"""CPU regression tests; no downloaded model or text encoder required.

Run: python -m unittest testing.test_moss_training_adapter
"""

import copy
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

# Import only MOSS: the extension registry eagerly imports CUDA-only models.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions_built_in" / "diffusion_models"))
from moss_image.moss_image import MossImageModel
from moss_image.src.assistant_lora import (
    load_training_adapter,
    prepare_adapter_weights,
)
from moss_image.src.transformer import (
    MossImageParams,
    MossImageTransformer,
)
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.base_model import BaseModel


def tiny_transformer():
    return MossImageTransformer(MossImageParams(
        dim=16, ffn_dim=32, num_heads=2, num_layers=2,
        text_dim=8, freq_dim=8, latent_channels=4,
    )).eval().requires_grad_(False)


def adapter_weights():
    weights = {}
    for i, (name, rank) in enumerate((("q", 2), ("k", 2), ("v", 2), ("o", 1))):
        prefix = f"diffusion_model.layers.0.proj.{name}"
        weights[prefix + ".lora_down.weight"] = torch.randn(rank, 16) * 0.1
        weights[prefix + ".lora_up.weight"] = torch.randn(16, rank) * 0.1
        weights[prefix + ".alpha"] = torch.tensor(float(i + 1))
    return weights


class MossTrainingAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "adapter.safetensors")
        self.weights = adapter_weights()
        save_file(self.weights, self.path)

    def model(self, strength=1.0):
        return MossImageModel("cpu", ModelConfig(
            name_or_path="unused", arch="moss_image",
            assistant_lora_path=self.path, assistant_lora_strength=strength,
        ), dtype="float32")

    def test_default_and_signed_config(self):
        self.assertEqual(ModelConfig(name_or_path="unused").assistant_lora_strength, 1.0)
        self.assertEqual(self.model(-0.75).model_config.assistant_lora_strength, -0.75)
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model(value)
        with self.assertRaisesRegex(ValueError, "only for MOSS"):
            ModelConfig(name_or_path="unused", arch="flux", assistant_lora_strength=-1)

    def test_independent_qkv_ranks_and_alpha_are_preserved(self):
        transformer, model = tiny_transformer(), self.model()
        weights, dims = prepare_adapter_weights(model, transformer, self.weights)
        prefix = "transformer.layers.0.proj."
        self.assertEqual(dims[(prefix + "qkv").replace(".", "$$")], 6)
        self.assertEqual(dims[(prefix + "o").replace(".", "$$")], 1)
        actual = weights[prefix + "qkv.lora_B.weight"] @ weights[prefix + "qkv.lora_A.weight"]
        expected = []
        for name in ("q", "k", "v"):
            base = f"diffusion_model.layers.0.proj.{name}."
            expected.append(self.weights[base + "lora_up.weight"] @ self.weights[base + "lora_down.weight"]
                            * self.weights[base + "alpha"] / 2)
        torch.testing.assert_close(actual, torch.cat(expected))

    def test_signed_merge_and_inverse_preview_do_not_accumulate(self):
        for strength in (-1.0, -0.5, 0.75, 1.0):
            with self.subTest(strength=strength):
                transformer, model = tiny_transformer(), self.model(strength)
                before = copy.deepcopy(transformer.state_dict())
                prepared, _ = prepare_adapter_weights(model, transformer, self.weights)
                x = torch.randn(2, 3, 16)
                original = transformer.layers[0].proj.qkv(x)
                delta = prepared["transformer.layers.0.proj.qkv.lora_B.weight"] @ prepared["transformer.layers.0.proj.qkv.lora_A.weight"]
                load_training_adapter(model, transformer)
                assistant = model.assistant_lora
                self.assertFalse(assistant.is_active)
                self.assertTrue(model.invert_assistant_lora)
                self.assertTrue(all(not p.requires_grad for p in assistant.parameters()))
                torch.testing.assert_close(transformer.layers[0].proj.qkv.weight,
                                           before["layers.0.proj.qkv.weight"] + strength * delta)
                merged = transformer.layers[0].proj.qkv.weight.detach().clone()
                training = transformer.layers[0].proj.qkv(x)
                self.assertFalse(torch.allclose(training, original))
                for _ in range(3):
                    assistant.is_active = True
                    torch.testing.assert_close(transformer.layers[0].proj.qkv(x), original)
                    assistant.is_active = False
                    torch.testing.assert_close(transformer.layers[0].proj.qkv(x), training)
                    torch.testing.assert_close(transformer.layers[0].proj.qkv.weight, merged)

    def test_zero_does_not_read_or_modify_checkpoint(self):
        transformer, model = tiny_transformer(), self.model(0)
        before = copy.deepcopy(transformer.state_dict())
        with mock.patch("moss_image.src.assistant_lora.load_file") as load:
            load_training_adapter(model, transformer)
        load.assert_not_called()
        self.assertIsNone(model.model_config.assistant_lora_path)
        self.assertIsNone(model.assistant_lora)
        for key, value in transformer.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_invalid_adapter_fails_without_modifying_model(self):
        for kind in ("unknown", "missing_up", "wrong_shape", "lokr", "incomplete_qkv"):
            with self.subTest(kind=kind):
                weights = dict(self.weights)
                if kind == "unknown":
                    weights["diffusion_model.unknown.alpha"] = torch.tensor(1.)
                elif kind == "missing_up":
                    del weights["diffusion_model.layers.0.proj.o.lora_up.weight"]
                elif kind == "wrong_shape":
                    weights["diffusion_model.layers.0.proj.o.lora_up.weight"] = torch.randn(17, 1)
                elif kind == "lokr":
                    weights = {"diffusion_model.layers.0.proj.o.lokr_w1": torch.randn(2, 2)}
                else:
                    weights = {k: v for k, v in weights.items() if ".proj.k." not in k}
                transformer = tiny_transformer()
                before = copy.deepcopy(transformer.state_dict())
                with self.assertRaises(ValueError):
                    prepare_adapter_weights(self.model(), transformer, weights)
                for key, value in transformer.state_dict().items():
                    torch.testing.assert_close(value, before[key])

    def test_duplicate_aliases_and_conflicting_paths_are_rejected(self):
        weights = dict(self.weights)
        weights["transformer.layers.0.proj.q.lora_A.weight"] = weights["diffusion_model.layers.0.proj.q.lora_down.weight"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_adapter_weights(self.model(), tiny_transformer(), weights)
        for key in ("lora_path", "inference_lora_path"):
            model = self.model(-1)
            setattr(model.model_config, key, "another.safetensors")
            with mock.patch.object(model, "_load_transformer") as load:
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    model.load_model()
                load.assert_not_called()

    def test_new_trainable_network_is_separate_and_receives_gradients(self):
        model, transformer = self.model(-1), tiny_transformer()
        load_training_adapter(model, transformer)
        config = NetworkConfig(type="lora", linear=2, linear_alpha=2)
        network = LoRASpecialNetwork(
            text_encoder=None, unet=transformer, train_text_encoder=False,
            train_unet=True, lora_dim=2, alpha=2, is_transformer=True,
            target_lin_modules=["MossImageTransformer"], network_config=config,
            base_model=model,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.is_active = True
        network._update_torch_multiplier()
        x = torch.randn(2, 3, 16)
        transformer.layers[0].proj.qkv(x).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters()))
        self.assertTrue(all(p.grad is None for p in model.assistant_lora.parameters()))
        self.assertFalse(set(network.state_dict()) == set(model.assistant_lora.state_dict()))

    def test_sampling_cleanup_even_when_generation_raises(self):
        for fails in (False, True):
            with self.subTest(fails=fails):
                model = self.model(-1)
                load_training_adapter(model, tiny_transformer())
                def sample(*args, **kwargs):
                    model.assistant_lora.is_active = True
                    self.assertEqual(model.assistant_lora.torch_multiplier.item(), 1.)
                    if fails:
                        raise RuntimeError("preview failed")
                    return "preview"
                with mock.patch.object(BaseModel, "generate_images", side_effect=sample):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "preview failed"):
                            model.generate_images([])
                    else:
                        self.assertEqual(model.generate_images([]), "preview")
                self.assertFalse(model.assistant_lora.is_active)
                self.assertTrue(all(p.device.type == "cpu" for p in model.assistant_lora.parameters()))

    def test_native_eight_step_sampler_reverses_adapter(self):
        model, transformer = self.model(-1), tiny_transformer()
        baseline = self.model()
        baseline.model_config.assistant_lora_path = None
        baseline.model = copy.deepcopy(transformer)
        model.model = transformer
        load_training_adapter(model, transformer)
        merged = {k: v.clone() for k, v in transformer.state_dict().items()}
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.randn(3, 8)])
        captured = []
        for current in (baseline, model):
            current.sample_prompts_cache = [{"conditional": embeds, "unconditional": embeds}]
            config = GenerateImageConfig(
                prompt="geometric shapes", width=32, height=32, seed=10,
                num_inference_steps=8, guidance_scale=1,
                output_path=str(Path(self.temp.name) / "sample.png"),
            )
            def decode(latents, **kwargs):
                captured.append(latents.clone())
                return latents[:, :3]
            with mock.patch.object(current, "save_device_state"), \
                 mock.patch.object(current, "restore_device_state"), \
                 mock.patch.object(current, "set_device_state_preset"), \
                 mock.patch.object(current, "decode_latents", side_effect=decode), \
                 mock.patch.object(config, "save_image_atomic"), \
                 mock.patch.object(config, "log_image"):
                current.generate_images([config])
        torch.testing.assert_close(captured[0], captured[1], atol=1e-5, rtol=1e-5)
        self.assertFalse(model.assistant_lora.is_active)
        for key, value in transformer.state_dict().items():
            torch.testing.assert_close(value, merged[key], rtol=0, atol=0)

    def test_full_moss_lokr_training_step_keeps_assistant_frozen(self):
        model, transformer = self.model(-1), tiny_transformer()
        load_training_adapter(model, transformer)
        assistant_before = {k: v.clone() for k, v in model.assistant_lora.state_dict().items()}
        config = NetworkConfig(type="lokr", linear=8, linear_alpha=8, lokr_full_rank=True, lokr_factor=8)
        network = LoRASpecialNetwork(
            text_encoder=None, unet=transformer, train_text_encoder=False,
            train_unet=True, lora_dim=config.linear, alpha=config.linear_alpha,
            is_transformer=True, network_type="lokr", network_config=config,
            target_lin_modules=["MossImageTransformer"], base_model=model,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.is_active = True
        network._update_torch_multiplier()
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
        assistant_ids = {id(p) for p in model.assistant_lora.parameters()}
        self.assertTrue(assistant_ids.isdisjoint(id(p) for group in optimizer.param_groups for p in group["params"]))
        latents, text = torch.randn(2, 4, 2, 2), torch.randn(2, 3, 8)
        predicted = transformer(latents, torch.tensor([500., 700.]), text)
        loss = (predicted - torch.randn_like(predicted)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters()))
        optimizer.step()
        for key, value in model.assistant_lora.state_dict().items():
            torch.testing.assert_close(value, assistant_before[key], rtol=0, atol=0)

    def test_adapter_loaded_before_quantization(self):
        model, transformer = self.model(-1), tiny_transformer()
        events = []
        def post_load(**kwargs):
            events.append("post_load")
            self.assertIsNotNone(model.assistant_lora)
        with mock.patch.object(model, "_load_transformer", return_value=transformer), \
             mock.patch.object(transformer, "aitk_post_load", side_effect=post_load), \
             mock.patch.object(model, "_load_text_encoder", return_value=mock.Mock()), \
             mock.patch.object(model, "_load_tokenizer", return_value=mock.Mock()), \
             mock.patch.object(model, "_load_vae", return_value=mock.Mock()):
            model.load_model()
        self.assertEqual(events, ["post_load"])

    def test_quantized_base_stays_quantized_during_preview_cycles(self):
        from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer
        transformer, model = tiny_transformer(), self.model(-1)
        model.model_config.qtype = "convrot8"
        proj = transformer.layers[0].proj.qkv
        convert_linear_to_ostris(proj, get_ostris_quantizer("convrot8"))
        load_training_adapter(model, transformer)
        self.assertTrue(proj.is_ostris_quantized)
        stored = {k: v.clone() for k, v in proj.state_dict().items()}
        x = torch.randn(2, 3, 16)
        training = proj(x)
        for _ in range(3):
            model.assistant_lora.is_active = True
            preview = proj(x)
            self.assertFalse(torch.allclose(preview, training))
            model.assistant_lora.is_active = False
            torch.testing.assert_close(proj(x), training)
        for key, value in proj.state_dict().items():
            torch.testing.assert_close(value, stored[key], rtol=0, atol=0)

    def test_prequantized_merge_is_seeded_repeatable_and_signed(self):
        from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer
        for strength in (-1., 1.):
            with self.subTest(strength=strength):
                transformer = tiny_transformer()
                proj = transformer.layers[0].proj.qkv
                convert_linear_to_ostris(proj, get_ostris_quantizer("convrot8"))
                clone = copy.deepcopy(transformer)
                quantizer = proj.ostris_quantizer
                with mock.patch.object(quantizer, "requantize_", wraps=quantizer.requantize_) as requantize:
                    model = self.model(strength)
                    load_training_adapter(model, transformer)
                self.assertEqual(requantize.call_count, 1)
                self.assertIsInstance(requantize.call_args.kwargs["stochastic_seed"], int)
                self.assertEqual(model.assistant_lora.multiplier, -strength)
                self.assertFalse(model.assistant_lora.is_active)
                # Changing the global RNG must not change the merge realization.
                torch.manual_seed(9876)
                load_training_adapter(self.model(strength), clone)
                other = clone.layers[0].proj.qkv
                torch.testing.assert_close(proj.cr8_qdata, other.cr8_qdata, rtol=0, atol=0)
                torch.testing.assert_close(proj.cr8_scales, other.cr8_scales, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for layer offloading")
    def test_inverse_adapter_with_layer_offloading(self):
        from toolkit.memory_management.manager import MemoryManager
        from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                transformer, model = tiny_transformer(), self.model(-1)
                model.device_torch = torch.device("cuda:0")
                proj = transformer.layers[0].proj.qkv
                if quantized:
                    convert_linear_to_ostris(proj, get_ostris_quantizer("convrot8"))
                load_training_adapter(model, transformer)
                # Attach after the assistant, matching MOSS load_model order.
                MemoryManager.attach(transformer, model.device_torch, offload_percent=1.0)
                try:
                    assistant = model.assistant_lora
                    assistant.force_to(model.device_torch, model.torch_dtype)
                    assistant._update_torch_multiplier()
                    # Image-token-sized input exercises GEMM, avoiding the separate
                    # decode-only Triton GEMV kernel (unsupported on Turing).
                    x = torch.randn(2, 32, 16, device=model.device_torch)
                    training = proj(x)
                    assistant.is_active = True
                    preview = proj(x)
                    self.assertTrue(torch.isfinite(preview).all())
                    self.assertFalse(torch.allclose(training, preview))
                    assistant.is_active = False
                    torch.testing.assert_close(proj(x), training)
                    self.assertEqual(assistant.torch_multiplier.device.type, "cuda")
                finally:
                    MemoryManager.detach(transformer)


class SeededInt8RoundingTest(unittest.TestCase):
    def input(self, device):
        # Exact INT8-grid base with both endpoints present in every row.
        generator = torch.Generator(device=device).manual_seed(11)
        codes = torch.randint(-126, 127, (512, 256), device=device, generator=generator)
        codes[:, 0], codes[:, 1] = 127, -127
        return codes.float() * .01

    def test_small_signed_updates_survive_without_consuming_global_rng(self):
        from toolkit.util.convrot_quant import quantize_int8_rows
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            base = self.input(device)
            for sign in (-1, 1):
                with self.subTest(device=device, sign=sign):
                    delta = torch.full_like(base, sign * .0005)
                    q_nearest, scales = quantize_int8_rows(base + delta)
                    nearest = q_nearest.float() * scales[:, None] - base
                    state = torch.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state() if device != "cpu" else None
                    q, scales = quantize_int8_rows(base + delta, stochastic_seed=1234)
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    if cuda_state is not None:
                        self.assertTrue(torch.equal(cuda_state, torch.cuda.get_rng_state()))
                    actual = q.float() * scales[:, None] - base
                    denominator = delta.square().sum()
                    self.assertLess(abs(float((nearest * delta).sum() / denominator)), .1)
                    self.assertAlmostEqual(float((actual * delta).sum() / denominator), 1., delta=.1)
                    repeat, _ = quantize_int8_rows(base + delta, stochastic_seed=1234)
                    other, _ = quantize_int8_rows(base + delta, stochastic_seed=5678)
                    torch.testing.assert_close(q, repeat, rtol=0, atol=0)
                    self.assertFalse(torch.equal(q, other))

    def test_default_rounding_and_zero_rows_are_unchanged(self):
        from toolkit.util.convrot_quant import quantize_int8_rows
        x = torch.cat((torch.zeros(1, 256), self.input("cpu")[:3]))
        expected_scale = x.abs().amax(dim=1) / 127
        expected_scale = torch.where(expected_scale > 0, expected_scale, torch.ones_like(expected_scale))
        expected = torch.round(x / expected_scale[:, None]).clamp(-127, 127).to(torch.int8)
        q, scales = quantize_int8_rows(x)
        torch.testing.assert_close(q, expected, rtol=0, atol=0)
        torch.testing.assert_close(scales, expected_scale, rtol=0, atol=0)
        q, _ = quantize_int8_rows(x[:1], stochastic_seed=1234)
        self.assertFalse(q.any())


if __name__ == "__main__":
    unittest.main()
