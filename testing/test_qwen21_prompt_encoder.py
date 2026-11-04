"""Offline regressions using real, randomly initialized Qwen3-VL modules.

Run from the repository root: python -m unittest testing.test_qwen21_prompt_encoder -v
No checkpoint download or dataset writes. CUDA/convrot8/offloading coverage is
enabled automatically when CUDA is available. Compare each quantized model to
itself, not to unquantized weights: the allowed embedding difference is zero.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from transformers import BatchFeature, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


_spec = importlib.util.spec_from_file_location(
    "qwen21_pipeline",
    Path(__file__).resolve().parents[1]
    / "extensions_built_in/diffusion_models/qwen_image_2/src/pipeline.py",
)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def tiny_model(device="cpu"):
    torch.manual_seed(123)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=512,
            rope_scaling=dict(rope_type="default", mrope_section=[4, 6, 6]),
        ),
        vision_config=dict(
            depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
            patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=128, num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
        image_token_id=250, video_token_id=251,
        vision_start_token_id=252, vision_end_token_id=253,
    )
    model = Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    model.set_attn_implementation("eager")
    return model.to(device)


class Processor:
    """Deterministic left-padded text and valid tiny image inputs; no tokenizer IO."""

    tokenizer = SimpleNamespace(encode=lambda text: [250])

    def apply_chat_template(self, *args, **kwargs):
        return [[10, 11]]  # drop the two system tokens

    def __call__(self, text, images=None, **kwargs):
        rows = []
        for index, _ in enumerate(text):
            prefix = [10, 11, 252, 250, 253] if images else [10, 11]
            rows.append(prefix + [20 + index] * (index + 2))
        width = max(map(len, rows))
        ids = torch.tensor([[0] * (width - len(row)) + row for row in rows])
        mask = (ids != 0).long()
        types = (ids == 250).long()
        result = dict(input_ids=ids, attention_mask=mask, mm_token_type_ids=types)
        if images:
            # Four 2x2x2 RGB patches merge to one image token per image.
            result.update(
                pixel_values=torch.arange(len(images) * 4 * 24).float().reshape(-1, 24) / 100,
                image_grid_thw=torch.tensor([[1, 2, 2]] * len(images)),
            )
        return BatchFeature(result)


class LegacyBackbone:
    """Reproduce the old full LM call and hidden_states[-1] selection."""

    def __init__(self, model):
        self.encoder = model
        self.language_model = model.model.language_model

    def __call__(self, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs.pop("use_cache")  # use the old default (True)
        outputs = self.encoder(**kwargs)
        return SimpleNamespace(last_hidden_state=outputs.hidden_states[-1])


def legacy_encoder(model, processor):
    return pipeline.QwenImage21PromptEncoder(
        SimpleNamespace(model=LegacyBackbone(model), device=model.device), processor
    )


class PromptEncoderTests(unittest.TestCase):
    def compare(self, model, images=None):
        processor = Processor()
        prompts = ["", "longer prompt", "different length"]
        reference = legacy_encoder(model, processor).encode(prompts, images=images)
        captures = []
        outputs = []
        handle = model.model.register_forward_pre_hook(
            lambda module, args, kwargs: captures.append(dict(kwargs)), with_kwargs=True
        )
        output_handle = model.model.register_forward_hook(
            lambda module, args, output: outputs.append(output)
        )
        try:
            with mock.patch.object(model.lm_head, "forward", side_effect=AssertionError("lm_head called")):
                actual = pipeline.QwenImage21PromptEncoder(model, processor).encode(prompts, images=images)
        finally:
            handle.remove()
            output_handle.remove()
        for expected_group, actual_group in zip(reference, actual):
            for expected, result in zip(expected_group, actual_group):
                self.assertEqual(expected.shape, result.shape)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertFalse(captures[0]["output_hidden_states"])
        self.assertFalse(captures[0]["use_cache"])
        self.assertTrue(captures[0]["return_dict"])
        self.assertIsNone(outputs[0].hidden_states)
        self.assertIsNone(outputs[0].past_key_values)
        self.assertEqual([len(e) for e in actual[0]], [2, 3, 4] if images is None else [5, 6, 7])
        self.assertTrue(all(mask.dtype == torch.long and bool(mask.all()) for mask in actual[1]))
        self.assertTrue(all(mask.dtype == torch.bool for mask in actual[2]))
        self.assertEqual([int(mask.sum()) for mask in actual[2]], [0, 0, 0] if images is None else [1, 1, 1])
        self.assertFalse(model.model.language_model.norm._forward_hooks)

    def test_text_hidden_states_and_masks_are_exact(self):
        self.compare(tiny_model())

    def test_multimodal_hidden_states_and_image_slots_are_exact(self):
        self.compare(tiny_model(), images=[[object()], [object()], [object()]])

    def test_explicit_return_dict_overrides_model_config(self):
        model = tiny_model()
        model.config.return_dict = False
        model.model.config.return_dict = False
        result = pipeline.QwenImage21PromptEncoder(model, Processor()).encode(["text"])
        self.assertEqual(result[0][0].shape, (2, 128))

    def test_norm_hook_removed_on_failure(self):
        model = tiny_model()
        with mock.patch.object(model.model, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                pipeline.QwenImage21PromptEncoder(model, Processor()).encode(["text"])
        self.assertFalse(model.model.language_model.norm._forward_hooks)

    def quantize(self, model):
        from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer
        quantizer = get_ostris_quantizer("convrot8")
        count = sum(
            convert_linear_to_ostris(module, quantizer)
            for module in list(model.modules()) if isinstance(module, torch.nn.Linear)
        )
        self.assertGreater(count, 1)

    def test_convrot8_cpu_hidden_states_are_exact(self):
        model = tiny_model()
        self.quantize(model)
        self.compare(model)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for int8 GEMM/offloading")
    def test_convrot8_cuda_and_full_layer_offloading(self):
        from toolkit.memory_management import MemoryManager
        model = tiny_model().to(dtype=torch.bfloat16)
        self.quantize(model)
        MemoryManager.attach(model, torch.device("cuda"), offload_percent=1.0)
        self.compare(model)
        self.compare(model, images=[[object()], [object()], [object()]])
        self.assertTrue(all(
            module.cr8_qdata.device.type == "cpu"
            for module in model.modules() if hasattr(module, "cr8_qdata")
        ))


if __name__ == "__main__":
    unittest.main()
