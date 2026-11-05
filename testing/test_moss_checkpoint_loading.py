"""MOSS component and ComfyUI-saved checkpoint loading, including INT8 sidecars."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

# Reuse the tiny real MOSS fixture; imports only MOSS, not the CUDA registry.
from testing.test_moss_training_adapter import (
    MossImageModel, MossImageTransformer, ModelConfig, adapter_weights, tiny_transformer,
)
from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer


class MossCheckpointLoadingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def checkpoint(self, *, quantized=False):
        model = tiny_transformer()
        if quantized:
            convert_linear_to_ostris(model.layers[0].proj.qkv, get_ostris_quantizer("convrot8"))
        path = str(Path(self.temp.name) / "base.safetensors")
        model.save_model(path, quantized=quantized)
        return model, load_file(path)

    def test_raw_and_wrapped_checkpoints_load_identically(self):
        original, weights = self.checkpoint()
        for prefix in ("", "model.diffusion_model.", "diffusion_model.", "transformer."):
            with self.subTest(prefix=prefix):
                wrapped = {prefix + k: v for k, v in weights.items()}
                # A full checkpoint may contain other components too.
                if prefix:
                    wrapped["first_stage_model.unrelated.weight"] = torch.ones(1)
                original_keys = set(wrapped)
                restored = MossImageTransformer.load_from_state_dict(wrapped, torch.float32)
                self.assertEqual(set(wrapped), original_keys)
                self.assertEqual(restored.params, original.params)
                for key, value in restored.state_dict().items():
                    torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)

    def test_wrapped_int8_weights_preserve_codes_scales_and_forward(self):
        original, weights = self.checkpoint(quantized=True)
        x = torch.randn(2, 32, 16)
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
            with self.subTest(prefix=prefix):
                wrapped = {prefix + k: v for k, v in weights.items()}
                restored = MossImageTransformer.load_from_state_dict(wrapped, torch.float32)
                actual, expected = restored.layers[0].proj.qkv, original.layers[0].proj.qkv
                self.assertTrue(restored.aitk_is_quantized)
                torch.testing.assert_close(actual.cr8_qdata, expected.cr8_qdata, rtol=0, atol=0)
                torch.testing.assert_close(actual.cr8_scales, expected.cr8_scales, rtol=0, atol=0)
                torch.testing.assert_close(actual(x), expected(x), rtol=0, atol=0)

    def test_comfyui_checkpoint_loads_with_negative_assistant(self):
        _, weights = self.checkpoint(quantized=True)
        checkpoint = str(Path(self.temp.name) / "comfyui.safetensors")
        save_file({"model.diffusion_model." + k: v for k, v in weights.items()}, checkpoint)
        adapter = str(Path(self.temp.name) / "turbo.safetensors")
        save_file(adapter_weights(), adapter)
        model = MossImageModel("cpu", ModelConfig(
            name_or_path=checkpoint, arch="moss_image", quantize=True, qtype="convrot8",
            assistant_lora_path=adapter, assistant_lora_strength=-1,
        ), dtype="float32")
        with mock.patch.object(model, "_load_text_encoder", return_value=mock.Mock()), \
             mock.patch.object(model, "_load_tokenizer", return_value=mock.Mock()), \
             mock.patch.object(model, "_load_vae", return_value=mock.Mock()):
            model.load_model()
        self.assertIsNotNone(model.assistant_lora)
        self.assertEqual(model.assistant_lora.multiplier, 1)
        self.assertFalse(model.assistant_lora.is_active)
        self.assertTrue(model.model.layers[0].proj.qkv.is_ostris_quantized)


if __name__ == "__main__":
    unittest.main()
