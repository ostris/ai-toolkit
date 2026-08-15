import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn


class _FakeRotary(nn.Module):
    def forward(self, inputs_embeds, position_ids):
        return (torch.zeros_like(inputs_embeds), torch.zeros_like(inputs_embeds))


class _FakeLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _FakeLanguageModel(nn.Module):
    def __init__(self, hidden_size=2):
        super().__init__()
        self.config = types.SimpleNamespace()
        self.embed_tokens = nn.Embedding(100, hidden_size)
        nn.init.zeros_(self.embed_tokens.weight)
        self.rotary_emb = _FakeRotary()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(36)])
        self.gradient_checkpointing = False


class _FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()
        self.is_gradient_checkpointing = False


class _FakeActivator:
    atomic_token_id = 99
    lookup_token_id = 1

    def __init__(self):
        self.received_indices = None
        self.runtime_modes = []
        self.module_lora_installs = 0

    def install_module_lora(self, language_model):
        self.module_lora_installs += 1

    def set_runtime_mode(self, mode):
        self.runtime_modes.append(mode)

    def apply_embedding(self, hidden_states, token_mask=None, token_indices=None, runtime_mode=None, **kwargs):
        self.received_indices = token_indices.detach().clone()
        output = hidden_states.clone()
        output[token_mask.bool()] = token_indices[token_mask.bool()].to(output.dtype).unsqueeze(-1) + 1
        return output


class Ideogram4TriggerPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        transformers = sys.modules.get("transformers")
        masking_utils = types.ModuleType("transformers.masking_utils")
        masking_utils.create_causal_mask = lambda **kwargs: None
        sys.modules.setdefault("transformers", transformers or types.ModuleType("transformers"))
        sys.modules["transformers.masking_utils"] = masking_utils

        diffusers = sys.modules.get("diffusers") or types.ModuleType("diffusers")
        diffusers_utils = types.ModuleType("diffusers.utils")
        torch_utils = types.ModuleType("diffusers.utils.torch_utils")
        torch_utils.randn_tensor = torch.randn
        sys.modules["diffusers"] = diffusers
        sys.modules["diffusers.utils"] = diffusers_utils
        sys.modules["diffusers.utils.torch_utils"] = torch_utils

        package_name = "extensions_built_in.diffusion_models.ideogram4.src"
        package = sys.modules.get(package_name) or types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
        transformer = types.ModuleType(f"{package_name}.transformer")
        transformer.IMAGE_POSITION_OFFSET = 0
        transformer.LLM_TOKEN_INDICATOR = 3
        transformer.OUTPUT_IMAGE_INDICATOR = 2
        transformer.QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
        transformer.SEQUENCE_PADDING_INDICATOR = -1
        transformer.Ideogram4Transformer2DModel = nn.Module
        sys.modules[f"{package_name}.transformer"] = transformer

        pipeline_path = (
            Path(__file__).resolve().parents[1]
            / "extensions_built_in"
            / "diffusion_models"
            / "ideogram4"
            / "src"
            / "pipeline.py"
        )
        spec = importlib.util.spec_from_file_location(f"{package_name}.pipeline_v8_test", pipeline_path)
        cls.pipeline = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.pipeline)

    def test_virtual_token_indices_reach_embedding_and_preserve_shape(self):
        text_encoder = _FakeTextEncoder()
        activator = _FakeActivator()
        token_ids = torch.tensor([[5, 99, 99, 6]])
        attention_mask = torch.ones_like(token_ids)
        trigger_mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
        token_indices = torch.tensor([[0, 0, 1, 0]])
        pos_2d = torch.arange(4).unsqueeze(0)

        features = self.pipeline.get_qwen3_vl_features(
            text_encoder,
            token_ids,
            attention_mask,
            pos_2d,
            trigger_mask=trigger_mask,
            token_indices=token_indices,
            text_activator=activator,
            runtime_mode="full",
        )

        self.assertEqual(features.shape, (1, 4, 26))
        self.assertTrue(torch.equal(activator.received_indices, token_indices))
        self.assertEqual(activator.module_lora_installs, 1)
        self.assertIn("full", activator.runtime_modes)
        self.assertTrue(torch.equal(features[0, 1, :2], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(features[0, 2, :2], torch.tensor([2.0, 2.0])))

    def test_plain_path_without_activator_is_unchanged(self):
        text_encoder = _FakeTextEncoder()
        token_ids = torch.tensor([[5, 6]])
        attention_mask = torch.ones_like(token_ids)
        pos_2d = torch.arange(2).unsqueeze(0)
        features = self.pipeline.get_qwen3_vl_features(
            text_encoder,
            token_ids,
            attention_mask,
            pos_2d,
        )
        self.assertEqual(features.shape, (1, 2, 26))
        self.assertTrue(torch.equal(features, torch.zeros_like(features)))


if __name__ == "__main__":
    unittest.main()
