import unittest

import torch
from torch import nn

from toolkit.models.ideogram4_trigger_activator import (
    AtomicLearnedEmbedding,
    BoundedGamma,
    DEFAULT_TAP_LAYERS,
    MaskedLowRankAdapter,
    TextActivator,
    trigger_runtime,
)


class _FakeQwen(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(4)])
        for block in self.blocks:
            nn.init.eye_(block.weight)

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class _FakeMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.down_proj.weight)

    def forward(self, hidden_states):
        return self.down_proj(hidden_states)


class _FakeLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = _FakeMLP(hidden_size)

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


class _FakeLanguageModel(nn.Module):
    def __init__(self, hidden_size, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(hidden_size) for _ in range(layers)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Ideogram4TriggerActivatorTest(unittest.TestCase):
    def test_atomic_embedding_learned_frozen_and_bypass(self):
        initializer = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        embedding = AtomicLearnedEmbedding(4, initializer=initializer)
        hidden = torch.zeros(1, 3, 4)
        mask = torch.tensor([[0, 1, 0]], dtype=torch.bool)

        embedding.weight.data.fill_(9.0)
        learned = embedding(hidden, mask, mode="learned")
        frozen = embedding(hidden, mask, mode="frozen")
        bypass = embedding(hidden, mask, mode="bypass")

        self.assertTrue(torch.equal(learned[0, 1], torch.full((4,), 9.0)))
        self.assertTrue(torch.equal(frozen[0, 1], initializer[0]))
        self.assertTrue(torch.equal(frozen[0, 0], hidden[0, 0]))
        self.assertIs(bypass, hidden)
        self.assertFalse(embedding.frozen_initializer.requires_grad)

    def test_multi_vector_embedding_uses_virtual_token_indices(self):
        initializer = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        embedding = AtomicLearnedEmbedding(4, tokens=2, initializer=initializer)
        hidden = torch.zeros(1, 5, 4)
        mask = torch.tensor([[0, 1, 1, 0, 1]], dtype=torch.bool)
        indices = torch.tensor([[0, 0, 1, 0, 1]])
        output = embedding(hidden, mask, token_indices=indices)
        self.assertTrue(torch.equal(output[0, 1], initializer[0]))
        self.assertTrue(torch.equal(output[0, 2], initializer[1]))
        self.assertTrue(torch.equal(output[0, 4], initializer[1]))
        self.assertTrue(torch.equal(output[0, 0], hidden[0, 0]))

    def test_same_shape_frozen_initializer_bypass(self):
        initializer = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        activator = TextActivator(2, embedding_tokens=2, initializer=initializer)
        activator.embedding.weight.data.fill_(9.0)
        hidden = torch.zeros(1, 3, 2)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        indices = torch.tensor([[0, 1, 0]])
        activator.set_runtime_mode("activator_bypass")
        output = activator.apply_embedding(
            hidden,
            token_mask=mask,
            token_indices=indices,
            runtime_mode="activator_bypass",
        )
        self.assertTrue(torch.equal(output[0, :2], initializer))
        self.assertEqual(output.shape, hidden.shape)

    def test_masked_low_rank_adapter_only_changes_trigger_span(self):
        adapter = MaskedLowRankAdapter(4, rank=2, alpha=2)
        nn.init.ones_(adapter.down.weight)
        nn.init.ones_(adapter.up.weight)
        hidden = torch.ones(1, 3, 4)
        mask = torch.tensor([[0, 1, 0]], dtype=torch.bool)

        output = adapter(hidden, mask)

        self.assertTrue(torch.equal(output[:, 0], hidden[:, 0]))
        self.assertFalse(torch.equal(output[:, 1], hidden[:, 1]))
        self.assertTrue(torch.equal(output[:, 2], hidden[:, 2]))

    def test_runtime_context_supplies_mask_without_hard_dependency(self):
        adapter = MaskedLowRankAdapter(2, rank=1)
        nn.init.ones_(adapter.down.weight)
        nn.init.ones_(adapter.up.weight)
        hidden = torch.ones(1, 2, 2)
        with trigger_runtime({"token_mask": torch.tensor([[1, 0]])}):
            output = adapter(hidden)
        self.assertFalse(torch.equal(output[:, 0], hidden[:, 0]))
        self.assertTrue(torch.equal(output[:, 1], hidden[:, 1]))

    def test_exactly_thirteen_taps_are_keyed_by_actual_layer(self):
        activator = TextActivator(4, tap_layers=DEFAULT_TAP_LAYERS)
        self.assertEqual(activator.tap_layers, DEFAULT_TAP_LAYERS)
        self.assertEqual(len(activator.tap_adapters), 13)
        with self.assertRaises(KeyError):
            activator.apply_tap(1, torch.zeros(1, 1, 4), torch.ones(1, 1))
        with self.assertRaises(ValueError):
            TextActivator(4, tap_layers=DEFAULT_TAP_LAYERS[:-1])

    def test_component_active_trainable_and_parameter_groups(self):
        te_adapter = MaskedLowRankAdapter(4, rank=1)
        activator = TextActivator(4, te_adapter=te_adapter)
        activator.set_component_mode("embedding", active=False, trainable=False)
        activator.set_component_mode("te_adapter", active=True, trainable=True)
        groups = activator.parameter_groups({"te_adapter": 2.0e-4, "tap_adapters": 1.0e-4})

        self.assertFalse(activator.component_active["embedding"])
        self.assertTrue(all(not parameter.requires_grad for parameter in activator.embedding.parameters()))
        self.assertEqual(
            [group["name"] for group in groups],
            ["text_activator.te_adapter", "text_activator.tap_adapters"],
        )
        self.assertEqual([group["lr"] for group in groups], [2.0e-4, 1.0e-4])

    def test_bounded_gamma_scales_active_components_and_stays_bounded(self):
        gamma = BoundedGamma(initial=0.5, minimum=0.0, maximum=1.0, trainable=True)
        self.assertGreater(gamma.value(), 0.49)
        self.assertLess(gamma.value(), 0.51)
        gamma.raw.data.fill_(100.0)
        self.assertLessEqual(gamma.value(), 1.0)
        gamma.raw.data.fill_(-100.0)
        self.assertGreaterEqual(gamma.value(), 0.0)

        activator = TextActivator(2, gamma_init=0.5)
        activator.embedding.weight.data.fill_(2.0)
        hidden = torch.zeros(1, 1, 2)
        output = activator.apply_embedding(hidden, token_mask=torch.ones(1, 1))
        self.assertTrue(torch.allclose(output, torch.ones_like(output), atol=1.0e-4))
        activator.set_component_mode("embedding", active=False)
        self.assertIs(activator.apply_embedding(hidden, token_mask=torch.ones(1, 1)), hidden)

    def test_module_lora_targets_each_layer_with_independent_parameters(self):
        qwen = _FakeLanguageModel(4, layers=3)
        activator = TextActivator(
            4,
            te_adapter_mode="module_lora",
            te_rank=2,
            te_target_modules=("down_proj",),
        )
        activator.install_module_lora(qwen)
        self.assertEqual(len(activator.module_lora_adapters), 3)
        adapters = list(activator.module_lora_adapters.values())
        self.assertIsNot(adapters[0].down.weight, adapters[1].down.weight)
        for adapter in adapters:
            nn.init.ones_(adapter.down.weight)
            nn.init.ones_(adapter.up.weight)
        hidden = torch.ones(1, 2, 4)
        mask = torch.tensor([[1, 0]], dtype=torch.bool)
        with trigger_runtime({"token_mask": mask}):
            activated = qwen(hidden)
        self.assertFalse(torch.equal(activated[:, 0], hidden[:, 0]))
        self.assertTrue(torch.equal(activated[:, 1], hidden[:, 1]))

    def test_qwen_internal_hooks_apply_and_are_removable(self):
        qwen = _FakeQwen(4)
        activator = TextActivator(4)
        tap = activator.tap_adapters[str(DEFAULT_TAP_LAYERS[0])]
        nn.init.ones_(tap.down.weight)
        nn.init.ones_(tap.up.weight)
        hidden = torch.ones(1, 2, 4)
        mask = torch.tensor([[1, 0]], dtype=torch.bool)

        with trigger_runtime({"token_mask": mask}):
            baseline = qwen(hidden)
            activator.install_qwen_hooks(
                qwen,
                tap_module_names={DEFAULT_TAP_LAYERS[0]: "blocks.1"},
            )
            activated = qwen(hidden)
            diagnostics = activator.probe_diagnostics()
            activator.remove_qwen_hooks()
            restored = qwen(hidden)

        self.assertFalse(torch.equal(activated[:, 0], baseline[:, 0]))
        self.assertTrue(torch.equal(activated[:, 1], baseline[:, 1]))
        self.assertTrue(torch.equal(restored, baseline))
        self.assertEqual(diagnostics.hook_count, 1)
        self.assertGreater(diagnostics.adapter_update_norms[str(DEFAULT_TAP_LAYERS[0])], 0.0)

    def test_wrapper_mode_restores_original_module(self):
        qwen = _FakeQwen(4)
        original = qwen.blocks[2]
        activator = TextActivator(4, te_adapter=MaskedLowRankAdapter(4, rank=1))
        activator.install_qwen_hooks(qwen, te_module_names=["blocks.2"], use_wrappers=True)
        self.assertIsNot(qwen.blocks[2], original)
        activator.remove_qwen_hooks()
        self.assertIs(qwen.blocks[2], original)

    def test_runtime_metadata_is_serializable(self):
        import json

        activator = TextActivator(
            4,
            embedding_tokens=4,
            te_adapter_mode="module_lora",
            te_rank=2,
            tap_rank=2,
            gamma_init=0.5,
        )
        metadata = activator.runtime_metadata()
        encoded = json.dumps(metadata)
        self.assertIn('"architecture_version": 8', encoded)
        self.assertEqual(metadata["virtual_tokens"], 4)
        self.assertEqual(metadata["te_adapter_mode"], "module_lora")

    def test_namespaced_state_dict_and_strict_validation(self):
        activator = TextActivator(4, te_adapter=MaskedLowRankAdapter(4, rank=1))
        state = activator.state_dict()
        self.assertTrue(state)
        self.assertTrue(all(key.startswith("ideogram4_text_activator.") for key in state))

        clone = TextActivator(4, te_adapter=MaskedLowRankAdapter(4, rank=1))
        clone.load_state_dict(state, strict=True)
        missing = dict(state)
        missing.pop(next(iter(missing)))
        with self.assertRaises(RuntimeError):
            clone.load_state_dict(missing, strict=True)
        foreign = dict(state)
        foreign["unrelated.weight"] = torch.ones(1)
        with self.assertRaises(RuntimeError):
            clone.load_state_dict(foreign, strict=True)
        wrong_shape = dict(state)
        first_key = next(iter(wrong_shape))
        wrong_shape[first_key] = torch.ones(999)
        with self.assertRaises(RuntimeError):
            clone.load_state_dict(wrong_shape, strict=True)


if __name__ == "__main__":
    unittest.main()
