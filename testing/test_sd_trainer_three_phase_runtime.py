import ast
import contextlib
import importlib
import inspect
import os
import types
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch


def _load_runtime_methods():
    source_path = Path(__file__).parents[1] / 'extensions_built_in' / 'sd_trainer' / 'SDTrainer.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SDTrainer')
    names = {
        'three_phase_enabled', '_load_trigger_binding_modules', '_call_supported', '_first_callable',
        '_phase_config', '_activator_component_flags', '_configure_phase_trainability',
        'hook_add_extra_train_params', '_activator_mode', '_calculate_trigger_binding_loss',
        '_install_trigger_binding_prompt_encoder', 'encode_static_prompt',
    }
    selected = [node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    module = ast.Module(body=[ast.ClassDef(name='SDTrainerRuntimeHarness', bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        'contextlib': contextlib,
        'importlib': importlib,
        'inspect': inspect,
        'os': os,
        'MethodType': MethodType,
        'torch': torch,
        'get_torch_dtype': lambda _dtype: torch.float32,
        'shared_loss_target': lambda trainer, noise, batch, timesteps: trainer.sd.get_loss_target(
            noise=noise, batch=batch, timesteps=timesteps
        ).detach(),
    }
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['SDTrainerRuntimeHarness']


SDTrainer = _load_runtime_methods()


class _FakeActivator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(2, 2, bias=False)
        self.te_adapter = torch.nn.Linear(2, 2, bias=False)
        self.tap_adapters = torch.nn.Linear(2, 2, bias=False)
        self.component_active = {}

    def set_component_mode(self, component, active=None, trainable=None):
        self.component_active[component] = active
        getattr(self, component).requires_grad_(trainable)

    def parameter_groups(self, learning_rates=None):
        learning_rates = learning_rates or {}
        groups = []
        for name in ('embedding', 'te_adapter', 'tap_adapters'):
            params = [parameter for parameter in getattr(self, name).parameters() if parameter.requires_grad]
            if params:
                group = {'params': params, 'name': name}
                if name in learning_rates:
                    group['lr'] = learning_rates[name]
                groups.append(group)
        return groups


class ThreePhaseRuntimeTest(unittest.TestCase):
    def _trainer(self, phase):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.runtime_phase = phase
        trainer.text_activator = _FakeActivator()
        trainer.network = torch.nn.Linear(2, 2, bias=False)
        phase_config = SimpleNamespace(
            train={
                'embedding': phase != 'b',
                'internal': False,
                'tap': phase == 'a2',
                'diffusion_lora': phase == 'b',
            },
            learning_rates={'embedding': 1e-3, 'tap_adapters': 2e-3},
        )
        trainer.three_phase_trigger_training = SimpleNamespace(
            enabled=True,
            phase_a1=phase_config,
            phase_b=phase_config,
            phase_a2=phase_config,
        )
        return trainer

    def test_phase_whitelist_freezes_non_targets(self):
        trainer = self._trainer('a2')
        params = [{'params': list(trainer.network.parameters())}]
        filtered = trainer.hook_add_extra_train_params(params)
        selected = {id(parameter) for group in filtered for parameter in group['params']}
        self.assertTrue(all(id(parameter) not in selected for parameter in trainer.network.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in trainer.text_activator.embedding.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in trainer.text_activator.tap_adapters.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.text_activator.te_adapter.parameters()))

    def test_b_phase_keeps_only_diffusion_lora_and_freezes_activator(self):
        trainer = self._trainer('b')
        params = [{'params': list(trainer.network.parameters()) + list(trainer.text_activator.parameters())}]
        filtered = trainer.hook_add_extra_train_params(params)
        selected = {id(parameter) for group in filtered for parameter in group['params']}
        self.assertEqual(selected, {id(parameter) for parameter in trainer.network.parameters()})
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.text_activator.parameters()))

    def test_static_prompt_bypasses_required_trigger_binding(self):
        trainer = self._trainer('a1')
        modes = []

        class _ModeContext:
            def __init__(self, mode):
                self.mode = mode

            def __enter__(self):
                modes.append(self.mode)

            def __exit__(self, *_args):
                modes.append('restored')

        trainer._activator_mode = lambda mode: _ModeContext(mode)
        trainer.sd = SimpleNamespace(
            encode_prompt=lambda prompt, **_kwargs: ('encoded', prompt),
        )

        result = trainer.encode_static_prompt([''])

        self.assertEqual(result, ('encoded', ['']))
        self.assertEqual(modes, ['activator_bypass', 'restored'])

    def test_prompt_encoder_allows_static_prompt_only_in_bypass_mode(self):
        trainer = self._trainer('a1')
        original_calls = []

        class _SD:
            text_activator_runtime_mode = 'activator_bypass'

            def get_prompt_embeds(self, prompt, **kwargs):
                original_calls.append((prompt, kwargs))
                return ('plain', prompt)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        trainer._install_trigger_binding_prompt_encoder(SimpleNamespace())

        result = trainer.sd.get_prompt_embeds([''])

        self.assertEqual(result, ('plain', ['']))
        self.assertEqual(original_calls[0][1]['runtime_mode'], 'activator_bypass')

    def test_a_phase_loss_receives_shared_latent_noise_timestep_and_target(self):
        trainer = self._trainer('a1')
        trainer.device_torch = torch.device('cpu')
        trainer.do_long_prompts = False
        trainer.additional_logs = {}
        trainer.sd = SimpleNamespace(
            encode_prompt=lambda prompts, **kwargs: torch.ones(len(prompts), 1, 2),
            get_loss_target=lambda noise, batch, timesteps: noise + 1,
        )
        trainer.predict_noise = lambda noisy_latents, **kwargs: noisy_latents * trainer.text_activator.embedding.weight.mean()
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)
        batch = SimpleNamespace(
            file_items=[SimpleNamespace(caption_template='x [trigger]', raw_caption='unused')],
            latents=torch.zeros(1, 2),
        )
        noisy = torch.randn(1, 2)
        noise = torch.randn(1, 2)
        timesteps = torch.tensor([10])
        captured = {}

        def fake_losses(**kwargs):
            captured.update(kwargs)
            return {'loss': (kwargs['active_prediction'] - kwargs['target']).pow(2).mean(), 'metrics': {'paired': 1}}

        trainer._trigger_binding_modules = {
            'losses': types.SimpleNamespace(calculate_trigger_binding_losses=fake_losses)
        }
        loss = trainer._calculate_trigger_binding_loss(
            noisy, noise, timesteps, batch, {}, 1.0, torch.float32
        )
        self.assertTrue(loss.requires_grad)
        self.assertIs(captured['noisy_latents'], noisy)
        self.assertIs(captured['noise'], noise)
        self.assertIs(captured['timesteps'], timesteps)
        self.assertTrue(torch.equal(captured['target'], noise + 1))
        self.assertEqual(trainer.additional_logs['phase/a1/paired'], 1.0)


if __name__ == '__main__':
    unittest.main()
