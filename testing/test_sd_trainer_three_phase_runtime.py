import ast
import contextlib
import importlib
import inspect
import os
import tempfile
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
        'hook_add_extra_train_params', '_activator_mode', '_write_trigger_binding_metrics',
        '_phase_caption_source_weights', '_prompt_tap_batch', '_check_first_trigger_gradient',
        '_calculate_trigger_binding_loss', '_install_trigger_binding_prompt_encoder', 'encode_static_prompt',
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
        'F': torch.nn.functional,
        'get_torch_dtype': lambda _dtype: torch.float32,
        'shared_loss_target': lambda trainer, noise, batch, timesteps: trainer.sd.get_loss_target(
            noise=noise, batch=batch, timesteps=timesteps
        ).detach(),
    }
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['SDTrainerRuntimeHarness']


SDTrainer = _load_runtime_methods()


class _FakeEmbeds:
    def __init__(self, batch_size, token_count=3, active=True, with_taps=False):
        scale = 1.0 if active else 0.0
        self.text_embeds = [torch.full((token_count, 2), scale) for _ in range(batch_size)]
        self.trigger_masks = [
            torch.tensor([True] + [False] * (token_count - 1)) for _ in range(batch_size)
        ]
        if with_taps:
            taps = torch.zeros(13, token_count, 2)
            taps[:, 0, 0] = scale
            self.text_taps = [taps.clone() for _ in range(batch_size)]

    def __contains__(self, key):
        return hasattr(self, key)

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        detached = _FakeEmbeds.__new__(_FakeEmbeds)
        detached.text_embeds = [tensor.detach() for tensor in self.text_embeds]
        detached.trigger_masks = [tensor.detach() for tensor in self.trigger_masks]
        if hasattr(self, 'text_taps'):
            detached.text_taps = [tensor.detach() for tensor in self.text_taps]
        return detached


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
            losses={'diffusion_mse': {'enabled': True, 'weight': 1.0}},
            activator_gain_floor=SimpleNamespace(
                enabled=False,
                weight=0.0,
                schedule=SimpleNamespace(keyframes=[], interpolation='smoothstep'),
            ),
            context_consistency=SimpleNamespace(
                enabled=False,
                weight=0.0,
                pooling='mean',
                detach_reference=False,
                magnitude_weight=0.0,
                min_delta_norm=1e-6,
                warmup_steps=0,
                tap_layers=None,
            ),
        )
        trainer.three_phase_trigger_training = SimpleNamespace(
            enabled=True,
            phase_a1=phase_config,
            phase_b=phase_config,
            phase_a2=phase_config,
            phase_runtime=SimpleNamespace(caption_sources={}),
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

    def test_prompt_encoder_bypass_keeps_binding_metadata_for_trigger_caption(self):
        trainer = self._trainer('a1')

        class _Batch:
            input_ids = torch.tensor([[1, 2, 3]])
            attention_mask = torch.tensor([[1, 1, 1]])
            trigger_mask = torch.tensor([[False, True, False]])

            def to(self, _device):
                return self

        class _SD:
            text_activator_runtime_mode = 'activator_bypass'
            tokenizer = object()
            text_encoder = SimpleNamespace(device=torch.device('cpu'))
            torch_dtype = torch.float32
            max_text_length = 16

            def get_prompt_embeds(self, prompt, **kwargs):
                return ('plain', prompt, kwargs)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        modules = SimpleNamespace(bind_trigger_batch=lambda *_args, **_kwargs: _Batch())
        fake_pipeline = SimpleNamespace(get_qwen3_vl_features=lambda *_args, **_kwargs: (
            torch.ones(1, 3, 2), [torch.ones(1, 3, 2) for _ in range(13)]
        ))
        with patch('importlib.import_module', return_value=fake_pipeline):
            trainer._install_trigger_binding_prompt_encoder({'runtime': modules})
            embeds = trainer.sd.get_prompt_embeds(['x [trigger]'], return_taps=True)

        self.assertIn('text_taps', embeds)
        self.assertIn('trigger_masks', embeds)
        self.assertEqual(embeds.text_taps[0].shape, (13, 3, 2))
        torch.testing.assert_close(embeds.trigger_masks[0], torch.tensor([False, True, False]))

    def test_prompt_encoder_bypasses_already_injected_literal_caption(self):
        trainer = self._trainer('a1')
        original_calls = []

        class _SD:
            text_activator_runtime_mode = 'full'

            def get_prompt_embeds(self, prompt, **kwargs):
                original_calls.append((prompt, kwargs))
                return ('plain', prompt)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        trainer._install_trigger_binding_prompt_encoder(SimpleNamespace())

        result = trainer.sd.get_prompt_embeds(['caption with <trigger> already injected'])

        self.assertEqual(result, ('plain', ['caption with <trigger> already injected']))
        self.assertEqual(original_calls[0][1]['runtime_mode'], 'activator_bypass')

    def test_prompt_encoder_forwards_return_taps_in_bypass_without_breaking_calls(self):
        trainer = self._trainer('a1')
        original_calls = []

        class _SD:
            text_activator_runtime_mode = 'activator_bypass'

            def get_prompt_embeds(self, prompt, **kwargs):
                original_calls.append(kwargs)
                return ('plain', prompt)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        trainer._install_trigger_binding_prompt_encoder(SimpleNamespace())

        trainer.sd.get_prompt_embeds([''], return_taps=True)

        self.assertTrue(original_calls[0]['return_taps'])
        self.assertEqual(original_calls[0]['runtime_mode'], 'activator_bypass')

    def test_prompt_encoder_still_rejects_caption_without_placeholder_or_literal(self):
        trainer = self._trainer('a1')

        class _SD:
            text_activator_runtime_mode = 'full'

            def get_prompt_embeds(self, prompt, **kwargs):
                return ('plain', prompt, kwargs)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        trainer._install_trigger_binding_prompt_encoder(SimpleNamespace())

        with self.assertRaisesRegex(ValueError, 'every training caption must contain'):
            trainer.sd.get_prompt_embeds(['caption without the required token'])

    def test_phase_metrics_are_written_independently_and_once_per_step(self):
        trainer = self._trainer('a1')
        trainer.step_num = 7
        trainer._trigger_binding_last_metrics = {'gain': 0.25}
        trainer._trigger_binding_last_metrics_written_step = None
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.save_root = temp_dir
            trainer.three_phase_trigger_training.run_root = temp_dir
            trainer.three_phase_trigger_training.artifacts = SimpleNamespace(
                phase_a1=SimpleNamespace(metrics_file='metrics.jsonl'),
            )
            trainer._write_trigger_binding_metrics(torch.tensor(0.5))
            trainer._write_trigger_binding_metrics(torch.tensor(0.75))
            metrics_path = Path(temp_dir) / 'phase_a1' / 'metrics.jsonl'
            records = metrics_path.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(records), 1)
            self.assertIn('"step": 7', records[0])
            self.assertIn('"loss": 0.5', records[0])
            self.assertIn('"gain": 0.25', records[0])

    def test_a_phase_single_source_fallback_shares_latent_noise_timestep_and_target(self):
        from toolkit import trigger_binding_losses

        trainer = self._trainer('a1')
        trainer.device_torch = torch.device('cpu')
        trainer.do_long_prompts = False
        trainer.additional_logs = {}
        trainer.sd = SimpleNamespace(
            encode_prompt=lambda prompts, **kwargs: _FakeEmbeds(len(prompts)),
            get_loss_target=lambda noise, batch, timesteps: noise + 1,
        )
        calls = []

        def predict(noisy_latents, timesteps, **kwargs):
            calls.append((noisy_latents, timesteps, kwargs['conditional_embeds']))
            if getattr(trainer, '_mode', None) == 'activator_bypass':
                return noisy_latents.detach() + 1.0
            return noisy_latents * trainer.text_activator.embedding.weight.mean()

        trainer.predict_noise = predict
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)
        trainer._check_first_trigger_gradient = lambda *_args: None
        batch = SimpleNamespace(
            file_items=[SimpleNamespace(caption_template='x [trigger]', raw_caption='unused')],
            latents=torch.zeros(1, 2),
        )
        noisy = torch.randn(1, 2)
        noise = torch.randn(1, 2)
        timesteps = torch.tensor([10])
        trainer._trigger_binding_modules = {'losses': trigger_binding_losses}
        trainer._write_trigger_binding_metrics = lambda _loss: None
        loss = trainer._calculate_trigger_binding_loss(
            noisy, noise, timesteps, batch, {}, 1.0, torch.float32
        )
        self.assertTrue(loss.requires_grad)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[0] is noisy and call[1] is timesteps for call in calls))
        self.assertIn('a1/source/primary/diffusion_mse', trainer._trigger_binding_last_metrics)
        self.assertIn('a1/aggregate_loss', trainer._trigger_binding_last_metrics)

    def test_paired_sources_use_all_templates_and_add_context_once(self):
        from toolkit import trigger_binding_losses

        trainer = self._trainer('a2')
        trainer.device_torch = torch.device('cpu')
        trainer.do_long_prompts = False
        trainer.step_num = 0
        trainer.additional_logs = {}
        trainer.three_phase_trigger_training.phase_runtime.caption_sources = {
            'enabled': True,
            'sources': [{'name': 'structured'}, {'name': 'natural'}],
            'schedule': {
                'normalize_weights': True,
                'keyframes': [{'step': 0, 'structured': 0.75, 'natural': 0.25}],
            },
        }
        phase = trainer._phase_config()
        phase.context_consistency.enabled = True
        phase.context_consistency.weight = 0.5
        phase.context_consistency.tap_layers = list(range(13))
        encoded_prompts = []

        def encode(prompts, **kwargs):
            encoded_prompts.append((tuple(prompts), kwargs.get('return_taps')))
            token_count = 3 if prompts[0].startswith('S') else 5
            active = getattr(trainer, '_mode', None) == 'full'
            embeds = _FakeEmbeds(len(prompts), token_count=token_count, active=active, with_taps=True)
            if active and prompts[0].startswith('Natural'):
                for taps in embeds.text_taps:
                    taps[:, 0, 0] = 0.0
                    taps[:, 0, 1] = 1.0
            return embeds

        trainer.sd = SimpleNamespace(
            encode_prompt=encode,
            get_loss_target=lambda noise, batch, timesteps: noise,
        )
        trainer.predict_noise = lambda noisy_latents, **kwargs: (
            noisy_latents * trainer.text_activator.embedding.weight.mean()
            if getattr(trainer, '_mode', None) == 'full' else noisy_latents.detach() + 1.0
        )
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)
        trainer._check_first_trigger_gradient = lambda *_args: None
        trainer._trigger_binding_modules = {'losses': trigger_binding_losses}
        trainer._write_trigger_binding_metrics = lambda _loss: None
        items = [SimpleNamespace(
            caption_template='fallback [trigger]',
            raw_caption='unused',
            caption_source_templates={'structured': 'S [trigger]', 'natural': 'Natural words [trigger]'},
        )]
        batch = SimpleNamespace(
            file_items=items,
            get_caption_source_templates=lambda names: [item.caption_source_templates[name] for item, name in zip(items, names)],
        )
        noisy = torch.ones(1, 2)
        loss = trainer._calculate_trigger_binding_loss(
            noisy, torch.zeros_like(noisy), torch.tensor([5]), batch, {}, 1.0, torch.float32
        )
        self.assertTrue(loss.requires_grad)
        self.assertEqual(len(encoded_prompts), 4)
        self.assertTrue(all(return_taps for _, return_taps in encoded_prompts))
        metrics = trainer._trigger_binding_last_metrics
        self.assertEqual(metrics['a2/source_weight/structured'], 0.75)
        self.assertEqual(metrics['a2/source_weight/natural'], 0.25)
        self.assertIn('a2/source/structured/activator_gain', metrics)
        self.assertIn('a2/source/natural/gain_floor_loss', metrics)
        self.assertGreater(metrics['a2/context_weighted'], 0.0)
        self.assertAlmostEqual(
            metrics['a2/aggregate_loss'],
            metrics['a2/aggregate_source_objective'] + metrics['a2/context_weighted'],
            places=6,
        )

    def test_first_real_loss_reachability_runs_once_and_raises(self):
        trainer = self._trainer('a1')
        trainer.params = list(trainer.text_activator.parameters())
        trainer.optimizer = torch.optim.SGD(trainer.params, lr=0.1)
        trainer._trigger_gradient_reachability_checked = False
        loss = trainer.text_activator.embedding.weight.square().mean()
        calls = []

        def check(*args, **kwargs):
            calls.append((args, kwargs))

        fake_module = SimpleNamespace(check_gradient_reachability=check)
        with patch('importlib.import_module', return_value=fake_module):
            trainer._check_first_trigger_gradient(loss, torch.ones(1), torch.zeros(1))
            trainer._check_first_trigger_gradient(loss, torch.ones(1), torch.zeros(1))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]['raise_on_error'])

        trainer._trigger_gradient_reachability_checked = False
        fake_module.check_gradient_reachability = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('unreachable'))
        with patch('importlib.import_module', return_value=fake_module):
            with self.assertRaisesRegex(RuntimeError, 'unreachable'):
                trainer._check_first_trigger_gradient(loss, torch.ones(1), torch.zeros(1))
        self.assertFalse(trainer._trigger_gradient_reachability_checked)

    def test_context_enabled_fails_fast_without_paired_sources(self):
        trainer = self._trainer('a2')
        trainer._phase_config().context_consistency.enabled = True
        trainer.three_phase_trigger_training.phase_runtime.caption_sources = {}
        trainer.sd = SimpleNamespace()
        batch = SimpleNamespace(file_items=[SimpleNamespace(caption_template='x [trigger]', raw_caption='x')])
        with self.assertRaisesRegex(ValueError, 'requires at least two'):
            trainer._calculate_trigger_binding_loss(
                torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([1]), batch, {}, 1.0, torch.float32
            )


if __name__ == '__main__':
    unittest.main()
