import json
import os
import tempfile
import unittest
from collections import OrderedDict

import yaml

from extensions_built_in.sd_trainer.ThreePhaseTriggerTrainer import ThreePhaseTriggerTrainer
from toolkit.config_modules import (
    ThreePhaseTriggerTrainingConfig,
    validate_three_phase_trigger_training_config,
)


def _three_phase_block(enabled=True):
    return {
        'enabled': enabled,
        'trigger': {
            'placeholder': '[trigger]',
            'literal': '<r1X1dOn9mA2>',
            'span_detection': 'offsets',
            'mask_all_occurrences': True,
            'occurrence_mode': 'additive',
        },
        'text_activator': {
            'embedding': {
                'enabled': True,
                'tokens': 1,
                'init_mode': 'semantic',
                'init_words': 'illustration',
            },
            'te_adapter': {'enabled': False},
            'tap_adapters': {'enabled': False},
        },
        'reachability_probe': {'enabled': True},
        'phase_a1': {
            'enabled': True,
            'steps': 10,
            'optimizer': 'adamw',
            'learning_rates': {'embedding': 0.001},
            'train': {'train_embedding': True},
            'save_steps': [5, 10],
        },
        'phase_b': {
            'enabled': True,
            'steps': 20,
            'optimizer': 'adamw8bit',
            'optimizer_params': {'weight_decay': 0.00005},
            'learning_rates': {'diffusion_lora': 0.00008},
            'train': {'train_unet': True, 'train_text_encoder': False},
            'text_activator_source': {'phase': 'a1', 'step': 'final'},
            'save_steps': [10, 20],
        },
        'phase_a2': {
            'enabled': True,
            'steps': 8,
            'optimizer': 'adamw',
            'learning_rates': {'embedding': 0.0001},
            'train': {'train_embedding': True},
            'text_activator_source': {'phase': 'a1', 'step': 'final'},
            'diffusion_lora_source': {'phase': 'b', 'step': 'final'},
            'save_steps': [4, 8],
            'losses': {
                'context_consistency': {
                    'enabled': True,
                    'weight': 0.05,
                    'loss_type': 'cosine',
                    'warmup_steps': 2,
                    'min_delta_norm': 1.0e-6,
                },
                'activator_gain_floor': {
                    'enabled': True,
                    'weight': 1.0,
                    'schedule': {
                        'keyframes': [
                            {'step': 0, 'value': 0.0},
                            {'step': 8, 'value': 0.1},
                        ],
                    },
                },
            },
        },
    }


class _Job:
    def __init__(self, process_config):
        self.name = 'parent_job'
        self.meta = OrderedDict({'purpose': 'unit-test'})
        self.raw_config = OrderedDict({
            'job': 'extension',
            'config': OrderedDict({
                'name': self.name,
                'process': [process_config],
            }),
            'meta': self.meta,
        })


class ThreePhaseTriggerTrainingConfigTest(unittest.TestCase):
    def test_disabled_config_keeps_legacy_path_compatible(self):
        validate_three_phase_trigger_training_config(
            ThreePhaseTriggerTrainingConfig(enabled=False),
            None,
        )

    def test_valid_three_phase_config(self):
        config = ThreePhaseTriggerTrainingConfig(**_three_phase_block())
        validate_three_phase_trigger_training_config(config, '<r1X1dOn9mA2>')
        self.assertEqual(config.get_phase('b').steps, 20)
        self.assertEqual(config.literal, '<r1X1dOn9mA2>')

    def test_enabled_config_requires_native_placeholder(self):
        raw = _three_phase_block()
        raw['trigger']['placeholder'] = '<trigger>'
        with self.assertRaisesRegex(ValueError, 'native'):
            validate_three_phase_trigger_training_config(
                ThreePhaseTriggerTrainingConfig(**raw),
                '<r1X1dOn9mA2>',
            )

    def test_enabled_config_rejects_invalid_phase_dependency(self):
        raw = _three_phase_block()
        raw['phase_a2']['diffusion_lora_source']['phase'] = 'a1'
        with self.assertRaisesRegex(ValueError, 'phase b'):
            validate_three_phase_trigger_training_config(
                ThreePhaseTriggerTrainingConfig(**raw),
                '<r1X1dOn9mA2>',
            )

    def test_enabled_config_rejects_missing_trainable_component(self):
        raw = _three_phase_block()
        raw['phase_b']['train'] = {'train_unet': False}
        with self.assertRaisesRegex(ValueError, 'trainable component'):
            validate_three_phase_trigger_training_config(
                ThreePhaseTriggerTrainingConfig(**raw),
                '<r1X1dOn9mA2>',
            )


class ThreePhaseTriggerTrainerTest(unittest.TestCase):
    def test_disabled_orchestrator_is_noop_compatible(self):
        process_config = OrderedDict({
            'type': 'three_phase_trigger_trainer',
            'name': 'disabled_binding_run',
            'three_phase_trigger_training': {'enabled': False},
        })
        process = ThreePhaseTriggerTrainer(0, _Job(process_config), process_config)
        process.run()
        self.assertFalse(hasattr(process, 'run_root'))

    def _make_process(self, temp_dir):
        process_config = OrderedDict({
            'type': 'three_phase_trigger_trainer',
            'name': 'binding_run',
            'training_folder': temp_dir,
            'trigger_word': '<r1X1dOn9mA2>',
            'network': {'type': 'lora', 'linear': 32},
            'train': {'dtype': 'bf16', 'steps': 999, 'optimizer': 'adamw'},
            'model': {'name_or_path': 'test/model', 'arch': 'ideogram4'},
            'datasets': [{'folder_path': 'dataset'}],
            'save': {'save_every': 100},
            'sample': {'samples': []},
            'three_phase_trigger_training': _three_phase_block(),
        })
        return ThreePhaseTriggerTrainer(0, _Job(process_config), process_config)

    def test_build_child_config_maps_phase_and_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            child_job = process.build_child_job_config('a2')
            child = child_job['config']['process'][0]
            self.assertEqual(child['type'], 'sd_trainer')
            self.assertEqual(child['name'], 'phase_a2')
            self.assertEqual(child['train']['steps'], 8)
            self.assertEqual(child['train']['optimizer'], 'adamw')
            self.assertAlmostEqual(child['train']['embedding_lr'], 0.0001)
            self.assertTrue(child['train']['train_embedding'])
            runtime = child['three_phase_trigger_training']['runtime']
            self.assertEqual(runtime['active_phase'], 'a2')
            self.assertTrue(runtime['orchestrated'])
            sources = child['three_phase_trigger_training']['phase_runtime']['sources']
            self.assertEqual(
                sources['embedding'],
                os.path.join(process.run_root, 'phase_a1', 'final', 'trigger_embedding.safetensors'),
            )
            self.assertEqual(
                sources['diffusion_lora'],
                os.path.join(process.run_root, 'phase_b', 'final', 'diffusion_lora.safetensors'),
            )

    def test_snapshot_and_completion_contract_are_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            snapshot_path = process.write_phase_snapshot('a1')
            contract_path = process.write_completion_contract('a1', 'completed', 0)
            self.assertTrue(os.path.isfile(snapshot_path))
            self.assertTrue(os.path.isfile(contract_path))
            with open(snapshot_path, 'r', encoding='utf-8') as handle:
                snapshot = yaml.safe_load(handle)
            self.assertEqual(snapshot['config']['process'][0]['type'], 'sd_trainer')
            with open(contract_path, 'r', encoding='utf-8') as handle:
                contract = json.load(handle)
            self.assertEqual(contract['status'], 'completed')
            self.assertEqual(contract['return_code'], 0)
            self.assertEqual(contract['phase'], 'a1')
            self.assertTrue(contract['artifacts']['embedding'].endswith('trigger_embedding.safetensors'))


if __name__ == '__main__':
    unittest.main()
