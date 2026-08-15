import hashlib
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
            'caption_sources': {
                'enabled': True,
                'sources': [
                    {
                        'name': 'json',
                        'use_main_dataset': True,
                        'caption_ext': '.json',
                        'format': 'json',
                    },
                ],
            },
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
            'caption_sources': {
                'enabled': True,
                'sources': [
                    {
                        'name': 'phase_b_json',
                        'use_main_dataset': True,
                        'caption_ext': '.json',
                        'format': 'json',
                    },
                ],
            },
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
            'caption_sources': {
                'enabled': True,
                'sources': [
                    {
                        'name': 'json',
                        'use_main_dataset': True,
                        'caption_ext': '.json',
                        'format': 'json',
                    },
                ],
            },
            'save_steps': [4, 8],
            'losses': {
                'context_consistency': {
                    'enabled': True,
                    'weight': 0.05,
                    'alignment': 'trigger_pooled',
                    'mask': 'trigger',
                    'pooling': 'mean',
                    'detach_reference': False,
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

    def test_context_consistency_defaults_to_trigger_pooled_cosine(self):
        raw = _three_phase_block()
        raw['phase_a2']['losses']['context_consistency'] = {'enabled': True, 'weight': 0.05}
        config = ThreePhaseTriggerTrainingConfig(**raw)
        validate_three_phase_trigger_training_config(config, '<r1X1dOn9mA2>')
        consistency = config.phase_a2.context_consistency
        self.assertEqual(consistency.alignment, 'trigger_pooled')
        self.assertEqual(consistency.mask, 'trigger')
        self.assertEqual(consistency.pooling, 'mean')
        self.assertEqual(consistency.loss_type, 'cosine')
        self.assertFalse(consistency.detach_reference)

    def test_context_consistency_rejects_invalid_pooled_mask(self):
        raw = _three_phase_block()
        raw['phase_a2']['losses']['context_consistency']['mask'] = 'nontrigger'
        with self.assertRaisesRegex(ValueError, 'mask is invalid'):
            validate_three_phase_trigger_training_config(
                ThreePhaseTriggerTrainingConfig(**raw),
                '<r1X1dOn9mA2>',
            )

    def test_context_consistency_rejects_non_boolean_detach_reference(self):
        raw = _three_phase_block()
        raw['phase_a2']['losses']['context_consistency']['detach_reference'] = 'false'
        with self.assertRaisesRegex(ValueError, 'detach_reference must be boolean'):
            validate_three_phase_trigger_training_config(
                ThreePhaseTriggerTrainingConfig(**raw),
                '<r1X1dOn9mA2>',
            )

    def test_phase_runtime_is_parsed_into_explicit_fields(self):
        raw = _three_phase_block()
        raw['runtime'] = {
            'active_phase': 'b',
            'orchestrated': True,
            'run_root': '/tmp/run',
            'config_snapshot': '/tmp/phase_b.yaml',
            'completion_contract': '/tmp/phase_b.json',
        }
        raw['phase_runtime'] = {
            'caption_sources': raw['phase_b']['caption_sources'],
            'losses': {'tst_v3': {'enabled': True}},
            'save_steps': [10, 20],
            'resume': {'enabled': True, 'checkpoint': '/tmp/resume.json'},
            'sources': {
                'embedding': '/tmp/embedding.safetensors',
                'diffusion_lora': '/tmp/lora.safetensors',
            },
        }
        config = ThreePhaseTriggerTrainingConfig(**raw)
        validate_three_phase_trigger_training_config(config, '<r1X1dOn9mA2>')
        self.assertEqual(config.phase_runtime.caption_sources['sources'][0]['name'], 'phase_b_json')
        self.assertTrue(config.phase_runtime.losses['tst_v3']['enabled'])
        self.assertEqual(config.phase_runtime.save_steps, [10, 20])
        self.assertEqual(config.phase_runtime.resume.checkpoint, '/tmp/resume.json')
        self.assertEqual(config.phase_runtime.sources.embedding, '/tmp/embedding.safetensors')
        self.assertEqual(config.phase_runtime.sources.diffusion_lora, '/tmp/lora.safetensors')

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
            phase_runtime = child['three_phase_trigger_training']['phase_runtime']
            sources = phase_runtime['sources']
            self.assertEqual(
                sources['embedding'],
                os.path.join(process.run_root, 'phase_a1', 'final', 'trigger_embedding.safetensors'),
            )
            self.assertEqual(
                sources['diffusion_lora'],
                os.path.join(process.run_root, 'phase_b', 'final', 'diffusion_lora.safetensors'),
            )
            self.assertEqual(phase_runtime['caption_sources']['sources'][0]['name'], 'json')
            self.assertEqual(
                child['trigger_selective_training']['caption_sources'],
                phase_runtime['caption_sources'],
            )
            self.assertIsNone(
                child['three_phase_trigger_training']['phase_a2']['text_activator_source']['path']
            )
            self.assertEqual(child['network']['pretrained_lora_path'], sources['diffusion_lora'])

    def test_a1_paired_caption_source_names_resolve_parent_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            process.raw_process_config['trigger_selective_training'] = {
                'enabled': True,
                'caption_sources': {
                    'enabled': True,
                    'sources': [
                        {'name': 'json', 'use_main_dataset': True, 'caption_ext': '.json', 'format': 'json'},
                        {'name': 'natural', 'path': '/mirror', 'caption_ext': '.txt', 'format': 'text'},
                    ],
                },
            }
            process.three_phase_config.phase_a1.caption_sources = {
                'enabled': True,
                'paired': ['json', 'natural'],
                'weights': {'json': 0.5, 'natural': 0.5},
            }
            child = process.build_child_job_config('a1')['config']['process'][0]
            caption_sources = child['trigger_selective_training']['caption_sources']
            self.assertEqual([source['name'] for source in caption_sources['sources']], ['json', 'natural'])
            self.assertEqual(caption_sources['schedule']['keyframes'][0]['json'], 0.5)

    def test_local_phase_sources_take_precedence_when_paired_is_true(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            process.raw_process_config['trigger_selective_training'] = {
                'enabled': True,
                'caption_sources': {
                    'enabled': True,
                    'sources': [
                        {'name': 'json', 'use_main_dataset': True},
                        {'name': 'natural', 'path': '/mirror'},
                    ],
                },
            }
            process.three_phase_config.phase_a1.caption_sources = {
                'enabled': True,
                'paired': True,
                'sources': [
                    {'name': 'structured', 'use_main_dataset': True},
                    {'name': 'natural', 'path': '/structured-mirror'},
                ],
                'weights': {'structured': 0.5, 'natural': 0.5},
            }
            child = process.build_child_job_config('a1')['config']['process'][0]
            caption_sources = child['trigger_selective_training']['caption_sources']
            self.assertEqual(
                [source['name'] for source in caption_sources['sources']],
                ['structured', 'natural'],
            )
            self.assertEqual(caption_sources['schedule']['keyframes'][0]['structured'], 0.5)

    def test_phase_b_caption_sources_override_parent_and_json_only_is_scheduled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            process.raw_process_config['trigger_selective_training'] = {
                'enabled': True,
                'caption_sources': {
                    'enabled': True,
                    'sources': [
                        {'name': 'parent', 'use_main_dataset': True, 'caption_ext': '.txt', 'format': 'text'},
                    ],
                },
            }
            child = process.build_child_job_config('b')['config']['process'][0]
            caption_sources = child['trigger_selective_training']['caption_sources']
            self.assertEqual([source['name'] for source in caption_sources['sources']], ['phase_b_json'])
            self.assertEqual(
                caption_sources['schedule']['keyframes'],
                [{'step': 0, 'phase_b_json': 1.0}],
            )

    def test_phase_b_native_checkpoint_is_published_for_a2_handoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            phase_root = process._phase_root('b')
            os.makedirs(phase_root, exist_ok=True)
            native_path = os.path.join(phase_root, 'phase_b.safetensors')
            with open(native_path, 'wb') as handle:
                handle.write(b'diffusion-lora')

            published = process._publish_phase_b_diffusion_lora()

            expected = os.path.join(phase_root, 'final', 'diffusion_lora.safetensors')
            self.assertEqual(published, expected)
            with open(expected, 'rb') as handle:
                self.assertEqual(handle.read(), b'diffusion-lora')
            process.three_phase_config.phase_a2.text_activator_source.phase = None
            process.three_phase_config.phase_a2.text_activator_source.path = None
            process._verify_phase_inputs('a2')

    def test_completed_phase_b_contract_self_heals_missing_standard_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            phase_root = process._phase_root('b')
            os.makedirs(phase_root, exist_ok=True)
            with open(os.path.join(phase_root, 'phase_b.safetensors'), 'wb') as handle:
                handle.write(b'diffusion-lora')
            process.write_phase_snapshot('b')
            process.write_completion_contract('b', 'completed', 0)

            self.assertTrue(process._contract_is_verified('b'))
            self.assertTrue(os.path.isfile(os.path.join(
                phase_root, 'final', 'diffusion_lora.safetensors'
            )))

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

    def test_completion_contract_hashes_existing_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            input_path = os.path.join(temp_dir, 'embedding.safetensors')
            with open(input_path, 'wb') as handle:
                handle.write(b'phase-input')
            process.three_phase_config.phase_b.text_activator_source.phase = None
            process.three_phase_config.phase_b.text_activator_source.path = input_path
            process.write_phase_snapshot('b')
            contract = process.completion_contract('b', 'running')
            self.assertEqual(contract['inputs']['embedding']['path'], input_path)
            self.assertEqual(
                contract['inputs']['embedding']['sha256'],
                hashlib.sha256(b'phase-input').hexdigest(),
            )

    def test_phase_input_verification_fails_fast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            with self.assertRaisesRegex(FileNotFoundError, 'missing required input artifact'):
                process._verify_phase_inputs('b')

    def test_verified_contract_rejects_changed_input_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._make_process(temp_dir)
            input_path = os.path.join(temp_dir, 'embedding.safetensors')
            with open(input_path, 'wb') as handle:
                handle.write(b'original')
            process.three_phase_config.phase_b.text_activator_source.phase = None
            process.three_phase_config.phase_b.text_activator_source.path = input_path
            process.write_phase_snapshot('b')
            process.write_completion_contract('b', 'completed', 0)
            self.assertTrue(process._contract_is_verified('b'))
            with open(input_path, 'wb') as handle:
                handle.write(b'changed')
            self.assertFalse(process._contract_is_verified('b'))


if __name__ == '__main__':
    unittest.main()
