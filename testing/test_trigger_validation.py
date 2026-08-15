import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from toolkit.trigger_data_split import (
    assert_no_split_leakage,
    create_data_split_manifest,
    filter_items_for_split,
    get_or_create_data_split_manifest,
    heldout_item_count,
    load_data_split_manifest,
    normalize_dataset_relative_item_id,
    paired_caption_item_id,
    persist_data_split_manifest,
)
from toolkit.trigger_validation import (
    JSONLWriter,
    aggregate_results,
    evaluate_gain,
    isolated_rng,
    make_python_rng,
    make_torch_generator,
    validate_trigger_data_split_config,
    validate_trigger_validation_config,
)


class _ValidationConfig:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get('enabled', False)
        self.every = kwargs.get('every', 0)
        self.seed = kwargs.get('seed', 0)
        self.fixed_timesteps = kwargs.get('fixed_timesteps', [])
        self.fixed_sigmas = kwargs.get('fixed_sigmas', [])
        self.train_probe_manifest = kwargs.get('train_probe_manifest')
        self.heldout_manifest = kwargs.get('heldout_manifest')
        self.data_split_manifest = kwargs.get('data_split_manifest')
        self.caption_sources = kwargs.get('caption_sources', [])
        self.negative_phrases = kwargs.get('negative_phrases', [])
        self.train_probe_output_filename = kwargs.get(
            'train_probe_output_filename', 'train_probe_validation.jsonl'
        )
        self.heldout_output_filename = kwargs.get(
            'heldout_output_filename', 'heldout_validation.jsonl'
        )
        self.aggregate_output_filename = kwargs.get(
            'aggregate_output_filename', 'trigger_validation_aggregate.jsonl'
        )
        self.gain_epsilon = kwargs.get('gain_epsilon', 1.0e-6)


class _DataSplitConfig:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get('enabled', False)
        self.heldout_fraction = kwargs.get('heldout_fraction', 0.1)
        self.seed = kwargs.get('seed', 0)
        self.manifest_path = kwargs.get('manifest_path')
        self.reuse_existing = kwargs.get('reuse_existing', True)


class TriggerValidationTest(unittest.TestCase):
    def test_rng_isolation_restores_all_global_states(self):
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(404)

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

        with isolated_rng(999):
            random.random()
            np.random.rand()
            torch.rand(3)
            if torch.cuda.is_available():
                torch.rand(3, device='cuda')

        self.assertEqual(random.getstate(), python_state)
        restored_numpy = np.random.get_state()
        self.assertEqual(restored_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
        self.assertEqual(restored_numpy[2:], numpy_state[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        for restored, expected in zip(torch.cuda.get_rng_state_all(), cuda_states):
            torch.testing.assert_close(restored, expected)

    def test_independent_generators_do_not_change_global_rng(self):
        random.seed(10)
        torch.manual_seed(20)
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()

        python_rng = make_python_rng(30)
        torch_generator = make_torch_generator(40)
        first_python = python_rng.random()
        first_torch = torch.rand(2, generator=torch_generator)

        self.assertEqual(random.getstate(), python_state)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        self.assertEqual(first_python, make_python_rng(30).random())
        torch.testing.assert_close(first_torch, torch.rand(2, generator=make_torch_generator(40)))

    def test_evaluate_gain_from_scalar_losses(self):
        result = evaluate_gain(
            lambda: 4.0,
            lambda: 2.0,
            lambda: 5.0,
            lambda: 6.0,
            epsilon=1.0e-6,
        )
        expected_trigger = 1.0 - 2.0 / 4.000001
        expected_decoy = 1.0 - 6.0 / 5.000001
        self.assertAlmostEqual(result.trigger_gain, expected_trigger)
        self.assertAlmostEqual(result.decoy_gain, expected_decoy)
        self.assertAlmostEqual(result.raw_gap, expected_trigger - expected_decoy)
        self.assertAlmostEqual(result.effective_gap, expected_trigger)

    def test_evaluate_gain_from_predictions_uses_shared_target(self):
        target = torch.zeros(1, 2)
        result = evaluate_gain(
            lambda: torch.tensor([[2.0, 2.0]]),
            lambda: torch.tensor([[1.0, 1.0]]),
            lambda: torch.tensor([[1.0, 1.0]]),
            lambda: torch.tensor([[0.5, 0.5]]),
            target=target,
        )
        self.assertAlmostEqual(result.base_trigger_loss, 4.0)
        self.assertAlmostEqual(result.student_trigger_loss, 1.0)
        self.assertAlmostEqual(result.base_decoy_loss, 1.0)
        self.assertAlmostEqual(result.student_decoy_loss, 0.25)
        self.assertAlmostEqual(result.trigger_gain, 0.7500000625, places=6)
        self.assertAlmostEqual(result.decoy_gain, 0.75000025, places=6)

    def test_aggregate_results(self):
        first = evaluate_gain(lambda: 4.0, lambda: 2.0, lambda: 4.0, lambda: 4.0)
        second = evaluate_gain(lambda: 2.0, lambda: 3.0, lambda: 2.0, lambda: 2.0)
        aggregate = aggregate_results([first, second])
        self.assertEqual(aggregate['count'], 2)
        self.assertAlmostEqual(
            aggregate['trigger_gain'],
            (first.trigger_gain + second.trigger_gain) / 2.0,
        )
        self.assertAlmostEqual(
            aggregate['effective_gap'],
            (first.effective_gap + second.effective_gap) / 2.0,
        )
        self.assertEqual(aggregate['trigger_gain_positive_rate'], 0.5)
        self.assertEqual(aggregate['effective_gap_positive_rate'], 0.5)

    def test_jsonl_writer_appends_sorted_records(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = JSONLWriter(directory, 'validation.jsonl')
            writer.write({'z': 1, 'name': 'heldout'})
            writer.write({'z': 2, 'name': 'train-probe'})
            with open(writer.path, 'r', encoding='utf-8') as handle:
                lines = handle.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0]), {'name': 'heldout', 'z': 1})
            self.assertEqual(json.loads(lines[1]), {'name': 'train-probe', 'z': 2})

    def test_config_validation_accepts_complete_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            train_probe = os.path.join(directory, 'train_probe.jsonl')
            heldout = os.path.join(directory, 'heldout.jsonl')
            for path in (train_probe, heldout):
                with open(path, 'w', encoding='utf-8') as handle:
                    handle.write('{}\n')
            config = _ValidationConfig(
                enabled=True,
                every=100,
                seed=123,
                fixed_timesteps=[10, 20],
                train_probe_manifest=train_probe,
                heldout_manifest=heldout,
                caption_sources=['json', 'natural'],
                negative_phrases=['painting', 'anime'],
                train_probe_output_filename='train_probe.jsonl',
                heldout_output_filename='heldout.jsonl',
                aggregate_output_filename='aggregate.jsonl',
            )
            validate_trigger_validation_config(config)

    def test_data_split_config_and_single_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            split_path = os.path.join(directory, 'split.json')
            with open(split_path, 'w', encoding='utf-8') as handle:
                handle.write('{}')
            split_config = _DataSplitConfig(
                enabled=True,
                heldout_fraction=0.1,
                seed=42,
                manifest_path=split_path,
            )
            validate_trigger_data_split_config(split_config, require_manifest_file=True)
            validation_config = _ValidationConfig(
                enabled=True,
                every=10,
                fixed_timesteps=[10],
                data_split_manifest=split_path,
                caption_sources=['json'],
                negative_phrases=['painting'],
            )
            validate_trigger_validation_config(validation_config)
            managed_validation = _ValidationConfig(
                enabled=True,
                every=10,
                fixed_timesteps=[10],
                caption_sources=['json'],
                negative_phrases=['painting'],
            )
            validate_trigger_validation_config(
                managed_validation,
                data_split_config=split_config,
            )

    def test_data_split_is_reproducible_and_keeps_both_sides_non_empty(self):
        item_ids = ['z/image.png', 'a/image.png', 'nested\\other.jpg', 'third.webp']
        first = create_data_split_manifest(item_ids, seed=123, heldout_fraction=0.1)
        second = create_data_split_manifest(reversed(item_ids), seed=123, heldout_fraction=0.1)
        self.assertEqual(first, second)
        self.assertEqual(len(first.heldout_item_ids), 1)
        self.assertTrue(first.train_item_ids)
        self.assertTrue(first.heldout_item_ids)
        self.assertEqual(first.heldout_item_ids, tuple(sorted(first.heldout_item_ids)))
        self.assertEqual(heldout_item_count(2, 0.99), 1)
        self.assertEqual(heldout_item_count(15, 0.1), 2)
        self.assertEqual(heldout_item_count(5, 0.1), 1)

    def test_data_split_manifest_persists_and_fails_fast_on_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'split.json')
            manifest = create_data_split_manifest(['a.png', 'b.png', 'c.png'], seed=7)
            persist_data_split_manifest(manifest, path)
            loaded = load_data_split_manifest(path, item_ids=['a.png', 'b.png', 'c.png'])
            self.assertEqual(loaded, manifest)
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                load_data_split_manifest(path, item_ids=['a.png', 'b.png', 'changed.png'])
            reused = get_or_create_data_split_manifest(
                ['a.png', 'b.png', 'c.png'], path, seed=7, heldout_fraction=0.1
            )
            self.assertEqual(reused, manifest)

    def test_data_split_allowlist_and_leakage_helpers(self):
        manifest = create_data_split_manifest(['a.png', 'b.png', 'c.png', 'd.png'], seed=2)
        items = [{'item_id': item_id} for item_id in manifest.train_item_ids + manifest.heldout_item_ids]
        train = filter_items_for_split(items, manifest, 'train')
        heldout = filter_items_for_split(items, manifest, 'heldout')
        self.assertEqual({item['item_id'] for item in train}, set(manifest.train_item_ids))
        self.assertEqual({item['item_id'] for item in heldout}, set(manifest.heldout_item_ids))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            assert_no_split_leakage(['a.png'], ['a.png'])

    def test_item_ids_normalize_and_pair_captions_by_image_id(self):
        self.assertEqual(normalize_dataset_relative_item_id('nested\\image.png'), 'nested/image.png')
        with tempfile.TemporaryDirectory() as directory:
            image_root = os.path.join(directory, 'images')
            os.makedirs(os.path.join(image_root, 'nested'))
            caption = os.path.join(image_root, 'nested', 'image.json')
            image_ids = ['nested/image.png', 'other.png']
            with open(caption, 'w', encoding='utf-8') as handle:
                handle.write('{}')
            self.assertEqual(paired_caption_item_id(caption, image_root, image_ids), 'nested/image.png')

    def test_config_validation_rejects_invalid_paths_and_parameters(self):
        config = _ValidationConfig(
            enabled=True,
            every=0,
            fixed_timesteps=[10],
            train_probe_manifest='missing-train.jsonl',
            heldout_manifest='missing-heldout.jsonl',
            caption_sources=['json'],
            negative_phrases=['painting'],
        )
        with self.assertRaisesRegex(ValueError, 'every must be positive'):
            validate_trigger_validation_config(config, require_manifest_files=False)

        config.every = 10
        config.fixed_sigmas = [1.0]
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            validate_trigger_validation_config(config, require_manifest_files=False)

        config.fixed_sigmas = []
        config.data_split_manifest = 'split.json'
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            validate_trigger_validation_config(config, require_manifest_files=False)

        config.data_split_manifest = None
        config.train_probe_output_filename = '../escape.jsonl'
        with self.assertRaisesRegex(ValueError, 'filename, not a path'):
            validate_trigger_validation_config(config, require_manifest_files=False)


if __name__ == '__main__':
    unittest.main()
