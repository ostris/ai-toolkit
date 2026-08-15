import json
import os
import random
import tempfile
import unittest

import torch

from toolkit.config_modules import TriggerSelectiveTrainingConfig
from toolkit.data_loader import discover_tst_caption_sources
from toolkit.dataloader_mixins import read_caption_source
from toolkit.trigger_selective_training import (
    apply_differential_guidance_target,
    get_scheduled_caption_source_weights,
    get_scheduled_gain_floor,
    get_scheduled_loss_weights,
    get_scheduled_margin,
    network_disabled,
    normalized_gain,
    resolve_prompt_variants,
    sample_caption_sources,
    sample_negative_styles,
    trigger_advantage_hinge,
    trigger_gain_floor_hinge,
    validate_trigger_selective_config,
)


class _Network:
    def __init__(self):
        self.is_active = True


class TriggerSelectiveTrainingTest(unittest.TestCase):
    def setUp(self):
        self.config = TriggerSelectiveTrainingConfig(
            enabled=True,
            negative_styles={
                'expected_category_count': 2,
                'categories': [
                    {'name': 'neutral', 'probability': 0.5, 'phrases': ['']},
                    {'name': 'hard', 'probability': 0.5, 'phrases': ['painting', 'illustration']},
                ],
            },
            path3={
                'margin_schedule': {
                    'interpolation': 'linear',
                    'keyframes': [{'step': 0, 'value': 0.02}, {'step': 100, 'value': 0.12}],
                },
            },
            loss_schedule={
                'interpolation': 'linear',
                'keyframes': [
                    {'step': 0, 'path1': 0.8, 'path2': 0.1, 'path3': 0.1},
                    {'step': 100, 'path1': 0.6, 'path2': 0.15, 'path3': 0.25},
                ],
            },
        )

    def test_validation_and_schedule_clamping(self):
        validate_trigger_selective_config(self.config, '<trigger>')
        self.assertAlmostEqual(get_scheduled_margin(self.config, 0), 0.02)
        self.assertAlmostEqual(get_scheduled_margin(self.config, 50), 0.07)
        self.assertAlmostEqual(get_scheduled_margin(self.config, 1000), 0.12)
        weights = get_scheduled_loss_weights(self.config, 50)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights['path1'], 0.7)

    def test_sampling_and_shared_placeholder_resolution(self):
        samples = sample_negative_styles(self.config, 8, random.Random(3))
        trigger, decoy = resolve_prompt_variants(
            ['a [trigger] portrait [trigger]'] * 8,
            '<trigger>',
            samples,
        )
        self.assertTrue(all(prompt.count('<trigger>') == 2 for prompt in trigger))
        self.assertTrue(all('[trigger]' not in prompt for prompt in decoy))
        self.assertEqual(len(trigger), len(decoy))

    def test_gain_and_hinge_stop_gradient_on_decoy_gain(self):
        student_loss = torch.tensor([2.0], requires_grad=True)
        base_loss = torch.tensor([4.0])
        decoy_gain = normalized_gain(student_loss, base_loss, 1.0e-6)
        trigger_gain = torch.tensor([0.01], requires_grad=True)
        loss = trigger_advantage_hinge(trigger_gain, decoy_gain, 0.1).sum()
        loss.backward()
        self.assertIsNotNone(trigger_gain.grad)
        self.assertIsNone(student_loss.grad)

    def test_positive_clamped_decoy_gain_keeps_only_positive_gradient(self):
        student_loss = torch.tensor([2.0], requires_grad=True)
        base_loss = torch.tensor([4.0])
        decoy_gain = normalized_gain(student_loss, base_loss, 1.0e-6)
        trigger_gain = torch.tensor([0.01], requires_grad=True)
        loss = trigger_advantage_hinge(
            trigger_gain,
            decoy_gain,
            0.1,
            decoy_gain_mode='positive_clamped',
        ).sum()
        loss.backward()
        self.assertIsNotNone(trigger_gain.grad)
        self.assertIsNotNone(student_loss.grad)
        self.assertLess(student_loss.grad.item(), 0.0)

    def test_positive_clamped_decoy_gain_stops_below_base(self):
        student_loss = torch.tensor([5.0], requires_grad=True)
        base_loss = torch.tensor([4.0])
        decoy_gain = normalized_gain(student_loss, base_loss, 1.0e-6)
        trigger_gain = torch.tensor([0.01], requires_grad=True)
        loss = trigger_advantage_hinge(
            trigger_gain,
            decoy_gain,
            0.1,
            decoy_gain_mode='positive_clamped',
        ).sum()
        loss.backward()
        self.assertIsNotNone(student_loss.grad)
        self.assertEqual(student_loss.grad.item(), 0.0)

    def test_validation_accepts_v2_decoy_gain_mode(self):
        self.config.path3.decoy_gain_mode = 'positive_clamped'
        validate_trigger_selective_config(self.config, '<trigger>')

    def test_differential_guidance_target_is_detached_and_shared(self):
        class _Config:
            do_guidance_loss = True
            do_differential_guidance = True
            differential_guidance_scale = 3.0

        class _Trainer:
            train_config = _Config()

        target = torch.ones(1, 2)
        prediction = torch.zeros(1, 2, requires_grad=True)
        shared = apply_differential_guidance_target(_Trainer(), target, prediction)
        self.assertFalse(shared.requires_grad)
        self.assertTrue(torch.equal(shared, torch.full_like(target, 3.0)))

    def test_differential_guidance_preserves_effective_v3_behavior(self):
        class _Config:
            do_guidance_loss = False
            do_differential_guidance = True
            differential_guidance_scale = 3.0

        class _Trainer:
            train_config = _Config()

        target = torch.ones(1, 2)
        prediction = torch.zeros(1, 2, requires_grad=True)
        shared = apply_differential_guidance_target(_Trainer(), target, prediction)
        self.assertIs(shared, target)

    def test_network_state_is_restored(self):
        network = _Network()
        with network_disabled(network):
            self.assertFalse(network.is_active)
        self.assertTrue(network.is_active)
        network.is_active = False
        with network_disabled(network):
            self.assertFalse(network.is_active)
        self.assertFalse(network.is_active)

    def test_v3_source_schedule_and_gain_floor(self):
        config = TriggerSelectiveTrainingConfig(
            enabled=True,
            caption_sources={
                'enabled': True,
                'sources': [
                    {'name': 'json', 'use_main_dataset': True, 'caption_ext': '.json', 'format': 'json'},
                    {'name': 'natural', 'path': '/mirror', 'caption_ext': '.txt', 'format': 'text'},
                ],
                'schedule': {
                    'interpolation': 'linear',
                    'keyframes': [
                        {'step': 0, 'json': 1.0, 'natural': 0.0},
                        {'step': 100, 'json': 0.5, 'natural': 0.5},
                    ],
                },
            },
            negative_styles={
                'categories': [{'name': 'neutral', 'probability': 1.0, 'phrases': ['']}],
            },
            path3={
                'decoy_gain_mode': 'positive_clamped',
                'margin_schedule': {'keyframes': [{'step': 0, 'value': 0.1}]},
                'gain_floor': {
                    'enabled': True,
                    'weight': 0.5,
                    'schedule': {
                        'interpolation': 'linear',
                        'keyframes': [{'step': 0, 'value': 0.0}, {'step': 100, 'value': 0.2}],
                    },
                },
            },
            loss_schedule={
                'keyframes': [{'step': 0, 'path1': 0.8, 'path2': 0.1, 'path3': 0.1}],
            },
        )
        validate_trigger_selective_config(config, '<trigger>')
        self.assertEqual(get_scheduled_caption_source_weights(config, 0), {'json': 1.0, 'natural': 0.0})
        weights = get_scheduled_caption_source_weights(config, 50)
        self.assertAlmostEqual(weights['json'], 0.75)
        self.assertAlmostEqual(weights['natural'], 0.25)
        selected, _ = sample_caption_sources(config, 0, 8, random.Random(2))
        self.assertEqual(selected, ['json'] * 8)
        self.assertAlmostEqual(get_scheduled_gain_floor(config, 50), 0.1)

    def test_gain_floor_has_trigger_only_gradient(self):
        trigger_gain = torch.tensor([0.05], requires_grad=True)
        decoy_gain = torch.tensor([0.3], requires_grad=True)
        floor_loss = trigger_gain_floor_hinge(trigger_gain, 0.1).sum()
        combined = trigger_advantage_hinge(
            trigger_gain,
            decoy_gain,
            0.1,
            decoy_gain_mode='positive_clamped',
        ).sum() + 0.5 * floor_loss
        combined.backward()
        self.assertLess(trigger_gain.grad.item(), 0.0)
        self.assertGreater(decoy_gain.grad.item(), 0.0)
        floor_only_trigger = torch.tensor([0.05], requires_grad=True)
        floor_only_decoy = torch.tensor([0.3], requires_grad=True)
        floor_only = trigger_gain_floor_hinge(floor_only_trigger, 0.1).sum() + floor_only_decoy * 0
        floor_only.backward()
        self.assertLess(floor_only_trigger.grad.item(), 0.0)
        self.assertEqual(floor_only_decoy.grad.item(), 0.0)

    def test_caption_source_reading_and_relative_pairing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            main_root = os.path.join(temp_dir, 'main')
            mirror_root = os.path.join(temp_dir, 'mirror')
            os.makedirs(os.path.join(main_root, 'nested'))
            os.makedirs(os.path.join(mirror_root, 'nested'))
            main_image = os.path.join(main_root, 'nested', 'item.png')
            mirror_image = os.path.join(mirror_root, 'nested', 'item.png')
            open(main_image, 'wb').close()
            open(mirror_image, 'wb').close()
            json_path = os.path.splitext(main_image)[0] + '.json'
            text_path = os.path.splitext(mirror_image)[0] + '.txt'
            with open(json_path, 'w', encoding='utf-8') as handle:
                json.dump({'caption': 'structured [trigger] caption'}, handle)
            with open(text_path, 'w', encoding='utf-8') as handle:
                handle.write('natural [trigger] caption')
            self.assertEqual(read_caption_source(json_path, 'json', 'caption'), 'structured [trigger] caption')
            config = TriggerSelectiveTrainingConfig(
                caption_sources={
                    'enabled': True,
                    'sources': [
                        {'name': 'json', 'use_main_dataset': True, 'caption_ext': '.json', 'format': 'json'},
                        {'name': 'natural', 'path': mirror_root, 'caption_ext': '.txt', 'format': 'text'},
                    ],
                },
            )
            result = discover_tst_caption_sources(main_root, [main_image], config.caption_sources)
            item = result[os.path.abspath(main_image)]
            self.assertEqual(item['item_id'], os.path.join('nested', 'item.png'))
            self.assertEqual(item['sources']['natural']['caption'], 'natural [trigger] caption')

    def test_json_only_main_caption_source_is_supported(self):
        config = TriggerSelectiveTrainingConfig(
            enabled=True,
            caption_sources={
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
            negative_styles={
                'categories': [{'name': 'neutral', 'probability': 1.0, 'phrases': ['']}],
            },
            path3={
                'margin_schedule': {'keyframes': [{'step': 0, 'value': 0.1}]},
            },
            loss_schedule={
                'keyframes': [{'step': 0, 'path1': 1.0, 'path2': 0.0, 'path3': 0.0}],
            },
        )
        validate_trigger_selective_config(config, '<trigger>')
        self.assertEqual(config.caption_sources.schedule.keyframes, [{'step': 0, 'json': 1.0}])
        selected, probabilities = sample_caption_sources(config, 0, 3, random.Random(1))
        self.assertEqual(selected, ['json', 'json', 'json'])
        self.assertEqual(probabilities, {'json': 1.0})

    def test_disabled_config_does_not_require_tst_fields(self):
        validate_trigger_selective_config(TriggerSelectiveTrainingConfig(enabled=False), None)


if __name__ == '__main__':
    unittest.main()
