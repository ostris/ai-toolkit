import random
import unittest

import torch

from toolkit.config_modules import TriggerSelectiveTrainingConfig
from toolkit.trigger_selective_training import (
    apply_differential_guidance_target,
    get_scheduled_loss_weights,
    get_scheduled_margin,
    network_disabled,
    normalized_gain,
    resolve_prompt_variants,
    sample_negative_styles,
    trigger_advantage_hinge,
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

    def test_disabled_config_does_not_require_tst_fields(self):
        validate_trigger_selective_config(TriggerSelectiveTrainingConfig(enabled=False), None)


if __name__ == '__main__':
    unittest.main()
