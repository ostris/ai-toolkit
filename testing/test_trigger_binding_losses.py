import unittest

import torch

from toolkit.trigger_binding_losses import (
    activator_gain_floor_hinge,
    aggregate_paired_source_losses,
    compute_a1_loss,
    compute_a2_loss,
    delta_context_consistency,
    normalized_activator_gain,
    per_item_diffusion_mse,
    scheduled_gain_floor,
)


class TriggerBindingLossesTest(unittest.TestCase):
    def test_per_item_diffusion_mse(self):
        prediction = torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]])
        target = torch.tensor([[[0.0, 1.0]], [[2.0, 2.0]]])
        result = per_item_diffusion_mse(prediction, target)
        torch.testing.assert_close(result, torch.tensor([2.5, 2.0]))

    def test_normalized_gain_detaches_denominator(self):
        activator_loss = torch.tensor([2.0], requires_grad=True)
        bypass_loss = torch.tensor([4.0], requires_grad=True)
        gain = normalized_activator_gain(activator_loss, bypass_loss)
        gain.sum().backward()
        self.assertAlmostEqual(gain.item(), 0.5000001, places=5)
        self.assertLess(activator_loss.grad.item(), 0.0)
        self.assertIsNone(bypass_loss.grad)

    def test_gain_floor_schedule_and_hinge(self):
        keyframes = [
            {'step': 0, 'value': 0.0},
            {'step': 100, 'value': 0.2},
        ]
        self.assertEqual(scheduled_gain_floor(-1, keyframes, 'linear'), 0.0)
        self.assertAlmostEqual(scheduled_gain_floor(50, keyframes, 'linear'), 0.1)
        self.assertEqual(scheduled_gain_floor(200, keyframes, 'smoothstep'), 0.2)
        gain = torch.tensor([0.05, 0.15], requires_grad=True)
        loss = activator_gain_floor_hinge(gain, 0.1)
        torch.testing.assert_close(loss, torch.tensor([0.05, 0.0]))
        loss.sum().backward()
        torch.testing.assert_close(gain.grad, torch.tensor([-1.0, 0.0]))

    def test_context_consistency_cosine_magnitude_mask_gate_and_warmup(self):
        shape = (1, 13, 3, 2)
        bypass = torch.zeros(shape)
        reference_bypass = torch.zeros(shape)
        reference_on = torch.zeros(shape)
        on = torch.zeros(shape, requires_grad=True)
        reference_on[:, :, 1] = torch.tensor([1.0, 0.0])
        reference_on[:, :, 2] = torch.tensor([2.0, 0.0])
        with torch.no_grad():
            on[:, :, 1] = torch.tensor([0.0, 1.0])
            on[:, :, 2] = torch.tensor([1.0, 0.0])
        trigger_mask = torch.tensor([[True, False, False]])
        token_mask = torch.tensor([[True, True, True]])
        result = delta_context_consistency(
            on,
            bypass,
            reference_on,
            reference_bypass,
            token_mask=token_mask,
            trigger_mask=trigger_mask,
            mask_mode='nontrigger',
            cosine_weight=1.0,
            magnitude_weight=0.5,
            min_delta_norm=0.5,
            step=5,
            warmup_steps=10,
        )
        self.assertAlmostEqual(result.warmup_scale, 0.5)
        self.assertEqual(result.valid_taps_per_item.item(), 26.0)
        self.assertAlmostEqual(result.cosine_per_item.item(), 0.5, places=6)
        self.assertAlmostEqual(result.magnitude_per_item.item(), 0.25, places=6)
        self.assertAlmostEqual(result.per_item.item(), 0.3125, places=6)
        result.loss.backward()
        self.assertIsNotNone(on.grad)
        self.assertIsNone(bypass.grad)

    def test_context_trigger_mask_only_selects_trigger_tokens(self):
        reference_on = torch.zeros(1, 13, 2, 2)
        reference_on[:, :, 0, 0] = 1.0
        reference_on[:, :, 1, 0] = 1.0
        on = reference_on.clone()
        on[:, :, 1] = torch.tensor([0.0, 1.0])
        result = delta_context_consistency(
            on,
            torch.zeros_like(on),
            reference_on,
            torch.zeros_like(reference_on),
            trigger_mask=torch.tensor([[True, False]]),
            mask_mode='trigger',
        )
        torch.testing.assert_close(result.per_item, torch.zeros(1))
        self.assertEqual(result.valid_taps_per_item.item(), 13.0)

    def test_context_rejects_non_13_tap_input(self):
        taps = torch.zeros(1, 12, 2, 3)
        with self.assertRaisesRegex(ValueError, 'expected 13'):
            delta_context_consistency(taps, taps, taps, taps)

    def test_paired_source_aggregation(self):
        aggregate, weighted, weights = aggregate_paired_source_losses(
            {
                'json': torch.tensor([1.0, 3.0]),
                'natural': torch.tensor([5.0, 1.0]),
            },
            {'json': 3.0, 'natural': 1.0},
        )
        torch.testing.assert_close(aggregate, torch.tensor([2.0, 2.5]))
        torch.testing.assert_close(weighted['json'], torch.tensor([0.75, 2.25]))
        self.assertEqual(weights, {'json': 0.75, 'natural': 0.25})

    def test_a1_result_has_detailed_metrics(self):
        prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        target = torch.zeros_like(prediction)
        result = compute_a1_loss(prediction, target)
        self.assertEqual(result.per_item.shape, (2,))
        self.assertIn('a1/diffusion_mse', result.metrics)
        self.assertIn('a1/source/primary', result.metrics)
        result.loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_a1_gain_floor_cannot_worsen_bypass(self):
        activator_prediction = torch.tensor([[1.0, 1.0]], requires_grad=True)
        bypass_prediction = torch.tensor([[2.0, 2.0]], requires_grad=True)
        target = torch.zeros_like(activator_prediction)
        result = compute_a1_loss(
            activator_prediction,
            target,
            bypass_prediction=bypass_prediction,
            gain_floor=0.8,
            diffusion_weight=0.0,
            gain_floor_weight=1.0,
        )
        result.loss.backward()
        self.assertGreater(activator_prediction.grad.abs().sum().item(), 0.0)
        self.assertIsNone(bypass_prediction.grad)
        self.assertIn('a1/activator_gain', result.metrics)

    def test_a2_cannot_get_gradient_by_worsening_bypass(self):
        activator_prediction = torch.tensor([[1.0, 1.0]], requires_grad=True)
        bypass_prediction = torch.tensor([[2.0, 2.0]], requires_grad=True)
        target = torch.zeros_like(activator_prediction)
        result = compute_a2_loss(
            activator_prediction,
            bypass_prediction,
            target,
            gain_floor=0.8,
            diffusion_weight=0.0,
            gain_floor_weight=1.0,
        )
        self.assertGreater(result.gain_floor_per_item.item(), 0.0)
        result.loss.backward()
        self.assertIsNotNone(activator_prediction.grad)
        self.assertGreater(activator_prediction.grad.abs().sum().item(), 0.0)
        self.assertIsNone(bypass_prediction.grad)

    def test_a2_combined_objective_still_never_updates_bypass(self):
        activator_prediction = torch.tensor([[1.0, -1.0]], requires_grad=True)
        bypass_prediction = torch.tensor([[0.5, -0.5]], requires_grad=True)
        target = torch.zeros_like(activator_prediction)
        result = compute_a2_loss(
            activator_prediction,
            bypass_prediction,
            target,
            gain_floor=0.5,
            diffusion_weight=1.0,
            gain_floor_weight=1.0,
        )
        result.loss.backward()
        self.assertGreater(activator_prediction.grad.abs().sum().item(), 0.0)
        self.assertIsNone(bypass_prediction.grad)
        self.assertIn('a2/activator_gain', result.metrics)
        self.assertIn('a2/gain_floor_satisfied', result.metrics)
        self.assertIn('a2/bypass_diffusion_mse', result.metrics)


if __name__ == '__main__':
    unittest.main()
