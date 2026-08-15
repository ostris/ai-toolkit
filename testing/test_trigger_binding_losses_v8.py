import unittest

import torch

from toolkit.trigger_binding_losses import (
    adjacent_response_hierarchy_loss,
    causal_response_decomposition,
    clamped_response_floor,
    condition_local_response_target,
    huber_response_floor,
    off_direction_penalty,
    per_item_response_mse,
    soft_response_floor,
    structured_natural_effect_consistency,
)


class TriggerBindingLossesV8Test(unittest.TestCase):
    def test_condition_local_target_endpoints_and_per_item_mse(self):
        base = torch.tensor([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
        target = torch.tensor([[5.0, 7.0], [6.0, 8.0]])
        rho = torch.tensor([0.0, 1.0])
        response_target = condition_local_response_target(base, target, rho)
        torch.testing.assert_close(response_target[0], base.detach()[0])
        torch.testing.assert_close(response_target[1], target[1])
        prediction = torch.tensor([[2.0, 4.0], [5.0, 9.0]], requires_grad=True)
        mse = per_item_response_mse(prediction, base, target, rho)
        torch.testing.assert_close(mse, torch.tensor([1.0, 1.0]))
        mse.sum().backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(base.grad)

    def test_condition_local_target_supports_fractional_per_item_rho(self):
        base = torch.zeros(2, 2, 2)
        target = torch.full_like(base, 4.0)
        result = condition_local_response_target(base, target, torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(result[0], torch.ones(2, 2))
        torch.testing.assert_close(result[1], torch.full((2, 2), 3.0))
        with self.assertRaisesRegex(ValueError, 'closed interval'):
            condition_local_response_target(base, target, 1.1)

    def test_condition_local_decomposition_matches_exact_epsilon_identity(self):
        base = torch.tensor([[1.0, -1.0, 2.0], [0.5, 2.0, -3.0]])
        target = torch.tensor([[3.0, 2.0, -2.0], [1.5, -1.0, 1.0]])
        response = torch.tensor([[2.0, 0.0, 0.0], [1.0, 1.0, -1.0]], requires_grad=True)
        epsilon = 1.0e-4
        diagnostics = causal_response_decomposition(
            response,
            base,
            target,
            epsilon=epsilon,
            omega_tolerance=1.0e-6,
        )
        expected_error = epsilon / (diagnostics.base_mse.detach() + epsilon)
        torch.testing.assert_close(
            diagnostics.reconstruction_error,
            expected_error,
            rtol=1.0e-5,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            diagnostics.old_gain,
            diagnostics.reconstructed_gain + diagnostics.reconstruction_error,
        )
        self.assertFalse(diagnostics.uses_shared_reference)
        self.assertTrue(diagnostics.all_omega_within_tolerance)

    def test_shared_reference_decomposition_uses_common_direction(self):
        base = torch.tensor([[10.0, -2.0], [-4.0, 3.0]])
        target = torch.zeros_like(base)
        shared_direction = torch.tensor([2.0, -1.0])
        response = (base + 0.5 * shared_direction).requires_grad_()
        diagnostics = causal_response_decomposition(
            response,
            base,
            target,
            v_ref=shared_direction,
            epsilon=1.0e-8,
        )
        torch.testing.assert_close(diagnostics.alpha, torch.full((2,), 0.5))
        torch.testing.assert_close(diagnostics.beta, torch.full((2,), 0.25))
        torch.testing.assert_close(diagnostics.omega, torch.zeros(2), atol=1.0e-7, rtol=0.0)
        self.assertTrue(diagnostics.uses_shared_reference)
        diagnostics.alpha.sum().backward()
        torch.testing.assert_close(
            response.grad,
            shared_direction.expand_as(response) / shared_direction.square().sum(),
        )

    def test_alpha_floor_gradient_moves_response_toward_target_direction(self):
        base = torch.zeros(1, 2)
        target = torch.tensor([[2.0, 0.0]])
        response = torch.tensor([[0.2, 0.5]], requires_grad=True)
        diagnostics = causal_response_decomposition(response, base, target, epsilon=1.0e-8)
        loss = huber_response_floor(diagnostics.alpha, 0.8, delta=0.2).mean()
        loss.backward()
        self.assertLess(response.grad[0, 0].item(), 0.0)
        self.assertAlmostEqual(response.grad[0, 1].item(), 0.0, places=7)

    def test_off_direction_penalty_removes_orthogonal_component(self):
        base = torch.zeros(1, 2)
        target = torch.tensor([[2.0, 0.0]])
        response = torch.tensor([[1.0, 1.0]], requires_grad=True)
        diagnostics = causal_response_decomposition(response, base, target, epsilon=1.0e-8)
        loss = off_direction_penalty(diagnostics.omega, tolerance=0.0).mean()
        loss.backward()
        self.assertAlmostEqual(response.grad[0, 0].item(), 0.0, places=6)
        self.assertGreater(response.grad[0, 1].item(), 0.0)

    def test_omega_roundoff_is_accepted_with_tolerance(self):
        base = torch.zeros(2, 1024, dtype=torch.float32)
        target = torch.randn(2, 1024, generator=torch.Generator().manual_seed(7))
        response = 0.37 * target
        diagnostics = causal_response_decomposition(
            response,
            base,
            target,
            epsilon=1.0e-12,
            omega_tolerance=2.0e-6,
        )
        self.assertGreaterEqual(diagnostics.omega.min().item(), -2.0e-6)
        self.assertTrue(diagnostics.all_omega_within_tolerance)
        torch.testing.assert_close(
            off_direction_penalty(diagnostics.omega, tolerance=2.0e-6),
            torch.zeros(2),
        )

    def test_floor_variants_are_finite_and_clamped(self):
        response = torch.tensor([-10.0, 0.9, 1.1], requires_grad=True)
        soft = soft_response_floor(response, 1.0, temperature=0.1)
        huber = huber_response_floor(response, 1.0, delta=0.2)
        clamped = clamped_response_floor(response, 1.0, max_deficit=0.5)
        self.assertTrue(torch.isfinite(soft).all())
        torch.testing.assert_close(huber[2], torch.tensor(0.0))
        torch.testing.assert_close(clamped, torch.tensor([0.25, 0.01, 0.0]))
        clamped.mean().backward()
        self.assertEqual(response.grad[0].item(), 0.0)
        self.assertLess(response.grad[1].item(), 0.0)
        self.assertEqual(response.grad[2].item(), 0.0)

    def test_hierarchy_uses_class_means_not_itemwise_ordering(self):
        result = adjacent_response_hierarchy_loss(
            {
                'far': torch.tensor([0.8, -0.8]),
                'neutral': torch.tensor([-0.7, 0.9]),
                'hard': torch.tensor([0.2, 0.4]),
                'trigger': torch.tensor([0.6, 0.8]),
            },
            margins=[0.05, 0.05, 0.05],
            mode='clamped',
            max_deficit=1.0,
        )
        torch.testing.assert_close(result.class_means['far'], torch.tensor(0.0))
        torch.testing.assert_close(result.class_means['neutral'], torch.tensor(0.1))
        torch.testing.assert_close(result.loss, torch.tensor(0.0))
        self.assertEqual(set(result.adjacent_losses), {'far->neutral', 'neutral->hard', 'hard->trigger'})

    def test_hierarchy_gradient_pushes_adjacent_class_means_apart(self):
        far = torch.tensor([0.0, 0.0], requires_grad=True)
        neutral = torch.tensor([0.0, 0.0], requires_grad=True)
        hard = torch.tensor([0.0, 0.0], requires_grad=True)
        trigger = torch.tensor([0.0, 0.0], requires_grad=True)
        result = adjacent_response_hierarchy_loss(
            {'far': far, 'neutral': neutral, 'hard': hard, 'trigger': trigger},
            margins=0.2,
            mode='huber',
            huber_delta=0.5,
        )
        result.loss.backward()
        self.assertGreater(far.grad.mean().item(), 0.0)
        self.assertLess(trigger.grad.mean().item(), 0.0)
        self.assertAlmostEqual(neutral.grad.mean().item(), 0.0, places=7)
        self.assertAlmostEqual(hard.grad.mean().item(), 0.0, places=7)

    def test_structured_natural_consistency_supports_paired_and_mean(self):
        structured = torch.tensor([0.1, 0.9], requires_grad=True)
        natural = torch.tensor([0.4, 0.6], requires_grad=True)
        paired = structured_natural_effect_consistency(structured, natural)
        mean_only = structured_natural_effect_consistency(structured, natural, reduction='mean')
        torch.testing.assert_close(paired.per_item, torch.tensor([0.09, 0.09]))
        torch.testing.assert_close(mean_only.loss, torch.tensor(0.0))
        paired.loss.backward()
        self.assertLess(structured.grad[0].item(), 0.0)
        self.assertGreater(structured.grad[1].item(), 0.0)
        self.assertGreater(natural.grad[0].item(), 0.0)
        self.assertLess(natural.grad[1].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
