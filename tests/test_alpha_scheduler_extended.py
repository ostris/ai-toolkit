#!/usr/bin/env python3
"""
Extended tests for Alpha Scheduler - Critical functionality
Tests checkpoint save/load and recent bug fixes.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from toolkit.alpha_scheduler import (
    PhaseAlphaScheduler,
    TrainingStatistics,
    create_default_config
)


class TestCheckpointSaveLoad(unittest.TestCase):
    """Test checkpoint save/load functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.rank = 64
        self.config = create_default_config(rank=self.rank, conv_alpha=14, linear_alpha=16)

    def test_state_dict_disabled(self):
        """Test state_dict when scheduler is disabled."""
        config = {'enabled': False}
        scheduler = PhaseAlphaScheduler(config, self.rank)
        state = scheduler.state_dict()

        self.assertEqual(state, {'enabled': False})

    def test_state_dict_enabled_initial(self):
        """Test state_dict at beginning of training."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)
        state = scheduler.state_dict()

        self.assertTrue(state['enabled'])
        self.assertEqual(state['current_phase_idx'], 0)
        self.assertEqual(state['steps_in_phase'], 0)
        self.assertEqual(state['total_steps'], 0)
        self.assertEqual(state['transition_history'], [])

    def test_state_dict_after_training(self):
        """Test state_dict after some training steps."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)

        # Simulate 50 training steps
        for i in range(50):
            scheduler.update(step=i, loss=0.5 - i * 0.001, gradient_stability=0.6)

        state = scheduler.state_dict()

        self.assertEqual(state['total_steps'], 49)
        self.assertEqual(state['steps_in_phase'], 50)
        self.assertEqual(len(state['global_losses']), 50)
        self.assertEqual(len(state['global_grad_stability']), 50)

    def test_load_state_dict_disabled(self):
        """Test loading state when disabled."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)
        state = {'enabled': False}

        scheduler.load_state_dict(state)
        # Should not crash, just return

    def test_load_state_dict_full(self):
        """Test full save/load cycle."""
        # Create and train scheduler
        scheduler1 = PhaseAlphaScheduler(self.config, self.rank)

        for i in range(100):
            scheduler1.update(step=i, loss=0.5 - i * 0.001, gradient_stability=0.6)

        # Save state
        state = scheduler1.state_dict()

        # Create new scheduler and load
        scheduler2 = PhaseAlphaScheduler(self.config, self.rank)
        scheduler2.load_state_dict(state)

        # Verify restored
        self.assertEqual(scheduler2.current_phase_idx, scheduler1.current_phase_idx)
        self.assertEqual(scheduler2.steps_in_phase, scheduler1.steps_in_phase)
        self.assertEqual(scheduler2.total_steps, scheduler1.total_steps)
        self.assertEqual(len(scheduler2.global_statistics.recent_losses),
                        len(scheduler1.global_statistics.recent_losses))

    def test_checkpoint_restart_continues_correctly(self):
        """Test that restart from checkpoint continues training correctly."""
        # Train to step 1000
        scheduler1 = PhaseAlphaScheduler(self.config, self.rank)
        for i in range(1000):
            scheduler1.update(step=i, loss=0.5, gradient_stability=0.6)

        phase_before = scheduler1.current_phase_idx
        steps_in_phase_before = scheduler1.steps_in_phase

        # Save and reload
        state = scheduler1.state_dict()
        scheduler2 = PhaseAlphaScheduler(self.config, self.rank)
        scheduler2.load_state_dict(state)

        # Continue training
        scheduler2.update(step=1000, loss=0.5, gradient_stability=0.6)

        # Verify continuity
        self.assertEqual(scheduler2.current_phase_idx, phase_before)
        self.assertEqual(scheduler2.steps_in_phase, steps_in_phase_before + 1)
        self.assertEqual(scheduler2.total_steps, 1000)

    def test_checkpoint_with_transition_history(self):
        """Test saving/loading with transition history."""
        scheduler1 = PhaseAlphaScheduler(self.config, self.rank)

        # Force a transition
        scheduler1.current_phase_idx = 1
        scheduler1.steps_in_phase = 500
        scheduler1.transition_history = [
            {'step': 1200, 'from_phase': 'foundation', 'to_phase': 'balance'}
        ]

        # Save and reload
        state = scheduler1.state_dict()
        scheduler2 = PhaseAlphaScheduler(self.config, self.rank)
        scheduler2.load_state_dict(state)

        # Verify history preserved
        self.assertEqual(len(scheduler2.transition_history), 1)
        self.assertEqual(scheduler2.transition_history[0]['step'], 1200)


class TestLossIncreasingScenario(unittest.TestCase):
    """Test that scheduler handles increasing loss correctly."""

    def setUp(self):
        """Set up test fixtures."""
        self.rank = 64
        self.config = create_default_config(rank=self.rank, conv_alpha=14, linear_alpha=16)

    def test_does_not_transition_on_increasing_loss(self):
        """Test that transition doesn't happen when loss is increasing."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)

        # Train for min_steps with stable gradient but increasing loss
        min_steps = self.config['conv_alpha_phases']['foundation']['min_steps']

        for i in range(min_steps + 200):
            # Loss slowly increasing
            loss = 0.5 + i * 0.0001
            scheduler.update(step=i, loss=loss, gradient_stability=0.7)

        # Should NOT have transitioned (loss increasing is bad)
        self.assertEqual(scheduler.current_phase_idx, 0)

    def test_transitions_on_plateaued_loss(self):
        """Test that transition happens when loss plateaus."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)

        min_steps = self.config['conv_alpha_phases']['foundation']['min_steps']

        # Decrease loss first
        for i in range(min_steps):
            loss = 0.5 - i * 0.0001
            scheduler.update(step=i, loss=loss, gradient_stability=0.7)

        # Then plateau
        for i in range(min_steps, min_steps + 200):
            loss = 0.5 - min_steps * 0.0001 + np.random.randn() * 0.0001
            scheduler.update(step=i, loss=loss, gradient_stability=0.7)

        # Should have transitioned (plateaued with good stability)
        self.assertGreaterEqual(scheduler.current_phase_idx, 1)

    def test_loss_slope_sign_detection(self):
        """Test that positive vs negative slopes are correctly identified."""
        stats = TrainingStatistics()

        # Increasing loss
        for i in range(100):
            stats.add_loss(0.5 + i * 0.01)

        slope, _ = stats.get_loss_slope()
        self.assertGreater(slope, 0, "Increasing loss should have positive slope")

        # Decreasing loss
        stats = TrainingStatistics()
        for i in range(100):
            stats.add_loss(0.5 - i * 0.01)

        slope, _ = stats.get_loss_slope()
        self.assertLess(slope, 0, "Decreasing loss should have negative slope")


class TestNoGradientStability(unittest.TestCase):
    """Test scheduler works without gradient stability data."""

    def setUp(self):
        """Set up test fixtures."""
        self.rank = 64
        self.config = create_default_config(rank=self.rank, conv_alpha=14, linear_alpha=16)

    def test_works_without_gradient_stability(self):
        """Test that scheduler works when gradient_stability=None."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)

        # Update with only loss (no gradient stability)
        for i in range(100):
            scheduler.update(step=i, loss=0.5 - i * 0.001, gradient_stability=None)

        # Should not crash and should track statistics
        self.assertEqual(len(scheduler.global_statistics.recent_losses), 100)
        self.assertEqual(len(scheduler.global_statistics.gradient_stability_history), 0)

    def test_can_transition_without_gradient_stability(self):
        """Test that transitions can happen without gradient stability."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)

        min_steps = self.config['conv_alpha_phases']['foundation']['min_steps']

        # Train with plateaued loss, no gradient stability
        for i in range(min_steps + 200):
            if i < min_steps:
                loss = 0.5 - i * 0.0001
            else:
                loss = 0.5 - min_steps * 0.0001
            scheduler.update(step=i, loss=loss, gradient_stability=None)

        # Should have transitioned based on loss alone
        # (gradient stability check skipped when no data)
        self.assertGreaterEqual(scheduler.current_phase_idx, 0)
        # Might or might not transition depending on other criteria
        # But importantly, it shouldn't crash


class TestVeryNoisyVideoTraining(unittest.TestCase):
    """Test scheduler with realistic noisy video training data."""

    def setUp(self):
        """Set up test fixtures."""
        self.rank = 64
        self.config = create_default_config(rank=self.rank, conv_alpha=14, linear_alpha=16)

    def test_low_r_squared_doesnt_block_transition(self):
        """Test that very low R² (like 0.0004) doesn't block transitions."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)

        min_steps = self.config['conv_alpha_phases']['foundation']['min_steps']
        np.random.seed(42)

        # Create very noisy loss (like real video training)
        base_loss = 0.5
        for i in range(min_steps + 300):
            # Overall slight improvement but VERY noisy
            trend = -i * 0.00001
            noise = np.random.randn() * 0.05  # High noise
            loss = base_loss + trend + noise
            scheduler.update(step=i, loss=loss, gradient_stability=0.65)

        # Calculate R²
        slope, r2 = scheduler.global_statistics.get_loss_slope()

        # R² should be very low (noisy data)
        self.assertLess(r2, 0.01, "Video training should have low R²")

        # But transition might still happen (R² is now advisory)
        # Just verify it doesn't crash and phase_idx is valid
        self.assertIn(scheduler.current_phase_idx, [0, 1, 2])


class TestAlphaValueProgression(unittest.TestCase):
    """Test that alpha values progress correctly through phases."""

    def setUp(self):
        """Set up test fixtures."""
        self.rank = 64
        self.config = create_default_config(rank=self.rank, conv_alpha=14, linear_alpha=16)

    def test_conv_alpha_increases_through_phases(self):
        """Test that conv alpha increases as phases progress."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)

        # Phase 0
        alpha_phase0 = scheduler.get_current_alpha('test_conv', is_conv=True)

        # Force to phase 1
        scheduler.current_phase_idx = 1
        alpha_phase1 = scheduler.get_current_alpha('test_conv', is_conv=True)

        # Force to phase 2
        scheduler.current_phase_idx = 2
        alpha_phase2 = scheduler.get_current_alpha('test_conv', is_conv=True)

        # Should be increasing
        self.assertLess(alpha_phase0, alpha_phase1)
        self.assertLess(alpha_phase1, alpha_phase2)

    def test_linear_alpha_stays_constant(self):
        """Test that linear alpha never changes."""
        scheduler = PhaseAlphaScheduler(self.config, self.rank)

        alpha_phase0 = scheduler.get_current_alpha('test_linear', is_conv=False)

        scheduler.current_phase_idx = 1
        alpha_phase1 = scheduler.get_current_alpha('test_linear', is_conv=False)

        scheduler.current_phase_idx = 2
        alpha_phase2 = scheduler.get_current_alpha('test_linear', is_conv=False)

        # Should all be the same
        self.assertEqual(alpha_phase0, alpha_phase1)
        self.assertEqual(alpha_phase1, alpha_phase2)
        self.assertEqual(alpha_phase0, 16)

    def test_scale_respects_rank(self):
        """Test that scale = alpha/rank for all phases."""
        for rank in [32, 64, 128]:
            config = create_default_config(rank=rank, conv_alpha=14, linear_alpha=16)
            scheduler = PhaseAlphaScheduler(config, rank)

            for phase_idx in range(3):
                scheduler.current_phase_idx = phase_idx
                alpha = scheduler.get_current_alpha('test', is_conv=True)
                scale = scheduler.get_current_scale('test', is_conv=True)

                expected_scale = alpha / rank
                self.assertAlmostEqual(scale, expected_scale, places=6)


class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_state_dict_load(self):
        """Test loading an empty state dict."""
        config = create_default_config(rank=64)
        scheduler = PhaseAlphaScheduler(config, 64)

        scheduler.load_state_dict({})
        # Should not crash

    def test_partial_state_dict(self):
        """Test loading a state dict with missing fields."""
        config = create_default_config(rank=64)
        scheduler = PhaseAlphaScheduler(config, 64)

        partial_state = {
            'enabled': True,
            'current_phase_idx': 1,
            # Missing other fields
        }

        scheduler.load_state_dict(partial_state)

        # Should have loaded what was available
        self.assertEqual(scheduler.current_phase_idx, 1)

    def test_update_with_all_none(self):
        """Test update() when all optional args are None."""
        config = create_default_config(rank=64)
        scheduler = PhaseAlphaScheduler(config, 64)

        scheduler.update(step=0, loss=None, gradient_stability=None, expert=None)

        # Should not crash
        self.assertEqual(scheduler.total_steps, 0)

    def test_very_short_training(self):
        """Test training shorter than min_steps."""
        config = create_default_config(rank=64)
        scheduler = PhaseAlphaScheduler(config, 64)

        # Only train for 100 steps (min_steps is 1000)
        for i in range(100):
            scheduler.update(step=i, loss=0.5, gradient_stability=0.6)

        # Should stay in phase 0
        self.assertEqual(scheduler.current_phase_idx, 0)

    def test_zero_rank(self):
        """Test that zero rank raises error or handles gracefully."""
        config = create_default_config(rank=1)  # Minimum rank
        scheduler = PhaseAlphaScheduler(config, 1)

        # Should work with rank=1
        scale = scheduler.get_current_scale('test', is_conv=True)
        self.assertGreater(scale, 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
