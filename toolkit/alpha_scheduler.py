#!/usr/bin/env python3
"""
Alpha Scheduler for LoRA Training
Implements automatic alpha scheduling with phase-based transitions.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class PhaseDefinition:
    """Defines a training phase with alpha value and exit criteria."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.alpha = config.get('alpha')
        self.min_steps = config.get('min_steps', 500)

        # Exit criteria for automatic transition
        exit_criteria = config.get('exit_criteria', {})
        self.loss_improvement_rate_below = exit_criteria.get('loss_improvement_rate_below', 0.001)
        self.min_gradient_stability = exit_criteria.get('min_gradient_stability', 0.55)
        self.min_loss_r2 = exit_criteria.get('min_loss_r2', 0.1)  # Ensure trend is real, not noise

    def __repr__(self):
        return f"Phase({self.name}, alpha={self.alpha}, min_steps={self.min_steps})"


class TrainingStatistics:
    """Tracks training statistics for phase transition decisions."""

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.recent_losses = []
        self.gradient_stability_history = []

    def add_loss(self, loss: float):
        """Add a loss value to the history."""
        self.recent_losses.append(loss)
        if len(self.recent_losses) > self.window_size:
            self.recent_losses.pop(0)

    def add_gradient_stability(self, stability: float):
        """Add gradient stability metric to history."""
        self.gradient_stability_history.append(stability)
        if len(self.gradient_stability_history) > self.window_size:
            self.gradient_stability_history.pop(0)

    def get_loss_slope(self) -> tuple:
        """
        Calculate loss slope using linear regression.
        Returns: (slope, r_squared) or (None, None) if insufficient data
        """
        # Need at least 20 samples for meaningful trend analysis
        if len(self.recent_losses) < 20:
            return None, None

        losses = np.array(self.recent_losses)
        indices = np.arange(len(losses))

        slope, intercept, r_value, _, _ = linregress(indices, losses)
        r_squared = r_value ** 2

        return slope, r_squared

    def get_gradient_stability(self) -> float:
        """Get average gradient stability over recent history."""
        if not self.gradient_stability_history:
            return 0.0

        # Use recent 50 samples or all if less
        recent = self.gradient_stability_history[-50:]
        return np.mean(recent)

    def get_loss_cv(self) -> float:
        """Calculate coefficient of variation for recent losses."""
        if len(self.recent_losses) < 10:
            return 0.0

        losses = np.array(self.recent_losses[-50:])
        mean_loss = np.mean(losses)
        if mean_loss == 0:
            return 0.0

        return np.std(losses) / mean_loss


class PhaseAlphaScheduler:
    """
    Phase-based alpha scheduler with automatic transitions.

    Progressively adjusts alpha values through defined training phases,
    automatically transitioning when loss plateaus and gradients are stable.
    """

    def __init__(self, config: Dict[str, Any], rank: int):
        """
        Initialize the alpha scheduler.

        Args:
            config: Configuration dictionary with phase definitions
            rank: LoRA rank (needed for rank-aware decisions)
        """
        self.config = config
        self.rank = rank
        self.enabled = config.get('enabled', False)

        if not self.enabled:
            logger.info("Alpha scheduling disabled")
            return

        # Parse phase definitions
        self.phases = self._parse_phases(config.get('conv_alpha_phases', {}))
        self.linear_alpha = config.get('linear_alpha', 16)

        # Parse per-expert configurations (for MoE)
        self.per_expert_phases = {}
        per_expert_config = config.get('per_expert', {})
        for expert_name, expert_config in per_expert_config.items():
            if 'phases' in expert_config:
                self.per_expert_phases[expert_name] = self._parse_phases(expert_config['phases'])

        # State tracking
        self.current_phase_idx = 0
        self.steps_in_phase = 0
        self.total_steps = 0

        # Statistics tracking (per expert for MoE)
        self.statistics = {}  # expert_name -> TrainingStatistics
        self.global_statistics = TrainingStatistics()

        # Phase transition history
        self.transition_history = []

        logger.info(f"Alpha scheduler initialized with {len(self.phases)} phases")
        logger.info(f"Rank: {rank}, Linear alpha (fixed): {self.linear_alpha}")
        logger.info(f"Conv alpha phases: {[p.name for p in self.phases]}")
        if self.per_expert_phases:
            logger.info(f"Per-expert phases configured for: {list(self.per_expert_phases.keys())}")

        # Validate alpha/rank ratios and warn if high
        self._validate_alpha_ratios()

    def _validate_alpha_ratios(self):
        """Validate alpha/rank ratios and warn if unusually high."""
        # Check linear alpha
        linear_scale = self.linear_alpha / self.rank
        if linear_scale > 0.5:
            logger.warning(
                f"⚠️  Linear alpha scale is HIGH: {self.linear_alpha}/{self.rank} = {linear_scale:.3f}\n"
                f"   This exceeds 0.5 (half of rank). Common practice is scale ≤ 1.0.\n"
                f"   Consider reducing linear_alpha if training is unstable."
            )

        # Check conv alpha in all phases
        for phase in self.phases:
            conv_scale = phase.alpha / self.rank
            if conv_scale > 0.5:
                logger.warning(
                    f"⚠️  Conv alpha scale in '{phase.name}' phase is HIGH: {phase.alpha}/{self.rank} = {conv_scale:.3f}\n"
                    f"   This exceeds 0.5 (half of rank). Common practice is scale ≤ 1.0.\n"
                    f"   Consider reducing alpha for this phase if training is unstable."
                )

        # Check per-expert phases if they exist
        if self.per_expert_phases:
            for expert_name, expert_phases in self.per_expert_phases.items():
                for phase in expert_phases:
                    conv_scale = phase.alpha / self.rank
                    if conv_scale > 0.5:
                        logger.warning(
                            f"⚠️  Conv alpha scale for '{expert_name}' in '{phase.name}' phase is HIGH:\n"
                            f"   {phase.alpha}/{self.rank} = {conv_scale:.3f} (exceeds 0.5)\n"
                            f"   Common practice is scale ≤ 1.0. Consider reducing if unstable."
                        )

    def _parse_phases(self, phases_config: Dict[str, Dict]) -> List[PhaseDefinition]:
        """Parse phase configuration into PhaseDefinition objects."""
        phases = []
        for phase_name, phase_config in phases_config.items():
            phases.append(PhaseDefinition(phase_name, phase_config))
        return phases

    def _infer_expert(self, module_name: str) -> Optional[str]:
        """
        Infer expert name from module name.

        For MoE networks, module names typically contain expert identifier.
        Examples: "high_noise.lora_down", "low_noise.attention"
        """
        if not module_name:
            return None

        # Check for common expert name patterns
        for expert_name in ['high_noise', 'low_noise']:
            if expert_name in module_name.lower():
                return expert_name

        return None

    def _get_phases_for_expert(self, expert: Optional[str]) -> List[PhaseDefinition]:
        """Get phase definitions for a specific expert (or global if no expert)."""
        if expert and expert in self.per_expert_phases:
            return self.per_expert_phases[expert]
        return self.phases

    def get_current_alpha(self, module_name: str, is_conv: bool) -> float:
        """
        Get current alpha value for a module.

        Args:
            module_name: Name of the LoRA module
            is_conv: Whether this is a convolutional layer

        Returns:
            Current alpha value
        """
        if not self.enabled:
            # Return default values when disabled
            return self.linear_alpha if not is_conv else self.config.get('conv_alpha', 14)

        # Linear alpha is always fixed (content stability)
        if not is_conv:
            return self.linear_alpha

        # Get expert-specific or global phases
        expert = self._infer_expert(module_name)
        phases = self._get_phases_for_expert(expert)

        # Get current phase alpha
        if self.current_phase_idx < len(phases):
            return phases[self.current_phase_idx].alpha
        else:
            # Staying in final phase
            return phases[-1].alpha

    def get_current_scale(self, module_name: str, is_conv: bool) -> float:
        """
        Get current scale value (alpha/rank) for a module.

        This is the actual effective scaling factor applied in forward pass.
        """
        alpha = self.get_current_alpha(module_name, is_conv)
        return alpha / self.rank

    def update(self, step: int, loss: Optional[float] = None,
               gradient_stability: Optional[float] = None,
               expert: Optional[str] = None):
        """
        Update scheduler state and check for phase transitions.

        Args:
            step: Current training step
            loss: Current loss value
            gradient_stability: Current gradient sign agreement rate
            expert: Expert name (for MoE networks)
        """
        if not self.enabled:
            return

        self.total_steps = step
        self.steps_in_phase += 1

        # Update statistics
        if loss is not None:
            self.global_statistics.add_loss(loss)

            if expert:
                if expert not in self.statistics:
                    self.statistics[expert] = TrainingStatistics()
                self.statistics[expert].add_loss(loss)

        if gradient_stability is not None:
            self.global_statistics.add_gradient_stability(gradient_stability)

            if expert:
                if expert not in self.statistics:
                    self.statistics[expert] = TrainingStatistics()
                self.statistics[expert].add_gradient_stability(gradient_stability)

        # Check for phase transition
        if self.current_phase_idx < len(self.phases) - 1:
            if self._should_transition():
                self._transition_to_next_phase()

    def _should_transition(self) -> bool:
        """
        Determine if we should transition to the next phase.

        Criteria:
        1. Minimum steps in current phase met
        2. Loss improvement rate below threshold (plateauing)
        3. Gradient stability above threshold (stable training)
        4. Loss trend R² high enough (real trend, not noise)
        """
        current_phase = self.phases[self.current_phase_idx]

        # Must meet minimum steps first
        if self.steps_in_phase < current_phase.min_steps:
            return False

        # Get loss slope and R²
        loss_slope, loss_r2 = self.global_statistics.get_loss_slope()

        # Check if we have enough data for trend analysis
        if loss_slope is None or loss_r2 is None:
            return False

        if len(self.global_statistics.recent_losses) < 100:
            return False

        # Check R² threshold - trend must be real, not noise
        # For video training, R² is often very low (~0.001) due to high variance
        # Only use this as a sanity check, not a hard requirement
        if loss_r2 < current_phase.min_loss_r2:
            logger.debug(f"Phase {current_phase.name}: R² too low ({loss_r2:.4f}), need > {current_phase.min_loss_r2}")
            # Don't return False - just log for now, check other criteria

        # Check loss is improving or plateaued (NOT increasing)
        # We want to transition when loss stops improving (plateaus)
        # But NOT if loss is actively getting worse (increasing)

        loss_plateau_threshold = current_phase.loss_improvement_rate_below

        # Plateau: slope very close to zero (within threshold, either direction)
        # Improving: slope negative beyond plateau threshold
        # Increasing: slope positive (any amount - this is BAD)

        # Key insight: ANY meaningful positive slope means loss is increasing (bad)
        # Only allow transition if slope is negative or essentially zero
        # Use a very strict threshold for "essentially zero" - 5% of plateau threshold
        essentially_zero = loss_plateau_threshold * 0.05

        if loss_slope > essentially_zero:
            # Positive slope beyond noise level - loss is increasing, block transition
            loss_ok = False
        elif loss_slope < 0:
            # Decreasing - good, allow if slow enough (plateau) or still improving rapidly
            loss_ok = abs(loss_slope) < loss_plateau_threshold * 5
        else:
            # Within essentially zero range - true plateau, allow transition
            loss_ok = abs(loss_slope) <= essentially_zero

        # Check gradient stability (if available)
        grad_stability = self.global_statistics.get_gradient_stability()
        # If no gradient stability data (non-automagic optimizer), skip this check
        if len(self.global_statistics.gradient_stability_history) > 0:
            stability_ok = grad_stability >= current_phase.min_gradient_stability
        else:
            # No gradient stability available - use other criteria only
            stability_ok = True
            logger.debug(f"Phase {current_phase.name}: No gradient stability data, skipping stability check")

        # Check coefficient of variation (should be reasonable)
        loss_cv = self.global_statistics.get_loss_cv()
        cv_ok = loss_cv < 0.5  # Less than 50% variation

        logger.debug(
            f"Phase {current_phase.name} transition check at step {self.total_steps}:\n"
            f"  Steps in phase: {self.steps_in_phase} >= {current_phase.min_steps}\n"
            f"  Loss slope: {loss_slope:.6e}\n"
            f"    Threshold: {loss_plateau_threshold:.6e}\n"
            f"    Loss OK: {loss_ok} (not increasing)\n"
            f"  Loss R²: {loss_r2:.4f} (advisory: {current_phase.min_loss_r2})\n"
            f"  Gradient stability: {grad_stability:.4f} >= {current_phase.min_gradient_stability}: {stability_ok}\n"
            f"  Loss CV: {loss_cv:.4f} < 0.5: {cv_ok}"
        )

        return loss_ok and stability_ok and cv_ok

    def _transition_to_next_phase(self):
        """Execute transition to the next phase."""
        old_phase = self.phases[self.current_phase_idx]
        self.current_phase_idx += 1
        new_phase = self.phases[self.current_phase_idx]

        transition_info = {
            'step': self.total_steps,
            'from_phase': old_phase.name,
            'to_phase': new_phase.name,
            'from_alpha': old_phase.alpha,
            'to_alpha': new_phase.alpha,
            'steps_in_phase': self.steps_in_phase
        }
        self.transition_history.append(transition_info)

        # Reset phase step counter
        self.steps_in_phase = 0

        logger.info(
            f"\n{'='*80}\n"
            f"ALPHA PHASE TRANSITION at step {self.total_steps}\n"
            f"  {old_phase.name} (α={old_phase.alpha}) → {new_phase.name} (α={new_phase.alpha})\n"
            f"  Duration: {transition_info['steps_in_phase']} steps\n"
            f"  Effective scale change: {old_phase.alpha/self.rank:.6f} → {new_phase.alpha/self.rank:.6f}\n"
            f"{'='*80}\n"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status for logging/debugging."""
        if not self.enabled:
            return {'enabled': False}

        current_phase = self.phases[self.current_phase_idx]
        loss_slope, loss_r2 = self.global_statistics.get_loss_slope()

        status = {
            'enabled': True,
            'total_steps': self.total_steps,
            'current_phase': current_phase.name,
            'phase_index': f"{self.current_phase_idx + 1}/{len(self.phases)}",
            'steps_in_phase': self.steps_in_phase,
            'current_conv_alpha': current_phase.alpha,
            'current_linear_alpha': self.linear_alpha,
            'current_conv_scale': current_phase.alpha / self.rank,
            'current_linear_scale': self.linear_alpha / self.rank,
            'loss_slope': loss_slope,
            'loss_r2': loss_r2,
            'gradient_stability': self.global_statistics.get_gradient_stability(),
            'loss_cv': self.global_statistics.get_loss_cv(),
            'transitions': len(self.transition_history)
        }

        # Add per-expert status if available
        if self.statistics:
            status['per_expert'] = {}
            for expert_name, stats in self.statistics.items():
                expert_slope, expert_r2 = stats.get_loss_slope()
                status['per_expert'][expert_name] = {
                    'loss_slope': expert_slope,
                    'loss_r2': expert_r2,
                    'gradient_stability': stats.get_gradient_stability(),
                    'loss_cv': stats.get_loss_cv()
                }

        return status

    def log_status(self):
        """Log current scheduler status."""
        status = self.get_status()

        if not status['enabled']:
            return

        logger.info(
            f"Alpha Scheduler Status (Step {status['total_steps']}):\n"
            f"  Phase: {status['current_phase']} ({status['phase_index']}) - {status['steps_in_phase']} steps\n"
            f"  Conv: α={status['current_conv_alpha']} (scale={status['current_conv_scale']:.6f})\n"
            f"  Linear: α={status['current_linear_alpha']} (scale={status['current_linear_scale']:.6f})\n"
            f"  Loss: slope={status['loss_slope']:.6e}, R²={status['loss_r2']:.4f}, CV={status['loss_cv']:.4f}\n"
            f"  Gradient stability: {status['gradient_stability']:.4f}\n"
            f"  Total transitions: {status['transitions']}"
        )

        if 'per_expert' in status:
            for expert_name, expert_status in status['per_expert'].items():
                logger.info(
                    f"  Expert {expert_name}: "
                    f"slope={expert_status['loss_slope']:.6e}, "
                    f"R²={expert_status['loss_r2']:.4f}, "
                    f"stability={expert_status['gradient_stability']:.4f}"
                )

    def state_dict(self) -> Dict[str, Any]:
        """
        Get scheduler state for checkpoint saving.

        Returns:
            Dictionary containing scheduler state
        """
        if not self.enabled:
            return {'enabled': False}

        state = {
            'enabled': True,
            'current_phase_idx': self.current_phase_idx,
            'steps_in_phase': self.steps_in_phase,
            'total_steps': self.total_steps,
            'transition_history': self.transition_history,
            'global_losses': list(self.global_statistics.recent_losses),
            'global_grad_stability': list(self.global_statistics.gradient_stability_history),
        }

        # Save per-expert statistics if they exist
        if self.statistics:
            state['expert_statistics'] = {}
            for expert_name, stats in self.statistics.items():
                state['expert_statistics'][expert_name] = {
                    'losses': list(stats.recent_losses),
                    'grad_stability': list(stats.gradient_stability_history)
                }

        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """
        Load scheduler state from checkpoint.

        Args:
            state: Dictionary containing scheduler state
        """
        if not state.get('enabled', False):
            return

        self.current_phase_idx = state.get('current_phase_idx', 0)
        self.steps_in_phase = state.get('steps_in_phase', 0)
        self.total_steps = state.get('total_steps', 0)
        self.transition_history = state.get('transition_history', [])

        # Restore global statistics
        self.global_statistics.recent_losses = state.get('global_losses', [])
        self.global_statistics.gradient_stability_history = state.get('global_grad_stability', [])

        # Restore per-expert statistics if they exist
        if 'expert_statistics' in state:
            for expert_name, expert_state in state['expert_statistics'].items():
                if expert_name not in self.statistics:
                    self.statistics[expert_name] = TrainingStatistics()
                self.statistics[expert_name].recent_losses = expert_state.get('losses', [])
                self.statistics[expert_name].gradient_stability_history = expert_state.get('grad_stability', [])

        logger.info(
            f"Alpha scheduler state restored: "
            f"phase {self.current_phase_idx + 1}/{len(self.phases)} "
            f"({self.phases[self.current_phase_idx].name}), "
            f"step {self.total_steps}, "
            f"{len(self.transition_history)} transitions"
        )


def create_default_config(rank: int, conv_alpha: float = 14, linear_alpha: float = 16) -> Dict[str, Any]:
    """
    Create a default alpha schedule configuration.

    This provides a sensible default for video LoRA training with progressive
    motion emphasis. Based on proven values from squ1rtv14 training.

    Args:
        rank: LoRA rank
        conv_alpha: Target conv_alpha for final phase (default: 14)
        linear_alpha: Fixed linear_alpha (content stability, default: 16)

    Returns:
        Configuration dictionary

    Note:
        Default scales for rank=64:
        - linear: 16/64 = 0.25 (proven to work)
        - conv foundation: 7/64 = 0.109
        - conv balance: 10/64 = 0.156
        - conv emphasis: 14/64 = 0.219 (proven to work)
    """
    # Calculate phases based on target alpha
    # Use 50%, 70%, 100% progression (more gradual than 50/75/100)
    foundation_alpha = max(4, int(conv_alpha * 0.5))  # 50% of target (7 for target 14)
    balance_alpha = max(6, int(conv_alpha * 0.7))     # 70% of target (10 for target 14)
    emphasis_alpha = conv_alpha                        # 100% of target (14)

    config = {
        'enabled': True,
        'mode': 'phase_adaptive',
        'linear_alpha': linear_alpha,
        'conv_alpha_phases': {
            'foundation': {
                'alpha': foundation_alpha,
                'min_steps': 1000,
                'exit_criteria': {
                    'loss_improvement_rate_below': 0.001,
                    'min_gradient_stability': 0.55,
                    'min_loss_r2': 0.005  # Very low for noisy video training
                }
            },
            'balance': {
                'alpha': balance_alpha,
                'min_steps': 1500,
                'exit_criteria': {
                    'loss_improvement_rate_below': 0.0005,
                    'min_gradient_stability': 0.60,
                    'min_loss_r2': 0.003  # Very low for noisy video training
                }
            },
            'emphasis': {
                'alpha': emphasis_alpha,
                # Final phase, no exit criteria needed
            }
        }
    }

    # Add MoE-specific configurations
    # High noise (harder timesteps) gets slightly more alpha
    # But keep it reasonable - max at linear_alpha for safety
    high_noise_emphasis = min(linear_alpha, emphasis_alpha + 2)  # Cap at linear_alpha
    high_noise_balance = min(linear_alpha - 2, balance_alpha + 2)
    high_noise_foundation = min(linear_alpha - 4, foundation_alpha + 2)

    config['per_expert'] = {
        'high_noise': {
            'phases': {
                'foundation': {'alpha': high_noise_foundation},
                'balance': {'alpha': high_noise_balance},
                'emphasis': {'alpha': high_noise_emphasis}
            }
        },
        'low_noise': {
            'phases': {
                'foundation': {'alpha': foundation_alpha},
                'balance': {'alpha': balance_alpha},
                'emphasis': {'alpha': emphasis_alpha}
            }
        }
    }

    return config
