"""
Alpha Scheduler Metrics Logger
Collects and exports training metrics for UI visualization.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class AlphaMetricsLogger:
    """Collects and exports alpha scheduler metrics for UI."""

    def __init__(self, output_dir: str, job_name: str):
        """
        Initialize metrics logger.

        Args:
            output_dir: Base output directory for the job
            job_name: Name of the training job
        """
        self.output_dir = output_dir
        self.job_name = job_name
        self.metrics_file = os.path.join(output_dir, f"metrics_{job_name}.jsonl")

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Track if we've written the header
        self._initialized = os.path.exists(self.metrics_file)

    def log_step(self,
                 step: int,
                 loss: Optional[float] = None,
                 gradient_stability: Optional[float] = None,
                 expert: Optional[str] = None,
                 scheduler = None,
                 learning_rate: Optional[float] = None,
                 learning_rates: Optional[list] = None):
        """
        Log metrics for current training step.

        Args:
            step: Current training step number
            loss: Loss value for this step
            gradient_stability: Gradient sign agreement rate (0-1)
            expert: Expert name if using MoE ('high_noise', 'low_noise', etc.)
            scheduler: PhaseAlphaScheduler instance (optional)
            learning_rate: Single learning rate (for non-MoE)
            learning_rates: List of learning rates per expert (for MoE)
        """
        metrics = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'loss': loss,
            'gradient_stability': gradient_stability,
            'expert': expert
        }

        # Add learning rate data
        if learning_rates is not None and len(learning_rates) > 0:
            # MoE: multiple learning rates
            for i, lr in enumerate(learning_rates):
                lr_val = lr.item() if hasattr(lr, 'item') else lr
                metrics[f'lr_{i}'] = lr_val
        elif learning_rate is not None:
            # Single learning rate
            metrics['learning_rate'] = learning_rate

        # Add alpha scheduler state if available
        if scheduler and hasattr(scheduler, 'enabled') and scheduler.enabled:
            try:
                phase_names = ['foundation', 'balance', 'emphasis']
                current_phase = phase_names[scheduler.current_phase_idx] if scheduler.current_phase_idx < len(phase_names) else 'unknown'

                metrics.update({
                    'alpha_enabled': True,
                    'phase': current_phase,
                    'phase_idx': scheduler.current_phase_idx,
                    'steps_in_phase': scheduler.steps_in_phase,
                    'conv_alpha': scheduler.get_current_alpha('conv', is_conv=True),
                    'linear_alpha': scheduler.get_current_alpha('linear', is_conv=False),
                })

                # Add loss statistics if available
                if hasattr(scheduler, 'global_statistics'):
                    stats = scheduler.global_statistics
                    if hasattr(stats, 'get_loss_slope'):
                        slope, r2 = stats.get_loss_slope()
                        # Only add if we have enough samples (not None)
                        if slope is not None:
                            metrics['loss_slope'] = slope
                            metrics['loss_r2'] = r2
                            metrics['loss_samples'] = len(stats.recent_losses)
                        else:
                            metrics['loss_samples'] = len(stats.recent_losses)

                    if hasattr(stats, 'get_gradient_stability'):
                        metrics['gradient_stability_avg'] = stats.get_gradient_stability()

                    # Add EMA metrics for charting
                    if hasattr(stats, 'loss_ema_10'):
                        metrics['loss_ema_10'] = stats.loss_ema_10
                    if hasattr(stats, 'loss_ema_50'):
                        metrics['loss_ema_50'] = stats.loss_ema_50
                    if hasattr(stats, 'loss_ema_100'):
                        metrics['loss_ema_100'] = stats.loss_ema_100
                    if hasattr(stats, 'grad_ema_10'):
                        metrics['grad_ema_10'] = stats.grad_ema_10
                    if hasattr(stats, 'grad_ema_50'):
                        metrics['grad_ema_50'] = stats.grad_ema_50
                    if hasattr(stats, 'grad_ema_100'):
                        metrics['grad_ema_100'] = stats.grad_ema_100

            except Exception as e:
                # Don't fail training if metrics collection fails
                print(f"Warning: Failed to collect alpha scheduler metrics: {e}")
                metrics['alpha_enabled'] = False
        else:
            metrics['alpha_enabled'] = False

        # Write to JSONL file (one line per step)
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write metrics: {e}")

    def get_metrics_file_path(self) -> str:
        """Get the path to the metrics file."""
        return self.metrics_file

    def get_latest_metrics(self, n: int = 100) -> list:
        """
        Read the last N metrics entries.

        Args:
            n: Number of recent entries to read

        Returns:
            List of metric dictionaries
        """
        if not os.path.exists(self.metrics_file):
            return []

        try:
            with open(self.metrics_file, 'r') as f:
                lines = f.readlines()

            # Get last N lines
            recent_lines = lines[-n:] if len(lines) > n else lines

            # Parse JSON
            metrics = []
            for line in recent_lines:
                line = line.strip()
                if line:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            return metrics
        except Exception as e:
            print(f"Warning: Failed to read metrics: {e}")
            return []

    def cleanup_metrics_after_step(self, resume_step: int):
        """
        Remove metrics entries beyond the resume step.
        This is needed when training is resumed from a checkpoint - metrics logged
        after the checkpoint step should be removed.

        Args:
            resume_step: Step number we're resuming from
        """
        if not os.path.exists(self.metrics_file):
            return

        try:
            with open(self.metrics_file, 'r') as f:
                lines = f.readlines()

            # Filter to keep only metrics at or before resume_step
            valid_lines = []
            removed_count = 0
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        metric = json.loads(line)
                        if metric.get('step', 0) <= resume_step:
                            valid_lines.append(line + '\n')
                        else:
                            removed_count += 1
                    except json.JSONDecodeError:
                        continue

            # Rewrite file with valid lines only
            if removed_count > 0:
                with open(self.metrics_file, 'w') as f:
                    f.writelines(valid_lines)
                print(f"Cleaned up {removed_count} metrics entries beyond step {resume_step}")

        except Exception as e:
            print(f"Warning: Failed to cleanup metrics: {e}")
