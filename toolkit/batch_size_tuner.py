"""
Smart Batch Size Scaling for AI Toolkit.

This module provides automatic batch size tuning to maximize GPU utilization
while avoiding OOM (Out Of Memory) errors.

Features:
- Auto-detection: Find optimal batch size based on available GPU memory
- Batch size warmup: Gradually increase batch size during training
- OOM recovery: Automatically reduce batch size when OOM occurs
- Memory monitoring: Track GPU memory usage and headroom

Example:
    >>> from toolkit.batch_size_tuner import BatchSizeTuner
    >>> tuner = BatchSizeTuner(
    ...     initial_batch_size=4,
    ...     min_batch_size=1,
    ...     max_batch_size=32,
    ...     auto_scale=True
    ... )
    >>> # During training
    >>> current_bs = tuner.get_batch_size(step=100)
    >>> # After OOM
    >>> tuner.handle_oom()
    >>> new_bs = tuner.get_batch_size(step=100)
"""

import torch
from typing import Optional, Tuple
import math
from toolkit.print import print_acc


class BatchSizeTuner:
    """
    Automatically tune batch size for optimal GPU utilization.

    This class helps find and maintain the optimal batch size by:
    - Auto-detecting the largest batch size that fits in GPU memory
    - Gradually warming up batch size during training
    - Recovering from OOM errors by reducing batch size
    - Monitoring GPU memory usage

    Args:
        initial_batch_size: Starting batch size (default: 4)
        min_batch_size: Minimum allowed batch size (default: 1)
        max_batch_size: Maximum allowed batch size (default: 32)
        auto_scale: Enable automatic batch size scaling (default: False)
        warmup_steps: Number of steps for batch size warmup (default: 100)
        scale_factor: Factor to scale batch size by (default: 2)
        safety_margin: GPU memory safety margin as fraction (default: 0.9)
        oom_backoff_factor: Factor to reduce batch size on OOM (default: 0.75)
    """

    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        auto_scale: bool = False,
        warmup_steps: int = 100,
        scale_factor: float = 2.0,
        safety_margin: float = 0.9,
        oom_backoff_factor: float = 0.75,
    ):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = max(1, min_batch_size)
        self.max_batch_size = max_batch_size
        self.auto_scale = auto_scale
        self.warmup_steps = warmup_steps
        self.scale_factor = scale_factor
        self.safety_margin = safety_margin
        self.oom_backoff_factor = oom_backoff_factor

        # Current state
        self.current_batch_size = initial_batch_size
        self.stable_batch_size = initial_batch_size  # Last known stable batch size
        self.oom_count = 0
        self.successful_steps = 0
        self.warmup_complete = False

        # Memory tracking
        self.peak_memory_mb = 0
        self.total_memory_mb = 0
        if torch.cuda.is_available():
            self.total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    def get_batch_size(self, step: int) -> int:
        """
        Get the current batch size for the given training step.

        Args:
            step: Current training step

        Returns:
            Current batch size to use
        """
        if not self.auto_scale:
            return self.current_batch_size

        # Warmup phase: Gradually increase batch size
        if step < self.warmup_steps and not self.warmup_complete:
            # Linear warmup from initial to stable batch size
            progress = step / self.warmup_steps
            warmup_bs = int(self.initial_batch_size +
                          (self.stable_batch_size - self.initial_batch_size) * progress)
            self.current_batch_size = max(self.min_batch_size,
                                         min(warmup_bs, self.max_batch_size))

        return self.current_batch_size

    def handle_oom(self) -> Tuple[int, bool]:
        """
        Handle an OOM error by reducing batch size.

        Returns:
            Tuple of (new_batch_size, should_abort)
            - new_batch_size: Reduced batch size to try
            - should_abort: Whether training should be aborted (too many OOMs)
        """
        self.oom_count += 1

        # Calculate new batch size (reduce by backoff factor)
        new_batch_size = max(
            self.min_batch_size,
            int(self.current_batch_size * self.oom_backoff_factor)
        )

        # If we can't reduce further, abort
        if new_batch_size == self.current_batch_size:
            print_acc(f"Cannot reduce batch size below {self.min_batch_size}")
            return new_batch_size, True

        # If too many OOMs, abort
        if self.oom_count >= 5:
            print_acc(f"Too many OOM errors ({self.oom_count}), aborting")
            return new_batch_size, True

        old_batch_size = self.current_batch_size
        self.current_batch_size = new_batch_size
        self.stable_batch_size = new_batch_size

        print_acc(f"OOM detected! Reducing batch size: {old_batch_size} → {new_batch_size}")
        print_acc(f"  OOM count: {self.oom_count}/5")

        return new_batch_size, False

    def handle_success(self):
        """
        Record a successful training step.

        After enough successful steps, we may try to increase batch size.
        """
        self.successful_steps += 1
        self.oom_count = max(0, self.oom_count - 1)  # Decay OOM count on success

        # After warmup, try to increase batch size if we've been stable
        if self.warmup_complete and self.auto_scale:
            # Every 100 successful steps, try to increase batch size
            if self.successful_steps % 100 == 0 and self.successful_steps > 0:
                self._try_increase_batch_size()

    def _try_increase_batch_size(self):
        """
        Attempt to increase batch size if memory allows.
        """
        if self.current_batch_size >= self.max_batch_size:
            return  # Already at maximum

        # Check GPU memory headroom
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
            memory_available = self.total_memory_mb
            utilization = memory_used / memory_available

            # Only increase if we have headroom
            if utilization < self.safety_margin:
                # Try increasing by a factor (but not doubling immediately)
                increment = max(1, int(self.current_batch_size * 0.25))
                new_batch_size = min(
                    self.max_batch_size,
                    self.current_batch_size + increment
                )

                if new_batch_size > self.current_batch_size:
                    print_acc(f"Increasing batch size: {self.current_batch_size} → {new_batch_size}")
                    print_acc(f"  GPU memory: {utilization*100:.1f}% used")
                    self.current_batch_size = new_batch_size

    def complete_warmup(self):
        """
        Mark warmup as complete.
        """
        self.warmup_complete = True
        print_acc(f"Batch size warmup complete. Stable batch size: {self.stable_batch_size}")

    def get_memory_stats(self) -> dict:
        """
        Get current GPU memory statistics.

        Returns:
            Dictionary with memory stats (in MB)
        """
        if not torch.cuda.is_available():
            return {
                'allocated': 0,
                'reserved': 0,
                'peak': 0,
                'total': 0,
                'utilization': 0.0
            }

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        total = self.total_memory_mb
        utilization = allocated / total if total > 0 else 0.0

        self.peak_memory_mb = max(self.peak_memory_mb, peak)

        return {
            'allocated': allocated,
            'reserved': reserved,
            'peak': peak,
            'total': total,
            'utilization': utilization
        }

    def print_stats(self):
        """
        Print current batch size tuner statistics.
        """
        stats = self.get_memory_stats()
        print_acc("=== Batch Size Tuner Stats ===")
        print_acc(f"  Current batch size: {self.current_batch_size}")
        print_acc(f"  Stable batch size: {self.stable_batch_size}")
        print_acc(f"  Successful steps: {self.successful_steps}")
        print_acc(f"  OOM count: {self.oom_count}")
        print_acc(f"  Warmup complete: {self.warmup_complete}")
        if torch.cuda.is_available():
            print_acc(f"  GPU memory:")
            print_acc(f"    Allocated: {stats['allocated']:.1f} MB")
            print_acc(f"    Reserved: {stats['reserved']:.1f} MB")
            print_acc(f"    Peak: {stats['peak']:.1f} MB")
            print_acc(f"    Total: {stats['total']:.1f} MB")
            print_acc(f"    Utilization: {stats['utilization']*100:.1f}%")

    def reset(self):
        """
        Reset the tuner to initial state.
        """
        self.current_batch_size = self.initial_batch_size
        self.stable_batch_size = self.initial_batch_size
        self.oom_count = 0
        self.successful_steps = 0
        self.warmup_complete = False
        self.peak_memory_mb = 0


def auto_detect_batch_size(
    test_fn,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    initial_guess: int = 8,
) -> int:
    """
    Automatically detect the optimal batch size by binary search.

    This function tests progressively larger batch sizes until OOM,
    then uses binary search to find the largest size that fits.

    Args:
        test_fn: Function that takes batch_size and returns True if successful
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        initial_guess: Initial batch size guess

    Returns:
        Optimal batch size

    Example:
        >>> def test_batch(bs):
        ...     try:
        ...         # Try training step with batch size bs
        ...         return True
        ...     except torch.cuda.OutOfMemoryError:
        ...         return False
        >>> optimal_bs = auto_detect_batch_size(test_batch)
    """
    print_acc("Auto-detecting optimal batch size...")

    # Binary search for optimal batch size
    low = min_batch_size
    high = max_batch_size
    best_batch_size = min_batch_size

    while low <= high:
        mid = (low + high) // 2
        print_acc(f"  Testing batch size: {mid}")

        # Clear CUDA cache before test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            success = test_fn(mid)
            if success:
                best_batch_size = mid
                print_acc(f"    ✓ Batch size {mid} succeeded")
                low = mid + 1  # Try larger
            else:
                print_acc(f"    ✗ Batch size {mid} failed")
                high = mid - 1  # Try smaller
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print_acc(f"    ✗ Batch size {mid} OOM")
                high = mid - 1  # Try smaller
            else:
                raise

        # Clear cache after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_acc(f"Optimal batch size detected: {best_batch_size}")
    return best_batch_size
