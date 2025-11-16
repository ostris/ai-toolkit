"""
Training Metrics Collector for AI Toolkit.

Provides comprehensive metrics tracking for training processes including:
- Real-time memory usage (GPU and CPU)
- Dataloader throughput (batches/sec, samples/sec)
- Cache hit/miss rates
- Worker utilization statistics
- Training step timing breakdown

The metrics collector integrates with the existing Timer infrastructure
and provides a dashboard-style display of all metrics.
"""

import time
import torch
from collections import deque, OrderedDict
from typing import Dict, List, Optional, Tuple
import psutil
import os


class MetricsCollector:
    """
    Comprehensive metrics collector for training progress monitoring.

    Tracks various aspects of training performance:
    - Memory: GPU/CPU usage, peak allocations
    - Throughput: Batches/sec, samples/sec, images/sec
    - Dataloader: Fetch times, cache stats, worker stats
    - Training: Step times, loss values, learning rates

    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.start_step()
        >>> # ... training step ...
        >>> metrics.end_step(loss=0.123, lr=1e-4, batch_size=4)
        >>> metrics.print_dashboard()
    """

    def __init__(
        self,
        window_size: int = 100,
        enable_memory_tracking: bool = True,
        enable_throughput_tracking: bool = True,
        enable_dataloader_tracking: bool = True,
    ):
        """
        Initialize the metrics collector.

        Args:
            window_size: Number of recent samples to keep for averaging (default: 100)
            enable_memory_tracking: Track memory usage (default: True)
            enable_throughput_tracking: Track throughput metrics (default: True)
            enable_dataloader_tracking: Track dataloader metrics (default: True)
        """
        self.window_size = window_size
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_throughput_tracking = enable_throughput_tracking
        self.enable_dataloader_tracking = enable_dataloader_tracking

        # Step tracking
        self.step_count = 0
        self.step_start_time = None
        self.step_times = deque(maxlen=window_size)
        self.global_start_time = time.time()

        # Loss and LR tracking
        self.losses = OrderedDict()  # {loss_name: deque}
        self.learning_rates = deque(maxlen=window_size)

        # Memory tracking
        self.gpu_memory_allocated = deque(maxlen=window_size)
        self.gpu_memory_reserved = deque(maxlen=window_size)
        self.cpu_memory_percent = deque(maxlen=window_size)
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0

        # Throughput tracking
        self.batch_sizes = deque(maxlen=window_size)
        self.samples_per_second = deque(maxlen=window_size)
        self.total_samples_processed = 0

        # Dataloader tracking
        self.dataloader_fetch_times = deque(maxlen=window_size)
        self.cache_hits = 0
        self.cache_misses = 0
        self.worker_busy_time = deque(maxlen=window_size)

        # Timing breakdown (from Timer integration)
        self.timing_breakdown = OrderedDict()

    def start_step(self):
        """Mark the start of a training step."""
        self.step_start_time = time.time()

    def end_step(
        self,
        loss: Optional[float] = None,
        loss_dict: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Mark the end of a training step and record metrics.

        Args:
            loss: Primary loss value
            loss_dict: Dictionary of loss components
            lr: Current learning rate
            batch_size: Batch size for this step
        """
        if self.step_start_time is None:
            return

        # Record step time
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        self.step_count += 1

        # Record losses
        if loss is not None:
            if 'loss' not in self.losses:
                self.losses['loss'] = deque(maxlen=self.window_size)
            self.losses['loss'].append(loss)

        if loss_dict is not None:
            for key, value in loss_dict.items():
                if key not in self.losses:
                    self.losses[key] = deque(maxlen=self.window_size)
                self.losses[key].append(value)

        # Record learning rate
        if lr is not None:
            self.learning_rates.append(lr)

        # Record batch size and calculate throughput
        if batch_size is not None and self.enable_throughput_tracking:
            self.batch_sizes.append(batch_size)
            self.total_samples_processed += batch_size

            # Calculate samples/sec
            if step_time > 0:
                samples_per_sec = batch_size / step_time
                self.samples_per_second.append(samples_per_sec)

        # Record memory stats
        if self.enable_memory_tracking:
            self._record_memory_stats()

        # Reset step timer
        self.step_start_time = None

    def record_dataloader_fetch(self, fetch_time: float):
        """
        Record a dataloader fetch operation time.

        Args:
            fetch_time: Time taken to fetch a batch (seconds)
        """
        if self.enable_dataloader_tracking:
            self.dataloader_fetch_times.append(fetch_time)

    def record_cache_hit(self):
        """Record a cache hit."""
        if self.enable_dataloader_tracking:
            self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        if self.enable_dataloader_tracking:
            self.cache_misses += 1

    def record_worker_time(self, busy_time: float):
        """
        Record worker busy time.

        Args:
            busy_time: Time worker spent processing (seconds)
        """
        if self.enable_dataloader_tracking:
            self.worker_busy_time.append(busy_time)

    def update_timing_breakdown(self, timing_dict: Dict[str, float]):
        """
        Update timing breakdown from Timer integration.

        Args:
            timing_dict: Dictionary of {operation_name: avg_time}
        """
        self.timing_breakdown = timing_dict

    def _record_memory_stats(self):
        """Record current memory usage statistics."""
        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

            self.gpu_memory_allocated.append(allocated)
            self.gpu_memory_reserved.append(reserved)
            self.peak_gpu_memory = max(self.peak_gpu_memory, peak)

        # CPU memory
        try:
            process = psutil.Process()
            cpu_mem_percent = process.memory_percent()
            self.cpu_memory_percent.append(cpu_mem_percent)
            self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_mem_percent)
        except Exception:
            pass  # psutil might not be available

    def get_metrics_summary(self) -> Dict:
        """
        Get a summary of all current metrics.

        Returns:
            Dictionary containing all metrics averages and stats
        """
        summary = {}

        # Step metrics
        summary['step_count'] = self.step_count
        summary['total_time'] = time.time() - self.global_start_time
        if self.step_times:
            summary['avg_step_time'] = sum(self.step_times) / len(self.step_times)
            summary['steps_per_second'] = 1.0 / summary['avg_step_time'] if summary['avg_step_time'] > 0 else 0

        # Loss metrics
        for loss_name, loss_values in self.losses.items():
            if loss_values:
                summary[f'avg_{loss_name}'] = sum(loss_values) / len(loss_values)

        # Learning rate
        if self.learning_rates:
            summary['current_lr'] = self.learning_rates[-1]

        # Memory metrics
        if self.gpu_memory_allocated:
            summary['avg_gpu_allocated'] = sum(self.gpu_memory_allocated) / len(self.gpu_memory_allocated)
            summary['current_gpu_allocated'] = self.gpu_memory_allocated[-1]
            summary['peak_gpu_memory'] = self.peak_gpu_memory

        if self.cpu_memory_percent:
            summary['avg_cpu_memory_percent'] = sum(self.cpu_memory_percent) / len(self.cpu_memory_percent)

        # Throughput metrics
        if self.batch_sizes:
            summary['avg_batch_size'] = sum(self.batch_sizes) / len(self.batch_sizes)

        if self.samples_per_second:
            summary['avg_samples_per_second'] = sum(self.samples_per_second) / len(self.samples_per_second)

        summary['total_samples_processed'] = self.total_samples_processed

        # Dataloader metrics
        if self.dataloader_fetch_times:
            summary['avg_dataloader_fetch_time'] = sum(self.dataloader_fetch_times) / len(self.dataloader_fetch_times)

        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops > 0:
            summary['cache_hit_rate'] = self.cache_hits / total_cache_ops

        if self.worker_busy_time:
            summary['avg_worker_busy_time'] = sum(self.worker_busy_time) / len(self.worker_busy_time)

        return summary

    def print_dashboard(self, include_timing_breakdown: bool = False):
        """
        Print a dashboard-style summary of all metrics.

        Args:
            include_timing_breakdown: Include detailed timing breakdown (default: False)
        """
        summary = self.get_metrics_summary()

        print("\n" + "="*70)
        print("TRAINING METRICS DASHBOARD".center(70))
        print("="*70)

        # Training Progress
        print("\n📊 Training Progress:")
        print(f"  Steps completed: {summary.get('step_count', 0)}")
        print(f"  Total time: {summary.get('total_time', 0):.1f}s")
        if 'avg_step_time' in summary:
            print(f"  Avg step time: {summary['avg_step_time']:.3f}s")
        if 'steps_per_second' in summary:
            print(f"  Steps/sec: {summary['steps_per_second']:.2f}")

        # Loss Metrics
        if self.losses:
            print("\n📉 Loss Metrics:")
            for loss_name in self.losses.keys():
                avg_key = f'avg_{loss_name}'
                if avg_key in summary:
                    print(f"  {loss_name}: {summary[avg_key]:.6f}")

        # Learning Rate
        if 'current_lr' in summary:
            print(f"\n📈 Learning Rate: {summary['current_lr']:.2e}")

        # Memory Usage
        if self.enable_memory_tracking:
            print("\n💾 Memory Usage:")
            if 'current_gpu_allocated' in summary:
                print(f"  GPU allocated: {summary['current_gpu_allocated']:.2f} GB")
            if 'avg_gpu_allocated' in summary:
                print(f"  GPU avg: {summary['avg_gpu_allocated']:.2f} GB")
            if 'peak_gpu_memory' in summary:
                print(f"  GPU peak: {summary['peak_gpu_memory']:.2f} GB")
            if 'avg_cpu_memory_percent' in summary:
                print(f"  CPU usage: {summary['avg_cpu_memory_percent']:.1f}%")

        # Throughput
        if self.enable_throughput_tracking and 'avg_samples_per_second' in summary:
            print("\n⚡ Throughput:")
            print(f"  Samples/sec: {summary['avg_samples_per_second']:.1f}")
            print(f"  Total samples: {summary['total_samples_processed']}")
            if 'avg_batch_size' in summary:
                print(f"  Avg batch size: {summary['avg_batch_size']:.1f}")

        # Dataloader Stats
        if self.enable_dataloader_tracking:
            print("\n📦 Dataloader:")
            if 'avg_dataloader_fetch_time' in summary:
                print(f"  Avg fetch time: {summary['avg_dataloader_fetch_time']:.4f}s")
            if 'cache_hit_rate' in summary:
                print(f"  Cache hit rate: {summary['cache_hit_rate']*100:.1f}%")
                print(f"    Hits: {self.cache_hits}, Misses: {self.cache_misses}")

        # Timing Breakdown
        if include_timing_breakdown and self.timing_breakdown:
            print("\n⏱️  Timing Breakdown:")
            for operation, avg_time in sorted(self.timing_breakdown.items(),
                                             key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {operation}: {avg_time:.4f}s")

        print("\n" + "="*70 + "\n")

    def reset(self):
        """Reset all metrics (useful for per-epoch resets)."""
        self.step_count = 0
        self.step_times.clear()
        self.losses.clear()
        self.learning_rates.clear()
        self.gpu_memory_allocated.clear()
        self.gpu_memory_reserved.clear()
        self.cpu_memory_percent.clear()
        self.batch_sizes.clear()
        self.samples_per_second.clear()
        self.dataloader_fetch_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.worker_busy_time.clear()
        self.global_start_time = time.time()

    def export_to_dict(self) -> Dict:
        """
        Export all metrics to a dictionary for logging/saving.

        Returns:
            Dictionary with all current metrics
        """
        return {
            'summary': self.get_metrics_summary(),
            'step_count': self.step_count,
            'total_samples_processed': self.total_samples_processed,
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            'timing_breakdown': dict(self.timing_breakdown),
        }
