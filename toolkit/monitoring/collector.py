"""
Main monitoring orchestrator that coordinates metric collection during training.
Runs a background thread for periodic sampling and hooks into training events.
"""

import os
import time
import threading
from typing import Optional, Dict, Any
from .database import MonitoringDatabase, MetricsSample, TrainingEvent, StepMetric
from .samplers import collect_all_metrics, is_unified_memory_system


class MonitoringCollector:
    """
    Orchestrates metric collection during training.

    Features:
    - Background thread for periodic system sampling
    - Event hooks for training milestones
    - Step-based metric collection
    - Automatic buffer flushing
    """

    def __init__(
        self,
        db_path: str,
        job_id: str,
        sample_interval_seconds: float = 5.0,
        track_per_process: bool = True,
        enabled: bool = True,
    ):
        self.db_path = db_path
        self.job_id = job_id
        self.sample_interval = sample_interval_seconds
        self.track_per_process = track_per_process
        self.enabled = enabled

        self.db = MonitoringDatabase(db_path) if enabled else None
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._pid = os.getpid()
        self._is_unified_memory = is_unified_memory_system()
        self._gpu_type = 'unified' if self._is_unified_memory else 'auto'
        self._start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None

    def start(self):
        """Start the background sampling thread."""
        if not self.enabled:
            return

        self._start_time = time.time()
        self._stop_event.clear()

        # Record training start event
        self.record_event('training_start', {
            'pid': self._pid,
            'is_unified_memory': self._is_unified_memory,
        })

        # Start background sampler
        self._sampler_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name='MonitoringSampler'
        )
        self._sampler_thread.start()

    def stop(self):
        """Stop the background sampling thread and flush remaining data."""
        if not self.enabled:
            return

        self._stop_event.set()

        if self._sampler_thread:
            self._sampler_thread.join(timeout=2.0)

        # Flush any remaining buffered samples
        if self.db:
            self.db.flush_buffer()

        # Record training end event
        duration = time.time() - self._start_time if self._start_time else 0
        self.record_event('training_end', {
            'duration_seconds': duration,
        })

    def _sampling_loop(self):
        """Background thread that periodically samples system metrics."""
        while not self._stop_event.is_set():
            try:
                self._collect_sample()
            except Exception as e:
                # Log error but don't crash the training
                print(f"[Monitoring] Error collecting sample: {e}")

            # Sleep in small increments to allow quick shutdown
            for _ in range(int(self.sample_interval * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

    def _collect_sample(self):
        """Collect and store a single metrics sample."""
        if not self.db:
            return

        metrics = collect_all_metrics(
            pid=self._pid if self.track_per_process else None,
            gpu_type=self._gpu_type
        )

        sample = MetricsSample(
            job_id=self.job_id,
            timestamp=time.time(),
            total_memory_gb=metrics.get('total_memory_gb', 0),
            used_memory_gb=metrics.get('used_memory_gb', 0),
            available_memory_gb=metrics.get('available_memory_gb', 0),
            swap_used_gb=metrics.get('swap_used_gb', 0),
            main_process_memory_gb=metrics.get('main_process_memory_gb', 0),
            worker_memory_gb=metrics.get('worker_memory_gb', 0),
            worker_count=metrics.get('worker_count', 0),
            gpu_memory_used_gb=metrics.get('gpu_memory_used_gb', 0),
            gpu_utilization_percent=metrics.get('gpu_utilization_percent', 0),
            cpu_utilization_percent=metrics.get('cpu_utilization_percent', 0),
        )

        self.db.buffer_sample(sample)

    def record_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        """
        Record a training event with optional memory snapshot.

        Args:
            event_type: Type of event (e.g., 'training_start', 'oom_error', 'batch_adjustment')
            details: Optional dictionary with event-specific information
        """
        if not self.enabled or not self.db:
            return

        import json

        # Take a memory snapshot at event time
        snapshot_id = None
        try:
            metrics = collect_all_metrics(
                pid=self._pid if self.track_per_process else None,
                gpu_type=self._gpu_type
            )
            sample = MetricsSample(
                job_id=self.job_id,
                timestamp=time.time(),
                total_memory_gb=metrics.get('total_memory_gb', 0),
                used_memory_gb=metrics.get('used_memory_gb', 0),
                available_memory_gb=metrics.get('available_memory_gb', 0),
                swap_used_gb=metrics.get('swap_used_gb', 0),
                main_process_memory_gb=metrics.get('main_process_memory_gb', 0),
                worker_memory_gb=metrics.get('worker_memory_gb', 0),
                worker_count=metrics.get('worker_count', 0),
                gpu_memory_used_gb=metrics.get('gpu_memory_used_gb', 0),
                gpu_utilization_percent=metrics.get('gpu_utilization_percent', 0),
                cpu_utilization_percent=metrics.get('cpu_utilization_percent', 0),
            )
            snapshot_id = self.db.insert_sample(sample)
        except Exception:
            pass

        event = TrainingEvent(
            job_id=self.job_id,
            timestamp=time.time(),
            event_type=event_type,
            details=json.dumps(details) if details else None,
            memory_snapshot_id=snapshot_id
        )

        self.db.insert_event(event)

    def record_step(
        self,
        step: int,
        loss: Optional[float] = None,
        batch_size: Optional[int] = None,
        gradient_norm: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """
        Record metrics for a training step.

        Args:
            step: Current training step number
            loss: Training loss value
            batch_size: Current batch size
            gradient_norm: Gradient norm (if computed)
            learning_rate: Current learning rate
        """
        if not self.enabled or not self.db:
            return

        current_time = time.time()
        step_time = None

        if self._last_step_time is not None:
            step_time = current_time - self._last_step_time
        self._last_step_time = current_time

        metric = StepMetric(
            job_id=self.job_id,
            step=step,
            timestamp=current_time,
            loss=loss,
            batch_size=batch_size,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            step_time_seconds=step_time
        )

        self.db.insert_step_metric(metric)

    def record_oom_error(self, error_message: str, batch_size: Optional[int] = None):
        """Record an out-of-memory error event."""
        self.record_event('oom_error', {
            'error_message': error_message,
            'batch_size': batch_size,
        })

    def record_batch_adjustment(self, old_batch_size: int, new_batch_size: int, reason: str = ''):
        """Record a batch size adjustment event."""
        self.record_event('batch_adjustment', {
            'old_batch_size': old_batch_size,
            'new_batch_size': new_batch_size,
            'reason': reason,
        })

    def record_sampling_start(self, prompt: str = ''):
        """Record start of image sampling."""
        self.record_event('sampling_start', {
            'prompt': prompt[:100] if prompt else '',  # Truncate long prompts
        })

    def record_sampling_end(self, duration_seconds: float = 0):
        """Record end of image sampling."""
        self.record_event('sampling_end', {
            'duration_seconds': duration_seconds,
        })

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            # Record the error before stopping
            self.record_event('training_error', {
                'error_type': str(exc_type.__name__) if exc_type else 'Unknown',
                'error_message': str(exc_val) if exc_val else '',
            })
        self.stop()
        return False  # Don't suppress exceptions
