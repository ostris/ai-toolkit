"""
Post-training analysis engine that generates optimization recommendations.
Analyzes metrics, log files, and training patterns to identify issues.
"""

import os
import json
from typing import List, Dict, Any, Optional
from .database import MonitoringDatabase
from .log_parser import parse_training_log


class TrainingAnalyzer:
    """
    Analyzes training metrics and logs to generate optimization recommendations.
    """

    def __init__(
        self,
        db_path: str,
        job_id: str,
        training_folder: str = 'output',
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
    ):
        self.db = MonitoringDatabase(db_path)
        self.job_id = job_id
        self.training_folder = training_folder
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def analyze(self, job_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform complete analysis and generate report.

        Args:
            job_name: Name of the job (for log file location)

        Returns:
            Complete analysis report with recommendations
        """
        # Gather all data
        samples = self.db.get_samples_for_job(self.job_id)
        events = self.db.get_events_for_job(self.job_id)
        step_metrics = self.db.get_step_metrics_for_job(self.job_id)

        # Parse log file if available
        log_errors = []
        if job_name:
            log_path = os.path.join(self.training_folder, job_name, 'log.txt')
            if os.path.exists(log_path):
                log_data = parse_training_log(log_path)
                log_errors = log_data.get('errors', []) + log_data.get('oom_events', [])

        # Generate analysis
        summary = self._generate_summary(samples, events, step_metrics)
        recommendations = self._generate_recommendations(samples, events, step_metrics, log_errors)

        # Determine health score
        health_score = self._calculate_health_score(summary, recommendations, log_errors)

        # Save report to database
        self.db.save_analysis_report(
            job_id=self.job_id,
            summary=summary,
            recommendations=recommendations,
            log_errors=log_errors,
            peak_memory_gb=summary.get('peak_memory_gb', 0),
            health_score=health_score
        )

        return {
            'summary': summary,
            'recommendations': recommendations,
            'log_errors': log_errors,
            'health_score': health_score,
            'metrics_timeline': samples,
        }

    def _generate_summary(
        self,
        samples: List[Dict],
        events: List[Dict],
        step_metrics: List[Dict]
    ) -> Dict[str, Any]:
        """Generate summary statistics from collected data."""
        summary = {
            'peak_memory_gb': 0.0,
            'total_memory_gb': 0.0,
            'avg_memory_utilization': 0.0,
            'peak_gpu_memory_gb': 0.0,
            'avg_gpu_utilization': 0.0,
            'steps_completed': 0,
            'duration_minutes': 0.0,
            'training_completed': False,
            'oom_count': 0,
            'batch_adjustments': 0,
            'worker_count': 0,
            'swap_usage_detected': False,
        }

        if not samples:
            return summary

        # Memory statistics
        memory_values = [s['used_memory_gb'] for s in samples if s['used_memory_gb']]
        if memory_values:
            summary['peak_memory_gb'] = max(memory_values)
            summary['avg_memory_utilization'] = sum(memory_values) / len(memory_values)

        # Total memory (from first sample)
        if samples:
            summary['total_memory_gb'] = samples[0].get('total_memory_gb', 0)

        # GPU statistics
        gpu_memory = [s['gpu_memory_used_gb'] for s in samples if s['gpu_memory_used_gb']]
        if gpu_memory:
            summary['peak_gpu_memory_gb'] = max(gpu_memory)

        gpu_util = [s['gpu_utilization_percent'] for s in samples if s['gpu_utilization_percent']]
        if gpu_util:
            summary['avg_gpu_utilization'] = sum(gpu_util) / len(gpu_util)

        # Worker information
        worker_counts = [s['worker_count'] for s in samples if s['worker_count']]
        if worker_counts:
            summary['worker_count'] = max(worker_counts)

        # Swap usage
        swap_values = [s['swap_used_gb'] for s in samples if s['swap_used_gb'] > 0.1]
        summary['swap_usage_detected'] = len(swap_values) > 0

        # Step metrics
        if step_metrics:
            summary['steps_completed'] = max(m['step'] for m in step_metrics)

        # Duration
        if samples and len(samples) >= 2:
            duration_seconds = samples[-1]['timestamp'] - samples[0]['timestamp']
            summary['duration_minutes'] = duration_seconds / 60

        # Events analysis
        for event in events:
            if event['event_type'] == 'training_end':
                summary['training_completed'] = True
            elif event['event_type'] == 'oom_error':
                summary['oom_count'] += 1
            elif event['event_type'] == 'batch_adjustment':
                summary['batch_adjustments'] += 1

        return summary

    def _generate_recommendations(
        self,
        samples: List[Dict],
        events: List[Dict],
        step_metrics: List[Dict],
        log_errors: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate priority-ranked optimization recommendations."""
        recommendations = []

        if not samples:
            return recommendations

        # Calculate key metrics
        peak_memory = max((s['used_memory_gb'] for s in samples), default=0)
        total_memory = samples[0].get('total_memory_gb', 0) if samples else 0
        memory_ratio = peak_memory / total_memory if total_memory > 0 else 0

        worker_count = max((s['worker_count'] for s in samples), default=0)
        worker_memory = max((s['worker_memory_gb'] for s in samples), default=0)

        has_oom = any(e['event_type'] == 'oom_error' for e in events)
        has_oom_in_logs = any('oom' in str(err.get('type', '')).lower() for err in log_errors)

        swap_detected = any(s['swap_used_gb'] > 0.1 for s in samples)

        # Check current batch size from step metrics
        current_batch_size = None
        if step_metrics:
            batch_sizes = [m['batch_size'] for m in step_metrics if m['batch_size']]
            if batch_sizes:
                current_batch_size = batch_sizes[-1]

        # === CRITICAL RECOMMENDATIONS (OOM or >95% memory) ===

        if has_oom or has_oom_in_logs or memory_ratio >= self.critical_threshold:
            severity = 'critical'

            # Recommendation: Reduce workers
            if worker_count > 0:
                worker_savings = worker_memory
                recommendations.append({
                    'severity': severity,
                    'title': 'Reduce worker count',
                    'action': f'Set num_workers: 0',
                    'impact': f'Saves ~{worker_savings:.1f}GB memory ({worker_count} workers currently using {worker_memory:.1f}GB)',
                    'config_path': 'config.process[0].datasets[0].num_workers',
                    'current_value': worker_count,
                    'recommended_value': 0,
                    'priority': 1,
                })

            # Recommendation: Enable quantization
            recommendations.append({
                'severity': severity,
                'title': 'Enable model quantization',
                'action': 'Set quantize: true, qtype: qfloat8',
                'impact': 'Reduces model memory by ~50% (uses 8-bit weights)',
                'config_path': 'config.process[0].model.quantize',
                'current_value': None,
                'recommended_value': True,
                'priority': 2,
            })

            # Recommendation: Reduce batch size
            if current_batch_size and current_batch_size > 1:
                new_batch = max(1, current_batch_size // 2)
                recommendations.append({
                    'severity': severity,
                    'title': 'Reduce batch size',
                    'action': f'Set max_batch_size: {new_batch}',
                    'impact': f'Reduces peak VRAM usage (current: {current_batch_size})',
                    'config_path': 'config.process[0].train.max_batch_size',
                    'current_value': current_batch_size,
                    'recommended_value': new_batch,
                    'priority': 3,
                })

            # Recommendation: Enable gradient checkpointing
            recommendations.append({
                'severity': severity,
                'title': 'Enable gradient checkpointing',
                'action': 'Set gradient_checkpointing: true',
                'impact': 'Trades ~30% compute for significant memory savings',
                'config_path': 'config.process[0].train.gradient_checkpointing',
                'current_value': None,
                'recommended_value': True,
                'priority': 4,
            })

            # Recommendation: Disable EMA
            recommendations.append({
                'severity': severity,
                'title': 'Disable EMA',
                'action': 'Set use_ema: false',
                'impact': 'Saves ~1GB memory (EMA maintains copy of weights)',
                'config_path': 'config.process[0].train.ema_config.use_ema',
                'current_value': None,
                'recommended_value': False,
                'priority': 5,
            })

        # === WARNING RECOMMENDATIONS (>85% memory but stable) ===

        elif memory_ratio >= self.warning_threshold:
            severity = 'warning'

            if swap_detected:
                recommendations.append({
                    'severity': severity,
                    'title': 'Swap usage detected',
                    'action': 'Reduce memory usage to avoid swapping',
                    'impact': 'Swap is 10-100x slower than RAM, significantly impacts training speed',
                    'config_path': None,
                    'current_value': None,
                    'recommended_value': None,
                    'priority': 1,
                })

            if worker_count > 0:
                recommendations.append({
                    'severity': severity,
                    'title': 'Consider reducing workers',
                    'action': f'Try num_workers: 0 or {max(0, worker_count - 1)}',
                    'impact': f'Workers using {worker_memory:.1f}GB, reducing may prevent OOM',
                    'config_path': 'config.process[0].datasets[0].num_workers',
                    'current_value': worker_count,
                    'recommended_value': max(0, worker_count - 1),
                    'priority': 2,
                })

            # Resolution recommendation
            recommendations.append({
                'severity': severity,
                'title': 'Consider lower resolution',
                'action': 'Reduce training resolution (e.g., 1024 instead of 1536)',
                'impact': 'Memory scales quadratically with resolution',
                'config_path': 'config.process[0].datasets[0].resolution',
                'current_value': None,
                'recommended_value': None,
                'priority': 3,
            })

        # === INFO RECOMMENDATIONS (optimization opportunities) ===

        else:
            severity = 'info'

            # Check if batch size could be increased
            if memory_ratio < 0.7 and current_batch_size:
                headroom = total_memory - peak_memory
                # Rough estimate: each batch increase uses ~2GB more
                potential_increase = int(headroom / 3)
                if potential_increase > 0:
                    new_batch = current_batch_size + potential_increase
                    recommendations.append({
                        'severity': severity,
                        'title': 'Batch size can be increased',
                        'action': f'Memory allows batch_size: {new_batch}',
                        'impact': f'Currently using {current_batch_size}, {headroom:.1f}GB headroom available',
                        'config_path': 'config.process[0].train.batch_size',
                        'current_value': current_batch_size,
                        'recommended_value': new_batch,
                        'priority': 1,
                    })

            # If memory comfortable, can disable gradient checkpointing for speed
            if memory_ratio < 0.6:
                recommendations.append({
                    'severity': severity,
                    'title': 'Can disable gradient checkpointing',
                    'action': 'Set gradient_checkpointing: false for faster training',
                    'impact': '~30% speed improvement, memory usage is comfortable',
                    'config_path': 'config.process[0].train.gradient_checkpointing',
                    'current_value': None,
                    'recommended_value': False,
                    'priority': 2,
                })

        # Sort by priority
        recommendations.sort(key=lambda x: (
            {'critical': 0, 'warning': 1, 'info': 2}.get(x['severity'], 3),
            x.get('priority', 99)
        ))

        return recommendations

    def _calculate_health_score(
        self,
        summary: Dict[str, Any],
        recommendations: List[Dict],
        log_errors: List[Dict]
    ) -> str:
        """Calculate overall health score for the training run."""
        # Critical conditions
        if summary.get('oom_count', 0) > 0:
            return 'critical'

        if any('oom' in str(err.get('type', '')).lower() for err in log_errors):
            return 'critical'

        total_memory = summary.get('total_memory_gb', 1)
        peak_memory = summary.get('peak_memory_gb', 0)
        memory_ratio = peak_memory / total_memory if total_memory > 0 else 0

        if memory_ratio >= self.critical_threshold:
            return 'critical'

        # Warning conditions
        if memory_ratio >= self.warning_threshold:
            return 'warning'

        if summary.get('swap_usage_detected', False):
            return 'warning'

        if any(r['severity'] == 'critical' for r in recommendations):
            return 'critical'

        if any(r['severity'] == 'warning' for r in recommendations):
            return 'warning'

        # Good health
        return 'good'


def analyze_training_job(
    db_path: str,
    job_id: str,
    job_name: Optional[str] = None,
    training_folder: str = 'output'
) -> Dict[str, Any]:
    """
    Convenience function to analyze a completed training job.

    Args:
        db_path: Path to the SQLite database
        job_id: Unique identifier for the training job
        job_name: Name of the job (for log file)
        training_folder: Base folder for training outputs

    Returns:
        Complete analysis report
    """
    analyzer = TrainingAnalyzer(db_path, job_id, training_folder)
    return analyzer.analyze(job_name)
