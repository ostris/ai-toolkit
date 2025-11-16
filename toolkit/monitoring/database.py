"""
Database operations for monitoring metrics storage.
Auto-creates tables on first use.
"""

import sqlite3
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


CREATE_MONITORING_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS MetricsSample (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    total_memory_gb REAL,
    used_memory_gb REAL,
    available_memory_gb REAL,
    swap_used_gb REAL,
    main_process_memory_gb REAL,
    worker_memory_gb REAL,
    worker_count INTEGER,
    gpu_memory_used_gb REAL,
    gpu_utilization_percent REAL,
    cpu_utilization_percent REAL
);

CREATE TABLE IF NOT EXISTS TrainingEvent (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    details TEXT,
    memory_snapshot_id INTEGER,
    FOREIGN KEY (memory_snapshot_id) REFERENCES MetricsSample(id)
);

CREATE TABLE IF NOT EXISTS StepMetric (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    loss REAL,
    batch_size INTEGER,
    gradient_norm REAL,
    learning_rate REAL,
    step_time_seconds REAL
);

CREATE TABLE IF NOT EXISTS AnalysisReport (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    summary TEXT NOT NULL,
    recommendations TEXT NOT NULL,
    log_errors TEXT,
    peak_memory_gb REAL,
    health_score TEXT
);

CREATE INDEX IF NOT EXISTS idx_metrics_job_id ON MetricsSample(job_id);
CREATE INDEX IF NOT EXISTS idx_events_job_id ON TrainingEvent(job_id);
CREATE INDEX IF NOT EXISTS idx_steps_job_id ON StepMetric(job_id);
CREATE INDEX IF NOT EXISTS idx_analysis_job_id ON AnalysisReport(job_id);
"""


@dataclass
class MetricsSample:
    job_id: str
    timestamp: float
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    swap_used_gb: float = 0.0
    main_process_memory_gb: float = 0.0
    worker_memory_gb: float = 0.0
    worker_count: int = 0
    gpu_memory_used_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0


@dataclass
class TrainingEvent:
    job_id: str
    timestamp: float
    event_type: str
    details: Optional[str] = None
    memory_snapshot_id: Optional[int] = None


@dataclass
class StepMetric:
    job_id: str
    step: int
    timestamp: float
    loss: Optional[float] = None
    batch_size: Optional[int] = None
    gradient_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    step_time_seconds: Optional[float] = None


class MonitoringDatabase:
    """Handles all database operations for monitoring data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_tables()
        self._sample_buffer: List[MetricsSample] = []
        self._buffer_size = 10  # Batch writes every 10 samples

    def _ensure_tables(self):
        """Create monitoring tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(CREATE_MONITORING_TABLES_SQL)
            conn.commit()
        finally:
            conn.close()

    def insert_sample(self, sample: MetricsSample) -> int:
        """Insert a single metrics sample."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO MetricsSample (
                    job_id, timestamp, total_memory_gb, used_memory_gb,
                    available_memory_gb, swap_used_gb, main_process_memory_gb,
                    worker_memory_gb, worker_count, gpu_memory_used_gb,
                    gpu_utilization_percent, cpu_utilization_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.job_id, sample.timestamp, sample.total_memory_gb,
                    sample.used_memory_gb, sample.available_memory_gb,
                    sample.swap_used_gb, sample.main_process_memory_gb,
                    sample.worker_memory_gb, sample.worker_count,
                    sample.gpu_memory_used_gb, sample.gpu_utilization_percent,
                    sample.cpu_utilization_percent
                )
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def buffer_sample(self, sample: MetricsSample):
        """Add sample to buffer and flush if buffer is full."""
        self._sample_buffer.append(sample)
        if len(self._sample_buffer) >= self._buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        """Write all buffered samples to database."""
        if not self._sample_buffer:
            return

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                """
                INSERT INTO MetricsSample (
                    job_id, timestamp, total_memory_gb, used_memory_gb,
                    available_memory_gb, swap_used_gb, main_process_memory_gb,
                    worker_memory_gb, worker_count, gpu_memory_used_gb,
                    gpu_utilization_percent, cpu_utilization_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.job_id, s.timestamp, s.total_memory_gb,
                        s.used_memory_gb, s.available_memory_gb,
                        s.swap_used_gb, s.main_process_memory_gb,
                        s.worker_memory_gb, s.worker_count,
                        s.gpu_memory_used_gb, s.gpu_utilization_percent,
                        s.cpu_utilization_percent
                    )
                    for s in self._sample_buffer
                ]
            )
            conn.commit()
            self._sample_buffer.clear()
        finally:
            conn.close()

    def insert_event(self, event: TrainingEvent) -> int:
        """Insert a training event."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO TrainingEvent (
                    job_id, timestamp, event_type, details, memory_snapshot_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.job_id, event.timestamp, event.event_type,
                    event.details, event.memory_snapshot_id
                )
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def insert_step_metric(self, metric: StepMetric) -> int:
        """Insert a step metric."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO StepMetric (
                    job_id, step, timestamp, loss, batch_size,
                    gradient_norm, learning_rate, step_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.job_id, metric.step, metric.timestamp,
                    metric.loss, metric.batch_size, metric.gradient_norm,
                    metric.learning_rate, metric.step_time_seconds
                )
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_samples_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all metric samples for a job."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM MetricsSample WHERE job_id = ? ORDER BY timestamp",
                (job_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_events_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all events for a job."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM TrainingEvent WHERE job_id = ? ORDER BY timestamp",
                (job_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_step_metrics_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all step metrics for a job."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM StepMetric WHERE job_id = ? ORDER BY step",
                (job_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def save_analysis_report(
        self,
        job_id: str,
        summary: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        log_errors: List[Dict[str, Any]],
        peak_memory_gb: float,
        health_score: str
    ) -> int:
        """Save analysis report for a job."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO AnalysisReport (
                    job_id, created_at, summary, recommendations,
                    log_errors, peak_memory_gb, health_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, time.time(), json.dumps(summary),
                    json.dumps(recommendations), json.dumps(log_errors),
                    peak_memory_gb, health_score
                )
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_latest_analysis(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis report for a job."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT * FROM AnalysisReport
                WHERE job_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (job_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['summary'] = json.loads(result['summary'])
                result['recommendations'] = json.loads(result['recommendations'])
                result['log_errors'] = json.loads(result['log_errors']) if result['log_errors'] else []
                return result
            return None
        finally:
            conn.close()

    def get_peak_memory(self, job_id: str) -> float:
        """Get peak memory usage for a job."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT MAX(used_memory_gb) FROM MetricsSample WHERE job_id = ?",
                (job_id,)
            )
            result = cursor.fetchone()[0]
            return result if result else 0.0
        finally:
            conn.close()
