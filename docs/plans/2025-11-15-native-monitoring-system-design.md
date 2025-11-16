# Native Training Monitoring System Design

## Overview

A self-monitoring system that collects memory, performance, and training metrics during training runs, stores them in SQLite, and provides post-run analysis with optimization recommendations.

## Goals

1. Automatically collect comprehensive system metrics during training
2. Store metrics in existing `aitk_db.db` SQLite database
3. Analyze data after training to identify optimization opportunities
4. Display recommendations in the job details UI page
5. Help users optimize settings for their specific hardware (especially unified memory systems)

## Data Collection Architecture

Three collectors run concurrently during training:

### 1. Step-based Collector
- Triggers every N steps (configured by `performance_log_every`)
- Records: step number, loss, batch size, gradient norm, learning rate, step time
- Minimal overhead, integrated into existing training loop

### 2. Event-based Collector
- Hooks into key training events
- Events: training start/end, OOM errors, batch size adjustments, sampling start/end
- Records: timestamp, event type, config snapshot, memory state at that moment

### 3. Background Sampler (separate thread)
- Samples system state every 5 seconds (configurable)
- Platform-specific collection:
  - **macOS/Grace Hopper (unified memory)**: `vm_stat`, `ps aux`
  - **Linux**: `/proc/meminfo`, `/proc/<pid>/status`
  - **NVIDIA discrete GPU**: `nvidia-smi --query-gpu=memory.used,utilization.gpu`
- Tracks: total memory, used memory, swap, per-process breakdown, GPU utilization

## Database Schema

New tables in `aitk_db.db`:

### MetricsSample
```sql
CREATE TABLE MetricsSample (
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
```

### TrainingEvent
```sql
CREATE TABLE TrainingEvent (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  timestamp REAL NOT NULL,
  event_type TEXT NOT NULL,
  details TEXT,
  memory_snapshot_id INTEGER,
  FOREIGN KEY (memory_snapshot_id) REFERENCES MetricsSample(id)
);
```

### StepMetric
```sql
CREATE TABLE StepMetric (
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
```

### AnalysisReport
```sql
CREATE TABLE AnalysisReport (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  created_at REAL NOT NULL,
  summary JSON NOT NULL,
  recommendations JSON NOT NULL,
  log_errors JSON,
  peak_memory_gb REAL,
  health_score TEXT
);
```

Tables are created automatically on first use (no manual migration required).

## Analysis Engine

Runs after training completes (or crashes). Performs:

### Memory Analysis
- Peak memory usage vs available system memory
- Worker memory proportion of total
- Swap usage indicating memory pressure
- Correlation of memory with batch size changes

### Log File Analysis
- Parses `output/{job_name}/log.txt`
- Extracts warnings and errors with timestamps
- Detects: OOM errors, CUDA errors, configuration issues
- Correlates log timestamps with metric samples

### Recommendation Generation

Priority-ranked suggestions based on patterns:

**Critical (OOM or >90% memory):**
1. "Set num_workers: 0" - Saves {X}GB per worker
2. "Enable quantization (qtype: qfloat8)" - Reduces model memory by 50%
3. "Disable EMA (use_ema: false)" - Saves ~1GB for LoRA
4. "Reduce max_batch_size to {N}" - Based on actual peak usage

**Warning (>85% memory but stable):**
1. "Enable gradient checkpointing" - Trades compute for memory
2. "Consider lower resolution" - Quadratic memory scaling

**Info (optimization opportunities):**
1. "Memory allows batch_size: {N}, currently using {M}" - Can increase throughput
2. "Disable gradient checkpointing for speed" - If memory is comfortable

Output is structured JSON with:
- Health score: Good/Warning/Critical
- Peak metrics summary
- Priority-ranked recommendations with expected impact
- Config paths for each suggestion

## UI Integration

### Job Details Page Enhancement

New "Performance Analysis" panel on `/jobs/[jobID]` page:

1. **Summary Card**
   - Health score badge (Good/Warning/Critical)
   - Peak memory: "119GB / 120GB (99%)"
   - Training duration and completion status
   - Error count from log analysis

2. **Recommendations List**
   - Priority-ordered cards
   - Each shows: severity, what to change, expected impact, config path
   - "Apply Recommended Settings" button pre-fills new job

3. **Memory Timeline Chart**
   - Line chart of memory over time
   - Event annotations (batch changes, OOM, sampling)
   - Threshold indicators (warning/critical zones)

4. **Log Errors Section**
   - Extracted warnings/errors from log.txt
   - Timestamp and correlation with memory state
   - Stack traces where relevant

### API Endpoint

`GET /api/jobs/[jobID]/analysis`

Returns:
```json
{
  "summary": {
    "health_score": "critical",
    "peak_memory_gb": 119.2,
    "total_memory_gb": 120,
    "training_completed": false,
    "steps_completed": 450,
    "duration_minutes": 32.5
  },
  "recommendations": [
    {
      "severity": "critical",
      "title": "Reduce worker count",
      "action": "Set num_workers: 0",
      "impact": "Saves ~16GB memory",
      "config_path": "config.process[0].datasets[0].num_workers",
      "current_value": 2,
      "recommended_value": 0
    }
  ],
  "log_errors": [
    {
      "timestamp": 1731723012.5,
      "level": "ERROR",
      "message": "RuntimeError: CUDA out of memory",
      "memory_at_time_gb": 119.2
    }
  ],
  "metrics_timeline": [...]
}
```

## Configuration

Optional config section (uses sensible defaults if omitted):

```yaml
config:
  process:
    - monitoring:
        enabled: true
        sample_interval_seconds: 5
        track_per_process: true
        analyze_on_complete: true
        memory_warning_threshold: 0.85
        memory_critical_threshold: 0.95
```

## Implementation Components

### Python (Backend)
- `toolkit/monitoring/collector.py` - Main monitoring orchestrator
- `toolkit/monitoring/samplers.py` - Platform-specific metric collection
- `toolkit/monitoring/database.py` - SQLite operations and auto-migration
- `toolkit/monitoring/analyzer.py` - Post-run analysis and recommendations
- `toolkit/monitoring/log_parser.py` - Parse log.txt for warnings/errors
- Modify `extensions_built_in/sd_trainer/DiffusionTrainer.py` - Hook into training loop

### TypeScript (Frontend)
- `ui/src/app/api/jobs/[jobID]/analysis/route.ts` - Analysis API endpoint
- `ui/src/app/jobs/[jobID]/components/PerformanceAnalysis.tsx` - Main panel
- `ui/src/app/jobs/[jobID]/components/MemoryTimeline.tsx` - Chart component
- `ui/src/app/jobs/[jobID]/components/RecommendationList.tsx` - Suggestions list
- Modify `ui/src/app/jobs/[jobID]/page.tsx` - Add analysis panel

## Performance Overhead

- Background thread sleeps between samples: <1% CPU
- SQLite writes batched (every 10 samples): minimal I/O
- Estimated disk usage: ~50MB per hour of training
- No impact on training throughput

## Migration Strategy

Automatic on first use:
```python
def ensure_monitoring_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='MetricsSample'"
    )
    if cursor.fetchone() is None:
        conn.executescript(CREATE_MONITORING_TABLES_SQL)
    conn.close()
```

User never needs to run manual migrations.

## Success Criteria

1. System automatically tracks memory usage without user configuration
2. Analysis correctly identifies OOM causes (workers, EMA, batch size, etc.)
3. Recommendations are specific and actionable with expected impact
4. UI clearly shows what went wrong and how to fix it
5. Users can optimize training runs based on data, not guesswork
