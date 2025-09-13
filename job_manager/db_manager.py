
import sqlite3
from pathlib import Path
from collections import OrderedDict
import json
import uuid
from typing import Dict

class DatabaseManager:

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).absolute()
        self._initialize_schema()

    def _initialize_schema(self):
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Job (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    gpu_ids TEXT NOT NULL,
                    job_config TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'stopped',
                    stop BOOLEAN DEFAULT 0,
                    step INTEGER DEFAULT 0,
                    info TEXT DEFAULT '',
                    speed_string TEXT DEFAULT ''
                )
            ''')
            conn.commit()

    def create_job_entry(self, job_config: OrderedDict, job_name: str, gpu_ids: str) -> str:
        job_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO Job (id, name, gpu_ids, job_config, status, info)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (job_id, job_name, gpu_ids, json.dumps(job_config), 'running', 'Starting job...'))
            conn.commit()
        return job_id

    def get_job_status(self, job_id: str) -> Dict:
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT status, info, step, speed_string FROM Job WHERE id = ?",
                    (job_id,)
                )
                result = cursor.fetchone()
                if result:
                    status, info, step, speed = result
                    return {
                        "job_id": job_id,
                        "status": status,
                        "info": info,
                        "step": step,
                        "speed_string": speed
                    }
                return {"job_id": job_id, "status": "error", "info": "Job not found"}
        except Exception as e:
            return {"job_id": job_id, "status": "error", "info": f"Status error: {str(e)}"}

    def set_stop_flag(self, job_id: str) -> bool:
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE Job SET stop = 1 WHERE id = ?", (job_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False

    def update_job_status_on_failure(self, job_id: str, exit_code: int):
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM Job WHERE id = ?", (job_id,))
            result = cursor.fetchone()
            if result and result[0] == 'running':
                cursor.execute(
                    "UPDATE Job SET status = ?, info = ? WHERE id = ?",
                    ('error', f"Process exited with code {exit_code}", job_id)
                )
                conn.commit()
