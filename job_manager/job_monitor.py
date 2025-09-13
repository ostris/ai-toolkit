
import sqlite3
import time
import threading
from pathlib import Path
from typing import Callable, Dict, List

class JobMonitor:

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).absolute()

    def monitor_job_status(self, job_id: str, stop_event: threading.Event, status_callback: Callable[[Dict], None]):
        while not stop_event.is_set():
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT status, info, step, speed_string, stop FROM Job WHERE id = ?",
                        (job_id,)
                    )
                    result = cursor.fetchone()
                    if result:
                        status, info, step, speed, stop_flag = result
                        status_update = {
                            "job_id": job_id,
                            "status": status,
                            "info": info,
                            "step": step,
                            "speed_string": speed
                        }
                        status_callback(status_update)
                        if status in ['completed', 'stopped', 'error']:
                            break
            except Exception as e:
                status_callback({"job_id": job_id, "status": "error", "info": f"Monitor error: {str(e)}"})
            time.sleep(2)

    def monitor_logs(self, log_path: str, stop_event: threading.Event, log_callback: Callable[[str], None]):
        last_pos = 0
        while not stop_event.is_set():
            try:
                with open(log_path, 'r') as f:
                    f.seek(last_pos)
                    lines = f.readlines()
                    last_pos = f.tell()
                    for line in lines:
                        log_callback(line.strip())
            except Exception as e:
                log_callback(f"Log monitor error: {str(e)}")
            time.sleep(1)

    def get_rolling_logs(self, log_path: str, max_lines: int = 100) -> List[str]:
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                return lines[-max_lines:] if lines else ["No logs available"]
        except Exception as e:
            return [f"Log error: {str(e)}"]
