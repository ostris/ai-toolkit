"""UI job bookkeeping for long-running extension processes.

When a process is launched by the UI it gets AITK_JOB_ID and a sqlite db
path; status / step / info are written to the Job row and the UI's stop
button sets Job.stop. This mirrors what BaseCaptioner does inline, as a
standalone helper so other UI-driven processes (the inference engine) can
share it. Outside the UI (no job id or no db file) every call is a no-op.
"""

import os
import sqlite3
import threading
import time
from typing import Callable, Literal, Optional

AITK_Status = Literal["running", "stopped", "error", "completed", "queued"]


class UIJobStatus:
    def __init__(self, sqlite_db_path: str = "./aitk_db.db"):
        self.sqlite_db_path = sqlite_db_path
        job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = job_id.strip() if job_id else None
        self.is_ui_job = self.job_id is not None and os.path.exists(sqlite_db_path)
        if self.is_ui_job:
            print(f"Using SQLite database at {sqlite_db_path}")
            print(f'Job ID: "{self.job_id}"')
        self._lock = threading.Lock()
        self._watcher = None
        # collapse bursts of identical writes (progress ticks)
        self._last_info = None

    # ---- db plumbing ----
    def _connect(self):
        conn = sqlite3.connect(self.sqlite_db_path, timeout=30.0)
        conn.isolation_level = None
        return conn

    def _retry(self, fn, max_retries=3, base_delay=2.0):
        last = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except sqlite3.OperationalError as e:
                if "database is locked" not in str(e):
                    raise
                last = e
                if attempt < max_retries:
                    time.sleep(base_delay * (2**attempt))
        raise last

    def _write(self, query: str, params: tuple):
        if not self.is_ui_job:
            return

        def _do():
            with self._lock, self._connect() as conn:
                cur = conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                try:
                    cur.execute(query, params)
                finally:
                    cur.execute("COMMIT")

        try:
            self._retry(_do)
        except Exception as e:
            print(f"[AITK] Warning: db write failed: {e}")

    def _read_flag(self, column: str) -> bool:
        if not self.is_ui_job:
            return False

        def _do():
            with self._lock, self._connect() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT {column} FROM Job WHERE id = ?", (self.job_id,))
                row = cur.fetchone()
                return False if row is None else row[0] == 1

        return self._retry(_do)

    # ---- public api ----
    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        if info is None:
            self._write("UPDATE Job SET status = ? WHERE id = ?", (status, self.job_id))
        else:
            self._last_info = info
            self._write(
                "UPDATE Job SET status = ?, info = ? WHERE id = ?",
                (status, info, self.job_id),
            )

    def update_info(self, info: str):
        if info == self._last_info:
            return
        self._last_info = info
        self._write("UPDATE Job SET info = ? WHERE id = ?", (info, self.job_id))

    def update_key(self, key: str, value):
        # key is a column name from our own code, never user input
        self._write(f"UPDATE Job SET {key} = ? WHERE id = ?", (str(value), self.job_id))

    def should_stop(self) -> bool:
        return self._read_flag("stop")

    def should_return_to_queue(self) -> bool:
        return self._read_flag("return_to_queue")

    def start_stop_watcher(self, on_stop: Callable[[str], None], interval_sec: float = 2.0):
        """Poll the stop / return_to_queue flags on a daemon thread and call
        on_stop("stopped" | "queued") once when one is set."""
        if not self.is_ui_job or self._watcher is not None:
            return

        def _loop():
            while True:
                try:
                    if self.should_stop():
                        on_stop("stopped")
                        return
                    if self.should_return_to_queue():
                        on_stop("queued")
                        return
                except Exception:
                    pass
                time.sleep(interval_sec)

        self._watcher = threading.Thread(target=_loop, daemon=True)
        self._watcher.start()
