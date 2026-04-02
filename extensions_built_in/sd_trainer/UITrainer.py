from collections import OrderedDict
import os
import sqlite3
import asyncio
import concurrent.futures
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from typing import Literal, Optional
import threading
import time
import signal
import torch

AITK_Status = Literal["running", "stopped", "error", "completed"]


class UITrainer(SDTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(UITrainer, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        if not os.path.exists(self.sqlite_db_path):
            raise Exception(
                f"SQLite database not found at {self.sqlite_db_path}")
        print(f"Using SQLite database at {self.sqlite_db_path}")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = self.job_id.strip() if self.job_id is not None else None
        print(f"Job ID: \"{self.job_id}\"")
        if self.job_id is None:
            raise Exception("AITK_JOB_ID not set")
        self.is_stopping = False
        self.is_returning_to_queue = False
        # Create a thread pool for database operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Track all async tasks
        self._async_tasks = []
        # Initialize the status
        self._run_async_operation(self._update_status("running", "Starting"))
        self._last_speed_update = 0.0
        self._stop_watcher_started = False
        self.start_stop_watcher(interval_sec=2.0)
    
    def start_stop_watcher(self, interval_sec: float = 5.0):
        """
        Start a daemon thread that periodically checks should_stop()
        and terminates the process immediately when triggered.
        In distributed mode, only rank 0 polls the DB; it kills the
        entire process group so all ranks terminate.
        """
        # Only rank 0 should poll the DB to avoid SQLite locking issues
        if self.accelerator.num_processes > 1 and not self.accelerator.is_main_process:
            return
        if getattr(self, "_stop_watcher_started", False):
            return
        self._stop_watcher_started = True
        t = threading.Thread(
            target=self._stop_watcher_thread, args=(interval_sec,), daemon=True
        )
        t.start()

    def _stop_watcher_thread(self, interval_sec: float):
        while True:
            try:
                if self.should_stop():
                    self.is_stopping = True
                    print("")
                    print("****************************************************")
                    print("    Stop signal received; terminating process.      ")
                    print("****************************************************")
                    if self.accelerator.num_processes > 1:
                        # FSDP: don't kill — let end_step_hook broadcast the flag
                        # so all ranks save a temp checkpoint together.
                        # Don't shut down thread_pool here; on_error() still needs it.
                        # Force kill after 5 min as a last-resort fallback.
                        for _ in range(150):
                            time.sleep(2)
                        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
                    else:
                        # Single GPU: update status, flush, then SIGINT.
                        self._run_async_operation(
                            self._update_status("stopped", "Job stopped")
                        )
                        try:
                            asyncio.run(self.wait_for_all_async())
                        except RuntimeError:
                            pass
                        try:
                            self.thread_pool.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            self.thread_pool.shutdown(wait=False)
                        os.kill(os.getpid(), signal.SIGINT)
                    break  # don't loop — avoid repeated signals interrupting save
                time.sleep(interval_sec)
            except Exception:
                time.sleep(interval_sec)

    def _run_async_operation(self, coro):
        """Helper method to run an async coroutine and track the task."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Prune completed tasks periodically to avoid unbounded growth
        if len(self._async_tasks) > 100:
            self._async_tasks = [t for t in self._async_tasks if not t.done()]

        # Create a task and track it
        if loop.is_running():
            task = asyncio.run_coroutine_threadsafe(coro, loop)
            self._async_tasks.append(asyncio.wrap_future(task))
        else:
            task = loop.create_task(coro)
            self._async_tasks.append(task)
            loop.run_until_complete(task)

    async def _execute_db_operation(self, operation_func):
        """Execute a database operation in a separate thread to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, operation_func)

    def _db_connect(self):
        """Create a new connection for each operation to avoid locking."""
        conn = sqlite3.connect(self.sqlite_db_path, timeout=10.0)
        conn.isolation_level = None  # Enable autocommit mode
        return conn

    def should_stop(self):
        # In distributed mode, only rank 0 polls the DB to avoid SQLite locking
        if self.accelerator.num_processes > 1 and not self.accelerator.is_main_process:
            return False
        def _check_stop():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop FROM Job WHERE id = ?", (self.job_id,))
                stop = cursor.fetchone()
                return False if stop is None else stop[0] == 1

        return _check_stop()

    def should_return_to_queue(self):
        # In distributed mode, only rank 0 polls the DB to avoid SQLite locking
        if self.accelerator.num_processes > 1 and not self.accelerator.is_main_process:
            return False
        def _check_return_to_queue():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT return_to_queue FROM Job WHERE id = ?", (self.job_id,))
                return_to_queue = cursor.fetchone()
                return False if return_to_queue is None else return_to_queue[0] == 1

        return _check_return_to_queue()

    def maybe_stop(self):
        # In distributed mode, only rank 0 checks stop signals.
        # Non-rank-0 processes are terminated via the stop watcher's os.killpg().
        # We CANNOT use dist.broadcast() here because maybe_stop() is called from
        # rank-0-only code paths (inside sample(), save(), sample_step_hook()) where
        # other ranks have already diverged. A broadcast would deadlock.
        if self.accelerator.num_processes > 1 and not self.accelerator.is_main_process:
            return
        if self.should_stop():
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self.is_stopping = True
            raise Exception("Job stopped")
        if self.should_return_to_queue():
            self._run_async_operation(
                self._update_status("queued", "Job queued"))
            self.is_stopping = True
            self.is_returning_to_queue = True
            raise Exception("Job returning to queue")

    async def _update_key(self, key, value):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    # Convert the value to string if it's not already
                    if isinstance(value, str):
                        value_to_insert = value
                    else:
                        value_to_insert = str(value)

                    # Use parameterized query for both the column name and value
                    update_query = f"UPDATE Job SET {key} = ? WHERE id = ?"
                    cursor.execute(
                        update_query, (value_to_insert, self.job_id))
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_step(self):
        """Non-blocking update of the step count."""
        if self.accelerator.is_main_process:
            self._run_async_operation(self._update_key("step", self.step_num))

    def update_db_key(self, key, value):
        """Non-blocking update a key in the database."""
        if self.accelerator.is_main_process:
            self._run_async_operation(self._update_key(key, value))

    async def _update_status(self, status: AITK_Status, info: Optional[str] = None):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    if info is not None:
                        cursor.execute(
                            "UPDATE Job SET status = ?, info = ? WHERE id = ?",
                            (status, info, self.job_id)
                        )
                    else:
                        cursor.execute(
                            "UPDATE Job SET status = ? WHERE id = ?",
                            (status, self.job_id)
                        )
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        """Non-blocking update of status."""
        if self.accelerator.is_main_process:
            self._run_async_operation(self._update_status(status, info))

    async def wait_for_all_async(self):
        """Wait for all tracked async operations to complete."""
        if not self._async_tasks:
            return

        try:
            await asyncio.gather(*self._async_tasks)
        except Exception as e:
            pass
        finally:
            # Clear the task list after completion
            self._async_tasks.clear()

    def on_error(self, e: Exception):
        super(UITrainer, self).on_error(e)
        if self.accelerator.is_main_process:
            if self.is_returning_to_queue:
                self.update_status("queued", "Job queued")
            elif self.is_stopping:
                self.update_status("stopped", "Job stopped")
            else:
                self.update_status("error", str(e))
        self.update_db_key("step", self.last_save_step)
        try:
            asyncio.run(self.wait_for_all_async())
        except RuntimeError:
            pass
        self.thread_pool.shutdown(wait=True)

    def done_hook(self):
        super(UITrainer, self).done_hook()
        self.update_status("completed", "Training completed")
        # Wait for all async operations to finish before shutting down
        asyncio.run(self.wait_for_all_async())
        self.thread_pool.shutdown(wait=True)

    def end_step_hook(self):
        super(UITrainer, self).end_step_hook()
        self.update_step()
        # Update speed_string every ~10 seconds using timer's rolling average
        train_loop_timings = self.timer.timers.get('train_loop')
        if train_loop_timings and (time.time() - self._last_speed_update) >= 10.0:
            seconds_per_iter = sum(train_loop_timings) / len(train_loop_timings)
            if seconds_per_iter <= 0:
                pass
            elif seconds_per_iter < 1:
                self.update_db_key("speed_string", f"{1 / seconds_per_iter:.2f} iter/sec")
            else:
                self.update_db_key("speed_string", f"{seconds_per_iter:.2f} sec/iter")
            self._last_speed_update = time.time()
        # FSDP: broadcast stop flag so all ranks exit together for collective save.
        # Only uses the watcher's is_stopping flag (no DB query in hot path).
        if self.accelerator.num_processes > 1:
            import torch.distributed as dist
            flag = 1 if self.is_stopping else 0
            stop_tensor = torch.tensor([flag], device=self.accelerator.device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                self.is_stopping = True
                raise Exception("Job stopped")
        else:
            self.maybe_stop()

    def hook_before_model_load(self):
        super().hook_before_model_load()
        self.maybe_stop()
        self.update_status("running", "Loading model")

    def before_dataset_load(self):
        super().before_dataset_load()
        self.maybe_stop()
        self.update_status("running", "Loading dataset")

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self.maybe_stop()
        self.update_step()
        self.update_status("running", "Training")
    def status_update_hook_func(self, string):
        self.update_status("running", string)

    def hook_after_sd_init_before_load(self):
        super().hook_after_sd_init_before_load()
        self.maybe_stop()
        self.sd.add_status_update_hook(self.status_update_hook_func)

    def sample_step_hook(self, img_num, total_imgs):
        super().sample_step_hook(img_num, total_imgs)
        self.maybe_stop()
        self.update_status(
            "running", f"Generating images - {img_num + 1}/{total_imgs}")

    def sample(self, step=None, is_first=False):
        self.maybe_stop()
        total_imgs = len(self.sample_config.prompts)
        self.update_status("running", f"Generating images - 0/{total_imgs}")
        super().sample(step, is_first)
        self.maybe_stop()
        self.update_status("running", "Training")

    def save(self, step=None, is_temp=False):
        if not is_temp:
            # Under FSDP, skip pre-save maybe_stop() — it could throw on rank 0
            # before the collective save, deadlocking other ranks in full_tensor().
            if not self.use_fsdp:
                self.maybe_stop()
            self.update_status("running", "Saving model")
        super().save(step, is_temp=is_temp)
        if not is_temp:
            if not self.use_fsdp:
                self.maybe_stop()
            self.update_status("running", "Training")
