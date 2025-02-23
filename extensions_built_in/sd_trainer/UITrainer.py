from collections import OrderedDict
import os
import sqlite3
import asyncio
import concurrent.futures
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from typing import Literal, Optional

AITK_Status = Literal["running", "stopped", "error", "completed"]


class UITrainer(SDTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(UITrainer, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        print(f"Using SQLite database at {self.sqlite_db_path}")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        if self.job_id is None:
            raise Exception("AITK_JOB_ID not set")
        self.is_stopping = False
        # Create a thread pool for database operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Initialize the status
        self._run_async_operation(self._update_status("running", "Starting"))

    def _run_async_operation(self, coro):
        """Helper method to run an async coroutine in a new event loop."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the coroutine in the event loop
        if loop.is_running():
            # If we're already in an event loop, create a future
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            # We could wait for the result if needed: future.result()
        else:
            # If no loop is running, run the coroutine and close the loop
            loop.run_until_complete(coro)

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
        def _check_stop():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop FROM Job WHERE id = ?", (self.job_id,))
                stop = cursor.fetchone()
                return False if stop is None else stop[0] == 1

        # For this one we need a synchronous result, so we'll run it directly
        return _check_stop()

    def maybe_stop(self):
        if self.should_stop():
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self.is_stopping = True
            raise Exception("Job stopped")

    async def _update_step(self):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")  # Get an immediate lock
                try:
                    cursor.execute(
                        "UPDATE Job SET step = ? WHERE id = ?",
                        (self.step_num, self.job_id)
                    )
                finally:
                    cursor.execute("COMMIT")  # Release the lock

        await self._execute_db_operation(_do_update)

    def update_step(self):
        """Non-blocking update of the step count."""
        if self.accelerator.is_main_process:
            # Use the helper method to run the async operation
            self._run_async_operation(self._update_step())

    async def _update_status(self, status: AITK_Status, info: Optional[str] = None):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")  # Get an immediate lock
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
                    cursor.execute("COMMIT")  # Release the lock

        await self._execute_db_operation(_do_update)

    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        """Non-blocking update of status."""
        if self.accelerator.is_main_process:
            # Use the helper method to run the async operation
            self._run_async_operation(self._update_status(status, info))

    def on_error(self, e: Exception):
        super(UITrainer, self).on_error(e)
        if self.accelerator.is_main_process and not self.is_stopping:
            self.update_status("error", str(e))

    def done_hook(self):
        super(UITrainer, self).done_hook()
        self.update_status("completed", "Training completed")
        # Make sure we clean up the thread pool
        self.thread_pool.shutdown(wait=False)

    def end_step_hook(self):
        super(UITrainer, self).end_step_hook()
        self.update_step()
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

    def save(self, step=None):
        self.maybe_stop()
        self.update_status("running", "Saving model")
        super().save(step)
        self.maybe_stop()
        self.update_status("running", "Training")
