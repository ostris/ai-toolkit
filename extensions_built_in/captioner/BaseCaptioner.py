import asyncio
from collections import OrderedDict

import sqlite3
import os
from typing import Literal, Optional
import threading
import time
import signal
import concurrent.futures
from PIL import Image

import torch
from jobs.process import BaseExtensionProcess
import tqdm

from toolkit.train_tools import get_torch_dtype

AITK_Status = Literal["running", "stopped", "error", "completed"]


class CaptionConfig:
    def __init__(self, **kwargs):
        self.model_name_or_path = kwargs.get("model_name_or_path", None)
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required in config")
        self.model_name_or_path2 = kwargs.get("model_name_or_path2", None)
        self.extensions = kwargs.get("extensions", [])
        if self.extensions is None or len(self.extensions) == 0:
            raise ValueError("At least one extension is required in config")
        self.path_to_caption = kwargs.get("path_to_caption", None)
        if self.path_to_caption is None:
            raise ValueError("path_to_caption is required in config")
        self.dtype = kwargs.get("dtype", "bf16")
        self.device = kwargs.get("device", "cuda")
        self.quantize = kwargs.get("quantize", False)
        self.qtype = kwargs.get("qtype", "float8")
        self.low_vram = kwargs.get("low_vram", False)
        self.caption_extension = kwargs.get("caption_extension", "txt")
        self.recaption = kwargs.get("recaption", False)
        self.max_res = kwargs.get("max_res", 512)
        self.max_new_tokens = kwargs.get("max_new_tokens", 128)
        self.caption_prompt = kwargs.get(
            "caption_prompt", "Describe this image in detail."
        )


class BaseCaptioner(BaseExtensionProcess):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(BaseCaptioner, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = self.job_id.strip() if self.job_id is not None else None
        self.is_ui_captioner = True
        if not os.path.exists(self.sqlite_db_path):
            self.is_ui_captioner = False
        else:
            print(f"Using SQLite database at {self.sqlite_db_path}")
        if self.job_id is None:
            self.is_ui_captioner = False
        else:
            print(f'Job ID: "{self.job_id}"')

        self.is_stopping = False

        if self.is_ui_captioner:
            self.is_stopping = False
            # Create a thread pool for database operations
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # Track all async tasks
            self._async_tasks = []
            # Initialize the status
            self._run_async_operation(self._update_status("running", "Starting"))
            self._stop_watcher_started = False
            # self.start_stop_watcher(interval_sec=2.0)

        self.caption_config = CaptionConfig(**self.get_conf("caption", {}))
        self.model = None
        self.processor = None
        self.model2 = None
        self.processor2 = None
        self.file_paths = []
        self.device_torch = torch.device(self.caption_config.device)
        self.torch_dtype = get_torch_dtype(self.caption_config.dtype)

    def run(self):
        super(BaseCaptioner, self).run()
        self.start_stop_watcher()
        self.update_status("running", "Loading Model")
        self.load_model()
        self.update_status("running", "Looking for files")
        self.find_files()
        self.update_status("running", f"Captioning {len(self.file_paths)} files")
        self.run_caption_loop()
        self.update_status("completed", "Captioning completed")
        print("")

        print("****************************************************")
        print("Captioning complete")
        print("****************************************************")

    def run_caption_loop(self):
        for file_path in tqdm.tqdm(
            self.file_paths, desc="Captioning files", unit="file"
        ):
            if self.is_ui_captioner:
                self.maybe_stop()
                if self.is_stopping:
                    break
            try:
                file_caption = self.get_caption_for_file(file_path)
                if file_caption is not None:
                    self.save_caption_for_file(file_path, file_caption)
            except Exception as e:
                print(f"Error captioning file {file_path}: {e}")
                continue

    def load_pil_image(self, file_path: str, max_res: Optional[int] = None) -> Image:
        image = Image.open(file_path).convert("RGB")
        if max_res is not None:
            max_pixels = max_res * max_res
            image_pixels = image.width * image.height
            if image_pixels > max_pixels:
                scale_factor = (max_pixels / image_pixels) ** 0.5
                new_width = int(image.width * scale_factor)
                new_height = int(image.height * scale_factor)
                image = image.resize((new_width, new_height), resample=Image.BICUBIC)
        return image

    def save_caption_for_file(self, file_path: str, caption: str):
        filename_no_ext = os.path.splitext(file_path)[0]
        caption_file_path = f"{filename_no_ext}.{self.caption_config.caption_extension}"
        # delete it if it already exists
        if os.path.exists(caption_file_path):
            os.remove(caption_file_path)
        with open(caption_file_path, "w", encoding="utf-8") as f:
            f.write(caption)

    def get_caption_for_file(self, file_path: str) -> str:
        raise NotImplementedError("Captioning not implemented for this captioner")

    def print_and_status_update(self, status: str):
        print(status)
        self.update_status("running", status)

    def find_files(self):
        # recursivly find all the files in the path_to_caption with the specified extensions and save the paths to self.file_paths
        for root, dirs, files in os.walk(self.caption_config.path_to_caption):
            for file in files:
                if any(
                    file.lower().endswith(f".{ext}")
                    for ext in self.caption_config.extensions
                ):
                    full_path = os.path.join(root, file)
                    self.file_paths.append(full_path)
        # sort
        self.file_paths.sort()
        # it not recaption, remove the ones with captions
        if not self.caption_config.recaption:
            filtered_file_paths = []
            for file_path in self.file_paths:
                filename_no_ext = os.path.splitext(file_path)[0]
                caption_file_path = (
                    f"{filename_no_ext}.{self.caption_config.caption_extension}"
                )
                if not os.path.exists(caption_file_path):
                    filtered_file_paths.append(file_path)
            print(
                f"Found {len(self.file_paths)} files. {len(filtered_file_paths)} need captioning."
            )
            self.file_paths = filtered_file_paths
        else:
            print(f"Found {len(self.file_paths)} files to caption")

    def load_model(self):
        raise NotImplementedError("Model loading not implemented for this captioner")

    def start_stop_watcher(self, interval_sec: float = 5.0):
        """
        Start a daemon thread that periodically checks should_stop()
        and terminates the process immediately when triggered.
        """
        if not self.is_ui_captioner:
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
                    # Mark and update status (non-blocking; uses existing infra)
                    self.is_stopping = True
                    self._run_async_operation(
                        self._update_status("stopped", "Job stopped (remote)")
                    )
                    # Best-effort flush pending async ops
                    try:
                        asyncio.run(self.wait_for_all_async())
                    except RuntimeError:
                        pass
                    # Try to stop DB thread pool quickly
                    try:
                        self.thread_pool.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        self.thread_pool.shutdown(wait=False)
                    print("")
                    print("****************************************************")
                    print("    Stop signal received; terminating process.      ")
                    print("****************************************************")
                    os.kill(os.getpid(), signal.SIGINT)
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

        # Create a task and track it
        if loop.is_running():
            task = asyncio.run_coroutine_threadsafe(coro, loop)
            self._async_tasks.append(asyncio.wrap_future(task))
        else:
            task = loop.create_task(coro)
            self._async_tasks.append(task)
            loop.run_until_complete(task)

    async def _execute_db_operation(self, operation_func):
        """Execute a database operation in a separate thread with retry on lock."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, lambda: self._retry_db_operation(operation_func)
        )

    def _db_connect(self):
        """Create a new connection for each operation to avoid locking."""
        conn = sqlite3.connect(self.sqlite_db_path, timeout=30.0)
        conn.isolation_level = None  # Enable autocommit mode
        return conn

    def _retry_db_operation(self, operation_func, max_retries=3, base_delay=2.0):
        """Retry a database operation with exponential backoff on lock errors."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return operation_func()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt)  # 2s, 4s, 8s
                        print(
                            f"[AITK] Database locked (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"[AITK] Database locked after {max_retries + 1} attempts, giving up."
                        )
                else:
                    raise
        raise last_error

    def should_stop(self):
        if not self.is_ui_captioner:
            return False

        def _check_stop():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT stop FROM Job WHERE id = ?", (self.job_id,))
                stop = cursor.fetchone()
                return False if stop is None else stop[0] == 1

        return self._retry_db_operation(_check_stop)

    def should_return_to_queue(self):
        if not self.is_ui_captioner:
            return False

        def _check_return_to_queue():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT return_to_queue FROM Job WHERE id = ?", (self.job_id,)
                )
                return_to_queue = cursor.fetchone()
                return False if return_to_queue is None else return_to_queue[0] == 1

        return self._retry_db_operation(_check_return_to_queue)

    def maybe_stop(self):
        if not self.is_ui_captioner:
            return
        if self.should_stop():
            self._run_async_operation(self._update_status("stopped", "Job stopped"))
            self.is_stopping = True
            raise Exception("Job stopped")
        if self.should_return_to_queue():
            self._run_async_operation(self._update_status("queued", "Job queued"))
            self.is_stopping = True
            raise Exception("Job returning to queue")

    async def _update_key(self, key, value):
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
                    cursor.execute(update_query, (value_to_insert, self.job_id))
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_step(self):
        """Non-blocking update of the step count."""
        if self.is_ui_captioner:
            self._run_async_operation(self._update_key("step", self.step_num))

    def update_db_key(self, key, value):
        """Non-blocking update a key in the database."""
        if self.is_ui_captioner:
            self._run_async_operation(self._update_key(key, value))

    async def _update_status(self, status: AITK_Status, info: Optional[str] = None):
        if not self.is_ui_captioner:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    if info is not None:
                        cursor.execute(
                            "UPDATE Job SET status = ?, info = ? WHERE id = ?",
                            (status, info, self.job_id),
                        )
                    else:
                        cursor.execute(
                            "UPDATE Job SET status = ? WHERE id = ?",
                            (status, self.job_id),
                        )
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        if self.is_ui_captioner:
            """Non-blocking update of status."""
            self._run_async_operation(self._update_status(status, info))

    def on_error(self, e: Exception):
        super(BaseCaptioner, self).on_error(e)
        if self.is_ui_captioner:
            try:
                if not self.is_stopping:
                    self.update_status("error", str(e))
                asyncio.run(self.wait_for_all_async())
            except Exception as db_err:
                print(
                    f"[AITK] Warning: failed to update DB during error handling: {db_err}"
                )
            finally:
                self.thread_pool.shutdown(wait=True)

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
