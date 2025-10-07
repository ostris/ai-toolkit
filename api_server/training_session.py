import contextlib
import io
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Generator, List, Optional

from toolkit.job import get_job

from api_server.epoch_controller import StepBudgetController


class _LogTee(io.TextIOBase):
    def __init__(self, session: 'TrainingSession', original: io.TextIOBase) -> None:
        self._session = session
        self._original = original

    def write(self, data: str) -> int:
        self._original.write(data)
        self._session._append_log(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()


class TrainingSession:
    def __init__(
        self,
        session_id: str,
        config: Dict[str, Any],
        *,
        max_steps: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.max_steps = max_steps

        self._status_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._log_buffer = ''
        self._stopped = threading.Event()

        self.status: str = 'initializing'
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.error: Optional[str] = None

        self.log_history: Deque[str] = deque(maxlen=2000)
        self.log_queue: 'queue.Queue[Optional[str]]' = queue.Queue()

        self.job = get_job(config)
        if not getattr(self.job, 'process', None):
            raise ValueError('config did not produce any processes')
        if len(self.job.process) != 1:
            raise ValueError('training API currently supports configs with exactly one process')

        self.trainer = self.job.process[0]
        self.controller = StepBudgetController(
            total_steps=max_steps,
            on_budget_exhausted=self._on_budget_exhausted,
        )
        if hasattr(self.trainer, 'set_pause_controller'):
            self.trainer.set_pause_controller(self.controller)
        else:
            setattr(self.trainer, 'pause_controller', self.controller)

        self._last_budget_report = 0

        self._thread = threading.Thread(
            target=self._training_loop,
            name=f'training-session-{session_id}',
            daemon=True,
        )
        self._thread.start()

        # Wait for trainer to reach initial pause or finish
        self._wait_for_initial_ready()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _wait_for_initial_ready(self) -> None:
        start = time.time()
        while True:
            if self.controller.pause_event.is_set() or self.controller.finished_event.is_set():
                break
            if not self._thread.is_alive():
                break
            if time.time() - start > 300:  # 5 minutes safeguard
                break
            time.sleep(0.05)
        with self._status_lock:
            if self.controller.finished_event.is_set():
                if self.controller.abort_event.is_set():
                    self.status = 'aborted'
                else:
                    self.status = 'completed'
            elif self.controller.pause_event.is_set():
                self.status = 'paused'
            else:
                self.status = 'running'
        self.current_step = self.controller.completed_steps
        self.current_epoch = self.controller.current_epoch

    def _training_loop(self) -> None:
        final_status = 'completed'
        try:
            with self._capture_output():
                self.trainer.run()
            aborted = getattr(self.trainer, '_training_aborted', False) or self.controller.abort_event.is_set()
            final_status = 'aborted' if aborted else 'completed'
        except Exception as exc:  # pylint: disable=broad-exception-caught
            final_status = 'error'
            self.error = f'{type(exc).__name__}: {exc}'
            self._append_log(f'Exception raised in training thread: {self.error}\n')
        finally:
            try:
                self.job.cleanup()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            if not self.controller.finished_event.is_set():
                self.controller.notify_training_stopped(
                    aborted=final_status in ('aborted', 'error'),
                    step=getattr(self.trainer, 'step_num', -1),
                    epoch=getattr(self.trainer, 'epoch_num', 0),
                )
            with self._status_lock:
                self.status = final_status
                self.current_step = max(self.current_step, self.controller.completed_steps)
                self.current_epoch = self.controller.current_epoch
            self._stopped.set()
            # signal log consumers to exit
            self.log_queue.put(None)

    @contextlib.contextmanager
    def _capture_output(self) -> Generator[None, None, None]:
        stdout_original = sys.stdout
        stderr_original = sys.stderr
        tee_out = _LogTee(self, stdout_original)
        tee_err = _LogTee(self, stderr_original)
        try:
            sys.stdout = tee_out
            sys.stderr = tee_err
            yield
        finally:
            sys.stdout = stdout_original
            sys.stderr = stderr_original

    def _append_log(self, chunk: str) -> None:
        if not chunk:
            return
        chunk = chunk.replace('\r', '')
        with self._log_lock:
            self._log_buffer += chunk
            while '\n' in self._log_buffer:
                line, remainder = self._log_buffer.split('\n', 1)
                self._log_buffer = remainder
                stripped = line.rstrip()
                if stripped:
                    self.log_history.append(stripped)
                    self.log_queue.put(stripped)

    def _on_budget_exhausted(self, completed_steps: int, epoch: int) -> None:
        with self._status_lock:
            self.current_step = completed_steps
            self.current_epoch = epoch
            if self.status not in ('completed', 'aborted', 'error'):
                self.status = 'paused'
        delta = max(0, completed_steps - self._last_budget_report)
        self._last_budget_report = completed_steps
        if delta > 0:
            self.log_queue.put(
                f'[session {self.session_id}] consumed {delta} steps (total {completed_steps}) at epoch {epoch}'
            )

    def _latest_checkpoint_path(self) -> Optional[str]:
        save_root = getattr(self.trainer, 'save_root', None)
        job = getattr(self.trainer, 'job', None)
        step = getattr(self.trainer, 'last_save_step', None)
        if not save_root or job is None or not getattr(job, 'name', None):
            return None
        filename = job.name
        if step is not None and step >= 0:
            filename = f"{filename}_{step:09d}"
        filename = f"{filename}.safetensors"
        return os.path.join(save_root, filename)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def allocate_steps(self, count: int, timeout: Optional[float] = None) -> Dict[str, Any]:
        granted = 0
        consumed = False
        timed_out = False
        already_finished = self.controller.finished_event.is_set()

        if count <= 0:
            return {
                'granted': granted,
                'consumed': consumed,
                'timed_out': timed_out,
                'already_finished': already_finished,
            }

        with self._status_lock:
            if self.status in ('completed', 'aborted', 'error'):
                return {
                    'granted': granted,
                    'consumed': consumed,
                    'timed_out': timed_out,
                    'already_finished': True,
                }
            self.status = 'running'

        granted = self.controller.allow_steps(count)
        if granted == 0:
            already_finished = already_finished or self.controller.finished_event.is_set()
            with self._status_lock:
                if self.status not in ('completed', 'aborted', 'error'):
                    self.status = 'completed' if already_finished else 'paused'
            return {
                'granted': granted,
                'consumed': consumed,
                'timed_out': timed_out,
                'already_finished': already_finished,
            }

        start = time.time()
        while True:
            if self.controller.finished_event.is_set():
                consumed = True
                break
            if self.controller.wait_until_paused(timeout=0.1):
                consumed = True
                with self._status_lock:
                    if self.status not in ('completed', 'aborted', 'error'):
                        self.status = 'paused'
                break
            if timeout is not None and (time.time() - start) > timeout:
                timed_out = True
                break

        return {
            'granted': granted,
            'consumed': consumed,
            'timed_out': timed_out,
            'already_finished': self.controller.finished_event.is_set(),
        }

    def abort(self) -> None:
        self.controller.request_abort()
        with self._status_lock:
            if self.status not in ('completed', 'error'):
                self.status = 'aborting'
        self.log_queue.put(f'[session {self.session_id}] abort requested')

    def wait_until_stopped(self, timeout: Optional[float] = None) -> bool:
        return self._stopped.wait(timeout)

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def get_status(self) -> Dict[str, Any]:
        with self._status_lock:
            status = self.status
            epoch = self.current_epoch
            step = self.current_step
            error = self.error
        controller = self.controller
        return {
            'session_id': self.session_id,
            'status': status,
            'current_epoch': epoch,
            'current_step': step,
            'allowed_steps': controller.allowed_steps,
            'completed_steps': controller.completed_steps,
            'remaining_steps': controller.remaining_budget,
            'last_step_index': controller.last_step,
            'max_steps': self.max_steps,
            'finished': controller.finished_event.is_set(),
            'error': error,
            'last_checkpoint_step': getattr(self.trainer, 'last_save_step', None),
            'last_checkpoint_path': self._latest_checkpoint_path(),
        }

    def get_logs(self, limit: Optional[int] = None) -> List[str]:
        with self._log_lock:
            if limit is None:
                return list(self.log_history)
            return list(self.log_history)[-limit:]

    def log_stream(self) -> Generator[str, None, None]:
        while True:
            item = self.log_queue.get()
            if item is None:
                break
            yield item

    def dispose(self) -> None:
        self.abort()
        self.wait_until_stopped(timeout=10)
        self.join(timeout=2)
        # Ensure consumers exit
        self.log_queue.put(None)
