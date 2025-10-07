import threading
from typing import Callable, Optional


class StepBudgetController:
    """Coordinates pause/resume behaviour based on step budgets."""

    def __init__(
        self,
        total_steps: Optional[int] = None,
        on_budget_exhausted: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.total_steps = total_steps
        self.on_budget_exhausted = on_budget_exhausted

        self.allowed_steps = 0
        self.completed_steps = 0
        self._last_step: Optional[int] = None
        self._last_epoch: Optional[int] = None
        self._budget_notified_at: Optional[int] = None

        self._lock = threading.Lock()
        self.resume_event = threading.Event()
        self.abort_event = threading.Event()
        self.pause_event = threading.Event()
        self.finished_event = threading.Event()

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------
    def allow_steps(self, count: int) -> int:
        """Grant additional step budget. Returns the number of steps granted."""
        if count <= 0:
            return 0
        with self._lock:
            previous = self.allowed_steps
            new_allowed = previous + count
            if self.total_steps is not None:
                new_allowed = min(new_allowed, self.total_steps)
            granted = max(0, new_allowed - previous)
            if granted == 0:
                return 0
            self.allowed_steps = new_allowed
            self.resume_event.set()
            self.pause_event.clear()
            self._budget_notified_at = None
        return granted

    def wait_if_needed(self, current_epoch: int, current_step: int) -> bool:
        """Block until the current step is authorised. Returns True if abort requested."""
        with self._lock:
            self._last_epoch = current_epoch
        while True:
            if self.abort_event.is_set():
                return True
            with self._lock:
                allowed = self.allowed_steps
                total = self.total_steps
            if total is not None and current_step >= total:
                self.pause_event.set()
                return False
            if current_step < allowed:
                self.pause_event.clear()
                return False
            self.pause_event.set()
            self.resume_event.wait(0.1)

    def on_step_end(self, step: int, epoch: int) -> str:
        """Called after each training step. Returns 'pause', 'continue', or 'abort'."""
        callback: Optional[Callable[[int, int], None]] = None
        should_pause = False
        last_step_index = step - 1 if step > 0 else None
        with self._lock:
            if last_step_index is not None:
                self._last_step = last_step_index
            self._last_epoch = epoch
            if step >= 0:
                self.completed_steps = max(self.completed_steps, max(step, 0))
            total = self.total_steps
            allowed = self.allowed_steps
            if total is not None and self.completed_steps >= total:
                should_pause = True
            elif self.completed_steps >= allowed:
                should_pause = True
            if should_pause:
                self.pause_event.set()
                if self.on_budget_exhausted is not None and self._budget_notified_at != self.completed_steps:
                    self._budget_notified_at = self.completed_steps
                    callback = self.on_budget_exhausted
            else:
                self.pause_event.clear()
        if callback is not None:
            try:
                callback(self.completed_steps, epoch)
            except Exception:
                pass
        if self.abort_event.is_set():
            return "abort"
        return "pause" if should_pause else "continue"

    def wait_for_resume(self) -> str:
        """Wait for additional budget or abort. Returns 'resume', 'abort', or 'complete'."""
        while True:
            if self.abort_event.is_set():
                return "abort"
            with self._lock:
                total = self.total_steps
                completed = self.completed_steps
                allowed = self.allowed_steps
            if total is not None and completed >= total:
                return "complete"
            if completed < allowed:
                self.resume_event.clear()
                self.pause_event.clear()
                return "resume"
            self.resume_event.wait(0.1)

    def request_abort(self) -> None:
        self.abort_event.set()
        self.resume_event.set()
        self.pause_event.set()

    def wait_until_paused(self, timeout: Optional[float] = None) -> bool:
        return self.pause_event.wait(timeout)

    def wait_until_finished(self, timeout: Optional[float] = None) -> bool:
        return self.finished_event.wait(timeout)

    def notify_training_stopped(self, aborted: bool, step: int, epoch: int) -> None:
        with self._lock:
            if step >= 0:
                self.completed_steps = max(self.completed_steps, max(step, 0))
                self._last_step = step - 1 if step > 0 else self._last_step
            self._last_epoch = epoch
            if aborted:
                self.pause_event.set()
                self.resume_event.set()
        self.finished_event.set()

    @property
    def current_epoch(self) -> int:
        with self._lock:
            return 0 if self._last_epoch is None else self._last_epoch

    @property
    def last_step(self) -> Optional[int]:
        with self._lock:
            return self._last_step

    @property
    def remaining_budget(self) -> int:
        with self._lock:
            remaining = self.allowed_steps - self.completed_steps
            if self.total_steps is not None:
                remaining = min(remaining, self.total_steps - self.completed_steps)
            return max(0, remaining)


EpochPauseController = StepBudgetController

__all__ = ["StepBudgetController", "EpochPauseController"]
