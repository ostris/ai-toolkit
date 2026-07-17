import pytest

from jobs.BaseJob import BaseJob


class _CleanupProcess:
    def __init__(self, name, calls, error=None):
        self.name = name
        self.calls = calls
        self.error = error
        self.job = object()

    def cleanup(self):
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error


def test_base_job_cleanup_releases_every_process_after_failure():
    calls = []
    job = BaseJob.__new__(BaseJob)
    job.process = [
        _CleanupProcess("first", calls),
        _CleanupProcess("second", calls, RuntimeError("boom")),
    ]
    processes = list(job.process)

    with pytest.raises(RuntimeError, match="_CleanupProcess: boom"):
        job.cleanup()

    assert calls == ["second", "first"]
    assert job.process == []
    assert all(process.job is None for process in processes)


def test_base_job_cleanup_is_safe_before_processes_are_loaded():
    job = BaseJob.__new__(BaseJob)

    job.cleanup()

    assert job.process == []
