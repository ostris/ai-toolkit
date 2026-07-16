import pytest

import toolkit.timer as timer_module
from toolkit.timer import Timer


def test_timer_print_ignores_canceled_empty_measurements(capsys):
    timer = Timer("recovered step")
    reports = []
    timer.add_after_print_hook(reports.append)

    with pytest.raises(RuntimeError, match="recoverable failure"):
        with timer("failed operation"):
            raise RuntimeError("recoverable failure")

    timer.print()

    assert reports == [{}]
    assert "failed operation" not in capsys.readouterr().out


def test_timer_print_still_reports_completed_measurements(monkeypatch, capsys):
    clock = iter((10.0, 12.5))
    monkeypatch.setattr(timer_module.time, "time", lambda: next(clock))
    timer = Timer("completed step")
    reports = []
    timer.add_after_print_hook(reports.append)

    with timer("completed operation"):
        pass

    timer.print()

    assert reports == [{"completed operation": 2.5}]
    assert "2.5000s avg - completed operation" in capsys.readouterr().out
