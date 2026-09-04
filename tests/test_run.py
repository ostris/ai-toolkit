import sys

import run


def test_main_skips_on_error_when_job_construction_fails(monkeypatch):
    messages = []

    def fail_to_construct_job(config_file, name):
        raise RuntimeError("invalid job configuration")

    monkeypatch.setattr(sys, "argv", ["run.py", "--recover", "broken.yaml"])
    monkeypatch.setattr(run, "get_job", fail_to_construct_job)
    monkeypatch.setattr(run, "print_acc", messages.append)

    run.main()

    assert "Error running job: invalid job configuration" in messages
    assert not any("Error running on_error" in message for message in messages)
