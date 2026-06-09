"""Tests for the launch/monitor/lifecycle side of the remote pipeline.

Run directly: python testing/test_remote_monitor.py

U5 (scripts/remote/launch.py) tests live here now; U6 (monitor.py) and U7
(lifecycle.py) tests will EXTEND this file — keep TestCase classes cleanly
separated per unit and reuse the fixture helpers at the top.

No network: every ssh/rsync invocation goes through an injected fake runner
returning scripted CompletedProcess objects (the DispatchRunner pattern from
testing/test_remote_transport.py).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.remote import contract, launch, manifest, pod
from scripts.remote.manifest import RunManifest
from scripts.remote.transport import Endpoint

# ---------------------------------------------------------------------------
# Shared fixture helpers (reused by U6/U7 tests later)
# ---------------------------------------------------------------------------


def proc(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


class DispatchRunner:
    """Fake runner: first handler whose needle appears in the joined argv wins."""

    def __init__(self, handlers, default=None):
        self.handlers = handlers      # list of (needle, CompletedProcess | callable)
        self.default = default if default is not None else proc("")
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        joined = " ".join(str(c) for c in cmd)
        for needle, resp in self.handlers:
            if needle in joined:
                return resp(cmd) if callable(resp) else resp
        return self.default

    def commands(self, needle):
        return [c for c in self.calls if needle in " ".join(str(x) for x in c)]

    def first_index(self, needle):
        for i, c in enumerate(self.calls):
            if needle in " ".join(str(x) for x in c):
                return i
        return None


EP = Endpoint(host="1.2.3.4", port=10022)
NOW = 1768500000.0
RUN = "balfua_v3"
JOB_DIR = contract.remote_job_dir(RUN)

# Log-tail fixtures (records separated the way the teed log really is: tqdm
# writes \r-separated updates).
TRACEBACK_TAIL = (
    "Loading checkpoint shards\n"
    "Traceback (most recent call last):\n"
    '  File "/app/ai-toolkit/run.py", line 90, in <module>\n'
    "    main()\n"
    "ValueError: boom\n"
)
PROGRESS_TAIL = (
    "balfua_v3:   2%|2         | 86/4000 [21:26<12:10:03, 13.97s/it, lr: 1.0e-04 loss: 4.012e-01]\r"
    "balfua_v3:   2%|2         | 87/4000 [21:40<12:09:00, 13.95s/it, lr: 1.0e-04 loss: 3.998e-01]"
)
WARMING_TAIL = (
    "Fetching 23 files\n"
    "Loading pipeline components ...\n"
    "Loading checkpoint shards\n"
)
SILENT_TAIL = "container started\n"


def _touch_commands(script):
    """Non-comment lines that invoke `touch` (sentinels must carry content)."""
    return [l for l in script.splitlines()
            if not l.lstrip().startswith("#")
            and (l.lstrip().startswith("touch ") or "&& touch " in l or "; touch " in l)]


def make_manifest(**kw):
    defaults = dict(run_name=RUN, job_name=RUN, prompt_count=12,
                    state=contract.RunState.SYNCED.value)
    defaults.update(kw)
    return RunManifest(**defaults)


def launch_handlers(tail=PROGRESS_TAIL, has_session_rc=1, pgrep_stdout="",
                    pgrep_rc=1, ckpt_stdout="", ckpt_rc=1, extra=None):
    """Scripted responses for a launch_run cycle (order: specific first)."""
    handlers = [
        ("tmux has-session", proc("", returncode=has_session_rc)),
        ("pgrep", proc(pgrep_stdout, returncode=pgrep_rc)),
        (".safetensors", proc(ckpt_stdout, returncode=ckpt_rc)),
        ("mv ", proc("")),
        ("mkdir -p", proc("")),
        ("rsync", proc("")),
        ("tmux new-session", proc("")),
        ("tmux set-option", proc("")),
        ("tail -c", proc(tail)),
    ]
    return (extra or []) + handlers


class LaunchBase(unittest.TestCase):
    """Temp base dir with a saved manifest; launch helper with frozen time."""

    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-launch-test-")
        manifest.save(make_manifest(), self.base)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def launch(self, runner, **kw):
        kw.setdefault("ep", EP)
        kw.setdefault("base_dir", self.base)
        kw.setdefault("now", NOW)
        kw.setdefault("tail_timeout_s", 0)
        kw.setdefault("tail_poll_s", 1)
        kw.setdefault("_sleep", lambda s: None)
        return launch.launch_run(RUN, runner=runner, **kw)


# ---------------------------------------------------------------------------
# U5: pre-launch guards (R10)
# ---------------------------------------------------------------------------


class TestLaunchGuards(LaunchBase):
    def test_existing_tmux_session_refused_mentioning_status_attach(self):
        runner = DispatchRunner(launch_handlers(has_session_rc=0))
        with self.assertRaises(launch.LaunchRefusedError) as ctx:
            self.launch(runner)
        msg = str(ctx.exception)
        self.assertIn("status", msg)
        self.assertIn("attach", msg)
        self.assertEqual(runner.commands("tmux new-session"), [])

    def test_running_trainer_process_refused(self):
        runner = DispatchRunner(launch_handlers(pgrep_stdout="4242\n", pgrep_rc=0))
        with self.assertRaises(launch.LaunchRefusedError) as ctx:
            self.launch(runner)
        msg = str(ctx.exception)
        self.assertIn("status", msg)
        self.assertIn("attach", msg)
        self.assertEqual(runner.commands("tmux new-session"), [])

    def test_remote_checkpoints_without_flag_refused_naming_both_flags(self):
        listing = f"{JOB_DIR}/{RUN}_000001000.safetensors\n{JOB_DIR}/{RUN}_000001250.safetensors\n"
        runner = DispatchRunner(launch_handlers(ckpt_stdout=listing, ckpt_rc=0))
        with self.assertRaises(launch.LaunchRefusedError) as ctx:
            self.launch(runner)
        msg = str(ctx.exception)
        self.assertIn("--resume", msg)
        self.assertIn("--fresh", msg)
        self.assertEqual(runner.commands("tmux new-session"), [])
        self.assertEqual(runner.commands("mv "), [])

    def test_resume_proceeds_leaving_checkpoints(self):
        listing = f"{JOB_DIR}/{RUN}_000001000.safetensors\n"
        runner = DispatchRunner(launch_handlers(ckpt_stdout=listing, ckpt_rc=0))
        result = self.launch(runner, resume=True)
        self.assertEqual(result.outcome, "running")
        self.assertEqual(runner.commands("mv "), [])  # checkpoints untouched
        self.assertTrue(runner.commands("tmux new-session"))

    def test_fresh_moves_output_dir_aside_before_launch(self):
        listing = f"{JOB_DIR}/{RUN}_000001000.safetensors\n"
        runner = DispatchRunner(launch_handlers(ckpt_stdout=listing, ckpt_rc=0))
        self.launch(runner, fresh=True)
        mv_calls = runner.commands("mv ")
        self.assertEqual(len(mv_calls), 1)
        mv_joined = " ".join(mv_calls[0])
        self.assertIn(JOB_DIR, mv_joined)
        self.assertIn(f".old.{int(NOW)}", mv_joined)
        self.assertLess(runner.first_index("mv "),
                        runner.first_index("tmux new-session"))

    def test_resume_and_fresh_mutually_exclusive(self):
        runner = DispatchRunner(launch_handlers())
        with self.assertRaises(launch.LaunchError):
            self.launch(runner, resume=True, fresh=True)


# ---------------------------------------------------------------------------
# U5: wrapper script generation (R8/R28)
# ---------------------------------------------------------------------------


class TestWrapperScript(unittest.TestCase):
    def test_body_order_env_run_exitcode_ack_selfstop(self):
        m = make_manifest()
        script = launch.build_wrapper_script(m, max_grace_seconds=1800)
        exit_path = contract.remote_sentinel_path(RUN, contract.EXIT_CODE_FILE)
        exit_write = f'echo "$code" > {contract.shell_quote(exit_path)}'

        idx_env = script.index("/proc/1/environ")
        idx_run = script.index("python run.py")
        idx_exit = script.index(exit_write)
        idx_ack = script.index(contract.PULLED_OK_FILE)
        idx_stop = script.index("rest.runpod.io")
        self.assertLess(idx_env, idx_run)
        self.assertLess(idx_run, idx_exit)
        self.assertLess(idx_exit, idx_ack)
        self.assertLess(idx_ack, idx_stop)

        # the PID-1 bridge covers all three HF vars, before the trainer runs
        for var in ("HF_TOKEN", "HF_HOME", "HF_HUB_ENABLE_HF_TRANSFER"):
            self.assertLess(script.index(var), idx_run)
        # exit code written WITH CONTENT (echo redirect), never a bare touch
        self.assertIn(exit_write, script)
        self.assertFalse(_touch_commands(script),
                         "wrapper must never touch a sentinel (invisible to transport)")
        # ack wait bounded by the max-grace deadline computed from SECONDS
        self.assertIn("SECONDS + 1800", script)
        self.assertIn(f"sleep {launch.ACK_POLL_SECONDS}", script)
        # the self-stop content is pod.py's, embedded verbatim
        self.assertIn("RUNPOD_STOP_KEY", script)

    def test_max_grace_flows_into_deadline(self):
        script = launch.build_wrapper_script(make_manifest(), max_grace_seconds=900)
        self.assertIn("SECONDS + 900", script)
        self.assertNotIn("SECONDS + 1800", script)

    def test_no_provider_strings_in_launch_source(self):
        with open(launch.__file__, "r") as f:
            src = f.read()
        for marker in ("rest.runpod.io", "runpod.io/graphql", "api.runpod.io"):
            self.assertNotIn(marker, src,
                             f"provider string {marker!r} originates in launch.py")
        # ...while the generated ARTIFACT does carry them, received from pod.py
        script = launch.build_wrapper_script(make_manifest(), 1800)
        self.assertIn("rest.runpod.io", script)
        self.assertIn("rest.runpod.io", pod.self_stop_script())

    def test_user_values_appear_only_inside_single_quoted_literals(self):
        # A run name with a space puts a space into every path. launch_run
        # validates names long before this (R28); the quoting layer must hold
        # regardless.
        spaced = "has space"
        m = RunManifest(run_name=spaced, job_name=spaced)
        script = launch.build_wrapper_script(m, 1800)
        paths = [
            contract.remote_config_path(spaced),
            contract.remote_log_path(spaced),
            contract.remote_sentinel_path(spaced, contract.EXIT_CODE_FILE),
            contract.remote_sentinel_path(spaced, contract.PULLED_OK_FILE),
        ]
        for path in paths:
            quoted = contract.shell_quote(path)
            self.assertGreater(script.count(quoted), 0, f"{path} missing")
            self.assertEqual(script.count(path), script.count(quoted),
                             f"{path} appears outside a single-quoted literal")


# ---------------------------------------------------------------------------
# U5: timer script + manifest deadline (R8)
# ---------------------------------------------------------------------------


class TestTimerScript(LaunchBase):
    def test_targets_timer_session_with_two_phase_order(self):
        m = make_manifest()
        script = launch.build_timer_script(m, max_hours=2.0)
        self.assertIn(contract.tmux_timer_session(RUN), script)
        timed_out_path = contract.remote_sentinel_path(RUN, contract.TIMED_OUT_FILE)
        timed_out_write = f"date +%s > {contract.shell_quote(timed_out_path)}"

        idx_sleep = script.index(f"sleep {int(2.0 * 3600)}")
        idx_pkill = script.index("pkill -f")
        idx_timed_out = script.index(timed_out_write)
        idx_stop = script.index("rest.runpod.io")
        self.assertLess(idx_sleep, idx_pkill)
        self.assertLess(idx_pkill, idx_timed_out)
        self.assertLess(idx_timed_out, idx_stop)
        # pkill targets THIS run's trainer; the sentinel is written WITH CONTENT
        pkill_line = [l for l in script.splitlines() if l.startswith("pkill")][0]
        self.assertIn(RUN, pkill_line)
        self.assertIn(timed_out_write, script)
        self.assertFalse(_touch_commands(script),
                         "timer must never touch a sentinel (invisible to transport)")

    def test_max_hours_flows_into_sleep(self):
        script = launch.build_timer_script(make_manifest(), max_hours=0.5)
        self.assertIn("sleep 1800", script)

    def test_launch_records_absolute_deadline_and_sessions(self):
        runner = DispatchRunner(launch_handlers())
        self.launch(runner, max_hours=2.0, max_grace_seconds=600)
        m = manifest.load(RUN, self.base)
        self.assertEqual(m.launched_at, NOW)
        self.assertEqual(m.max_runtime_deadline, NOW + 2.0 * 3600)
        self.assertEqual(m.max_grace_seconds, 600)
        self.assertEqual(m.tmux_session, contract.tmux_train_session(RUN))
        self.assertEqual(m.timer_session, contract.tmux_timer_session(RUN))
        self.assertEqual(m.state, contract.RunState.RUNNING.value)

    def test_timer_spawns_in_its_own_tmux_session(self):
        runner = DispatchRunner(launch_handlers())
        self.launch(runner)
        spawns = runner.commands("tmux new-session")
        self.assertEqual(len(spawns), 2)
        joined = [" ".join(c) for c in spawns]
        train = [j for j in joined if contract.tmux_train_session(RUN) in j]
        timer = [j for j in joined if contract.tmux_timer_session(RUN) in j]
        self.assertEqual(len(train), 1)
        self.assertEqual(len(timer), 1)
        # each session executes its uploaded FILE — never an inline command
        self.assertIn(contract.WRAPPER_SCRIPT, train[0])
        self.assertIn(contract.TIMER_SCRIPT, timer[0])
        for j in joined:
            self.assertNotIn("python run.py", j)
        # both scripts were uploaded (rsync) before the sessions spawned
        self.assertLess(runner.first_index("rsync"),
                        runner.first_index("tmux new-session"))
        # remain-on-exit set on the training session
        remain = [" ".join(c) for c in runner.commands("remain-on-exit on")]
        self.assertTrue(any(contract.tmux_train_session(RUN) in j for j in remain))


# ---------------------------------------------------------------------------
# U5: early tail (R11)
# ---------------------------------------------------------------------------


class TestEarlyTail(unittest.TestCase):
    def _tail(self, tail_proc, *, timeout_s=900, poll_s=15):
        sleeps = []
        runner = DispatchRunner([("tail -c", tail_proc)])
        outcome, detail = launch.early_tail(
            EP, make_manifest(), timeout_s=timeout_s, poll_s=poll_s,
            runner=runner, _sleep=sleeps.append)
        return outcome, detail, sleeps

    def test_traceback_returns_crashed_with_traceback_text(self):
        outcome, detail, _ = self._tail(proc(TRACEBACK_TAIL))
        self.assertEqual(outcome, "crashed")
        self.assertTrue(detail.startswith("Traceback (most recent call last)"))
        self.assertIn("ValueError: boom", detail)

    def test_warming_returns_promptly_without_waiting_for_step_one(self):
        outcome, detail, sleeps = self._tail(proc(WARMING_TAIL))
        self.assertEqual(outcome, "warming")
        self.assertIn("Loading", detail)
        self.assertEqual(sleeps, [])  # prompt return: no polling sleeps at all

    def test_tqdm_progress_returns_running(self):
        # \r-separated records, as the teed log really is
        outcome, detail, sleeps = self._tail(proc(PROGRESS_TAIL))
        self.assertEqual(outcome, "running")
        self.assertEqual(sleeps, [])

    def test_silent_past_timeout_reports_raw_tail(self):
        outcome, detail, sleeps = self._tail(proc(SILENT_TAIL),
                                             timeout_s=60, poll_s=15)
        self.assertEqual(outcome, "silent")
        self.assertEqual(detail, SILENT_TAIL)  # raw tail, verbatim
        self.assertEqual(sleeps, [15, 15, 15, 15])

    def test_empty_log_past_timeout_is_silent(self):
        outcome, detail, sleeps = self._tail(proc("", returncode=1),
                                             timeout_s=30, poll_s=15)
        self.assertEqual(outcome, "silent")
        self.assertEqual(detail, "")
        self.assertEqual(len(sleeps), 2)


class TestLaunchOutcomes(LaunchBase):
    def test_traceback_marks_manifest_crashed_and_carries_text(self):
        runner = DispatchRunner(launch_handlers(tail=TRACEBACK_TAIL))
        result = self.launch(runner)
        self.assertEqual(result.outcome, "crashed")
        self.assertIn("ValueError: boom", result.detail)
        m = manifest.load(RUN, self.base)
        self.assertEqual(m.state, contract.RunState.CRASHED.value)
        # the pod is left up for diagnosis: launch never stops/terminates
        self.assertEqual(runner.commands("stop"), [])

    def test_progress_keeps_manifest_running(self):
        runner = DispatchRunner(launch_handlers(tail=PROGRESS_TAIL))
        result = self.launch(runner)
        self.assertEqual(result.outcome, "running")
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.RUNNING.value)

    def test_warming_keeps_manifest_running(self):
        runner = DispatchRunner(launch_handlers(tail=WARMING_TAIL))
        result = self.launch(runner)
        self.assertEqual(result.outcome, "warming")
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.RUNNING.value)

    def test_silence_leaves_running_state_with_raw_tail(self):
        runner = DispatchRunner(launch_handlers(tail=SILENT_TAIL))
        result = self.launch(runner)
        self.assertEqual(result.outcome, "silent")
        self.assertEqual(result.detail, SILENT_TAIL)
        # silence is NOT a crash: state stays RUNNING, pod left up
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.RUNNING.value)


# ---------------------------------------------------------------------------
# U5: generated scripts must be valid bash
# ---------------------------------------------------------------------------


class TestScriptSyntax(unittest.TestCase):
    def _bash_n(self, content):
        fd, path = tempfile.mkstemp(suffix=".sh")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            res = subprocess.run(["bash", "-n", path],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0,
                             f"bash -n failed:\n{res.stderr}\n--- script ---\n{content}")
        finally:
            os.unlink(path)

    def test_wrapper_script_parses(self):
        self._bash_n(launch.build_wrapper_script(make_manifest(), 1800))

    def test_timer_script_parses(self):
        self._bash_n(launch.build_timer_script(make_manifest(), 24.0))

    def test_scripts_parse_with_hostile_run_name(self):
        m = RunManifest(run_name="has space", job_name="has space")
        self._bash_n(launch.build_wrapper_script(m, 1800))
        self._bash_n(launch.build_timer_script(m, 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
