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

from scripts.remote import contract, launch, lifecycle, manifest, monitor, pod
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


# ---------------------------------------------------------------------------
# U6 fixtures: mock RunPod SDK + scripted ssh responses for a status cycle
# ---------------------------------------------------------------------------


RAW_RUNNING = {
    "id": "pod-1", "desiredStatus": "RUNNING", "costPerHr": 1.89,
    "machine": {"gpuDisplayName": "NVIDIA A100 80GB PCIe"},
    "runtime": {"ports": [{"privatePort": 22, "isIpPublic": True,
                           "ip": EP.host, "publicPort": EP.port}]},
}
RAW_EXITED = {"id": "pod-1", "desiredStatus": "EXITED"}


class FakeSdk:
    """get_pod returns the scripted raws in order, holding on the last one."""

    def __init__(self, *raws):
        self.raws = list(raws)

    def get_pod(self, pod_id):
        return self.raws.pop(0) if len(self.raws) > 1 else self.raws[0]


def activity_output(log_age=30, sample_age=None, now=NOW):
    log = f"{now - log_age:.0f}" if log_age is not None else ""
    sample = f"{now - sample_age:.4f}" if sample_age is not None else ""
    return f"NOW={now:.0f}\nLOG={log}\nSAMPLE={sample}\n"


def monitor_handlers(tail=PROGRESS_TAIL, log_age=30, sample_age=None,
                     timed_out=None, exit_code=None, tmux_rc=0,
                     progress="865,0.4012", df_pct=42, extra=None):
    """Scripted ssh/rsync responses for a full RUNNING-pod status cycle.

    Order matters: specific needles first ('stat -c %Y' before 'date +%s'
    because the activity probe contains both).
    """
    df_line = f"/dev/vda1 104857600 47185920 57671680 {df_pct}% /workspace"
    handlers = [
        ("stat -c %Y", proc(activity_output(log_age, sample_age))),
        ("df -P /workspace", proc(df_line)),
        ("python3 -c", proc(progress)),
        ("timed_out", proc(timed_out or "", returncode=0 if timed_out else 1)),
        ("exit_code", proc(exit_code or "", returncode=0 if exit_code else 1)),
        ("tmux has-session", proc("", returncode=tmux_rc)),
        ("tail -c", proc(tail)),
        ("date +%s", proc(f"{NOW:.0f}")),
        ("-printf '%P", proc("")),  # empty remote artifact inventory
        ("rsync", proc("")),
    ]
    return (extra or []) + handlers


class MonitorBase(unittest.TestCase):
    """Temp base dir with a RUNNING manifest wired to pod-1."""

    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-monitor-test-")
        self.save_manifest()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def save_manifest(self, **kw):
        kw.setdefault("pod_id", "pod-1")
        kw.setdefault("state", contract.RunState.RUNNING.value)
        kw.setdefault("total_steps", 4000)
        manifest.save(make_manifest(**kw), self.base)

    def write_mirror(self, sentinel, content):
        run_dir = contract.local_run_dir(RUN, self.base)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, sentinel), "w") as f:
            f.write(content)

    def status(self, runner=None, sdk=None):
        return monitor.get_status(
            RUN, base_dir=self.base,
            sdk=sdk if sdk is not None else FakeSdk(RAW_RUNNING),
            runner=runner if runner is not None else DispatchRunner(monitor_handlers()))


# ---------------------------------------------------------------------------
# U6: state precedence (R14)
# ---------------------------------------------------------------------------


class TestStatePrecedence(MonitorBase):
    def test_gone_pod_is_pod_lost_despite_stale_local_sentinel(self):
        self.write_mirror(contract.EXIT_CODE_FILE, "0\n")
        runner = DispatchRunner([])
        report = self.status(runner=runner, sdk=FakeSdk(None))
        self.assertEqual(report.state, contract.RunState.POD_LOST.value)
        self.assertEqual(runner.calls, [])  # no ssh on a gone pod

    def test_exited_with_mirrored_zero_is_completed_with_zero_ssh_calls(self):
        self.write_mirror(contract.EXIT_CODE_FILE, "0\n")
        runner = DispatchRunner([])
        report = self.status(runner=runner, sdk=FakeSdk(RAW_EXITED))
        self.assertEqual(report.state, contract.RunState.COMPLETED.value)
        self.assertEqual(runner.calls, [])

    def test_exited_with_mirrored_nonzero_is_crashed_reporting_the_code(self):
        self.write_mirror(contract.EXIT_CODE_FILE, "137\n")
        runner = DispatchRunner([])
        report = self.status(runner=runner, sdk=FakeSdk(RAW_EXITED))
        self.assertEqual(report.state, contract.RunState.CRASHED.value)
        self.assertIn("137", report.detail)
        self.assertEqual(runner.calls, [])

    def test_exited_nonzero_after_stop_honors_manifest_stopped(self):
        self.save_manifest(state=contract.RunState.STOPPED.value)
        self.write_mirror(contract.EXIT_CODE_FILE, "143\n")
        report = self.status(runner=DispatchRunner([]), sdk=FakeSdk(RAW_EXITED))
        self.assertEqual(report.state, contract.RunState.STOPPED.value)

    def test_exited_with_mirrored_timed_out_wins_over_exit_code(self):
        self.write_mirror(contract.TIMED_OUT_FILE, "1768500000\n")
        self.write_mirror(contract.EXIT_CODE_FILE, "143\n")
        report = self.status(runner=DispatchRunner([]), sdk=FakeSdk(RAW_EXITED))
        self.assertEqual(report.state, contract.RunState.TIMED_OUT.value)

    def test_exited_with_no_sentinel_resolves_timed_out_conservatively(self):
        report = self.status(runner=DispatchRunner([]), sdk=FakeSdk(RAW_EXITED))
        self.assertEqual(report.state, contract.RunState.TIMED_OUT.value)

    def test_running_remote_exit_zero_is_completed(self):
        runner = DispatchRunner(monitor_handlers(exit_code="0\n"))
        report = self.status(runner=runner)
        self.assertEqual(report.state, contract.RunState.COMPLETED.value)

    def test_running_remote_exit_nonzero_is_crashed_with_code(self):
        runner = DispatchRunner(monitor_handlers(exit_code="1\n"))
        report = self.status(runner=runner)
        self.assertEqual(report.state, contract.RunState.CRASHED.value)
        self.assertIn("1", report.detail)

    def test_running_remote_timed_out_sentinel_wins(self):
        runner = DispatchRunner(monitor_handlers(timed_out="1768500000\n",
                                                 exit_code="143\n"))
        report = self.status(runner=runner)
        self.assertEqual(report.state, contract.RunState.TIMED_OUT.value)

    def test_running_tmux_gone_no_sentinel_is_unknown(self):
        runner = DispatchRunner(monitor_handlers(tmux_rc=1))
        report = self.status(runner=runner)
        self.assertEqual(report.state, contract.RunState.UNKNOWN.value)


# ---------------------------------------------------------------------------
# U6: running substates + health (R13/R14/R19/R26)
# ---------------------------------------------------------------------------


OOM_TAIL = (
    "balfua_v3:  10%|#         | 400/4000 [1:00:00<9:00:00, 9.00s/it]\r"
    "# OOM during training step, skipping batch\n"
    "balfua_v3:  11%|#1        | 440/4000 [1:06:00<8:54:00, 9.00s/it]\r"
    "# OOM during training step, skipping batch\n"
    "balfua_v3:  12%|#2        | 480/4000 [1:12:00<8:48:00, 9.00s/it]\r"
    "# OOM during training step, skipping batch\n"
)


class TestRunningSubstates(MonitorBase):
    def test_active_progress_is_running_with_step_and_loss(self):
        report = self.status()
        self.assertEqual(report.state, contract.RunState.RUNNING.value)
        self.assertEqual(report.step, 865)
        self.assertAlmostEqual(report.recent_loss, 0.4012)
        self.assertEqual(report.oom_skips, 0)
        self.assertEqual(report.total_steps, 4000)

    def test_three_scattered_oom_lines_is_degraded_count_three(self):
        runner = DispatchRunner(monitor_handlers(tail=OOM_TAIL))
        report = self.status(runner=runner)
        self.assertEqual(report.state, contract.RunState.DEGRADED.value)
        self.assertEqual(report.oom_skips, 3)
        self.assertIn("3", report.detail)

    def test_no_advance_15min_with_fresh_samples_is_sampling(self):
        runner = DispatchRunner(monitor_handlers(log_age=900, sample_age=60))
        report = self.status(runner=runner)
        self.assertEqual(report.state, contract.RunState.SAMPLING.value)

    def test_no_advance_25min_with_no_new_files_flags_stalled(self):
        runner = DispatchRunner(monitor_handlers(log_age=1500, sample_age=None))
        report = self.status(runner=runner)
        self.assertEqual(report.state, contract.RunState.RUNNING.value)
        self.assertIn("stalled", report.detail.lower())

    def test_missing_loss_db_tolerated_as_warming(self):
        runner = DispatchRunner(monitor_handlers(progress=""))
        report = self.status(runner=runner)
        self.assertEqual(report.state, contract.RunState.RUNNING.value)
        self.assertIsNone(report.step)
        self.assertIn("warming", report.detail.lower())

    def test_disk_91_percent_warns_naming_workspace(self):
        runner = DispatchRunner(monitor_handlers(df_pct=91))
        report = self.status(runner=runner)
        self.assertEqual(report.disk_used_pct, 91)
        self.assertTrue(report.disk_warning)
        self.assertIn("/workspace", report.detail)

    def test_disk_below_threshold_no_warning(self):
        report = self.status()
        self.assertEqual(report.disk_used_pct, 42)
        self.assertFalse(report.disk_warning)

    def test_drift_true_when_derived_config_tampered(self):
        import hashlib
        run_dir = contract.local_run_dir(RUN, self.base)
        os.makedirs(run_dir, exist_ok=True)
        cfg = os.path.join(run_dir, "remote_config.yaml")
        with open(cfg, "w") as f:
            f.write("job: extension\n")
        with open(cfg, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.save_manifest(config_hash=digest)
        self.assertFalse(self.status(runner=DispatchRunner(monitor_handlers())).drift)
        with open(cfg, "a") as f:
            f.write("steps: 9999\n")  # tamper after launch
        report = self.status(runner=DispatchRunner(monitor_handlers()))
        self.assertTrue(report.drift)
        self.assertIn("drift", report.detail.lower())

    def test_progress_command_is_readonly_stdlib_sqlite_not_a_cli(self):
        cmd = monitor.build_progress_command(make_manifest())
        self.assertIn("mode=ro", cmd)
        self.assertIn("python3", cmd)
        self.assertNotIn("sqlite3 ", cmd)  # never the sqlite3 CLI...
        self.assertFalse(cmd.lstrip().startswith("sqlite3"))
        self.assertIn("import sqlite3", cmd)  # ...always the stdlib module
        runner = DispatchRunner(monitor_handlers())
        self.status(runner=runner)
        joined = [" ".join(c) for c in runner.commands("python3 -c")]
        self.assertEqual(len(joined), 1)
        self.assertIn("mode=ro", joined[0])


# ---------------------------------------------------------------------------
# U6: reviewability (R16)
# ---------------------------------------------------------------------------


class TestReviewableSteps(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-review-test-")
        self.samples = os.path.join(contract.local_output_dir(RUN, self.base),
                                    "samples")
        os.makedirs(self.samples)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def write_samples(self, step, count, age_s, now=NOW):
        for i in range(count):
            path = os.path.join(self.samples,
                                f"1768500{i:03d}__{step:09d}_{i}.jpg")
            with open(path, "w") as f:
                f.write("x")
            os.utime(path, (now - age_s, now - age_s))

    def steps(self, **kw):
        m = make_manifest(**kw)
        return monitor.reviewable_steps(m, base_dir=self.base, now=NOW)

    def test_full_batch_is_reviewable_even_when_fresh(self):
        self.write_samples(500, 12, age_s=5)  # 12/12 matches prompt_count
        self.assertEqual(self.steps(), [500])

    def test_fresh_partial_batch_is_not_reviewable(self):
        self.write_samples(750, 4, age_s=30)  # 4/12, newest 30s old
        self.assertEqual(self.steps(), [])

    def test_old_partial_batch_is_reviewable_with_warning(self):
        self.write_samples(750, 4, age_s=600)  # 4/12 but settled
        self.assertEqual(self.steps(), [750])

    def test_steps_at_or_below_last_reviewed_are_excluded(self):
        self.write_samples(500, 12, age_s=600)
        self.write_samples(750, 12, age_s=600)
        self.assertEqual(self.steps(last_reviewed_step=500), [750])

    def test_no_prompt_count_falls_back_to_stability_only(self):
        self.write_samples(500, 3, age_s=600)
        self.write_samples(750, 3, age_s=10)
        self.assertEqual(self.steps(prompt_count=None), [500])


# ---------------------------------------------------------------------------
# U6: watch loop + exit codes (R17) and attach
# ---------------------------------------------------------------------------


import contextlib
import io
import json as _json


class TestWatchAndAttach(MonitorBase):
    def watch(self, runner=None, sdk=None, **kw):
        kw.setdefault("base_dir", self.base)
        kw.setdefault("sdk", sdk if sdk is not None else FakeSdk(RAW_RUNNING))
        kw.setdefault("runner",
                      runner if runner is not None else DispatchRunner(monitor_handlers()))
        kw.setdefault("_sleep", lambda s: None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = monitor.watch(RUN, **kw)
        return code, out.getvalue()

    def write_local_samples(self, step, count):
        samples = os.path.join(contract.local_output_dir(RUN, self.base),
                               "samples")
        os.makedirs(samples, exist_ok=True)
        for i in range(count):
            with open(os.path.join(samples,
                                   f"1768500{i:03d}__{step:09d}_{i}.jpg"), "w") as f:
                f.write("x")

    def test_terminal_states_produce_distinct_documented_exit_codes(self):
        cases = [
            ("0\n", None, contract.EXIT_COMPLETED),
            ("1\n", None, contract.EXIT_CRASHED),
            ("143\n", contract.RunState.STOPPED.value, contract.EXIT_STOPPED),
            (None, None, contract.EXIT_TIMED_OUT),  # no sentinel: conservative
        ]
        for exit_code, state, expected in cases:
            with self.subTest(exit_code=exit_code, state=state):
                self.setUp()
                if state:
                    self.save_manifest(state=state)
                if exit_code is not None:
                    self.write_mirror(contract.EXIT_CODE_FILE, exit_code)
                code, _ = self.watch(runner=DispatchRunner([]),
                                     sdk=FakeSdk(RAW_EXITED), once=True)
                self.assertEqual(code, expected)
                self.tearDown()

    def test_pod_lost_and_unknown_have_their_own_exit_codes(self):
        code, _ = self.watch(runner=DispatchRunner([]), sdk=FakeSdk(None),
                             once=True)
        self.assertEqual(code, contract.EXIT_POD_LOST)
        code, _ = self.watch(runner=DispatchRunner(monitor_handlers(tmux_rc=1)),
                             once=True)
        self.assertEqual(code, contract.EXIT_UNKNOWN)

    def test_once_still_running_nothing_new_exits_running(self):
        code, _ = self.watch(once=True)
        self.assertEqual(code, contract.EXIT_RUNNING)

    def test_once_json_with_new_reviewable_steps_is_valid_json_exit_10(self):
        self.write_local_samples(500, 12)  # full batch → reviewable
        code, out = self.watch(once=True, json_out=True)
        self.assertEqual(code, contract.EXIT_NEW_SAMPLES)
        data = _json.loads(out)  # ONE valid json object, nothing else
        self.assertEqual(data["run"], RUN)
        self.assertEqual(data["state"], contract.RunState.RUNNING.value)
        self.assertEqual(data["step"], 865)
        self.assertEqual(data["total_steps"], 4000)
        self.assertAlmostEqual(data["loss"], 0.4012)
        self.assertEqual(data["reviewable_steps"], [500])
        self.assertEqual(data["last_reviewed_step"], 0)
        self.assertEqual(data["exit_code"], contract.EXIT_NEW_SAMPLES)
        self.assertEqual(data["oom_skips"], 0)
        self.assertEqual(data["disk_used_pct"], 42)
        self.assertFalse(data["drift"])

    def test_once_json_terminal_reports_terminal_exit_code(self):
        self.write_mirror(contract.EXIT_CODE_FILE, "0\n")
        code, out = self.watch(runner=DispatchRunner([]),
                               sdk=FakeSdk(RAW_EXITED), once=True,
                               json_out=True)
        self.assertEqual(code, contract.EXIT_COMPLETED)
        data = _json.loads(out)
        self.assertEqual(data["state"], contract.RunState.COMPLETED.value)
        self.assertEqual(data["exit_code"], contract.EXIT_COMPLETED)

    def test_watch_loop_prints_transition_and_exits_on_terminal(self):
        self.write_mirror(contract.EXIT_CODE_FILE, "0\n")  # used on cycle 2
        sleeps = []
        code, out = self.watch(sdk=FakeSdk(RAW_RUNNING, RAW_EXITED),
                               interval_s=600, _sleep=sleeps.append)
        self.assertEqual(code, contract.EXIT_COMPLETED)
        self.assertEqual(sleeps, [600])  # one full cycle, then terminal
        self.assertIn("-> RUNNING", out)
        self.assertIn("RUNNING -> COMPLETED", out)

    def test_watch_announces_newly_reviewable_steps_once(self):
        self.write_local_samples(500, 12)
        self.write_mirror(contract.EXIT_CODE_FILE, "0\n")
        code, out = self.watch(sdk=FakeSdk(RAW_RUNNING, RAW_EXITED))
        self.assertEqual(out.count("new reviewable sample steps: [500]"), 1)

    def test_watch_persists_resolved_state_to_manifest(self):
        self.write_mirror(contract.EXIT_CODE_FILE, "1\n")
        code, _ = self.watch(runner=DispatchRunner([]),
                             sdk=FakeSdk(RAW_EXITED), once=True)
        self.assertEqual(code, contract.EXIT_CRASHED)
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.CRASHED.value)

    def test_attach_reports_same_state_as_a_continuous_watcher(self):
        # DEGRADED is the watcher-visible state for this fixture set; attach
        # must rebuild it from manifest + API + remote log alone.
        runner = DispatchRunner(monitor_handlers(tail=OOM_TAIL))
        expected = monitor.get_status(RUN, base_dir=self.base,
                                      sdk=FakeSdk(RAW_RUNNING), runner=runner)
        report = monitor.attach(RUN, base_dir=self.base,
                                sdk=FakeSdk(RAW_RUNNING),
                                runner=DispatchRunner(monitor_handlers(tail=OOM_TAIL)))
        self.assertEqual(report.state, expected.state)
        self.assertEqual(report.state, contract.RunState.DEGRADED.value)
        self.assertEqual(report.oom_skips, 3)
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.DEGRADED.value)


# ---------------------------------------------------------------------------
# U7 fixtures: SDK fake recording lifecycle ops + local artifact helpers
# ---------------------------------------------------------------------------


class LifecycleSdk(FakeSdk):
    """FakeSdk that also records stop/terminate/resume calls in order."""

    def __init__(self, *raws):
        super().__init__(*raws)
        self.ops = []

    def get_pod(self, pod_id):
        self.ops.append("get_pod")
        return super().get_pod(pod_id)

    def stop_pod(self, pod_id):
        self.ops.append("stop_pod")
        return {"id": pod_id}

    def resume_pod(self, pod_id, gpu_count):
        self.ops.append(("resume_pod", gpu_count))
        return {"id": pod_id}

    def terminate_pod(self, pod_id):
        self.ops.append("terminate_pod")
        return {"id": pod_id}


class LifecycleBase(MonitorBase):
    """MonitorBase plus local checkpoint writers and stdout-capturing calls."""

    def write_checkpoints(self, steps, final=False):
        out_dir = contract.local_output_dir(RUN, self.base)
        os.makedirs(out_dir, exist_ok=True)
        for step in steps:
            with open(os.path.join(out_dir, f"{RUN}_{step:09d}.safetensors"),
                      "w") as f:
                f.write("x")
        if final:
            with open(os.path.join(out_dir, f"{RUN}.safetensors"), "w") as f:
                f.write("x")

    def early_stop_manifest(self, **kw):
        """Run stopped at step 1500 of 5000, save_every 250 (R9 scenario)."""
        kw.setdefault("state", contract.RunState.STOPPED.value)
        kw.setdefault("total_steps", 5000)
        kw.setdefault("save_every", 250)
        kw.setdefault("last_pulled_checkpoint_step", 1500)
        kw.setdefault("hourly_rate", 1.89)
        kw.setdefault("provisioned_at", NOW - 7200)
        self.save_manifest(**kw)

    def stop(self, runner=None, sdk=None):
        runner = runner if runner is not None else DispatchRunner(monitor_handlers())
        sdk = sdk if sdk is not None else FakeSdk(RAW_RUNNING)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report = lifecycle.stop_run(RUN, base_dir=self.base, sdk=sdk,
                                        runner=runner)
        return report, out.getvalue(), runner

    def down(self, sdk, runner=None, **kw):
        runner = runner if runner is not None else DispatchRunner(monitor_handlers())
        kw.setdefault("now", NOW)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = lifecycle.down_run(RUN, base_dir=self.base, sdk=sdk,
                                      runner=runner, **kw)
        return code, out.getvalue(), runner

    def rescue(self, sdk, runner=None, **kw):
        runner = runner if runner is not None else DispatchRunner(monitor_handlers())
        kw.setdefault("now", NOW)
        kw.setdefault("_sleep", lambda s: None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = lifecycle.rescue_run(RUN, base_dir=self.base, sdk=sdk,
                                        runner=runner, **kw)
        return code, out.getvalue(), runner


# ---------------------------------------------------------------------------
# U7: stop semantics — kill the trainer, never the tmux session
# ---------------------------------------------------------------------------


class TestStopRun(LifecycleBase):
    def test_kill_targets_trainer_pattern_and_never_the_tmux_session(self):
        report, _out, runner = self.stop()
        kills = runner.commands("pkill")
        self.assertEqual(len(kills), 1)
        self.assertIn(f"[r]un.py.*{RUN}", " ".join(kills[0]))
        # NO command may kill the train tmux session: the wrapper must
        # survive to write exit_code, await the ack, and self-stop.
        for cmd in runner.calls:
            joined = " ".join(str(c) for c in cmd)
            self.assertNotIn("kill-session", joined)
            self.assertNotIn("tmux kill", joined)
        # stop always flows into a pull (the inventory find ran)
        self.assertTrue(runner.commands("-printf '%P"))
        self.assertLess(runner.first_index("pkill"),
                        runner.first_index("-printf '%P"))

    def test_settle_window_holds_stopped_when_sentinel_not_yet_written(self):
        # default handlers: no exit_code sentinel yet, tmux session alive —
        # the pull resolves a live state, but STOPPED must survive it.
        report, _out, _runner = self.stop()
        self.assertEqual(report.state, contract.RunState.STOPPED.value)
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.STOPPED.value)

    def test_sentinel_after_kill_resolves_stopped_not_crashed(self):
        runner = DispatchRunner(monitor_handlers(exit_code="143\n"))
        report, _out, _runner = self.stop(runner=runner)
        self.assertEqual(report.state, contract.RunState.STOPPED.value)
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.STOPPED.value)

    def test_stop_refuses_when_pod_not_reachable(self):
        with self.assertRaises(lifecycle.LifecycleError) as ctx:
            lifecycle.stop_run(RUN, base_dir=self.base,
                               sdk=FakeSdk(RAW_EXITED),
                               runner=DispatchRunner([]))
        self.assertIn("rescue", str(ctx.exception))


# ---------------------------------------------------------------------------
# U7: verification expectations (R9)
# ---------------------------------------------------------------------------


class TestVerifyArtifacts(LifecycleBase):
    def test_early_stop_verifies_with_observed_count_not_full_run(self):
        # stopped at 1500 of 5000, save_every 250: 6 checkpoints, NOT 20.
        # No watermark set — the observed step must come from the inventory.
        self.early_stop_manifest(last_pulled_checkpoint_step=0)
        self.write_checkpoints(range(250, 1501, 250))
        v = lifecycle.verify_artifacts(manifest.load(RUN, self.base),
                                       base_dir=self.base)
        self.assertTrue(v.ok)
        self.assertEqual(v.expected_count, 6)
        self.assertNotEqual(v.expected_count, 20)
        self.assertEqual(v.observed_max_step, 1500)

    def test_loss_mirror_feeds_observed_max_step(self):
        import sqlite3 as _sqlite3
        self.early_stop_manifest(last_pulled_checkpoint_step=0)
        db = os.path.join(contract.local_run_dir(RUN, self.base),
                          "loss_log.db")
        os.makedirs(os.path.dirname(db), exist_ok=True)
        con = _sqlite3.connect(db)
        con.execute("CREATE TABLE metrics (step INT, key TEXT, value_real REAL)")
        con.execute("INSERT INTO metrics VALUES (1500, 'loss', 0.4)")
        con.commit()
        con.close()
        # mirror says 1500 but NO checkpoints were pulled: 6 missing, named.
        v = lifecycle.verify_artifacts(manifest.load(RUN, self.base),
                                       base_dir=self.base)
        self.assertFalse(v.ok)
        self.assertEqual(v.observed_max_step, 1500)
        self.assertEqual(len(v.missing), 6)
        self.assertIn(f"{RUN}_000000250.safetensors", v.missing)

    def test_completed_demands_full_count_and_final_no_suffix_save(self):
        self.early_stop_manifest(state=contract.RunState.COMPLETED.value,
                                 last_pulled_checkpoint_step=5000)
        self.write_checkpoints(range(250, 5001, 250), final=True)
        v = lifecycle.verify_artifacts(manifest.load(RUN, self.base),
                                       base_dir=self.base)
        self.assertTrue(v.ok)
        self.assertEqual(v.expected_count, 20)  # floor(5000/250), R9

    def test_completed_missing_final_safetensors_fails(self):
        self.early_stop_manifest(state=contract.RunState.COMPLETED.value,
                                 last_pulled_checkpoint_step=5000)
        self.write_checkpoints(range(250, 5001, 250), final=False)
        v = lifecycle.verify_artifacts(manifest.load(RUN, self.base),
                                       base_dir=self.base)
        self.assertFalse(v.ok)
        self.assertIn(f"{RUN}.safetensors", v.missing)


# ---------------------------------------------------------------------------
# U7: down (R9)
# ---------------------------------------------------------------------------


class TestDownRun(LifecycleBase):
    def test_verified_down_acks_terminates_and_prints_cost(self):
        self.early_stop_manifest()
        self.write_checkpoints(range(250, 1501, 250))
        sdk = LifecycleSdk(RAW_RUNNING)
        code, out, runner = self.down(sdk)
        self.assertEqual(code, lifecycle.DOWN_OK)
        # pulled.ok written REMOTELY with content (echo redirect, not touch)
        acks = runner.commands(contract.PULLED_OK_FILE)
        self.assertEqual(len(acks), 1)
        joined = " ".join(acks[0])
        self.assertIn(f"echo {int(NOW)} >", joined)
        self.assertNotIn("touch", joined)
        # the final pull ran before the ack
        self.assertLess(runner.first_index("-printf '%P"),
                        runner.first_index(contract.PULLED_OK_FILE))
        self.assertIn("terminate_pod", sdk.ops)
        self.assertNotIn("stop_pod", sdk.ops)
        # cost printed: 2h at $1.89/hr
        self.assertIn("$3.78", out)
        m = manifest.load(RUN, self.base)
        self.assertEqual(m.state, contract.RunState.TERMINATED.value)
        self.assertEqual(m.terminated_at, NOW)

    def test_failed_verification_stops_pod_names_missing_no_ack(self):
        self.early_stop_manifest()
        steps = [s for s in range(250, 1501, 250) if s != 1000]
        self.write_checkpoints(steps)
        sdk = LifecycleSdk(RAW_RUNNING)
        code, out, runner = self.down(sdk)
        self.assertEqual(code, lifecycle.DOWN_VERIFY_FAILED)
        self.assertIn("stop_pod", sdk.ops)
        self.assertNotIn("terminate_pod", sdk.ops)
        # exactly what is missing, named in the output
        self.assertIn(f"{RUN}_000001000.safetensors", out)
        # no pulled.ok written on failure
        self.assertEqual(runner.commands(contract.PULLED_OK_FILE), [])
        self.assertNotEqual(manifest.load(RUN, self.base).state,
                            contract.RunState.TERMINATED.value)

    def test_force_terminates_despite_missing_artifacts(self):
        self.early_stop_manifest()
        steps = [s for s in range(250, 1501, 250) if s != 1000]
        self.write_checkpoints(steps)
        sdk = LifecycleSdk(RAW_RUNNING)
        code, out, runner = self.down(sdk, force=True)
        self.assertEqual(code, lifecycle.DOWN_OK)
        self.assertIn("terminate_pod", sdk.ops)
        self.assertNotIn("stop_pod", sdk.ops)
        self.assertIn(f"{RUN}_000001000.safetensors", out)  # still named
        self.assertEqual(runner.commands(contract.PULLED_OK_FILE), [])
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.TERMINATED.value)

    def test_down_on_gone_pod_succeeds_quietly_still_verifying(self):
        self.early_stop_manifest()
        self.write_checkpoints(range(250, 1501, 250))
        sdk = LifecycleSdk(None)  # GONE from the API
        runner = DispatchRunner([])
        code, out, runner = self.down(sdk, runner=runner)
        self.assertEqual(code, lifecycle.DOWN_OK)
        self.assertEqual(runner.calls, [])          # zero ssh on a gone pod
        self.assertEqual(sdk.ops, ["get_pod"])      # no terminate of a ghost
        self.assertIn("6 checkpoint(s) verified", out)
        self.assertIn("$", out)                      # cost from the manifest
        self.assertEqual(manifest.load(RUN, self.base).state,
                         contract.RunState.TERMINATED.value)


# ---------------------------------------------------------------------------
# U7: rescue (R24)
# ---------------------------------------------------------------------------


class TestRescueRun(LifecycleBase):
    def test_rescue_starts_zero_gpu_reresolves_pulls_verifies_terminates(self):
        self.early_stop_manifest()
        self.write_checkpoints(range(250, 1501, 250))
        # stopped pod first, RUNNING after the zero-GPU start
        sdk = LifecycleSdk(RAW_EXITED, RAW_RUNNING)
        code, out, runner = self.rescue(sdk)
        self.assertEqual(code, lifecycle.RESCUE_OK)
        # zero-GPU start issued, and the endpoint RE-resolved AFTER it
        self.assertIn(("resume_pod", 0), sdk.ops)
        resume_at = sdk.ops.index(("resume_pod", 0))
        self.assertIn("get_pod", sdk.ops[resume_at + 1:])
        # pull ran (inventory find), then terminate
        self.assertTrue(runner.commands("-printf '%P"))
        self.assertEqual(sdk.ops[-1], "terminate_pod")
        self.assertIn("$", out)  # cost reported
        m = manifest.load(RUN, self.base)
        self.assertEqual(m.state, contract.RunState.TERMINATED.value)
        self.assertEqual(m.terminated_at, NOW)

    def test_rescue_verification_failure_restops_not_terminates(self):
        self.early_stop_manifest()
        steps = [s for s in range(250, 1501, 250) if s != 1250]
        self.write_checkpoints(steps)
        sdk = LifecycleSdk(RAW_EXITED, RAW_RUNNING)
        code, out, _runner = self.rescue(sdk)
        self.assertEqual(code, lifecycle.RESCUE_VERIFY_FAILED)
        self.assertIn("stop_pod", sdk.ops)
        self.assertNotIn("terminate_pod", sdk.ops)
        self.assertIn(f"{RUN}_000001250.safetensors", out)

    def test_rescue_refuses_running_pod_pointing_at_stop_down(self):
        sdk = LifecycleSdk(RAW_RUNNING)
        with self.assertRaises(lifecycle.LifecycleError) as ctx:
            self.rescue(sdk)
        msg = str(ctx.exception)
        self.assertIn("RUNNING", msg)
        self.assertIn("stop", msg)
        self.assertIn("down", msg)
        self.assertNotIn(("resume_pod", 0), sdk.ops)

    def test_rescue_refuses_gone_pod_reporting_local_state(self):
        self.early_stop_manifest()
        self.write_checkpoints(range(250, 1501, 250))
        sdk = LifecycleSdk(None)
        with self.assertRaises(lifecycle.LifecycleError) as ctx:
            self.rescue(sdk)
        msg = str(ctx.exception)
        self.assertIn("nothing to rescue", msg)
        self.assertIn("6 checkpoint(s)", msg)            # local state reported
        self.assertIn(contract.RunState.STOPPED.value, msg)
        self.assertEqual(sdk.ops, ["get_pod"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
