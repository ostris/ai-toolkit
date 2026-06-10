"""Status, health, and watch loop for the remote-GPU training pipeline (U6).

One command that tells the truth about a run — step, loss, health, disk,
lifecycle state — and a loop that keeps local artifacts current.

State precedence (R13/R14) branches on RunPod API pod STATUS, not existence:
  1. pod gone from the API            → POD_LOST (even over a stale mirror)
  2. pod exited/stopped               → locally MIRRORED sentinels only (R25;
                                        SSH is unavailable): timed_out →
                                        TIMED_OUT; exit_code 0 → COMPLETED,
                                        nonzero → CRASHED (or STOPPED when the
                                        manifest recorded a stop); no sentinel
                                        at all → TIMED_OUT, the conservative
                                        default. NO ssh on this branch.
  3. pod running                      → remote sentinels over ssh cat:
                                        timed_out / exit_code as above.
  4. no sentinel, tmux session alive  → RUNNING / DEGRADED (OOM-skip lines in
                                        the tail, count reported) / SAMPLING
                                        (no step advance but fresh sample
                                        mtimes); >= 25 min with no new files
                                        flags "stalled" in the detail.
  5. pod running, tmux gone           → UNKNOWN (container restart or tmux
                                        death), its own watch exit code.

Progress (R13): loss_log.db is queried IN PLACE over ssh via `python3 -c`
with stdlib sqlite3 in read-only URI mode (file:...?mode=ro) — the image
ships no sqlite3 CLI, and the WAL db must never be copied mid-write. A
missing db is tolerated (step None: the trainer is still warming).

Disk (R26): `df -P /workspace` with a usage-percent warning threshold —
keep-all checkpoints make disk-full a designed-in risk.

Reviewability (R16, decided HERE): a pulled sample step is reviewable when
its local file count matches the manifest's prompt count, or when files exist
and the newest is older than a stability threshold (partial batch, warned).
Only steps beyond manifest.last_reviewed_step are reported.

Watch (R17): status + pull every interval; prints state transitions and newly
reviewable steps; exits with contract.WATCH_EXIT_CODES on terminal states.
--once returns the terminal code, else EXIT_NEW_SAMPLES when new reviewable
steps exist, else EXIT_RUNNING; --json emits ONE structured object.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.remote import contract, manifest, pod, transport
from scripts.remote.manifest import RunManifest
from scripts.remote.transport import Endpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_ACTIVE_SECONDS = 300     # log written this recently → steps are advancing
SAMPLE_FRESH_SECONDS = 600    # newest sample younger than this → SAMPLING proof
STALL_SOFT_SECONDS = 1200     # < 20 min without advance is a NORMAL stall
STALL_HARD_SECONDS = 1500     # >= 25 min without advance and no new files → stalled
DISK_WARN_PCT = 85            # /workspace use% warning threshold (R26)
REVIEW_STABILITY_SECONDS = 120  # local partial-batch age before reviewable (R16)
DEFAULT_WATCH_INTERVAL_S = 600  # watch cadence (R17)


@dataclass
class StatusReport:
    run_name: str
    state: str
    step: int = None
    total_steps: int = None
    recent_loss: float = None
    oom_skips: int = 0
    disk_used_pct: int = None
    disk_warning: bool = False
    drift: bool = False
    cost_estimate: float = None
    detail: str = ""
    reviewable: list = field(default_factory=list)  # sample steps ready for review


@dataclass
class _Resolution:
    state: str
    detail: str
    oom_skips: int = 0


def _warn(message: str):
    print(f"[monitor] warning: {message}", file=sys.stderr)


def _endpoint(pod_info) -> Endpoint:
    """Endpoint from a live PodInfo, or None when SSH is unreachable."""
    if pod_info is not None and pod_info.ssh_host and pod_info.ssh_port:
        return Endpoint(host=pod_info.ssh_host, port=int(pod_info.ssh_port))
    return None


def _pod_info(m: RunManifest, sdk):
    if not m.pod_id:
        return None
    return pod.get_pod_info(m.pod_id, sdk=sdk)


# ---------------------------------------------------------------------------
# Sentinel readers
# ---------------------------------------------------------------------------
# Presence convention (shared with transport): a sentinel exists when it is
# readable WITH non-empty content — the wrapper/timer write content, never a
# bare touch.

def _read_local_sentinel(m: RunManifest, sentinel: str, base_dir: str) -> str:
    """Mirrored sentinel content from runs/<run>/, or None when absent."""
    path = os.path.join(contract.local_run_dir(m.run_name, base_dir), sentinel)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        content = f.read().strip()
    return content if content else None


def _cat_remote_sentinel(ep: Endpoint, run_name: str, sentinel: str, runner) -> str:
    """Remote sentinel content via ssh cat, or None when absent/empty."""
    remote = contract.remote_sentinel_path(run_name, sentinel)
    res = transport.ssh_run(ep, f"cat {contract.shell_quote(remote)} 2>/dev/null",
                            runner=runner)
    if res.returncode == 0 and (res.stdout or "").strip():
        return res.stdout.strip()
    return None


def _exit_code_resolution(m: RunManifest, content: str, source: str) -> _Resolution:
    """Map exit_code sentinel content to COMPLETED/CRASHED/STOPPED (R14)."""
    try:
        code = int(content.strip())
    except (ValueError, AttributeError):
        code = None
    if code == 0:
        return _Resolution(contract.RunState.COMPLETED.value,
                           f"trainer exited 0 ({source} exit_code sentinel)")
    if m.state == contract.RunState.STOPPED.value:
        return _Resolution(contract.RunState.STOPPED.value,
                           f"trainer exited with code {content.strip()} after a "
                           f"requested stop ({source} sentinel)")
    return _Resolution(contract.RunState.CRASHED.value,
                       f"trainer exited with code {content.strip()} "
                       f"({source} exit_code sentinel)")


def _resolve_from_local_mirrors(m: RunManifest, pod_info, base_dir: str) -> _Resolution:
    """Pod exited/stopped: resolve from mirrored sentinels only — NO ssh (R25)."""
    timed_out = _read_local_sentinel(m, contract.TIMED_OUT_FILE, base_dir)
    if timed_out is not None:
        return _Resolution(contract.RunState.TIMED_OUT.value,
                           "max-runtime timer fired (mirrored timed_out sentinel)")
    exit_code = _read_local_sentinel(m, contract.EXIT_CODE_FILE, base_dir)
    if exit_code is not None:
        return _exit_code_resolution(m, exit_code, source="mirrored")
    return _Resolution(
        contract.RunState.TIMED_OUT.value,
        f"pod is {pod_info.status} with no mirrored sentinel at all — "
        "resolving TIMED_OUT as the conservative default (R14); use rescue "
        "to read the volume directly")


# ---------------------------------------------------------------------------
# Running substate: RUNNING / DEGRADED / SAMPLING (+ stalled detail)
# ---------------------------------------------------------------------------

def _activity_command(m: RunManifest) -> str:
    """One ssh command printing pod-clock now, log mtime, newest sample mtime.

    All three from the SAME clock so ages are pod-relative (R15's lesson). The
    log mtime is the step-advance proxy: tqdm rewrites the teed log on every
    step, so a stale log means no step has finished since.
    """
    q = contract.shell_quote
    log_path = contract.remote_log_path(m.run_name)
    samples_dir = contract.remote_samples_dir(m.run_name, m.job_name or m.run_name)
    return (
        f"echo NOW=$(date +%s); "
        f"echo LOG=$(stat -c %Y {q(log_path)} 2>/dev/null); "
        f"echo SAMPLE=$(find {q(samples_dir)} -type f -printf '%T@\\n' "
        "2>/dev/null | sort -n | tail -1)"
    )


def _parse_activity(stdout: str):
    """(log_age_s, sample_age_s) from the activity probe; None when absent."""
    values = {}
    for line in (stdout or "").splitlines():
        if "=" not in line:
            continue
        key, _, raw = line.partition("=")
        raw = raw.strip()
        if raw:
            try:
                values[key.strip()] = float(raw)
            except ValueError:
                continue
    now = values.get("NOW")
    if now is None:
        return None, None
    log_age = (now - values["LOG"]) if "LOG" in values else None
    sample_age = (now - values["SAMPLE"]) if "SAMPLE" in values else None
    return log_age, sample_age


def _running_substate(ep: Endpoint, m: RunManifest, runner) -> _Resolution:
    """Branch 4: tmux alive, no sentinel → RUNNING / DEGRADED / SAMPLING."""
    log_path = contract.remote_log_path(m.run_name)
    res = transport.ssh_run(
        ep, f"tail -c {transport.LOG_TAIL_BYTES} "
            f"{contract.shell_quote(log_path)} 2>/dev/null", runner=runner)
    tail = res.stdout if (res.returncode == 0 and res.stdout) else ""

    oom = contract.count_oom_skips(tail)
    if oom > 0:
        return _Resolution(
            contract.RunState.DEGRADED.value,
            f"{oom} OOM-skip line(s) in the recent log — training continues "
            "but batches are being dropped", oom_skips=oom)

    res = transport.ssh_run(ep, _activity_command(m), runner=runner)
    log_age, sample_age = _parse_activity(res.stdout if res.returncode == 0 else "")

    if log_age is None or log_age < STEP_ACTIVE_SECONDS:
        return _Resolution(contract.RunState.RUNNING.value, "")
    fresh_samples = sample_age is not None and sample_age < SAMPLE_FRESH_SECONDS
    if fresh_samples:
        return _Resolution(
            contract.RunState.SAMPLING.value,
            f"no step advance for {int(log_age)}s but sample files are still "
            f"being written (newest {int(sample_age)}s old) — sampling, not hung")
    if log_age >= STALL_HARD_SECONDS:
        return _Resolution(
            contract.RunState.RUNNING.value,
            f"STALLED: no step advance for {int(log_age)}s and no new sample "
            f"files (threshold {STALL_HARD_SECONDS}s) — inspect with tmux attach")
    return _Resolution(
        contract.RunState.RUNNING.value,
        f"no step advance for {int(log_age)}s — normal below "
        f"{STALL_SOFT_SECONDS}s (sampling batches / checkpoint saves)")


# ---------------------------------------------------------------------------
# State resolution (R14 precedence)
# ---------------------------------------------------------------------------

def _resolve(m: RunManifest, pod_info, *, ep=None, runner=subprocess.run,
             base_dir: str = ".") -> _Resolution:
    # 1) pod gone from the API → POD_LOST, even over a stale local mirror.
    if pod_info is None or pod_info.status == "GONE":
        return _Resolution(
            contract.RunState.POD_LOST.value,
            "pod is gone from the RunPod API (terminated externally?) — local "
            "mirrors under runs/<run>/ are all that remain")

    # 2) pod exited/stopped → mirrored local sentinels only; NO ssh.
    if pod_info.status != "RUNNING":
        return _resolve_from_local_mirrors(m, pod_info, base_dir)

    # 3) pod running → remote sentinels over ssh.
    if ep is None:
        return _Resolution(
            contract.RunState.UNKNOWN.value,
            "pod reports RUNNING but exposes no direct SSH endpoint yet")
    timed_out = _cat_remote_sentinel(ep, m.run_name, contract.TIMED_OUT_FILE, runner)
    if timed_out is not None:
        return _Resolution(contract.RunState.TIMED_OUT.value,
                           "max-runtime timer fired (remote timed_out sentinel)")
    exit_code = _cat_remote_sentinel(ep, m.run_name, contract.EXIT_CODE_FILE, runner)
    if exit_code is not None:
        return _exit_code_resolution(m, exit_code, source="remote")

    # 4/5) no sentinel → tmux session presence decides.
    session = m.tmux_session or contract.tmux_train_session(m.run_name)
    res = transport.ssh_run(
        ep, f"tmux has-session -t {contract.shell_quote(session)} 2>/dev/null",
        runner=runner)
    if res.returncode != 0:
        return _Resolution(
            contract.RunState.UNKNOWN.value,
            f"pod is running but tmux session '{session}' is gone with no "
            "sentinel — container restart or tmux death; inspect manually")
    return _running_substate(ep, m, runner)


def resolve_state(m: RunManifest, pod_info, *, ep=None, runner=subprocess.run,
                  base_dir: str = "."):
    """Resolve the run's lifecycle state. Returns (state_str, detail_str)."""
    r = _resolve(m, pod_info, ep=ep, runner=runner, base_dir=base_dir)
    return r.state, r.detail


# ---------------------------------------------------------------------------
# Progress: loss_log.db queried in place over ssh (R13)
# ---------------------------------------------------------------------------

def build_progress_command(m: RunManifest) -> str:
    """The remote command printing 'step,loss' for the latest logged step.

    stdlib sqlite3 via `python3 -c` in READ-ONLY URI mode (file:...?mode=ro):
    the image ships no sqlite3 CLI, and mode=ro never touches the WAL of a db
    the trainer is mid-writing. A missing/locked db prints nothing (warming).
    """
    db_path = contract.remote_loss_db_path(m.run_name, m.job_name or m.run_name)
    uri = "file:" + db_path.replace("'", "\\'") + "?mode=ro"
    code = (
        "import sqlite3\n"
        "try:\n"
        f" con=sqlite3.connect('{uri}',uri=True)\n"
        " row=con.execute(\"SELECT step,value_real FROM metrics"
        " WHERE key='loss' ORDER BY step DESC LIMIT 1\").fetchone()\n"
        " print('' if row is None else"
        " '%s,%s'%(row[0],'' if row[1] is None else row[1]))\n"
        "except Exception:\n"
        " print('')\n"
    )
    return f"python3 -c {contract.shell_quote(code)}"


def query_progress(ep: Endpoint, m: RunManifest, *, runner=subprocess.run):
    """(step, loss) from the remote loss db; (None, None) when unavailable."""
    res = transport.ssh_run(ep, build_progress_command(m), runner=runner)
    text = (res.stdout or "").strip() if res.returncode == 0 else ""
    if not text or "," not in text:
        return None, None
    step_s, _, loss_s = text.partition(",")
    try:
        step = int(step_s)
    except ValueError:
        return None, None
    try:
        loss = float(loss_s) if loss_s.strip() else None
    except ValueError:
        loss = None
    return step, loss


# ---------------------------------------------------------------------------
# Disk (R26)
# ---------------------------------------------------------------------------

def check_disk(ep: Endpoint, *, runner=subprocess.run):
    """(/workspace use%, warning_bool); (None, False) when unreadable."""
    res = transport.ssh_run(ep, "df -P /workspace | tail -1", runner=runner)
    if res.returncode != 0 or not (res.stdout or "").strip():
        return None, False
    fields = res.stdout.split()
    if len(fields) < 2:
        return None, False
    try:
        pct = int(fields[-2].rstrip("%"))
    except ValueError:
        return None, False
    return pct, pct >= DISK_WARN_PCT


# ---------------------------------------------------------------------------
# Reviewability (R16 — the decision lives HERE)
# ---------------------------------------------------------------------------

def reviewable_steps(m: RunManifest, base_dir: str = ".", now: float = None) -> list:
    """Locally pulled sample steps ready for review, beyond last_reviewed_step.

    A step is reviewable when its local pulled file count matches the
    manifest's prompt_count, or when files exist and the newest is older than
    REVIEW_STABILITY_SECONDS (a partial batch — reported with a warning).
    Fresh partial batches are never handed to the reviewer.
    """
    now = time.time() if now is None else float(now)
    samples_dir = os.path.join(contract.local_output_dir(m.run_name, base_dir),
                               "samples")
    if not os.path.isdir(samples_dir):
        return []
    groups = {}
    for name in os.listdir(samples_dir):
        parsed = contract.parse_sample_filename(name)
        if parsed is None:
            continue
        step, _count = parsed
        if step <= (m.last_reviewed_step or 0):
            continue
        try:
            mtime = os.path.getmtime(os.path.join(samples_dir, name))
        except OSError:
            continue
        groups.setdefault(step, []).append(mtime)
    out = []
    for step, mtimes in sorted(groups.items()):
        count = len(mtimes)
        if m.prompt_count and count >= m.prompt_count:
            out.append(step)
        elif count > 0 and (now - max(mtimes)) >= REVIEW_STABILITY_SECONDS:
            out.append(step)
            if m.prompt_count and count < m.prompt_count:
                _warn(f"step {step}: partial sample batch ({count}/"
                      f"{m.prompt_count}) is old enough to review — treating "
                      "as reviewable with a partial-batch warning")
    return out


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _status_from(m: RunManifest, pod_info, ep, *, base_dir: str, runner) -> StatusReport:
    r = _resolve(m, pod_info, ep=ep, runner=runner, base_dir=base_dir)
    step = loss = None
    disk_pct, disk_warn = None, False
    if pod_info is not None and pod_info.status == "RUNNING" and ep is not None:
        step, loss = query_progress(ep, m, runner=runner)
        disk_pct, disk_warn = check_disk(ep, runner=runner)

    details = [r.detail] if r.detail else []
    if step is None and r.state in contract.LIVE_STATES:
        details.append("loss_log.db not readable yet (trainer still warming)")
    if disk_warn:
        details.append(f"disk warning: /workspace at {disk_pct}% used "
                       f"(threshold {DISK_WARN_PCT}%) — checkpoints are "
                       "keep-all; a full disk kills the next save (R26)")
    drift = transport.check_config_drift(m, base_dir=base_dir)
    if drift:
        details.append("config drift: runs/<run>/remote_config.yaml no longer "
                       "matches the hash recorded at launch (R19)")

    return StatusReport(
        run_name=m.run_name,
        state=r.state,
        step=step,
        total_steps=m.total_steps,
        recent_loss=loss,
        oom_skips=r.oom_skips,
        disk_used_pct=disk_pct,
        disk_warning=disk_warn,
        drift=drift,
        cost_estimate=m.estimated_cost(),
        detail="; ".join(details),
        reviewable=reviewable_steps(m, base_dir=base_dir),
    )


def get_status(run_name: str, *, base_dir: str = ".", sdk=None,
               runner=subprocess.run) -> StatusReport:
    """One truthful report: state, step, loss, health, disk, drift, cost."""
    m = manifest.load(run_name, base_dir)
    pod_info = _pod_info(m, sdk)
    ep = _endpoint(pod_info)
    return _status_from(m, pod_info, ep, base_dir=base_dir, runner=runner)


# ---------------------------------------------------------------------------
# Pull + watch (R17)
# ---------------------------------------------------------------------------

_VALID_STATES = {s.value for s in contract.RunState}


def _persist_state(m: RunManifest, state: str, base_dir: str):
    """Persist the freshly resolved state (a cached hint, re-derived on read).

    DETACHED is computed, never stored — _resolve never produces it, so this
    cannot violate that invariant.
    """
    if state in _VALID_STATES:
        m.state = state
    manifest.save(m, base_dir)


def pull_once(run_name: str, *, base_dir: str = ".", sdk=None,
              runner=subprocess.run):
    """One status+pull cycle. Returns (StatusReport, PullResult|None).

    When the pod is reachable: pull_artifacts (which also mirrors sentinels,
    R25), then resolve state and persist it with the pull watermarks. When it
    is not, status resolves from local mirrors with no ssh at all.
    """
    m = manifest.load(run_name, base_dir)
    pod_info = _pod_info(m, sdk)
    ep = _endpoint(pod_info)
    pull = None
    if pod_info is not None and pod_info.status == "RUNNING" and ep is not None:
        pull = transport.pull_artifacts(ep, m, base_dir=base_dir, runner=runner)
    report = _status_from(m, pod_info, ep, base_dir=base_dir, runner=runner)
    _persist_state(m, report.state, base_dir)
    return report, pull


def _emit(run_name: str, report: StatusReport, code: int, json_out: bool,
          base_dir: str) -> int:
    """Final emission: ONE json object when --json, then the exit code."""
    if json_out:
        m = manifest.load(run_name, base_dir)
        print(json.dumps({
            "run": report.run_name,
            "state": report.state,
            "step": report.step,
            "total_steps": report.total_steps,
            "loss": report.recent_loss,
            "oom_skips": report.oom_skips,
            "disk_used_pct": report.disk_used_pct,
            "drift": report.drift,
            "reviewable_steps": report.reviewable,
            "last_reviewed_step": m.last_reviewed_step,
            "cost_estimate": report.cost_estimate,
            "exit_code": code,
        }))
    return code


def watch(run_name: str, *, interval_s: float = DEFAULT_WATCH_INTERVAL_S,
          once: bool = False, json_out: bool = False, base_dir: str = ".",
          sdk=None, runner=subprocess.run, _sleep=time.sleep) -> int:
    """Status + pull every interval (R17). Returns the exit code.

    Prints state transitions and newly reviewable sample steps; exits with the
    distinct contract.WATCH_EXIT_CODES code on terminal states so agent loops
    can branch. once=True performs a single cycle: terminal code if terminal,
    else EXIT_NEW_SAMPLES when new reviewable steps exist, else EXIT_RUNNING.
    json_out prints ONE structured object (the schedule-driven agent contract).
    """
    last_state = None
    announced = set()
    while True:
        report, _pull = pull_once(run_name, base_dir=base_dir, sdk=sdk,
                                  runner=runner)
        new_steps = [s for s in report.reviewable if s not in announced]
        announced.update(new_steps)
        if not json_out:
            if report.state != last_state:
                suffix = f" — {report.detail}" if report.detail else ""
                print(f"[watch] {run_name}: {last_state or '(start)'} -> "
                      f"{report.state}{suffix}")
            if new_steps:
                print(f"[watch] {run_name}: new reviewable sample steps: "
                      f"{new_steps}")
        try:
            state_enum = contract.RunState(report.state)
        except ValueError:
            state_enum = None
        if state_enum in contract.WATCH_EXIT_CODES:
            return _emit(run_name, report,
                         contract.WATCH_EXIT_CODES[state_enum], json_out,
                         base_dir)
        if once:
            code = (contract.EXIT_NEW_SAMPLES if new_steps
                    else contract.EXIT_RUNNING)
            return _emit(run_name, report, code, json_out, base_dir)
        last_state = report.state
        _sleep(interval_s)


def attach(run_name: str, *, base_dir: str = ".", sdk=None,
           runner=subprocess.run) -> StatusReport:
    """Rebuild watcher state purely from manifest + API + sentinels/log.

    A fresh session with zero context reaches the exact state a continuous
    watcher would hold; the re-derived state is persisted to the manifest.
    """
    m = manifest.load(run_name, base_dir)
    pod_info = _pod_info(m, sdk)
    ep = _endpoint(pod_info)
    report = _status_from(m, pod_info, ep, base_dir=base_dir, runner=runner)
    _persist_state(m, report.state, base_dir)
    return report
