"""Guarded, sentinel-wrapped launch (U5).

Starts training exactly once, crash-visibly, and cost-safely (R8/R10/R11):

  Guards (R10) — refuse when a training session/process already exists for
      this run on the pod; refuse when remote checkpoints exist unless
      resume=True (leave them — ai-toolkit silently auto-resumes from the
      max-ctime checkpoint) or fresh=True (move the remote output dir aside).

  Wrapper (R8) — training runs inside tmux (remain-on-exit on) as a generated
      shell script UPLOADED AS A FILE, never an inline quoted command. Body:
      bridge HF_* env from PID 1, run the trainer, write the numeric exit code
      as CONTENT into the exit_code sentinel (transport defines presence as
      "readable with non-empty content"; a bare touch is invisible), wait for
      the laptop's pulled.ok ack bounded by a max-grace deadline, then
      self-stop via the snippet provided by pod.py. Provider-specific strings
      NEVER originate in this module — pod.py owns every one of them; this
      module only embeds what pod.py hands over. (The uploaded wrapper
      artifact is necessarily provider-specific; the isolation claim is about
      source modules.)

  Timer (R8) — a hard max-runtime timer runs in its OWN tmux session so
      `stop` cannot kill it. Two-phase on fire: pkill the trainer (the
      wrapper's sentinel path runs), wait a settling beat, then — still
      executing means the pod is still up — write a timestamp as CONTENT into
      the timed_out sentinel and force-stop via the pod.py snippet.

  Early tail (R11) — after spawn, poll the remote log tail (byte-offset tail,
      classified by contract.classify_log_tail): traceback → CRASHED with the
      traceback text, pod left up for diagnosis; warming output → return
      PROMPTLY with a warming note (first-run cold starts download weights
      for 20-60 minutes before step 1 — never wait for it); training-step
      progress → RUNNING; silence past the timeout → report the raw tail and
      leave the pod up rather than marking CRASHED.

All user-supplied values are substituted into generated scripts as escaped
single-quoted literals via contract.shell_quote (R28).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.remote import contract, manifest, pod, transport
from scripts.remote.manifest import RunManifest
from scripts.remote.transport import Endpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_HOURS = 24.0          # hard max-runtime backstop (R8)
DEFAULT_MAX_GRACE_SECONDS = 1800  # ack-wait bound after training ends
ACK_POLL_SECONDS = 30             # wrapper's pulled.ok polling cadence
TIMER_SETTLE_SECONDS = 120        # beat between pkill and force-stop
EARLY_TAIL_TIMEOUT_S = 900        # bounded no-output window (R11)
EARLY_TAIL_POLL_S = 15


class LaunchError(RuntimeError):
    pass


class LaunchRefusedError(LaunchError):
    """A pre-launch guard refused to start training (R10)."""


@dataclass
class LaunchResult:
    outcome: str          # 'running' | 'warming' | 'crashed' | 'silent'
    detail: str           # traceback text or raw log tail
    manifest: RunManifest


def _check(res: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        raise LaunchError(f"{what} failed (exit {res.returncode}): {stderr[:2000]}")
    return res


# ---------------------------------------------------------------------------
# Script generation (pure functions)
# ---------------------------------------------------------------------------

def _self_stop_snippet() -> str:
    """The pod-side self-stop body, received from pod.py at generation time.

    pod.py owns every provider-specific string in the pipeline; this module
    only embeds the snippet verbatim (shebang stripped — it lands mid-script).
    """
    lines = pod.self_stop_script().splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    return "\n".join(lines).strip("\n")


# Canonical home is contract.trainer_pkill_pattern (shared with lifecycle);
# thin private alias kept for this module's internal call sites.
_trainer_pkill_pattern = contract.trainer_pkill_pattern


def _train_command(m: RunManifest, config_q: str, log_q: str) -> str:
    """The trainer invocation embedded in the wrapper.

    Single GPU (gpu_count <= 1): plain `python run.py` — the fully-validated
    default path. Multi-GPU: `accelerate launch --multi_gpu --num_processes N`
    — ai-toolkit's run.py is built for this (model/LoRA/optimizer are
    accelerator.prepare()'d, sampling/saving/logging are is_main_process
    guarded). mixed_precision/dynamo are left off so ai-toolkit owns dtype
    (it casts via train.dtype) and its own compile path. The exact flag set
    is validated on the first real multi-GPU run (see the remote-launch
    skill); override via AITK_ACCELERATE_ARGS if a pod needs different flags.
    """
    n = int(getattr(m, "gpu_count", 1) or 1)
    if n <= 1:
        return f"python run.py {config_q} --log {log_q}"
    extra = os.environ.get("AITK_ACCELERATE_ARGS", "")
    accel = (f"accelerate launch --multi_gpu --num_processes {n} "
             f"--num_machines 1 --mixed_precision no --dynamo_backend no")
    if extra:
        accel += f" {extra}"
    return f"{accel} run.py {config_q} --log {log_q}"


def build_wrapper_script(m: RunManifest,
                         max_grace_seconds: int = DEFAULT_MAX_GRACE_SECONDS) -> str:
    """Generate the sentinel wrapper bash script (pure; R8/R28).

    Body order is contract (test-enforced):
      (a) HF env bridge from PID 1  (b) run.py  (c) exit-code write WITH
      CONTENT  (d) ack-wait bounded by max-grace  (e) self-stop snippet.
    """
    run = m.run_name
    q = contract.shell_quote
    config_path = contract.remote_config_path(run)
    log_path = contract.remote_log_path(run)
    exit_path = contract.remote_sentinel_path(run, contract.EXIT_CODE_FILE)
    ack_path = contract.remote_sentinel_path(run, contract.PULLED_OK_FILE)
    grace = int(max_grace_seconds)
    train_cmd = _train_command(m, q(config_path), q(log_path))
    return f"""#!/bin/bash
# Training wrapper for run {q(run)} — generated by scripts/remote/launch.py (U5).
# Runs inside tmux session {q(contract.tmux_train_session(run))} with
# remain-on-exit on. Sentinels are written WITH CONTENT: transport defines
# presence as "readable with non-empty content"; a bare touch is invisible.

# (a) Bridge HF env from PID 1's environment. The image's start.sh bridges
# only its own provider vars and PATH into SSH sessions, so the trainer would
# never see these otherwise. Export each var only if PID 1 actually has it.
for _var in HF_TOKEN HF_HOME HF_HUB_ENABLE_HF_TRANSFER; do
    _val=$(tr '\\0' '\\n' < /proc/1/environ | grep "^${{_var}}=" | head -n 1 | cut -d= -f2-)
    if [ -n "$_val" ]; then
        export "${{_var}}=${{_val}}"
    fi
done

# (b) Train.
cd {q(contract.REMOTE_REPO_DIR)} && {train_cmd}
code=$?

# (c) Exit-code sentinel: the numeric code as CONTENT, never a bare touch.
echo "$code" > {q(exit_path)}

# (d) Ack wait: hold the pod until the laptop's verified final pull lands the
# ack file, bounded by the max-grace deadline (computed from $SECONDS).
_grace_deadline=$(( SECONDS + {grace} ))
until [ -e {q(ack_path)} ]; do
    if [ "$SECONDS" -ge "$_grace_deadline" ]; then
        echo "wrapper: max-grace deadline reached without ack; self-stopping" >&2
        break
    fi
    sleep {ACK_POLL_SECONDS}
done

# (e) Self-stop. Snippet provided by pod.py — provider strings live there.
{_self_stop_snippet()}
"""


def build_timer_script(m: RunManifest, max_hours: float = DEFAULT_MAX_HOURS) -> str:
    """Generate the max-runtime timer bash script (pure; R8).

    Runs in its own tmux session (contract.tmux_timer_session) so `stop` —
    which targets the training session/process — can never kill it. Two-phase
    on fire: pkill the trainer (the wrapper's sentinel path runs), settle,
    then write the timed_out sentinel WITH CONTENT and force-stop.
    """
    run = m.run_name
    q = contract.shell_quote
    timed_out_path = contract.remote_sentinel_path(run, contract.TIMED_OUT_FILE)
    sleep_seconds = int(float(max_hours) * 3600)
    return f"""#!/bin/bash
# Max-runtime timer for run {q(run)} — generated by scripts/remote/launch.py (U5).
# Runs in its own tmux session {q(contract.tmux_timer_session(run))} so `stop`
# (which targets the training session/process) can never kill it (R8).

sleep {sleep_seconds}

# Phase 1: kill the trainer, NOT the wrapper — the wrapper's sentinel path
# (exit-code write, ack wait, self-stop) must still execute.
pkill -f {q(_trainer_pkill_pattern(run))}

# Settling beat: give the wrapper time to write its sentinel / start its ack wait.
sleep {TIMER_SETTLE_SECONDS}

# Phase 2: this line executing at all means the pod is still up — record the
# timeout as a sentinel WITH CONTENT, then force-stop via the pod.py snippet.
date +%s > {q(timed_out_path)}
{_self_stop_snippet()}
"""


# ---------------------------------------------------------------------------
# Pre-launch guards (R10)
# ---------------------------------------------------------------------------

def _guard_not_running(ep: Endpoint, run_name: str, *, runner) -> None:
    """Refuse when a training session or trainer process already exists."""
    q = contract.shell_quote
    session = contract.tmux_train_session(run_name)
    res = transport.ssh_run(ep, f"tmux has-session -t {q(session)} 2>/dev/null",
                            runner=runner)
    if res.returncode == 0:
        raise LaunchRefusedError(
            f"training session '{session}' already exists on the pod for run "
            f"'{run_name}' — refusing to double-launch. Use `status` to check "
            "progress or `attach` to reattach a watcher."
        )
    pattern = _trainer_pkill_pattern(run_name)
    res = transport.ssh_run(ep, f"pgrep -f {q(pattern)}", runner=runner)
    if res.returncode == 0 and (res.stdout or "").strip():
        pids = " ".join((res.stdout or "").split())
        raise LaunchRefusedError(
            f"a trainer process for run '{run_name}' is already running on the "
            f"pod (pid {pids}) — refusing to double-launch. Use `status` to "
            "check progress or `attach` to reattach a watcher."
        )


def _guard_checkpoints(ep: Endpoint, m: RunManifest, *, resume: bool,
                       fresh: bool, now: float, runner) -> None:
    """Refuse on existing remote checkpoints unless resume/fresh (R10).

    ai-toolkit silently auto-resumes from the max-ctime checkpoint in the job
    dir, so an unflagged relaunch over old checkpoints would quietly continue
    a run the user thought was starting over.
    """
    q = contract.shell_quote
    job_dir = contract.remote_job_dir(m.run_name, m.job_name or m.run_name)
    res = transport.ssh_run(ep, f"ls {q(job_dir)}/*.safetensors 2>/dev/null",
                            runner=runner)
    names = [n for n in (res.stdout or "").split() if n]
    if res.returncode != 0 or not names:
        return  # no checkpoints, nothing to guard
    if fresh:
        aside = f"{job_dir}.old.{int(now)}"
        _check(transport.ssh_run(ep, f"mv {q(job_dir)} {q(aside)}", runner=runner),
               f"moving remote output dir aside to {aside}")
        return
    if resume:
        return  # leave checkpoints in place; the trainer resumes from the latest
    raise LaunchRefusedError(
        f"{len(names)} remote checkpoint(s) already exist under {job_dir} "
        f"(e.g. {os.path.basename(names[0])}). ai-toolkit silently auto-resumes "
        "from the newest one. Pass --resume to continue that run intentionally, "
        "or --fresh to move the remote output dir aside and start over."
    )


# ---------------------------------------------------------------------------
# Early tail (R11)
# ---------------------------------------------------------------------------

def _extract_traceback(tail: str) -> str:
    match = contract.TRACEBACK_RE.search(tail)
    return tail[match.start():] if match else tail


def early_tail(ep: Endpoint, m: RunManifest, *, timeout_s: float = EARLY_TAIL_TIMEOUT_S,
               poll_s: float = EARLY_TAIL_POLL_S, runner=subprocess.run,
               _sleep=time.sleep):
    """Block-tail the remote log after spawn and classify (R11).

    Returns (outcome, detail):
      'crashed' — traceback seen; detail is the traceback text (the caller
                  marks CRASHED and leaves the pod up for diagnosis)
      'running' — a training-step progress record seen
      'warming' — model download/load/cache output; returns PROMPTLY (cold
                  starts take 20-60 minutes before step 1 — never wait)
      'silent'  — nothing recognizable within timeout_s; detail is the raw
                  tail; the pod is left up, never marked CRASHED on silence
    """
    log_path = contract.remote_log_path(m.run_name)
    tail_cmd = (f"tail -c {transport.LOG_TAIL_BYTES} "
                f"{contract.shell_quote(log_path)} 2>/dev/null")
    elapsed = 0.0
    last_tail = ""
    while True:
        res = transport.ssh_run(ep, tail_cmd, runner=runner)
        tail = res.stdout if (res.returncode == 0 and res.stdout) else ""
        if tail.strip():
            last_tail = tail
            kind = contract.classify_log_tail(tail)
            if kind == "traceback":
                return "crashed", _extract_traceback(tail)
            if kind == "progress":
                return "running", tail
            if kind == "warming":
                return "warming", tail
        if elapsed >= timeout_s:
            return "silent", last_tail
        _sleep(poll_s)
        elapsed += poll_s


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def _upload_script(ep: Endpoint, content: str, remote_path: str, *, runner) -> str:
    """Write a generated script to a temp file and upload it AS A FILE.

    The wrapper/timer are never passed to tmux as inline quoted commands —
    quoting through ssh+tmux subprocess layers is the likeliest failure mode
    (plan KTD), so the only thing tmux ever executes is `bash <file>`.
    """
    fd, tmp = tempfile.mkstemp(prefix=".aitk-script-", suffix=".sh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        transport.upload_file(ep, tmp, remote_path, runner=runner)
    finally:
        os.unlink(tmp)
    return remote_path


def launch_run(run_name: str, *, ep: Endpoint, resume: bool = False,
               fresh: bool = False, max_hours: float = DEFAULT_MAX_HOURS,
               max_grace_seconds: int = DEFAULT_MAX_GRACE_SECONDS,
               base_dir: str = ".", runner=subprocess.run, now: float = None,
               tail_timeout_s: float = EARLY_TAIL_TIMEOUT_S,
               tail_poll_s: float = EARLY_TAIL_POLL_S,
               _sleep=time.sleep) -> LaunchResult:
    """Guarded, sentinel-wrapped launch (R8/R10/R11). See module docstring."""
    contract.validate_run_name(run_name)
    if resume and fresh:
        raise LaunchError("--resume and --fresh are mutually exclusive")
    m = manifest.load(run_name, base_dir)
    now = time.time() if now is None else float(now)
    q = contract.shell_quote

    # --- pre-launch guards (R10) ------------------------------------------
    _guard_not_running(ep, run_name, runner=runner)
    _guard_checkpoints(ep, m, resume=resume, fresh=fresh, now=now, runner=runner)

    # --- multi-GPU effective-batch warning --------------------------------
    n_gpu = int(getattr(m, "gpu_count", 1) or 1)
    if n_gpu > 1:
        print(f"[launch] {run_name}: multi-GPU ({n_gpu} GPUs) via "
              f"`accelerate launch`. Effective batch scales ~{n_gpu}x — the "
              f"dataset is seen ~{n_gpu}x faster per step, so step counts/LR "
              f"tuned for 1 GPU will OVER-train. Cut train.steps roughly "
              f"{n_gpu}x (or raise LR with care) for equivalent training.",
              file=sys.stderr)

    # --- fresh start: invalidate all pull/review progress (#2) -------------
    # The remote output dir was moved aside by the guard; the LOCAL mirror and
    # the manifest watermarks describe the OLD attempt and must reset too, or
    # the first pull of the new run would skip everything below the old marks.
    if fresh:
        m.last_pulled_sample_step = 0
        m.last_pulled_checkpoint_step = 0
        m.last_reviewed_step = 0
        m.optimizer_pairing_step = None
        local_out = contract.local_output_dir(run_name, base_dir)
        if os.path.isdir(local_out):
            os.rename(local_out, f"{local_out}.old.{int(now)}")

    # --- clear stale sentinels + rotate the prior log (#8) ------------------
    # A relaunch over leftovers would otherwise resolve instantly as
    # COMPLETED/CRASHED/TIMED_OUT from the previous attempt's sentinels, and
    # early_tail could match an OLD traceback (toolkit/print.py opens the log
    # in append mode).
    sentinel_paths = [contract.remote_sentinel_path(run_name, s)
                      for s in (contract.EXIT_CODE_FILE, contract.TIMED_OUT_FILE,
                                contract.PULLED_OK_FILE)]
    _check(transport.ssh_run(
        ep, "rm -f " + " ".join(q(p) for p in sentinel_paths), runner=runner),
        "clearing stale sentinels")
    log_path = contract.remote_log_path(run_name)
    _check(transport.ssh_run(
        ep, f"if [ -f {q(log_path)} ]; then mv {q(log_path)} "
            f"{q(f'{log_path}.{int(now)}')}; fi", runner=runner),
        "rotating remote log")

    # --- generate + upload the wrapper and timer as files ------------------
    run_root = contract.remote_run_root(run_name)
    wrapper_remote = f"{run_root}/{contract.WRAPPER_SCRIPT}"
    timer_remote = f"{run_root}/{contract.TIMER_SCRIPT}"
    _upload_script(ep, build_wrapper_script(m, max_grace_seconds),
                   wrapper_remote, runner=runner)
    _upload_script(ep, build_timer_script(m, max_hours), timer_remote,
                   runner=runner)

    # --- spawn: training session, then the timer in its OWN session --------
    train_session = contract.tmux_train_session(run_name)
    timer_session = contract.tmux_timer_session(run_name)
    _check(transport.ssh_run(
        ep, f"tmux new-session -d -s {q(train_session)} bash {q(wrapper_remote)}",
        runner=runner), "spawning training session")
    _check(transport.ssh_run(
        ep, f"tmux set-option -t {q(train_session)} remain-on-exit on",
        runner=runner), "setting remain-on-exit on training session")
    _check(transport.ssh_run(
        ep, f"tmux new-session -d -s {q(timer_session)} bash {q(timer_remote)}",
        runner=runner), "spawning max-runtime timer session")
    _check(transport.ssh_run(
        ep, f"tmux set-option -t {q(timer_session)} remain-on-exit on",
        runner=runner), "setting remain-on-exit on timer session")

    # --- record launch facts; RUNNING is the optimistic post-spawn state ---
    m.launched_at = now
    m.max_runtime_deadline = now + float(max_hours) * 3600.0
    m.max_grace_seconds = int(max_grace_seconds)
    m.tmux_session = train_session
    m.timer_session = timer_session
    m.state = contract.RunState.RUNNING.value
    manifest.save(m, base_dir)

    # --- early tail: the three R11 outcomes --------------------------------
    outcome, detail = early_tail(ep, m, timeout_s=tail_timeout_s,
                                 poll_s=tail_poll_s, runner=runner, _sleep=_sleep)
    if outcome == "crashed":
        m.state = contract.RunState.CRASHED.value
        manifest.save(m, base_dir)
    return LaunchResult(outcome=outcome, detail=detail, manifest=m)
