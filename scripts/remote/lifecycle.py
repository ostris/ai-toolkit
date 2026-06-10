"""Stop, down, and rescue lifecycle flows (U7).

Terminal operations that never destroy unverified data (R9) plus the rescue
path that retrieves artifacts from a stopped pod (R24).

stop   — pkill the run.py trainer process over ssh, NEVER the tmux session:
         the wrapper must survive to write the exit_code sentinel, wait for
         the pulled.ok ack, and self-stop. STOPPED rides the same sentinel
         mechanics as CRASHED, and always flows into a pull. The settle
         window is tolerated: the sentinel may land after the pull resolves,
         so a live-looking resolution never clobbers the recorded STOPPED.

down   — final incremental pull → verify against manifest expectations →
         write pulled.ok remotely WITH CONTENT (only when the pod is
         reachable) → terminate → record TERMINATED + cost report. The
         expected checkpoint count derives from the OBSERVED max trained
         step — floor(observed / save_every) — so reviewer-driven early
         stops verify cleanly; the configured-total formula and the final
         no-suffix safetensors requirement apply ONLY when the run state is
         COMPLETED (R9). On verification failure the pod is STOPPED (not
         terminated), exactly what is missing is printed, and a failure code
         returns — unless force=True, which terminates anyway. down on an
         already-terminated/GONE pod succeeds quietly, still verifying local
         artifacts, with cost from the manifest.

rescue — for a stopped pod (self-stopped or stop-on-failed-pull): start it
         with zero GPUs, re-resolve the SSH endpoint via the API (the
         manifest endpoint is cache; RunPod reassigns IP/port across
         stop/start), pull + mirror, verify, terminate. Refuses on a RUNNING
         pod (use stop/down) and on a GONE pod (nothing to rescue).
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.remote import contract, manifest, monitor, pod, transport
from scripts.remote.manifest import RunManifest
from scripts.remote.transport import Endpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOWN_OK = 0
DOWN_VERIFY_FAILED = 1
RESCUE_OK = 0
RESCUE_VERIFY_FAILED = 1

# After a zero-GPU start, RunPod reassigns the endpoint; bounded re-resolution.
RESCUE_ENDPOINT_TIMEOUT_S = 600
RESCUE_ENDPOINT_POLL_S = 10



class LifecycleError(RuntimeError):
    pass


def _endpoint(pod_info) -> Endpoint:
    if pod_info is not None and pod_info.ssh_host and pod_info.ssh_port:
        return Endpoint(host=pod_info.ssh_host, port=int(pod_info.ssh_port))
    return None


# ---------------------------------------------------------------------------
# Verification (R9)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    ok: bool
    missing: list                 # artifact filenames expected but absent locally
    observed_max_step: int
    expected_count: int
    found_steps: list = field(default_factory=list)
    final_present: bool = False


def _loss_mirror_max_step(m: RunManifest, base_dir: str):
    """MAX(step) from the locally mirrored loss_log.db, or None.

    The mirror lands in runs/<run>/ on terminal pulls (R25); read-only URI
    mode, every failure tolerated — the mirror is corroborating evidence,
    never a hard requirement.
    """
    path = os.path.join(contract.local_run_dir(m.run_name, base_dir),
                        transport.LOSS_DB_FILE)
    if not os.path.exists(path):
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT MAX(step) FROM metrics").fetchone()
        finally:
            con.close()
        return int(row[0]) if row and row[0] is not None else None
    except Exception:  # noqa: BLE001 — any unreadable mirror is just absent
        return None


def verify_artifacts(m: RunManifest, *, base_dir: str = ".") -> VerificationResult:
    """Verify locally pulled artifacts against manifest expectations (R9).

    observed_max_step = max(last pulled checkpoint watermark, steps parsed
    from the local checkpoint inventory, MAX(step) from the mirrored loss
    db). Expected checkpoint count is floor(observed / save_every) so
    early-stopped runs verify cleanly; ONLY when the run state is COMPLETED
    does the full-run formula floor(total_steps / save_every) apply, plus the
    final no-suffix safetensors requirement.
    """
    job_name = m.job_name or m.run_name
    out_dir = contract.local_output_dir(m.run_name, base_dir)
    found_steps = []
    final_present = False
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            step = contract.parse_checkpoint_filename(name, job_name)
            if step is not None:
                found_steps.append(step)
            elif contract.is_final_checkpoint(name, job_name):
                final_present = True
    found_steps.sort()

    candidates = [m.last_pulled_checkpoint_step or 0] + found_steps
    mirror_step = _loss_mirror_max_step(m, base_dir)
    if mirror_step is not None:
        candidates.append(mirror_step)
    observed = max(candidates)

    completed = m.state == contract.RunState.COMPLETED.value
    if completed and m.total_steps and m.save_every:
        expected = int(m.total_steps) // int(m.save_every)
    else:
        expected = m.expected_checkpoint_count(observed)

    missing = []
    have = set(found_steps)
    if m.save_every:
        for i in range(1, expected + 1):
            step = i * int(m.save_every)
            if step not in have:
                missing.append(f"{job_name}_{step:09d}.safetensors")
    if completed and not final_present:
        missing.append(f"{job_name}.safetensors")

    return VerificationResult(ok=not missing, missing=missing,
                              observed_max_step=observed, expected_count=expected,
                              found_steps=found_steps, final_present=final_present)


def _print_missing(run_name: str, v: VerificationResult, *, prefix: str):
    print(f"[{prefix}] {run_name}: verification FAILED — "
          f"{len(v.missing)} expected artifact(s) missing locally "
          f"(observed max step {v.observed_max_step}, "
          f"expected {v.expected_count} checkpoint(s), "
          f"found {len(v.found_steps)}):")
    for name in v.missing:
        print(f"[{prefix}]   missing: {name}")


# ---------------------------------------------------------------------------
# stop (R9 / KTD "kill the trainer, not the wrapper")
# ---------------------------------------------------------------------------

def stop_run(run_name: str, *, base_dir: str = ".", sdk=None,
             runner=subprocess.run):
    """pkill the run.py trainer over ssh — NEVER the tmux session.

    The wrapper must survive to write exit_code, wait for the pulled.ok ack,
    and self-stop; killing the session would orphan all three. The kill uses
    the same '[r]un.py.*<run>' bracket pattern launch.py uses, records
    STOPPED, then always flows into a pull (monitor.pull_once), tolerating
    the settle window before the sentinel lands. Returns the StatusReport.
    """
    m = manifest.load(run_name, base_dir)
    if not m.pod_id:
        raise LifecycleError(
            f"run '{run_name}' has no pod recorded in its manifest — "
            "nothing to stop")
    info = pod.get_pod_info(m.pod_id, sdk=sdk)
    ep = _endpoint(info)
    if info.status != "RUNNING" or ep is None:
        raise LifecycleError(
            f"cannot stop run '{run_name}': pod {m.pod_id} is {info.status} "
            "with no SSH path — there is no reachable trainer to kill. Use "
            "`down` to tear down, or `rescue` if the pod is stopped.")

    # Kill the trainer PROCESS only. The [r] bracket keeps the pattern from
    # matching the ssh-side shell that carries it. No tmux kill-session, ever.
    pattern = contract.trainer_pkill_pattern(run_name)
    res = transport.ssh_run(ep, f"pkill -f {contract.shell_quote(pattern)}",
                            runner=runner)
    if res.returncode not in (0, 1):  # 1 = nothing matched (already exited)
        raise LifecycleError(
            f"pkill over ssh failed (exit {res.returncode}): "
            f"{(res.stderr or '').strip()[:500]}")
    if res.returncode == 1:
        print(f"[stop] {run_name}: no trainer process matched — "
              "already exited; pulling anyway")

    # Record the stop BEFORE the pull: monitor maps a nonzero exit_code to
    # STOPPED (not CRASHED) only when the manifest recorded the stop.
    m.state = contract.RunState.STOPPED.value
    manifest.save(m, base_dir)

    # Always flow into a pull (sentinels + any final artifacts).
    report, _pull = monitor.pull_once(run_name, base_dir=base_dir, sdk=sdk,
                                      runner=runner)

    # Settle window: if the wrapper has not written exit_code yet, the pull
    # resolves a live-looking state — hold STOPPED so the eventual sentinel
    # read still maps to STOPPED.
    m = manifest.load(run_name, base_dir)
    # SETTLE_STATES: the wrapper may not have written exit_code yet.
    if m.state in contract.SETTLE_STATES:
        m.state = contract.RunState.STOPPED.value
        manifest.save(m, base_dir)
        report.state = contract.RunState.STOPPED.value
        note = ("stop requested; exit_code sentinel not yet written "
                "(settle window) — state held at STOPPED")
        report.detail = f"{report.detail}; {note}" if report.detail else note
    return report


# ---------------------------------------------------------------------------
# down (R9)
# ---------------------------------------------------------------------------

def down_run(run_name: str, *, force: bool = False, base_dir: str = ".",
             sdk=None, runner=subprocess.run, now: float = None) -> int:
    """Final pull → verify → ack → terminate → cost report (R9).

    See the module docstring for the full decision table. Returns DOWN_OK or
    DOWN_VERIFY_FAILED.
    """
    now = time.time() if now is None else float(now)
    m = manifest.load(run_name, base_dir)
    info = pod.get_pod_info(m.pod_id, sdk=sdk) if m.pod_id else None
    gone = info is None or info.status == "GONE"
    ep = _endpoint(info) if not gone else None
    reachable = (not gone) and info.status == "RUNNING" and ep is not None

    # Final incremental pull (mirrors sentinels too, R25) when reachable.
    if reachable:
        transport.pull_artifacts(ep, m, base_dir=base_dir, runner=runner)
        manifest.save(m, base_dir)

    v = verify_artifacts(m, base_dir=base_dir)
    if not v.ok:
        _print_missing(run_name, v, prefix="down")
        if gone:
            print(f"[down] {run_name}: pod is already gone — nothing left to "
                  "preserve; proceeding with local teardown")
        elif force:
            print(f"[down] {run_name}: force=True — terminating DESPITE "
                  f"{len(v.missing)} missing artifact(s)")
        else:
            if info.status == "RUNNING":
                pod.stop_pod(m.pod_id, sdk=sdk)
                print(f"[down] {run_name}: pod {m.pod_id} STOPPED (not "
                      "terminated) — artifacts preserved on its volume. "
                      "Use `rescue` to retrieve them, or re-run `down "
                      "--force` to terminate anyway.")
            else:
                print(f"[down] {run_name}: pod {m.pod_id} is already "
                      f"{info.status} — artifacts remain on its volume. Use "
                      "`rescue` to retrieve them, or `down --force` to "
                      "terminate anyway.")
            return DOWN_VERIFY_FAILED

    # Ack: write pulled.ok remotely WITH CONTENT (never a bare touch — the
    # presence convention is "readable with non-empty content"). Only after a
    # passed verification, and only when the pod is reachable.
    if v.ok and reachable:
        ack = contract.remote_sentinel_path(run_name, contract.PULLED_OK_FILE)
        res = transport.ssh_run(
            ep, f"echo {int(now)} > {contract.shell_quote(ack)}", runner=runner)
        if res.returncode != 0:
            print(f"[down] {run_name}: warning: pulled.ok ack write failed "
                  "(pod is being terminated next; the wrapper's max-grace "
                  "deadline covers this)")

    if not gone:
        pod.terminate_pod(m.pod_id, sdk=sdk)  # idempotent on a vanishing pod

    if m.terminated_at is None:
        m.terminated_at = now
    m.state = contract.RunState.TERMINATED.value
    manifest.save(m, base_dir)

    cost = m.estimated_cost()
    cost_s = f"${cost:.2f}" if cost is not None else "unknown (no rate recorded)"
    verified_s = (f"{len(v.found_steps)} checkpoint(s) verified" if v.ok
                  else f"{len(v.missing)} artifact(s) missing")
    pod_s = "pod already gone" if gone else f"pod {m.pod_id} terminated"
    print(f"[down] {run_name}: {pod_s}; {verified_s}; estimated cost {cost_s}")
    return DOWN_OK


# ---------------------------------------------------------------------------
# rescue (R24)
# ---------------------------------------------------------------------------

def rescue_run(run_name: str, *, base_dir: str = ".", sdk=None,
               runner=subprocess.run, now: float = None,
               wait_timeout_s: float = RESCUE_ENDPOINT_TIMEOUT_S,
               poll_s: float = RESCUE_ENDPOINT_POLL_S,
               _sleep=time.sleep) -> int:
    """Retrieve artifacts from a STOPPED pod: zero-GPU start → re-resolve
    endpoint → pull + mirror → verify → terminate (R24).

    The documented recovery for both pull-failure stops and post-self-stop
    retrieval — stopped pods keep billing volume storage, so terminate is the
    expected endgame. Refuses on a RUNNING pod (use stop/down) and on a GONE
    pod (nothing to rescue). Returns RESCUE_OK or RESCUE_VERIFY_FAILED.
    """
    now = time.time() if now is None else float(now)
    m = manifest.load(run_name, base_dir)
    if not m.pod_id:
        raise LifecycleError(
            f"run '{run_name}' has no pod recorded in its manifest — "
            "nothing to rescue")
    info = pod.get_pod_info(m.pod_id, sdk=sdk)
    if info.status == "GONE":
        v = verify_artifacts(m, base_dir=base_dir)
        raise LifecycleError(
            f"nothing to rescue for run '{run_name}': pod {m.pod_id} is gone "
            "from the RunPod API (its volume no longer exists). Local state: "
            f"manifest says {m.state}; {len(v.found_steps)} checkpoint(s) "
            f"pulled (observed max step {v.observed_max_step}, expected "
            f"{v.expected_count}); local mirrors under runs/{run_name}/ are "
            "all that remain.")
    if info.status == "RUNNING":
        raise LifecycleError(
            f"refusing to rescue run '{run_name}': pod {m.pod_id} is RUNNING "
            "— rescue is for stopped pods only. Use `stop` to end training "
            "or `down` to pull and terminate.")

    # Stopped pod: start CPU-only, then RE-RESOLVE the endpoint via the API —
    # the manifest endpoint is cache, and RunPod reassigns IP/port across
    # stop/start.
    info = pod.start_zero_gpu(m.pod_id, sdk=sdk)
    elapsed = 0.0
    while not (info.ssh_host and info.ssh_port):
        if elapsed >= wait_timeout_s:
            raise LifecycleError(
                f"pod {m.pod_id} did not expose an SSH endpoint within "
                f"{int(wait_timeout_s)}s of the zero-GPU start — it is left "
                "running CPU-only (billing storage + CPU); retry rescue or "
                "stop it from the RunPod console.")
        _sleep(poll_s)
        elapsed += poll_s
        info = pod.get_pod_info(m.pod_id, sdk=sdk)
    ep = Endpoint(host=info.ssh_host, port=int(info.ssh_port))
    m.ssh_host, m.ssh_port = info.ssh_host, int(info.ssh_port)

    # Pull + mirror (sentinels and, on terminal states, loss_log.db — R25).
    transport.pull_artifacts(ep, m, base_dir=base_dir, runner=runner)
    manifest.save(m, base_dir)

    v = verify_artifacts(m, base_dir=base_dir)
    if not v.ok:
        _print_missing(run_name, v, prefix="rescue")
        pod.stop_pod(m.pod_id, sdk=sdk)
        print(f"[rescue] {run_name}: pod {m.pod_id} re-STOPPED (not "
              "terminated) — artifacts preserved on its volume; inspect, "
              "then retry rescue or use `down --force`.")
        return RESCUE_VERIFY_FAILED

    pod.terminate_pod(m.pod_id, sdk=sdk)
    if m.terminated_at is None:
        m.terminated_at = now
    m.state = contract.RunState.TERMINATED.value
    manifest.save(m, base_dir)

    cost = m.estimated_cost()
    cost_s = f"${cost:.2f}" if cost is not None else "unknown (no rate recorded)"
    print(f"[rescue] {run_name}: pod {m.pod_id} terminated; "
          f"{len(v.found_steps)} checkpoint(s) verified; "
          f"estimated cost {cost_s}")
    return RESCUE_OK
