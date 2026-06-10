#!/usr/bin/env python3
"""CLI entrypoint for the remote GPU training pipeline (U8).

Usage:
    python scripts/remote/cli.py <subcommand> [...]
    python -m scripts.remote.cli <subcommand> [...]

Thirteen subcommands wired to the scripts.remote.* modules:
    preflight  provision  sync  launch  up
    status     pull       watch stop    down  attach  rescue
    mark-reviewed

This module is THIN: argument parsing + wiring only. Business logic lives in
preflight/pod/transport/launch/monitor/lifecycle; cross-unit strings live in
contract.py. Exit codes: `watch` returns monitor's distinct codes (see
contract.WATCH_EXIT_CODES and README); every other command exits 0/1.

Secrets come from the environment / .env (R21): RUNPOD_API_KEY (laptop-side
control plane), RUNPOD_STOP_API_KEY (restricted pod-side self-stop key),
HF_TOKEN (forwarded to the pod). See scripts/remote/README.md, Setup.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.remote import contract, lifecycle, manifest, monitor, pod, transport
from scripts.remote import launch as launch_mod
from scripts.remote import preflight as preflight_mod
from scripts.remote.transport import Endpoint

README_SETUP = "scripts/remote/README.md (section: Setup — RunPod API key)"
INSTALL_HINT = "pip install -r scripts/remote/requirements.txt"


class CliError(RuntimeError):
    pass


def _warn(message: str):
    print(f"[cli] WARNING: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Environment: .env loading + actionable requirement checks
# ---------------------------------------------------------------------------

def _parse_env_file(path: str) -> dict:
    """Tiny .env parser used when python-dotenv is not importable."""
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def load_env(base_dir: str = ".") -> None:
    """Load .env without ever overriding the real environment.

    python-dotenv when importable (the trainer venv already depends on it);
    otherwise a tiny local parser — the CLI never hard-requires dotenv.
    """
    path = os.path.join(base_dir, ".env")
    try:
        from dotenv import load_dotenv
        load_dotenv(path if os.path.exists(path) else None)
        return
    except ImportError:
        pass
    if not os.path.exists(path):
        return
    for key, value in _parse_env_file(path).items():
        os.environ.setdefault(key, value)


def require_runpod_sdk() -> None:
    try:
        import runpod  # noqa: F401
    except ImportError:
        raise CliError(
            "the 'runpod' SDK is not installed in this environment.\n"
            f"Install the laptop-only dependencies with: {INSTALL_HINT}\n"
            "(these are never merged into the repo's training requirements)"
        ) from None


def require_api_key() -> None:
    if not os.environ.get("RUNPOD_API_KEY"):
        raise CliError(
            "RUNPOD_API_KEY is not set.\n"
            "Create an API key at runpod.io (Console -> Settings -> API Keys "
            "-> '+ API Key'), then add `RUNPOD_API_KEY=...` to the repo .env "
            f"file. Step-by-step instructions: {README_SETUP}"
        )


def check_rsync() -> None:
    """macOS openrsync detection: warn (only) recommending GNU rsync 3.

    transport already restricts itself to portable flags, so this never
    blocks — GNU rsync is just faster and better-behaved on incremental pulls.
    """
    try:
        res = subprocess.run(["rsync", "--version"], capture_output=True,
                             text=True, timeout=10)
        out = res.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        _warn("could not run `rsync --version` — is rsync installed?")
        return
    if "version 3" not in out:
        _warn(
            "rsync does not look like GNU rsync 3.x (macOS ships openrsync). "
            "Transport sticks to portable flags so this still works, but "
            "`brew install rsync` is recommended."
        )


# ---------------------------------------------------------------------------
# Shared wiring helpers
# ---------------------------------------------------------------------------

def _load_manifest(run_name: str, base_dir: str):
    try:
        return manifest.load(run_name, base_dir)
    except manifest.ManifestNotFoundError as e:
        raise CliError(f"{e}\nRun `preflight <config> --run-name {run_name}` "
                       "(or `up`) first.") from e


def _resolve_endpoint(m, *, base_dir: str):
    """Always via pod.get_pod_info — the manifest endpoint is cache (KTD)."""
    if not m.pod_id:
        raise CliError(
            f"run '{m.run_name}' has no pod recorded in its manifest — "
            "run `provision` (or `up`) first")
    info = pod.get_pod_info(m.pod_id)
    if info.status != "RUNNING" or not (info.ssh_host and info.ssh_port):
        raise CliError(
            f"pod {m.pod_id} is {info.status} with no direct SSH endpoint — "
            "cannot reach it. Use `status` for state, `rescue` if it is "
            "stopped, or `provision` for a new pod.")
    m.ssh_host, m.ssh_port = info.ssh_host, int(info.ssh_port)
    manifest.save(m, base_dir)
    return info, Endpoint(host=info.ssh_host, port=int(info.ssh_port))


def _ssh_check(ep: Endpoint, command: str, what: str) -> None:
    res = transport.ssh_run(ep, command)
    if res.returncode != 0:
        raise CliError(f"{what} failed (exit {res.returncode}): "
                       f"{(res.stderr or '').strip()[:500]}")


def _upload_dir(ep: Endpoint, local_dir: str, remote_dir: str) -> None:
    """Upload one directory's contents to an exact remote dir (R5 filters)."""
    _ssh_check(ep, f"mkdir -p {contract.shell_quote(remote_dir)}",
               f"mkdir {remote_dir}")
    cmd = transport.build_rsync_up(local_dir.rstrip("/") + "/", ep,
                                   remote_dir.rstrip("/") + "/")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise CliError(f"upload of {local_dir} failed (exit {res.returncode}): "
                       f"{(res.stderr or '').strip()[:500]}")


def _upload_self_stop(ep: Endpoint, run_name: str) -> str:
    """Upload pod.py's self-stop tool (provider strings live in pod.py)."""
    remote = f"{contract.remote_run_root(run_name)}/{contract.SELF_STOP_SCRIPT}"
    fd, tmp = tempfile.mkstemp(prefix=".aitk-selfstop-", suffix=".sh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(pod.self_stop_script())
        transport.upload_file(ep, tmp, remote)
    finally:
        os.unlink(tmp)
    _ssh_check(ep, f"chmod +x {contract.shell_quote(remote)}", f"chmod {remote}")
    return remote


def _print_report(report) -> None:
    step = f"{report.step}/{report.total_steps}" if report.step is not None else "-"
    loss = f"{report.recent_loss:.4f}" if report.recent_loss is not None else "-"
    cost = f"${report.cost_estimate:.2f}" if report.cost_estimate is not None else "-"
    print(f"[{report.run_name}] state={report.state} step={step} "
          f"loss={loss} cost~{cost}")
    if report.oom_skips:
        print(f"  oom_skips={report.oom_skips} (DEGRADED health signal)")
    if report.disk_used_pct is not None:
        flag = " ** WARNING **" if report.disk_warning else ""
        print(f"  /workspace disk: {report.disk_used_pct}% used{flag}")
    if report.drift:
        print("  CONFIG DRIFT: local derived config no longer matches the "
              "hash this run launched with")
    if report.reviewable:
        print(f"  reviewable sample steps: {report.reviewable}")
    if report.detail:
        print(f"  detail: {report.detail}")


def _print_pull(pull) -> None:
    if pull is None:
        print("  pull: pod not reachable — no transfer this cycle")
        return
    print(f"  pulled {pull.pulled_files} file(s); "
          f"new sample steps {pull.new_sample_steps or []}; "
          f"new checkpoints {pull.new_checkpoint_steps or []}")
    if pull.partial_steps:
        print(f"  partial sample batches (old mtime, pulled with warning): "
              f"{pull.partial_steps}")
    if pull.skipped_unstable:
        print(f"  skipped this cycle (unstable mtime/size): "
              f"{len(pull.skipped_unstable)} file(s)")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_preflight(args) -> int:
    result = preflight_mod.run_preflight(
        args.config, run_name=args.run_name, base_dir=args.base_dir,
        allow_uncaptioned=args.allow_uncaptioned)
    print(f"[preflight] OK run='{result.run_name}' job='{result.job_name}'")
    print(f"  derived config: {result.derived_config_path}")
    print(f"  config hash:    {result.config_hash}")
    for report in result.dataset_reports:
        print(f"  dataset {report.folder}: {report.image_count} image(s), "
              f"{report.caption_count} caption(s), "
              f"{report.total_bytes / 1e6:.1f} MB to upload")
    print(f"  upload set: {len(result.upload_set)} path(s)")
    for change in result.changes:
        print(f"  remap: {change}")
    for warning in result.warnings:
        _warn(warning)
    return 0


def _provision(run_name: str, args, base_dir: str):
    """Shared by `provision` and `up`. Returns (pod_id|None, hint_printed)."""
    m = _load_manifest(run_name, base_dir)
    # Double-provision guard (#3): a manifest that already records a pod
    # refuses a second one unless that pod is GONE or --force-new is passed.
    if m.pod_id and not getattr(args, "force_new", False):
        info = pod.get_pod_info(m.pod_id)
        if info.status != "GONE":
            raise CliError(
                f"run '{run_name}' already has pod {m.pod_id} "
                f"({info.status}) recorded in its manifest — refusing to "
                "provision a second pod that would bill in parallel. Use "
                "`status` to inspect it, `down`/`rescue` to tear it down, "
                "or pass --force-new to provision anyway.")
    disk_gb = args.disk_gb
    if disk_gb is None:
        if not (m.total_steps and m.save_every):
            raise CliError(
                "cannot derive --disk-gb: the manifest has no total_steps/"
                "save_every (re-run preflight) — or pass --disk-gb explicitly")
        disk_gb = contract.disk_size_gb(m.total_steps, m.save_every)
    gpu_fallback = tuple(args.gpu_fallback or ())
    info = pod.create_pod(run_name, gpu_type=args.gpu, image_tag=args.image,
                          disk_gb=disk_gb, gpu_fallback=gpu_fallback,
                          dry_run=args.dry_run)
    if args.dry_run:
        print(f"[provision] dry run for '{run_name}' "
              f"(disk {disk_gb} GB) — no pod created")
        return None, disk_gb
    m.pod_id = info.pod_id
    m.image_tag = (args.image or os.environ.get("AITK_REMOTE_IMAGE")
                   or pod.DEFAULT_IMAGE_TAG)
    m.gpu_requested = pod.resolve_gpu_type(args.gpu)
    m.disk_gb = int(disk_gb)
    m.provisioned_at = time.time()
    m.state = contract.RunState.PROVISIONING.value
    manifest.save(m, base_dir)
    try:
        info = pod.wait_for_ready(info.pod_id, timeout_s=getattr(args, 'ready_timeout', 2400))
    except Exception:
        m.state = contract.RunState.PROVISION_FAILED.value
        manifest.save(m, base_dir)
        raise
    m.gpu_provisioned = info.gpu_type
    m.hourly_rate = info.hourly_rate
    m.ssh_host, m.ssh_port = info.ssh_host, int(info.ssh_port)
    m.state = contract.RunState.POD_READY.value
    manifest.save(m, base_dir)
    rate = f"${info.hourly_rate:.2f}/hr" if info.hourly_rate else "rate unknown"
    print(f"[provision] pod {info.pod_id} ready: {info.gpu_type} ({rate}), "
          f"disk {disk_gb} GB, ssh root@{info.ssh_host} -p {info.ssh_port}")
    return info.pod_id, disk_gb


def cmd_provision(args) -> int:
    _provision(args.run, args, args.base_dir)
    return 0


def _sync(result, args, base_dir: str) -> None:
    """Shared by `sync` and `up`: uploads against a fresh PreflightResult."""
    run_name = result.run_name
    m = _load_manifest(run_name, base_dir)
    _info, ep = _resolve_endpoint(m, base_dir=base_dir)
    for local, remote in result.upload_set:
        if os.path.isdir(local):
            _upload_dir(ep, local, remote)
        else:
            transport.upload_file(ep, local, remote)
        print(f"[sync] uploaded {local} -> {remote}")
    transport.upload_file(ep, result.derived_config_path,
                          contract.remote_config_path(run_name))
    print(f"[sync] uploaded derived config -> "
          f"{contract.remote_config_path(run_name)}")
    _upload_self_stop(ep, run_name)
    facts = transport.upload_overlay(ep, _REPO_ROOT)
    m = _load_manifest(run_name, base_dir)
    m.overlay_git_sha = facts.get("overlay_git_sha")
    m.overlay_dirty_hash = facts.get("overlay_dirty_hash")
    m.image_repo_commit = facts.get("image_repo_commit")
    if m.state not in contract.LIVE_STATES:
        m.state = contract.RunState.SYNCED.value
    manifest.save(m, base_dir)
    print(f"[sync] overlay uploaded (local {facts.get('overlay_git_sha')}, "
          f"image repo {facts.get('image_repo_commit')})")


def cmd_sync(args) -> int:
    check_rsync()
    run_name = args.run_name
    if run_name is None:
        # peek at config.name so the drift pre-check can read the manifest
        cfg = preflight_mod.load_source_config(args.config, args.base_dir)
        run_name = (cfg.get("config") or {}).get("name")
    old = None
    if run_name:
        try:
            old = manifest.load(run_name, args.base_dir)
        except manifest.ManifestNotFoundError:
            pass
    live = (old is not None and old.state in contract.LIVE_STATES
            and bool(old.pod_id))
    result = preflight_mod.run_preflight(
        args.config, run_name=run_name, base_dir=args.base_dir,
        allow_uncaptioned=args.allow_uncaptioned)
    if live:
        if old.config_hash and result.config_hash != old.config_hash:
            _warn(f"CONFIG DRIFT: run '{result.run_name}' is {old.state} on "
                  "the pod with a DIFFERENT config than this sync just "
                  "derived — the running trainer keeps its old config; the "
                  "new config applies at the next launch")
        # R19/#19: re-preflight just overwrote the manifest with the NEW
        # config's facts, but the RUNNING training still follows the
        # launch-time ones — restore them so status/verify stay truthful.
        # The new derived file stays on disk for the next launch.
        m = _load_manifest(result.run_name, args.base_dir)
        for name in ("state", "job_name", "total_steps", "save_every",
                     "sample_every", "prompt_count", "config_hash"):
            setattr(m, name, getattr(old, name))
        manifest.save(m, args.base_dir)
    _sync(result, args, args.base_dir)
    return 0


def _launch(run_name: str, args, base_dir: str) -> int:
    m = _load_manifest(run_name, base_dir)
    _info, ep = _resolve_endpoint(m, base_dir=base_dir)
    result = launch_mod.launch_run(
        run_name, ep=ep, resume=args.resume, fresh=args.fresh,
        max_hours=args.max_hours, max_grace_seconds=args.max_grace,
        base_dir=base_dir)
    print(f"[launch] {run_name}: {result.outcome}")
    if result.outcome == "crashed":
        print(result.detail)
        print("[launch] pod left up for diagnosis — "
              f"`attach {run_name}` / ssh + tmux to inspect, `down` to tear down")
        return 1
    if result.outcome == "warming":
        print("[launch] model download/load in progress (cold starts take "
              "20-60 min before step 1) — use `watch` to follow")
    elif result.outcome == "silent":
        _warn("no recognizable trainer output within the early-tail window; "
              "pod left up — inspect with `status` / `attach`. Raw tail:")
        print(result.detail)
    return 0


def cmd_launch(args) -> int:
    return _launch(args.run, args, args.base_dir)


def cmd_up(args) -> int:
    """preflight -> provision -> sync -> launch; teardown on post-provision
    failure so a half-set-up pod never sits billing."""
    check_rsync()
    result = preflight_mod.run_preflight(
        args.config, run_name=args.run_name, base_dir=args.base_dir,
        allow_uncaptioned=args.allow_uncaptioned)
    run_name = result.run_name
    print(f"[up] preflight OK for '{run_name}'")
    pod_id, _disk = _provision(run_name, args, args.base_dir)
    if args.dry_run:
        print("[up] dry run stops after the provision request preview")
        return 0
    try:
        _sync(result, args, args.base_dir)
        return _launch(run_name, args, args.base_dir)
    except Exception as e:
        _warn(f"`up` failed after the pod was provisioned: {e}")
        try:
            pod.terminate_pod(pod_id)
            m = _load_manifest(run_name, args.base_dir)
            m.state = contract.RunState.TERMINATED.value
            m.terminated_at = time.time()
            manifest.save(m, args.base_dir)
            print(f"[up] pod {pod_id} TERMINATED so it does not keep billing. "
                  "Fix the error above and re-run `up`.")
        except Exception as cleanup_err:  # noqa: BLE001
            _warn(f"TEARDOWN FAILED ({cleanup_err}) — terminate pod {pod_id} "
                  "manually in the RunPod console NOW; it is still billing")
        return 1


def cmd_status(args) -> int:
    report = monitor.get_status(args.run, base_dir=args.base_dir)
    _print_report(report)
    return 0


def cmd_pull(args) -> int:
    check_rsync()
    report, pull = monitor.pull_once(args.run, base_dir=args.base_dir)
    _print_report(report)
    _print_pull(pull)
    return 0


def cmd_watch(args) -> int:
    check_rsync()
    return monitor.watch(args.run, interval_s=args.interval, once=args.once,
                         json_out=args.json, base_dir=args.base_dir)


def cmd_stop(args) -> int:
    report = lifecycle.stop_run(args.run, base_dir=args.base_dir)
    _print_report(report)
    return 0


def cmd_down(args) -> int:
    check_rsync()
    return lifecycle.down_run(args.run, force=args.force,
                              base_dir=args.base_dir)


def cmd_attach(args) -> int:
    report = monitor.attach(args.run, base_dir=args.base_dir)
    _print_report(report)
    return 0


def cmd_rescue(args) -> int:
    check_rsync()
    return lifecycle.rescue_run(args.run, base_dir=args.base_dir)


def cmd_mark_reviewed(args) -> int:
    """Advance the review watermark (#14) — without this, `watch --once`
    re-reports the same reviewable steps forever."""
    m = _load_manifest(args.run, args.base_dir)
    try:
        step = int(args.step)
    except (TypeError, ValueError):
        raise CliError(f"step must be an integer, got {args.step!r}") from None
    current = m.last_reviewed_step or 0
    if step < current:
        raise CliError(
            f"step {step} is below the current last_reviewed_step "
            f"({current}) — the review watermark only moves forward")
    m.last_reviewed_step = step
    manifest.save(m, args.base_dir)
    print(f"[mark-reviewed] {args.run}: last_reviewed_step={step}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# subcommands that talk to the RunPod API (preflight is the only local one)
_NEEDS_API = {"provision", "sync", "launch", "up", "status", "pull", "watch",
              "stop", "down", "attach", "rescue"}


def _add_config_args(p):
    p.add_argument("config", help="path to the source training config YAML")
    p.add_argument("--run-name", default=None,
                   help="run name (default: config.name; [A-Za-z0-9_-]{1,64})")
    p.add_argument("--allow-uncaptioned", action="store_true",
                   help="proceed even when images lack .txt caption sidecars")


def _add_provision_args(p):
    p.add_argument("--gpu", default=pod.DEFAULT_GPU,
                   help=f"GPU type or alias (default: {pod.DEFAULT_GPU})")
    p.add_argument("--gpu-fallback", action="append", default=[],
                   help="fallback GPU type tried when --gpu is out of stock "
                        "(repeatable, ordered)")
    p.add_argument("--image", default=None,
                   help="ostris/aitoolkit image tag (default: "
                        "$AITK_REMOTE_IMAGE or the pinned default)")
    p.add_argument("--disk-gb", type=int, default=None,
                   help="volume size; default derived from steps/save_every")
    p.add_argument("--dry-run", action="store_true",
                   help="print the create request without calling the API")
    p.add_argument("--force-new", action="store_true",
                   help="provision even when the manifest already records a "
                        "live pod (default: refuse unless that pod is GONE)")


def _add_launch_args(p):
    p.add_argument("--resume", action="store_true",
                   help="allow launch over existing remote checkpoints "
                        "(ai-toolkit auto-resumes from the newest)")
    p.add_argument("--fresh", action="store_true",
                   help="move existing remote output aside and start over")
    p.add_argument("--max-hours", type=float, default=launch_mod.DEFAULT_MAX_HOURS,
                   help="hard max-runtime backstop (default: %(default)s)")
    p.add_argument("--max-grace", type=int,
                   default=launch_mod.DEFAULT_MAX_GRACE_SECONDS,
                   help="post-training ack-wait bound in seconds before "
                        "self-stop (default: %(default)s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Remote GPU LoRA training on RunPod — agentic workflow "
                    "guide in scripts/remote/README.md")
    parser.add_argument("--base-dir", default=".",
                        help=argparse.SUPPRESS)  # test/fixture hook
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="validate config + dataset; write "
                       "the derived remote config (no pod, no network)")
    _add_config_args(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("provision", help="create a pod and wait until SSH-ready")
    p.add_argument("run", help="run name (preflight must have run)")
    _add_provision_args(p)
    p.set_defaults(func=cmd_provision)

    p = sub.add_parser("sync", help="upload dataset, ctrl images, derived "
                       "config, self-stop tool, and the repo overlay")
    _add_config_args(p)
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("launch", help="guarded, sentinel-wrapped training start")
    p.add_argument("run", help="run name")
    _add_launch_args(p)
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("up", help="preflight -> provision -> sync -> launch; "
                       "tears the pod down if a post-provision step fails")
    _add_config_args(p)
    _add_provision_args(p)
    _add_launch_args(p)
    p.set_defaults(func=cmd_up)

    p = sub.add_parser("status", help="state, step, loss, health, disk, cost")
    p.add_argument("run", help="run name")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("pull", help="one status + incremental artifact pull")
    p.add_argument("run", help="run name")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("watch", help="status + pull on an interval; exits "
                       "with distinct codes on terminal states (see README)")
    p.add_argument("run", help="run name")
    p.add_argument("--interval", type=float,
                   default=monitor.DEFAULT_WATCH_INTERVAL_S,
                   help="seconds between cycles (default: %(default)s)")
    p.add_argument("--once", action="store_true",
                   help="one cycle, then exit (the agent polling contract)")
    p.add_argument("--json", action="store_true",
                   help="emit one JSON object (use with --once)")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("stop", help="kill the trainer (sentinel path still "
                       "runs); always flows into a pull")
    p.add_argument("run", help="run name")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("down", help="final pull, verify, ack, terminate")
    p.add_argument("run", help="run name")
    p.add_argument("--force", action="store_true",
                   help="terminate even when artifact verification fails")
    p.set_defaults(func=cmd_down)

    p = sub.add_parser("attach", help="rebuild watcher state from the "
                       "manifest + API alone (fresh-session re-entry)")
    p.add_argument("run", help="run name")
    p.set_defaults(func=cmd_attach)

    p = sub.add_parser("rescue", help="zero-GPU start a stopped pod, pull, "
                       "verify, terminate")
    p.add_argument("run", help="run name")
    p.set_defaults(func=cmd_rescue)

    p = sub.add_parser("mark-reviewed", help="advance the review watermark "
                       "after reviewing samples (watch re-reports steps "
                       "until this is run)")
    p.add_argument("run", help="run name")
    p.add_argument("step", help="highest sample step just reviewed")
    p.set_defaults(func=cmd_mark_reviewed)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    load_env(args.base_dir)
    try:
        if args.command in _NEEDS_API:
            require_runpod_sdk()
            require_api_key()
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[cli] interrupted", file=sys.stderr)
        return 1
    except (CliError, preflight_mod.PreflightError, pod.PodError,
            launch_mod.LaunchError, lifecycle.LifecycleError,
            manifest.ManifestNotFoundError, RuntimeError,
            subprocess.TimeoutExpired) as e:
        # RuntimeError/TimeoutExpired cover transport failures (#17): exit 1
        # with the message instead of a traceback.
        print(f"[cli] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
