"""AI Toolkit manager CLI.

Runs with any Python >= 3.8 and no dependencies, so it works before the
training environment exists. This is the single entry point every installer
frontend (shell scripts, the desktop launcher, the web UI) shells out to.

    python3 -m manager install          first-time environment setup
    python3 -m manager check [--json]   is an update / dep sync needed?
    python3 -m manager update           git pull + dependency sync + migrations
    python3 -m manager sync             dependency sync only (no git pull)
    python3 -m manager launch           start the web UI
    python3 -m manager detect [--json]  show detected hardware
    python3 -m manager doctor           full environment diagnostics
"""

import argparse
import os
import subprocess
import sys

# allow `python manager/__main__.py` as well as `python -m manager`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manager import detect as detect_mod
from manager import env, gitops, launch, spec as spec_mod, util
from manager.util import die, info, ok, print_json, warn


def _resolve_spec(args):
    detection = detect_mod.detect()
    try:
        return detection, spec_mod.build_spec(
            detection, allow_cpu=getattr(args, "cpu", False)
        )
    except RuntimeError as e:
        die(str(e))


def cmd_detect(args):
    detection = detect_mod.detect()
    try:
        s = spec_mod.build_spec(detection, allow_cpu=True)
        detection["spec"] = s.as_dict()
    except RuntimeError as e:
        detection["spec_error"] = str(e)
    if args.json:
        print_json(detection)
    else:
        backend = detection.get("spec", {}).get("backend", "unknown")
        info("os=%s arch=%s backend=%s" % (detection["os"], detection["arch"], backend))
        if detection["nvidia"]:
            for gpu in detection["nvidia"]["gpus"]:
                info("gpu: %s (%s)" % (gpu["name"], gpu["memory"]))


def cmd_install(args):
    detection, s = _resolve_spec(args)
    env.sync(s, detection, dry_run=args.dry_run, force=args.force)
    if not args.dry_run:
        ok("Install complete. Start the UI with: python3 -m manager launch")


def cmd_sync(args):
    detection, s = _resolve_spec(args)
    env.sync(s, detection, dry_run=args.dry_run, force=args.force)


def cmd_check(args):
    _, s = _resolve_spec(args)
    fetched = gitops.fetch()
    behind = gitops.behind_count()
    data = {
        "version": _toolkit_version(),
        "branch": gitops.current_branch(),
        "commit": gitops.current_commit(),
        "remote_commit": gitops.remote_commit(),
        "dirty": gitops.is_dirty(),
        "fetch_ok": fetched,
        "behind": behind,
        "incoming": gitops.incoming_log(),
        "venv": env.venv_exists(),
        "deps_in_sync": env.venv_exists()
        and env.torch_matches(s)
        and env.requirements_in_sync(s),
        "backend": s.backend,
    }
    data["update_available"] = bool(behind) or not data["deps_in_sync"]
    if args.json:
        print_json(data)
        return
    info("AI Toolkit %s (%s @ %s)" % (data["version"], data["branch"], data["commit"]))
    if not fetched:
        warn("Could not reach the remote (offline?) — update status may be stale.")
    if behind:
        info("Update available: %d new commit(s)." % behind)
        for line in data["incoming"]:
            print("    " + line)
    elif behind == 0:
        ok("Code is up to date.")
    if not data["deps_in_sync"]:
        warn("Dependencies are out of sync. Run: python3 -m manager sync")
    elif behind == 0:
        ok("Dependencies are in sync.")


def cmd_update(args):
    """git pull (never destructive) + dependency sync.

    Local work is sacred: a dirty tree either aborts (default), or with
    --auto is skipped with a warning so run scripts can continue to launch.
    We never reset/clean; even a forced pull is --ff-only, which git itself
    aborts rather than overwriting local changes.
    """
    auto = getattr(args, "auto", False)
    skip_pull = False

    if gitops.is_dirty() and not args.force:
        if auto:
            warn(
                "Local changes detected — skipping the code update to protect "
                "your work. Commit or stash your changes to receive updates."
            )
            skip_pull = True
        else:
            die(
                "You have local changes to tracked files. Commit or stash them, "
                "or re-run with --force to attempt the update anyway (git will "
                "still refuse rather than overwrite your changes)."
            )

    if not skip_pull and not gitops.fetch():
        if auto:
            warn("Could not reach the git remote — skipping the update check.")
            skip_pull = True
        else:
            die("Could not reach the git remote. Check your network and try again.")

    if not skip_pull:
        behind = gitops.behind_count()
        if behind is None:
            warn("Current branch has no upstream; skipping git pull.")
        elif behind == 0:
            ok("Code already up to date.")
        else:
            info("Pulling %d new commit(s)..." % behind)
            gitops.pull_ff()
            ok("Code updated to %s." % gitops.current_commit())
            # Re-exec so the freshly pulled manager code runs its own dependency
            # sync and migrations (the in-memory copy of this module is stale now).
            cmd = [sys.executable, "-m", "manager", "sync"]
            if args.dry_run:
                cmd.append("--dry-run")
            sys.exit(subprocess.call(cmd, cwd=util.REPO_ROOT))
    # nothing was pulled — safe to sync with the code already loaded
    detection, s = _resolve_spec(args)
    env.sync(s, detection, dry_run=args.dry_run)


def cmd_launch(args):
    sys.exit(launch.launch_ui(open_browser=not args.no_browser))


def cmd_doctor(args):
    from manager import doctor

    doctor.run_doctor()


def cmd_version(args):
    print(_toolkit_version())


def _toolkit_version():
    version = {}
    try:
        with open(os.path.join(util.REPO_ROOT, "version.py")) as f:
            exec(f.read(), version)
        return version.get("VERSION", "unknown")
    except OSError:
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="manager", description="AI Toolkit install / update manager"
    )
    sub = parser.add_subparsers(dest="command")

    def add(name, fn, **kwargs):
        p = sub.add_parser(name, **kwargs)
        p.set_defaults(fn=fn)
        return p

    p = add("detect", cmd_detect, help="show detected hardware and env spec")
    p.add_argument("--json", action="store_true")

    for name, fn, help_text in (
        ("install", cmd_install, "first-time environment setup"),
        ("sync", cmd_sync, "sync dependencies for the current checkout"),
    ):
        p = add(name, fn, help=help_text)
        p.add_argument("--cpu", action="store_true", help="allow CPU-only install")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument(
            "--force",
            action="store_true",
            help="reinstall requirements even if in sync",
        )

    p = add("check", cmd_check, help="check for updates (use --json for machines)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)

    p = add("update", cmd_update, help="git pull + dependency sync + migrations")
    p.add_argument("--cpu", action="store_true", help="allow CPU-only install")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force", action="store_true", help="update even with local changes"
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="unattended mode (run scripts): on local changes or an unreachable "
        "remote, warn and skip the code update instead of failing; deps still sync",
    )

    p = add("launch", cmd_launch, help="start the web UI")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser when the UI is ready",
    )
    add("doctor", cmd_doctor, help="diagnose the environment")
    add("version", cmd_version, help="print the toolkit version")

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    util.set_json_mode(bool(getattr(args, "json", False)))
    try:
        args.fn(args)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
