"""Environment diagnostics: `python -m manager doctor`."""

import os
import shutil
import subprocess
import sys

from . import detect as detect_mod
from . import env, ffmpeg, gitops, nodejs
from .util import REPO_ROOT, clean_env, find_uv, venv_dir, venv_python


def _check(label, passed, detail=""):
    if sys.stdout.isatty():
        mark = "\033[32mOK\033[0m " if passed else "\033[31mFAIL\033[0m"
    else:
        mark = "OK  " if passed else "FAIL"
    print("  [%s] %-18s %s" % (mark, label, detail))
    return passed


def run_doctor():
    print("AI Toolkit doctor\n")
    d = detect_mod.detect()

    from . import gitwin

    _check("os / arch", True, "%s %s" % (d["os"], d["arch"]))
    git = gitwin.find_git()
    _check(
        "git",
        git is not None,
        git or "not found (manager sync installs a local copy on Windows)",
    )
    uv = find_uv()
    _check("uv", True, uv or "not found (optional, recommended)")

    if d["nvidia"]:
        names = ", ".join(g["name"] for g in d["nvidia"]["gpus"])
        _check(
            "gpu",
            True,
            "%s (driver %s, CUDA %s)"
            % (names, d["nvidia"]["driver"], d["nvidia"]["cuda_version"]),
        )
    elif d["rocm"]:
        _check("gpu", True, "AMD ROCm (experimental)")
    elif d["backend"] == "mps":
        _check("gpu", True, "Apple Silicon (MPS)")
    else:
        _check("gpu", False, "no supported GPU detected")

    has_venv = env.venv_exists()
    _check("venv", has_venv, venv_dir() if has_venv else "not created yet")
    if has_venv:
        torch = env.installed_torch()
        _check("torch", torch is not None, torch or "not installed")
        if torch and d["backend"] == "cuda":
            try:
                out = subprocess.run(
                    [
                        venv_python(),
                        "-c",
                        "import torch; print(torch.cuda.is_available())",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                    env=clean_env(),
                )
                avail = out.stdout.decode().strip() == "True"
                _check(
                    "torch sees gpu",
                    avail,
                    "" if avail else "torch.cuda.is_available() is False",
                )
            except (OSError, subprocess.TimeoutExpired):
                _check("torch sees gpu", False, "could not query")

    node_exe, node_major = nodejs.have_usable_node()
    if node_exe:
        _check("node", True, "%s (v%s)" % (node_exe, node_major))
    else:
        _check(
            "node",
            False,
            "none >= %d found (manager sync installs a local copy)"
            % nodejs.MIN_NODE_MAJOR,
        )

    if os.path.isfile(ffmpeg.ffmpeg_exe()):
        # run it with the launch env so missing shared libs are caught
        from . import launch

        try:
            out = subprocess.run(
                [ffmpeg.ffmpeg_exe(), "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                env=launch.build_env(),
            )
            works = out.returncode == 0
            detail = (
                out.stdout.decode().splitlines()[0]
                if works
                else "installed but fails to run"
            )
        except (OSError, subprocess.TimeoutExpired):
            works, detail = False, "installed but fails to run"
        _check("ffmpeg (local)", works, detail)
    else:
        _check("ffmpeg (local)", False, "not installed (manager sync installs it)")

    try:
        free_gb = shutil.disk_usage(REPO_ROOT).free / (1024**3)
        _check("disk space", free_gb > 30, "%.0f GB free" % free_gb)
    except OSError:
        pass

    branch = gitops.current_branch()
    _check(
        "git checkout",
        True,
        "%s @ %s%s"
        % (branch, gitops.current_commit(), " (dirty)" if gitops.is_dirty() else ""),
    )
