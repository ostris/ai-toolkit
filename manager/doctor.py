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

    arch_detail = "%s %s" % (d["os"], d["arch"])
    if d["os"] == "windows" and d["arch"] == "aarch64":
        from . import spec as spec_mod_arch

        try:
            _s = spec_mod_arch.build_spec(d, allow_cpu=True)
            if _s.backend == "cu134":
                arch_detail += " (RTX Spark: native win_arm64 CUDA stack)"
            else:
                arch_detail += " (Windows-on-ARM: x64 stack via emulation)"
        except RuntimeError:
            pass
    _check("os / arch", True, arch_detail)
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
        from . import spec as spec_mod

        stack = env.torch_stack()
        torch = stack.get("torch")
        if torch and torch.startswith("ERROR"):
            torch = None
        _check("torch", torch is not None, stack.get("torch") or "not installed")
        # torchvision/torchaudio ship C++ extensions linked against libtorch —
        # a version skew here fails at import, not at install
        for name in ("torchvision", "torchaudio"):
            found = stack.get(name)
            _check(
                name,
                bool(found) and not found.startswith("ERROR"),
                found or "not installed",
            )
        try:
            want = spec_mod.build_spec(d, allow_cpu=True)
            _check(
                "torch stack pins",
                env.torch_matches(want),
                "expected %s (%s)"
                % (
                    ", ".join(
                        "%s %s" % (k, v) for k, v in sorted(want.torch_packages.items())
                    ),
                    want.backend,
                ),
            )
        except RuntimeError as e:
            _check("torch stack pins", False, str(e))
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
