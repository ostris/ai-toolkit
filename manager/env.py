"""Python environment provisioning and dependency sync.

Strategy:
- If a venv already exists (.venv or venv), use it.
- Otherwise create one: prefer uv (downloads the exact Python version needed),
  fall back to the running Python's venv module if it is new enough.
- Installs go through `uv pip` when uv is available (much faster), else pip.

State (torch backend, requirements hash, applied migrations) is stored inside
the venv so a deleted venv means a clean slate — which is correct.
"""

import json
import os
import subprocess
import sys

from .util import (
    REPO_ROOT,
    IS_WINDOWS,
    clean_env,
    die,
    file_hash,
    find_uv,
    info,
    ok,
    run,
    venv_dir,
    venv_python,
    warn,
)

STATE_FILE = "aitk_manager_state.json"

MIN_SYSTEM_PYTHON = (3, 10)


# ---------------------------------------------------------------- state


def state_path(venv=None):
    return os.path.join(venv or venv_dir(), STATE_FILE)


def load_state():
    try:
        with open(state_path(), "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state):
    with open(state_path(), "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------- venv


def venv_exists():
    return os.path.isfile(venv_python())


def _venv_platform():
    """sysconfig platform of the existing venv ('win-amd64', 'win-arm64', ...)."""
    if not venv_exists():
        return None
    try:
        out = subprocess.run(
            [venv_python(), "-c", "import sysconfig; print(sysconfig.get_platform())"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=clean_env(),
        )
        if out.returncode != 0:
            return None
        return out.stdout.decode().strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _uv_python_platform(uv_python):
    """'win-arm64' / 'win-amd64' expected for a pinned uv interpreter request."""
    if not uv_python:
        return None
    if "windows-aarch64" in uv_python:
        return "win-arm64"
    if "windows-x86_64" in uv_python:
        return "win-amd64"
    return None


def ensure_venv(spec, dry_run=False):
    """Create the venv if missing. Returns path to the venv python."""
    if venv_exists():
        # Switching stacks (e.g. Spark emulated x64 <-> native arm64) needs a
        # different interpreter arch; the venv is disposable by design, so
        # recreate it rather than install unresolvable wheels into it.
        want = _uv_python_platform(spec.uv_python)
        have = _venv_platform()
        if want and have and want != have:
            if dry_run:
                info(
                    "[dry-run] venv is %s but this spec needs %s — would "
                    "recreate the venv." % (have, want)
                )
                return venv_python()
            warn(
                "Existing venv is %s but this spec needs %s — recreating the "
                "venv (all packages will be reinstalled)." % (have, want)
            )
            import shutil

            shutil.rmtree(venv_dir(), ignore_errors=True)
        else:
            return venv_python()
    if venv_exists():
        return venv_python()

    target = venv_dir()
    uv = find_uv()
    # spec.uv_python pins the full interpreter build (arch included) where the
    # default choice would be wrong — e.g. Windows-on-ARM must stay x86_64
    python_request = spec.uv_python or spec.python_version
    if dry_run:
        info(
            "[dry-run] would create venv at %s (python %s, via %s)"
            % (target, python_request, "uv" if uv else "venv module")
        )
        return venv_python(target)

    if uv:
        info("Creating venv with uv (python %s) at %s" % (python_request, target))
        run(
            [uv, "venv", target, "--python", python_request, "--seed"],
            env=clean_env(),
        )
    else:
        if sys.version_info < MIN_SYSTEM_PYTHON:
            die(
                "Python %d.%d is too old (need >= %d.%d) and uv is not installed.\n"
                "Install uv (https://docs.astral.sh/uv/) or a newer Python, then re-run."
                % (sys.version_info[:2] + MIN_SYSTEM_PYTHON)
            )
        if spec.uv_python:
            warn(
                "uv not found — the venv needs the %s interpreter and the "
                "system Python may be a different build. Install uv if the "
                "torch install below fails to resolve." % spec.uv_python
            )
        pyver = "%d.%d" % (sys.version_info[:2])
        if pyver != spec.python_version:
            warn(
                "Recommended Python is %s but using system Python %s "
                "(install uv to get the exact version automatically)."
                % (spec.python_version, pyver)
            )
        info("Creating venv at %s" % target)
        run([sys.executable, "-m", "venv", target])
    ok("Virtual environment ready.")
    return venv_python(target)


def _pip_install(args, dry_run=False, upgrade=False, check=True):
    """Install into the venv, via uv pip if available. Returns exit code."""
    uv = find_uv()
    if uv:
        cmd = [uv, "pip", "install", "--python", venv_python()]
    else:
        cmd = [venv_python(), "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd += args
    if dry_run:
        info("[dry-run] would run: %s" % " ".join(cmd))
        return 0
    code, _ = run(cmd, stream=True, env=clean_env(), check=check)
    return code


def _pip_install_no_deps(pkg, dry_run=False):
    """Install a single package with --no-deps. Returns exit code."""
    uv = find_uv()
    if uv:
        cmd = [uv, "pip", "install", "--python", venv_python(), "--no-deps", pkg]
    else:
        cmd = [venv_python(), "-m", "pip", "install", "--no-deps", pkg]
    if dry_run:
        info("[dry-run] would run: %s" % " ".join(cmd))
        return 0
    code, _ = run(cmd, stream=True, env=clean_env(), check=False)
    return code


def _pip_uninstall(packages, dry_run=False):
    uv = find_uv()
    if uv:
        cmd = [uv, "pip", "uninstall", "--python", venv_python()] + packages
    else:
        cmd = [venv_python(), "-m", "pip", "uninstall", "-y"] + packages
    if dry_run:
        info("[dry-run] would run: %s" % " ".join(cmd))
        return
    # non-fatal: the package may simply not be installed yet
    run(cmd, check=False, env=clean_env())


def venv_python_version():
    """'3.12' etc. from the venv interpreter, or None."""
    if not venv_exists():
        return None
    try:
        out = subprocess.run(
            [venv_python(), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=clean_env(),
        )
        if out.returncode != 0:
            return None
        return out.stdout.decode().strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------- torch


def installed_torch():
    """Returns torch.__version__ from the venv (e.g. '2.9.1+cu128'), or None."""
    if not venv_exists():
        return None
    try:
        out = subprocess.run(
            [venv_python(), "-c", "import torch; print(torch.__version__)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=120,
            env=clean_env(),
        )
        if out.returncode != 0:
            return None
        return out.stdout.decode().strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def torch_stack():
    """{package: version or 'ERROR: ...'} for torch, torchvision, torchaudio.

    Each one is imported rather than read from metadata: the failure mode this
    guards against is a C++ extension (libtorchaudio, torchvision's ops) linked
    against a libtorch that is no longer the installed one, and that only shows
    up at import time — the recorded version looks perfectly fine.
    """
    if not venv_exists():
        return {}
    code = (
        "import importlib, json\n"
        "r = {}\n"
        "for m in ('torch', 'torchvision', 'torchaudio'):\n"
        "    try:\n"
        "        r[m] = importlib.import_module(m).__version__\n"
        "    except Exception as e:\n"
        "        r[m] = 'ERROR: %s: %s' % (type(e).__name__, e)\n"
        "print(json.dumps(r))\n"
    )
    try:
        out = subprocess.run(
            [venv_python(), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=300,
            env=clean_env(),
        )
        return json.loads(out.stdout.decode().strip() or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {}


def _version_matches(current, want, backend):
    if not current or current.startswith("ERROR"):
        return False
    # local version tag carries the backend: "2.9.1+cu128"
    if "+" in current:
        version, local = current.split("+", 1)
        return version == want and local == backend
    # PyPI wheels (mac) have no local tag
    return current == want and backend in ("mps", "cpu")


def torch_matches(spec):
    """True only if the whole trio is at the pinned version AND imports.

    Checking torch alone is not enough — a resolver backtrack downgrades
    torchvision while leaving torch untouched, and torchaudio keeps its version
    number even when its extension can no longer load.
    """
    stack = torch_stack()
    if not stack:
        return False
    return all(
        _version_matches(stack.get(name), version, spec.backend)
        for name, version in spec.torch_packages.items()
    )


def ensure_torch(spec, dry_run=False):
    if torch_matches(spec):
        ok(
            "PyTorch %s (%s) already installed."
            % (spec.torch_packages["torch"], spec.backend)
        )
        return False
    current = installed_torch()
    if current:
        info(
            "PyTorch %s installed, need %s (%s) — reinstalling."
            % (current, spec.torch_packages["torch"], spec.backend)
        )
    else:
        info(
            "Installing PyTorch %s (%s)..."
            % (spec.torch_packages["torch"], spec.backend)
        )
    _pip_install(spec.torch_args(), dry_run=dry_run)
    return True


CONSTRAINTS_FILE = "aitk_torch_constraints.txt"


def _torch_pin_args(spec, dry_run=False):
    """`--constraint`/`--find-links` args nailing torch down for later passes."""
    path = os.path.join(venv_dir(), CONSTRAINTS_FILE)
    if not dry_run:
        with open(path, "w") as f:
            f.write(
                "# Generated by the AI Toolkit manager (manager/env.py).\n"
                "# Keeps requirements.txt and prebuilt accelerator wheels from\n"
                "# replacing the GPU torch build. Do not edit.\n"
            )
            f.write("\n".join(spec.torch_constraints()) + "\n")
    args = ["--constraint", path]
    for url in spec.torch_find_links():
        args += ["--find-links", url]
    return args


def _verify_torch(spec, dry_run=False):
    """Last line of defence: no install pass may leave torch swapped out.

    The constraints normally prevent this outright; this catches the cases they
    can't (a package vendoring its own torch, a pip fallback that ignores the
    constraint) before the venv is declared good.
    """
    if dry_run or torch_matches(spec):
        return False
    stack = torch_stack()
    found = ", ".join(
        "%s %s" % (name, stack.get(name) or "missing")
        for name in sorted(spec.torch_packages)
    )
    warn(
        "The PyTorch stack was disturbed during dependency install (%s) — "
        "restoring the pinned %s build." % (found, spec.backend)
    )
    _pip_install(spec.torch_args(), dry_run=dry_run)
    return True


# ---------------------------------------------------------------- requirements


def requirements_hash(spec):
    """Hash of every requirements file plus the spec itself."""
    req_files = [
        os.path.join(REPO_ROOT, f)
        for f in os.listdir(REPO_ROOT)
        if f.startswith("requirements") and f.endswith(".txt")
    ]
    req_files.append(os.path.join(REPO_ROOT, "dgx_requirements.txt"))
    base = file_hash(req_files)
    import hashlib

    h = hashlib.sha256()
    h.update(base.encode())
    h.update(json.dumps(spec.as_dict(), sort_keys=True).encode())
    return h.hexdigest()


def requirements_in_sync(spec):
    if not venv_exists():
        return False
    return load_state().get("req_hash") == requirements_hash(spec)


def _git_pinned_packages(spec):
    """{package_name: full git+ requirement line} from the requirements files.

    pip skips reinstalling a git pin whose version number didn't change even
    when the commit hash did, so pins whose URL changed since the last sync
    get uninstalled first to force the new commit (the trick the community
    Windows installer uses for diffusers).
    """
    pins = {}
    seen_files = set()

    def scan(path):
        if path in seen_files or not os.path.isfile(path):
            return
        seen_files.add(path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("-r "):
                    scan(os.path.join(os.path.dirname(path), line[3:].strip()))
                elif "git+" in line and not line.startswith("#"):
                    # e.g. git+https://github.com/huggingface/diffusers.git@<sha>
                    tail = line.split("/")[-1]
                    name = tail.split(".git")[0].split("@")[0]
                    if name:
                        pins[name] = line

    scan(spec.requirements_path())
    return pins


def _stale_git_pins(spec):
    """Git-pinned packages whose pin (commit) changed since the last sync."""
    current = _git_pinned_packages(spec)
    state = load_state()
    stored = state.get("git_pins")
    if stored is None:
        # no record of what's installed (pre-tracking env): if deps were ever
        # installed here, play it safe and force-reinstall all git pins once
        return list(current) if state.get("req_hash") else []
    return [name for name, line in current.items() if stored.get(name) != line]


# optional packages whose import name differs from the distribution name
_IMPORT_ALIASES = {"flash_linear_attention": "fla"}
# companion dists to remove on rollback (fla-core provides the `fla` module
# itself; leaving it behind would keep the broken import resolvable)
_ROLLBACK_EXTRAS = {"flash_linear_attention": ["fla-core"]}


def _optional_names(pkg):
    """(distribution, import) names for an optional package spec or wheel URL."""
    if "://" in pkg:
        name = os.path.basename(pkg).split("-")[0]
    else:
        name = pkg
        for sep in ("==", ">=", "<=", "<", ">", "["):
            name = name.split(sep)[0]
    name = name.strip().replace("-", "_")
    return name, _IMPORT_ALIASES.get(name, name)


def _venv_import_ok(module_name):
    try:
        out = subprocess.run(
            [venv_python(), "-c", "import %s" % module_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
            env=clean_env(),
        )
        return out.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _filter_extras(extras):
    """Drop wheel URLs whose cpXY tag doesn't match the venv python."""
    pyver = venv_python_version()
    cp_tag = "cp" + pyver.replace(".", "") if pyver else None
    kept = []
    for pkg in extras:
        if "cp3" in pkg and cp_tag and cp_tag not in pkg:
            warn(
                "Skipping %s (built for a different python than venv %s)."
                % (os.path.basename(pkg), pyver)
            )
            continue
        kept.append(pkg)
    return kept


def ensure_requirements(spec, dry_run=False, force=False):
    if not force and requirements_in_sync(spec):
        ok("Requirements already in sync.")
        return False
    # force git-pinned deps (diffusers) onto a newly pinned commit — pip won't
    # reinstall them on its own because the version number stays the same
    stale = _stale_git_pins(spec)
    if stale:
        info("Git pin changed — reinstalling: %s" % ", ".join(stale))
        _pip_uninstall(stale, dry_run=dry_run)
    # every pass below carries the torch pins so nothing can swap the GPU build
    pins = _torch_pin_args(spec, dry_run=dry_run)
    find_links = list(pins)
    for url in spec.find_links:
        find_links += ["--find-links", url]
    info("Installing requirements from %s..." % spec.requirements_file)
    # find_links included: on Spark the requirements themselves resolve
    # self-built wheels (opencv, soxr, ...) from the spark wheel set
    _pip_install(["-r", spec.requirements_path()] + find_links, dry_run=dry_run)
    extras = _filter_extras(spec.extra_packages)
    if extras:
        info("Installing platform extras...")
        _pip_install(extras + find_links, dry_run=dry_run, upgrade=True)
    # the optional import checks below only mean anything against the torch we
    # actually intend to ship, so repair it first if something got through
    _verify_torch(spec, dry_run=dry_run)
    # accelerators (flash-attn, NATTEN, ...): install one-by-one, warn on
    # failure — training works without them, so never fail the whole install.
    # No --upgrade here: every optional spec is an exact pin or wheel URL, and
    # uv's -U eagerly upgrades the whole dependency closure, blowing past
    # requirements.txt pins (numpy/transformers) that only the requirements
    # pass enforces.
    for pkg in _filter_extras(spec.optional_packages):
        label = os.path.basename(pkg) if "://" in pkg else pkg
        info("Installing optional accelerator: %s" % label)
        code = _pip_install([pkg] + find_links, dry_run=dry_run, check=False)
        if code != 0:
            warn("Optional package failed to install (continuing): %s" % label)
            continue
        if dry_run:
            continue
        # prebuilt accelerator wheels are sometimes built against a torch
        # nightly and fail to load against the release ABI — verify the
        # import and roll back rather than leaving a broken wheel installed
        dist_name, import_name = _optional_names(pkg)
        if not _venv_import_ok(import_name):
            warn(
                "%s installed but fails to import against this torch build — "
                "removing it (training falls back to native attention)."
                % import_name
            )
            _pip_uninstall([dist_name] + _ROLLBACK_EXTRAS.get(dist_name, []))
    # packages whose dependency metadata is unsatisfiable on this platform but
    # which work fine without it (e.g. tensorboard's grpcio on win_arm64)
    for pkg in spec.no_deps_packages:
        info("Installing (no-deps): %s" % pkg)
        code = _pip_install_no_deps(pkg, dry_run=dry_run)
        if code != 0:
            warn("No-deps package failed to install (continuing): %s" % pkg)
    _verify_torch(spec, dry_run=dry_run)
    if not dry_run:
        state = load_state()
        state["req_hash"] = requirements_hash(spec)
        state["backend"] = spec.backend
        state["git_pins"] = _git_pinned_packages(spec)
        save_state(state)
    return True


# ---------------------------------------------------------------- sitecustomize


def _msvc_runtime_env():
    """{env: value} + [bin dirs] from vcvarsarm64, for triton's runtime JIT.

    Triton compiles its kernel launcher stubs with cl.exe at runtime (cached
    afterwards in ~/.triton), which needs INCLUDE/LIB and cl on PATH. Capture
    the values once at sync time and bake them into sitecustomize.
    """
    vcvars = (
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
        r"\VC\Auxiliary\Build\vcvarsarm64.bat"
    )
    if not os.path.isfile(vcvars):
        return {}, []
    try:
        # string form: list2cmdline would mangle the nested quoting
        out = subprocess.run(
            'cmd /s /c "call "%s" >nul && set"' % vcvars,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if out.returncode != 0:
            return {}, []
        env = {}
        for line in out.stdout.decode(errors="replace").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env[k.upper()] = v
        bin_dirs = [
            d for d in env.get("PATH", "").split(os.pathsep)
            if os.path.isfile(os.path.join(d, "cl.exe"))
        ][:1]
        keep = {k: env[k] for k in ("INCLUDE", "LIB") if env.get(k)}
        if keep and bin_dirs:
            keep["CC"] = "cl"
            return keep, bin_dirs
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {}, []


def write_sitecustomize(dry_run=False, spec=None):
    """Drop a sitecustomize.py into the venv exposing runtime DLL dirs.

    sitecustomize is imported automatically at interpreter startup, so ANY use
    of the venv python (UI-spawned training jobs, run.py from a terminal) gets
    .ffmpeg/bin on PATH — and on Windows, os.add_dll_directory so torchcodec
    finds the FFmpeg DLLs. On Spark the spec also carries the CUDA/cuDNN/APL
    bin dirs, because the native torch wheel does not bundle its DLLs.
    """
    from . import ffmpeg

    if not venv_exists():
        return
    try:
        out = subprocess.run(
            [
                venv_python(),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=clean_env(),
        )
        site_packages = out.stdout.decode().strip()
    except (OSError, subprocess.TimeoutExpired):
        site_packages = ""
    if not site_packages or not os.path.isdir(site_packages):
        warn("Could not locate venv site-packages — skipping sitecustomize.")
        return
    target = os.path.join(site_packages, "sitecustomize.py")
    dll_dirs = [ffmpeg.bin_dir()] + list(getattr(spec, "runtime_dll_dirs", []) or [])
    runtime_env = dict(getattr(spec, "runtime_env", {}) or {})
    if getattr(spec, "backend", None) == "cu134":
        # triton's runtime launcher JIT needs the MSVC environment
        msvc_env, msvc_bins = _msvc_runtime_env()
        runtime_env.update(msvc_env)
        dll_dirs += msvc_bins
    content = (
        "# Generated by the AI Toolkit manager (manager/env.py). Do not edit;\n"
        "# regenerated on every `manager sync`.\n"
        "import os\n"
        "for _k, _v in %r.items():\n"
        "    os.environ.setdefault(_k, _v)\n"
        "_DLL_DIRS = %r\n"
        "_FFMPEG_LIB = %r\n"
        "for _d in _DLL_DIRS:\n"
        "    if os.path.isdir(_d):\n"
        "        os.environ['PATH'] = _d + os.pathsep + os.environ.get('PATH', '')\n"
        "        if hasattr(os, 'add_dll_directory'):\n"
        "            try:\n"
        "                os.add_dll_directory(_d)\n"
        "            except OSError:\n"
        "                pass\n"
        "if os.path.isdir(_FFMPEG_LIB):\n"
        "    # inherited by child processes (the ffmpeg/ffprobe executables\n"
        "    # need it to find their own shared libs)\n"
        "    _prev = os.environ.get('LD_LIBRARY_PATH', '')\n"
        "    if _FFMPEG_LIB not in _prev.split(os.pathsep):\n"
        "        os.environ['LD_LIBRARY_PATH'] = (\n"
        "            _FFMPEG_LIB + ((os.pathsep + _prev) if _prev else '')\n"
        "        )\n"
    ) % (runtime_env, dll_dirs, ffmpeg.lib_dir())
    if dry_run:
        info("[dry-run] would write %s" % target)
        return
    with open(target, "w") as f:
        f.write(content)


# ---------------------------------------------------------------- sync


def sync(spec, detection, dry_run=False, force=False):
    """Bring the environment fully up to date for this checkout."""
    from . import ffmpeg, gitwin, migrations, nodejs, uvbin

    for note in spec.notes:
        warn(note)
    uvbin.ensure_uv(dry_run=dry_run)
    gitwin.ensure_git(dry_run=dry_run)
    if spec.backend == "cu134":
        # native Spark stack: provision CUDA/cuDNN/APL runtime DLLs, VC
        # redist, and (best-effort) MSVC for triton's kernel launcher JIT
        from . import sparkdeps

        sparkdeps.ensure_spark_runtime(dry_run=dry_run)
    ensure_venv(spec, dry_run=dry_run)
    # sitecustomize must exist BEFORE any torch import check below: on Spark
    # the torch wheel is unbundled and only imports once the CUDA/cuDNN/BLAS
    # DLL dirs from the spec are exposed to the interpreter
    write_sitecustomize(dry_run=dry_run, spec=spec)
    changed_torch = ensure_torch(spec, dry_run=dry_run)
    # a torch reinstall can clobber pinned deps; force req pass afterwards
    ensure_requirements(spec, dry_run=dry_run, force=force or changed_torch)
    ffmpeg.ensure_ffmpeg(detection, dry_run=dry_run, spec=spec)
    nodejs.ensure_node(detection, dry_run=dry_run)
    nodejs.ensure_ui_deps(dry_run=dry_run)
    write_sitecustomize(dry_run=dry_run, spec=spec)
    migrations.run_pending(dry_run=dry_run)
    ok("Environment is up to date.")
