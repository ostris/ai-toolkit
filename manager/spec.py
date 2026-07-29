"""Maps detected hardware to an environment spec.

This is the single source of truth for "what does this machine need to run
this commit of AI Toolkit". One universal torch version across all platforms;
per-platform accelerator extras (flash-attn, NATTEN, triton) wherever prebuilt
wheels exist. **Update the pins below together with the README install
instructions and run_mac.zsh.**

Wheel coverage for the pinned set (verified 2026-07, flash-attn + NATTEN GPU
kernels smoke-tested on an RTX 5090 / sm120 with torch 2.13.0+cu130):
- torch 2.13.0: cu126/cu130 wheels for linux x86_64 + aarch64 + windows; PyPI
  wheels for mac arm64. torchaudio is in maintenance mode — 2.11.0 is the
  current release and is torch-version-agnostic (no torch dep in metadata).
- torchcodec 0.15.0: supports torch >= 2.11, wheels on all platforms.
- flash-attn 2.8.3: prebuilt by mjun0812/flash-attention-prebuild-wheels for
  {cu126,cu130} x {cp310..cp314} x {linux x86_64, linux aarch64, windows}.
  NOTE: the torch2.12 linux wheels there were built against a torch nightly
  and fail to import on 2.12.0 final — the torch2.13 batches (v0.9.47+) are
  verified good. Re-verify imports whenever bumping torch.
- NATTEN 0.21.7: prebuilt at whl.natten.org for {cu126,cu130,cu132} x
  {cp310..cp314} x {linux x86_64, linux aarch64}. No Windows/mac wheels.
- triton: bundled with torch on Linux (incl. aarch64; torch 2.13 bundles
  triton 3.7.1); triton-windows 3.7.x matches on Windows; nothing for MPS.
- flash-linear-attention 0.5.2: pure-Python (py3-none-any) Triton kernels —
  installs anywhere, needs triton>=3.3 at runtime. Installed bare (no backend
  extra) so it never pulls its own torch/triton over our pinned stack; usable
  on every platform with a working triton (CUDA linux/windows, Spark, ROCm),
  not on MPS/CPU.
- Windows-on-ARM (verified 2026-07): there are NO win_arm64 wheels for any of
  the CUDA stack — torch cu130, triton-windows, flash-attn and Prisma's node
  engine are all x64-only. The supported configuration is therefore the x64
  stack end to end (Python, torch, Node) running under Windows' x64 emulation,
  with the GPU driven natively by the NVIDIA driver. build_spec() pins the
  interpreter arch explicitly so uv can never flip the venv to native aarch64
  (where torch would not resolve). Revisit if pytorch.org ever publishes
  win_arm64 CUDA wheels.

Requirements files must never pin anything torch itself depends on below what
the pinned torch needs (torch 2.13 wants setuptools>=77.0.3). The resolver does
not report that as a conflict — torch is an unpinned transitive dep of timm /
peft / accelerate / torchvision, so it just silently backtracks to an older
torch and drags torchvision down with it. `torch_constraints()` below turns
that class of mistake into a hard resolution error instead of a broken venv.

extra_packages are installed AFTER requirements.txt with --upgrade so they can
override requirement pins (e.g. torchcodec). optional_packages are installed
one-by-one and only warn on failure (accelerators the training code can live
without). Wheel URLs containing a cpXY tag are skipped automatically if the
venv python doesn't match.
"""

import os

from .util import REPO_ROOT

# ---- version pins (edit these to move the fleet forward) -------------------

TORCH = {"torch": "2.13.0", "torchvision": "0.28.0", "torchaudio": "2.11.0"}
TORCH_TAG = "2.13"  # as it appears in flash-attn / natten wheel names

TORCHCODEC = "torchcodec==0.15.0"
TRITON_WINDOWS = "triton-windows>=3.7,<3.8"

NATTEN_VERSION = "0.21.7"
NATTEN_FIND_LINKS = "https://whl.natten.org"

# pure-Python triton kernels; bare install (no [cuda]/[rocm] extra) on purpose —
# the extras only add torch/triton pins we already manage per-platform
FLA = "flash-linear-attention==0.5.2"

FLASH_ATTN_VERSION = "2.8.3"
_FA_BASE = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
)
# (os, arch) -> (release tag, wheel platform tag) — tags are per torch
# version; these carry the torch2.13 builds
_FA_BUILDS = {
    ("linux", "x86_64"): ("v0.9.47", "manylinux_2_24_x86_64.manylinux_2_28_x86_64"),
    ("linux", "aarch64"): ("v0.9.48", "manylinux_2_34_aarch64"),
    ("windows", "x86_64"): ("v0.9.52", "win_amd64"),
}

# helper build tools some sdists need on Windows
_WIN_HELPERS = ["wheel", "setuptools", "poetry-core", "hf_xet"]

PYTORCH_INDEX = "https://download.pytorch.org/whl/"

# ---- NVIDIA RTX Spark: native Windows-on-ARM CUDA ---------------------------
# The CUDA 13.4 developer preview added native win_arm64 CUDA. No public wheels
# exist for the GPU stack, so we build them ourselves (torch from main +
# pytorch/pytorch#190448, plus torchvision/torchaudio/torchcodec and the deps
# with no win_arm64 wheels). The manager installs them from a find-links
# source: the local wheels/spark/ dir during development, or the hosted URL
# once published. Without that source (or with a pre-13.4 driver) Spark
# machines fall back to the emulated x64 stack below, which also works.
SPARK_BACKEND = "cu134"
SPARK_TORCH = {
    "torch": "2.14.0.dev20260727",
    "torchvision": "0.29.0.dev20260727",
    "torchaudio": "2.11.0.dev20260727",
}
SPARK_WHEELS_DIR = os.path.join(REPO_ROOT, "wheels", "spark")
# GitHub's expanded_assets endpoint serves plain HTML anchors — a valid
# pip/uv find-links page pointing at the release assets.
SPARK_WHEELS_URL = (
    "https://github.com/ostris/ai-toolkit-spark-wheels/releases/"
    "expanded_assets/cu134-20260727"
)
SPARK_UV_PYTHON = "cpython-3.12-windows-aarch64-none"
# Runtime DLL homes for the unbundled native torch (TH_BINARY_BUILD=0) are
# resolved dynamically (system installs of any version, else the downloadable
# runtime bundle) — see sparkdeps.resolve_dll_dirs().


def _spark_wheels_source():
    """find-links source holding the self-built win_arm64 wheels, or None."""
    if os.path.isdir(SPARK_WHEELS_DIR):
        for name in os.listdir(SPARK_WHEELS_DIR):
            if name.startswith("torch-") and "win_arm64" in name:
                return SPARK_WHEELS_DIR
    return SPARK_WHEELS_URL


def _spark_capable(detection):
    """Driver new enough for native win_arm64 CUDA (R616+ reports CUDA 13.4)."""
    nvidia = detection.get("nvidia") or {}
    try:
        cuda = tuple(int(x) for x in (nvidia.get("cuda_version") or "").split("."))
    except ValueError:
        return False
    return cuda >= (13, 4)


class EnvSpec(object):
    def __init__(
        self,
        backend,
        torch_packages,
        torch_index=None,
        python_version="3.12",
        requirements_file="requirements.txt",
        extra_packages=None,
        optional_packages=None,
        find_links=None,
        notes=None,
        uv_python=None,
        torch_links=None,
        no_deps_packages=None,
        runtime_dll_dirs=None,
    ):
        self.backend = backend  # cu134 / cu130 / cu126 / rocm7.1 / mps / cpu
        self.torch_packages = torch_packages  # {name: version}
        self.torch_index = torch_index  # None = PyPI
        self.python_version = python_version
        self.requirements_file = requirements_file
        self.extra_packages = extra_packages or []
        self.optional_packages = optional_packages or []
        self.find_links = find_links or []
        self.notes = notes or []
        # full uv interpreter request (e.g. "cpython-3.12-windows-x86_64-none")
        # when the venv arch must not be left to uv's default; None = just
        # python_version
        self.uv_python = uv_python
        # --find-links sources for the torch trio itself (self-built wheels);
        # used when torch_index is None
        self.torch_links = torch_links or []
        # installed with --no-deps after everything else (e.g. tensorboard on
        # Spark, whose grpcio dep has no win_arm64 wheels but is only needed
        # for the server, not the log writer)
        self.no_deps_packages = no_deps_packages or []
        # extra DLL dirs the venv needs at runtime (unbundled CUDA/cuDNN/BLAS
        # on Spark); baked into sitecustomize.py, missing dirs skipped
        self.runtime_dll_dirs = runtime_dll_dirs or []
        # env vars every venv python needs (sitecustomize setdefault)
        self.runtime_env = {}

    def torch_args(self):
        args = ["%s==%s" % (k, v) for k, v in sorted(self.torch_packages.items())]
        if self.torch_index:
            args += ["--index-url", self.torch_index]
        for links in self.torch_links:
            args += ["--find-links", links]
        return args

    def torch_constraints(self):
        """Exact pins for the torch trio, carrying the backend local tag.

        Written to a constraints file and passed to every install pass after
        torch itself, so nothing in requirements.txt (or a prebuilt accelerator
        wheel that merely declares `torch`) can quietly swap the GPU build for
        a PyPI one. Without this, a single conflicting pin anywhere in the tree
        makes the resolver silently backtrack to an older torch instead of
        failing, and the accelerator wheels then reinstall a plain PyPI torch
        on top — which leaves torchaudio/torchvision linked against a libtorch
        that is no longer there.
        """
        tag = "+%s" % self.backend if self.torch_index else ""
        return [
            "%s==%s%s" % (k, v, tag) for k, v in sorted(self.torch_packages.items())
        ]

    def torch_find_links(self):
        """Per-package wheel pages making the constrained pins resolvable.

        Deliberately per-package `--find-links` rather than `--extra-index-url`:
        the pytorch index also mirrors numpy/pillow/setuptools/... and with uv's
        default first-index strategy an extra index would win for those too,
        pinning them to whatever stale copy pytorch happens to host.
        """
        if not self.torch_index:
            return []
        base = self.torch_index.rstrip("/")
        return ["%s/%s/" % (base, name) for name in sorted(self.torch_packages)]

    def requirements_path(self):
        return os.path.join(REPO_ROOT, self.requirements_file)

    def as_dict(self):
        return {
            "backend": self.backend,
            "torch_packages": self.torch_packages,
            "torch_index": self.torch_index,
            "python_version": self.python_version,
            "requirements_file": self.requirements_file,
            "extra_packages": self.extra_packages,
            "optional_packages": self.optional_packages,
            "find_links": self.find_links,
            "notes": self.notes,
            "uv_python": self.uv_python,
            "torch_links": self.torch_links,
            "no_deps_packages": self.no_deps_packages,
            "runtime_dll_dirs": self.runtime_dll_dirs,
            "runtime_env": self.runtime_env,
        }


def _flash_attn_url(flavor, os_name, arch, python_version):
    build = _FA_BUILDS.get((os_name, arch))
    if build is None:
        return None
    tag, plat = build
    cp = "cp" + python_version.replace(".", "")
    return "%s%s/flash_attn-%s+%storch%s-%s-%s-%s.whl" % (
        _FA_BASE,
        tag,
        FLASH_ATTN_VERSION,
        flavor,
        TORCH_TAG,
        cp,
        cp,
        plat,
    )


def _natten_pin(flavor):
    # natten wheel local tags use the full torch version without dots: torch2120cu130
    return "natten==%s+torch%s%s" % (
        NATTEN_VERSION,
        TORCH["torch"].replace(".", ""),
        flavor.replace(".", ""),
    )


def _cuda_flavor(detection):
    """Pick a cuda wheel flavor the installed driver can actually run."""
    nvidia = detection.get("nvidia") or {}
    cuda = None
    if nvidia.get("cuda_version"):
        try:
            cuda = tuple(int(x) for x in nvidia["cuda_version"].split("."))
        except ValueError:
            cuda = None
    if cuda is None:
        # driver present but version unknown — assume current
        return "cu130", []
    if cuda >= (13, 0):
        return "cu130", []
    # non-GPU rows (the NPU on ARM hybrids) report compute_cap as "[N/A]"
    caps = []
    for g in nvidia.get("gpus", []):
        try:
            caps.append(float(g.get("compute_cap")))
        except (TypeError, ValueError):
            pass
    has_blackwell = any(c >= 12.0 for c in caps)
    if cuda >= (12, 6):
        if has_blackwell:
            raise RuntimeError(
                "Blackwell GPU detected but the NVIDIA driver only supports "
                "CUDA %s. Blackwell needs the cu130 build — update your "
                "driver to 580+ and re-run." % nvidia["cuda_version"]
            )
        return "cu126", [
            "NVIDIA driver only supports CUDA %s — installing cu126 wheels. "
            "Updating your driver is recommended." % nvidia["cuda_version"]
        ]
    raise RuntimeError(
        "NVIDIA driver only supports CUDA %s, which is too old for the pinned "
        "torch build. Update your NVIDIA driver, then re-run install."
        % nvidia["cuda_version"]
    )


def _cuda_spec(detection):
    os_name = detection["os"]
    arch = detection["arch"]
    flavor, notes = _cuda_flavor(detection)
    python_version = "3.12"
    requirements = (
        "dgx_requirements.txt" if detection.get("is_dgx") else "requirements.txt"
    )
    if detection.get("is_dgx"):
        # the old "Python 3.11 on DGX OS" constraint was for conda/system
        # installs; uv provisions 3.12 and all aarch64 cp312 wheels exist now
        notes = notes + [
            "DGX OS / Grace detected: using %s wheels and dgx_requirements.txt."
            % flavor
        ]

    extras = [TORCHCODEC]
    optional = [FLA]
    find_links = []

    # Windows-on-ARM runs the x64 wheel stack (see module docstring), so wheel
    # selection uses x86_64 there regardless of the host arch.
    wheel_arch = "x86_64" if os_name == "windows" else arch
    fa_url = _flash_attn_url(flavor, os_name, wheel_arch, python_version)
    if fa_url:
        optional.append(fa_url)

    if os_name == "linux":
        optional.append(_natten_pin(flavor))
        find_links.append(NATTEN_FIND_LINKS)
    elif os_name == "windows":
        extras = _WIN_HELPERS + extras + [TRITON_WINDOWS]
        notes = notes + ["NATTEN has no Windows wheels — skipping it."]

    return EnvSpec(
        flavor,
        TORCH,
        torch_index=PYTORCH_INDEX + flavor,
        python_version=python_version,
        requirements_file=requirements,
        extra_packages=extras,
        optional_packages=optional,
        find_links=find_links,
        notes=notes,
    )


def _spark_spec(detection, wheels_source):
    """Native win_arm64 CUDA 13.4 stack from self-built wheels (RTX Spark)."""
    from . import sparkdeps

    dll_dirs = sparkdeps.resolve_dll_dirs()
    spec = _make_spark_spec(detection, wheels_source, dll_dirs)
    # OpenCV's runtime CPU detection is blind to FP16/DOTPROD on Windows
    # ARM64 and aborts the process at import even though the N1X supports
    # both; the self-built cv2 wheels rely on this skip.
    spec.runtime_env["OPENCV_SKIP_CPU_BASELINE_CHECK"] = "1"
    # our triton wheel does not bundle NVIDIA's compiler tools (preview
    # licensing) — point its knobs at the user's CUDA toolkit
    spec.runtime_env.update(sparkdeps.triton_tool_env())
    return spec


def _make_spark_spec(detection, wheels_source, dll_dirs):
    return EnvSpec(
        SPARK_BACKEND,
        SPARK_TORCH,
        torch_index=None,
        python_version="3.12",
        requirements_file="spark_requirements.txt",
        # all pins resolve from the spark wheel set (self-built win_arm64):
        # torchcodec, plus our triton port (torch.compile / compiled flex
        # attention) — see the wheel-set build notes
        extra_packages=_WIN_HELPERS + [
            "torchcodec==0.15.0",
            "triton==3.8.0+git8743423b",
        ],
        # import-verified with rollback, like accelerators on other platforms
        optional_packages=[
            "flash-attn==2.8.3+cu134torch2.14",
            "natten==0.21.7",
            FLA,
        ],
        find_links=[wheels_source],
        notes=[
            "RTX Spark native mode: win_arm64 CUDA %s stack from the "
            "ai-toolkit wheel set (CUDA 13.4 developer preview), including "
            "self-built flash-attn, NATTEN and triton (torch.compile)."
            % SPARK_BACKEND,
        ],
        uv_python=SPARK_UV_PYTHON,
        torch_links=[wheels_source],
        no_deps_packages=["tensorboard"],
        runtime_dll_dirs=dll_dirs,
    )


def build_spec(detection, allow_cpu=False):
    """Returns EnvSpec, or raises RuntimeError with a user-facing message."""
    if (
        detection["os"] == "windows"
        and detection["arch"] == "aarch64"
        and detection.get("backend") == "cuda"
        and os.environ.get("AITK_SPARK_NATIVE", "1") != "0"
        and _spark_capable(detection)
    ):
        from . import sparkdeps

        wheels_source = _spark_wheels_source()
        # the CUDA toolkit is the one manual install (preview EULA) — without
        # it the native wheels cannot run, so fall back to the x64 stack
        if wheels_source and sparkdeps.cuda_bin_dir():
            return _spark_spec(detection, wheels_source)

    spec = _build_spec(detection, allow_cpu=allow_cpu)
    if detection["os"] == "windows" and detection["arch"] == "aarch64":
        # Emulated-x64 fallback (no native wheel source, old driver, or
        # AITK_SPARK_NATIVE=0). Pin the venv interpreter to x64 explicitly:
        # uv currently defaults to an emulated x86_64 Python on arm64 hosts,
        # but says it will flip to native aarch64 once it considers support
        # mature — which would silently leave a venv where the cu130 torch
        # wheels don't resolve.
        spec.uv_python = "cpython-%s-windows-x86_64-none" % spec.python_version
        spec.notes.append(
            "Windows-on-ARM detected: using the x64 stack under Windows' "
            "emulation (GPU work still runs natively via the NVIDIA driver). "
            "Native mode needs a CUDA 13.4+ driver and the Spark wheel set."
        )
    return spec


def _build_spec(detection, allow_cpu=False):
    os_name = detection["os"]

    if os_name == "mac":
        notes = [
            "flash-attn / NATTEN / triton / flash-linear-attention are "
            "unavailable on macOS."
        ]
        if detection["backend"] != "mps":
            notes.append("Intel Mac detected — training will be extremely slow.")
        return EnvSpec(
            "mps",
            TORCH,
            python_version="3.12",
            extra_packages=[TORCHCODEC],
            notes=notes,
        )

    if detection["backend"] == "cuda":
        return _cuda_spec(detection)

    if detection["backend"] == "rocm":
        return EnvSpec(
            "rocm7.1",
            TORCH,
            torch_index=PYTORCH_INDEX + "rocm7.1",
            extra_packages=[TORCHCODEC],
            # runs on ROCm via the triton bundled with rocm torch
            optional_packages=[FLA],
            notes=[
                "AMD ROCm support is experimental and largely untested.",
                "flash-attn / NATTEN prebuilt wheels are unavailable for ROCm.",
            ],
        )

    # CPU fallback
    if not allow_cpu:
        raise RuntimeError(
            "No supported GPU detected (NVIDIA CUDA, AMD ROCm, or Apple Silicon). "
            "Training on CPU is not practical. Pass --cpu to install anyway."
        )
    return EnvSpec(
        "cpu",
        TORCH,
        torch_index=PYTORCH_INDEX + "cpu",
        extra_packages=[TORCHCODEC],
        notes=["CPU-only install: training will be impractically slow."],
    )
