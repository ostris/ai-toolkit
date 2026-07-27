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
    ):
        self.backend = backend  # cu130 / cu126 / rocm7.1 / mps / cpu
        self.torch_packages = torch_packages  # {name: version}
        self.torch_index = torch_index  # None = PyPI
        self.python_version = python_version
        self.requirements_file = requirements_file
        self.extra_packages = extra_packages or []
        self.optional_packages = optional_packages or []
        self.find_links = find_links or []
        self.notes = notes or []

    def torch_args(self):
        args = ["%s==%s" % (k, v) for k, v in sorted(self.torch_packages.items())]
        if self.torch_index:
            args += ["--index-url", self.torch_index]
        return args

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
    caps = [g.get("compute_cap") for g in nvidia.get("gpus", [])]
    has_blackwell = any(c and float(c) >= 12.0 for c in caps if c)
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
    optional = []
    find_links = []

    fa_url = _flash_attn_url(flavor, os_name, arch, python_version)
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


def build_spec(detection, allow_cpu=False):
    """Returns EnvSpec, or raises RuntimeError with a user-facing message."""
    os_name = detection["os"]

    if os_name == "mac":
        notes = ["flash-attn / NATTEN / triton are unavailable on macOS."]
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
