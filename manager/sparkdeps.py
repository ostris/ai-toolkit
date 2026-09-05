"""RTX Spark native-stack runtime provisioning (Windows on ARM, cu134).

Goal: a fresh Spark machine runs run_windows.bat and gets as close to
zero-manual-setup as licensing allows. The native wheels (torch etc.) do not
bundle CUDA / cuDNN / BLAS DLLs, and triton's launcher JIT wants MSVC. Policy:
NVIDIA components are never redistributed by us.

- CUDA 13.4 toolkit (developer preview): MANUAL install — the preview EULA
  requires NVIDIA's own click-through, so the manager only detects it and
  prints instructions when missing. This is the single manual step.
- cuDNN (arm64): auto-downloaded from NVIDIA's own official installer URL and
  installed silently — fetched directly from NVIDIA, not redistributed.
- Arm Performance Libraries: auto-install via winget (official Arm package).
- MSVC Build Tools (triton torch.compile JIT only): auto-install via winget;
  failure downgrades gracefully (training works, no torch.compile).
- VC redistributable (arm64): auto-install via winget when msvcp140 missing.

Everything is best-effort with warnings; the training stack itself only hard-
requires the CUDA + cuDNN + APL DLL dirs.
"""

import glob
import os
import subprocess

from .util import download, info, ok, warn, which

CUDA_DOWNLOAD_PAGE = (
    "https://developer.nvidia.com/cuda-13-4-0-download-archive"
    "?target_os=Windows&target_arch=arm64"
)
# NVIDIA's official public installer for cuDNN on Windows arm64. Downloaded
# straight from NVIDIA at install time (we do not redistribute it). Update
# together with the wheel set when moving to a newer cuDNN.
CUDNN_INSTALLER_URL = (
    "https://developer.download.nvidia.com/compute/cudnn/9.25.0/"
    "local_installers/cudnn_9.25.0_windows_arm64.exe"
)

# System install roots, newest version preferred (globs, not pinned versions)
_CUDA_BIN_GLOB = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\arm64"
_CUDNN_BIN_GLOB = r"C:\Program Files\NVIDIA\CUDNN\v*\bin\*\arm64"
_ARMPL_BIN_GLOB = r"C:\Program Files\Arm Performance Libraries\armpl_*\bin"

_VS_BUILDTOOLS = (
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
)


def _newest(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def cuda_bin_dir():
    return _newest(_CUDA_BIN_GLOB)


def cuda_root():
    d = cuda_bin_dir()
    # <root>\bin\arm64 -> <root>
    return os.path.dirname(os.path.dirname(d)) if d else None


def cudnn_bin_dir():
    return _newest(_CUDNN_BIN_GLOB)


def armpl_bin_dir():
    return _newest(_ARMPL_BIN_GLOB)


def resolve_dll_dirs():
    """All runtime DLL dirs for the native stack (existing ones only)."""
    return [d for d in (cuda_bin_dir(), cudnn_bin_dir(), armpl_bin_dir()) if d]


def runtime_complete():
    return bool(cuda_bin_dir() and cudnn_bin_dir() and armpl_bin_dir())


def triton_tool_env():
    """TRITON_*_PATH env for ptxas etc. from the system CUDA install.

    Our triton wheel deliberately does NOT bundle NVIDIA's compiler tools
    (developer-preview licensing); resolve them from the user's toolkit.
    """
    root = cuda_root()
    if not root:
        return {}
    env = {}
    for var, exe in (
        ("TRITON_PTXAS_PATH", "ptxas.exe"),
        ("TRITON_PTXAS_BLACKWELL_PATH", "ptxas.exe"),
        ("TRITON_CUOBJDUMP_PATH", "cuobjdump.exe"),
        ("TRITON_NVDISASM_PATH", "nvdisasm.exe"),
    ):
        path = os.path.join(root, "bin", exe)
        if os.path.isfile(path):
            env[var] = path
    return env


def check_cuda():
    """CUDA toolkit is the one manual install (preview EULA). Detect + guide."""
    if cuda_bin_dir():
        return True
    warn(
        "The CUDA 13.4 toolkit (arm64) is not installed. NVIDIA's developer "
        "preview license requires installing it manually:\n"
        "    1. Download from %s\n"
        "    2. Install with default settings, then re-run this setup.\n"
        "The RTX Spark developer driver (R616+) is required as well."
        % CUDA_DOWNLOAD_PAGE
    )
    return False


def ensure_cudnn(dry_run=False):
    """Fetch + silently run NVIDIA's official cuDNN installer if missing."""
    if cudnn_bin_dir():
        return True
    if dry_run:
        info("[dry-run] would download and install cuDNN from NVIDIA")
        return False
    import tempfile

    tmp = tempfile.mkdtemp(prefix="aitk_cudnn_")
    try:
        exe = os.path.join(tmp, os.path.basename(CUDNN_INSTALLER_URL))
        download(CUDNN_INSTALLER_URL, exe, label="cuDNN (from NVIDIA)")
        info("Installing cuDNN (silent)...")
        code = subprocess.call([exe, "-s"])
        if code != 0:
            warn("cuDNN installer exited with %d." % code)
        return cudnn_bin_dir() is not None
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def have_msvc():
    return bool(
        glob.glob(os.path.join(_VS_BUILDTOOLS, "VC", "Tools", "MSVC", "*",
                               "bin", "Hostarm64", "arm64", "cl.exe"))
    )


def _winget_install(args, label, dry_run=False):
    winget = which("winget")
    if not winget:
        warn("winget not available — cannot auto-install %s." % label)
        return False
    if dry_run:
        info("[dry-run] would winget install %s" % label)
        return False
    info("Installing %s (one-time, may take several minutes)..." % label)
    code = subprocess.call(
        [winget, "install", "--exact", "--source", "winget",
         "--accept-source-agreements", "--accept-package-agreements"] + args,
        stdout=subprocess.DEVNULL,
    )
    if code != 0:
        warn("%s install failed (winget exit %d)." % (label, code))
    return code == 0


def ensure_armpl(dry_run=False):
    if armpl_bin_dir():
        return True
    return _winget_install(
        ["--id", "Arm.ArmPerformanceLibraries"],
        "Arm Performance Libraries",
        dry_run=dry_run,
    )


def ensure_vcredist(dry_run=False):
    """VC runtime (msvcp140 etc.) — required by the native wheels."""
    sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    if os.path.isfile(os.path.join(sysdir, "msvcp140.dll")):
        return True
    return _winget_install(
        ["--id", "Microsoft.VCRedist.2015+.arm64"],
        "Visual C++ Redistributable (arm64)",
        dry_run=dry_run,
    )


def ensure_msvc(dry_run=False):
    """MSVC Build Tools — only needed for triton's runtime kernel launchers.

    Best effort: without it, training still works; torch.compile / triton
    JIT is unavailable until the user installs Build Tools.
    """
    if have_msvc():
        return True
    done = _winget_install(
        ["--id", "Microsoft.VisualStudio.2022.BuildTools", "--override",
         "--quiet --wait --norestart "
         "--add Microsoft.VisualStudio.Workload.VCTools "
         "--add Microsoft.VisualStudio.Component.VC.Tools.ARM64 "
         "--add Microsoft.VisualStudio.Component.Windows11SDK.26100"],
        "MSVC Build Tools (for torch.compile/triton)",
        dry_run=dry_run,
    )
    if not done and not dry_run:
        warn(
            "torch.compile/triton kernel JIT will be unavailable until MSVC "
            "Build Tools are installed; training itself is unaffected."
        )
    return done


def ensure_spark_runtime(dry_run=False):
    """Full best-effort provisioning for the native Spark stack."""
    check_cuda()
    ensure_vcredist(dry_run=dry_run)
    ensure_cudnn(dry_run=dry_run)
    ensure_armpl(dry_run=dry_run)
    ensure_msvc(dry_run=dry_run)
    if runtime_complete():
        ok("Spark native runtime present (CUDA + cuDNN + Arm PL).")
