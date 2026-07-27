"""Hardware / platform detection. Stdlib only, safe to run anywhere."""

import os
import platform
import re
import subprocess

from .util import which, IS_WINDOWS, IS_MAC, IS_LINUX


def _run_quiet(cmd):
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15
        )
        if out.returncode != 0:
            return None
        return out.stdout.decode("utf-8", errors="replace")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def detect_nvidia():
    """Returns dict with gpus + driver info, or None if no working nvidia-smi."""
    smi = which("nvidia-smi")
    if not smi:
        return None
    fields = "name,memory.total,driver_version,compute_cap"
    csv = _run_quiet([smi, "--query-gpu=" + fields, "--format=csv,noheader"])
    if not csv:
        # older drivers don't know the compute_cap field
        csv = _run_quiet(
            [
                smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
    if not csv:
        return None
    gpus = []
    driver = None
    for line in csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpu = {"name": parts[0], "memory": parts[1]}
            if len(parts) >= 4:
                gpu["compute_cap"] = parts[3]
            gpus.append(gpu)
            driver = parts[2]
    if not gpus:
        return None
    # Max CUDA version the driver supports only appears in the banner output.
    # Windows/WDDM labels it "CUDA UMD Version:" on recent drivers, Linux just
    # "CUDA Version:" — without the optional word we silently fall through to
    # the "driver present but version unknown, assume current" path and can
    # hand an old driver cu130 wheels it cannot run.
    banner = _run_quiet([smi]) or ""
    m = re.search(r"CUDA(?:\s+[A-Z]+)?\s+Version:\s*([0-9]+\.[0-9]+)", banner)
    cuda_version = m.group(1) if m else None
    return {"gpus": gpus, "driver": driver, "cuda_version": cuda_version}


def detect_rocm():
    """Returns dict if an AMD ROCm stack is present, else None."""
    if not IS_LINUX:
        return None
    smi = which("rocm-smi")
    if not smi and not os.path.isdir("/opt/rocm"):
        return None
    gpus = []
    out = _run_quiet([smi, "--showproductname"]) if smi else None
    if out:
        for m in re.finditer(r"Card [Ss]eries:\s*(.+)", out):
            gpus.append({"name": m.group(1).strip()})
    return {"gpus": gpus}


def detect():
    """Full platform detection. Returns a plain dict (json-serializable)."""
    system = platform.system()  # Linux / Darwin / Windows
    arch = platform.machine().lower()  # x86_64 / amd64 / arm64 / aarch64
    if arch == "amd64":
        arch = "x86_64"
    if arch == "arm64" and not IS_MAC:
        arch = "aarch64"

    result = {
        "os": {"Linux": "linux", "Darwin": "mac", "Windows": "windows"}.get(
            system, system.lower()
        ),
        "arch": arch,
        "python": platform.python_version(),
        "nvidia": None,
        "rocm": None,
        "backend": "cpu",
    }

    nvidia = detect_nvidia()
    if nvidia:
        result["nvidia"] = nvidia
        result["backend"] = "cuda"
    else:
        rocm = detect_rocm()
        if rocm:
            result["rocm"] = rocm
            result["backend"] = "rocm"

    if IS_MAC:
        result["backend"] = "mps" if arch == "arm64" else "cpu"

    # DGX OS / Grace (GB10, DGX Spark): NVIDIA GPU on aarch64 Linux
    result["is_dgx"] = bool(result["os"] == "linux" and arch == "aarch64" and nvidia)
    return result
