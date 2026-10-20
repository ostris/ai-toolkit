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


KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _gfx_from_target_version(value):
    # KFD packs the ISA as major*10000 + minor*100 + stepping, and the
    # stepping is printed in hex: 110501 -> gfx1151, 90010 -> gfx90a
    return "gfx%d%d%x" % (value // 10000, (value // 100) % 100, value % 100)


def _rocm_gpus_linux():
    """[{name, gfx}] from the amdgpu KFD topology; needs only the kernel driver."""
    gpus = []
    try:
        nodes = sorted(os.listdir(KFD_NODES), key=lambda n: int(n) if n.isdigit() else 0)
    except OSError:
        return gpus
    for node in nodes:
        props = {}
        try:
            with open(os.path.join(KFD_NODES, node, "properties")) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        props[parts[0]] = int(parts[1])
        except OSError:
            continue
        # CPU nodes have no SIMDs
        if not props.get("simd_count") or not props.get("gfx_target_version"):
            continue
        gfx = _gfx_from_target_version(props["gfx_target_version"])
        gpus.append({"name": "AMD GPU (%s)" % gfx, "gfx": gfx})
    smi = which("rocm-smi")
    out = _run_quiet([smi, "--showproductname"]) if smi else None
    names = re.findall(r"Card [Ss]eries:\s*(.+)", out or "")
    if names and len(names) == len(gpus):
        for gpu, name in zip(gpus, names):
            gpu["name"] = name.strip()
    elif names and not gpus:
        gpus = [{"name": name.strip()} for name in names]
    return gpus


def _rocm_gpus_windows():
    """[{name, gfx, memory}] from clinfo, which the AMD display driver installs."""
    smi = which("clinfo")
    out = _run_quiet([smi]) if smi else None
    gpus = []
    if not out:
        return gpus
    # one block per device; each starts at its "Device Type:" line
    for block in re.split(r"\n\s*Device Type:", out)[1:]:
        if not block.lstrip().startswith("CL_DEVICE_TYPE_GPU"):
            continue
        if "Advanced Micro Devices" not in block:
            continue
        gfx = re.search(r"^\s*Name:\s*(gfx[0-9a-f]+)", block, re.M)
        if not gfx:
            continue
        board = re.search(r"^\s*Board name:\s*(.+)$", block, re.M)
        mem = re.search(r"^\s*Global memory size:\s*(\d+)", block, re.M)
        gpu = {
            "name": board.group(1).strip() if board else "AMD GPU",
            "gfx": gfx.group(1),
        }
        if mem:
            gpu["memory"] = "%d MiB" % (int(mem.group(1)) // (1024 * 1024))
        gpus.append(gpu)
    return gpus


def detect_rocm():
    """Returns dict if an AMD GPU usable by ROCm PyTorch is present, else None.

    ROCm PyTorch wheels bundle the runtime, so only the GPU driver is needed:
    the amdgpu KFD topology on Linux, the Adrenalin driver's clinfo on Windows.
    """
    if IS_LINUX:
        gpus = _rocm_gpus_linux()
        if not gpus and os.path.isdir("/opt/rocm"):
            return {"gpus": []}
    elif IS_WINDOWS:
        gpus = _rocm_gpus_windows()
    else:
        return None
    if not gpus:
        return None
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
    return result
