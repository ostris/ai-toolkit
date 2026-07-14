"""Cross-platform NVML sensor for TRUE physical VRAM occupancy.

Why this exists
---------------
``torch.cuda.mem_get_info`` does NOT report physical card availability. It
reports what the driver is willing to promise *this process*, which on WDDM is
a budget the OS intends to satisfy by paging other processes out. Measured on
an RTX 4070 (11.99 GiB) while a second process held 9 GiB:

    nvidia-smi / NVML physical free : 0.55 GiB   <- the truth
    torch.cuda.mem_get_info free    : 4.70 GiB   <- over-reports ~8x
    DXGI LOCAL Budget - CurrentUsage: 4.58 GiB   <- also over-reports

Planning residency against the optimistic numbers is exactly how a run silently
crosses the dedicated-VRAM cliff: the allocation "succeeds", WDDM pages to
system RAM, and throughput collapses with no error and no allocator retry to
detect it (observed: a training job at 304 s/it instead of 73 s/it because an
orphaned process was squatting on the card).

NVML reports device-wide physical occupancy across *every* process, so it stays
correct when a game, a ComfyUI server, or a leftover job shares the GPU. It is
also the only such sensor that works on both Windows and Linux -- DXGI is
Windows-only and, per the numbers above, measures a permission rather than an
availability.

This module is ctypes-only and adds no dependency: it binds ``nvml.dll`` on
Windows and ``libnvidia-ml.so.1`` on Linux, both shipped with the NVIDIA driver.
Every public function returns ``None`` when NVML is unavailable (no NVIDIA
driver, non-NVIDIA GPU) so callers can fall back to their previous behavior.

Cost: ``nvmlDeviceGetMemoryInfo`` measures ~3 us/call, ~25x cheaper than
``torch.cuda.mem_get_info`` (~80 us), so it is safe to call on hot paths.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import NamedTuple

_NVML_SUCCESS = 0


class NvmlMemoryInfo(NamedTuple):
    total_bytes: int
    free_bytes: int
    used_bytes: int


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


_lock = threading.Lock()
_lib = None
_init_failed = False
_handles: dict[int, ctypes.c_void_p] = {}
_warned: set[str] = set()


def _log_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(message)


def _load_library():
    names = ("nvml.dll",) if os.name == "nt" else ("libnvidia-ml.so.1", "libnvidia-ml.so")
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _library():
    """Load + initialize NVML once. None when unavailable."""
    global _lib, _init_failed
    if _lib is not None:
        return _lib
    if _init_failed:
        return None
    lib = _load_library()
    if lib is None:
        _init_failed = True
        _log_once(
            "nvml_missing",
            "[NVML] library not found; falling back to torch.cuda.mem_get_info "
            "(device-free readings will not see other processes)",
        )
        return None
    try:
        rc = lib.nvmlInit_v2()
    except Exception as exc:  # pragma: no cover - driver-level failure
        _init_failed = True
        _log_once("nvml_init_raise", f"[NVML] init raised: {exc}; using mem_get_info")
        return None
    if rc != _NVML_SUCCESS:
        _init_failed = True
        _log_once("nvml_init_failed", f"[NVML] nvmlInit_v2 failed rc={rc}; using mem_get_info")
        return None
    _lib = lib
    return _lib


def _torch_device_uuid(cuda_device_index: int) -> str | None:
    """UUID of a CUDA device as torch sees it, normalized. None if unavailable."""
    try:
        import torch

        raw = getattr(torch.cuda.get_device_properties(cuda_device_index), "uuid", None)
    except Exception:
        return None
    if raw is None:
        return None
    return str(raw).replace("GPU-", "").replace("-", "").lower()


def _nvml_device_uuid(lib, handle: ctypes.c_void_p) -> str | None:
    buf = ctypes.create_string_buffer(96)
    try:
        rc = lib.nvmlDeviceGetUUID(handle, buf, ctypes.c_uint(96))
    except Exception:  # pragma: no cover - symbol missing on ancient drivers
        return None
    if rc != _NVML_SUCCESS:
        return None
    return buf.value.decode(errors="replace").replace("GPU-", "").replace("-", "").lower()


def _resolve_handle(lib, cuda_device_index: int) -> ctypes.c_void_p | None:
    """Map a CUDA device index to an NVML handle.

    Prefer UUID matching: NVML enumerates *physical* devices, while CUDA indices
    are affected by CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER, so index-to-index
    identity is not guaranteed on a multi-GPU box. Fall back to the index only
    when UUIDs are unavailable.
    """
    count = ctypes.c_uint()
    if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != _NVML_SUCCESS:
        return None

    want = _torch_device_uuid(cuda_device_index)
    if want:
        for idx in range(int(count.value)):
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(idx), ctypes.byref(handle)) != _NVML_SUCCESS:
                continue
            if _nvml_device_uuid(lib, handle) == want:
                return handle
        _log_once(
            "uuid_no_match",
            f"[NVML] no NVML device matched CUDA device {cuda_device_index} by UUID; "
            "falling back to index mapping",
        )

    if cuda_device_index >= int(count.value):
        return None
    handle = ctypes.c_void_p()
    if lib.nvmlDeviceGetHandleByIndex_v2(
        ctypes.c_uint(cuda_device_index), ctypes.byref(handle)
    ) != _NVML_SUCCESS:
        return None
    return handle


def _handle_for(cuda_device_index: int) -> ctypes.c_void_p | None:
    with _lock:
        cached = _handles.get(cuda_device_index)
        if cached is not None:
            return cached
        lib = _library()
        if lib is None:
            return None
        try:
            handle = _resolve_handle(lib, cuda_device_index)
        except Exception as exc:  # pragma: no cover
            _log_once("handle_raise", f"[NVML] device resolution raised: {exc}")
            return None
        if handle is None:
            _log_once(
                "handle_missing",
                f"[NVML] could not resolve NVML handle for CUDA device {cuda_device_index}",
            )
            return None
        _handles[cuda_device_index] = handle
        return handle


def query_device_memory_info(cuda_device_index: int = 0) -> NvmlMemoryInfo | None:
    """Physical VRAM totals for a device, across ALL processes on the GPU.

    ``free_bytes`` is the real bytes left on the card -- unlike
    ``torch.cuda.mem_get_info``, it accounts for other processes (a game, a
    ComfyUI server, an orphaned training job). Returns None when NVML is
    unavailable, so callers must keep a fallback.
    """
    lib = _library()
    if lib is None:
        return None
    handle = _handle_for(cuda_device_index)
    if handle is None:
        return None
    mem = _NvmlMemory()
    try:
        rc = lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
    except Exception as exc:  # pragma: no cover
        _log_once("query_raise", f"[NVML] memory query raised: {exc}")
        return None
    if rc != _NVML_SUCCESS:
        _log_once("query_failed", f"[NVML] nvmlDeviceGetMemoryInfo failed rc={rc}")
        return None
    return NvmlMemoryInfo(
        total_bytes=int(mem.total),
        free_bytes=int(mem.free),
        used_bytes=int(mem.used),
    )


def physical_free_bytes(cuda_device_index: int = 0) -> int | None:
    """True physical free bytes on the card, or None when NVML is unavailable."""
    info = query_device_memory_info(cuda_device_index)
    return None if info is None else info.free_bytes


def is_available() -> bool:
    """True when NVML can be used to read physical occupancy."""
    return _library() is not None
