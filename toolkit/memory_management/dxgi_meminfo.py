"""Windows DXGI video-memory budget sensor.

This module is intentionally ctypes-only and Windows-only. On non-Windows, or
when DXGI is unavailable, public query functions return None so callers can use
their existing host-RAM proxy.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import NamedTuple, Optional


_DXGI_ERROR_NOT_FOUND = 0x887A0002
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
_DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1
_NVIDIA_VENDOR_ID = 0x10DE


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


IID_IDXGIFactory1 = GUID(
    0x770AAE78,
    0xF26F,
    0x4DBA,
    (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)
IID_IDXGIAdapter3 = GUID(
    0x645967A4,
    0x1392,
    0x4310,
    (ctypes.c_ubyte * 8)(0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD),
)


class LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", ctypes.c_uint32),
        ("HighPart", ctypes.c_int32),
    ]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", ctypes.c_uint32),
    ]


class _DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


class DxgiMemoryInfo(NamedTuple):
    budget_bytes: int
    current_usage_bytes: int
    available_for_reservation_bytes: int
    current_reservation_bytes: int


# Which adapter-match methods are trustworthy enough to *drive control*
# (aggressive pin-for-speed), versus only telemetry. Auto-detected confident
# matches (single_nvidia, and the Stage-B luid match) are safe. An explicit
# env override is honored for control but is a human-forced adapter, so it is
# logged distinctly as "manual" -- a misconfiguration must be visible, not
# silently trusted as if auto-verified. Everything else (sole_hardware_adapter,
# global_conservative, or no adapter at all) stays conservative.
_CONTROL_SAFE_METHODS = frozenset({"single_nvidia", "luid"})
_CONTROL_MANUAL_METHODS = frozenset({"env_override"})


def safe_for_control(match_method: Optional[str]) -> bool:
    """Is a reading from this match method trustworthy enough to drive control?

    Pure classifier (unit-testable, no ctypes). True for confidently
    auto-detected adapters and for an explicit manual override; False for
    ambiguous/fallback selections and when the sensor is unavailable.
    """
    if not match_method:
        return False
    return match_method in _CONTROL_SAFE_METHODS or match_method in _CONTROL_MANUAL_METHODS


def is_manual_control(match_method: Optional[str]) -> bool:
    """True when control is driven off a human-forced (env-override) adapter."""
    return bool(match_method) and match_method in _CONTROL_MANUAL_METHODS


class DxgiAdapterInfo(NamedTuple):
    index: int
    description: str
    vendor_id: int
    device_id: int
    luid: str
    match_method: str
    safe_for_control: bool
    manual_control: bool


class DxgiAdapterRecord(NamedTuple):
    index: int
    description: str
    vendor_id: int
    device_id: int
    luid: str
    flags: int
    dedicated_video_memory_bytes: int
    dedicated_system_memory_bytes: int
    shared_system_memory_bytes: int
    is_software: bool


class _AdapterSelection(NamedTuple):
    ptr: ctypes.c_void_p
    info: DxgiAdapterInfo


_selection_lock = threading.Lock()
_selection: Optional[_AdapterSelection] = None
_selection_unavailable = False
_query_lock = threading.Lock()
_query_cache: dict[tuple[int, int], tuple[float, DxgiMemoryInfo]] = {}
_warned: set[str] = set()


def _log_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(message)


def _failed(hr: int) -> bool:
    return int(hr) < 0


def _hr_u32(hr: int) -> int:
    return int(hr) & 0xFFFFFFFF


def _luid_text(luid: LUID) -> str:
    return f"{int(luid.HighPart) & 0xFFFFFFFF:08x}:{int(luid.LowPart):08x}"


def _com_call(ptr, vtbl_index, restype, argtypes, *args):
    """Dispatch a COM vtable call against ``ptr``."""
    if not ptr:
        raise RuntimeError("null COM pointer")
    winfunctype = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
    raw_ptr = ctypes.c_void_p(ptr) if isinstance(ptr, int) else ptr
    vtbl = ctypes.cast(raw_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    address = vtbl[vtbl_index]
    prototype = winfunctype(restype, ctypes.c_void_p, *argtypes)
    return prototype(address)(raw_ptr, *args)


def _create_dxgi_factory1():
    if os.name != "nt":
        return None
    factory = ctypes.c_void_p()
    hr = ctypes.windll.dxgi.CreateDXGIFactory1(
        ctypes.byref(IID_IDXGIFactory1),
        ctypes.byref(factory),
    )
    if _failed(hr):
        raise OSError(f"CreateDXGIFactory1 failed hr=0x{_hr_u32(hr):08x}")
    return factory


def _get_desc1(adapter) -> DXGI_ADAPTER_DESC1:
    desc = DXGI_ADAPTER_DESC1()
    # IDXGIAdapter1::GetDesc1: IUnknown 0-2, IDXGIObject 3-6,
    # IDXGIAdapter 7-9, IDXGIAdapter1 slot 10.
    hr = _com_call(
        adapter,
        10,
        ctypes.c_long,
        [ctypes.POINTER(DXGI_ADAPTER_DESC1)],
        ctypes.byref(desc),
    )
    if _failed(hr):
        raise OSError(f"IDXGIAdapter1::GetDesc1 failed hr=0x{_hr_u32(hr):08x}")
    return desc


def _record_from_desc(index: int, desc: DXGI_ADAPTER_DESC1) -> DxgiAdapterRecord:
    return DxgiAdapterRecord(
        index=index,
        description=str(desc.Description).rstrip("\x00"),
        vendor_id=int(desc.VendorId),
        device_id=int(desc.DeviceId),
        luid=_luid_text(desc.AdapterLuid),
        flags=int(desc.Flags),
        dedicated_video_memory_bytes=int(desc.DedicatedVideoMemory),
        dedicated_system_memory_bytes=int(desc.DedicatedSystemMemory),
        shared_system_memory_bytes=int(desc.SharedSystemMemory),
        is_software=bool(int(desc.Flags) & _DXGI_ADAPTER_FLAG_SOFTWARE),
    )


def _enum_adapters1(factory) -> list[tuple[ctypes.c_void_p, DxgiAdapterRecord]]:
    adapters = []
    index = 0
    while True:
        adapter = ctypes.c_void_p()
        # IDXGIFactory1::EnumAdapters1: IUnknown 0-2, IDXGIObject 3-6,
        # IDXGIFactory 7-11, IDXGIFactory1 slot 12.
        hr = _com_call(
            factory,
            12,
            ctypes.c_long,
            [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
            ctypes.c_uint32(index),
            ctypes.byref(adapter),
        )
        if _hr_u32(hr) == _DXGI_ERROR_NOT_FOUND:
            break
        if _failed(hr):
            raise OSError(f"IDXGIFactory1::EnumAdapters1 failed hr=0x{_hr_u32(hr):08x}")
        desc = _get_desc1(adapter)
        adapters.append((adapter, _record_from_desc(index, desc)))
        index += 1
    return adapters


def enumerate_adapters() -> list[DxgiAdapterRecord]:
    """Return all DXGI adapters for diagnostics/probes, or [] if unavailable."""
    if os.name != "nt":
        return []
    try:
        factory = _create_dxgi_factory1()
        if factory is None:
            return []
        return [record for _adapter, record in _enum_adapters1(factory)]
    except Exception as exc:
        _log_once(
            "enumerate_failed",
            f"[DXGI] adapter enumeration unavailable: {exc}",
        )
        return []


def _query_interface_adapter3(adapter) -> ctypes.c_void_p:
    adapter3 = ctypes.c_void_p()
    # IUnknown::QueryInterface slot 0.
    hr = _com_call(
        adapter,
        0,
        ctypes.c_long,
        [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)],
        ctypes.byref(IID_IDXGIAdapter3),
        ctypes.byref(adapter3),
    )
    if _failed(hr):
        raise OSError(f"IDXGIAdapter3 QueryInterface failed hr=0x{_hr_u32(hr):08x}")
    return adapter3


def _select_hardware_adapter3() -> Optional[_AdapterSelection]:
    if os.name != "nt":
        return None

    factory = _create_dxgi_factory1()
    adapters = _enum_adapters1(factory)
    if not adapters:
        _log_once("no_adapters", "[DXGI] no adapters found; using legacy pin proxy")
        return None

    override = os.environ.get("AI_TOOLKIT_WDDM_DXGI_ADAPTER_INDEX", "").strip()
    if override:
        try:
            override_index = int(override)
            for adapter, record in adapters:
                if record.index == override_index:
                    return _selection_from_record(
                        adapter, record, match_method="env_override"
                    )
        except Exception:
            pass
        _log_once(
            "bad_override",
            "[DXGI] AI_TOOLKIT_WDDM_DXGI_ADAPTER_INDEX did not match a DXGI "
            "adapter; using legacy pin proxy",
        )
        return None

    hardware = [
        (adapter, record)
        for adapter, record in adapters
        if not record.is_software
    ]
    nvidia = [
        (adapter, record)
        for adapter, record in hardware
        if record.vendor_id == _NVIDIA_VENDOR_ID
    ]
    if len(nvidia) == 1:
        adapter, record = nvidia[0]
        return _selection_from_record(adapter, record, match_method="single_nvidia")
    if len(nvidia) == 0 and len(hardware) == 1:
        adapter, record = hardware[0]
        _log_once(
            "sole_hardware_adapter",
            "[DXGI] no NVIDIA adapter found; using sole hardware adapter for "
            "shared-memory pin budget",
        )
        return _selection_from_record(
            adapter, record, match_method="sole_hardware_adapter"
        )
    _log_once(
        "ambiguous_adapter",
        "[DXGI] adapter selection is ambiguous in Stage A; using legacy pin proxy",
    )
    return None


def _selection_from_record(adapter, record: DxgiAdapterRecord, match_method: str):
    adapter3 = _query_interface_adapter3(adapter)
    manual = is_manual_control(match_method)
    if manual:
        # A human forced the adapter via AI_TOOLKIT_WDDM_DXGI_ADAPTER_INDEX. It
        # still drives control, but log it distinctly so a misconfiguration is
        # visible rather than lumped in with confident auto-detection.
        _log_once(
            "manual_control_override",
            f"[DXGI] manual adapter override in effect (index={record.index} "
            f"{record.description!r}); driving control as 'manual' -- verify "
            "this is the training GPU",
        )
    info = DxgiAdapterInfo(
        index=record.index,
        description=record.description,
        vendor_id=record.vendor_id,
        device_id=record.device_id,
        luid=record.luid,
        match_method=match_method,
        safe_for_control=safe_for_control(match_method),
        manual_control=manual,
    )
    return _AdapterSelection(adapter3, info)


def _selected_adapter() -> Optional[_AdapterSelection]:
    global _selection, _selection_unavailable
    if os.name != "nt":
        return None
    with _selection_lock:
        if _selection is not None:
            return _selection
        if _selection_unavailable:
            return None
        try:
            _selection = _select_hardware_adapter3()
        except Exception as exc:
            _log_once(
                "selection_failed",
                f"[DXGI] adapter selection unavailable: {exc}; using legacy pin proxy",
            )
            _selection = None
        if _selection is None:
            _selection_unavailable = True
        return _selection


def selected_adapter_info() -> Optional[DxgiAdapterInfo]:
    selection = _selected_adapter()
    return selection.info if selection is not None else None


def control_is_eligible(cuda_device_index: int = 0) -> bool:
    """True when the resolved adapter is trustworthy enough to drive control.

    Consumed by the pin-for-speed policy: aggressive full-pin engages only when
    this is True (confident auto-detect or an explicit manual override). When
    DXGI is unavailable/ambiguous this is False and the policy stays
    conservative. ``cuda_device_index`` is accepted for the Stage-B per-device
    match; Stage A resolves the single cached adapter.
    """
    del cuda_device_index  # Stage A: single cached adapter.
    info = selected_adapter_info()
    return bool(info is not None and info.safe_for_control)


def query_video_memory_info(
    cuda_device_index: int = 0,
    segment_group: int = _DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    min_interval_s: float = 0.5,
) -> Optional[DxgiMemoryInfo]:
    """Query LOCAL or NON_LOCAL DXGI video-memory info for the selected adapter."""
    del cuda_device_index  # Stage A uses one cached adapter; Stage B matches per CUDA device.
    if os.name != "nt":
        return None
    selection = _selected_adapter()
    if selection is None:
        return None
    cache_key = (0, int(segment_group))
    now = time.monotonic()
    with _query_lock:
        cached = _query_cache.get(cache_key)
        if cached is not None and min_interval_s > 0:
            ts, info = cached
            if now - ts < min_interval_s:
                return info
    try:
        raw = _DXGI_QUERY_VIDEO_MEMORY_INFO()
        # IDXGIAdapter3::QueryVideoMemoryInfo: IUnknown 0-2, IDXGIObject 3-6,
        # IDXGIAdapter 7-9, IDXGIAdapter1 10, IDXGIAdapter2 11-13,
        # IDXGIAdapter3 slot 14.
        hr = _com_call(
            selection.ptr,
            14,
            ctypes.c_long,
            [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(_DXGI_QUERY_VIDEO_MEMORY_INFO)],
            ctypes.c_uint32(0),
            ctypes.c_int(int(segment_group)),
            ctypes.byref(raw),
        )
        if _failed(hr):
            raise OSError(
                f"IDXGIAdapter3::QueryVideoMemoryInfo failed hr=0x{_hr_u32(hr):08x}"
            )
        info = DxgiMemoryInfo(
            budget_bytes=int(raw.Budget),
            current_usage_bytes=int(raw.CurrentUsage),
            available_for_reservation_bytes=int(raw.AvailableForReservation),
            current_reservation_bytes=int(raw.CurrentReservation),
        )
    except Exception as exc:
        _log_once(
            "query_failed",
            f"[DXGI] video-memory query unavailable: {exc}; using legacy pin proxy",
        )
        return None
    with _query_lock:
        _query_cache[cache_key] = (now, info)
    return info


def query_non_local_video_memory_info(
    cuda_device_index: int = 0,
    min_interval_s: float = 0.5,
) -> Optional[DxgiMemoryInfo]:
    return query_video_memory_info(
        cuda_device_index=cuda_device_index,
        segment_group=_DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
        min_interval_s=min_interval_s,
    )


def query_local_video_memory_info(
    cuda_device_index: int = 0,
    min_interval_s: float = 0.5,
) -> Optional[DxgiMemoryInfo]:
    return query_video_memory_info(
        cuda_device_index=cuda_device_index,
        segment_group=_DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        min_interval_s=min_interval_s,
    )


def compute_non_local_headroom_bytes(
    budget_bytes: int,
    current_usage_bytes: int,
    spill_reserve_bytes: int,
) -> int:
    return max(
        0,
        int(budget_bytes) - int(current_usage_bytes) - max(0, int(spill_reserve_bytes)),
    )
