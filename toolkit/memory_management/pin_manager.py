"""Single authority for process-local pinned host memory.

Pinned host memory is a shared budget on Windows/WDDM.  This module keeps a
per-consumer ledger and centralizes the DXGI/RAM headroom checks that used to
live in individual consumers.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

import torch


try:
    import psutil as _psutil
except Exception:
    _psutil = None


GIB = 1024 ** 3


class PinBudgetExceeded(RuntimeError):
    """Raised when a must-pin allocation cannot fit the current host-pin budget."""


class PinReleaseError(RuntimeError):
    """Raised when a registered host allocation could not be unpinned."""


@dataclass
class PinHandle:
    tensor: torch.Tensor
    nbytes: int
    kind: str
    pinned: bool
    # "alloc" = cudaHostAlloc via torch's caching host allocator (bytes only
    # return DXGI budget after _empty_host_pin_cache, and the allocator
    # rounds requests up to power-of-two buckets -- the DXGI cost can be up
    # to 2x nbytes). "register" = exact-size pageable tensor pinned with
    # cudaHostRegister (DXGI cost == nbytes, returned immediately on
    # release).
    mechanism: str = "alloc"


_LOCK = threading.RLock()
_LEDGER: dict[str, int] = {}

# Weight-tier consumers use only the fixed spill-reserve floor because their
# static commitments do not need the percentage reserve held for dynamic pins.
_WEIGHT_TIER_KINDS = ("weights",)

dxgi_meminfo = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _cuda_device_index(device=None) -> int:
    if device is None:
        return 0
    try:
        dev = torch.device(device)
    except Exception:
        return 0
    if dev.type != "cuda":
        return 0
    if dev.index is not None:
        return int(dev.index)
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def total_pinned_bytes() -> int:
    with _LOCK:
        return int(sum(_LEDGER.values()))


def pinned_bytes_by_kind() -> dict[str, int]:
    with _LOCK:
        return dict(_LEDGER)


def register_pinned_bytes(n: int, kind: str = "unknown") -> None:
    n = int(n)
    if n <= 0:
        return
    kind = str(kind or "unknown")
    with _LOCK:
        _LEDGER[kind] = _LEDGER.get(kind, 0) + n


def release_pinned_bytes(n: int, kind: str = "unknown") -> None:
    n = int(n)
    if n <= 0:
        return
    kind = str(kind or "unknown")
    with _LOCK:
        # Clamp within the kind only: an unmatched release (e.g. a consumer
        # releasing bytes it never registered because pinning was disabled)
        # must not drain other consumers' ledger entries.
        if kind in _LEDGER:
            _LEDGER[kind] = max(0, _LEDGER[kind] - n)
            if _LEDGER[kind] == 0:
                del _LEDGER[kind]


def reset_for_tests() -> None:
    with _LOCK:
        _LEDGER.clear()


def _spill_reserve_floor_gib() -> float:
    for name, default in (
        ("AI_TOOLKIT_WDDM_SPILL_RESERVE_FLOOR_GIB", None),
        ("AI_TOOLKIT_WDDM_SPILL_RESERVE_GIB", "1.0"),
    ):
        raw = os.environ.get(name)
        if raw is None:
            if default is None:
                continue
            raw = default
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
    return 2.0


def _spill_reserve_pct() -> float:
    try:
        return max(0.0, float(os.environ.get("AI_TOOLKIT_WDDM_SPILL_RESERVE_PCT", "0.10")))
    except (TypeError, ValueError):
        return 0.10


def dxgi_spill_reserve_bytes(budget_bytes: Optional[int] = None) -> int:
    floor_bytes = int(_spill_reserve_floor_gib() * GIB)
    if budget_bytes and int(budget_bytes) > 0:
        return max(floor_bytes, int(_spill_reserve_pct() * float(budget_bytes)))
    return floor_bytes


def _spill_reserve_for_kind(kind: str, budget_bytes: Optional[int]) -> int:
    # Weight-tier pins are one-shot static commitments sized at attach/enable:
    # per-tensor legacy weight pins and canonical Arena storage (kind="weights").
    # The pct-based reserve exists as slack for the *dynamic* streaming
    # consumer (the bounce pool), which grows at runtime -- a static commitment
    # does not need it and keeps only the floor.
    if kind in _WEIGHT_TIER_KINDS:
        return int(_spill_reserve_floor_gib() * GIB)
    return dxgi_spill_reserve_bytes(budget_bytes)


def get_dxgi_meminfo():
    if _env_bool("AI_TOOLKIT_WDDM_DXGI_DISABLE"):
        return None
    global dxgi_meminfo
    if dxgi_meminfo is None:
        try:
            from . import dxgi_meminfo as _dxgi_meminfo
        except Exception:
            return None
        dxgi_meminfo = _dxgi_meminfo
    return dxgi_meminfo


def dxgi_pinned_headroom(
    cuda_device_index: Optional[int] = None, *, kind: str = "unknown"
) -> Optional[int]:
    if _env_bool("AI_TOOLKIT_WDDM_DXGI_CONTROL_DISABLE"):
        return None
    dxgi = get_dxgi_meminfo()
    if dxgi is None:
        return None
    info = dxgi.query_non_local_video_memory_info(
        cuda_device_index=0 if cuda_device_index is None else int(cuda_device_index),
        min_interval_s=0.0,
    )
    if info is None:
        return None
    return dxgi.compute_non_local_headroom_bytes(
        info.budget_bytes,
        info.current_usage_bytes,
        _spill_reserve_for_kind(kind, info.budget_bytes),
    )


def pinned_bytes_headroom(
    cuda_device_index: Optional[int] = None, *, kind: str = "unknown"
) -> Optional[int]:
    headroom = dxgi_pinned_headroom(cuda_device_index, kind=kind)
    if headroom is not None:
        return headroom
    if _psutil is None:
        return None
    try:
        total = _psutil.virtual_memory().total
    except Exception:
        return None
    try:
        fraction = float(os.environ.get("AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION", "0.25"))
    except (TypeError, ValueError):
        fraction = 0.25
    if fraction <= 0:
        return None
    return max(0, int(total * fraction) - total_pinned_bytes())


def host_cache_reserve_bytes(mode: Optional[str] = None) -> int:
    if mode == "sampling":
        default = "0.0"
    else:
        default = os.environ.get("AI_TOOLKIT_PIN_HOST_CACHE_RESERVE_GIB", "1.0")
    try:
        return int(max(0.0, float(default)) * GIB)
    except (TypeError, ValueError):
        return 0


def available_for_pin(
    *,
    kind: str = "unknown",
    nbytes: int = 0,
    device=None,
    reserve_bytes: int = 0,
    mode: Optional[str] = None,
) -> Optional[int]:
    reserve = max(0, int(reserve_bytes or 0)) + host_cache_reserve_bytes(mode)
    headroom = pinned_bytes_headroom(_cuda_device_index(device), kind=kind)
    if headroom is None:
        return None
    return max(0, int(headroom) - reserve)


def _empty_host_pin_cache() -> None:
    for name in ("_host_emptyCache", "_accelerator_emptyHostCache"):
        try:
            fn = getattr(torch._C, name, None)
            if fn is not None:
                fn()
        except Exception:
            pass


def reconcile() -> None:
    """Return cached host-pin allocations before retrying a pin request."""
    _empty_host_pin_cache()


_REGISTERED_HOST_PIN_LOCK = threading.Lock()
_REGISTERED_HOST_PINS: dict[int, tuple[int, str]] = {}


def pin_tensor_in_place(t: torch.Tensor, kind: str = "weights", *, device=None) -> bool:
    """Pin an existing CPU tensor storage with cudaHostRegister when possible."""
    if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.is_pinned():
        return False
    size = int(t.numel() * t.element_size())
    if size <= 0 or not torch.cuda.is_available():
        return False
    available = available_for_pin(kind=kind, nbytes=size, device=device)
    if available is not None and size > available:
        reconcile()
        available = available_for_pin(kind=kind, nbytes=size, device=device)
        if available is not None and size > available:
            return False
    try:
        from torch.cuda import _pin_memory_utils as pin_memory_utils
        ptr = int(t.data_ptr())
        if ptr == 0:
            return False
        pin_memory_utils.pin_memory(ptr, size)
    except Exception:
        return False
    with _REGISTERED_HOST_PIN_LOCK:
        _REGISTERED_HOST_PINS[int(t.data_ptr())] = (size, str(kind or "unknown"))
    register_pinned_bytes(size, kind)
    return True


def is_host_pinned(t: torch.Tensor) -> bool:
    """True if this tensor's storage is usable as pinned host memory.

    torch's ``is_pinned()`` only recognizes buffers allocated by its own
    caching host allocator; memory pinned in place with cudaHostRegister
    (``pin_tensor_in_place`` / register-mechanism pinning -- the
    weight/arena tier)
    reports ``is_pinned() == False`` even though CUDA treats it as pinned for
    transfer purposes. Consult the registration table too so canonical Arena
    consumers recognize registered flats rather than falsely treating them as
    pageable.
    """
    if not isinstance(t, torch.Tensor):
        return False
    try:
        if t.is_pinned():
            return True
    except Exception:
        pass
    with _REGISTERED_HOST_PIN_LOCK:
        return int(t.data_ptr()) in _REGISTERED_HOST_PINS


def unpin_tensor_in_place(t: torch.Tensor, kind: Optional[str] = None) -> bool:
    if not isinstance(t, torch.Tensor):
        return False
    ptr = int(t.data_ptr())
    with _REGISTERED_HOST_PIN_LOCK:
        entry = _REGISTERED_HOST_PINS.pop(ptr, None)
    if entry is None:
        return False
    size, registered_kind = entry
    try:
        from torch.cuda import _pin_memory_utils as pin_memory_utils
        pin_memory_utils.unpin_memory(ptr)
    except Exception:
        with _REGISTERED_HOST_PIN_LOCK:
            _REGISTERED_HOST_PINS[ptr] = (size, registered_kind)
        return False
    release_pinned_bytes(size, kind or registered_kind)
    return True


# Storage-base data_ptrs owned by canonical arena flats. Registration is now a
# residency policy: streamed blocks and the next known demotion candidates are
# pinned, while other fully resident blocks remain pageable. This registry says
# only "canonical source storage," not "currently pinned"; callers that require
# direct async H2D must also consult is_host_pinned. Refcounting keeps ownership
# correct when an allocator later recycles the same storage base pointer.
_ARENA_BACKED_STORAGE_LOCK = threading.Lock()
_ARENA_BACKED_STORAGE_PTRS: dict[int, int] = {}


def _storage_base_ptr(t: torch.Tensor) -> Optional[int]:
    if not isinstance(t, torch.Tensor):
        return None
    try:
        if t.device.type != "cpu":
            return None
        ptr = int(t.untyped_storage().data_ptr())
    except Exception:
        return None
    return ptr if ptr != 0 else None


def register_arena_storage(t: torch.Tensor) -> None:
    """Mark a canonical arena flat's storage independently of pinnedness."""
    ptr = _storage_base_ptr(t)
    if ptr is None:
        return
    with _ARENA_BACKED_STORAGE_LOCK:
        _ARENA_BACKED_STORAGE_PTRS[ptr] = _ARENA_BACKED_STORAGE_PTRS.get(ptr, 0) + 1


def unregister_arena_storage(t: torch.Tensor) -> None:
    ptr = _storage_base_ptr(t)
    if ptr is None:
        return
    with _ARENA_BACKED_STORAGE_LOCK:
        count = _ARENA_BACKED_STORAGE_PTRS.get(ptr)
        if count is None:
            return
        if count <= 1:
            _ARENA_BACKED_STORAGE_PTRS.pop(ptr, None)
        else:
            _ARENA_BACKED_STORAGE_PTRS[ptr] = count - 1


def is_arena_backed(t: torch.Tensor) -> bool:
    """True if this CPU tensor is a view into canonical arena storage."""
    ptr = _storage_base_ptr(t)
    if ptr is None:
        return False
    with _ARENA_BACKED_STORAGE_LOCK:
        return ptr in _ARENA_BACKED_STORAGE_PTRS


def release(handle: PinHandle) -> None:
    """Return a pin_alloc grant to the ledger.

    Accounting is explicit and deterministic: every pinned PinHandle must be
    released exactly once by its owner (a GC finalizer on the handle's tensor
    was tried and rejected -- callers immediately re-view the tensor, so the
    original Python object dies while the pinned storage lives on)."""
    if handle is None or not getattr(handle, "pinned", False):
        return
    if getattr(handle, "mechanism", "alloc") == "register":
        # unpin_tensor_in_place does the ledger release itself (and the
        # cudaHostUnregister returns DXGI budget immediately).
        if not unpin_tensor_in_place(
            handle.tensor, getattr(handle, "kind", None)
        ):
            raise PinReleaseError(
                f"cudaHostUnregister failed for {getattr(handle, 'kind', 'unknown')}"
            )
    else:
        release_pinned_bytes(int(handle.nbytes), getattr(handle, "kind", "unknown"))
    handle.pinned = False
    handle.nbytes = 0

def pin_alloc(
    nbytes: int,
    kind: str,
    *,
    device=None,
    required: bool = True,
    reserve_bytes: int = 0,
    mode: Optional[str] = None,
) -> PinHandle:
    nbytes = int(nbytes)
    kind = str(kind or "unknown")
    if nbytes <= 0:
        tensor = torch.empty(max(0, nbytes), dtype=torch.uint8)
        return PinHandle(tensor=tensor, nbytes=0, kind=kind, pinned=False)
    if not torch.cuda.is_available():
        tensor = torch.empty(nbytes, dtype=torch.uint8)
        return PinHandle(tensor=tensor, nbytes=nbytes, kind=kind, pinned=False)

    available = available_for_pin(
        kind=kind, nbytes=nbytes, device=device, reserve_bytes=reserve_bytes, mode=mode
    )
    if available is not None and nbytes > available:
        reconcile()
        available = available_for_pin(
            kind=kind, nbytes=nbytes, device=device, reserve_bytes=reserve_bytes, mode=mode
        )
        if available is not None and nbytes > available:
            if not required:
                print(
                    f"[PinManager] pin refused ({kind}): {nbytes / GIB:.2f} GiB "
                    f"> {available / GIB:.2f} GiB available (ledger/DXGI budget); "
                    "returning pageable"
                )
                tensor = torch.empty(nbytes, dtype=torch.uint8)
                return PinHandle(tensor=tensor, nbytes=nbytes, kind=kind, pinned=False)
            raise PinBudgetExceeded(_budget_message(kind, nbytes, available, device=device))
    try:
        tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    except RuntimeError:
        reconcile()
        try:
            tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except RuntimeError as error:
            if not required:
                print(
                    f"[PinManager] pin failed at OS level ({kind}): "
                    f"{nbytes / GIB:.2f} GiB cudaHostAlloc/pin raised "
                    f"'{error}' (ledger said {'' if available is None else f'{available / GIB:.2f} GiB '}"
                    "available -- likely system RAM pressure); returning pageable"
                )
                tensor = torch.empty(nbytes, dtype=torch.uint8)
                return PinHandle(tensor=tensor, nbytes=nbytes, kind=kind, pinned=False)
            raise PinBudgetExceeded(_budget_message(kind, nbytes, available, device=device)) from error
    register_pinned_bytes(nbytes, kind)
    return PinHandle(tensor=tensor, nbytes=nbytes, kind=kind, pinned=True)


def pin_register_prepare(nbytes: int) -> tuple[torch.Tensor, int]:
    """Allocate the page-aligned pageable buffer a register-mechanism pin
    will need, WITHOUT pinning it yet.

    Split into prepare/commit steps so callers can populate the buffer (for
    example, copying leaf tensors into a block flat) on ordinary
    pageable memory -- a plain memcpy that faults in pages at normal RAM
    bandwidth -- before ``cudaHostRegister`` runs. Registering a virgin,
    never-touched buffer forces the OS to commit+pin every page during the
    syscall itself, which is measurably slower than registering pages that
    are already resident, saving roughly 1-1.5 seconds per full arena build.

    Returns ``(candidate, padded_nbytes)``; ``candidate`` is an untouched
    pageable view with the layout required by ``pin_register_commit``.
    """
    nbytes = int(nbytes)
    if nbytes <= 0:
        return torch.empty(0, dtype=torch.uint8), 0
    page = 4096
    padded = (nbytes + page - 1) // page * page
    # cudaHostRegister works at PAGE granularity: it registers every 4096-byte
    # page the range touches. Two buffers that share a page (small buffers
    # from the same allocator arena, or the boundary page between adjacent
    # mallocs) collide -- registering the second raises CUDA "resource already
    # mapped" (the 763bb75 root cause). Guarantee the registered range's pages
    # are exclusive to THIS allocation: over-allocate with a full slack page on
    # each side and register only the page-aligned interior, so no neighbor
    # allocation can own a page we register. The slack bases stay alive with
    # the returned tensor (the view keeps the base storage referenced).
    base = torch.empty(padded + 3 * page, dtype=torch.uint8)
    base_ptr = base.data_ptr()
    # First page boundary at least one full page into the allocation.
    aligned_start = ((base_ptr + page + page - 1) // page) * page
    offset = aligned_start - base_ptr
    candidate = base[offset:offset + padded]
    return candidate, padded


def pin_register_commit(
    candidate: torch.Tensor,
    nbytes: int,
    kind: str,
    *,
    device=None,
    required: bool = False,
) -> PinHandle:
    """Pin an already-prepared (and optionally already-populated) buffer
    from :func:`pin_register_prepare` with cudaHostRegister."""
    nbytes = int(nbytes)
    kind = str(kind or "unknown")
    if nbytes <= 0 or not torch.cuda.is_available():
        return PinHandle(tensor=candidate, nbytes=0, kind=kind,
                         pinned=False, mechanism="register")
    padded = candidate.numel() * candidate.element_size()
    # pin_tensor_in_place budget-checks (with reconcile) and does the ledger
    # accounting (of the padded size -- the true DXGI cost).
    if pin_tensor_in_place(candidate, kind, device=device):
        return PinHandle(tensor=candidate, nbytes=padded, kind=kind,
                         pinned=True, mechanism="register")
    if required:
        raise PinBudgetExceeded(
            _budget_message(
                kind, nbytes,
                available_for_pin(kind=kind, nbytes=nbytes, device=device),
                device=device,
            )
        )
    print(
        f"[PinManager] pin refused ({kind}): {nbytes / GIB:.2f} GiB "
        "cudaHostRegister denied (ledger/DXGI budget); returning pageable"
    )
    return PinHandle(tensor=candidate, nbytes=nbytes, kind=kind, pinned=False,
                     mechanism="register")


def snapshot(device=None, mode: Optional[str] = None) -> dict:
    headroom = pinned_bytes_headroom(_cuda_device_index(device))
    reserve = host_cache_reserve_bytes(mode)
    by_kind = pinned_bytes_by_kind()
    total = sum(by_kind.values())
    return {
        "pinned_total_bytes": total,
        "pinned_by_kind": by_kind,
        "headroom_bytes": headroom,
        "host_cache_reserve_bytes": reserve,
        "available_bytes": None if headroom is None else max(0, int(headroom) - reserve),
    }


def _budget_message(kind: str, nbytes: int, available: Optional[int], *, device=None) -> str:
    snap = snapshot(device=device)
    return (
        f"pinned host memory budget exceeded for {kind}: "
        f"request={nbytes / GIB:.2f} GiB "
        f"available={'unknown' if available is None else f'{available / GIB:.2f} GiB'} "
        f"ledger={snap['pinned_by_kind']}"
    )
