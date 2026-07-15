"""Shared WDDM allocator-cap mechanism for both memory backends."""

from __future__ import annotations

import sys

import torch

from . import vram_budget

GIB = 1024 ** 3
APPLIED_FRACTIONS: dict[int, float] = {}
RELIEF_BYTES: dict[int, int] = {}
CAP_RELIEF_BYTES = int(0.5 * GIB)


def _cuda_index(device) -> int | None:
    if not torch.cuda.is_available():
        return None
    dev = torch.device(device if device is not None else "cuda")
    if dev.type != "cuda":
        return None
    return dev.index if dev.index is not None else torch.cuda.current_device()


def tracked_allocator_fraction(device) -> float | None:
    """Return only allocator policy previously installed by Toolkit."""
    index = _cuda_index(device)
    return None if index is None else APPLIED_FRACTIONS.get(index)


def restore_tracked_allocator_fraction(device, previous: float | None) -> None:
    """Restore a Toolkit-owned allocator setting captured before arena setup."""
    index = _cuda_index(device)
    if index is None:
        return
    current = APPLIED_FRACTIONS.get(index)
    if previous is None:
        if current is not None and current < 1.0:
            torch.cuda.set_per_process_memory_fraction(1.0, index)
        APPLIED_FRACTIONS.pop(index, None)
        RELIEF_BYTES.pop(index, None)
        return
    previous = float(previous)
    if current is None or abs(current - previous) > 1e-12:
        torch.cuda.set_per_process_memory_fraction(previous, index)
    APPLIED_FRACTIONS[index] = previous




def wddm_cliff_cap_bytes(device, wddm_hard_gib=None) -> int:
    """Return the governing allocator cap at the current WDDM cliff."""
    dev = torch.device(device if device is not None else "cuda")
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        hard_gib = float(wddm_hard_gib) if wddm_hard_gib is not None else 1.0
    except (TypeError, ValueError):
        hard_gib = 1.0
    if hard_gib <= 0:
        hard_gib = 1.0
    total = vram_budget.device_total_bytes(index)
    free_bytes, _ = vram_budget.device_mem_info(index)
    reserved_bytes = torch.cuda.memory_reserved(index)
    return int(
        vram_budget.cap_fraction(
            total, free_bytes, reserved_bytes, hard_gib
        )
        * total
    )




def applied_cap_bytes(device) -> int | None:
    """Return the last cap bound on this device in physical allocator bytes."""
    if not torch.cuda.is_available():
        return None
    dev = torch.device(device if device is not None else "cuda")
    if dev.type != "cuda":
        return None
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    fraction = APPLIED_FRACTIONS.get(index)
    if fraction is None:
        return None
    return int(float(fraction) * vram_budget.real_device_total_bytes(index))


def configure_wddm_allocator_guard(
    device,
    wddm_hard_gib=None,
    *,
    target_cap_bytes=None,
    strict=False,
    log_prefix="[MemoryManager]",
):
    """Bind the allocator below the WDDM cliff in every guard mode.

    ``strict`` controls how the caller handles a cap rejection; it must not
    disable the cap or the FSM's allocator steering. Call only at a phase
    boundary. A simulated governing capacity is always converted against the
    physical card total.
    """
    if sys.platform != "win32" or not torch.cuda.is_available():
        return None
    dev = torch.device(device if device is not None else "cuda")
    if dev.type != "cuda":
        return None
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        hard_gib = float(wddm_hard_gib) if wddm_hard_gib is not None else 1.0
    except (TypeError, ValueError):
        hard_gib = 1.0
    if hard_gib <= 0:
        hard_gib = 1.0
    total = vram_budget.device_total_bytes(index)
    real_total = vram_budget.real_device_total_bytes(index)
    free_bytes, _governing_total = vram_budget.device_mem_info(index)
    reserved_bytes = torch.cuda.memory_reserved(index)
    cliff_fraction = vram_budget.cap_fraction(
        total, free_bytes, reserved_bytes, hard_gib
    )
    fraction = cliff_fraction
    reclaimed = False
    if target_cap_bytes is not None:
        target_fraction = float(target_cap_bytes) / float(total)
        fraction = max(0.1, min(cliff_fraction, target_fraction))
        reclaimed = fraction < cliff_fraction - 1e-9

    relief_bytes = RELIEF_BYTES.get(index, 0)
    if relief_bytes:
        fraction = min(1.0, fraction + relief_bytes / float(total))

    applied = fraction * total / float(real_total)
    previous = APPLIED_FRACTIONS.get(index)
    tolerance = (64 * 1024**2) / real_total
    if previous is not None and abs(previous - applied) < tolerance:
        return previous

    torch.cuda.set_per_process_memory_fraction(applied, index)
    APPLIED_FRACTIONS[index] = applied
    non_torch = max(0, (total - free_bytes) - reserved_bytes)
    source = (
        f"reclaim target, cliff {cliff_fraction * total / GIB:.2f} GiB"
        if reclaimed
        else "cliff bound"
    )
    if total != real_total:
        source += f"; SIMULATED {total / GIB:.2f} GiB card"
    if relief_bytes:
        source += f"; +{relief_bytes / GIB:.2f} GiB recovery relief"
    print(
        f"{log_prefix} WDDM allocator cap: "
        f"{fraction * total / GIB:.2f}/{total / GIB:.2f} GiB "
        f"({source}; margin {hard_gib:.2f} GiB, "
        f"non_torch {non_torch / GIB:.2f} GiB; allocation beyond this "
        "recycles cache or raises OOM instead of silently paging)"
    )
    return applied


def relieve_wddm_allocator_guard_after_oom(
    device, *, strict=False, context="training step", log_prefix="[MemoryManager]"
) -> bool:
    """Widen a capped allocator only when non-strict recovery has no layout relief."""
    if strict or sys.platform != "win32" or not torch.cuda.is_available():
        return False
    dev = torch.device(device if device is not None else "cuda")
    if dev.type != "cuda":
        return False
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    applied = APPLIED_FRACTIONS.get(index)
    if applied is None or applied >= 1.0:
        return False

    real_total = vram_budget.real_device_total_bytes(index)
    relief = RELIEF_BYTES.get(index, 0) + CAP_RELIEF_BYTES
    widened = min(1.0, applied + CAP_RELIEF_BYTES / float(real_total))
    torch.cuda.set_per_process_memory_fraction(widened, index)
    APPLIED_FRACTIONS[index] = widened
    RELIEF_BYTES[index] = relief
    print(
        f"{log_prefix} allocator cap rejected {context}: no resident layout "
        f"relief remained, widening {applied * real_total / GIB:.2f}->"
        f"{widened * real_total / GIB:.2f} GiB so the non-strict job can "
        "continue"
    )
    return True
