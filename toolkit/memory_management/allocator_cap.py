"""Shared WDDM allocator-cap mechanism for both memory backends."""

from __future__ import annotations

import sys

import torch

from . import vram_budget

GIB = 1024 ** 3
APPLIED_FRACTIONS: dict[int, float] = {}
RELIEF_BYTES: dict[int, int] = {}




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


def apply_wddm_hard_allocator_cap(
    device,
    wddm_hard_gib=None,
    *,
    target_cap_bytes=None,
    log_prefix="[MemoryManager]",
):
    """Bind torch's allocator below the WDDM dedicated-memory cliff.

    Call only at a phase boundary. The governing capacity may be a simulated
    smaller card, but torch's fraction is always converted against the physical
    card total.
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
    if relief_bytes:
        source += f"; +{relief_bytes / GIB:.2f} GiB post-violation relief"
    if total != real_total:
        source += f"; SIMULATED {total / GIB:.2f} GiB card"
    print(
        f"{log_prefix} WDDM hard allocator cap: "
        f"{fraction * total / GIB:.2f}/{total / GIB:.2f} GiB "
        f"({source}; margin {hard_gib:.2f} GiB, "
        f"non_torch {non_torch / GIB:.2f} GiB; allocation beyond this "
        "recycles cache or raises OOM instead of silently paging)"
    )
    return applied
