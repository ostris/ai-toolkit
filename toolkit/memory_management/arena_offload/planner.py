"""Cold-start, whole-block residency planning owned by arena offload."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from toolkit.quantization.storage import temporary_materialization_bytes

from .. import vram_budget
from .layout import flatten_leaves

GIB = 1024**3
DEFAULT_AUTO_WORKING_RESERVE_GIB = 5.0
# Full residency removes the transfer ring, but training still needs activation,
# adapter, dequantization, and allocator-fragmentation headroom. Keep a
# conservative cold-start reserve on the target 12 GiB class of devices;
# explicit policy values remain authoritative for measured workloads.
DEFAULT_ALL_RESIDENT_WORKING_RESERVE_GIB = 4.0
DEFAULT_RESIDENT_FLOOR_GIB = 2.0


def resolve_physical_vram_headroom_gib(
    device, value, *, hard_gib=0.0
) -> float:
    try:
        margin = float(value)
        automatic = margin < 0
    except (TypeError, ValueError):
        automatic = value is None or str(value).strip().lower() == "auto"
        margin = -1.0
    if automatic:
        margin = max(
            float(vram_budget.auto_physical_vram_headroom_gib(device)),
            float(hard_gib or 0.0),
        )
    return max(0.0, float(margin))


def training_pinned_keys_for_keep_last(model, keep_last, block_keys=None) -> set[str]:
    keep_last = max(0, int(keep_last or 0))
    if block_keys is not None:
        ordered = tuple(str(key) for key in block_keys)
        return set(ordered[max(0, len(ordered) - keep_last):])
    blocks = getattr(model, "blocks", None)
    if blocks is None or keep_last <= 0:
        return set()
    start = max(0, len(blocks) - keep_last)
    return {f"blocks.{index}" for index in range(start, len(blocks))}


def _tensor_storage_bytes(value) -> int:
    try:
        leaves = flatten_leaves(value)
    except Exception:
        leaves = (value,)
    return sum(
        int(leaf.numel() * leaf.element_size())
        for leaf in leaves
        if isinstance(leaf, torch.Tensor) and leaf.device.type != "meta"
    )


def _singleton_stats(model, canonical_modules) -> tuple[int, int, set[int]]:
    canonical_ids = {id(module) for module in canonical_modules}
    seen_parameters = set()
    total = 0
    largest_materialization = 0
    runtime_ids = set()
    for module in model.modules():
        if id(module) in canonical_ids:
            continue
        direct = tuple(module.parameters(recurse=False))
        if direct:
            runtime_ids.add(id(module))
        for parameter in direct:
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            total += _tensor_storage_bytes(parameter.data)
            largest_materialization = max(
                largest_materialization,
                temporary_materialization_bytes(parameter.data),
            )
    return total, largest_materialization, runtime_ids


def _planning_records(arena_or_build):
    """Return the record surface needed by the cold planner.

    A prepared canonical build exposes final allocation sizes and module/leaf
    membership before it publishes any Parameter views. Planning against that
    surface keeps predictable admission failures on the rollback-safe side of
    the canonical commit boundary.
    """
    if hasattr(arena_or_build, "block_keys"):
        return [
            arena_or_build.block_record(key)
            for key in arena_or_build.block_keys()
            if arena_or_build.block_record(key) is not None
        ]
    blocks = getattr(arena_or_build, "blocks", None)
    if blocks is None:
        raise TypeError("arena planner requires an arena or prepared build")
    return [
        SimpleNamespace(
            block_key=block.key,
            committed_bytes=int(block.layout.nbytes),
            modules=tuple(module for _name, module in block.entries),
            leaf_names=tuple(name for name, _module in block.entries),
        )
        for block in blocks
    ]


def impossible_training_plan_message(plan) -> str:
    """Describe a minimum-layout admission failure with actionable budgets."""
    return (
        "arena training minimum layout does not fit before canonical commit: "
        f"required_bytes={int(plan['minimum_required_bytes'])}, "
        f"available_bytes={int(plan['available_bytes'])}, "
        f"reserve_bytes={int(plan['working_reserve_bytes'])}, "
        f"ring_bytes={int(plan['ring_bytes'])}, "
        f"singleton_bytes={int(plan['singleton_resident_bytes'])}, "
        f"mandatory_resident_bytes={int(plan['pinned_resident_bytes'])}, "
        "physical_vram_headroom_bytes="
        f"{int(plan['physical_vram_headroom_bytes'])}"
    )


def build_training_plan(
    model, arena_or_build, canonical_modules, device, config, *, block_keys=None
) -> dict:
    """Choose an initial whole-block layout without the legacy manager."""
    device = torch.device(device)
    policy = config._policy
    try:
        working_value = float(policy.working_reserve_gib)
        automatic = working_value < 0
    except (TypeError, ValueError):
        automatic = policy.working_reserve_gib is None or str(
            policy.working_reserve_gib
        ).strip().lower() == "auto"
        working_value = DEFAULT_AUTO_WORKING_RESERVE_GIB
    if automatic:
        working_value = DEFAULT_AUTO_WORKING_RESERVE_GIB

    hard_gib = float(policy.wddm_hard_gib or 1.0)
    physical_headroom_gib = resolve_physical_vram_headroom_gib(
        device, policy.physical_vram_headroom_gib, hard_gib=hard_gib
    )
    free_bytes, total_bytes = vram_budget.device_mem_info(device)
    working_bytes = int(max(0.0, working_value) * GIB)
    physical_headroom_bytes = int(physical_headroom_gib * GIB)
    hard_bytes = int(hard_gib * GIB)

    singleton_bytes, largest_singleton_dequant, runtime_ids = _singleton_stats(
        model, canonical_modules
    )
    records = _planning_records(arena_or_build)
    block_bytes = sum(record.committed_bytes for record in records)
    all_resident_working_bytes = working_bytes
    if automatic:
        all_resident_working_bytes = min(
            working_bytes,
            int(DEFAULT_ALL_RESIDENT_WORKING_RESERVE_GIB * GIB),
        )
    all_resident_fit = (
        singleton_bytes + block_bytes + all_resident_working_bytes
        <= max(0, int(free_bytes) - physical_headroom_bytes)
    )
    if all_resident_fit:
        # A transfer ring and the generic 5 GiB cold-start reserve are both
        # counterproductive when the complete compressed model plus the
        # evidence-backed execution reserve fits below the physical WDDM margin.
        working_bytes = all_resident_working_bytes

    pinned_keys = training_pinned_keys_for_keep_last(
        model, policy.checkpoint_keep_last, block_keys=block_keys
    )
    resident_keys = (
        {record.block_key for record in records}
        if all_resident_fit
        else {
            record.block_key
            for record in records
            if record.block_key in pinned_keys
        }
    )
    streamed = [record for record in records if record.block_key not in resident_keys]
    largest_stream = max((record.committed_bytes for record in streamed), default=0)
    ring_bytes = largest_stream * max(1, int(policy.prefetch_depth))
    usable = max(
        0, int(free_bytes) - physical_headroom_bytes - working_bytes
    )
    resident_budget = max(0, usable - singleton_bytes - ring_bytes)
    resident_bytes = sum(
        record.committed_bytes for record in records if record.block_key in resident_keys
    )
    floor = min(block_bytes, int(DEFAULT_RESIDENT_FLOOR_GIB * GIB))

    candidates = sorted(
        (record for record in records if record.block_key not in resident_keys),
        key=lambda record: (record.committed_bytes, record.block_key),
    )
    target = (
        block_bytes
        if all_resident_fit
        else resident_budget if not automatic else min(resident_budget, floor)
    )
    for record in candidates:
        if resident_bytes >= target:
            break
        if resident_bytes + record.committed_bytes > resident_budget:
            continue
        resident_keys.add(record.block_key)
        resident_bytes += record.committed_bytes

    offload_ids = {
        id(module)
        for record in records
        if record.block_key not in resident_keys
        for module in record.modules
    }
    protected = frozenset(
        (record.block_key, leaf_name)
        for record in records
        if record.block_key in pinned_keys
        for leaf_name in record.leaf_names
    )
    try:
        device_used = torch.cuda.device_memory_used(device)
        torch_reserved = torch.cuda.memory_reserved(device)
        system_reserve = max(0, int(device_used) - int(torch_reserved))
    except Exception:
        system_reserve = max(0, int(total_bytes) - int(free_bytes))

    pinned_resident_bytes = sum(
        record.committed_bytes
        for record in records
        if record.block_key in pinned_keys
    )
    available_bytes = max(0, int(free_bytes) - physical_headroom_bytes)
    minimum_required_bytes = (
        singleton_bytes + pinned_resident_bytes + ring_bytes + working_bytes
    )

    return {
        "offload_ids": offload_ids,
        "offloaded_layers": len(offload_ids),
        "candidate_layers": sum(len(record.modules) for record in records),
        "model_bytes": singleton_bytes + block_bytes,
        "resident_bytes": singleton_bytes + resident_bytes,
        "must_resident_bytes": singleton_bytes + pinned_resident_bytes,
        "must_resident_layer_keys": set(),
        "pinned_resident_bytes": pinned_resident_bytes,
        "pinned_resident_keys": set(pinned_keys),
        "protected_training_leaf_keys": protected,
        "generic_resident_bytes": max(0, resident_bytes),
        "ring_bytes": ring_bytes,
        "gpu_stream_need_bytes": ring_bytes,
        "gpu_stream_budget_bytes": ring_bytes,
        "working_reserve_bytes": working_bytes,
        "physical_vram_headroom_bytes": physical_headroom_bytes,
        "wddm_hard_bytes": hard_bytes,
        "system_reserve_bytes": system_reserve,
        "usable_bytes": usable,
        "free_bytes": int(free_bytes),
        "fits": minimum_required_bytes <= available_bytes,
        "minimum_required_bytes": minimum_required_bytes,
        "available_bytes": available_bytes,
        "singleton_resident_bytes": singleton_bytes,
        "largest_singleton_bf16_dequant_bytes": largest_singleton_dequant,
        "singleton_runtime_ids": runtime_ids,
        "auto_working_reserve": automatic,
        "all_resident_fit": all_resident_fit,
        "all_resident_working_reserve_bytes": all_resident_working_bytes,
    }
