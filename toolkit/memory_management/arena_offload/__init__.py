"""Arena offload: the block-native weight-streaming backend.

This package is the only supported integration surface for arena offload.
Model integrations and the shared trainer must go through `api`; they must not
construct `CanonicalArena`, `ResidencyState`, `ResidencyPlan`, or the immutable
runtime directly.

Dependency rule (three tiers):

    host_memory (pin_manager, vram_budget, nvml_meminfo, dxgi_meminfo)
        imports neither backend

    arena_offload  -> may import host_memory; must NOT import MemoryManager
    MemoryManager  -> may import host_memory; must NOT import arena_offload

Arena planning, transfer, FP8 transforms, and lifecycle cleanup are owned here;
the legacy manager remains a separate backend.
"""

__all__ = [
    "ArenaOffloadConfig",
    "close_arena_offload",
    "get_arena_runtime",
    "estimate_training_working_reserve_hint_bytes",
    "is_arena_offloaded",
    "model_load_arena_session",
    "prepare_canonical_storage",
    "prepare_canonical_storage_from_state_dict",
    "prepare_arena_offload",
    "validate_arena_training_mode",
]

_PUBLIC_ENTRY_POINTS = {
    "ArenaOffloadConfig": (".api", "ArenaOffloadConfig"),
    "close_arena_offload": (".api", "close_arena_offload"),
    "get_arena_runtime": (".api", "get_arena_runtime"),
    "estimate_training_working_reserve_hint_bytes": (
        ".api",
        "estimate_training_working_reserve_hint_bytes",
    ),
    "is_arena_offloaded": (".api", "is_arena_offloaded"),
    "model_load_arena_session": (".load_session", "model_load_arena_session"),
    "prepare_canonical_storage": (".api", "prepare_canonical_storage"),
    "prepare_canonical_storage_from_state_dict": (
        ".api",
        "prepare_canonical_storage_from_state_dict",
    ),
    "prepare_arena_offload": (".api", "prepare_arena_offload"),
    "validate_arena_training_mode": (".api", "validate_arena_training_mode"),
}


def __getattr__(name):
    entry_point = _PUBLIC_ENTRY_POINTS.get(name)
    if entry_point is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attribute_name = entry_point
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
