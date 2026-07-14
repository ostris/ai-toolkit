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

from .api import (
    ArenaOffloadConfig,
    close_arena_offload,
    get_arena_runtime,
    is_arena_offloaded,
    is_memory_managed,
    memory_runtime_owns_compile,
    prepare_canonical_storage,
    prepare_canonical_storage_from_state_dict,
    prepare_arena_offload,
    validate_arena_training_mode,
)
from .runtime import ArenaOffloadRuntime
from .dispatcher import DISPATCHER_GENERATION
from .discovery import BlockDiscoveryError, discover_blocks
from .errors import ArenaCleanupError, ArenaSetupFatalError
from .load_session import model_load_arena_session
from ..runtime import close_memory_runtime, get_memory_runtime

__all__ = [
    "ArenaOffloadConfig",
    "ArenaCleanupError",
    "ArenaOffloadRuntime",
    "ArenaSetupFatalError",
    "BlockDiscoveryError",
    "DISPATCHER_GENERATION",
    "close_arena_offload",
    "close_memory_runtime",
    "get_arena_runtime",
    "get_memory_runtime",
    "discover_blocks",
    "is_arena_offloaded",
    "is_memory_managed",
    "memory_runtime_owns_compile",
    "model_load_arena_session",
    "prepare_canonical_storage",
    "prepare_canonical_storage_from_state_dict",
    "prepare_arena_offload",
    "validate_arena_training_mode",
]
