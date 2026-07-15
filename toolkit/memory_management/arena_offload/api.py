"""Public integration surface for arena offload.

Everything a model integration or the shared trainer is allowed to touch lives
here. The rule the rest of the codebase must follow:

    from toolkit.memory_management.arena_offload import (
        prepare_arena_offload, get_arena_runtime, ...
    )

and nothing else. In particular, no `CanonicalArena`, `ResidencyState`,
`ResidencyPlan` or dispatcher-controller imports outside this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
import warnings

from ..runtime import (
    RUNTIME_ATTR,
    close_memory_runtime,
    get_memory_runtime,
    is_memory_managed,
    memory_runtime_owns_compile,
    unwrap_memory_model,
)
from .runtime import ArenaOffloadRuntime

GIB = 1024**3
_FP8_QTYPES = ("qfloat8", "float8")
_COMPATIBILITY_ALIASES = {
    "layer_offloading_smart_working_reserve_gb": (
        "layer_offloading_smart_headroom_gb",
    ),
    "layer_offloading_smart_physical_vram_headroom_gb": (
        "layer_offloading_smart_wddm_margin_gb",
        "layer_offloading_smart_buffer_gb",
    ),
    "layer_offloading_smart_wddm_hard_gb": (
        "layer_offloading_smart_hard_buffer_gb",
    ),
    "layer_offloading_smart_sampling_working_reserve_gb": (
        "layer_offloading_smart_sampling_headroom_gb",
    ),
    "layer_offloading_smart_sampling_physical_vram_headroom_gb": (
        "layer_offloading_smart_sampling_wddm_margin_gb",
        "layer_offloading_smart_sampling_buffer_gb",
    ),
    "layer_offloading_smart_sampling_wddm_hard_gb": (
        "layer_offloading_smart_sampling_hard_buffer_gb",
    ),
}


def estimate_training_working_reserve_hint_bytes(
    dataset_configs, *, batch_size: int = 1
) -> int | None:
    """Estimate the worst configured image-training working set.

    Dataset ``resolution`` is an area target: a 1024 bucket is approximately
    1024**2 pixels regardless of aspect ratio. Diffusion transformers normally
    see one token per 16x16 image pixels after VAE and patch compression.
    Planning from the largest configured bucket prevents earlier low-resolution
    steps from licensing residency that the later bucket cannot support.
    """
    max_image_tokens = 0
    for dataset in dataset_configs or ():
        resolution = getattr(dataset, "resolution", 0)
        if isinstance(resolution, Sequence) and not isinstance(
            resolution, (str, bytes)
        ):
            candidates = resolution
        else:
            candidates = (resolution,)
        for candidate in candidates:
            try:
                side = max(0, int(candidate))
            except (TypeError, ValueError):
                continue
            image_tokens = (side * side + 255) // 256
            max_image_tokens = max(max_image_tokens, image_tokens)
    if max_image_tokens <= 0:
        return None

    from ..vram_budget import estimate_training_working_reserve_bytes

    batch = max(1, int(batch_size or 1))
    return estimate_training_working_reserve_bytes(
        max_image_tokens * batch,
        text_tokens=512 * batch,
    )


def validate_arena_training_mode(
    *,
    full_finetune=False,
    mutates_base_weights=False,
    train_text_encoder=False,
    unload_text_encoder=True,
) -> None:
    """Reject mutable-base configurations before model loading.

    Canonical arena leaves are immutable frozen base weights. Full-parameter
    training or merge-in save workflows require a different storage
    architecture, independent of the model or quantization integration.
    """
    if full_finetune:
        raise ValueError(
            "arena offload requires frozen base transformer weights and does "
            "not support full-model fine-tuning"
        )
    if mutates_base_weights:
        raise ValueError(
            "arena offload requires immutable base transformer weights and "
            "does not support merge_network_on_save"
        )
    if train_text_encoder or not unload_text_encoder:
        raise ValueError(
            "arena offload does not support a text encoder during training; "
            "cache text embeddings and unload the text encoder"
        )


def unwrap(model):
    """Peel Accelerate / DDP / torch.compile wrappers without importing them.

    The arena package must stay importable from a bare CPU test process, so this
    does not go through `toolkit.accelerator.unwrap_model` (which constructs a
    global `Accelerator`).
    """
    return unwrap_memory_model(model)


@dataclass(frozen=True)
class _ArenaPolicyOptions:
    """Internal policy inputs retained while fork job aliases are migrated."""

    working_reserve_gib: float | None = None
    physical_vram_headroom_gib: float | None = None
    wddm_hard_gib: float | None = None
    checkpoint_keep_last: int = 0
    prefetch_depth: int = 3

    sampling_working_reserve_gib: float | None = None
    sampling_physical_vram_headroom_gib: float | None = None
    sampling_wddm_hard_gib: float | None = 1.0


@dataclass(frozen=True)
class ArenaOffloadConfig:
    """The narrow public configuration surface of arena offload.

    Fields prefixed with ``_`` are derived integration details, not additional
    user-facing arena controls.
    """

    enabled: bool = False
    fp8_forward: bool = False
    fp8_backward: bool = False
    fp8_sampling: bool = False
    compile_blocks: bool = False
    strict_vram_cap: bool = False
    _compile_dynamic: bool | None = True
    _compile_dynamic_hints: tuple[tuple[int, int | None, int | None], ...] = ()
    # Validation knob: pretend the card is this many GiB, so small-card
    # behaviour (deeper streaming, tighter caps, a residency plan that cannot
    # fit) is exercisable on a bigger one. 0/None = use the real card.
    _simulated_vram_gib: float | None = None
    _policy: _ArenaPolicyOptions = field(
        default_factory=_ArenaPolicyOptions, repr=False
    )

    @classmethod
    def from_model_config(
        cls, model_config, *, training_working_reserve_hint_bytes: int | None = None
    ) -> ArenaOffloadConfig:
        def get(name: str, default: Any = None) -> Any:
            if hasattr(model_config, name):
                return getattr(model_config, name)
            for alias in _COMPATIBILITY_ALIASES.get(name, ()):
                if hasattr(model_config, alias):
                    return getattr(model_config, alias)
            return default

        raw_working_reserve_gib = get("layer_offloading_smart_working_reserve_gb")
        working_reserve_gib = raw_working_reserve_gib
        if training_working_reserve_hint_bytes:
            try:
                is_auto = raw_working_reserve_gib is None or float(raw_working_reserve_gib) < 0
            except (TypeError, ValueError):
                is_auto = str(raw_working_reserve_gib).strip().lower() == "auto"
            if is_auto:
                # A caller-supplied, resolution-aware hint (see
                # vram_budget.estimate_training_working_reserve_bytes) beats the
                # planner's flat DEFAULT_AUTO_WORKING_RESERVE_GIB fallback, but
                # never overrides an explicit user value.
                working_reserve_gib = float(training_working_reserve_hint_bytes) / float(GIB)

        fp8_weights = bool(get("quantize", False)) and get("qtype") in _FP8_QTYPES
        requested_forward = bool(get("layer_offloading_fp8_forward", False))
        requested_backward = bool(get("layer_offloading_fp8_grad_input", False))
        requested_sampling = bool(get("layer_offloading_fp8_sampling", False))
        ignored = []
        if not fp8_weights:
            ignored.extend(
                name
                for name, requested in (
                    ("fp8_forward", requested_forward),
                    ("fp8_backward", requested_backward),
                    ("fp8_sampling", requested_sampling),
                )
                if requested
            )
        elif requested_backward and not requested_forward:
            ignored.append("fp8_backward_without_fp8_forward")
        if ignored:
            warnings.warn(
                "arena offload ignored irrelevant FP8 options: "
                + ", ".join(ignored),
                RuntimeWarning,
                stacklevel=2,
            )

        return cls(
            enabled=bool(
                get("layer_offloading", False)
                and get("layer_offloading_smart", False)
            ),
            fp8_forward=fp8_weights and requested_forward,
            fp8_backward=fp8_weights
            and requested_forward
            and requested_backward,
            fp8_sampling=fp8_weights
            and requested_sampling,
            # Arena execution has one shared block dispatcher for training
            # and sampling, so Toolkit's supported model compile setting owns
            # both phases.
            compile_blocks=bool(get("compile", False)),
            strict_vram_cap=bool(
                get("layer_offloading_strict_vram_cap", False)
            ),
            _compile_dynamic=(
                None
                if get("compile_dynamic", True) is None
                else bool(get("compile_dynamic", True))
            ),
            _compile_dynamic_hints=tuple(
                tuple(hint) for hint in (get("compile_dynamic_hints", ()) or ())
            ),
            _simulated_vram_gib=(
                float(get("layer_offloading_simulated_vram_gb") or 0.0) or None
            ),
            _policy=_ArenaPolicyOptions(
                working_reserve_gib=working_reserve_gib,
                physical_vram_headroom_gib=get(
                    "layer_offloading_smart_physical_vram_headroom_gb"
                ),
                wddm_hard_gib=get("layer_offloading_smart_wddm_hard_gb"),
                checkpoint_keep_last=max(
                    0, int(get("layer_offloading_checkpoint_keep_last", 0) or 0)
                ),
                prefetch_depth=int(get("layer_offloading_prefetch_depth", 3) or 3),
                sampling_working_reserve_gib=get(
                    "layer_offloading_smart_sampling_working_reserve_gb"
                ),
                sampling_physical_vram_headroom_gib=get(
                    "layer_offloading_smart_sampling_physical_vram_headroom_gb"
                ),
                sampling_wddm_hard_gib=get(
                    "layer_offloading_smart_sampling_wddm_hard_gb", 1.0
                ),
            ),
        )



def prepare_canonical_storage(
    transformer,
    *,
    block_names: Sequence[str] | None = None,
    device=None,
    defer_blocks: bool = False,
):
    """Prepare final arena destinations without publishing model Parameters."""
    from ..canonical_arena import CanonicalArena
    from .discovery import discover_blocks
    from .resources import ArenaRuntimeResources

    selection = discover_blocks(
        transformer, container_paths=tuple(block_names or ())
    )
    resources = None
    if device is not None:
        resources = ArenaRuntimeResources(transformer, device)
        resources.acquire_process_owner()
    try:
        entries = (
            {}
            if defer_blocks
            else {
                key: list(selection.entries_by_block[key])
                for key in selection.block_keys
            }
        )
        arena = CanonicalArena()
        build = arena.prepare(
            entries,
            model=transformer,
            pin_on_finish=False,
        )
        if resources is not None:
            resources.adopt_canonical_build(build)
        return build
    except BaseException:
        if resources is not None:
            resources.release()
        raise


def prepare_canonical_storage_from_state_dict(
    transformer,
    state_dict,
    *,
    block_names: Sequence[str] | None = None,
    device=None,
):
    """Build canonical storage incrementally from a mutable state mapping.

    Serialized source keys and physical tensor leaves are inferred from the
    transformer's module paths, each module's own state-dict surface, and its
    quantization storage declaration. The complete mapping is validated before
    allocation or destructive consumption begins.
    """
    from .discovery import discover_blocks

    selection = discover_blocks(
        transformer, container_paths=tuple(block_names or ())
    )
    entries_by_block = selection.entries_by_block

    from .construction import infer_state_dict_schema, validate_state_dict_schema

    schema = infer_state_dict_schema(transformer, entries_by_block)
    validate_state_dict_schema(state_dict, schema)
    build = prepare_canonical_storage(
        transformer,
        block_names=block_names,
        device=device,
        defer_blocks=True,
    )
    build.populate_from_state_dict_consuming(
        state_dict, blocks=entries_by_block.items()
    )
    return build


def prepare_arena_offload(
    transformer,
    *,
    device,
    config: ArenaOffloadConfig,
    block_names: Sequence[str] | None = None,
    ignore_modules: Sequence[Any] | None = None,
    canonical_build=None,
) -> ArenaOffloadRuntime:
    """Canonicalize the model's execution blocks and prepare the arena runtime.

    Call after the base weights are final and BEFORE the training network is
    applied. Loaders that populate final arena destinations directly pass their
    populated ``canonical_build``; other models use the compatibility source,
    which copies from the already-materialized model. The runtime comes back
    unfinalized; the trainer calls ``finalize()`` once the network is installed.

    The runtime is published on `transformer._arena_offload_runtime`.
    """
    if not config.enabled:
        raise ValueError("arena_offload_not_enabled")
    from .load_session import claim_pending_canonical_build

    pending_build = claim_pending_canonical_build(transformer)
    if canonical_build is None:
        canonical_build = pending_build
    elif pending_build is not None:
        pending_build.rollback()
        canonical_build.rollback()
        raise ValueError("arena_multiple_pending_canonical_builds")
    try:
        from .discovery import discover_blocks

        if not bool(getattr(transformer, "gradient_checkpointing", False)):
            raise ValueError(
                "arena training requires model gradient checkpointing before "
                "canonical storage is committed"
            )
        configured_keep_last = int(config._policy.checkpoint_keep_last)
        model_keep_last = getattr(transformer, "_checkpoint_keep_last", None)
        if model_keep_last is None:
            if configured_keep_last:
                raise ValueError(
                    "arena checkpoint keep-last requires a model-owned "
                    "keep-last declaration"
                )
        elif int(model_keep_last) != configured_keep_last:
            raise ValueError(
                "arena checkpoint keep-last does not match the model-owned "
                f"value: config={configured_keep_last} model={int(model_keep_last)}"
            )
        selection = discover_blocks(
            transformer,
            container_paths=tuple(block_names or ()),
        )
    except BaseException:
        if canonical_build is not None:
            canonical_build.rollback()
        raise
    return ArenaOffloadRuntime._prepare(
        transformer,
        device=device,
        selection=selection,
        config=config,
        ignore_modules=ignore_modules,
        canonical_build=canonical_build,
    )


def get_arena_runtime(model) -> ArenaOffloadRuntime | None:
    """The arena runtime for `model`, or None. Unwraps Accelerate/DDP/compile."""
    return get_memory_runtime(model)


def is_arena_offloaded(model) -> bool:
    return get_arena_runtime(model) is not None


def close_arena_offload(model) -> None:
    close_memory_runtime(model)
