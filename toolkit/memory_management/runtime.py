"""Generic memory-runtime discovery used by shared trainer code."""

from __future__ import annotations


RUNTIME_ATTR = "_arena_offload_runtime"


def unwrap_memory_model(model):
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        original = getattr(model, "_orig_mod", None)
        if original is not None and original is not model:
            model = original
            continue
        if getattr(model, RUNTIME_ATTR, None) is not None:
            return model
        if hasattr(model, "_memory_manager"):
            return model
        inner = getattr(model, "module", None)
        if inner is None or inner is model:
            return model
        model = inner
    return model


def get_memory_runtime(model):
    if model is None:
        return None
    return getattr(unwrap_memory_model(model), RUNTIME_ATTR, None)


def is_memory_managed(model) -> bool:
    if model is None:
        return False
    inner = unwrap_memory_model(model)
    return (
        get_memory_runtime(inner) is not None
        or hasattr(inner, "_memory_manager")
        or bool(getattr(inner, "_arena_offload_disposed", False))
    )


def memory_runtime_owns_compile(model) -> bool:
    runtime = get_memory_runtime(model)
    return bool(runtime is not None and getattr(runtime, "owns_compile", True))


def close_memory_runtime(model) -> None:
    runtime = get_memory_runtime(model)
    if runtime is not None:
        runtime.close()


def close_memory_runtime_preparation(model_owner) -> None:
    """Release a loader-scoped preparation that never published a runtime."""
    pending_models = getattr(model_owner, "_arena_pending_load_models", ())
    if hasattr(model_owner, "_arena_pending_load_models"):
        delattr(model_owner, "_arena_pending_load_models")
    if pending_models:
        from .arena_offload.load_session import discard_pending_canonical_build

        for model in pending_models:
            discard_pending_canonical_build(model)
    operation = getattr(model_owner, "cleanup_memory_runtime_preparation", None)
    if operation is not None:
        operation()
