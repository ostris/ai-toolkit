"""Generic production checkpoint-to-arena load session."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch


PENDING_CANONICAL_BUILD_ATTR = "_arena_pending_canonical_build"
_CURRENT_SESSION = ContextVar("arena_direct_load_session", default=None)


@dataclass
class _DirectLoadSession:
    device: object
    block_names: tuple[str, ...]
    pending: dict[int, tuple[object, object]] = field(default_factory=dict)
    unsupported_reason: str | None = None

    def publish(self, model, build) -> None:
        if getattr(model, PENDING_CANONICAL_BUILD_ATTR, None) is not None:
            raise RuntimeError("arena_direct_load_duplicate_pending_build")
        setattr(model, PENDING_CANONICAL_BUILD_ATTR, build)
        self.pending[id(model)] = (model, build)

    def claim(self, model):
        build = getattr(model, PENDING_CANONICAL_BUILD_ATTR, None)
        if build is None:
            return None
        delattr(model, PENDING_CANONICAL_BUILD_ATTR)
        self.pending.pop(id(model), None)
        return build

    def rollback_pending(self) -> None:
        for model, build in tuple(self.pending.values()):
            if getattr(model, PENDING_CANONICAL_BUILD_ATTR, None) is build:
                delattr(model, PENDING_CANONICAL_BUILD_ATTR)
            build.rollback()
        self.pending.clear()


def _arena_load_enabled(base_model, enabled) -> bool:
    if enabled is not None:
        return bool(enabled)
    config = getattr(base_model, "model_config", None)
    return bool(
        config is not None
        and getattr(config, "layer_offloading", False)
        and getattr(config, "layer_offloading_smart", False)
        and not getattr(base_model, "te_only", False)
    )


def _managed_source_keys(build) -> set[str]:
    return {
        source_key
        for block_schema in build.state_schema.values()
        for source_key in block_schema
    }


@contextmanager
def model_load_arena_session(base_model, *, enabled=None):
    """Offer generic direct arena ingestion during ``base_model.load_model``.

    Unsupported or non-inferable state schemas leave the state mapping intact
    and use the ordinary assignment path. A consumed mapping remains owned by
    the model until the shared trainer claims it through
    ``prepare_arena_offload`` or releases the unfinished preparation.
    """
    if not _arena_load_enabled(base_model, enabled):
        yield None
        return
    if _CURRENT_SESSION.get() is not None:
        raise RuntimeError("nested_arena_direct_load_session")
    block_names = ()
    provider = getattr(base_model, "get_transformer_block_names", None)
    if callable(provider):
        block_names = tuple(provider() or ())
    session = _DirectLoadSession(
        device=getattr(base_model, "device_torch", None),
        block_names=block_names,
    )
    token = _CURRENT_SESSION.set(session)
    original_load_state_dict = torch.nn.Module.load_state_dict

    def load_state_dict(module, state_dict, strict=True, assign=False):
        build = try_prepare_canonical_from_state_dict(module, state_dict)
        if build is None:
            return original_load_state_dict(
                module, state_dict, strict=strict, assign=assign
            )
        expected_missing = _managed_source_keys(build)
        try:
            incompatible = original_load_state_dict(
                module, state_dict, strict=False, assign=assign
            )
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            if missing != expected_missing or unexpected:
                raise RuntimeError(
                    "arena_direct_load_residual_mismatch:"
                    f"missing={sorted(missing - expected_missing)[:5]}:"
                    f"unconsumed={sorted(expected_missing - missing)[:5]}:"
                    f"unexpected={sorted(unexpected)[:5]}"
                )
            return type(incompatible)([], [])
        except BaseException:
            discard_pending_canonical_build(module)
            raise

    torch.nn.Module.load_state_dict = load_state_dict
    try:
        yield session
        if session.pending:
            setattr(
                base_model,
                "_arena_pending_load_models",
                tuple(model for model, _build in session.pending.values()),
            )
    except BaseException:
        session.rollback_pending()
        raise
    finally:
        torch.nn.Module.load_state_dict = original_load_state_dict
        _CURRENT_SESSION.reset(token)


def try_prepare_canonical_from_state_dict(model, state_dict):
    """Consume inferable managed state into a pending canonical build."""
    session = _CURRENT_SESSION.get()
    if session is None:
        return None
    from .api import prepare_canonical_storage_from_state_dict
    from .construction import (
        CanonicalBuildError,
        CanonicalStateConsumedError,
        CanonicalStateInferenceError,
    )
    from .discovery import BlockDiscoveryError
    from ..canonical_arena import CanonicalArenaError

    try:
        build = prepare_canonical_storage_from_state_dict(
            model,
            state_dict,
            block_names=session.block_names,
            device=session.device,
        )
    except CanonicalStateConsumedError:
        raise
    except (
        BlockDiscoveryError,
        CanonicalArenaError,
        CanonicalBuildError,
        CanonicalStateInferenceError,
    ) as error:
        session.unsupported_reason = str(error)
        return None
    try:
        session.publish(model, build)
    except BaseException:
        build.rollback()
        raise
    return build


def claim_pending_canonical_build(model):
    """Take a generic loader build at the normal arena-attach boundary."""
    session = _CURRENT_SESSION.get()
    if session is not None:
        return session.claim(model)
    build = getattr(model, PENDING_CANONICAL_BUILD_ATTR, None)
    if build is not None:
        delattr(model, PENDING_CANONICAL_BUILD_ATTR)
    return build


def discard_pending_canonical_build(model) -> None:
    """Rollback a pending build after residual assignment fails."""
    build = claim_pending_canonical_build(model)
    if build is not None:
        build.rollback()
