"""Preparation-scoped ownership for one arena runtime."""

from __future__ import annotations

from .errors import ArenaCleanupError
from .ownership import acquire_process_owner, release_process_owner


class ArenaRuntimeResources:
    """Own resources from pre-commit preparation through runtime close."""

    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device
        self.owner_token = None
        self.canonical_build = None
        self.arena = None
        self.residency = None
        self.executor = None
        self.runtime = None
        self.fp8_restores = []
        self.published_attributes = []
        self.canonical_modules = ()
        self.canonical_committed = False
        self.closing = False
        self.disposed = False
        self.released = False
        self._releasing = False
        self._original_movement = {
            name: getattr(model, name, None) for name in ("to", "cuda", "cpu")
        }
        self._original_forward = getattr(model, "forward", None)

    def acquire_process_owner(self) -> None:
        if self.owner_token is None:
            self.owner_token = acquire_process_owner(self.device)

    def adopt_canonical_build(self, build) -> None:
        self.canonical_build = build
        self.arena = build.arena
        build._arena_resources = self

    def mark_canonical_committed(self) -> None:
        self.canonical_committed = True
        if self.canonical_build is not None:
            # Rollback is invalid after the destructive boundary. Drop its
            # original Parameter references immediately so the resource owner
            # does not retain a second model-sized payload for runtime life.
            self.canonical_build._arena_resources = None
            self.canonical_build._originals.clear()
            self.canonical_build = None

    def adopt_residency(self, residency) -> None:
        self.residency = residency

    def adopt_executor(self, executor) -> None:
        self.executor = executor

    def adopt_runtime(self, runtime) -> None:
        self.runtime = runtime

    def record_fp8_restore(self, label, restore) -> None:
        self.fp8_restores.append((label, restore))

    def record_published_attribute(self, owner, name, value) -> None:
        setattr(owner, name, value)
        self.published_attributes.append((owner, name, value))

    def _install_disposed_movement_guard(self) -> None:
        def disposed(*_args, **_kwargs):
            raise RuntimeError("arena_offload_transformer_disposed")

        for name in self._original_movement:
            if self._original_movement[name] is not None:
                setattr(self.model, name, disposed)
        if self._original_forward is not None:
            setattr(self.model, "forward", disposed)
        setattr(self.model, "_arena_offload_disposed", True)

    def release(self) -> None:
        if self.released or self._releasing:
            return
        self._releasing = True
        self.closing = True
        failures = []

        if self.executor is not None and getattr(
            self.executor, "active_executions", 0
        ):
            self._releasing = False
            self.closing = False
            raise ArenaCleanupError(
                (("active execution", RuntimeError("cannot_close_during_execution")),)
            )

        def attempt(label, operation):
            try:
                operation()
            except BaseException as error:
                failures.append((label, error))

        from . import transfer

        try:
            transfer_cleanup_failed = False
            if self.owner_token is not None:
                try:
                    transfer.drain_fetch_runtime(owner_token=self.owner_token)
                except BaseException as error:
                    transfer_cleanup_failed = True
                    failures.append(("transfer tickets", error))
            if self.executor is not None:
                attempt("immutable executor", self.executor.close)
            for label, restore in reversed(self.fp8_restores):
                attempt(label, restore)
            self.fp8_restores.clear()
            if self.residency is not None:
                attempt("resident sidecars", self.residency.clear)

            for owner, name, value in reversed(self.published_attributes):
                if getattr(owner, name, None) is value:
                    attempt(
                        f"published attribute {name}",
                        lambda owner=owner, name=name: delattr(owner, name),
                    )
            self.published_attributes.clear()

            if self.canonical_committed:
                if self.arena is not None:
                    attempt(
                        "movement interception",
                        lambda: self.arena.unguard_whole_model_to(self.model),
                    )
                    attempt("canonical arena", self.arena.release)
                self.disposed = True
                self._install_disposed_movement_guard()
            elif self.canonical_build is not None:
                attempt("canonical build", self.canonical_build.rollback)

            if self.owner_token is not None:
                try:
                    transfer.release_fetch_runtime(self.owner_token)
                except BaseException as error:
                    transfer_cleanup_failed = True
                    failures.append(("transfer runtime", error))
                if not transfer_cleanup_failed:
                    try:
                        release_process_owner(self.owner_token)
                    except BaseException as error:
                        failures.append(("process ownership", error))
                    else:
                        self.owner_token = None
        finally:
            self.closing = False
            self.released = self.owner_token is None
            self._releasing = False
            if self.runtime is not None:
                self.runtime._closed = True
                self.runtime._disposed = bool(self.canonical_committed)
            self.canonical_modules = ()
            self.canonical_build = None
            self.residency = None
            self.executor = None
        if failures:
            raise ArenaCleanupError(failures)
