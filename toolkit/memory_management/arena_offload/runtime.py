"""The arena offload runtime facade.

One object owns the arena, the residency state, the training plan, and the
immutable block executor. Model integrations and the shared trainer hold a
reference to it and nothing else -- no `_mm_*` field reads, no arena
construction, no residency manipulation.

Cold planning, live policy, FP8 transforms, and teardown are arena-owned. The
legacy per-linear manager is deliberately outside this package.
"""

from __future__ import annotations

import contextlib
import time
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .. import allocator_cap
from ..canonical_arena import CanonicalArena
from ..residency import ResidencyPlan, ResidencyState
from ..vram_budget import apply_simulated_card
from .policy import (
    AGGRESSIVE_PROMOTION_MIN_CAPACITY,
    ArenaResidencyController,
    TrainingSignalWindow,
)
from .errors import ArenaCleanupError, ArenaSetupFatalError
from .fp8 import disable as disable_fp8
from .fp8 import enable as enable_fp8
from .fp8 import set_fp8_grad_input_enabled
from .planner import (
    build_training_plan,
    impossible_training_plan_message,
    resolve_physical_vram_headroom_gib,
)
from .ownership import normalize_device
from .resources import ArenaRuntimeResources

RUNTIME_ATTR = "_arena_offload_runtime"

GIB = 1024**3
BOOTSTRAP_MARGIN_BYTES = GIB
BOOTSTRAP_MIN_STEP = 2


class ArenaOffloadRuntime:
    """Lifecycle + execution contexts for one arena-offloaded transformer."""

    def __init__(
        self,
        model,
        *,
        device,
        config,
        arena,
        residency,
        executor,
        training_plan,
        smart_plan,
        canonical_modules,
        resources,
    ) -> None:
        self._model = model
        self._device = device
        self._config = config
        self._arena = arena
        self._residency = residency
        self._executor = executor
        self._training_plan = training_plan
        self._smart_plan = smart_plan
        self._canonical_modules = canonical_modules
        self._resources = resources
        self._closed = False
        self._disposed = False

        # Set by training_step(); the residency controller (git-bug 0c577ef)
        # reads these at the step boundary.
        self._last_shape_key: tuple | None = None
        self._last_step_num: int | None = None
        self._successful_training_steps = 0
        self._signals = TrainingSignalWindow()
        self._last_policy_error: str | None = None
        self._last_failure_event: dict | None = None
        self._policy = ArenaResidencyController()
        self._last_training_cap_target_bytes: int | None = None
        self._bootstrap_complete = False
        self._bootstrap_min_free_bytes: int | None = None
        self._bootstrap_budget_bytes = 0
        self._bootstrap_block_keys: tuple[str, ...] = ()
        self._training_fp8_restores = []
        self._training_fp8_canonical = 0
        self._training_fp8_singletons = 0
        self._sampling_fp8_canonical = 0
        self._sampling_fp8_singletons = 0
        self._permanent_placement = None
        self._device_state_parked_plan = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def _prepare(
        cls,
        transformer,
        *,
        device,
        selection,
        config,
        ignore_modules: Sequence[Any] | None = None,
        canonical_build=None,
    ) -> ArenaOffloadRuntime:
        existing = getattr(transformer, RUNTIME_ATTR, None)
        if getattr(transformer, "_arena_offload_disposed", False):
            raise RuntimeError("arena_offload_transformer_disposed")
        if existing is not None:
            if getattr(existing, "disposed", False):
                raise RuntimeError("arena_offload_transformer_disposed")
            raise RuntimeError("arena_offload_already_prepared")

        resources = getattr(canonical_build, "_arena_resources", None)
        if resources is None:
            resources = ArenaRuntimeResources(transformer, device)
            resources.acquire_process_owner()

        try:
            # Bind card simulation and allocator policy before any plan reads.
            apply_simulated_card(config._simulated_vram_gib, device=device)
            allocator_cap.configure_wddm_allocator_guard(
                device,
                config._policy.wddm_hard_gib,
                strict=config.strict_vram_cap,
                log_prefix="[ArenaOffload]",
            )
            set_fp8_grad_input_enabled(config.fp8_backward)

            blocks = selection.blocks
            block_keys = selection.block_keys
            entries_by_block = {
                key: list(selection.entries_by_block[key])
                for key in block_keys
            }
            if canonical_build is None:
                arena = CanonicalArena()
                canonical_build = arena.prepare(entries_by_block, model=transformer)
                resources.adopt_canonical_build(canonical_build)
                canonical_build.populate_from_model()
            else:
                resources.adopt_canonical_build(canonical_build)
                if canonical_build.model is not transformer:
                    raise RuntimeError("arena_canonical_build_model_mismatch")
                prepared_entries = {
                    key: tuple(module for _name, module in entries)
                    for key, entries in canonical_build.entries_by_block.items()
                }
                expected_entries = {
                    key: tuple(module for _name, module in entries)
                    for key, entries in entries_by_block.items()
                }
                if prepared_entries != expected_entries:
                    raise RuntimeError("arena_canonical_build_selection_mismatch")
                arena = canonical_build.arena

            canonical_modules = tuple(
                child for entries in entries_by_block.values() for _name, child in entries
            )
            smart_plan = build_training_plan(
                transformer,
                canonical_build,
                canonical_modules,
                device,
                config,
                block_keys=block_keys,
            )
            if not smart_plan["fits"]:
                message = impossible_training_plan_message(smart_plan)
                if config.strict_vram_cap:
                    raise ValueError(message)
                warnings.warn(
                    message
                    + "; continuing in production spill-permitted mode",
                    RuntimeWarning,
                    stacklevel=2,
                )

            canonical_build.commit()
            resources.mark_canonical_committed()
            resources.canonical_modules = canonical_modules
            residency = ResidencyState(arena, device)
            resources.adopt_residency(residency)
            training_plan = ResidencyPlan.from_smart_plan(
                arena, smart_plan, phase="train"
            )
            residency.reconcile(training_plan)

            policy = config._policy
            executor_kwargs = dict(
                depth=policy.prefetch_depth,
                compile_blocks=config.compile_blocks,
                compile_dynamic=config._compile_dynamic,
                compile_dynamic_hints=config._compile_dynamic_hints,
                protected_training_leaf_keys=smart_plan.get(
                    "protected_training_leaf_keys", ()
                ),
                owner_token=resources.owner_token,
            )
            from .dispatcher import prepare_block_dispatcher_runtime

            executor = prepare_block_dispatcher_runtime(
                transformer,
                residency,
                selection=selection,
                **executor_kwargs,
            )
            resources.adopt_executor(executor)

            runtime = cls(
                transformer,
                device=device,
                config=config,
                arena=arena,
                residency=residency,
                executor=executor,
                training_plan=training_plan,
                smart_plan=smart_plan,
                canonical_modules=canonical_modules,
                resources=resources,
            )
            resources.adopt_runtime(runtime)
            resources.record_published_attribute(
                transformer, RUNTIME_ATTR, runtime
            )
            return runtime
        except BaseException as error:
            committed = resources.canonical_committed
            try:
                resources.release()
            except ArenaCleanupError as cleanup_error:
                try:
                    error.add_note(str(cleanup_error))
                except AttributeError:
                    pass
            if committed:
                raise ArenaSetupFatalError(
                    "arena setup failed after canonical commit"
                ) from error
            raise

    # ------------------------------------------------------------------
    # read-only state
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def owns_compile(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def disposed(self) -> bool:
        return self._disposed

    def place_permanent_modules(self, device, dtype=None) -> None:
        """Move only noncanonical subtrees, preserving arena Parameter views."""
        self._require_open()
        import torch

        target = (torch.device(device), dtype)
        if getattr(self, "_permanent_placement", None) == target:
            return
        canonical = set(self._canonical_modules)

        def contains_canonical(module):
            return any(child in canonical for child in module.modules())

        def has_wrapped_parameter(module, *, recurse=True):
            for parameter in module.parameters(recurse=recurse):
                try:
                    names, _context = parameter.__tensor_flatten__()
                except Exception:
                    continue
                if names:
                    return True
            return False

        def move_local_state(module):
            local_dtype = (
                None
                if dtype is None or has_wrapped_parameter(module, recurse=False)
                else dtype
            )

            def convert(tensor):
                if local_dtype is not None and (
                    tensor.is_floating_point() or tensor.is_complex()
                ):
                    return tensor.to(device=device, dtype=local_dtype)
                return tensor.to(device=device)

            module._apply(convert, recurse=False)

        def move(module):
            if module in canonical:
                return
            if not contains_canonical(module):
                if dtype is None or has_wrapped_parameter(module):
                    module.to(device=device)
                else:
                    module.to(device=device, dtype=dtype)
                return
            # A mixed parent can own direct permanent state (for example
            # architecture pad tokens) in addition to canonical descendants.
            # Move that local state before recursing into child subtrees.
            move_local_state(module)
            for child in module.children():
                move(child)

        try:
            move(self._model)
            self._permanent_placement = target
        except BaseException as error:
            self._fatal_setup_failure(error)

    def handle_whole_model_move(
        self, device, *, dtype=None, memory_format=None
    ):
        """Interpret safe whole-model device intent without moving arena leaves."""
        self._require_open()
        if getattr(self._executor, "active_executions", 0):
            raise RuntimeError("arena_whole_model_move_during_execution")
        if memory_format is not None:
            raise RuntimeError("arena_whole_model_memory_format_unsupported")

        placement = self._permanent_placement
        if placement is None:
            raise RuntimeError("arena_whole_model_move_before_placement")
        placed_device, placed_dtype = placement
        placed_device = normalize_device(placed_device)
        if dtype is not None and dtype != placed_dtype:
            raise RuntimeError("arena_whole_model_dtype_change_unsupported")

        target_device = placed_device if device is None else normalize_device(device)
        arena_device = normalize_device(self._device)
        if target_device.type not in ("cpu", "cuda"):
            raise RuntimeError(
                f"arena_whole_model_device_unsupported:{target_device.type}"
            )
        if target_device.type == "cuda" and target_device != arena_device:
            raise RuntimeError(
                f"arena_whole_model_cuda_device_unsupported:{target_device}"
            )

        if target_device == placed_device:
            if (
                target_device == arena_device
                and self._device_state_parked_plan is not None
            ):
                self.restore_residency_after_external_phase()
            return self._model

        if target_device.type == "cpu":
            self.park_residency_for_external_phase()
            self.place_permanent_modules(target_device, placed_dtype)
            return self._model

        self.place_permanent_modules(arena_device, placed_dtype)
        self.restore_residency_after_external_phase()
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def config(self):
        return self._config

    @property
    def block_count(self) -> int:
        return len(self._arena.block_keys())

    @property
    def finalized(self) -> bool:
        return bool(getattr(self._executor, "finalized", False))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def park_residency_for_external_phase(self) -> None:
        """Drop device sidecars while another large model component is active."""
        self._require_open()
        if self._device_state_parked_plan is not None:
            return
        current = self._residency.plan
        if current.phase != self._executor.TRAIN:
            raise RuntimeError(
                f"arena_external_phase_requires_train:{current.phase}"
            )
        self._device_state_parked_plan = current
        parked = ResidencyPlan.build(self._executor.TRAIN, ())
        try:
            if self.finalized:
                self._executor.set_residency_plan(parked)
            else:
                self._residency.reconcile(parked)
        except BaseException:
            self._device_state_parked_plan = None
            raise

    def restore_residency_after_external_phase(self) -> None:
        """Restore the exact training residency saved by the matching park."""
        self._require_open()
        plan = self._device_state_parked_plan
        if plan is None:
            return
        if self.finalized:
            self._executor.set_residency_plan(plan)
        else:
            self._residency.reconcile(plan)
        self._training_plan = plan
        self._device_state_parked_plan = None

    def set_compile_dynamic_hints(self, hints) -> None:
        """Install mark_dynamic hints on the block kernels (see ImmutableRuntime).

        The trainer derives sequence bounds from the datasets, which do not exist
        when the runtime is prepared. Must be called before the first forward.
        """
        self._require_open()
        try:
            self._executor.set_compile_dynamic_hints(hints)
            self._config = replace(
                self._config,
                _compile_dynamic_hints=self._executor.compile_dynamic_hints,
            )
        except BaseException as error:
            self._fatal_setup_failure(error)

    def finalize(self, network=None):
        """Build the permanent train/sample programs, then activate TRAIN.

        Must run AFTER the training network is applied so the dispatcher saves
        the final installed block forwards. ``network`` is retained as a public
        lifecycle argument, but adapter execution remains owned by those saved
        model forwards rather than by arena-specific mappings.
        """
        self._require_open()
        try:
            self._bind_training_cap()
            del network
            if self._config.fp8_forward:
                canonical_ids = self._canonical_runtime_ids()
                singleton_ids = self._singleton_runtime_ids()
                self._training_fp8_restores = enable_fp8(
                    self._model,
                    include_ids=canonical_ids | singleton_ids,
                    live_ids=canonical_ids | singleton_ids,
                    training=True,
                    device=self._device,
                )
                installed_ids = {
                    restore[4] for restore in self._training_fp8_restores
                }
                self._training_fp8_canonical = len(installed_ids & canonical_ids)
                self._training_fp8_singletons = len(installed_ids & singleton_ids)
                self._resources.record_fp8_restore(
                    "FP8 training restore",
                    lambda restores=self._training_fp8_restores: disable_fp8(restores),
                )
            # Save block forwards only after the live-state FP8 transforms are
            # installed, so compiled dispatcher kernels trace the selected
            # execution policy rather than Quanto's materializing fallback.
            self._executor.finalize_execution()
            self._executor.activate(self._executor.TRAIN, self._training_plan)
            return self
        except BaseException as error:
            self._fatal_setup_failure(error)

    def close(self) -> None:
        """Release through the same owner used during preparation."""
        self._resources.release()

    def _fatal_setup_failure(self, error):
        try:
            self._resources.release()
        except ArenaCleanupError as cleanup_error:
            try:
                error.add_note(str(cleanup_error))
            except AttributeError:
                pass
        raise ArenaSetupFatalError(
            "arena setup failed after canonical commit"
        ) from error

    def _require_open(self) -> None:
        if self._closed:
            if self._disposed:
                raise RuntimeError("arena_offload_transformer_disposed")
            raise RuntimeError("arena_offload_runtime_closed")

    # ------------------------------------------------------------------
    # execution contexts
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def training_step(self, *, shape_key: tuple | None = None, step_num: int | None = None):
        """The training phase boundary. Spans forward AND backward.

        Backward must be inside: checkpoint recomputation re-enters the block
        runtime, so the source snapshot has to stay pinned for the whole step.

        This is the two-timescale residency controller's phase-boundary hook:
        enter = plan/act, exit = observe.
        """
        self._require_open()
        self._last_shape_key = shape_key
        self._last_step_num = step_num
        try:
            self._apply_training_policy()
        except BaseException as error:
            try:
                self._handle_training_failure(
                    error, shape_key=shape_key, step_num=step_num
                )
            except Exception as cleanup_error:
                self._last_policy_error = (
                    "training_policy_cleanup: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self._device)
        except Exception:
            pass
        started_at = time.perf_counter()
        succeeded = False
        try:
            with self._executor.execution(self._executor.TRAIN):
                yield self
            succeeded = True
        except BaseException as error:
            try:
                self._handle_training_failure(
                    error, shape_key=shape_key, step_num=step_num
                )
            except Exception as cleanup_error:
                self._last_policy_error = (
                    "training_failure_cleanup: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        finally:
            if succeeded:
                try:
                    self._observe_training_step(
                        shape_key=shape_key,
                        step_num=step_num,
                        step_wall_ms=(time.perf_counter() - started_at) * 1000.0,
                    )
                    self._last_policy_error = None
                except Exception as error:
                    # Diagnostics must never mask a successful training step.
                    self._last_policy_error = f"{type(error).__name__}: {error}"

    def _handle_training_failure(self, error, *, shape_key, step_num):
        """Clean arena-owned state after the executor has unwound."""
        import torch

        from . import transfer
        from ..vram_budget import device_free_bytes

        abandoned = transfer.drain_fetch_runtime()
        text = str(error)
        allocation_failure = (
            isinstance(error, torch.cuda.OutOfMemoryError)
            or "out of memory" in text.lower()
        )
        rollback = None
        if allocation_failure:
            decision = self._policy.allocation_failure()
            if decision.action == "rollback" and decision.block_key is not None:
                rollback_keys = tuple(decision.block_keys or ())
                if rollback_keys:
                    self.transition_training_blocks(
                        rollback_keys, resident=False
                    )
                else:
                    self.transition_training_block(
                        decision.block_key, resident=False
                    )
                if decision.target_cap_bytes is not None:
                    allocator_cap.configure_wddm_allocator_guard(
                        self._device,
                        self._config._policy.wddm_hard_gib,
                        target_cap_bytes=decision.target_cap_bytes,
                        strict=getattr(
                            self._config, "strict_vram_cap", False
                        ),
                        log_prefix="[ArenaOffload]",
                    )
                    self._last_training_cap_target_bytes = (
                        decision.target_cap_bytes
                    )
                rollback = (
                    list(decision.block_keys)
                    if decision.block_keys
                    else decision.block_key
                )
        try:
            stats = torch.cuda.memory_stats(self._device)
            peak_allocated = int(torch.cuda.max_memory_allocated(self._device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self._device))
            physical_free = int(device_free_bytes(self._device))
        except Exception:
            stats = {}
            peak_allocated = 0
            peak_reserved = 0
            physical_free = 0
        active_cap = allocator_cap.applied_cap_bytes(self._device)
        policy = getattr(getattr(self, "_config", None), "_policy", None)
        hard_gib = getattr(policy, "wddm_hard_gib", None)
        hard_bytes = int(max(1.0, float(hard_gib or 1.0)) * GIB)
        classification = "non_allocation_failure"
        if allocation_failure:
            classification = (
                "capped_allocator_rejection"
                if active_cap is not None and physical_free > hard_bytes / 2
                else "cuda_allocation_failure_unknown"
            )
        self._last_failure_event = {
            "event": "training_allocation_failure",
            "classification": classification,
            "exception_type": type(error).__name__,
            "exception": text,
            "shape_key": shape_key,
            "step_num": step_num,
            "active_cap_bytes": active_cap,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "physical_free_bytes": physical_free,
            "allocator_retry_delta": max(
                0,
                int(stats.get("num_alloc_retries", 0) or 0)
                - int(
                    (
                        getattr(self._signals, "_allocator_previous", None)
                        or {}
                    ).get("num_alloc_retries", 0)
                    or 0
                ),
            ),
            "rollback_block": rollback,
            "rejected_residency_bytes": (
                self._policy.last_rejected_residency_bytes
            ),
            "abandoned_fetch_tickets": int(abandoned),
        }

    def _singleton_runtime_ids(self):
        return set((self._smart_plan or {}).get("singleton_runtime_ids", ()))

    def _canonical_runtime_ids(self):
        return {id(module) for module in self._canonical_modules}

    @contextlib.contextmanager
    def sampling_session(self):
        """Wrap a sampling run and restore TRAIN once at the end."""
        self._require_open()
        sampling_restores = []
        if self._config.fp8_sampling:
            canonical_ids = self._canonical_runtime_ids()
            singleton_ids = self._singleton_runtime_ids()
            sampling_restores = enable_fp8(
                self._model,
                include_ids=canonical_ids | singleton_ids,
                live_ids=canonical_ids | singleton_ids,
                training=False,
                device=self._device,
            )
            # Counts are eligibility diagnostics; training keeps its own
            # persistent transforms underneath this temporary sampling layer.
            installed_ids = {restore[4] for restore in sampling_restores}
            self._sampling_fp8_canonical = len(installed_ids & canonical_ids)
            self._sampling_fp8_singletons = len(installed_ids & singleton_ids)
        try:
            yield self
        finally:
            if sampling_restores:
                disable_fp8(sampling_restores)
            self._bind_training_cap()
            self._executor.activate(self._executor.TRAIN, self._training_plan)

    @contextlib.contextmanager
    def sampling_image(self, *, shape_key: tuple, cold_working_bytes: int):
        """The sampling phase boundary for ONE image.

        Switches to the permanent SAMPLE program (forward-only, no
        checkpointing) over the same arena. TRAIN is restored by the enclosing
        `sampling_session()`.
        """
        self._require_open()
        policy = self._config._policy

        fixed_working_bytes = _fixed_working_bytes(policy.sampling_working_reserve_gib)
        hard_gib = (
            1.0
            if policy.sampling_wddm_hard_gib is None
            else float(policy.sampling_wddm_hard_gib)
        )
        allocator_cap.configure_wddm_allocator_guard(
            self._device,
            hard_gib,
            strict=getattr(self._config, "strict_vram_cap", False),
            log_prefix="[ArenaOffload]",
        )
        physical_headroom_gib = resolve_physical_vram_headroom_gib(
            self._device,
            policy.sampling_physical_vram_headroom_gib,
            hard_gib=hard_gib,
        )
        dequant_reserve = (
            0
            if self._config.fp8_sampling
            else int(
                (self._smart_plan or {}).get(
                    "largest_singleton_bf16_dequant_bytes", 0
                )
            )
        )
        with self._executor.sampling(
            shape_key=shape_key,
            cold_working_bytes=int(cold_working_bytes),
            fixed_working_bytes=fixed_working_bytes,
            cold_floor_bytes=(
                int(physical_headroom_gib * GIB) + dequant_reserve
            ),
            hot_floor_bytes=int((hard_gib + 0.25) * GIB) + dequant_reserve,
        ):
            yield self

    # ------------------------------------------------------------------
    def record_training_physical_free_min(self, free_bytes) -> None:
        """Publish one successful step's physical high-water for bootstrap."""
        if (
            self._bootstrap_complete
            or free_bytes is None
            or self._successful_training_steps < BOOTSTRAP_MIN_STEP
        ):
            return
        value = max(0, int(free_bytes))
        self._bootstrap_min_free_bytes = (
            value
            if self._bootstrap_min_free_bytes is None
            else min(self._bootstrap_min_free_bytes, value)
        )

    def _training_physical_vram_headroom_bytes(self) -> int:
        policy = self._config._policy
        hard_gib = float(getattr(policy, "wddm_hard_gib", None) or 1.0)
        headroom_gib = resolve_physical_vram_headroom_gib(
            self._device,
            getattr(policy, "physical_vram_headroom_gib", -1.0),
            hard_gib=hard_gib,
        )
        return int(headroom_gib * GIB)

    def _bootstrap_training_residency(self, active_cap_bytes) -> bool:
        if (
            self._bootstrap_complete
            or self._bootstrap_min_free_bytes is None
            or int(self._successful_training_steps) < BOOTSTRAP_MIN_STEP
        ):
            return False
        budget = max(
            0,
            self._bootstrap_min_free_bytes
            - self._training_physical_vram_headroom_bytes()
            - BOOTSTRAP_MARGIN_BYTES,
        )
        self._bootstrap_budget_bytes = budget
        candidates = []
        protected = self._protected_training_blocks()
        plan = getattr(self._residency, "plan", None) or self._training_plan
        for order, block_key in enumerate(self._arena.block_keys()):
            record = self._arena.block_record(block_key)
            keys = tuple((block_key, name) for name in record.leaf_names)
            if block_key in protected or any(
                key in plan.resident_leaf_keys for key in keys
            ):
                continue
            candidates.append(
                (int(record.committed_bytes), order, str(block_key))
            )
        selected = []
        used = 0
        for block_bytes, _order, block_key in sorted(candidates):
            if used + block_bytes > budget:
                continue
            selected.append(block_key)
            used += block_bytes
        if not selected:
            self._bootstrap_complete = True
            return False
        resident_before = (
            int(self._residency.resident_bytes())
            + int((self._smart_plan or {}).get("singleton_resident_bytes", 0))
        )
        result = self.transition_training_blocks(selected, resident=True)
        if not result.get("changed"):
            return False
        self._bootstrap_complete = True
        self._bootstrap_block_keys = tuple(result["block_keys"])
        self._policy.begin_bootstrap_promotion(
            self._bootstrap_block_keys,
            used,
            resident_before,
            active_cap_bytes,
        )
        return True

    def _protected_training_blocks(self):
        return frozenset(
            str(block)
            for block, _leaf in (self._smart_plan or {}).get(
                "protected_training_leaf_keys", ()
            )
        )

    def _promotion_candidates(self):
        plan = getattr(self._residency, "plan", None) or self._training_plan
        protected = self._protected_training_blocks()
        candidates = []
        for order, block_key in enumerate(self._arena.block_keys()):
            record = self._arena.block_record(block_key)
            keys = tuple((block_key, name) for name in record.leaf_names)
            if block_key in protected or any(
                key in plan.resident_leaf_keys for key in keys
            ):
                continue
            candidates.append(
                (int(record.committed_bytes), order, str(block_key))
            )
        return tuple(
            {"block_key": block_key, "block_bytes": block_bytes}
            for block_bytes, _order, block_key in sorted(candidates)
        )

    def _promotion_candidate(self):
        candidates = self._promotion_candidates()
        return candidates[0] if candidates else None

    def _aggressive_promotion_capacity(self, current_cap_bytes):
        """Blocks that fit under both worst-shape safety budgets."""
        candidates = self._promotion_candidates()
        allocator_slack = self._worst_shape_allocator_slack_bytes(
            current_cap_bytes
        )
        allocator_headroom = int(
            self._policy.allocator_cache_headroom_bytes
        )
        used = 0
        capacity = 0
        for candidate in candidates:
            used += int(candidate["block_bytes"])
            cumulative = {"block_bytes": used}
            if (
                self._worst_shape_candidate_physical_headroom_bytes(
                    cumulative
                )
                < 0
            ):
                break
            if allocator_slack <= used + allocator_headroom:
                break
            capacity += 1
        return capacity

    def _demotion_candidate(self):
        plan = getattr(self._residency, "plan", None) or self._training_plan
        protected = self._protected_training_blocks()
        candidates = []
        for order, block_key in enumerate(self._arena.block_keys()):
            record = self._arena.block_record(block_key)
            keys = tuple((block_key, name) for name in record.leaf_names)
            if block_key in protected or not all(
                key in plan.resident_leaf_keys for key in keys
            ):
                continue
            actual = sum(
                self._residency.resident_leaf_bytes(key) for key in keys
            )
            candidates.append(
                (actual or int(record.committed_bytes), -order, str(block_key))
            )
        if not candidates:
            return None
        block_bytes, _order, block_key = max(candidates)
        return {"block_key": block_key, "block_bytes": block_bytes}

    def _worst_shape_candidate_physical_headroom_bytes(self, candidate):
        if candidate is None:
            return 0
        signal = self._signals.last_signal
        peaks = self._signals.shape_peaks
        if signal is None or not peaks:
            return 0
        from .. import vram_budget

        total = int(vram_budget.device_total_bytes(self._device))
        worst_working = max(
            int(peak.working_peak_bytes)
            for peak in peaks.values()
            if peak.steps > 0
        ) if any(peak.steps > 0 for peak in peaks.values()) else 0
        current_resident = (
            int(self._residency.resident_bytes())
            + int((self._smart_plan or {}).get("singleton_resident_bytes", 0))
        )
        current_ring = self._training_ring_bytes()
        non_torch = max(
            0,
            total
            - int(signal.get("device_free_bytes", 0) or 0)
            - int(signal.get("peak_reserved_bytes", 0) or 0),
        )
        predicted_free = total - (
            worst_working
            + current_resident
            + current_ring
            + non_torch
            + int(candidate["block_bytes"])
        )
        return int(
            predicted_free
            - self._training_physical_vram_headroom_bytes()
        )

    def _worst_shape_allocator_slack_bytes(self, current_cap_bytes):
        peaks = self._signals.shape_peaks
        if not any(peak.steps > 0 for peak in peaks.values()):
            return 0
        from .. import vram_budget

        worst_working = max(
            int(peak.working_peak_bytes)
            for peak in peaks.values()
            if peak.steps > 0
        )
        current_resident = (
            int(self._residency.resident_bytes())
            + int((self._smart_plan or {}).get("singleton_resident_bytes", 0))
        )
        predicted_live = (
            worst_working
            + current_resident
            + self._training_ring_bytes()
        )
        return vram_budget.allocator_allowance_bytes(
            current_cap_bytes, predicted_live
        )

    def _apply_training_policy(self):
        import torch

        if torch.device(self._device).type != "cuda" or not torch.cuda.is_available():
            return
        candidate = self._promotion_candidate()
        demote_candidate = self._demotion_candidate()
        cliff_cap = allocator_cap.wddm_cliff_cap_bytes(
            self._device, self._config._policy.wddm_hard_gib
        )
        signal = self._signals.last_signal
        current_cap = min(
            cliff_cap,
            int(self._last_training_cap_target_bytes or cliff_cap),
        )
        if self._bootstrap_training_residency(current_cap):
            return
        aggressive_capacity = self._aggressive_promotion_capacity(current_cap)
        decision = self._policy.step(
            self._signals.last_signal,
            candidate=candidate,
            demote_candidate=demote_candidate,
            cliff_cap_bytes=cliff_cap,
            current_cap_bytes=current_cap,
            worst_shape_free_bytes=(
                self._worst_shape_candidate_physical_headroom_bytes(
                    candidate
                )
            ),
            worst_shape_allocator_slack_bytes=(
                self._worst_shape_allocator_slack_bytes(current_cap)
            ),
            aggressive_promotion_capacity=aggressive_capacity,
        )
        if decision.action == "promote":
            self.transition_training_block(decision.block_key, resident=True)
        elif decision.action in ("demote", "rollback"):
            rollback_keys = tuple(decision.block_keys or ())
            if rollback_keys:
                self.transition_training_blocks(
                    rollback_keys, resident=False
                )
            else:
                self.transition_training_block(
                    decision.block_key, resident=False
                )
            if (
                decision.action == "rollback"
                and decision.target_cap_bytes is not None
            ):
                allocator_cap.configure_wddm_allocator_guard(
                    self._device,
                    self._config._policy.wddm_hard_gib,
                    target_cap_bytes=decision.target_cap_bytes,
                    strict=getattr(
                        self._config, "strict_vram_cap", False
                    ),
                    log_prefix="[ArenaOffload]",
                )
                self._last_training_cap_target_bytes = (
                    decision.target_cap_bytes
                )
        elif decision.action == "raise_cap":
            allocator_cap.configure_wddm_allocator_guard(
                self._device,
                self._config._policy.wddm_hard_gib,
                target_cap_bytes=decision.target_cap_bytes,
                strict=getattr(self._config, "strict_vram_cap", False),
                log_prefix="[ArenaOffload]",
            )
            self._last_training_cap_target_bytes = decision.target_cap_bytes

    def transition_training_blocks(self, block_keys, *, resident: bool) -> dict:
        """Apply one executor-owned multi-block transaction at a boundary."""
        self._require_open()
        result = self._executor.transition_training_blocks(
            tuple(block_keys), resident=bool(resident)
        )
        if result.get("changed"):
            self._training_plan = result["plan"]
        return result

    def transition_training_block(self, block_key: str, *, resident: bool) -> dict:
        """Apply one executor-owned whole-block transaction at a boundary."""
        self._require_open()
        result = self._executor.transition_training_block(
            str(block_key), resident=bool(resident)
        )
        if result.get("changed"):
            self._training_plan = result["plan"]
        return result

    def _bind_training_cap(self) -> None:
        allocator_cap.configure_wddm_allocator_guard(
            self._device,
            self._config._policy.wddm_hard_gib,
            strict=getattr(self._config, "strict_vram_cap", False),
            log_prefix="[ArenaOffload]",
        )

    # diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """One stable dict. Shared logging prints it; nobody reconstructs it."""
        active_plan = getattr(self._residency, "plan", None) or self._training_plan
        canonical_resident = int(self._residency.resident_bytes())
        singleton_resident = int(
            (self._smart_plan or {}).get("singleton_resident_bytes", 0)
        )
        accounting = self._execution_accounting(active_plan)
        selection = getattr(self._executor, "selection", None)
        state_audit = getattr(selection, "accounting", None)
        return {
            "backend": "arena",
            "blocks": self.block_count,
            "finalized": self.finalized,
            "resident_bytes": singleton_resident + canonical_resident,
            "singleton_resident_bytes": singleton_resident,
            "canonical_resident_bytes": canonical_resident,
            "total_weight_resident_bytes": singleton_resident + canonical_resident,
            "plan_fingerprint": getattr(active_plan, "fingerprint", None),
            "checkpoint_owner": "model",
            "accounting": accounting,
            "state_audit": (
                None
                if state_audit is None
                else {
                    "managed_entries": int(state_audit.managed_entries),
                    "managed_bytes": int(state_audit.managed_bytes),
                    "trainable_entries": int(state_audit.trainable_entries),
                    "resident_entries": int(state_audit.resident_entries),
                    "resident_bytes": int(state_audit.resident_bytes),
                }
            ),
            "prefetch_depth": int(getattr(self._executor, "depth", 0)),
            "compile_blocks": bool(self._config.compile_blocks),
            "compile_dynamic": bool(self._config._compile_dynamic),
            "strict_vram_cap": bool(
                getattr(self._config, "strict_vram_cap", False)
            ),
            "fp8_forward": bool(self._config.fp8_forward),
            "fp8_backward": bool(self._config.fp8_backward),
            "fp8_sampling": bool(self._config.fp8_sampling),
            "training_fp8_canonical": getattr(
                self, "_training_fp8_canonical", 0
            ),
            "training_fp8_singletons": getattr(
                self, "_training_fp8_singletons", 0
            ),
            "sampling_fp8_canonical": getattr(
                self, "_sampling_fp8_canonical", 0
            ),
            "sampling_fp8_singletons": getattr(
                self, "_sampling_fp8_singletons", 0
            ),
            "largest_singleton_bf16_dequant_bytes": int(
                (self._smart_plan or {}).get(
                    "largest_singleton_bf16_dequant_bytes", 0
                )
            ),
            "training_cap_target_bytes": getattr(
                self, "_last_training_cap_target_bytes", None
            ),
            "bootstrap_complete": self._bootstrap_complete,
            "bootstrap_min_free_bytes": self._bootstrap_min_free_bytes,
            "bootstrap_margin_bytes": BOOTSTRAP_MARGIN_BYTES,
            "training_physical_vram_headroom_bytes": (
                self._training_physical_vram_headroom_bytes()
            ),
            "bootstrap_min_step": BOOTSTRAP_MIN_STEP,
            "bootstrap_budget_bytes": self._bootstrap_budget_bytes,
            "bootstrap_block_keys": self._bootstrap_block_keys,
            "working_reserve_bytes": int(
                (self._smart_plan or {}).get("working_reserve_bytes", 0)
            ),
            "all_resident_fit": bool(
                (self._smart_plan or {}).get("all_resident_fit", False)
            ),
            "all_resident_working_reserve_bytes": int(
                (self._smart_plan or {}).get(
                    "all_resident_working_reserve_bytes", 0
                )
            ),
            "last_shape_key": self._last_shape_key,
            "last_step_num": self._last_step_num,
            "successful_training_steps": self._successful_training_steps,
            "policy": {
                **self._signals.diagnostics(),
                "controller": self._policy.diagnostics(),
            },
            "policy_error": self._last_policy_error,
            "last_failure_event": self._last_failure_event,
        }

    def _execution_accounting(self, plan) -> dict:
        """Reconcile canonical payload, residency, and one execution's H2D plan."""
        from ..transfer_plan import build_transfer_plan

        resident_keys = frozenset(getattr(plan, "resident_leaf_keys", ()))
        canonical_committed = 0
        canonical_payload = 0
        streamed_payload = 0
        planned_forward_bytes = 0
        planned_copies = 0
        resident_leaves = 0
        streamed_leaves = 0
        resident_blocks = 0
        streamed_blocks = 0
        partial_blocks = 0

        for block_key in self._arena.block_keys():
            record = self._arena.block_record(block_key)
            canonical_committed += int(record.committed_bytes)
            all_leaves = tuple(record.leaf_names)
            canonical_payload += sum(
                int(tensor.nbytes)
                for leaf in all_leaves
                for tensor in record.leaf_spec(leaf).tensors
            )
            streamed = tuple(
                leaf
                for leaf in all_leaves
                if (block_key, leaf) not in resident_keys
            )
            resident_count = len(all_leaves) - len(streamed)
            resident_leaves += resident_count
            streamed_leaves += len(streamed)
            if not streamed:
                resident_blocks += 1
            elif resident_count == 0:
                streamed_blocks += 1
            else:
                partial_blocks += 1
            if streamed:
                transfer = build_transfer_plan(record, streamed)
                streamed_payload += sum(
                    int(tensor.nbytes)
                    for leaf in streamed
                    for tensor in record.leaf_spec(leaf).tensors
                )
                planned_forward_bytes += int(transfer.compact_nbytes)
                planned_copies += int(transfer.num_ranges)

        canonical_resident_payload = int(self._residency.resident_bytes())
        protected = self._protected_training_blocks()
        protected_resident = all(
            all(
                (block_key, leaf) in resident_keys
                for leaf in self._arena.block_record(block_key).leaf_names
            )
            for block_key in protected
        )
        training_multiplier = 2
        return {
            "phase": getattr(plan, "phase", None),
            "canonical_committed_bytes": canonical_committed,
            "canonical_payload_bytes": canonical_payload,
            "canonical_padding_bytes": canonical_committed - canonical_payload,
            "canonical_resident_payload_bytes": canonical_resident_payload,
            "streamed_payload_bytes": streamed_payload,
            "payload_reconciled": (
                canonical_payload
                == canonical_resident_payload + streamed_payload
            ),
            "resident_blocks": resident_blocks,
            "streamed_blocks": streamed_blocks,
            "partially_resident_blocks": partial_blocks,
            "resident_leaves": resident_leaves,
            "streamed_leaves": streamed_leaves,
            "mixed_residency": resident_leaves > 0 and streamed_leaves > 0,
            "planned_forward_h2d_bytes": planned_forward_bytes,
            "planned_forward_h2d_copies": planned_copies,
            "planned_training_h2d_bytes": (
                planned_forward_bytes * training_multiplier
            ),
            "planned_training_h2d_copies": planned_copies * training_multiplier,
            "protected_training_blocks": tuple(sorted(protected)),
            "protected_training_blocks_resident": protected_resident,
        }

    def _observe_training_step(self, *, shape_key, step_num, step_wall_ms) -> None:
        """Collect the completed step's policy signals."""
        import torch

        from . import transfer
        from ..vram_budget import device_free_bytes

        if torch.device(self._device).type != "cuda" or not torch.cuda.is_available():
            return
        try:
            stats = torch.cuda.memory_stats(self._device)
        except Exception:
            stats = {}
        allocator = {
            key: int(stats.get(key, 0) or 0)
            for key in ("num_alloc_retries", "num_device_alloc", "num_device_free")
        }
        transfer_stats = (
            transfer.lifetime_fetch_stats()
            if self._signals.transfer_snapshot_due
            else None
        )
        self._signals.observe(
            shape_key=shape_key,
            step_num=step_num,
            allocator_counters=allocator,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self._device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self._device),
            device_free_bytes=device_free_bytes(self._device),
            resident_bytes=(
                self._residency.resident_bytes()
                + int((self._smart_plan or {}).get("singleton_resident_bytes", 0))
            ),
            ring_bytes=self._training_ring_bytes(),
            compile_counters=_compile_counter_snapshot(torch),
            transfer_counters=transfer_stats,
            step_wall_ms=step_wall_ms,
        )
        self._successful_training_steps += 1

    def _training_ring_bytes(self) -> int:
        from ..transfer_plan import build_transfer_plan

        largest = 0
        plan = getattr(self._residency, "plan", None) or self._training_plan
        for block_key in self._arena.block_keys():
            record = self._arena.block_record(block_key)
            streamed = tuple(
                name
                for name in record.leaf_names
                if (block_key, name) not in plan.resident_leaf_keys
            )
            if streamed:
                transfer = build_transfer_plan(record, streamed)
                largest = max(largest, transfer.compact_nbytes)
        depth = max(1, int(getattr(self._executor, "depth", 1)))
        return int(largest * depth)

    def report_foreign_vram_once(self, *, phase: str) -> None:
        """Say so, once, if another tenant on the GPU is why we are streaming."""
        self._executor.report_foreign_vram_once(
            self._residency.device,
            phase=phase,
            working_reserve_bytes=int(
                (self._smart_plan or {}).get("working_reserve_bytes", 0)
            ),
        )

def _compile_counter_snapshot(torch_module):
    try:
        counters = torch_module._dynamo.utils.counters
    except AttributeError:
        return None
    frames = int(counters["frames"].get("total", 0) or 0)
    graphs = int(counters["stats"].get("unique_graphs", 0) or 0)
    if frames == 0 and graphs == 0:
        return None
    return {
        "frames": frames,
        "graphs": graphs,
        "graph_breaks": int(sum(counters["graph_break"].values())),
    }


def _fixed_working_bytes(value) -> int | None:
    """None when the sampling working reserve is auto (unset, negative, 'auto')."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        # Non-numeric (e.g. "auto") means auto-size, same as unset.
        return None
    if numeric < 0:
        return None
    return int(numeric * GIB)
