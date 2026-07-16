"""Arena-native training signals and two-timescale residency policy."""

from __future__ import annotations

from dataclasses import dataclass

from .. import vram_budget

_ALLOC = ("num_alloc_retries", "num_device_alloc", "num_device_free")
_COMPILE = ("frames", "graphs", "graph_breaks")

DEFAULT_ALLOCATOR_CACHE_HEADROOM_BYTES = 256 * 1024**2
AGGRESSIVE_PROMOTION_MIN_CAPACITY = 4


def transfer_benefits_from_residency(transfer) -> bool:
    """Return whether a valid window proves that weights are still streaming."""
    if not transfer or transfer.get("h2d_duty_overflow"):
        return False
    return (
        int(transfer.get("bytes", 0) or 0) > 0
        and float(transfer.get("h2d_ms", 0.0) or 0.0) > 0.0
    )


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    block_key: str | None = None
    block_bytes: int = 0
    target_cap_bytes: int | None = None
    reason: str = ""
    block_keys: tuple[str, ...] = ()


class ArenaResidencyController:
    """Stateful wiring around the pure two-timescale residency FSM."""

    def __init__(
        self,
        *,
        allocator_cache_headroom_bytes=(
            DEFAULT_ALLOCATOR_CACHE_HEADROOM_BYTES
        ),
    ):
        self.state = vram_budget.ResidencyFsmState()
        self.allocator_cache_headroom_bytes = max(
            0, int(allocator_cache_headroom_bytes)
        )
        self.last_action = "hold"
        self.last_reason = "cold_start"
        self.last_promoted_key = None
        self.last_block_key = None
        self.last_block_bytes = 0
        self.last_target_cap_bytes = None
        self.last_worst_shape_physical_headroom_bytes = None
        self.last_throughput_gate = None
        self.last_promote_gate = None
        self.last_cap_covers_promo = None
        self.last_worst_shape_allocator_slack_bytes = None
        self.last_aggressive_capacity = 0
        self.last_aggressive_gate = False
        self.last_needed_promotion_cap_bytes = None
        self.last_cache_pad_bytes = self.allocator_cache_headroom_bytes
        self.pending_promotion = None
        self.last_safe_residency_bytes = None
        self.last_rejected_residency_bytes = None
        self.bootstrapped = False

    def step(self, signal, *, candidate, demote_candidate, cliff_cap_bytes,
             worst_shape_free_bytes, worst_shape_allocator_slack_bytes=None,
             current_cap_bytes=None, aggressive_promotion_capacity=0,
             worst_shape_live_bytes=None, learned_cache_pad_bytes=None):
        if not self.bootstrapped:
            self.bootstrapped = True
            if demote_candidate is not None:
                return self._decision(
                    "demote",
                    demote_candidate,
                    reason="approach_from_below",
                )
        if signal is None:
            return self._hold("awaiting_signal")

        block_bytes = 0 if candidate is None else int(candidate["block_bytes"])
        allocator = signal.get("allocator") or {}
        retries = int(allocator.get("alloc_retries_delta", 0) or 0)
        device_frees = int(allocator.get("free_count_delta", 0) or 0)
        allocator_slack = int(
            signal.get("reclaimable_at_peak_bytes", 0)
            if worst_shape_allocator_slack_bytes is None
            else worst_shape_allocator_slack_bytes
        )
        exact_funding = (
            learned_cache_pad_bytes is not None
            and worst_shape_live_bytes is not None
        )
        cache_pad = (
            self.allocator_cache_headroom_bytes
            if learned_cache_pad_bytes is None
            else max(0, int(learned_cache_pad_bytes))
        )
        throughput_ok = transfer_benefits_from_residency(signal.get("transfer"))
        worst_ok = candidate is not None and int(worst_shape_free_bytes) >= 0
        if exact_funding:
            promotion_cap_possible = (
                candidate is not None
                and vram_budget.cap_can_host_promotion(
                    int(worst_shape_live_bytes),
                    block_bytes,
                    cache_pad,
                    int(cliff_cap_bytes),
                )
            )
            promote_ok = (
                candidate is not None
                and promotion_cap_possible
                and throughput_ok
                and worst_ok
                and retries == 0
                and device_frees == 0
            )
        else:
            promotion_cap_possible = True
            promote_ok = (
                candidate is not None
                and throughput_ok
                and worst_ok
                and vram_budget.residency_promote_ok(
                    retries,
                    allocator_slack,
                    block_bytes,
                    cache_pad,
                )
            )
        active_cap = (
            int(cliff_cap_bytes)
            if current_cap_bytes is None
            else int(current_cap_bytes)
        )
        if exact_funding:
            needed_promotion_cap = vram_budget.cap_bytes_for_live(
                int(worst_shape_live_bytes) + block_bytes,
                cache_pad,
                int(cliff_cap_bytes),
            )
            cap_covers = (
                candidate is not None
                and active_cap >= needed_promotion_cap
            )
            binding = retries > 0
            pressure_needed_cap = vram_budget.cap_bytes_for_live(
                int(worst_shape_live_bytes),
                cache_pad,
                int(cliff_cap_bytes),
            )
            needed_cap = (
                needed_promotion_cap
                if promote_ok and not cap_covers
                else pressure_needed_cap
            )
        else:
            needed_promotion_cap = None
            cap_covers = (
                candidate is not None
                and allocator_slack > block_bytes + cache_pad
            )
            binding = retries > 0 or int(worst_shape_free_bytes) < 0
            cap_raise_bytes = (
                block_bytes
                if promote_ok and block_bytes > 0
                else cache_pad
            )
            needed_cap = min(
                int(cliff_cap_bytes),
                active_cap + max(0, int(cap_raise_bytes)),
            )
        aggressive_capacity = max(0, int(aggressive_promotion_capacity or 0))
        aggressive_ok = (
            candidate is not None
            and aggressive_capacity >= AGGRESSIVE_PROMOTION_MIN_CAPACITY
            and not bool(signal.get("compile_invalid"))
            and retries == 0
            and device_frees == 0
            and worst_ok
            and allocator_slack > block_bytes + cache_pad
        )
        self.last_worst_shape_physical_headroom_bytes = int(
            worst_shape_free_bytes
        )
        self.last_worst_shape_allocator_slack_bytes = allocator_slack
        self.last_throughput_gate = bool(throughput_ok)
        self.last_promote_gate = bool(promote_ok)
        self.last_cap_covers_promo = bool(cap_covers)
        self.last_aggressive_capacity = aggressive_capacity
        self.last_aggressive_gate = bool(aggressive_ok)
        self.last_needed_promotion_cap_bytes = needed_promotion_cap
        self.last_cache_pad_bytes = cache_pad
        if (
            self.pending_promotion is not None
            and (retries > 0 or device_frees > 0)
        ):
            return self.reject_pending_promotion("promotion_allocator_gc")

        # Abundant measured headroom does not need the multi-window transfer
        # proof. Spend only one block per step, leaving at least three blocks of
        # measured capacity in reserve, and make the new block the rollback
        # candidate for the next boundary.
        if aggressive_ok:
            self._begin_promotion(
                candidate,
                resident_bytes_before=int(signal.get("resident_bytes", 0) or 0),
                active_cap_bytes=active_cap,
            )
            return self._decision(
                "promote", candidate, reason="abundant_four_block_headroom"
            )

        previous_state = self.state.name
        self.state, action = vram_budget.residency_fsm_step(
            self.state,
            {
                "measurements_invalid": bool(signal.get("compile_invalid")),
                "binding": binding,
                "cap_can_relieve": (
                    active_cap < needed_cap <= int(cliff_cap_bytes)
                ),
                "promote_gate": promote_ok,
                "cap_covers_promo": cap_covers,
            },
        )
        if (
            previous_state == vram_budget.FSM_PROMOTION_VERIFY
            and self.state.name == vram_budget.FSM_STABLE
        ):
            self.pending_promotion = None
            self.last_promoted_key = None
        if action == vram_budget.ACT_PROMOTE and candidate is not None:
            self._begin_promotion(
                candidate,
                resident_bytes_before=int(signal.get("resident_bytes", 0) or 0),
                active_cap_bytes=active_cap,
            )
            return self._decision(
                action, candidate, reason="safe_transfer_benefit"
            )
        if (
            action == vram_budget.ACT_ROLLBACK
            and self.pending_promotion is not None
        ):
            return self.reject_pending_promotion("promotion_bound")
        if action == vram_budget.ACT_DEMOTE and demote_candidate is not None:
            return self._decision(
                action, demote_candidate, reason="live_pressure"
            )
        if action == vram_budget.ACT_RAISE_CAP:
            self.last_action = action
            self.last_reason = "prefund_or_relieve"
            self.last_block_key = None
            self.last_block_bytes = 0
            self.last_target_cap_bytes = needed_cap
            return PolicyDecision(
                action, target_cap_bytes=needed_cap, reason=self.last_reason
            )
        reason = (
            "worst_shape_veto"
            if candidate is not None and not worst_ok
            else (
                "promotion_exceeds_cliff"
                if candidate is not None
                and exact_funding
                and not promotion_cap_possible
                else (
                "throughput_gate"
                if candidate is not None and not throughput_ok
                else (
                    "allocator_cache_headroom"
                    if candidate is not None
                    and not exact_funding
                    and allocator_slack <= block_bytes + cache_pad
                    else "fsm_hold"
                )
                )
            )
        )
        return self._hold(reason, candidate=candidate)

    def reject_pending_promotion(self, reason):
        pending = self.pending_promotion
        if pending is None:
            return self._hold(reason)
        self.last_safe_residency_bytes = int(
            pending["resident_bytes_before"]
        )
        self.last_rejected_residency_bytes = int(
            pending["resident_bytes_after"]
        )
        self.pending_promotion = None
        self.last_promoted_key = None
        self.state = vram_budget.ResidencyFsmState(
            vram_budget.FSM_COOLDOWN, 0
        )
        self.last_action = "rollback"
        self.last_reason = reason
        self.last_block_key = pending["block_key"]
        self.last_block_bytes = int(pending["block_bytes"])
        self.last_target_cap_bytes = int(pending["previous_cap_target_bytes"])
        return PolicyDecision(
            "rollback",
            pending["block_key"],
            int(pending["block_bytes"]),
            target_cap_bytes=int(pending["previous_cap_target_bytes"]),
            reason=reason,
            block_keys=tuple(
                pending.get("block_keys") or (pending["block_key"],)
            ),
        )

    def _begin_promotion(
        self, candidate, *, resident_bytes_before, active_cap_bytes
    ):
        block_bytes = int(candidate["block_bytes"])
        self.last_promoted_key = candidate["block_key"]
        self.pending_promotion = {
            "block_key": candidate["block_key"],
            "block_bytes": block_bytes,
            "resident_bytes_before": int(resident_bytes_before),
            "resident_bytes_after": int(resident_bytes_before) + block_bytes,
            "previous_cap_target_bytes": int(active_cap_bytes),
        }
        self.state = vram_budget.ResidencyFsmState(
            vram_budget.FSM_PROMOTION_VERIFY, 0
        )

    def allocation_failure(self):
        return self.reject_pending_promotion("promotion_allocation_failure")

    def record_physical_pressure_relief(self, block_keys, block_bytes):
        """Put the controller in cooldown after emergency layout relief."""
        keys = tuple(str(key) for key in block_keys)
        self.pending_promotion = None
        self.last_promoted_key = None
        self.state = vram_budget.ResidencyFsmState(
            vram_budget.FSM_COOLDOWN, 0
        )
        self.last_action = "demote" if keys else "hold"
        self.last_reason = "physical_pressure_relief"
        self.last_block_key = keys[-1] if keys else None
        self.last_block_bytes = int(block_bytes)
        self.last_target_cap_bytes = None

    def _decision(self, action, candidate, *, reason):
        self.last_action = action
        self.last_reason = reason
        self.last_block_key = candidate["block_key"]
        self.last_block_bytes = int(candidate["block_bytes"])
        self.last_target_cap_bytes = None
        return PolicyDecision(
            action,
            candidate["block_key"],
            int(candidate["block_bytes"]),
            reason=reason,
        )

    def _hold(self, reason, *, candidate=None):
        self.last_action = "hold"
        self.last_reason = reason
        self.last_block_key = (
            None if candidate is None else candidate["block_key"]
        )
        self.last_block_bytes = (
            0 if candidate is None else int(candidate["block_bytes"])
        )
        self.last_target_cap_bytes = None
        return PolicyDecision("hold", reason=reason)

    def diagnostics(self):
        return {
            "state": self.state.name,
            "windows_in_state": self.state.windows_in_state,
            "last_action": self.last_action,
            "last_reason": self.last_reason,
            "last_promoted_key": self.last_promoted_key,
            "last_block_key": self.last_block_key,
            "last_block_bytes": self.last_block_bytes,
            "last_target_cap_bytes": self.last_target_cap_bytes,
            "last_worst_shape_physical_headroom_bytes": (
                self.last_worst_shape_physical_headroom_bytes
            ),
            "last_throughput_gate": self.last_throughput_gate,
            "last_promote_gate": self.last_promote_gate,
            "last_cap_covers_promo": self.last_cap_covers_promo,
            "last_worst_shape_allocator_slack_bytes": (
                self.last_worst_shape_allocator_slack_bytes
            ),
            "last_aggressive_capacity": self.last_aggressive_capacity,
            "last_aggressive_gate": self.last_aggressive_gate,
            "last_needed_promotion_cap_bytes": (
                self.last_needed_promotion_cap_bytes
            ),
            "last_cache_pad_bytes": self.last_cache_pad_bytes,
            "pending_promotion": self.pending_promotion,
            "last_safe_residency_bytes": self.last_safe_residency_bytes,
            "last_rejected_residency_bytes": (
                self.last_rejected_residency_bytes
            ),
            "allocator_cache_headroom_bytes": (
                self.allocator_cache_headroom_bytes
            ),
        }


def _deltas(previous, current, keys, cast=int):
    result = {}
    for key in keys:
        now = cast((current or {}).get(key, 0) or 0)
        before = cast((previous or {}).get(key, 0) or 0)
        result[key] = now - before if now >= before else now
    return result


@dataclass(frozen=True)
class ShapePeak:
    steps: int = 0
    warmup_steps: int = 1
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    working_peak_bytes: int = 0


class TrainingSignalWindow:
    """Runtime-owned, CPU-testable training policy observations."""

    def __init__(self, *, transfer_window_steps=4):
        self.transfer_window_steps = max(2, int(transfer_window_steps))
        self._allocator_previous = None
        self._compile_previous = None
        self._transfer_previous = None
        self._transfer_steps = 0
        self._transfer_wall_ms = 0.0
        self._transfer_h2d_ms = 0.0
        self._transfer_bytes = 0
        self._shape_peaks = {}
        self._last_signal = None

    @property
    def shape_peaks(self):
        return dict(self._shape_peaks)

    @property
    def last_signal(self):
        return None if self._last_signal is None else dict(self._last_signal)

    @property
    def transfer_snapshot_due(self):
        return self._transfer_steps + 1 >= self.transfer_window_steps

    def prime_counters(self, *, allocator_counters=None, compile_counters=None):
        """Start a new observation phase without attributing old activity."""
        if allocator_counters is not None:
            self._allocator_previous = {
                key: int((allocator_counters or {}).get(key, 0) or 0)
                for key in _ALLOC
            }
        if compile_counters is not None:
            self._compile_previous = {
                key: int((compile_counters or {}).get(key, 0) or 0)
                for key in _COMPILE
            }

    def invalidate_shape_peaks(self):
        self._shape_peaks.clear()
        self._last_signal = None

    def observe(
        self, *, shape_key, step_num, allocator_counters,
        peak_allocated_bytes, peak_reserved_bytes, device_free_bytes,
        resident_bytes, ring_bytes, compile_counters=None,
        transfer_counters=None, step_wall_ms=0.0,
    ):
        alloc_delta = _deltas(self._allocator_previous, allocator_counters, _ALLOC)
        self._allocator_previous = {
            key: int((allocator_counters or {}).get(key, 0) or 0) for key in _ALLOC
        }
        compile_delta = _deltas(
            self._compile_previous, compile_counters, _COMPILE
        )
        compile_invalid = compile_counters is not None and compile_delta["frames"] > 0
        if compile_counters is not None:
            self._compile_previous = {
                key: int(compile_counters.get(key, 0) or 0) for key in _COMPILE
            }
        if compile_invalid:
            self.invalidate_shape_peaks()

        key = _shape_key(shape_key)
        if not compile_invalid:
            self._record_shape_peak(
                key,
                int(peak_allocated_bytes or 0),
                int(peak_reserved_bytes or 0),
                int(resident_bytes or 0),
                int(ring_bytes or 0),
            )
        transfer = self._observe_transfer(transfer_counters, float(step_wall_ms or 0.0))
        allocated = int(peak_allocated_bytes or 0)
        reserved = int(peak_reserved_bytes or 0)
        resident = int(resident_bytes or 0)
        ring = int(ring_bytes or 0)
        signal = {
            "shape_key": key,
            "step_num": None if step_num is None else int(step_num),
            "allocator": {
                "alloc_retries_delta": alloc_delta["num_alloc_retries"],
                "alloc_count_delta": alloc_delta["num_device_alloc"],
                "free_count_delta": alloc_delta["num_device_free"],
            },
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "reclaimable_at_peak_bytes": max(0, reserved - allocated),
            "device_free_bytes": int(device_free_bytes or 0),
            "resident_bytes": resident,
            "ring_bytes": ring,
            "live_bytes": resident + ring,
            "compile_invalid": bool(compile_invalid),
            "compile_delta": compile_delta,
            "transfer": transfer,
        }
        self._last_signal = signal
        return dict(signal)

    def diagnostics(self):
        return {
            "last_signal": self.last_signal,
            "shape_peaks": [
                {
                    "shape_key": key,
                    "steps": peak.steps,
                    "warmup_steps": peak.warmup_steps,
                    "peak_allocated_bytes": peak.peak_allocated_bytes,
                    "peak_reserved_bytes": peak.peak_reserved_bytes,
                    "working_peak_bytes": peak.working_peak_bytes,
                }
                for key, peak in self._shape_peaks.items()
            ],
            "transfer_window_steps": self.transfer_window_steps,
            "transfer_pending_steps": self._transfer_steps,
        }

    def _record_shape_peak(self, key, allocated, reserved, resident, ring):
        previous = self._shape_peaks.get(key)
        if previous is None:
            self._shape_peaks[key] = ShapePeak()
            return
        self._shape_peaks[key] = ShapePeak(
            steps=previous.steps + 1,
            warmup_steps=previous.warmup_steps,
            peak_allocated_bytes=max(previous.peak_allocated_bytes, allocated),
            peak_reserved_bytes=max(previous.peak_reserved_bytes, reserved),
            working_peak_bytes=max(
                previous.working_peak_bytes,
                max(0, allocated - resident - ring),
            ),
        )

    def _observe_transfer(self, counters, wall_ms):
        self._transfer_steps += 1
        self._transfer_wall_ms += max(0.0, wall_ms)
        if counters is not None:
            h2d = _deltas(
                self._transfer_previous, counters, ("h2d_ms",), float
            )["h2d_ms"]
            byte_count = _deltas(
                self._transfer_previous, counters, ("bytes",)
            )["bytes"]
            self._transfer_previous = {
                "h2d_ms": float(counters.get("h2d_ms", 0.0) or 0.0),
                "bytes": int(counters.get("bytes", 0) or 0),
            }
            self._transfer_h2d_ms += h2d
            self._transfer_bytes += byte_count
        if self._transfer_steps < self.transfer_window_steps:
            return None
        duty = (
            None if self._transfer_wall_ms <= 0.0
            else 100.0 * self._transfer_h2d_ms / self._transfer_wall_ms
        )
        gbps = (
            None if self._transfer_h2d_ms <= 0.0
            else self._transfer_bytes / (self._transfer_h2d_ms * 1_000_000.0)
        )
        result = {
            "steps": self._transfer_steps,
            "step_wall_ms": self._transfer_wall_ms,
            "h2d_ms": self._transfer_h2d_ms,
            "bytes": self._transfer_bytes,
            "h2d_duty_pct": duty,
            "h2d_duty_overflow": bool(duty is not None and duty > 100.0),
            "achieved_gbps": gbps,
        }
        self._transfer_steps = 0
        self._transfer_wall_ms = 0.0
        self._transfer_h2d_ms = 0.0
        self._transfer_bytes = 0
        return result


def _shape_key(value):
    if value is None:
        return ("unknown",)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)
