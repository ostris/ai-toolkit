"""Bounded allocator-cap calibration for arena training.

The PyTorch allocator cap is process-global, so bucket measurements feed one
worst-bucket cap. The calibrator moves that cap only during a bounded startup
or invalidation probe; it never switches caps on every bucket transition.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import vram_budget


CAP_WARMUP = "warmup"
CAP_PROBE_SETTLE = "probe_settle"
CAP_PROBE_VERIFY = "probe_verify"
CAP_SETTLED = "settled"
CAP_RESTORE_VERIFY = "restore_verify"

CAP_HOLD = "hold"
CAP_SET = "set_cap"

DEFAULT_CAP_NOTCH_BYTES = 256 * 1024**2


@dataclass
class BucketCapProfile:
    working_peak_bytes: int = 0
    valid_observations: int = 0
    seen_at_probe_cap: int | None = None
    dirty_at_probe_cap: int | None = None


@dataclass(frozen=True)
class CapCalibrationDecision:
    action: str = CAP_HOLD
    target_cap_bytes: int | None = None
    reason: str = ""
    hold_residency: bool = True


def _ceil_to_notch(value: int, notch_bytes: int) -> int:
    value = max(0, int(value))
    notch = max(1, int(notch_bytes))
    return ((value + notch - 1) // notch) * notch


class TrainingCapCalibrator:
    """Learn one safe allocator cap from per-bucket working peaks.

    A retry, allocator GC, OOM, or compile invalidation rejects a lowered cap.
    Runtime-owned cache trims must be excluded from the allocator counters
    before the probe window begins.
    Every bucket known when the probe begins must then produce a clean step.
    """

    def __init__(
        self,
        *,
        enabled=False,
        notch_bytes=DEFAULT_CAP_NOTCH_BYTES,
        monitor_settled=False,
    ):
        self.enabled = bool(enabled)
        self.monitor_settled = bool(monitor_settled)
        self.notch_bytes = max(1, int(notch_bytes))
        self.state = CAP_WARMUP
        self.bucket_profiles: dict[tuple, BucketCapProfile] = {}
        self.probe_cap_bytes: int | None = None
        self.last_clean_cap_bytes: int | None = None
        self.settled_cap_bytes: int | None = None
        self.learned_cache_pad_bytes: int | None = None
        self.predicted_initial_cap_bytes: int | None = None
        self.last_dirty_cap_bytes: int | None = None
        self._verify_pending: set[tuple] = set()
        self.last_action = CAP_HOLD
        self.last_reason = "disabled" if not self.enabled else "warmup"

    @property
    def active(self) -> bool:
        return self.enabled and self.state != CAP_SETTLED

    def step(
        self,
        signal,
        *,
        upcoming_shape_key,
        shape_peaks,
        resident_bytes,
        ring_bytes,
        cliff_cap_bytes,
        current_cap_bytes,
    ) -> CapCalibrationDecision:
        if not self.enabled:
            return self._decision(CAP_HOLD, reason="disabled", hold=False)

        cliff = max(0, int(cliff_cap_bytes))
        current = max(0, int(current_cap_bytes))
        upcoming = _shape_key(upcoming_shape_key)
        self._sync_profiles(shape_peaks)

        if signal is not None and bool(signal.get("compile_invalid")):
            self._reset_for_invalid_measurements()
            return self._set_or_hold(
                cliff, current, reason="compile_invalid", hold=True
            )

        decision = self._consume_previous_signal(
            signal,
            shape_peaks=shape_peaks,
            resident_bytes=resident_bytes,
            ring_bytes=ring_bytes,
            cliff_cap_bytes=cliff,
            current_cap_bytes=current,
        )
        if decision is not None:
            current = (
                current
                if decision.target_cap_bytes is None
                else int(decision.target_cap_bytes)
            )

        valid = self._valid_bucket_keys()
        if upcoming not in valid:
            self._restart_warmup(reason="unseen_bucket")
            return self._set_or_hold(
                cliff, current, reason="unseen_bucket", hold=True
            )

        if decision is not None:
            return decision

        if self.state == CAP_WARMUP:
            if not valid:
                return self._decision(CAP_HOLD, reason="awaiting_bucket", hold=True)
            allocator = (signal or {}).get("allocator") or {}
            if int(allocator.get("alloc_retries_delta", 0) or 0) > 0 or int(
                allocator.get("free_count_delta", 0) or 0
            ) > 0:
                return self._decision(
                    CAP_HOLD, reason="awaiting_clean_warmup", hold=True
                )
            return self._begin_initial_probe(
                resident_bytes=resident_bytes,
                ring_bytes=ring_bytes,
                cliff_cap_bytes=cliff,
                current_cap_bytes=current,
            )

        return self._decision(
            CAP_HOLD,
            reason="settled" if self.state == CAP_SETTLED else self.state,
            hold=self.state != CAP_SETTLED,
        )

    def allocation_failure(self, *, cliff_cap_bytes, current_cap_bytes):
        """Reject an active probe before residency/OOM recovery is attempted."""
        if not self.active or self.state == CAP_WARMUP:
            return None
        dirty_cap = self.probe_cap_bytes or int(current_cap_bytes)
        self._mark_dirty(None, dirty_cap)
        recovery_floor = min(
            int(cliff_cap_bytes),
            int(dirty_cap) + 2 * self.notch_bytes,
        )
        return self._begin_restore(
            cliff_cap_bytes=int(cliff_cap_bytes),
            current_cap_bytes=int(current_cap_bytes),
            reason="probe_allocation_failure",
            minimum_cap_bytes=recovery_floor,
        )

    def invalidate_for_residency_growth(self):
        """Keep working peaks but relearn the cap for a larger resident layout."""
        if not self.enabled:
            return
        self.last_clean_cap_bytes = None
        self.settled_cap_bytes = None
        self.learned_cache_pad_bytes = None
        self.predicted_initial_cap_bytes = None
        self.last_dirty_cap_bytes = None
        self._restart_warmup(reason="residency_growth")

    def _consume_previous_signal(
        self,
        signal,
        *,
        shape_peaks,
        resident_bytes,
        ring_bytes,
        cliff_cap_bytes,
        current_cap_bytes,
    ):
        if signal is None:
            return None
        allocator = signal.get("allocator") or {}
        retries = int(allocator.get("alloc_retries_delta", 0) or 0)
        frees = int(allocator.get("free_count_delta", 0) or 0)
        shape_key = _shape_key(signal.get("shape_key"))

        if self.state == CAP_SETTLED and self.monitor_settled:
            if retries > 0 or frees > 0:
                dirty_cap = int(self.settled_cap_bytes or current_cap_bytes)
                self._mark_dirty(shape_key, dirty_cap)
                widened = min(
                    int(cliff_cap_bytes), dirty_cap + self.notch_bytes
                )
                if widened > dirty_cap:
                    self.last_clean_cap_bytes = widened
                    self.settled_cap_bytes = None
                    self.state = CAP_RESTORE_VERIFY
                    return self._set_or_hold(
                        widened,
                        current_cap_bytes,
                        reason=(
                            "settled_allocator_retry"
                            if retries > 0
                            else "settled_allocator_gc"
                        ),
                        hold=True,
                    )
            return None

        if self.state == CAP_PROBE_SETTLE:
            if retries > 0 or frees > 0:
                self._mark_dirty(shape_key, self.probe_cap_bytes)
                return self._begin_restore(
                    cliff_cap_bytes=cliff_cap_bytes,
                    current_cap_bytes=current_cap_bytes,
                    reason=(
                        "probe_allocator_retry"
                        if retries > 0
                        else "probe_allocator_gc"
                    ),
                )
            self.state = CAP_PROBE_VERIFY
            self._verify_pending = set(self._valid_bucket_keys())
            return self._decision(CAP_HOLD, reason="probe_settled", hold=True)

        if self.state == CAP_PROBE_VERIFY:
            if retries > 0 or frees > 0:
                self._mark_dirty(shape_key, self.probe_cap_bytes)
                return self._begin_restore(
                    cliff_cap_bytes=cliff_cap_bytes,
                    current_cap_bytes=current_cap_bytes,
                    reason=(
                        "probe_allocator_retry"
                        if retries > 0
                        else "probe_allocator_gc"
                    ),
                )
            self._verify_pending.discard(shape_key)
            profile = self.bucket_profiles.get(shape_key)
            if profile is not None:
                profile.seen_at_probe_cap = self.probe_cap_bytes
            if self._verify_pending:
                return self._decision(
                    CAP_HOLD, reason="awaiting_probe_buckets", hold=True
                )
            return self._advance_clean_probe(
                shape_peaks=shape_peaks,
                resident_bytes=resident_bytes,
                ring_bytes=ring_bytes,
                cliff_cap_bytes=cliff_cap_bytes,
                current_cap_bytes=current_cap_bytes,
            )

        if self.state == CAP_RESTORE_VERIFY:
            if retries > 0 or frees > 0:
                self._restart_warmup(reason="restore_not_clean")
                return self._set_or_hold(
                    cliff_cap_bytes,
                    current_cap_bytes,
                    reason="restore_not_clean",
                    hold=True,
                )
            self.state = CAP_SETTLED
            self.settled_cap_bytes = self.last_clean_cap_bytes
            return self._decision(CAP_HOLD, reason="calibration_settled", hold=False)

        return None

    def _begin_initial_probe(
        self, *, resident_bytes, ring_bytes, cliff_cap_bytes, current_cap_bytes
    ):
        worst_live = self._worst_live_bytes(resident_bytes, ring_bytes)
        predicted = _ceil_to_notch(
            int(worst_live / vram_budget.GC_THRESHOLD), self.notch_bytes
        ) + self.notch_bytes
        candidate = min(int(cliff_cap_bytes), predicted)
        self.predicted_initial_cap_bytes = candidate
        self.last_clean_cap_bytes = int(cliff_cap_bytes)
        if candidate >= int(cliff_cap_bytes):
            self.state = CAP_SETTLED
            self.settled_cap_bytes = int(cliff_cap_bytes)
            self.learned_cache_pad_bytes = max(
                0,
                vram_budget.allocator_allowance_bytes(
                    self.settled_cap_bytes, worst_live
                ),
            )
            return self._decision(
                CAP_HOLD, reason="no_reclaimable_notch", hold=False
            )
        return self._begin_probe(candidate, current_cap_bytes, "initial_probe")

    def _advance_clean_probe(
        self,
        *,
        shape_peaks,
        resident_bytes,
        ring_bytes,
        cliff_cap_bytes,
        current_cap_bytes,
    ):
        clean_cap = int(self.probe_cap_bytes or current_cap_bytes)
        self.last_clean_cap_bytes = clean_cap
        worst_live = self._worst_live_bytes(resident_bytes, ring_bytes)
        self.learned_cache_pad_bytes = max(
            0, vram_budget.allocator_allowance_bytes(clean_cap, worst_live)
        )
        next_cap = clean_cap - self.notch_bytes
        minimum = _ceil_to_notch(
            int(worst_live / vram_budget.GC_THRESHOLD), self.notch_bytes
        ) - self.notch_bytes
        next_cap = max(self.notch_bytes, minimum, next_cap)
        if next_cap >= clean_cap:
            self.state = CAP_SETTLED
            self.settled_cap_bytes = clean_cap
            return self._decision(CAP_HOLD, reason="calibration_floor", hold=False)
        return self._begin_probe(next_cap, current_cap_bytes, "lower_probe")

    def _begin_probe(self, target_cap_bytes, current_cap_bytes, reason):
        self.state = CAP_PROBE_SETTLE
        self.probe_cap_bytes = int(target_cap_bytes)
        self._verify_pending.clear()
        return self._set_or_hold(
            self.probe_cap_bytes,
            int(current_cap_bytes),
            reason=reason,
            hold=True,
        )

    def _begin_restore(
        self,
        *,
        cliff_cap_bytes,
        current_cap_bytes,
        reason,
        minimum_cap_bytes=0,
    ):
        restore = min(
            int(cliff_cap_bytes),
            max(
                int(self.last_clean_cap_bytes or cliff_cap_bytes),
                int(minimum_cap_bytes),
            ),
        )
        self.last_clean_cap_bytes = restore
        self.state = CAP_RESTORE_VERIFY
        self.probe_cap_bytes = None
        self._verify_pending.clear()
        return self._set_or_hold(
            restore, int(current_cap_bytes), reason=reason, hold=True
        )

    def _sync_profiles(self, shape_peaks):
        for key, peak in (shape_peaks or {}).items():
            normalized = _shape_key(key)
            steps = int(getattr(peak, "steps", 0) or 0)
            if steps <= 0:
                continue
            profile = self.bucket_profiles.setdefault(
                normalized, BucketCapProfile()
            )
            profile.working_peak_bytes = max(
                profile.working_peak_bytes,
                int(getattr(peak, "working_peak_bytes", 0) or 0),
            )
            profile.valid_observations = max(profile.valid_observations, steps)

    def _valid_bucket_keys(self):
        return {
            key
            for key, profile in self.bucket_profiles.items()
            if profile.valid_observations > 0
        }

    def _worst_live_bytes(self, resident_bytes, ring_bytes):
        worst_working = max(
            (
                profile.working_peak_bytes
                for profile in self.bucket_profiles.values()
                if profile.valid_observations > 0
            ),
            default=0,
        )
        return (
            int(worst_working)
            + max(0, int(resident_bytes))
            + max(0, int(ring_bytes))
        )

    def _mark_dirty(self, shape_key, cap_bytes):
        self.last_dirty_cap_bytes = (
            None if cap_bytes is None else int(cap_bytes)
        )
        if shape_key is None:
            return
        profile = self.bucket_profiles.get(_shape_key(shape_key))
        if profile is not None:
            profile.dirty_at_probe_cap = (
                None if cap_bytes is None else int(cap_bytes)
            )

    def _reset_for_invalid_measurements(self):
        self.bucket_profiles.clear()
        self.last_clean_cap_bytes = None
        self.settled_cap_bytes = None
        self.learned_cache_pad_bytes = None
        self.predicted_initial_cap_bytes = None
        self.last_dirty_cap_bytes = None
        self._restart_warmup(reason="compile_invalid")

    def _restart_warmup(self, *, reason):
        self.state = CAP_WARMUP
        self.probe_cap_bytes = None
        self.settled_cap_bytes = None
        self._verify_pending.clear()
        self.last_reason = reason

    def _set_or_hold(self, target, current, *, reason, hold):
        if int(target) == int(current):
            return self._decision(CAP_HOLD, reason=reason, hold=hold)
        return self._decision(CAP_SET, int(target), reason=reason, hold=hold)

    def _decision(self, action, target=None, *, reason, hold):
        self.last_action = action
        self.last_reason = reason
        return CapCalibrationDecision(
            action=action,
            target_cap_bytes=target,
            reason=reason,
            hold_residency=bool(hold),
        )

    def diagnostics(self):
        return {
            "enabled": self.enabled,
            "monitor_settled": self.monitor_settled,
            "state": self.state,
            "notch_bytes": self.notch_bytes,
            "probe_cap_bytes": self.probe_cap_bytes,
            "last_clean_cap_bytes": self.last_clean_cap_bytes,
            "settled_cap_bytes": self.settled_cap_bytes,
            "learned_cache_pad_bytes": self.learned_cache_pad_bytes,
            "predicted_initial_cap_bytes": self.predicted_initial_cap_bytes,
            "last_dirty_cap_bytes": self.last_dirty_cap_bytes,
            "verify_pending": tuple(sorted(self._verify_pending, key=repr)),
            "last_action": self.last_action,
            "last_reason": self.last_reason,
            "buckets": [
                {
                    "shape_key": key,
                    "working_peak_bytes": profile.working_peak_bytes,
                    "valid_observations": profile.valid_observations,
                    "seen_at_probe_cap": profile.seen_at_probe_cap,
                    "dirty_at_probe_cap": profile.dirty_at_probe_cap,
                }
                for key, profile in sorted(
                    self.bucket_profiles.items(), key=lambda item: repr(item[0])
                )
            ],
        }


def _shape_key(value):
    if value is None:
        return ("unknown",)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)
