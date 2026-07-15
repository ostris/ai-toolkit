from types import SimpleNamespace

from toolkit.memory_management.arena_offload.cap_calibrator import (
    CAP_PROBE_SETTLE,
    CAP_PROBE_VERIFY,
    CAP_RESTORE_VERIFY,
    CAP_SET,
    CAP_SETTLED,
    TrainingCapCalibrator,
)


def peak(working, steps=2):
    return SimpleNamespace(working_peak_bytes=working, steps=steps)


def signal(shape, *, retries=0, frees=0, compile_invalid=False):
    return {
        "shape_key": shape,
        "compile_invalid": compile_invalid,
        "allocator": {
            "alloc_retries_delta": retries,
            "free_count_delta": frees,
        },
    }


def drive(calibrator, previous, upcoming, peaks, current):
    return calibrator.step(
        previous,
        upcoming_shape_key=upcoming,
        shape_peaks=peaks,
        resident_bytes=100,
        ring_bytes=0,
        cliff_cap_bytes=1000,
        current_cap_bytes=current,
    )


def test_calibrator_verifies_every_bucket_and_restores_last_clean_cap():
    calibrator = TrainingCapCalibrator(enabled=True, notch_bytes=100)
    peaks = {("a",): peak(300), ("b",): peak(400)}

    decision = drive(calibrator, None, ("a",), peaks, 1000)
    assert decision.action == CAP_SET
    assert decision.target_cap_bytes == 700
    assert calibrator.state == CAP_PROBE_SETTLE

    decision = drive(calibrator, signal(("a",)), ("a",), peaks, 700)
    assert decision.action != CAP_SET
    assert calibrator.state == CAP_PROBE_VERIFY

    decision = drive(calibrator, signal(("a",)), ("b",), peaks, 700)
    assert decision.reason == "awaiting_probe_buckets"
    decision = drive(calibrator, signal(("b",)), ("a",), peaks, 700)
    assert decision.target_cap_bytes == 600

    drive(calibrator, signal(("a",)), ("a",), peaks, 600)
    drive(calibrator, signal(("a",)), ("b",), peaks, 600)
    decision = drive(calibrator, signal(("b",)), ("a",), peaks, 600)
    assert decision.target_cap_bytes == 500

    decision = drive(
        calibrator, signal(("a",), frees=1), ("a",), peaks, 500
    )
    assert decision.action == CAP_SET
    assert decision.target_cap_bytes == 600
    assert decision.reason == "probe_allocator_gc"
    assert calibrator.state == CAP_RESTORE_VERIFY

    decision = drive(calibrator, signal(("a",)), ("a",), peaks, 600)
    assert decision.hold_residency is False
    assert calibrator.state == CAP_SETTLED
    assert calibrator.settled_cap_bytes == 600
    assert calibrator.learned_cache_pad_bytes == 70
    profiles = calibrator.diagnostics()["buckets"]
    assert {row["shape_key"] for row in profiles} == {("a",), ("b",)}
    assert any(row["dirty_at_probe_cap"] == 500 for row in profiles)


def test_first_gc_during_probe_restores_last_clean_cap():
    calibrator = TrainingCapCalibrator(enabled=True, notch_bytes=100)
    peaks = {("a",): peak(400)}
    drive(calibrator, None, ("a",), peaks, 1000)

    decision = drive(
        calibrator, signal(("a",), frees=1), ("a",), peaks, 700
    )

    assert decision.action == CAP_SET
    assert decision.target_cap_bytes == 1000
    assert decision.reason == "probe_allocator_gc"
    assert calibrator.last_dirty_cap_bytes == 700
    assert calibrator.state == CAP_RESTORE_VERIFY


def test_unseen_bucket_restores_cliff_and_invalidates_settlement():
    calibrator = TrainingCapCalibrator(enabled=True, notch_bytes=100)
    calibrator.state = CAP_SETTLED
    calibrator.settled_cap_bytes = 600
    calibrator.learned_cache_pad_bytes = 70
    calibrator.bucket_profiles[("a",)] = SimpleNamespace(
        working_peak_bytes=400,
        valid_observations=2,
        seen_at_probe_cap=600,
        dirty_at_probe_cap=None,
    )

    decision = drive(calibrator, signal(("a",)), ("new",), {}, 600)
    assert decision.action == CAP_SET
    assert decision.target_cap_bytes == 1000
    assert decision.hold_residency is True


def test_compile_invalid_restores_cliff_and_discards_bucket_profiles():
    calibrator = TrainingCapCalibrator(enabled=True, notch_bytes=100)
    peaks = {("a",): peak(400)}
    drive(calibrator, None, ("a",), peaks, 1000)

    decision = drive(
        calibrator,
        signal(("a",), compile_invalid=True),
        ("a",),
        {},
        700,
    )
    assert decision.action == CAP_SET
    assert decision.target_cap_bytes == 1000
    assert calibrator.bucket_profiles == {}


def test_probe_oom_restores_before_residency_recovery():
    calibrator = TrainingCapCalibrator(enabled=True, notch_bytes=100)
    peaks = {("a",): peak(400)}
    drive(calibrator, None, ("a",), peaks, 1000)

    decision = calibrator.allocation_failure(
        cliff_cap_bytes=1000, current_cap_bytes=700
    )
    assert decision.action == CAP_SET
    assert decision.target_cap_bytes == 1000
    assert calibrator.state == CAP_RESTORE_VERIFY


def test_probe_oom_adds_two_notches_when_last_clean_is_too_close():
    calibrator = TrainingCapCalibrator(enabled=True, notch_bytes=100)
    calibrator.state = CAP_PROBE_SETTLE
    calibrator.probe_cap_bytes = 500
    calibrator.last_clean_cap_bytes = 600

    decision = calibrator.allocation_failure(
        cliff_cap_bytes=1000, current_cap_bytes=500
    )

    assert decision.action == CAP_SET
    assert decision.target_cap_bytes == 700
    assert calibrator.last_clean_cap_bytes == 700
    assert calibrator.state == CAP_RESTORE_VERIFY
