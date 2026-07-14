import contextlib
import json
from types import SimpleNamespace

import pytest

from toolkit.memory_management.arena_offload.policy import (
    ArenaResidencyController,
    TrainingSignalWindow,
    transfer_benefits_from_residency,
)
from toolkit.memory_management.arena_offload.runtime import ArenaOffloadRuntime


def observe(window, **overrides):
    values = {
        "shape_key": (512, 512),
        "step_num": 1,
        "allocator_counters": {
            "num_alloc_retries": 10,
            "num_device_alloc": 20,
            "num_device_free": 30,
        },
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 140,
        "device_free_bytes": 500,
        "resident_bytes": 200,
        "ring_bytes": 80,
        "compile_counters": None,
        "transfer_counters": None,
        "step_wall_ms": 10.0,
    }
    values.update(overrides)
    return window.observe(**values)


def test_allocator_deltas_tolerate_counter_reset():
    window = TrainingSignalWindow()
    first = observe(window)
    assert first["allocator"] == {
        "alloc_retries_delta": 10,
        "alloc_count_delta": 20,
        "free_count_delta": 30,
    }
    second = observe(
        window,
        allocator_counters={
            "num_alloc_retries": 12,
            "num_device_alloc": 25,
            "num_device_free": 37,
        },
    )
    assert second["allocator"] == {
        "alloc_retries_delta": 2,
        "alloc_count_delta": 5,
        "free_count_delta": 7,
    }
    reset = observe(
        window,
        allocator_counters={
            "num_alloc_retries": 1,
            "num_device_alloc": 2,
            "num_device_free": 3,
        },
    )
    assert reset["allocator"] == {
        "alloc_retries_delta": 1,
        "alloc_count_delta": 2,
        "free_count_delta": 3,
    }


def test_per_shape_peaks_skip_warmup_and_track_independently():
    window = TrainingSignalWindow()
    observe(window, peak_allocated_bytes=100, peak_reserved_bytes=140)
    observe(window, peak_allocated_bytes=110, peak_reserved_bytes=150)
    observe(
        window,
        shape_key=(768, 768),
        peak_allocated_bytes=200,
        peak_reserved_bytes=260,
    )
    peaks = window.shape_peaks
    assert peaks[(512, 512)].steps == 1
    assert peaks[(512, 512)].peak_allocated_bytes == 110
    assert peaks[(512, 512)].peak_reserved_bytes == 150
    assert peaks[(768, 768)].steps == 0


def test_compile_activity_invalidates_shape_measurements():
    window = TrainingSignalWindow()
    observe(window)
    observe(window, peak_allocated_bytes=110)
    assert window.shape_peaks[(512, 512)].steps == 1

    invalid = observe(
        window,
        compile_counters={"frames": 4, "graphs": 2, "graph_breaks": 0},
    )
    assert invalid["compile_invalid"] is True
    assert window.shape_peaks == {}

    stable = observe(
        window,
        compile_counters={"frames": 4, "graphs": 2, "graph_breaks": 0},
    )
    assert stable["compile_invalid"] is False
    assert window.shape_peaks[(512, 512)].warmup_steps == 1


def test_transfer_metrics_require_settled_multi_step_window_and_handle_reset():
    window = TrainingSignalWindow(transfer_window_steps=3)
    assert observe(window)["transfer"] is None
    assert observe(window)["transfer"] is None
    settled = observe(
        window,
        transfer_counters={"h2d_ms": 12.0, "bytes": 96_000_000},
    )["transfer"]
    assert settled["steps"] == 3
    assert settled["h2d_duty_pct"] == pytest.approx(40.0)
    assert settled["achieved_gbps"] == pytest.approx(8.0)
    assert settled["h2d_duty_overflow"] is False

    observe(window)
    observe(window)
    after_reset = observe(
        window,
        transfer_counters={"h2d_ms": 6.0, "bytes": 42_000_000},
    )["transfer"]
    assert after_reset["h2d_duty_pct"] == pytest.approx(20.0)
    assert after_reset["achieved_gbps"] == pytest.approx(7.0)




def test_runtime_preserves_shape_peaks_after_layout_change():
    signals = TrainingSignalWindow()
    observe(signals)
    observe(signals, peak_allocated_bytes=110)
    assert signals.shape_peaks

    next_plan = object()
    executor = SimpleNamespace(
        transition_training_block=lambda key, resident: {
            "changed": True,
            "block_key": key,
            "resident": resident,
            "plan": next_plan,
        }
    )
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._executor = executor
    runtime._signals = signals
    runtime._training_plan = object()

    result = runtime.transition_training_block("blocks.3", resident=True)
    assert result["plan"] is next_plan
    assert runtime._training_plan is next_plan
    assert signals.shape_peaks


def test_worst_shape_allocator_slack_reconstructs_current_layout():
    signals = TrainingSignalWindow()
    observe(
        signals,
        shape_key=(512, 512),
        peak_allocated_bytes=700,
        resident_bytes=100,
        ring_bytes=50,
    )
    observe(
        signals,
        shape_key=(512, 512),
        peak_allocated_bytes=750,
        resident_bytes=100,
        ring_bytes=50,
    )
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._signals = signals
    runtime._residency = SimpleNamespace(resident_bytes=lambda: 120)
    runtime._smart_plan = {"singleton_resident_bytes": 30}
    runtime._training_ring_bytes = lambda: 50

    # working=600, current layout=150 resident + 50 ring => live=800.
    assert runtime._worst_shape_allocator_slack_bytes(1000) == 150


def test_aggressive_capacity_counts_exact_smallest_blocks_under_both_budgets():
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._policy = SimpleNamespace(slack_pad_bytes=5)
    runtime._promotion_candidates = lambda: tuple(
        {"block_key": f"blocks.{index}", "block_bytes": size}
        for index, size in enumerate((10, 20, 30, 40, 50))
    )
    runtime._worst_shape_allocator_slack_bytes = lambda _cap: 106
    runtime._worst_shape_candidate_margin_bytes = (
        lambda candidate: 120 - candidate["block_bytes"]
    )

    # Four blocks consume 100 bytes and leave the 5-byte allocator pad. The
    # fifth would exceed both the allocator and physical budgets.
    assert runtime._aggressive_promotion_capacity(1000) == 4


def test_training_cap_binding_uses_configured_phase_margin(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.runtime."
        "allocator_cap.apply_wddm_hard_allocator_cap",
        lambda device, hard, **kwargs: calls.append((device, hard, kwargs)),
    )
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._device = "cuda:1"
    runtime._config = SimpleNamespace(
        _policy=SimpleNamespace(wddm_hard_gib=1.25)
    )

    runtime._bind_training_cap()
    assert calls == [
        ("cuda:1", 1.25, {"log_prefix": "[ArenaOffload]"})
    ]
def test_shape_working_peak_excludes_residency():
    window = TrainingSignalWindow()
    observe(window, peak_allocated_bytes=900, resident_bytes=200)
    observe(window, peak_allocated_bytes=1000, resident_bytes=300)
    peak = window.shape_peaks[(512, 512)]
    assert peak.working_peak_bytes == 620


def test_signal_contains_memory_accounting():
    signal = observe(TrainingSignalWindow())
    assert signal["reclaimable_at_peak_bytes"] == 40
    assert signal["device_free_bytes"] == 500
    assert signal["resident_bytes"] == 200
    assert signal["ring_bytes"] == 80
    assert signal["live_bytes"] == 280
    json.dumps(TrainingSignalWindow().diagnostics())

def test_transfer_benefit_gate_requires_valid_nonzero_streaming():
    assert not transfer_benefits_from_residency(None)
    assert not transfer_benefits_from_residency({"bytes": 0, "h2d_ms": 20.0})
    assert not transfer_benefits_from_residency({"bytes": 100, "h2d_ms": 0.0})
    assert not transfer_benefits_from_residency(
        {"bytes": 100, "h2d_ms": 20.0, "h2d_duty_overflow": True}
    )
    assert transfer_benefits_from_residency(
        {"bytes": 100, "h2d_ms": 20.0, "h2d_duty_pct": 1.0}
    )


def test_controller_promotes_exact_candidate_then_rolls_it_back():
    controller = ArenaResidencyController(slack_pad_bytes=10)
    candidate = {"block_key": "blocks.3", "block_bytes": 20}
    clean = {
        "allocator": {"alloc_retries_delta": 0},
        "peak_allocated_bytes": 100,
        "resident_bytes": 200,
        "reclaimable_at_peak_bytes": 100,
        "compile_invalid": False,
        "transfer": {
            "bytes": 100,
            "h2d_ms": 20.0,
            "h2d_duty_pct": 80.0,
            "achieved_gbps": 10.0,
        },
    }
    actions = []
    for _ in range(4):
        decision = controller.step(
            clean,
            candidate=candidate,
            demote_candidate=None,
            cliff_cap_bytes=1000,
            worst_shape_free_bytes=100,
        )
        actions.append(decision)
    promoted = next(item for item in actions if item.action == "promote")
    assert promoted.block_key == "blocks.3"
    diagnostics = controller.diagnostics()
    assert diagnostics["last_block_key"] == "blocks.3"
    assert diagnostics["last_block_bytes"] == 20
    assert diagnostics["last_target_cap_bytes"] is None

    controller.step(
        clean,
        candidate=None,
        demote_candidate=None,
        cliff_cap_bytes=1000,
        worst_shape_free_bytes=0,
    )
    dirty = {
        **clean,
        "allocator": {
            "alloc_retries_delta": 0,
            "free_count_delta": 1,
        },
    }
    rollback = controller.step(
        dirty,
        candidate=None,
        demote_candidate=None,
        cliff_cap_bytes=1000,
        worst_shape_free_bytes=0,
    )
    assert rollback.action == "rollback"
    assert rollback.block_key == "blocks.3"
    assert controller.last_safe_residency_bytes == 200
    assert controller.last_rejected_residency_bytes == 220

    for _ in range(8):
        held = controller.step(
            clean,
            candidate=candidate,
            demote_candidate=None,
            cliff_cap_bytes=1000,
            worst_shape_free_bytes=100,
            worst_shape_allocator_slack_bytes=20,
        )
        assert held.action != "promote"
    assert held.reason == "allocator_headband"

    promoted_again = None
    for _ in range(4):
        promoted_again = controller.step(
            clean,
            candidate=candidate,
            demote_candidate=None,
            cliff_cap_bytes=1000,
            worst_shape_free_bytes=100,
            worst_shape_allocator_slack_bytes=31,
        )
        if promoted_again.action == "promote":
            break
    assert promoted_again.action == "promote"


def test_controller_cold_starts_one_whole_block_below():
    controller = ArenaResidencyController()
    decision = controller.step(
        None,
        candidate={"block_key": "blocks.4", "block_bytes": 20},
        demote_candidate={"block_key": "blocks.2", "block_bytes": 30},
        cliff_cap_bytes=1000,
        worst_shape_free_bytes=100,
    )
    assert decision.action == "demote"
    assert decision.block_key == "blocks.2"
    assert decision.reason == "approach_from_below"


def test_controller_promotes_each_step_with_four_block_headroom():
    controller = ArenaResidencyController(slack_pad_bytes=10)
    controller.bootstrapped = True
    clean = {
        "allocator": {
            "alloc_retries_delta": 0,
            "free_count_delta": 0,
        },
        "resident_bytes": 200,
        "compile_invalid": False,
        "transfer": None,
    }

    first = controller.step(
        clean,
        candidate={"block_key": "blocks.3", "block_bytes": 20},
        demote_candidate=None,
        cliff_cap_bytes=1000,
        current_cap_bytes=1000,
        worst_shape_free_bytes=100,
        worst_shape_allocator_slack_bytes=200,
        aggressive_promotion_capacity=4,
    )
    assert first.action == "promote"
    assert first.reason == "abundant_four_block_headroom"

    second = controller.step(
        {**clean, "resident_bytes": 220},
        candidate={"block_key": "blocks.4", "block_bytes": 20},
        demote_candidate=None,
        cliff_cap_bytes=1000,
        current_cap_bytes=1000,
        worst_shape_free_bytes=80,
        worst_shape_allocator_slack_bytes=180,
        aggressive_promotion_capacity=4,
    )
    assert second.action == "promote"
    assert second.block_key == "blocks.4"
    assert controller.pending_promotion["block_key"] == "blocks.4"


def test_controller_four_block_fast_lane_keeps_safety_vetoes():
    controller = ArenaResidencyController(slack_pad_bytes=10)
    controller.bootstrapped = True
    dirty = {
        "allocator": {
            "alloc_retries_delta": 1,
            "free_count_delta": 0,
        },
        "resident_bytes": 200,
        "compile_invalid": False,
        "transfer": None,
    }
    decision = controller.step(
        dirty,
        candidate={"block_key": "blocks.3", "block_bytes": 20},
        demote_candidate={"block_key": "blocks.1", "block_bytes": 20},
        cliff_cap_bytes=1000,
        current_cap_bytes=1000,
        worst_shape_free_bytes=100,
        worst_shape_allocator_slack_bytes=200,
        aggressive_promotion_capacity=4,
    )
    assert decision.action != "promote"
    assert controller.diagnostics()["last_aggressive_gate"] is False


def test_controller_does_not_bypass_bootstrap_verification():
    controller = ArenaResidencyController(slack_pad_bytes=10)
    controller.bootstrapped = True
    controller.begin_bootstrap_promotion(
        ("blocks.0", "blocks.1", "blocks.2", "blocks.3"),
        80,
        200,
        1000,
    )
    clean = {
        "allocator": {
            "alloc_retries_delta": 0,
            "free_count_delta": 0,
        },
        "resident_bytes": 280,
        "compile_invalid": False,
        "transfer": None,
    }
    decision = controller.step(
        clean,
        candidate={"block_key": "blocks.4", "block_bytes": 20},
        demote_candidate=None,
        cliff_cap_bytes=1000,
        current_cap_bytes=1000,
        worst_shape_free_bytes=100,
        worst_shape_allocator_slack_bytes=200,
        aggressive_promotion_capacity=4,
    )
    assert decision.action == "hold"
    assert controller.pending_promotion["block_keys"] == (
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
    )
    assert controller.diagnostics()["last_aggressive_gate"] is False


def test_controller_raises_cap_by_fixed_fsm_increment():
    controller = ArenaResidencyController(slack_pad_bytes=10)
    controller.bootstrapped = True
    controller.state = type(controller.state)("stable", 2)
    signal = {
        "allocator": {"alloc_retries_delta": 1},
        "peak_allocated_bytes": 400,
        "reclaimable_at_peak_bytes": 0,
        "compile_invalid": False,
        "transfer": None,
    }
    decision = controller.step(
        signal,
        candidate=None,
        demote_candidate={"block_key": "blocks.0", "block_bytes": 100},
        cliff_cap_bytes=1000,
        current_cap_bytes=700,
        worst_shape_free_bytes=100,
    )
    assert decision.action == "raise_cap"
    assert decision.target_cap_bytes == 710


def test_arena_sampling_binds_fp8_to_canonical_and_singletons(monkeypatch):
    model = SimpleNamespace()
    canonical = SimpleNamespace()
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._model = model
    runtime._device = "cuda"
    runtime._canonical_modules = (canonical,)
    runtime._config = SimpleNamespace(fp8_sampling=True)
    runtime._sampling_fp8_canonical = 0
    runtime._sampling_fp8_singletons = 0
    runtime._smart_plan = {"singleton_runtime_ids": {11, 22}}
    runtime._training_plan = object()
    runtime._executor = SimpleNamespace(
        TRAIN="train",
        activate=lambda *_args: None,
    )
    runtime._bind_training_cap = lambda: None
    calls = []

    def fake_enable(
        module, include_ids=None, live_ids=None, training=False, device=None
    ):
        ids = set(include_ids)
        calls.append(("enable", module, ids, set(live_ids), training, device))
        return [(None, None, None, None, value) for value in ids]

    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.runtime.enable_fp8",
        fake_enable,
    )
    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.runtime.disable_fp8",
        lambda restores: calls.append(("disable", list(restores))),
    )

    with runtime.sampling_session():
        assert runtime._sampling_fp8_canonical == 1
        assert runtime._sampling_fp8_singletons == 2

    expected_ids = {id(canonical), 11, 22}
    assert calls[0] == (
        "enable", model, expected_ids, expected_ids, False, "cuda"
    )
    assert calls[1][0] == "disable"
    assert {restore[4] for restore in calls[1][1]} == expected_ids


def test_arena_close_releases_all_owned_resources_after_executor_error():
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._resources = SimpleNamespace(
        release=lambda: (_ for _ in ()).throw(RuntimeError("executor boom"))
    )

    import pytest
    with pytest.raises(RuntimeError, match="executor boom"):
        runtime.close()

    assert not runtime._closed


def test_bf16_sampling_reserves_largest_singleton_dequant(monkeypatch):
    gib = 1024 ** 3
    dequant = 864 * 1024 ** 2
    captured = {}
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._device = "cpu"
    runtime._smart_plan = {
        "largest_singleton_bf16_dequant_bytes": dequant
    }
    runtime._config = SimpleNamespace(
        fp8_sampling=False,
        _policy=SimpleNamespace(
            sampling_working_reserve_gib="auto",
            sampling_wddm_hard_gib=1.0,
            sampling_wddm_margin_gib=1.0,
        ),
    )

    @contextlib.contextmanager
    def sampling(**kwargs):
        captured.update(kwargs)
        yield

    runtime._executor = SimpleNamespace(sampling=sampling)
    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.runtime."
        "allocator_cap.apply_wddm_hard_allocator_cap",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.runtime.resolve_margin_gib",
        lambda *_args, **_kwargs: 1.0,
    )

    with runtime.sampling_image(
        shape_key=(768, 768), cold_working_bytes=2 * gib
    ):
        pass

    assert captured["cold_floor_bytes"] == gib + dequant
    assert captured["hot_floor_bytes"] == int(1.25 * gib) + dequant


def test_bootstrap_uses_min_physical_free_and_one_gib_margin():
    gib = 1024 ** 3
    block_bytes = 200 * 1024 ** 2
    records = {
        f"blocks.{index}": SimpleNamespace(
            committed_bytes=block_bytes,
            leaf_names=("linear",),
        )
        for index in range(3)
    }
    plan = SimpleNamespace(resident_leaf_keys=frozenset())
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._bootstrap_complete = False
    runtime._bootstrap_min_free_bytes = 2 * gib + 450 * 1024 ** 2
    runtime._bootstrap_budget_bytes = 0
    runtime._bootstrap_block_keys = ()
    runtime._last_step_num = 50_000
    runtime._successful_training_steps = 1
    runtime._config = SimpleNamespace(
        _policy=SimpleNamespace(wddm_hard_gib=1.0)
    )
    runtime._model = SimpleNamespace()
    runtime._arena = SimpleNamespace(
        block_keys=lambda: tuple(records),
        block_record=lambda key: records[key],
    )
    runtime._residency = SimpleNamespace(
        plan=plan,
        resident_bytes=lambda: 0,
    )
    runtime._training_plan = plan
    runtime._smart_plan = {"singleton_resident_bytes": 100}
    runtime._policy = ArenaResidencyController()
    transitions = []
    runtime.transition_training_blocks = lambda keys, resident: (
        transitions.append((tuple(keys), resident))
        or {
            "changed": True,
            "block_keys": tuple(keys),
            "plan": object(),
        }
    )

    assert runtime._bootstrap_training_residency(10 * gib) is False
    assert runtime._bootstrap_complete is False

    runtime._successful_training_steps = 2
    assert runtime._bootstrap_training_residency(10 * gib) is True
    assert runtime._bootstrap_budget_bytes == 450 * 1024 ** 2
    assert transitions == [(("blocks.0", "blocks.1"), True)]
    assert runtime._policy.pending_promotion["block_keys"] == (
        "blocks.0",
        "blocks.1",
    )
    assert runtime._policy.pending_promotion["resident_bytes_before"] == 100


def test_bootstrap_ignores_first_runtime_sample_after_checkpoint_resume():
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._bootstrap_complete = False
    runtime._bootstrap_min_free_bytes = None
    runtime._successful_training_steps = 1

    runtime.record_training_physical_free_min(123)
    assert runtime._bootstrap_min_free_bytes is None

    runtime._successful_training_steps = 2
    runtime.record_training_physical_free_min(456)
    assert runtime._bootstrap_min_free_bytes == 456


def test_bootstrap_keeps_priority_over_four_block_fast_lane():
    gib = 1024 ** 3
    block_bytes = 100 * 1024 ** 2
    records = {
        f"blocks.{index}": SimpleNamespace(
            committed_bytes=block_bytes,
            leaf_names=("linear",),
        )
        for index in range(5)
    }
    plan = SimpleNamespace(resident_leaf_keys=frozenset())
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._bootstrap_complete = False
    runtime._bootstrap_min_free_bytes = 3 * gib
    runtime._bootstrap_budget_bytes = 0
    runtime._bootstrap_block_keys = ()
    runtime._successful_training_steps = 2
    runtime._config = SimpleNamespace(
        _policy=SimpleNamespace(wddm_hard_gib=1.0)
    )
    runtime._arena = SimpleNamespace(
        block_keys=lambda: tuple(records),
        block_record=lambda key: records[key],
    )
    runtime._residency = SimpleNamespace(
        plan=plan,
        resident_bytes=lambda: 0,
    )
    runtime._training_plan = plan
    runtime._smart_plan = {"singleton_resident_bytes": 0}
    runtime._policy = ArenaResidencyController()
    transitions = []
    runtime.transition_training_blocks = lambda keys, resident: (
        transitions.append((tuple(keys), resident))
        or {"changed": True, "block_keys": tuple(keys), "plan": object()}
    )

    assert runtime._bootstrap_training_residency(10 * gib) is True
    assert transitions == [
        (("blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4"), True)
    ]
    assert runtime._bootstrap_complete is True


def test_arena_allocation_failure_drains_and_rolls_back(monkeypatch):
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._device = "cpu"
    runtime._config = SimpleNamespace(
        _policy=SimpleNamespace(wddm_hard_gib=1.0)
    )
    runtime._last_training_cap_target_bytes = None
    runtime._signals = TrainingSignalWindow()
    runtime._policy = ArenaResidencyController()
    runtime._policy.pending_promotion = {
        "block_key": "blocks.7",
        "block_bytes": 20,
        "resident_bytes_before": 200,
        "resident_bytes_after": 220,
        "previous_cap_target_bytes": 1000,
    }
    transitions = []
    runtime.transition_training_blocks = (
        lambda keys, resident: transitions.append((tuple(keys), resident))
    )
    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.transfer.drain_fetch_runtime",
        lambda: 2,
    )
    cap_calls = []
    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.runtime."
        "allocator_cap.apply_wddm_hard_allocator_cap",
        lambda *args, **kwargs: cap_calls.append((args, kwargs)),
    )

    runtime._handle_training_failure(
        __import__("torch").cuda.OutOfMemoryError("synthetic allocator OOM"),
        shape_key=(768, 768),
        step_num=65,
    )

    assert transitions == [(("blocks.7",), False)]
    assert runtime._policy.last_safe_residency_bytes == 200
    assert runtime._policy.last_rejected_residency_bytes == 220
    assert runtime._last_failure_event["rollback_block"] == ["blocks.7"]
    assert runtime._last_failure_event["abandoned_fetch_tickets"] == 2
    assert cap_calls[0][1]["target_cap_bytes"] == 1000


def test_failed_training_step_preserves_original_error_if_cleanup_fails():
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._last_shape_key = None
    runtime._last_step_num = None
    runtime._device = "cpu"
    runtime._last_policy_error = None
    runtime._apply_training_policy = lambda: None
    runtime._handle_training_failure = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup failed")
        )
    )
    runtime._executor = SimpleNamespace(
        TRAIN="train",
        execution=lambda _mode: contextlib.nullcontext(),
    )

    with pytest.raises(ValueError, match="original"):
        with runtime.training_step(shape_key=(768, 768), step_num=2):
            raise ValueError("original")
    assert "cleanup failed" in runtime._last_policy_error


def test_failed_training_step_does_not_publish_partial_peak(monkeypatch):
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._last_shape_key = None
    runtime._last_step_num = None
    runtime._device = "cpu"
    runtime._apply_training_policy = lambda: None
    runtime._handle_training_failure = lambda *_args, **_kwargs: None
    runtime._executor = SimpleNamespace(
        TRAIN="train",
        execution=lambda _mode: contextlib.nullcontext(),
    )
    observed = []
    runtime._observe_training_step = lambda **kwargs: observed.append(kwargs)

    with pytest.raises(RuntimeError, match="synthetic OOM"):
        with runtime.training_step(shape_key=(768, 768), step_num=2):
            raise RuntimeError("synthetic OOM")

    assert observed == []
