"""Pure-CPU coverage for the two-timescale residency controller.

Validates the cap/allowance arithmetic against the measured Krea2 cap-descent
numbers (7.35 GiB floor, 6.77 GiB live, 0.21 GiB knee slack, ~0.375 GiB block)
and the hysteresis FSM's transitions. No CUDA -- these are the reason the
controller can be trusted without GPU CI (see the plan / cuda-testing policy).
"""

import pytest

from toolkit.memory_management import vram_budget as vb

GIB = vb.GIB
GC_THRESHOLD = 0.95


def gib(x):
    return int(x * GIB)


# --- Arithmetic vs the measured cap-descent -------------------------------

def test_allowance_matches_measured_knee():
    # 0.95*7.35 - 6.77 = +0.21 GiB (clean, the floor cap); 0.95*7.10 - 6.77 =
    # -0.02 GiB (dirty). The model validated to within one 0.25 notch.
    clean = vb.allocator_allowance_bytes(
        gib(7.35), gib(6.77), gc_threshold=GC_THRESHOLD
    )
    dirty = vb.allocator_allowance_bytes(
        gib(7.10), gib(6.77), gc_threshold=GC_THRESHOLD
    )
    assert clean == pytest.approx(gib(0.2125), abs=gib(0.01))
    assert dirty < 0


def test_cap_for_live_is_allowance_inverse():
    # To host 6.77 live + 0.21 cache budget the cap must be ~7.35 (the floor).
    cap = vb.cap_bytes_for_live(
        gib(6.77),
        gib(0.21),
        cliff_cap_bytes=gib(9.85),
        gc_threshold=GC_THRESHOLD,
    )
    assert cap == pytest.approx(gib(7.35), abs=gib(0.02))
    # Round-trips: that cap yields ~the requested cache budget back.
    assert vb.allocator_allowance_bytes(
        cap, gib(6.77), gc_threshold=GC_THRESHOLD
    ) == pytest.approx(gib(0.21), abs=gib(0.01))


def test_promotion_cap_growth_preserves_allocator_allowance():
    current = gib(6.5)
    promoted = gib(0.5)
    target = vb.cap_bytes_preserving_allowance_after_promotion(
        current, promoted, gib(9.5), gc_threshold=GC_THRESHOLD
    )

    before = vb.allocator_allowance_bytes(
        current, gib(5.5), gc_threshold=GC_THRESHOLD
    )
    after = vb.allocator_allowance_bytes(
        target, gib(6.0), gc_threshold=GC_THRESHOLD
    )
    assert target > current + promoted
    assert after >= before


def test_cap_for_live_clamped_to_cliff():
    cap = vb.cap_bytes_for_live(
        gib(11.0),
        gib(2.0),
        cliff_cap_bytes=gib(9.85),
        gc_threshold=GC_THRESHOLD,
    )
    assert cap == gib(9.85)


def test_promotion_precheck_uses_the_0p95_divisor():
    # need_cap = (6.77 + 0.375 + 0.21) / 0.95 = 7.742 GiB.
    live, block, slack = gib(6.77), gib(0.375), gib(0.21)
    # Sampling: cliff ~9.85 has room -> cap lever can fund the block.
    assert vb.cap_can_host_promotion(
        live, block, slack, gib(9.85), gc_threshold=GC_THRESHOLD
    ) is True
    # Training-like: cap pinned at the 7.35 floor/cliff -> must demote instead.
    assert vb.cap_can_host_promotion(
        live, block, slack, gib(7.35), gc_threshold=GC_THRESHOLD
    ) is False
    # The naive (no /0.95) test would wrongly pass at cliff = 7.36
    # (6.77+0.375+0.21 = 7.355 < 7.36); the real need_cap 7.742 rejects it.
    assert vb.cap_can_host_promotion(
        live, block, slack, gib(7.36), gc_threshold=GC_THRESHOLD
    ) is False


def test_promote_gate_needs_zero_retries_and_a_block_of_slack():
    block, slack = gib(0.375), gib(0.21)  # need > 0.585 GiB reclaimable
    assert vb.residency_promote_ok(0, gib(0.6), block, slack) is True
    assert vb.residency_promote_ok(0, gib(0.5), block, slack) is False   # not enough slack
    assert vb.residency_promote_ok(1, gib(0.6), block, slack) is False   # retries bind


# --- FSM transitions -------------------------------------------------------

def drive(state, signal, **kw):
    return vb.residency_fsm_step(state, signal, **kw)


CLEAN = {"binding": False}
BIND = {"binding": True}


def test_cold_settles_to_stable_after_k_clean():
    s = vb.ResidencyFsmState()  # COLD
    s, a = drive(s, CLEAN, k_clean=2)
    assert s.name == vb.FSM_COLD and a == vb.ACT_HOLD
    s, a = drive(s, CLEAN, k_clean=2)
    assert s.name == vb.FSM_STABLE


def test_cold_pressure_raises_cap_when_it_can_relieve():
    s = vb.ResidencyFsmState()
    s, a = drive(s, {"binding": True, "cap_can_relieve": True})
    assert s.name == vb.FSM_CAP_VERIFY and a == vb.ACT_RAISE_CAP


def test_cold_pressure_demotes_when_cap_is_pinned():
    s = vb.ResidencyFsmState()
    s, a = drive(s, {"binding": True, "cap_can_relieve": False})
    assert s.name == vb.FSM_COLD and a == vb.ACT_DEMOTE


def _stable():
    s = vb.ResidencyFsmState()
    for _ in range(2):
        s, _ = drive(s, CLEAN, k_clean=2)
    assert s.name == vb.FSM_STABLE
    return s


def test_stable_pressure_raises_cap_when_it_can_relieve():
    s = _stable()
    s, a = drive(s, {"binding": True, "cap_can_relieve": True})
    assert s.name == vb.FSM_CAP_VERIFY and a == vb.ACT_RAISE_CAP


def test_stable_pressure_demotes_when_cap_pinned():
    s = _stable()
    s, a = drive(s, {"binding": True, "cap_can_relieve": False})
    assert s.name == vb.FSM_COLD and a == vb.ACT_DEMOTE


def test_promotion_direct_when_cap_already_covers():
    s = _stable()
    # eligibility needs k_clean clean windows in STABLE first
    s, _ = drive(s, {"binding": False, "promote_gate": True, "cap_covers_promo": True}, k_clean=2)
    assert s.name == vb.FSM_STABLE  # w=1 < k_clean
    s, a = drive(s, {"binding": False, "promote_gate": True, "cap_covers_promo": True}, k_clean=2)
    assert s.name == vb.FSM_PROMOTION_VERIFY and a == vb.ACT_PROMOTE


def test_promotion_prefunds_cap_when_needed():
    s = _stable()
    sig = {"binding": False, "promote_gate": True, "cap_covers_promo": False}
    s, _ = drive(s, sig, k_clean=2)
    s, a = drive(s, sig, k_clean=2)
    assert s.name == vb.FSM_CAP_VERIFY and a == vb.ACT_RAISE_CAP


def test_cap_verify_clean_returns_to_stable():
    s = vb.ResidencyFsmState(vb.FSM_CAP_VERIFY, 0)
    s, _ = drive(s, CLEAN, k_verify=2)
    assert s.name == vb.FSM_CAP_VERIFY
    s, _ = drive(s, CLEAN, k_verify=2)
    assert s.name == vb.FSM_STABLE


def test_cap_verify_pressure_escalates_to_demote():
    s = vb.ResidencyFsmState(vb.FSM_CAP_VERIFY, 0)
    s, a = drive(s, BIND)
    assert s.name == vb.FSM_COLD and a == vb.ACT_DEMOTE


def test_promotion_verify_rolls_back_binding_first_window():
    s = vb.ResidencyFsmState(vb.FSM_PROMOTION_VERIFY, 0)
    s, a = drive(s, BIND, k_verify=2)
    assert s.name == vb.FSM_COOLDOWN and a == vb.ACT_ROLLBACK


def test_promotion_verify_clean_commits():
    s = vb.ResidencyFsmState(vb.FSM_PROMOTION_VERIFY, 0)
    for _ in range(3):  # first window ignored, then k_verify clean
        s, _ = drive(s, CLEAN, k_verify=2)
    assert s.name == vb.FSM_STABLE


def test_cooldown_bars_repromotion_then_releases():
    s = vb.ResidencyFsmState(vb.FSM_COOLDOWN, 0)
    for _ in range(3):
        s, _ = drive(s, {"binding": False, "promote_gate": True, "cap_covers_promo": True}, cooldown_n=4)
        assert s.name == vb.FSM_COOLDOWN  # still barred, no promote
    s, _ = drive(s, CLEAN, cooldown_n=4)
    assert s.name == vb.FSM_STABLE


def test_measurements_invalid_forces_cold():
    s = _stable()
    s, a = drive(s, {"measurements_invalid": True, "binding": False})
    assert s.name == vb.FSM_COLD and a == vb.ACT_HOLD


def test_no_oscillation_rollback_then_cooldown_holds():
    # A failed promotion must not immediately re-promote: rollback -> cooldown
    # bars promotion for cooldown_n windows even if the gate keeps firing.
    s = vb.ResidencyFsmState(vb.FSM_PROMOTION_VERIFY, 1)  # past the cold window
    s, a = drive(s, BIND)
    assert (s.name, a) == (vb.FSM_COOLDOWN, vb.ACT_ROLLBACK)
    promotes = 0
    for _ in range(3):
        s, a = drive(s, {"binding": False, "promote_gate": True, "cap_covers_promo": True}, cooldown_n=4)
        promotes += a == vb.ACT_PROMOTE
    assert promotes == 0


# --- Training worst-resolution promotion guard -----------------------------

def test_worst_shape_free_subtracts_all_cohabitants_plus_block():
    # resident 5.0 + block 0.375 + ring 0.5 + worst reserve 2.5 + other 1.15
    # = 9.525 used; on a 12 GiB card that leaves 2.475 free.
    free = vb.training_promotion_worst_shape_free_gib(
        resident_gib=5.0,
        added_block_gib=0.375,
        ring_gib=0.5,
        worst_working_reserve_gib=2.5,
        other_gib=1.15,
        total_gib=12.0,
    )
    assert free == pytest.approx(2.475, abs=1e-6)


def test_worst_shape_gate_vetoes_when_high_res_reserve_would_page():
    # Same layout, but the worst measured resolution needs a 3.6 GiB reserve.
    # A promotion decided on a roomy low-res step (that only needed ~1.7) would
    # push the high-res cohabitation peak past the promote floor -> veto.
    hold_high = 2.0
    roomy_lowres = vb.training_promotion_worst_shape_free_gib(
        resident_gib=5.0, added_block_gib=0.375, ring_gib=0.5,
        worst_working_reserve_gib=1.7, other_gib=1.15, total_gib=12.0,
    )
    worst_highres = vb.training_promotion_worst_shape_free_gib(
        resident_gib=5.0, added_block_gib=0.375, ring_gib=0.5,
        worst_working_reserve_gib=3.6, other_gib=1.15, total_gib=12.0,
    )
    # The current (low-res) step looks safe; the worst measured shape does not.
    assert roomy_lowres >= hold_high
    assert worst_highres < hold_high


def test_worst_shape_free_is_conservative_about_the_block():
    # Adding the block can only lower the predicted free (never hidden by a
    # shrinking ring) -- the from-below assumption.
    without_block = vb.training_promotion_worst_shape_free_gib(
        resident_gib=5.0, added_block_gib=0.0, ring_gib=0.5,
        worst_working_reserve_gib=2.5, other_gib=1.15, total_gib=12.0,
    )
    with_block = vb.training_promotion_worst_shape_free_gib(
        resident_gib=5.0, added_block_gib=0.375, ring_gib=0.5,
        worst_working_reserve_gib=2.5, other_gib=1.15, total_gib=12.0,
    )
    assert with_block == pytest.approx(without_block - 0.375, abs=1e-6)
