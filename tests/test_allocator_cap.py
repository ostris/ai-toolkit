import pytest
from unittest import mock

from toolkit.memory_management import allocator_cap


@pytest.fixture(autouse=True)
def _clear_applied_fractions():
    allocator_cap.APPLIED_FRACTIONS.clear()
    allocator_cap.RELIEF_BYTES.clear()
    yield
    allocator_cap.APPLIED_FRACTIONS.clear()
    allocator_cap.RELIEF_BYTES.clear()


def test_production_guard_binds_the_same_allocator_cap(monkeypatch):
    calls = []
    gib = 1024**3
    monkeypatch.setattr(allocator_cap.sys, "platform", "win32")
    monkeypatch.setattr(allocator_cap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(allocator_cap.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(allocator_cap.torch.cuda, "memory_reserved", lambda _i: 0)
    monkeypatch.setattr(
        allocator_cap.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, index: calls.append((fraction, index)),
    )
    monkeypatch.setattr(
        allocator_cap.vram_budget, "device_total_bytes", lambda _i: 12 * gib
    )
    monkeypatch.setattr(
        allocator_cap.vram_budget,
        "real_device_total_bytes",
        lambda _i: 12 * gib,
    )
    monkeypatch.setattr(
        allocator_cap.vram_budget,
        "device_mem_info",
        lambda _i: (12 * gib, 12 * gib),
    )

    result = allocator_cap.configure_wddm_allocator_guard(
        "cuda", 1.0, strict=False
    )

    assert result == 11 / 12
    assert calls == [(11 / 12, 0)]
    assert allocator_cap.APPLIED_FRACTIONS[0] == 11 / 12


def test_strict_development_guard_binds_allocator_cap(monkeypatch):
    calls = []
    gib = 1024**3
    monkeypatch.setattr(allocator_cap.sys, "platform", "win32")
    monkeypatch.setattr(allocator_cap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(allocator_cap.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(allocator_cap.torch.cuda, "memory_reserved", lambda _i: 0)
    monkeypatch.setattr(
        allocator_cap.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, index: calls.append((fraction, index)),
    )
    monkeypatch.setattr(
        allocator_cap.vram_budget, "device_total_bytes", lambda _i: 12 * gib
    )
    monkeypatch.setattr(
        allocator_cap.vram_budget,
        "real_device_total_bytes",
        lambda _i: 12 * gib,
    )
    monkeypatch.setattr(
        allocator_cap.vram_budget,
        "device_mem_info",
        lambda _i: (12 * gib, 12 * gib),
    )

    result = allocator_cap.configure_wddm_allocator_guard(
        "cuda", 1.0, strict=True
    )

    assert result == 11 / 12
    assert calls == [(11 / 12, 0)]
    assert allocator_cap.APPLIED_FRACTIONS[0] == 11 / 12


def test_only_non_strict_failure_policy_can_relax_a_rejected_cap(monkeypatch):
    calls = []
    gib = 1024**3
    allocator_cap.APPLIED_FRACTIONS[0] = 10 / 12
    monkeypatch.setattr(allocator_cap.sys, "platform", "win32")
    monkeypatch.setattr(allocator_cap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(allocator_cap.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        allocator_cap.vram_budget,
        "real_device_total_bytes",
        lambda _i: 12 * gib,
    )
    monkeypatch.setattr(
        allocator_cap.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, index: calls.append((fraction, index)),
    )

    assert allocator_cap.relieve_wddm_allocator_guard_after_oom(
        "cuda", strict=True
    ) is False
    assert allocator_cap.relieve_wddm_allocator_guard_after_oom(
        "cuda", strict=False
    ) is True

    assert calls == [((10.5 / 12), 0)]
    assert allocator_cap.RELIEF_BYTES[0] == allocator_cap.CAP_RELIEF_BYTES


def test_restore_reinstalls_previous_toolkit_fraction(monkeypatch):
    calls = []
    monkeypatch.setattr(allocator_cap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(allocator_cap.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        allocator_cap.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, index: calls.append((fraction, index)),
    )
    allocator_cap.APPLIED_FRACTIONS[0] = 0.75

    allocator_cap.restore_tracked_allocator_fraction("cuda", 0.5)

    assert calls == [(0.5, 0)]
    assert allocator_cap.APPLIED_FRACTIONS[0] == 0.5


def test_restore_removes_arena_fraction_only_after_cuda_succeeds(monkeypatch):
    monkeypatch.setattr(allocator_cap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(allocator_cap.torch.cuda, "current_device", lambda: 0)
    setter = mock.Mock(side_effect=(RuntimeError("restore failed"), None))
    monkeypatch.setattr(
        allocator_cap.torch.cuda,
        "set_per_process_memory_fraction",
        setter,
    )
    allocator_cap.APPLIED_FRACTIONS[0] = 0.5

    with pytest.raises(RuntimeError, match="restore failed"):
        allocator_cap.restore_tracked_allocator_fraction("cuda", None)
    assert allocator_cap.APPLIED_FRACTIONS[0] == 0.5

    allocator_cap.restore_tracked_allocator_fraction("cuda", None)
    assert setter.call_args_list == [mock.call(1.0, 0), mock.call(1.0, 0)]
    assert 0 not in allocator_cap.APPLIED_FRACTIONS
