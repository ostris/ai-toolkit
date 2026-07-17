import pytest

from toolkit.memory_management import vram_budget


def test_runtime_allocator_settings_prefers_torch_runtime(monkeypatch):
    monkeypatch.setenv(
        "PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.61"
    )
    monkeypatch.setattr(
        vram_budget.torch._C,
        "_accelerator_getAllocatorSettings",
        lambda: "garbage_collection_threshold:0.79",
    )

    assert vram_budget._runtime_allocator_settings() == (
        "garbage_collection_threshold:0.79"
    )


def test_runtime_allocator_settings_falls_back_for_older_torch(monkeypatch):
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.67"
    )
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setattr(
        vram_budget.torch._C,
        "_accelerator_getAllocatorSettings",
        None,
    )

    assert vram_budget._runtime_allocator_settings() == (
        "garbage_collection_threshold:0.67"
    )


def test_allocator_gc_threshold_reads_live_torch_settings(monkeypatch):
    monkeypatch.setattr(
        vram_budget,
        "_runtime_allocator_settings",
        lambda: (
            "expandable_segments:True,"
            "garbage_collection_threshold:0.73"
        ),
    )

    assert vram_budget.allocator_gc_threshold() == pytest.approx(0.73)
    assert vram_budget.allocator_allowance_bytes(1_000, 600) == 130
    assert vram_budget.cap_bytes_for_live(600, 130, 2_000) == 1_000
    assert vram_budget.sampling_allocator_budget_free_bytes(
        1_000, 600, 1.0, 0
    ) == 130


@pytest.mark.parametrize("settings", ["", "backend:cudaMallocAsync"])
def test_allocator_without_native_gc_uses_cap_as_effective_target(
    monkeypatch, settings
):
    monkeypatch.setattr(
        vram_budget, "_runtime_allocator_settings", lambda: settings
    )

    assert vram_budget.allocator_gc_threshold() == 1.0
    assert vram_budget.allocator_allowance_bytes(1_000, 600) == 400


def test_invalid_fallback_allocator_threshold_fails_closed(monkeypatch):
    monkeypatch.setattr(
        vram_budget,
        "_runtime_allocator_settings",
        lambda: "garbage_collection_threshold:not-a-number",
    )

    with pytest.raises(ValueError, match="invalid runtime"):
        vram_budget.allocator_gc_threshold()
