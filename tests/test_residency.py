from types import SimpleNamespace

import pytest
import torch

from toolkit.memory_management import pin_manager
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.immutable_runtime import ImmutableTransformerRuntime
from toolkit.memory_management.residency import (
    ResidencyError,
    ResidencyPlan,
    ResidencyState,
)




def _linear(seed=0, *, device="cpu", dtype=torch.float32):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(8, 8, bias=True, device=device, dtype=dtype)
    layer.requires_grad_(False)
    return layer


@pytest.fixture
def arena_layers():
    layers = {"a": _linear(1), "b": _linear(2), "c": _linear(3)}
    arena = CanonicalArena()
    arena.canonicalize({"blocks.0": list(layers.items())})
    try:
        yield arena, layers
    finally:
        arena.release()
def test_phase_plan_and_existing_planner_seam(arena_layers):
    arena, layers = arena_layers
    smart = {"offload_ids": {id(layers["a"]), id(layers["c"])}}
    plan = ResidencyPlan.from_smart_plan(arena, smart, phase="train")
    # Immutable policy normalizes any partially offloaded canonical block to
    # fully streamed. Source snapshots may still represent mixed layouts, but
    # controller/planner output is whole-block.
    assert plan.resident_leaf_keys == frozenset()
    assert plan == ResidencyPlan.from_smart_plan(arena, smart, phase="train")
    assert plan.fingerprint != ResidencyPlan.build("sample", plan.resident_leaf_keys).fingerprint


def test_runtime_training_transitions_are_whole_block(arena_layers):
    arena, layers = arena_layers
    block = SimpleNamespace(entries=tuple(layers.items()))
    model = SimpleNamespace(blocks=(block,))
    state = ResidencyState(arena, "cpu")
    state.reconcile(ResidencyPlan.build("train", ()))
    runtime = ImmutableTransformerRuntime(
        model,
        state,
        blocks=model.blocks,
        block_keys=("blocks.0",),
        entries_by_block={"blocks.0": tuple(layers.items())},
        compile_blocks=False,
    )

    growth = runtime.increase_training_residency(
        arena.block_record("blocks.0").committed_bytes,
    )
    expected = frozenset(
        ("blocks.0", leaf_name) for leaf_name in layers
    )
    assert state.plan.resident_leaf_keys == expected
    assert growth["added_blocks"] == ("blocks.0",)

    relief = runtime.reduce_training_residency(1)
    assert relief["removed_blocks"] == ("blocks.0",)
    assert state.plan.resident_leaf_keys == frozenset()

def test_exact_training_block_transaction_uses_stable_key(arena_layers):
    arena, layers = arena_layers
    block = SimpleNamespace(entries=tuple(layers.items()))
    model = SimpleNamespace(blocks=(block,))
    state = ResidencyState(arena, "cpu")
    state.reconcile(ResidencyPlan.build("train", ()))
    runtime = ImmutableTransformerRuntime(
        model,
        state,
        blocks=model.blocks,
        block_keys=("blocks.0",),
        entries_by_block={"blocks.0": tuple(layers.items())},
        compile_blocks=False,
    )

    promoted = runtime.transition_training_block("blocks.0", resident=True)
    expected = frozenset(("blocks.0", name) for name in layers)
    assert promoted["changed"] is True
    assert state.plan.resident_leaf_keys == expected

    unchanged = runtime.transition_training_block("blocks.0", resident=True)
    assert unchanged["changed"] is False

    demoted = runtime.transition_training_block("blocks.0", resident=False)
    assert demoted["changed"] is True
    assert state.plan.resident_leaf_keys == frozenset()


def test_cpu_reconcile_never_mutates_parameters_or_pin_ledger(arena_layers):
    arena, layers = arena_layers
    state = ResidencyState(arena, "cpu")
    identities = {name: id(layer.weight) for name, layer in layers.items()}
    pointers = {
        name: layer.weight.untyped_storage().data_ptr() for name, layer in layers.items()
    }
    pins = (pin_manager.total_pinned_bytes(), pin_manager.pinned_bytes_by_kind())

    delta = state.reconcile(
        ResidencyPlan.build("train", (("blocks.0", "a"), ("blocks.0", "c")))
    )
    assert delta.promoted == (("blocks.0", "a"), ("blocks.0", "c"))
    assert state.streamed_leaf_names("blocks.0") == ("b",)
    state.reconcile(ResidencyPlan.build("sample", (("blocks.0", "b"),)))
    state.clear()

    assert pins == (pin_manager.total_pinned_bytes(), pin_manager.pinned_bytes_by_kind())
    assert identities == {name: id(layer.weight) for name, layer in layers.items()}
    assert pointers == {
        name: layer.weight.untyped_storage().data_ptr() for name, layer in layers.items()
    }


def test_failed_multi_promotion_rolls_back_atomically(arena_layers, monkeypatch):
    arena, _layers = arena_layers
    state = ResidencyState(arena, "cpu")
    state.promote(("blocks.0", "c"), phase="seed")
    before_plan = state.plan
    before_sidecar = state.resident_leaf(("blocks.0", "c"))
    before_pins = (pin_manager.total_pinned_bytes(), pin_manager.pinned_bytes_by_kind())
    original = state._build_sidecar
    calls = 0

    def fail_second(key):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic promotion failure")
        return original(key)

    monkeypatch.setattr(state, "_build_sidecar", fail_second)
    target = ResidencyPlan.build(
        "train", (("blocks.0", "a"), ("blocks.0", "b"), ("blocks.0", "c"))
    )
    with pytest.raises(RuntimeError, match="synthetic promotion failure"):
        state.reconcile(target)

    assert state.plan is before_plan
    assert state.resident_leaf(("blocks.0", "c")) is before_sidecar
    assert state.resident_leaf(("blocks.0", "a")) is None
    assert state.resident_leaf(("blocks.0", "b")) is None
    assert before_pins == (pin_manager.total_pinned_bytes(), pin_manager.pinned_bytes_by_kind())


def test_unknown_leaf_fails_before_state_change(arena_layers):
    arena, _layers = arena_layers
    state = ResidencyState(arena, "cpu")
    with pytest.raises(ResidencyError, match="unknown_residency_leaf"):
        state.reconcile(ResidencyPlan.build("train", (("blocks.0", "missing"),)))
    assert state.resident_bytes() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_sidecars_match_canonical_values_and_demote_without_writeback(arena_layers):
    arena, layers = arena_layers
    state = ResidencyState(arena, "cuda")
    host_before = {name: layer.weight.detach().clone() for name, layer in layers.items()}
    state.reconcile(
        ResidencyPlan.build("train", (("blocks.0", "a"), ("blocks.0", "c")))
    )
    for name in ("a", "c"):
        sidecar = state.resident_leaf(("blocks.0", name))
        assert sidecar.weight.device.type == "cuda"
        torch.testing.assert_close(sidecar.weight.cpu(), host_before[name])
    state.demote(("blocks.0", "a"))
    torch.cuda.synchronize()
    # Frozen demotion drops the device copy and performs no D2H write-back.
    torch.testing.assert_close(layers["a"].weight, host_before["a"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_sidecar_preserves_wrapper_type_and_values():
    from optimum.quanto import freeze

    from toolkit.util.quantize import get_qtype, quantize

    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16))
    quantize(model, weights=get_qtype("qfloat8"))
    freeze(model)
    layer = model[0]
    layer.weight.requires_grad_(False)
    expected_type = type(layer.weight.data)
    expected = layer.weight.data.dequantize().clone()
    arena = CanonicalArena()
    arena.canonicalize({"blocks.0": [("q", layer)]})
    try:
        state = ResidencyState(arena, "cuda")
        state.promote(("blocks.0", "q"), phase="train")
        weight = state.resident_tensor(("blocks.0", "q"))
        assert type(weight) is expected_type
        assert all(leaf.device.type == "cuda" for leaf in weight.__tensor_flatten__()[0]
                   for leaf in [getattr(weight, leaf)])
        torch.testing.assert_close(weight.dequantize().cpu(), expected)
    finally:
        arena.release()
