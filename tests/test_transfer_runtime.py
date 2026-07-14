import pytest
import torch
import torch.nn.functional as F

from toolkit.memory_management.arena_offload import transfer as ingraph_stream
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.transfer_plan import build_transfer_plan

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _linear(seed):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(8, 8, bias=True)
    layer.requires_grad_(False)
    return layer


@pytest.fixture
def arena_block():
    layers = {name: _linear(seed) for seed, name in enumerate(("a", "b", "c"), 1)}
    arena = CanonicalArena()
    arena.canonicalize({"blocks.0": list(layers.items())})
    ingraph_stream.configure_fetch_runtime(depth=2)
    ingraph_stream.fetch_stats(reset=True)
    try:
        yield arena.block_record("blocks.0"), layers
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()


def _fetch(record, plan):
    token = torch.ops.mm.fetch_start_multi(
        record.host_flat, plan.ranges_tensor(), plan.compact_nbytes
    )
    flat = torch.ops.mm.fetch_wait(token, plan.compact_nbytes)
    return token, flat


def test_mixed_resident_streamed_numerical_parity_and_exact_stats(arena_block):
    record, layers = arena_block
    plan = build_transfer_plan(record, ("a", "c"))
    token, flat = _fetch(record, plan)
    x = torch.randn(4, 8, device="cuda")

    streamed = 0
    for name in ("a", "c"):
        weight = plan.compact_leaf_view(flat, name, "weight")
        bias = plan.compact_leaf_view(flat, name, "bias")
        streamed = streamed + F.linear(x, weight, bias)
    resident = F.linear(
        x, layers["b"].weight.to("cuda"), layers["b"].bias.to("cuda")
    )
    actual = streamed + resident
    expected = sum(F.linear(x, layer.weight.to("cuda"), layer.bias.to("cuda"))
                   for layer in layers.values())
    torch.ops.mm.fetch_free_after(token, actual)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected)
    stats = ingraph_stream.fetch_stats()
    assert stats["fetches"] == 1
    assert stats["bytes"] == plan.compact_nbytes
    assert stats["copies"] == plan.num_ranges


def test_fully_streamed_plan_uses_single_copy_fast_path(arena_block):
    record, _layers = arena_block
    plan = build_transfer_plan(record, ("a", "b", "c"))
    assert plan.fully_streamed and plan.num_ranges == 1
    token, flat = _fetch(record, plan)
    span = plan.ranges[0]
    expected = record.host_flat[span.src_offset:span.src_offset + span.nbytes].to("cuda")
    torch.testing.assert_close(flat, expected)
    torch.ops.mm.fetch_free(token)
    torch.cuda.synchronize()
    assert ingraph_stream.fetch_stats()["copies"] == 1


def test_depth_two_reuse_hammer(arena_block):
    record, _layers = arena_block
    plan = build_transfer_plan(record, ("a", "c"))
    expected = torch.empty(plan.compact_nbytes, dtype=torch.uint8)
    for span in plan.ranges:
        expected[span.dst_offset:span.dst_offset + span.nbytes].copy_(
            record.host_flat[span.src_offset:span.src_offset + span.nbytes]
        )
    expected = expected.to("cuda")

    for _ in range(50):
        first_token, first = _fetch(record, plan)
        second_token, second = _fetch(record, plan)
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, expected)
        torch.ops.mm.fetch_free(first_token)
        torch.ops.mm.fetch_free(second_token)
    torch.cuda.synchronize()
    stats = ingraph_stream.fetch_stats()
    assert stats["fetches"] == 100
    assert stats["copies"] == 100 * plan.num_ranges


@pytest.mark.parametrize(
    "ranges,compact_nbytes,match",
    [
        (torch.tensor([[-1, 0, 4]], dtype=torch.int64), 4, "invalid range"),
        (torch.tensor([[0, 0, 10**9]], dtype=torch.int64), 10**9, "out of bounds"),
        (torch.tensor([[0, 1, 4]], dtype=torch.int64), 5, "not compact"),
        (torch.tensor([[0, 0, 4], [2, 4, 4]], dtype=torch.int64), 8, "overlaps"),
    ],
)
def test_invalid_ranges_fail_closed(arena_block, ranges, compact_nbytes, match):
    record, _layers = arena_block
    with pytest.raises(RuntimeError, match=match):
        torch.ops.mm.fetch_start_multi(record.host_flat, ranges, compact_nbytes)


def test_pageable_non_arena_source_and_unknown_ticket_fail_closed():
    host = torch.empty(64, dtype=torch.uint8)
    ranges = torch.tensor([[0, 0, 64]], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="registered canonical arena"):
        torch.ops.mm.fetch_start_multi(host, ranges, 64)
    with pytest.raises(RuntimeError, match="unknown ticket"):
        torch.ops.mm.fetch_wait(torch.tensor([987654], dtype=torch.int64), 64)


def test_same_plan_new_storage_has_zero_recompiles(arena_block):
    record, _layers = arena_block
    plan = build_transfer_plan(record, ("a", "c"))
    ranges = plan.ranges_tensor()

    other_layers = {name: _linear(seed) for seed, name in enumerate(("a", "b", "c"), 11)}
    other_arena = CanonicalArena()
    other_arena.canonicalize({"blocks.0": list(other_layers.items())})
    other_record = other_arena.block_record("blocks.0")
    other_plan = build_transfer_plan(other_record, ("a", "c"))
    assert other_plan.ranges == plan.ranges

    def consume(host, static_ranges, guard):
        token = torch.ops.mm.fetch_start_multi_after(
            host, static_ranges, plan.compact_nbytes, guard
        )
        flat = torch.ops.mm.fetch_wait(token, plan.compact_nbytes)
        out = flat.float().sum() + guard
        torch.ops.mm.fetch_free_after(token, out)
        return out

    torch._dynamo.reset()
    compiled = torch.compile(consume, fullgraph=True, dynamic=False)
    guard = torch.ones((), device="cuda")
    try:
        compiled(record.host_flat, ranges, guard)
        torch.cuda.synchronize()
        graphs = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
        assert graphs >= 1
        compiled(other_record.host_flat, other_plan.ranges_tensor(), guard)
        torch.cuda.synchronize()
        assert torch._dynamo.utils.counters["stats"].get("unique_graphs", 0) == graphs
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    finally:
        ingraph_stream.drain_fetch_runtime()
        other_arena.release()
