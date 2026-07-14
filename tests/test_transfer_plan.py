import unittest
from itertools import pairwise

import torch
import torch.nn as nn

from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.transfer_plan import (
    TransferPlanError,
    build_transfer_plan,
)


def _linear(in_f=8, out_f=4, bias=True):
    layer = nn.Linear(in_f, out_f, bias=bias)
    layer.weight.requires_grad_(False)
    if bias:
        layer.bias.requires_grad_(False)
    return layer


@unittest.skipUnless(torch.cuda.is_available(), "canonical arena pinning requires CUDA")
class BuildTransferPlanTests(unittest.TestCase):
    def setUp(self):
        self.arena = CanonicalArena()
        self.a = _linear()
        self.b = _linear()
        self.c = _linear()
        self.arena.canonicalize(
            {"blocks.0": [("a", self.a), ("b", self.b), ("c", self.c)]}
        )
        self.addCleanup(self.arena.release)

    def test_fully_streamed_block_is_one_coalesced_range(self):
        record = self.arena.block_record("blocks.0")
        plan = build_transfer_plan(record, ["a", "b", "c"])
        self.assertTrue(plan.fully_streamed)
        self.assertEqual(plan.num_ranges, 1)
        self.assertEqual(plan.ranges[0].nbytes, plan.compact_nbytes)
        # Weight+bias for all 3 leaves must be present in the compact layout.
        for name in ("a", "b", "c"):
            self.assertIn("weight", plan.leaf_specs[name])
            self.assertIn("bias", plan.leaf_specs[name])

    def test_partial_streaming_breaks_coalescing_at_resident_leaf(self):
        record = self.arena.block_record("blocks.0")
        plan = build_transfer_plan(record, ["a", "c"])  # b resident, breaks the run
        self.assertFalse(plan.fully_streamed)
        self.assertEqual(plan.num_ranges, 2)
        self.assertEqual(set(plan.leaf_specs.keys()), {"a", "c"})

    def test_compact_offsets_are_exact_and_non_overlapping(self):
        record = self.arena.block_record("blocks.0")
        plan = build_transfer_plan(record, ["a", "b", "c"])
        spans = []
        for name, roles in plan.leaf_specs.items():
            for role, spec in roles.items():
                spans.append((spec.dst_offset, spec.dst_offset + spec.nbytes, name, role))
        spans.sort()
        for prev, nxt in pairwise(spans):
            self.assertLessEqual(prev[1], nxt[0])
        self.assertLessEqual(spans[-1][1], plan.compact_nbytes)

    def test_range_bytes_equal_compact_total_and_cover_every_leaf(self):
        record = self.arena.block_record("blocks.0")
        plan = build_transfer_plan(record, ["a", "b"])
        # Coalescing a small alignment gap (< LEAF_ALIGN) into one range is
        # allowed to copy a few extra padding bytes -- ranges always sum
        # exactly to compact_nbytes, and every real leaf byte must fall
        # within that budget (no leaf is ever short-changed or truncated).
        self.assertEqual(sum(r.nbytes for r in plan.ranges), plan.compact_nbytes)
        leaf_bytes = sum(
            spec.nbytes
            for name in ("a", "b")
            for spec in plan.leaf_specs[name].values()
        )
        self.assertLessEqual(leaf_bytes, plan.compact_nbytes)
        # Padding slack introduced by coalescing must stay under one
        # alignment quantum per streamed leaf sub-tensor (weight/bias each
        # contribute at most one internal gap).
        n_sub_tensors = sum(len(roles) for roles in plan.leaf_specs.values())
        self.assertLess(plan.compact_nbytes - leaf_bytes, 256 * n_sub_tensors)

    def test_fingerprint_stable_across_rebuilds_same_selection(self):
        record = self.arena.block_record("blocks.0")
        p1 = build_transfer_plan(record, ["a", "b"])
        p2 = build_transfer_plan(record, ["b", "a"])  # order-independent input
        self.assertEqual(p1.fingerprint, p2.fingerprint)

    def test_fingerprint_differs_for_different_selection(self):
        record = self.arena.block_record("blocks.0")
        p1 = build_transfer_plan(record, ["a", "b"])
        p2 = build_transfer_plan(record, ["a", "c"])
        self.assertNotEqual(p1.fingerprint, p2.fingerprint)

    def test_empty_streamed_set_fails_closed(self):
        record = self.arena.block_record("blocks.0")
        with self.assertRaises(TransferPlanError):
            build_transfer_plan(record, [])

    def test_ranges_tensor_shape_and_dtype(self):
        record = self.arena.block_record("blocks.0")
        plan = build_transfer_plan(record, ["a", "c"])
        t = plan.ranges_tensor()
        self.assertEqual(t.dtype, torch.int64)
        self.assertEqual(tuple(t.shape), (plan.num_ranges, 3))
        self.assertEqual(t.device.type, "cpu")

    def test_compact_leaf_view_reads_correct_bytes(self):
        record = self.arena.block_record("blocks.0")
        plan = build_transfer_plan(record, ["a", "b", "c"])
        # Simulate the compact device buffer directly on the (CPU-readable)
        # host flat's own bytes to check the view math without a real H2D
        # copy: build a compact CPU buffer by literally performing the
        # plan's ranges as CPU-to-CPU copies, then verify per-leaf values.
        compact = torch.empty(plan.compact_nbytes, dtype=torch.uint8)
        flat = record.host_flat
        for r in plan.ranges:
            compact[r.dst_offset:r.dst_offset + r.nbytes].copy_(
                flat[r.src_offset:r.src_offset + r.nbytes]
            )
        for name, layer in (("a", self.a), ("b", self.b), ("c", self.c)):
            view = plan.compact_leaf_view(compact, name, "weight")
            torch.testing.assert_close(view, layer.weight.data)
            view_b = plan.compact_leaf_view(compact, name, "bias")
            torch.testing.assert_close(view_b, layer.bias.data)



    def test_unknown_streamed_leaf_fails_closed(self):
        record = self.arena.block_record("blocks.0")
        with self.assertRaisesRegex(TransferPlanError, "transfer_plan_unknown_leaf"):
            build_transfer_plan(record, ["a", "missing"])

    def test_leaf_metadata_is_immutable(self):
        record = self.arena.block_record("blocks.0")
        plan = build_transfer_plan(record, ["a"])
        with self.assertRaises(TypeError):
            plan.leaf_specs["a"]["weight"] = None
if __name__ == "__main__":
    unittest.main()
