import io
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from toolkit.memory_management import pin_manager
from toolkit.memory_management.canonical_arena import (
    CanonicalArena,
    CanonicalArenaError,
)
def _linear(in_f=8, out_f=4, bias=True):
    layer = nn.Linear(in_f, out_f, bias=bias)
    layer.weight.requires_grad_(False)
    if bias:
        layer.bias.requires_grad_(False)
    return layer


class CanonicalizeTests(unittest.TestCase):
    def test_canonicalize_repoints_params_into_one_flat_per_block(self):
        a = _linear()
        b = _linear()
        arena = CanonicalArena()
        try:
            stats = arena.canonicalize({"blocks.0": [("lin_a", a), ("lin_b", b)]})
            self.assertEqual(stats.blocks, 1)
            flat_ptr = arena.block_pack("blocks.0").host_flat.untyped_storage().data_ptr()
            self.assertEqual(a.weight.untyped_storage().data_ptr(), flat_ptr)
            self.assertEqual(b.weight.untyped_storage().data_ptr(), flat_ptr)
            self.assertEqual(a.bias.untyped_storage().data_ptr(), flat_ptr)
            self.assertTrue(arena.canonicalized)
        finally:
            arena.release()

    def test_double_canonicalize_is_rejected(self):
        layer = _linear()
        arena = CanonicalArena()
        try:
            arena.canonicalize({"blocks.0": [("lin", layer)]})
            with self.assertRaises(CanonicalArenaError):
                arena.canonicalize({"blocks.0": [("lin", layer)]})
        finally:
            arena.release()

    def test_trainable_leaf_is_rejected_and_nothing_repointed(self):
        trainable = _linear()
        trainable.weight.requires_grad_(True)
        frozen = _linear()
        original_frozen_ptr = frozen.weight.untyped_storage().data_ptr()
        arena = CanonicalArena()
        with self.assertRaises(CanonicalArenaError):
            arena.canonicalize(
                {"blocks.0": [("a", trainable)], "blocks.1": [("b", frozen)]}
            )
        # Frozen block's leaf must never have been repointed: the frozen
        # check runs over EVERY block before ANY block is built.
        self.assertEqual(frozen.weight.untyped_storage().data_ptr(), original_frozen_ptr)
        self.assertFalse(arena.canonicalized)

    def test_trainable_bias_is_rejected(self):
        layer = _linear()
        layer.bias.requires_grad_(True)
        arena = CanonicalArena()
        with self.assertRaises(CanonicalArenaError):
            arena.canonicalize({"blocks.0": [("lin", layer)]})

    def test_entries_without_module_are_rejected(self):
        layer = _linear()
        arena = CanonicalArena()
        with self.assertRaises(CanonicalArenaError):
            arena.canonicalize({"blocks.0": [("lin", layer.weight, layer.bias)]})

    def test_state_dict_round_trip_after_canonicalize(self):
        layer = _linear()
        expected = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        arena = CanonicalArena()
        try:
            arena.canonicalize({"blocks.0": [("lin", layer)]})
            buffer = io.BytesIO()
            torch.save(layer.state_dict(), buffer)
            buffer.seek(0)
            loaded = torch.load(buffer, weights_only=True)
            for key, value in expected.items():
                self.assertTrue(torch.equal(value, loaded[key]), key)
            # load_state_dict must copy in place so Parameter identity and the
            # canonical arena storage view remain unchanged.
            flat_ptr = arena.block_pack("blocks.0").host_flat.untyped_storage().data_ptr()
            layer.load_state_dict(loaded)
            self.assertEqual(layer.weight.untyped_storage().data_ptr(), flat_ptr)
        finally:
            arena.release()

    def test_release_returns_pin_ledger_bytes(self):
        layer = _linear(in_f=512, out_f=512, bias=True)
        arena = CanonicalArena()
        arena.canonicalize({"blocks.0": [("lin", layer)]})
        pinned = arena.committed_pinned_bytes()
        self.assertGreater(pinned, 0)
        before = pin_manager.pinned_bytes_by_kind().get("weights", 0)
        arena.release()
        after = pin_manager.pinned_bytes_by_kind().get("weights", 0)
        # Ledger tracks page-rounded committed bytes (cudaHostRegister is
        # page-granular); committed_pinned_bytes() reports the unpadded
        # logical total, so release only guarantees the delta covers at
        # LEAST the logical bytes, not an exact match.
        self.assertGreaterEqual(before - after, pinned)
        self.assertEqual(arena.committed_pinned_bytes(), 0)
        self.assertEqual(arena.block_keys(), ())

    def test_pageable_build_can_register_and_unregister_same_storage(self):
        layer = _linear(in_f=64, out_f=64, bias=True)
        expected = layer.weight.detach().clone()
        arena = CanonicalArena()
        build = arena.prepare(
            {"blocks.0": [("lin", layer)]},
            pin_on_finish=False,
        )
        try:
            build.populate_from_model()
            stats = build.commit()
            record = arena.block_record("blocks.0")
            pointer = record.host_flat.data_ptr()
            self.assertEqual(stats.pinned_bytes, 0)
            self.assertFalse(record.pack.pinned)
            self.assertEqual(arena.committed_pinned_bytes(), 0)

            if torch.cuda.is_available():
                self.assertTrue(arena.pin_block("blocks.0", required=True))
                self.assertTrue(record.pack.pinned)
                self.assertEqual(record.host_flat.data_ptr(), pointer)
                self.assertTrue(arena.unpin_block("blocks.0"))
                self.assertFalse(record.pack.pinned)
                self.assertEqual(record.host_flat.data_ptr(), pointer)
            torch.testing.assert_close(layer.weight, expected)
        finally:
            arena.release()

    def test_unknown_block_lookup_returns_none(self):
        arena = CanonicalArena()
        try:
            arena.canonicalize({"blocks.0": [("lin", _linear())]})
            self.assertIsNone(arena.block_pack("blocks.1"))
            self.assertIsNone(arena.block_record("blocks.1"))
        finally:
            arena.release()


class WholeModelToGuardTests(unittest.TestCase):
    def test_guarded_to_raises(self):
        model = nn.Sequential(_linear())
        arena = CanonicalArena()
        try:
            arena.canonicalize({"blocks.0": [("lin", model[0])]})
            CanonicalArena.guard_whole_model_to(model)
            for move in (
                lambda: model.to(torch.device("cpu")),
                model.cpu,
                model.cuda,
            ):
                with self.assertRaises(CanonicalArenaError):
                    move()
        finally:
            CanonicalArena.unguard_whole_model_to(model)
            arena.release()

    def test_guard_is_idempotent(self):
        model = nn.Sequential(_linear())
        CanonicalArena.guard_whole_model_to(model)
        original = model.to
        CanonicalArena.guard_whole_model_to(model)
        self.assertIs(model.to, original)
        CanonicalArena.unguard_whole_model_to(model)

    def test_guarded_movement_routes_all_entry_points_to_runtime(self):
        model = nn.Sequential(_linear())
        runtime = SimpleNamespace(
            device=torch.device("cuda:0"),
            handle_whole_model_move=mock.Mock(return_value=model),
        )
        model._arena_offload_runtime = runtime
        CanonicalArena.guard_whole_model_to(model)
        try:
            self.assertIs(model.to(torch.device("cpu")), model)
            self.assertIs(model.cpu(), model)
            self.assertIs(model.cuda(), model)
            self.assertEqual(runtime.handle_whole_model_move.call_count, 3)
            self.assertEqual(
                runtime.handle_whole_model_move.call_args_list[0].args,
                (torch.device("cpu"),),
            )
            self.assertEqual(
                runtime.handle_whole_model_move.call_args_list[1].args,
                ("cpu",),
            )
            self.assertEqual(
                runtime.handle_whole_model_move.call_args_list[2].args,
                (torch.device("cuda:0"),),
            )
        finally:
            CanonicalArena.unguard_whole_model_to(model)
            del model._arena_offload_runtime

    def test_unguard_restores_normal_to(self):
        model = nn.Sequential(_linear())
        originals = (model.to, model.cuda, model.cpu)
        CanonicalArena.guard_whole_model_to(model)
        CanonicalArena.unguard_whole_model_to(model)
        self.assertEqual((model.to, model.cuda, model.cpu), originals)
        # Ordinary .to() must work again (no canonicalized leaves here).
        model.to(torch.device("cpu"))

    def test_unguarded_model_never_raises(self):
        model = nn.Sequential(_linear())
        model.to(torch.device("cpu"))  # sanity: no guard installed, no raise


if __name__ == "__main__":
    unittest.main()
