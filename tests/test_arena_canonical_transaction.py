import gc
import unittest
import weakref
from unittest import mock

import torch

from toolkit.memory_management import pin_manager
from toolkit.memory_management.arena_offload.construction import CanonicalBuildError
from toolkit.memory_management.canonical_arena import CanonicalArena


def frozen_linear():
    layer = torch.nn.Linear(4, 4)
    layer.requires_grad_(False)
    return layer


class FailingLinear(torch.nn.Linear):
    fail_publication = False

    def __setattr__(self, name, value):
        if name == "weight" and getattr(self, "fail_publication", False):
            raise RuntimeError("injected_commit_failure")
        super().__setattr__(name, value)


class CanonicalTransactionTests(unittest.TestCase):
    def test_prepare_does_not_mutate_and_direct_population_uses_final_views(self):
        model = torch.nn.Sequential(frozen_linear())
        layer = model[0]
        original = layer.weight
        arena = CanonicalArena()
        build = arena.prepare({"blocks.0": [("linear", layer)]}, model=model)
        self.assertIs(layer.weight, original)
        destination = build.destinations[("blocks.0", "linear", "weight")]
        expected = torch.full_like(destination, 3)
        build.populate(lambda destinations: destinations[("blocks.0", "linear", "weight")].copy_(expected))
        build.commit()
        try:
            self.assertEqual(layer.weight.data_ptr(), destination.data_ptr())
            torch.testing.assert_close(layer.weight, expected)
        finally:
            CanonicalArena.unguard_whole_model_to(model)
            arena.release()

    def test_model_source_leaves_match_destination_keys(self):
        model = torch.nn.Sequential(frozen_linear())
        layer = model[0]
        arena = CanonicalArena()
        build = arena.prepare({"blocks.0": [("linear", layer)]}, model=model)
        try:
            sources = dict(build.model_source_leaves())
            self.assertEqual(set(sources), set(build.destinations))
            torch.testing.assert_close(
                sources[("blocks.0", "linear", "weight")], layer.weight
            )
            torch.testing.assert_close(
                sources[("blocks.0", "linear", "bias")], layer.bias
            )
        finally:
            build.rollback()

    def test_incremental_direct_block_releases_bounded_source_before_commit(self):
        model = torch.nn.Sequential(frozen_linear())
        layer = model[0]
        expected = layer.weight.detach().clone()
        source = layer.weight
        source_ref = weakref.ref(source)
        arena = CanonicalArena()
        build = arena.prepare({}, model=model)
        build.add_block("blocks.0", [("linear", layer)])
        build.populate_block_from_model("blocks.0")
        build.release_block_sources_to_meta("blocks.0")
        source = None
        gc.collect()

        self.assertIsNone(source_ref())
        self.assertEqual(layer.weight.device.type, "meta")

        build.finish_population()
        build.commit()
        try:
            torch.testing.assert_close(layer.weight, expected)
        finally:
            CanonicalArena.unguard_whole_model_to(model)
            arena.release()

    def test_state_dict_population_consumes_each_source_before_next_block(self):
        layers = [frozen_linear(), frozen_linear()]
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList()
        for layer in layers:
            block = torch.nn.Module()
            block.linear = layer
            model.blocks.append(block)
        state = {
            f"blocks.{index}.linear.weight": torch.full_like(layer.weight, index + 1)
            for index, layer in enumerate(layers)
        }
        state.update({
            f"blocks.{index}.linear.bias": torch.full_like(layer.bias, index + 3)
            for index, layer in enumerate(layers)
        })
        expected = {key: value.clone() for key, value in state.items()}
        first_refs = (
            weakref.ref(state["blocks.0.linear.weight"]),
            weakref.ref(state["blocks.0.linear.bias"]),
        )
        residual = torch.tensor([9.0])
        state["head.weight"] = residual

        def blocks():
            yield "blocks.0", [("linear", layers[0])]
            gc.collect()
            self.assertNotIn("blocks.0.linear.weight", state)
            self.assertNotIn("blocks.0.linear.bias", state)
            self.assertTrue(all(ref() is None for ref in first_refs))
            self.assertIn("blocks.1.linear.weight", state)
            yield "blocks.1", [("linear", layers[1])]

        arena = CanonicalArena()
        build = arena.prepare({}, model=model)
        consumed = build.populate_from_state_dict_consuming(
            state,
            blocks=blocks(),
        )

        self.assertEqual(
            set(consumed),
            {
                "blocks.0.linear.weight",
                "blocks.0.linear.bias",
                "blocks.1.linear.weight",
                "blocks.1.linear.bias",
            },
        )
        self.assertEqual(set(state), {"head.weight"})
        self.assertIs(state["head.weight"], residual)

        build.commit()
        try:
            for index, layer in enumerate(layers):
                torch.testing.assert_close(
                    layer.weight,
                    expected[f"blocks.{index}.linear.weight"],
                )
                torch.testing.assert_close(
                    layer.bias,
                    expected[f"blocks.{index}.linear.bias"],
                )
        finally:
            arena.release()

    def test_direct_population_failure_rolls_back_without_marker_or_pin_leak(self):
        model = torch.nn.Sequential(frozen_linear())
        layer = model[0]
        original = layer.weight
        before = pin_manager.pinned_bytes_by_kind().get("weights", 0)
        build = CanonicalArena().prepare({"blocks.0": [("linear", layer)]}, model=model)
        with self.assertRaisesRegex(RuntimeError, "injected_population_failure"):
            build.populate(lambda _destinations: (_ for _ in ()).throw(RuntimeError("injected_population_failure")))
        self.assertIs(layer.weight, original)
        self.assertFalse(hasattr(model, "_arena_offload_runtime"))
        self.assertEqual(pin_manager.pinned_bytes_by_kind().get("weights", 0), before)
        model(torch.randn(1, 4))

    def test_pin_failure_after_an_earlier_block_releases_everything(self):
        layers = [frozen_linear(), frozen_linear()]
        originals = [layer.weight for layer in layers]
        arena = CanonicalArena()
        build = arena.prepare({
            "blocks.0": [("linear", layers[0])],
            "blocks.1": [("linear", layers[1])],
        })
        real_commit = pin_manager.pin_register_commit
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected_pin_failure")
            return real_commit(*args, **kwargs)

        with mock.patch.object(pin_manager, "pin_register_commit", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected_pin_failure"):
                build.populate_from_model()
        self.assertEqual([layer.weight is original for layer, original in zip(layers, originals)], [True, True])
        self.assertFalse(arena.canonicalized)

    def test_commit_failure_after_first_repoint_restores_parameter_identity(self):
        first = frozen_linear()
        second = FailingLinear(4, 4)
        second.requires_grad_(False)
        originals = (first.weight, second.weight)
        arena = CanonicalArena()
        build = arena.prepare({"blocks.0": [("first", first), ("second", second)]})
        build.populate_from_model()
        second.fail_publication = True
        with self.assertRaisesRegex(RuntimeError, "injected_commit_failure"):
            build.commit()
        second.fail_publication = False
        self.assertIs(first.weight, originals[0])
        self.assertIs(second.weight, originals[1])
        self.assertFalse(arena.canonicalized)
        first(torch.randn(1, 4))
        second(torch.randn(1, 4))


    def test_unsupported_layout_mid_stack_leaves_originals_untouched(self):
        layers = [frozen_linear(), frozen_linear()]
        originals = [layer.weight for layer in layers]
        arena = CanonicalArena()
        from toolkit.memory_management.arena_offload import construction

        real_inspect = construction.inspect_block
        calls = 0

        def fail_second(key, entries):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("unsupported_quant_wrapper:blocks.1:linear")
            return real_inspect(key, entries)

        with mock.patch.object(construction, "inspect_block", side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, "unsupported_quant_wrapper"):
                arena.prepare({
                    "blocks.0": [("linear", layers[0])],
                    "blocks.1": [("linear", layers[1])],
                })
        self.assertEqual(
            [layer.weight is original for layer, original in zip(layers, originals)],
            [True, True],
        )
        self.assertFalse(arena.canonicalized)

    def test_allocation_failure_after_first_block_prepared_is_atomic(self):
        layers = [frozen_linear(), frozen_linear()]
        originals = [layer.weight for layer in layers]
        arena = CanonicalArena()
        real_prepare = pin_manager.pin_register_prepare
        calls = 0

        def fail_second(nbytes):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected_allocation_failure")
            return real_prepare(nbytes)

        with mock.patch.object(pin_manager, "pin_register_prepare", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected_allocation_failure"):
                arena.prepare({
                    "blocks.0": [("linear", layers[0])],
                    "blocks.1": [("linear", layers[1])],
                })
        self.assertEqual(
            [layer.weight is original for layer, original in zip(layers, originals)],
            [True, True],
        )
        self.assertFalse(arena.canonicalized)

    def test_wrapper_validation_failure_releases_committed_pin(self):
        model = torch.nn.Sequential(frozen_linear())
        layer = model[0]
        original = layer.weight
        before = pin_manager.pinned_bytes_by_kind().get("weights", 0)
        build = CanonicalArena().prepare(
            {"blocks.0": [("linear", layer)]},
            model=model,
        )
        with mock.patch(
            "toolkit.memory_management.arena_offload.construction.linear_views",
            side_effect=RuntimeError("injected_wrapper_validation_failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected_wrapper_validation_failure"):
                build.populate_from_model()
        self.assertIs(layer.weight, original)
        self.assertEqual(pin_manager.pinned_bytes_by_kind().get("weights", 0), before)
        model(torch.randn(1, 4))

    def test_commit_without_population_rolls_back_prepared_build(self):
        layer = frozen_linear()
        original = layer.weight
        arena = CanonicalArena()
        build = arena.prepare({"blocks.0": [("linear", layer)]})
        with self.assertRaisesRegex(CanonicalBuildError, "canonical_build_not_populated"):
            build.commit()
        self.assertIs(layer.weight, original)
        self.assertFalse(arena.canonicalized)


if __name__ == "__main__":
    unittest.main()
