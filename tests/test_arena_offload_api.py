"""Public Arena facade and generic memory-runtime seam coverage."""

from dataclasses import fields
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import torch

import toolkit.memory_management.arena_offload as arena_offload
from toolkit.memory_management.arena_offload import (
    ArenaOffloadConfig,
    estimate_training_working_reserve_hint_bytes,
    get_arena_runtime,
    is_arena_offloaded,
    validate_arena_training_mode,
)
from toolkit.memory_management.arena_offload.api import unwrap
from toolkit.memory_management.arena_offload.runtime import _fixed_working_bytes
from toolkit.memory_management.arena_offload.runtime import ArenaOffloadRuntime
from toolkit.memory_management.runtime import (
    RUNTIME_ATTR,
    is_memory_managed,
    memory_runtime_owns_compile,
)

GIB = 1024**3


def test_canonical_arena_import_does_not_depend_on_facade_import_order():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import toolkit.memory_management.canonical_arena; "
                "from toolkit.memory_management.arena_offload import "
                "ArenaOffloadConfig; assert ArenaOffloadConfig"
            ),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_public_facade_is_limited_to_integration_entry_points():
    assert arena_offload.__all__ == [
        "ArenaOffloadConfig",
        "close_arena_offload",
        "get_arena_runtime",
        "estimate_training_working_reserve_hint_bytes",
        "is_arena_offloaded",
        "model_load_arena_session",
        "prepare_canonical_storage",
        "prepare_canonical_storage_from_state_dict",
        "prepare_arena_offload",
        "validate_arena_training_mode",
    ]
    for implementation_name in (
        "ArenaCleanupError",
        "ArenaOffloadRuntime",
        "ArenaSetupFatalError",
        "BlockDiscoveryError",
        "DISPATCHER_GENERATION",
        "close_memory_runtime",
        "discover_blocks",
        "get_memory_runtime",
        "is_memory_managed",
        "memory_runtime_owns_compile",
    ):
        assert not hasattr(arena_offload, implementation_name)


class _Wrapper(torch.nn.Module):
    """Stands in for Accelerate/DDP, which expose the real model at `.module`."""

    def __init__(self, inner):
        super().__init__()
        self.module = inner


class _FakeModelConfig:
    quantize = True
    qtype = "qfloat8"
    layer_offloading = True
    layer_offloading_smart = True
    layer_offloading_fp8_forward = True
    layer_offloading_fp8_grad_input = True
    layer_offloading_fp8_sampling = True
    compile = True
    compile_sample = True
    train_compile_blocks = False
    layer_offloading_smart_working_reserve_gb = -1.0
    layer_offloading_smart_physical_vram_headroom_gb = None
    layer_offloading_smart_wddm_hard_gb = 1.0
    layer_offloading_smart_cap_calibration = True
    layer_offloading_wddm_spill_reserve_pct = 0.10
    layer_offloading_block_stream_only = False
    layer_offloading_checkpoint_keep_last = 2
    layer_offloading_prefetch_depth = 3
    layer_offloading_smart_sampling_working_reserve_gb = -1.0
    layer_offloading_smart_sampling_physical_vram_headroom_gb = -1.0
    layer_offloading_smart_sampling_wddm_hard_gb = 1.0
    layer_offloading_strict_vram_cap = False


class ArenaOffloadHelpersTest(unittest.TestCase):
    def test_training_mode_requires_frozen_immutable_base_weights(self):
        validate_arena_training_mode()

        with self.assertRaisesRegex(ValueError, "full-model fine-tuning"):
            validate_arena_training_mode(full_finetune=True)
        with self.assertRaisesRegex(ValueError, "merge_network_on_save"):
            validate_arena_training_mode(mutates_base_weights=True)
        with self.assertRaisesRegex(ValueError, "text encoder during training"):
            validate_arena_training_mode(train_text_encoder=True)
        with self.assertRaisesRegex(ValueError, "text encoder during training"):
            validate_arena_training_mode(unload_text_encoder=False)

    def test_helpers_are_none_safe(self):
        self.assertIsNone(get_arena_runtime(None))
        self.assertFalse(is_arena_offloaded(None))
        self.assertFalse(is_memory_managed(None))
        self.assertFalse(memory_runtime_owns_compile(None))

    def test_plain_module_is_not_managed(self):
        model = torch.nn.Linear(4, 4)
        self.assertFalse(is_arena_offloaded(model))
        self.assertFalse(is_memory_managed(model))
        self.assertFalse(memory_runtime_owns_compile(model))

    def test_runtime_found_through_wrappers(self):
        inner = torch.nn.Linear(4, 4)
        runtime = object()
        setattr(inner, RUNTIME_ATTR, runtime)
        wrapped = _Wrapper(_Wrapper(inner))

        self.assertIs(unwrap(wrapped), inner)
        self.assertIs(get_arena_runtime(wrapped), runtime)
        self.assertTrue(is_arena_offloaded(wrapped))
        self.assertTrue(is_memory_managed(wrapped))
        self.assertTrue(memory_runtime_owns_compile(wrapped))

    def test_legacy_backend_is_managed_but_does_not_own_compile(self):
        """The distinction generic block compile depends on."""
        model = torch.nn.Linear(4, 4)
        model._memory_manager = object()

        self.assertTrue(is_memory_managed(model))
        self.assertFalse(is_arena_offloaded(model))
        self.assertFalse(memory_runtime_owns_compile(model))

    def test_unwrap_terminates_on_self_referential_wrapper(self):
        model = torch.nn.Linear(4, 4)
        model.module = model  # a module that is its own `.module`
        self.assertIs(unwrap(model), model)

    def test_place_permanent_modules_does_not_move_canonical_leaf(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.root_token = torch.nn.Parameter(
                    torch.ones(4), requires_grad=False
                )
                self.register_buffer("root_buffer", torch.ones(4))
                self.canonical = torch.nn.Linear(4, 4)
                self.permanent = torch.nn.Linear(4, 4)

        model = Model()
        runtime = object.__new__(ArenaOffloadRuntime)
        runtime._model = model
        runtime._canonical_modules = (model.canonical,)
        runtime._closed = False

        runtime.place_permanent_modules("cpu", torch.float64)

        self.assertEqual(model.canonical.weight.dtype, torch.float32)
        self.assertEqual(model.permanent.weight.dtype, torch.float64)
        self.assertEqual(model.root_token.dtype, torch.float64)
        self.assertEqual(model.root_buffer.dtype, torch.float64)

    def test_whole_model_move_parks_cpu_and_restores_arena_device(self):
        model = object()
        runtime = object.__new__(ArenaOffloadRuntime)
        runtime._model = model
        runtime._closed = False
        runtime._disposed = False
        runtime._device = torch.device("cuda:0")
        runtime._executor = SimpleNamespace(active_executions=0)
        runtime._device_state_parked_plan = None
        runtime._permanent_placement = (torch.device("cuda:0"), torch.float32)
        events = []

        def place(device, dtype=None):
            normalized = torch.device(device)
            runtime._permanent_placement = (normalized, dtype)
            events.append(("place", normalized, dtype))

        runtime.place_permanent_modules = place
        runtime.park_residency_for_external_phase = lambda: events.append(
            ("park",)
        )
        runtime.restore_residency_after_external_phase = lambda: events.append(
            ("restore",)
        )

        self.assertIs(runtime.handle_whole_model_move("cpu"), model)
        self.assertIs(runtime.handle_whole_model_move("cuda:0"), model)
        self.assertEqual(
            events,
            [
                ("park",),
                ("place", torch.device("cpu"), torch.float32),
                ("place", torch.device("cuda:0"), torch.float32),
                ("restore",),
            ],
        )

    def test_whole_model_move_rejects_unsupported_intent_before_mutation(self):
        runtime = object.__new__(ArenaOffloadRuntime)
        runtime._model = object()
        runtime._closed = False
        runtime._disposed = False
        runtime._device = torch.device("cuda:0")
        runtime._executor = SimpleNamespace(active_executions=0)
        runtime._device_state_parked_plan = None
        runtime._permanent_placement = (torch.device("cuda:0"), torch.float32)
        runtime.place_permanent_modules = unittest.mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "dtype_change"):
            runtime.handle_whole_model_move("cuda:0", dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, "cuda_device"):
            runtime.handle_whole_model_move("cuda:1")
        with self.assertRaisesRegex(RuntimeError, "memory_format"):
            runtime.handle_whole_model_move(
                "cuda:0", memory_format=torch.channels_last
            )
        runtime._executor.active_executions = 1
        with self.assertRaisesRegex(RuntimeError, "during_execution"):
            runtime.handle_whole_model_move("cpu")
        runtime.place_permanent_modules.assert_not_called()


class ArenaOffloadConfigTest(unittest.TestCase):
    def test_auto_training_reserve_uses_largest_configured_resolution(self):
        datasets = [
            SimpleNamespace(resolution=256),
            SimpleNamespace(resolution=512),
            SimpleNamespace(resolution=1024),
        ]

        hint = estimate_training_working_reserve_hint_bytes(datasets)
        config = ArenaOffloadConfig.from_model_config(
            _FakeModelConfig(),
            training_working_reserve_hint_bytes=hint,
        )

        self.assertIsNotNone(hint)
        self.assertAlmostEqual(config._policy.working_reserve_gib, hint / GIB)
        self.assertGreater(config._policy.working_reserve_gib, 5.0)

    def test_explicit_training_reserve_overrides_configured_shape_hint(self):
        class Manual(_FakeModelConfig):
            layer_offloading_smart_working_reserve_gb = 6.0

        config = ArenaOffloadConfig.from_model_config(
            Manual(),
            training_working_reserve_hint_bytes=10 * GIB,
        )

        self.assertEqual(config._policy.working_reserve_gib, 6.0)

    def test_from_model_config_maps_the_public_surface(self):
        config = ArenaOffloadConfig.from_model_config(_FakeModelConfig())

        self.assertTrue(config.enabled)
        self.assertTrue(config.fp8_forward)
        self.assertTrue(config.fp8_backward)
        self.assertTrue(config.fp8_sampling)
        # compile_blocks is derived, not its own public knob.
        self.assertTrue(config.compile_blocks)
        self.assertFalse(config.strict_vram_cap)
        self.assertEqual(config._policy.prefetch_depth, 3)
        self.assertEqual(config._policy.checkpoint_keep_last, 2)
        self.assertTrue(config._policy.cap_calibration)

    def test_public_surface_is_narrow(self):
        public = {field.name for field in fields(ArenaOffloadConfig) if not field.name.startswith("_")}
        self.assertEqual(
            public,
            {
                "enabled",
                "fp8_forward",
                "fp8_backward",
                "fp8_sampling",
                "compile_blocks",
                "strict_vram_cap",
            },
        )

    def test_fp8_flags_require_fp8_weights(self):
        """An fp8_* toggle on a non-fp8 model is a no-op, not a crash."""

        class NoQuant(_FakeModelConfig):
            quantize = False

        with self.assertWarnsRegex(RuntimeWarning, "ignored irrelevant FP8 options"):
            config = ArenaOffloadConfig.from_model_config(NoQuant())
        self.assertFalse(config.fp8_forward)
        self.assertFalse(config.fp8_backward)
        self.assertFalse(config.fp8_sampling)
        self.assertTrue(config.enabled)

    def test_old_torchao_disables_only_torchao_arena_fp8(self):
        class TorchAOFloat8(_FakeModelConfig):
            qtype = "float8"

        with unittest.mock.patch(
            "toolkit.memory_management.arena_offload.api."
            "torchao_arena_fp8_supported",
            return_value=False,
        ), unittest.mock.patch(
            "toolkit.memory_management.arena_offload.api.TORCHAO_VERSION",
            "0.10.0",
        ):
            with self.assertWarnsRegex(RuntimeWarning, "requires_0.17.0"):
                config = ArenaOffloadConfig.from_model_config(TorchAOFloat8())

        self.assertTrue(config.enabled)
        self.assertFalse(config.fp8_forward)
        self.assertFalse(config.fp8_backward)
        self.assertFalse(config.fp8_sampling)

    def test_quanto_fp8_does_not_require_new_torchao_tensor_format(self):
        with unittest.mock.patch(
            "toolkit.memory_management.arena_offload.api."
            "torchao_arena_fp8_supported",
            return_value=False,
        ):
            config = ArenaOffloadConfig.from_model_config(_FakeModelConfig())

        self.assertTrue(config.fp8_forward)
        self.assertTrue(config.fp8_backward)
        self.assertTrue(config.fp8_sampling)

    def test_missing_attributes_fall_back_to_defaults(self):
        config = ArenaOffloadConfig.from_model_config(object())
        self.assertFalse(config.enabled)
        self.assertFalse(config.compile_blocks)
        self.assertEqual(config._policy.prefetch_depth, 3)
        self.assertFalse(config._policy.cap_calibration)

    def test_dead_compile_aliases_do_not_enable_arena_compile(self):
        class DeadAliases:
            compile = False
            compile_sample = True
            train_compile_blocks = True

        config = ArenaOffloadConfig.from_model_config(DeadAliases())
        self.assertFalse(config.compile_blocks)

    def test_backward_without_fp8_forward_is_ignored_once(self):
        class Invalid:
            quantize = True
            qtype = "qfloat8"
            layer_offloading_fp8_grad_input = True

        with self.assertWarnsRegex(RuntimeWarning, "fp8_backward_without_fp8_forward"):
            config = ArenaOffloadConfig.from_model_config(Invalid())
        self.assertFalse(config.fp8_backward)


class SamplingReserveTest(unittest.TestCase):
    def test_auto_working_reserve_is_none(self):
        """Unset / negative / 'auto' all mean 'let the runtime size it'."""
        for value in (None, -1.0, "auto"):
            self.assertIsNone(_fixed_working_bytes(value))

    def test_explicit_working_reserve_is_bytes(self):
        self.assertEqual(_fixed_working_bytes(2.0), 2 * GIB)
        self.assertEqual(_fixed_working_bytes(0), 0)


if __name__ == "__main__":
    unittest.main()
