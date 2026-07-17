import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from toolkit.memory_management.arena_offload import (
    ArenaOffloadConfig,
    prepare_canonical_storage,
    prepare_arena_offload,
)
from toolkit.memory_management.arena_offload.errors import (
    ArenaCleanupError,
    ArenaSetupFatalError,
    is_fatal_arena_setup,
    recover_allows_next_job,
)
from toolkit.memory_management.arena_offload.discovery import BlockDiscoveryError
from toolkit.memory_management.arena_offload.ownership import (
    acquire_process_owner,
    active_process_owner,
    release_process_owner,
)
from toolkit.memory_management.arena_offload.resources import ArenaRuntimeResources
from toolkit.memory_management.arena_offload.runtime import ArenaOffloadRuntime
from toolkit.memory_management import pin_manager
from toolkit.memory_management import vram_budget
from toolkit.quantization.fp8_linear import (
    fp8_grad_input_enabled,
    set_fp8_grad_input_enabled,
)
from toolkit.memory_management.manager_modules import _DEVICE_STATE
from toolkit.memory_management.residency import ResidencyPlan
from toolkit.models.base_model import BaseModel
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from jobs.BaseJob import BaseJob


@pytest.fixture(autouse=True)
def _disable_cuda_host_registration(monkeypatch):
    """Lifecycle ownership tests do not need a live CUDA pin registration."""
    def fake_commit(candidate, nbytes, kind, **_kwargs):
        return SimpleNamespace(
            tensor=candidate,
            nbytes=int(nbytes),
            kind=kind,
            pinned=True,
            mechanism="register",
        )

    monkeypatch.setattr(pin_manager, "pin_register_commit", fake_commit)
    monkeypatch.setattr(pin_manager, "release", lambda _handle: None)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, value):
        return self.linear(value)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList((_Block(), _Block()))
        self.gradient_checkpointing = True
        self._checkpoint_keep_last = 0

    @property
    def weight(self):
        return self.blocks[0].linear.weight

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


def _frozen_linear():
    model = _Model()
    model.requires_grad_(False)
    return model


def test_process_owner_is_exclusive_sequential_and_stale_safe():
    first = acquire_process_owner("cpu")
    try:
        with pytest.raises(RuntimeError, match="arena_runtime_already_active"):
            acquire_process_owner("cpu")
    finally:
        release_process_owner(first)

    second = acquire_process_owner("cpu")
    try:
        with pytest.raises(RuntimeError, match="owner_mismatch"):
            release_process_owner(first)
        assert active_process_owner() is second
    finally:
        release_process_owner(second)


def test_precommit_failure_preserves_original_classification_and_model():
    model = _frozen_linear()
    original = model.weight

    with pytest.raises(BlockDiscoveryError, match="block_container_not_found"):
        prepare_arena_offload(
            model,
            device="cpu",
            block_names=("missing",),
            config=ArenaOffloadConfig(enabled=True),
        )

    assert model.weight is original
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_runtime")
    assert not hasattr(model, "_arena_offload_disposed")


def test_disabled_config_and_unsupported_architecture_fail_before_mutation():
    model = _frozen_linear()
    original = model.weight
    with pytest.raises(ValueError, match="arena_offload_not_enabled"):
        prepare_arena_offload(
            model,
            device="cpu",
            config=ArenaOffloadConfig(enabled=False),
        )
    unsupported = torch.nn.Linear(4, 4)
    unsupported.gradient_checkpointing = True
    with pytest.raises(BlockDiscoveryError, match="no_repeated_block_container"):
        prepare_arena_offload(
            unsupported,
            device="cpu",
            config=ArenaOffloadConfig(enabled=True),
        )
    assert model.weight is original
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_runtime")
    assert not hasattr(model, "_arena_offload_disposed")


def test_direct_loader_rollback_releases_preparation_owner():
    model = _frozen_linear()
    build = prepare_canonical_storage(model, block_names=("blocks",), device="cpu")
    assert active_process_owner() is not None
    build.rollback()
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_disposed")


def test_planning_failure_is_precommit_and_preserves_model():
    model = _frozen_linear()
    original = model.weight
    original_error = RuntimeError("residency construction failed")
    with mock.patch(
        "toolkit.memory_management.arena_offload.runtime.build_training_plan",
        side_effect=original_error,
    ):
        with pytest.raises(RuntimeError, match="residency construction failed") as caught:
            prepare_arena_offload(
                model,
                device="cpu",
                block_names=("blocks",),
                config=ArenaOffloadConfig(enabled=True),
            )

    assert caught.value is original_error
    assert active_process_owner() is None
    assert model.weight is original
    assert not hasattr(model, "_arena_offload_disposed")


def test_precommit_setup_failure_restores_process_policy():
    original_simulated = vram_budget.simulated_card_bytes()
    original_fp8 = fp8_grad_input_enabled()
    try:
        vram_budget.set_simulated_card_bytes(1234)
        set_fp8_grad_input_enabled(True)
        config = ArenaOffloadConfig(
            enabled=True,
            fp8_backward=False,
            _simulated_vram_gib=0.5,
        )
        with mock.patch(
            "toolkit.memory_management.arena_offload.runtime.apply_simulated_card",
            side_effect=lambda _value, device=None: (
                vram_budget.set_simulated_card_bytes(5678)
            ),
        ), mock.patch(
            "toolkit.memory_management.arena_offload.runtime.build_training_plan",
            side_effect=RuntimeError("planning failed"),
        ):
            with pytest.raises(RuntimeError, match="planning failed"):
                prepare_arena_offload(
                    _frozen_linear(),
                    device="cpu",
                    block_names=("blocks",),
                    config=config,
                )

        assert vram_budget.simulated_card_bytes() == 1234
        assert fp8_grad_input_enabled()
        assert active_process_owner() is None
    finally:
        vram_budget.set_simulated_card_bytes(original_simulated)
        set_fp8_grad_input_enabled(original_fp8)


def test_postcommit_setup_failure_restores_process_policy():
    original_simulated = vram_budget.simulated_card_bytes()
    original_fp8 = fp8_grad_input_enabled()
    model = _frozen_linear()
    plan = {
        "offload_ids": set(),
        "protected_training_leaf_keys": frozenset(),
        "fits": True,
    }
    try:
        vram_budget.set_simulated_card_bytes(1234)
        set_fp8_grad_input_enabled(True)
        config = ArenaOffloadConfig(
            enabled=True,
            fp8_backward=False,
            _simulated_vram_gib=0.5,
        )
        with mock.patch(
            "toolkit.memory_management.arena_offload.runtime.apply_simulated_card",
            side_effect=lambda _value, device=None: (
                vram_budget.set_simulated_card_bytes(5678)
            ),
        ), mock.patch(
            "toolkit.memory_management.arena_offload.runtime.build_training_plan",
            return_value=plan,
        ), mock.patch(
            "toolkit.memory_management.arena_offload.runtime.ResidencyState",
            side_effect=RuntimeError("postcommit failed"),
        ):
            with pytest.raises(ArenaSetupFatalError):
                prepare_arena_offload(
                    model,
                    device="cpu",
                    block_names=("blocks",),
                    config=config,
                )

        assert vram_budget.simulated_card_bytes() == 1234
        assert fp8_grad_input_enabled()
        assert active_process_owner() is None
        assert model._arena_offload_disposed
    finally:
        vram_budget.set_simulated_card_bytes(original_simulated)
        set_fp8_grad_input_enabled(original_fp8)


@pytest.mark.parametrize(
    "target",
    (
        "toolkit.memory_management.arena_offload.runtime.ResidencyState",
        "toolkit.memory_management.arena_offload.dispatcher.prepare_block_dispatcher_runtime",
    ),
)
def test_postcommit_fault_boundaries_are_fatal_and_never_fall_back(target):
    model = _frozen_linear()
    failure = RuntimeError(f"injected:{target.rsplit('.', 1)[-1]}")
    plan = {
        "offload_ids": set(),
        "protected_training_leaf_keys": frozenset(),
        "fits": True,
    }
    patches = [
        mock.patch(
            "toolkit.memory_management.arena_offload.runtime.build_training_plan",
            return_value=plan,
        ),
        mock.patch(target, side_effect=failure),
    ]
    with patches[0], patches[1]:
        with pytest.raises(ArenaSetupFatalError) as caught:
            prepare_arena_offload(
                model,
                device="cpu",
                block_names=("blocks",),
                config=ArenaOffloadConfig(enabled=True),
            )
    assert caught.value.__cause__ is failure
    assert active_process_owner() is None
    assert model._arena_offload_disposed
    assert not hasattr(model, "_memory_manager")


def test_fatal_setup_classification_blocks_recovery_through_wrappers():
    fatal = ArenaSetupFatalError("committed setup failed")
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = fatal

    assert is_fatal_arena_setup(wrapper)
    assert not recover_allows_next_job(wrapper, True)
    assert recover_allows_next_job(RuntimeError("ordinary"), True)


def test_resource_release_continues_after_cleanup_error_and_is_idempotent():
    calls = []
    model = _frozen_linear()
    resources = ArenaRuntimeResources(model, "cpu")
    resources.acquire_process_owner()
    resources.canonical_committed = True
    resources.arena = SimpleNamespace(
        unguard_whole_model_to=lambda _model: calls.append("unguard"),
        release=lambda: calls.append("arena"),
    )
    resources.residency = SimpleNamespace(clear=lambda: calls.append("residency"))
    executor_close = mock.Mock(side_effect=(RuntimeError("executor boom"), None))
    resources.executor = SimpleNamespace(
        active_executions=0, close=executor_close
    )

    with pytest.raises(ArenaCleanupError, match="executor boom"):
        resources.release()
    resources.release()

    assert executor_close.call_count == 2
    assert calls == ["residency", "unguard", "arena"]
    assert resources.released
    assert resources.disposed
    assert active_process_owner() is None


def test_arena_release_failure_retains_resource_for_retry():
    model = _frozen_linear()
    resources = ArenaRuntimeResources(model, "cpu")
    resources.acquire_process_owner()
    token = resources.owner_token
    arena = SimpleNamespace(
        unguard_whole_model_to=mock.Mock(),
        release=mock.Mock(
            side_effect=(pin_manager.PinReleaseError("unregister failed"), None)
        ),
    )
    resources.canonical_committed = True
    resources.arena = arena

    with pytest.raises(ArenaCleanupError, match="unregister failed"):
        resources.release()

    assert resources.arena is arena
    assert resources.owner_token is token
    assert active_process_owner() is token
    resources.release()
    assert resources.arena is None
    assert resources.released
    assert active_process_owner() is None


def test_release_restores_arena_owned_process_globals():
    original_simulated = vram_budget.simulated_card_bytes()
    original_fp8 = fp8_grad_input_enabled()
    try:
        vram_budget.set_simulated_card_bytes(1234)
        set_fp8_grad_input_enabled(True)
        resources = ArenaRuntimeResources(_frozen_linear(), "cpu")
        resources.acquire_process_owner()
        vram_budget.set_simulated_card_bytes(5678)
        set_fp8_grad_input_enabled(False)

        resources.release()

        assert vram_budget.simulated_card_bytes() == 1234
        assert fp8_grad_input_enabled()
        assert active_process_owner() is None
    finally:
        vram_budget.set_simulated_card_bytes(original_simulated)
        set_fp8_grad_input_enabled(original_fp8)


def test_process_global_restore_failure_is_retryable():
    original_simulated = vram_budget.simulated_card_bytes()
    try:
        vram_budget.set_simulated_card_bytes(1234)
        resources = ArenaRuntimeResources(_frozen_linear(), "cpu")
        resources.acquire_process_owner()
        token = resources.owner_token
        vram_budget.set_simulated_card_bytes(5678)

        with mock.patch(
            "toolkit.memory_management.vram_budget.set_simulated_card_bytes",
            side_effect=RuntimeError("restore failed"),
        ):
            with pytest.raises(ArenaCleanupError, match="restore failed"):
                resources.release()

        assert resources.owner_token is token
        assert active_process_owner() is token
        resources.release()
        assert vram_budget.simulated_card_bytes() == 1234
        assert active_process_owner() is None
    finally:
        vram_budget.set_simulated_card_bytes(original_simulated)


def test_transfer_cleanup_failure_retains_process_owner_until_retry():
    model = _frozen_linear()
    resources = ArenaRuntimeResources(model, "cpu")
    resources.acquire_process_owner()
    token = resources.owner_token

    with mock.patch(
        "toolkit.memory_management.arena_offload.transfer.release_fetch_runtime",
        side_effect=RuntimeError("transfer cleanup failed"),
    ):
        with pytest.raises(ArenaCleanupError, match="transfer cleanup failed"):
            resources.release()

    assert active_process_owner() is token
    assert not resources.released
    resources.release()
    assert active_process_owner() is None
    assert resources.released


def test_finalize_cap_failure_is_fatal_after_runtime_publication():
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._disposed = False
    runtime._resources = SimpleNamespace(release=mock.Mock())
    failure = RuntimeError("allocator cap failed")
    runtime._bind_training_cap = mock.Mock(side_effect=failure)

    with pytest.raises(ArenaSetupFatalError) as caught:
        runtime.finalize()

    assert caught.value.__cause__ is failure
    runtime._resources.release.assert_called_once_with()


def test_base_model_device_state_routes_arena_transformer_through_runtime():
    model = SimpleNamespace(
        vae=mock.Mock(),
        unet=mock.Mock(),
        text_encoder=mock.Mock(),
        adapter=None,
        refiner_unet=None,
    )
    runtime = mock.Mock()
    state = {
        "vae": {"training": False, "device": "cpu", "requires_grad": False},
        "unet": {"training": False, "device": "cpu", "requires_grad": False},
        "text_encoder": {
            "training": False,
            "device": "cpu",
            "requires_grad": False,
        },
    }

    with mock.patch(
        "toolkit.memory_management.runtime.get_memory_runtime",
        return_value=runtime,
    ), mock.patch("toolkit.models.base_model.flush"):
        BaseModel.set_device_state(model, state)

    runtime.place_permanent_modules.assert_called_once_with(torch.device("cpu"))
    runtime.park_residency_for_external_phase.assert_called_once_with()
    model.unet.to.assert_not_called()
    model.unet.requires_grad_.assert_not_called()


def test_arena_external_phase_parks_and_restores_exact_training_plan():
    original = ResidencyPlan.build(
        "train", (("blocks.0", "linear"), ("blocks.1", "linear"))
    )
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._disposed = False
    runtime._device_state_parked_plan = None
    runtime._residency = SimpleNamespace(plan=original)
    runtime._executor = SimpleNamespace(
        TRAIN="train",
        finalized=True,
        set_residency_plan=mock.Mock(),
    )
    runtime._training_plan = original

    runtime.park_residency_for_external_phase()
    parked = runtime._executor.set_residency_plan.call_args.args[0]
    assert parked.phase == "train"
    assert parked.resident_leaf_keys == frozenset()

    runtime.restore_residency_after_external_phase()
    assert runtime._executor.set_residency_plan.call_args.args[0] is original
    assert runtime._device_state_parked_plan is None


def test_base_model_restores_arena_only_after_text_encoder_offload():
    events = []
    text_encoder = mock.Mock()
    text_encoder.to.side_effect = lambda device: events.append(("text", device))
    runtime = mock.Mock()
    runtime.restore_residency_after_external_phase.side_effect = (
        lambda: events.append(("arena", "restore"))
    )
    model = SimpleNamespace(
        vae=mock.Mock(),
        unet=mock.Mock(),
        text_encoder=text_encoder,
        adapter=None,
        refiner_unet=None,
    )
    state = {
        "vae": {"training": False, "device": "cpu", "requires_grad": False},
        "unet": {
            "training": True,
            "device": torch.device("cuda"),
            "requires_grad": False,
        },
        "text_encoder": {
            "training": False,
            "device": "cpu",
            "requires_grad": False,
        },
    }

    with mock.patch(
        "toolkit.memory_management.runtime.get_memory_runtime",
        return_value=runtime,
    ), mock.patch("toolkit.models.base_model.flush"):
        BaseModel.set_device_state(model, state)

    assert events == [("text", "cpu"), ("arena", "restore")]


def test_base_model_text_cache_preset_activates_only_text_encoder():
    model = SimpleNamespace(
        save_device_state=mock.Mock(),
        set_device_state=mock.Mock(),
        vae=mock.Mock(),
        unet=mock.Mock(),
        text_encoder=mock.Mock(),
        adapter=None,
        refiner_unet=None,
        vae_device_torch=torch.device("cuda"),
        device_torch=torch.device("cuda"),
        te_device_torch=torch.device("cuda"),
    )

    BaseModel.set_device_state_preset(model, "cache_text_encoder")

    state = model.set_device_state.call_args.args[0]
    assert state["vae"]["device"] == "cpu"
    assert state["unet"]["device"] == "cpu"
    assert state["text_encoder"]["device"] == torch.device("cuda")


def test_accelerator_preparation_skips_arena_managed_transformer():
    unet = mock.Mock()
    accelerator = mock.Mock()
    accelerator.prepare.side_effect = lambda value, **_kwargs: value
    process = SimpleNamespace(
        accelerator=accelerator,
        sd=SimpleNamespace(
            vae=mock.Mock(),
            unet=unet,
            text_encoder=None,
            refiner_unet=None,
            network=None,
        ),
        train_config=SimpleNamespace(
            train_text_encoder=False,
            train_refiner=False,
        ),
        modules_being_trained=[],
        adapter=None,
        optimizer=mock.Mock(),
        lr_scheduler=None,
    )

    with mock.patch(
        "jobs.process.BaseSDTrainProcess.get_memory_runtime",
        return_value=mock.Mock(),
    ):
        BaseSDTrainProcess.prepare_accelerator(process)

    assert not any(
        call.args and call.args[0] is unet
        for call in accelerator.prepare.mock_calls
    )
    assert process.modules_being_trained == [unet]


def test_process_cleanup_retains_runtime_after_failure_and_retries():
    runtime = mock.Mock()
    runtime.close.side_effect = (RuntimeError("close failed"), None)
    process = SimpleNamespace(
        _cleanup_in_progress=False,
        _cleanup_completed=False,
        _arena_runtime=runtime,
        sd=SimpleNamespace(unet=mock.Mock(), text_encoder=None),
    )

    with mock.patch(
        "jobs.process.BaseSDTrainProcess.close_memory_runtime_preparation"
    ), mock.patch("jobs.process.BaseSDTrainProcess.MemoryManager.detach"):
        with pytest.raises(RuntimeError, match="close failed"):
            BaseSDTrainProcess.cleanup(process)
        assert process._arena_runtime is runtime
        assert not process._cleanup_completed

        BaseSDTrainProcess.cleanup(process)

    assert process._arena_runtime is None
    assert process._cleanup_completed
    assert runtime.close.call_count == 2


def test_sequential_success_cleanup_returns_global_owners_to_baseline():
    model = _frozen_linear()
    resources = ArenaRuntimeResources(model, "cpu")
    resources.acquire_process_owner()
    runtime = SimpleNamespace(close=resources.release)
    text_encoder = torch.nn.Linear(4, 4)
    text_encoder._memory_manager = SimpleNamespace(unmanaged_modules=[])
    process = SimpleNamespace(
        _cleanup_in_progress=False,
        _cleanup_completed=False,
        _arena_runtime=runtime,
        sd=SimpleNamespace(unet=model, text_encoder=text_encoder),
    )
    pin_baseline = pin_manager.total_pinned_bytes()
    fake_cuda_device = torch.device("cuda")
    _DEVICE_STATE[fake_cuda_device] = object()

    with mock.patch("torch.cuda.empty_cache"):
        BaseSDTrainProcess.cleanup(process)

    assert active_process_owner() is None
    assert process._arena_runtime is None
    assert process._cleanup_completed
    assert not hasattr(text_encoder, "_memory_manager")
    assert fake_cuda_device not in _DEVICE_STATE
    assert pin_manager.total_pinned_bytes() == pin_baseline


def test_base_job_retains_processes_until_retryable_cleanup_succeeds():
    process = SimpleNamespace(job=object(), cleanup=mock.Mock())
    process.cleanup.side_effect = (RuntimeError("retry me"), None)
    job = BaseJob.__new__(BaseJob)
    job.process = [process]

    with pytest.raises(RuntimeError, match="retry me"):
        job.cleanup()
    assert job.process == [process]
    assert process.job is not None

    job.cleanup()
    assert job.process == []
    assert process.job is None


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.add(base)
            imported.update(
                f"{base}.{alias.name}" if base else alias.name
                for alias in node.names
            )
    return imported


def test_backend_import_and_shared_private_state_boundaries():
    root = Path(__file__).parents[1]
    arena_imports = set()
    for path in (root / "toolkit" / "memory_management" / "arena_offload").glob(
        "*.py"
    ):
        arena_imports.update(_imported_modules(path))
    assert not any(
        name == "manager"
        or name.endswith(".manager")
        or name == "manager_modules"
        or name.endswith(".manager_modules")
        for name in arena_imports
    )

    for relative in (
        "jobs/process/BaseSDTrainProcess.py",
        "extensions_built_in/sd_trainer/SDTrainer.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Attribute) and node.attr.startswith("_mm_")
            for node in ast.walk(tree)
        )

    trainer_source = (
        root / "jobs" / "process" / "BaseSDTrainProcess.py"
    ).read_text(encoding="utf-8")
    assert "if runtime_owns_block_compile:" in trainer_source
    assert "if runtime_owns_block_compile and block_compile:" not in trainer_source


def test_legacy_manager_does_not_import_arena_backend():
    root = Path(__file__).parents[1]
    imports = _imported_modules(
        root / "toolkit" / "memory_management" / "manager.py"
    )
    assert not any("arena_offload" in name for name in imports)
