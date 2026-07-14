import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from toolkit.memory_management.arena_offload import (
    ArenaOffloadConfig,
    ArenaSetupFatalError,
    prepare_canonical_storage,
    prepare_arena_offload,
)
from toolkit.memory_management.arena_offload.errors import (
    ArenaCleanupError,
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


def test_postcommit_failure_is_fatal_disposes_and_releases_owner():
    model = _frozen_linear()
    original_error = RuntimeError("residency construction failed")
    with mock.patch(
        "toolkit.memory_management.arena_offload.runtime.build_training_plan",
        side_effect=original_error,
    ):
        with pytest.raises(ArenaSetupFatalError) as caught:
            prepare_arena_offload(
                model,
                device="cpu",
                block_names=("blocks",),
                config=ArenaOffloadConfig(enabled=True),
            )

    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = caught.value
    assert caught.value.__cause__ is original_error
    assert is_fatal_arena_setup(wrapper)
    assert not recover_allows_next_job(wrapper, True)
    assert recover_allows_next_job(RuntimeError("ordinary"), True)
    assert active_process_owner() is None
    assert model._arena_offload_disposed
    with pytest.raises(RuntimeError, match="transformer_disposed"):
        model(torch.randn(1, 4))
    with pytest.raises(RuntimeError, match="transformer_disposed"):
        model.to("cpu")


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
    resources.executor = SimpleNamespace(
        active_executions=0,
        close=lambda: (_ for _ in ()).throw(RuntimeError("executor boom")),
    )

    with pytest.raises(ArenaCleanupError, match="executor boom"):
        resources.release()
    resources.release()

    assert calls == ["residency", "unguard", "arena"]
    assert resources.released
    assert resources.disposed
    assert active_process_owner() is None


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


def test_phase7_import_and_private_state_boundaries():
    root = Path(__file__).parents[1]
    arena_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "toolkit" / "memory_management" / "arena_offload").glob("*.py")
    )
    assert "from ..manager import" not in arena_sources
    assert "manager_modules" not in arena_sources

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

def test_phase8_legacy_manager_has_no_arena_execution_bridge():
    root = Path(__file__).parents[1]
    manager = (
        root / "toolkit" / "memory_management" / "manager.py"
    ).read_text(encoding="utf-8")
    for obsolete in (
        "attach_smart_training_immutable",
        "smart_immutable",
        "_mm_immutable_",
        "_immutable_runtime",
        "_mm_canonical_leaf",
        "canonical_relief",
    ):
        assert obsolete not in manager
