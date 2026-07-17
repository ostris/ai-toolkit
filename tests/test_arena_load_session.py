from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from toolkit.memory_management.arena_offload import model_load_arena_session
from toolkit.memory_management.arena_offload.load_session import (
    PENDING_CANONICAL_BUILD_ATTR,
    claim_pending_canonical_build,
)
from toolkit.memory_management.arena_offload.construction import (
    CanonicalBuildError,
    CanonicalStateConsumedError,
    PreparedCanonicalBuild,
)
from toolkit.memory_management.runtime import close_memory_runtime_preparation


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])
        self.head = torch.nn.Linear(4, 4)


def base_model():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            layer_offloading=True,
            layer_offloading_smart=True,
        ),
        device_torch=torch.device("cpu"),
        te_only=False,
        get_transformer_block_names=lambda: ["blocks"],
    )


def frozen_transformer():
    model = Transformer()
    model.requires_grad_(False)
    return model


def test_common_load_state_dict_uses_generic_arena_session():
    source = frozen_transformer()
    expected_head = source.head.weight.detach().clone()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = frozen_transformer()

    with model_load_arena_session(base_model()):
        incompatible = target.load_state_dict(state, strict=True, assign=True)

    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert "blocks.0.linear.weight" not in state
    torch.testing.assert_close(target.head.weight, expected_head)
    build = claim_pending_canonical_build(target)
    assert build is not None
    build.rollback()


def test_unfinished_generic_load_is_released_by_shared_cleanup():
    owner = base_model()
    source = frozen_transformer()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = frozen_transformer()

    with model_load_arena_session(owner):
        target.load_state_dict(state, strict=True, assign=True)

    assert hasattr(target, PENDING_CANONICAL_BUILD_ATTR)
    close_memory_runtime_preparation(owner)
    assert not hasattr(target, PENDING_CANONICAL_BUILD_ATTR)


def test_trainable_target_falls_back_to_normal_assignment():
    source = Transformer()
    expected = source.blocks[0].linear.weight.detach().clone()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = Transformer()

    with model_load_arena_session(base_model()) as session:
        target.load_state_dict(state, strict=True, assign=True)

    assert session.unsupported_reason is not None
    assert not hasattr(target, PENDING_CANONICAL_BUILD_ATTR)
    torch.testing.assert_close(target.blocks[0].linear.weight, expected)


def test_first_block_preparation_failure_keeps_mapping_and_falls_back():
    source = frozen_transformer()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    original_keys = tuple(state)
    target = frozen_transformer()

    with mock.patch.object(
        PreparedCanonicalBuild,
        "add_block",
        side_effect=CanonicalBuildError("first block failed"),
    ):
        with model_load_arena_session(base_model()) as session:
            incompatible = target.load_state_dict(state, strict=True, assign=True)

    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert tuple(state) == original_keys
    assert session.unsupported_reason == "first block failed"


def test_failure_after_first_consumed_block_never_falls_back():
    source = frozen_transformer()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = frozen_transformer()
    original = PreparedCanonicalBuild.copy_state_entry
    calls = 0

    def fail_in_second_block(build, source_key, value):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise CanonicalBuildError("second block failed")
        return original(build, source_key, value)

    with mock.patch.object(
        PreparedCanonicalBuild,
        "copy_state_entry",
        fail_in_second_block,
    ):
        with model_load_arena_session(base_model()):
            with pytest.raises(CanonicalStateConsumedError) as caught:
                target.load_state_dict(state, strict=True, assign=True)

    assert isinstance(caught.value.__cause__, CanonicalBuildError)
    assert "blocks.0.linear.weight" not in state
    assert "blocks.1.linear.weight" in state


def test_final_population_failure_after_consumption_never_falls_back():
    source = frozen_transformer()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = frozen_transformer()
    original = PreparedCanonicalBuild._finish_population

    def fail_after_pin(build):
        original(build)
        raise CanonicalBuildError("final pin failed")

    with mock.patch.object(
        PreparedCanonicalBuild, "_finish_population", fail_after_pin
    ):
        with model_load_arena_session(base_model()):
            with pytest.raises(CanonicalStateConsumedError):
                target.load_state_dict(state, strict=True, assign=True)

    assert "blocks.0.linear.weight" not in state
    assert "blocks.1.linear.weight" not in state
