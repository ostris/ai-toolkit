from types import SimpleNamespace
from unittest import mock

from toolkit.memory_management.arena_offload.planner import GIB, build_training_plan


class _Arena:
    def __init__(self, records):
        self._records = {record.block_key: record for record in records}

    def block_keys(self):
        return tuple(self._records)

    def block_record(self, key):
        return self._records[key]


def _record(key, committed_gib):
    return SimpleNamespace(
        block_key=key,
        committed_bytes=int(committed_gib * GIB),
        modules=(object(),),
        leaf_names=("weight",),
    )


def _config():
    return SimpleNamespace(
        _policy=SimpleNamespace(
            working_reserve_gib=-1,
            wddm_hard_gib=1.0,
            wddm_margin_gib=1.0,
            checkpoint_keep_last=0,
            prefetch_depth=2,
        )
    )


def test_auto_plan_uses_all_resident_fast_path_when_complete_model_fits():
    records = [_record(f"blocks.{index}", 2.0) for index in range(3)]
    with (
        mock.patch(
            "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
            return_value=(12 * GIB, 12 * GIB),
        ),
        mock.patch(
            "toolkit.memory_management.arena_offload.planner._singleton_stats",
            return_value=(1 * GIB, 0, set()),
        ),
    ):
        plan = build_training_plan(
            SimpleNamespace(), _Arena(records), (), "cuda", _config()
        )

    assert plan["all_resident_fit"]
    assert plan["offloaded_layers"] == 0
    assert plan["generic_resident_bytes"] == 6 * GIB
    assert plan["ring_bytes"] == 0
    assert plan["working_reserve_bytes"] == 4 * GIB


def test_auto_plan_keeps_streaming_when_complete_model_does_not_fit():
    records = [_record(f"blocks.{index}", 2.0) for index in range(3)]
    with (
        mock.patch(
            "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
            return_value=(10 * GIB, 12 * GIB),
        ),
        mock.patch(
            "toolkit.memory_management.arena_offload.planner._singleton_stats",
            return_value=(1 * GIB, 0, set()),
        ),
    ):
        plan = build_training_plan(
            SimpleNamespace(), _Arena(records), (), "cuda", _config()
        )

    assert not plan["all_resident_fit"]
    assert plan["offloaded_layers"] > 0
    assert plan["ring_bytes"] > 0
    assert plan["working_reserve_bytes"] == 5 * GIB


def test_explicit_working_reserve_controls_all_resident_fit():
    records = [_record(f"blocks.{index}", 2.0) for index in range(3)]
    config = _config()
    config._policy.working_reserve_gib = 2.0
    with (
        mock.patch(
            "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
            return_value=(10 * GIB, 12 * GIB),
        ),
        mock.patch(
            "toolkit.memory_management.arena_offload.planner._singleton_stats",
            return_value=(1 * GIB, 0, set()),
        ),
    ):
        plan = build_training_plan(
            SimpleNamespace(), _Arena(records), (), "cuda", config
        )

    assert plan["all_resident_fit"]
    assert plan["offloaded_layers"] == 0
    assert plan["working_reserve_bytes"] == 2 * GIB
