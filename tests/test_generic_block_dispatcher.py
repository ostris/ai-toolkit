from types import MethodType
from unittest import mock
from dataclasses import replace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from toolkit.memory_management.arena_offload import (
    ArenaOffloadConfig,
    close_arena_offload,
    prepare_arena_offload,
)
from toolkit.memory_management.arena_offload.discovery import (
    BlockDiscoveryError,
    discover_blocks,
)
from toolkit.memory_management.arena_offload.dispatcher import (
    _first_output_tensor,
    _first_tensor_argument,
    _replace_tensor_argument,
)
from toolkit.memory_management.arena_offload.ownership import active_process_owner
from toolkit.memory_management.residency import ResidencyPlan
from toolkit.memory_management.runtime import get_memory_runtime


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, value):
        return torch.nn.functional.silu(self.proj(value))


class _Transformer(torch.nn.Module):
    def __init__(self, count=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(count)])
        self.gradient_checkpointing = False
        self._checkpoint_keep_last = 0

    def enable_gradient_checkpointing(self, keep_last=0):
        self.gradient_checkpointing = True
        self._checkpoint_keep_last = int(keep_last)

    def forward(self, value):
        cutoff = len(self.blocks) - self._checkpoint_keep_last
        for index, block in enumerate(self.blocks):
            if self.gradient_checkpointing and torch.is_grad_enabled() and index < cutoff:
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        return value


def _frozen_transformer():
    model = _Transformer()
    model.requires_grad_(False)
    return model


def _fp8_transformer(device, count=3, width=32, *, qtype="float8"):
    from optimum.quanto import freeze

    from toolkit.util.quantize import get_qtype, quantize

    class Fp8Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(width, width, bias=False)

        def forward(self, value):
            return torch.nn.functional.silu(self.proj(value))

    class Fp8Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(width, width, bias=False)
            self.blocks = torch.nn.ModuleList(Fp8Block() for _ in range(count))
            self.gradient_checkpointing = True
            self._checkpoint_keep_last = 1

        def forward(self, value):
            value = self.first(value)
            cutoff = len(self.blocks) - self._checkpoint_keep_last
            for index, block in enumerate(self.blocks):
                if torch.is_grad_enabled() and index < cutoff:
                    value = checkpoint(block, value, use_reentrant=False)
                else:
                    value = block(value)
            return value

    model = Fp8Transformer().to(device=device, dtype=torch.bfloat16)
    quantize(model, weights=get_qtype(qtype))
    if qtype in ("float8", "qfloat8"):
        freeze(model)
    model.requires_grad_(False)
    return model


def _fp8_runtime(model, device, *, forward, backward, compile_blocks):
    config = ArenaOffloadConfig(
        enabled=True,
        fp8_forward=forward,
        fp8_backward=backward,
        compile_blocks=compile_blocks,
        _compile_dynamic=False,
    )
    config = replace(
        config,
        _policy=replace(
            config._policy,
            working_reserve_gib=0.0,
            physical_vram_headroom_gib=0.0,
            wddm_hard_gib=1.0,
            checkpoint_keep_last=1,
        ),
    )
    with mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
        return_value=(1 * 1024**3, 12 * 1024**3),
    ):
        return prepare_arena_offload(
            model,
            device=device,
            block_names=("blocks",),
            config=config,
        )


def _fp8_train_once(model, runtime, device, step, network=None):
    import contextlib

    value = torch.randn(
        2, 3, 32, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    network_context = network if network is not None else contextlib.nullcontext()
    with runtime.training_step(shape_key=(2, 3, 32), step_num=step), network_context:
        output = model(value)
        output.float().square().mean().backward()
    assert value.grad is not None
    if network is not None:
        assert all(parameter.grad is not None for parameter in network.parameters())
    return output.detach()


class _AdapterBaseModel:
    arch = "synthetic"
    use_old_lokr_format = False

    def __init__(self, device, dtype):
        self.device_torch = torch.device(device)
        self.torch_dtype = dtype

    def get_transformer_block_names(self):
        return None


def _apply_lora(model, device, dtype):
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork

    config = NetworkConfig(
        type="lora", linear=8, linear_alpha=8.0, transformer_only=True
    )
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=model,
        lora_dim=8,
        multiplier=1.0,
        alpha=8.0,
        train_unet=True,
        train_text_encoder=False,
        network_config=config,
        network_type=config.type,
        transformer_only=True,
        is_transformer=True,
        target_lin_modules=[model.__class__.__name__],
        base_model=_AdapterBaseModel(device, dtype),
    )
    network.force_to(device, dtype=torch.float32)
    network._update_torch_multiplier()
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.can_merge_in = False
    network.prepare_grad_etc(None, model)
    return network


def test_declared_container_discovery_accounts_all_block_state():
    model = _frozen_transformer()
    selection = discover_blocks(model, container_paths=("blocks",))
    assert selection.container_paths == ("blocks",)
    assert selection.block_keys == ("blocks.0", "blocks.1", "blocks.2")
    assert all(len(entries) == 1 for entries in selection.entries_by_block.values())
    assert selection.accounting.managed_entries == 6
    assert selection.accounting.managed_bytes > 0


def test_dispatcher_abi_accepts_keyword_inputs_and_structured_outputs():
    hidden = torch.randn(2, 4, requires_grad=True)
    mask = torch.ones(2, 4, dtype=torch.bool)
    args = ("metadata", mask)
    kwargs = {"inputs": {"hidden_states": hidden}, "mask": None}

    selected, location = _first_tensor_argument(args, kwargs)
    assert selected is hidden
    replacement = hidden + 1
    updated_args, updated_kwargs = _replace_tensor_argument(
        args, kwargs, location, replacement
    )
    assert updated_args == args
    assert updated_kwargs["inputs"]["hidden_states"] is replacement

    output = ({"hidden_states": replacement}, (None, hidden))
    assert _first_output_tensor(output) is replacement


def test_shared_managed_state_is_rejected_before_construction():
    model = _frozen_transformer()
    shared = model.blocks[0].proj.weight
    model.blocks[1].proj.weight = shared
    with pytest.raises(BlockDiscoveryError, match="shared_managed"):
        discover_blocks(model, container_paths=("blocks",))


@pytest.mark.parametrize("view_kind", ("exact", "partial", "disjoint"))
def test_same_block_shared_managed_storage_is_rejected(view_kind):
    model = _frozen_transformer()
    model.blocks[0].other = torch.nn.Linear(4, 4, bias=False)
    storage = torch.randn(32)
    if view_kind == "exact":
        left = storage[:16]
        right = storage[:16]
    elif view_kind == "partial":
        left = storage[:16]
        right = storage[4:20]
    else:
        left = storage[:16]
        right = storage[16:32]
    model.blocks[0].proj.weight = torch.nn.Parameter(
        left.view(4, 4), requires_grad=False
    )
    model.blocks[0].other.weight = torch.nn.Parameter(
        right.view(4, 4), requires_grad=False
    )

    with pytest.raises(BlockDiscoveryError, match="shared_managed_storage"):
        discover_blocks(model, container_paths=("blocks",))


def test_checkpointing_rejection_precedes_canonical_commit():
    model = _frozen_transformer()
    original = model.blocks[0].proj.weight
    with pytest.raises(ValueError, match="requires model gradient checkpointing"):
        prepare_arena_offload(
            model,
            device="cpu",
            block_names=("blocks",),
            config=ArenaOffloadConfig(enabled=True),
        )
    assert model.blocks[0].proj.weight is original
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_runtime")


def test_sampling_callbacks_bracket_one_complete_transformer_forward():
    model = _frozen_transformer()
    model.enable_gradient_checkpointing(keep_last=1)
    with mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
        return_value=(8 * 1024**3, 12 * 1024**3),
    ), mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget."
        "auto_physical_vram_headroom_gib",
        return_value=1.0,
    ):
        config = ArenaOffloadConfig(enabled=True, compile_blocks=False)
        config = replace(
            config,
            _policy=replace(config._policy, checkpoint_keep_last=1),
        )
        runtime = prepare_arena_offload(
            model,
            device="cpu",
            block_names=("blocks",),
            config=config,
        )
    try:
        runtime.finalize()
        events = []
        runtime._executor.set_sampling_forward_callbacks(
            lambda: events.append("begin"),
            lambda: events.append("end"),
        )
        runtime._executor.activate(
            runtime._executor.SAMPLE,
            ResidencyPlan.build(runtime._executor.SAMPLE, ()),
        )
        promotions = []
        runtime._executor.set_residency_promotion_callback(
            lambda nbytes, plan: promotions.append((nbytes, plan.phase))
        )
        resident = [
            (block_key, leaf_name)
            for block_key in runtime._arena.block_keys()
            for leaf_name in runtime._arena.block_record(block_key).leaf_names
        ]
        runtime._executor.activate(
            runtime._executor.SAMPLE,
            ResidencyPlan.build(runtime._executor.SAMPLE, resident),
        )
        with torch.no_grad(), runtime._executor.execution(
            runtime._executor.SAMPLE
        ):
            model(torch.randn(2, 4))
        assert events == ["begin", "end"]
        assert promotions
        assert promotions[-1][0] > 0
        assert promotions[-1][1] == runtime._executor.SAMPLE
    finally:
        close_arena_offload(model)


def test_sampling_dispatch_retries_one_failed_compiled_block_in_eager_wrapper():
    model = _frozen_transformer()
    model.enable_gradient_checkpointing(keep_last=1)
    with mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
        return_value=(8 * 1024**3, 12 * 1024**3),
    ), mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget."
        "auto_physical_vram_headroom_gib",
        return_value=1.0,
    ):
        config = ArenaOffloadConfig(enabled=True, compile_blocks=False)
        config = replace(
            config,
            _policy=replace(config._policy, checkpoint_keep_last=1),
        )
        runtime = prepare_arena_offload(
            model,
            device="cpu",
            block_names=("blocks",),
            config=config,
        )
    try:
        runtime.finalize()
        executor = runtime._executor
        original_get_kernel = executor._get_dispatch_kernel
        attempts = []

        def flaky_get_kernel(index):
            kernel = original_get_kernel(index)

            def flaky(*args, **kwargs):
                attempts.append(index)
                if len(attempts) == 1:
                    raise torch.OutOfMemoryError("synthetic capped OOM")
                return kernel(*args, **kwargs)

            return flaky

        executor._get_dispatch_kernel = flaky_get_kernel
        recoveries = []
        executor.set_sampling_forward_callbacks(
            allocation_failure=lambda error: recoveries.append(error) or True
        )
        resident = [
            (block_key, leaf_name)
            for block_key in runtime._arena.block_keys()
            for leaf_name in runtime._arena.block_record(block_key).leaf_names
        ]
        executor.activate(
            executor.SAMPLE,
            ResidencyPlan.build(executor.SAMPLE, resident),
        )

        with torch.no_grad(), executor.execution(executor.SAMPLE):
            output = model(torch.randn(2, 4))

        assert output.shape == (2, 4)
        assert len(recoveries) == 1
        assert isinstance(recoveries[0], torch.OutOfMemoryError)
        assert attempts[:2] == [0, 0]
    finally:
        close_arena_offload(model)


def test_saved_installed_forward_checkpoint_backward_and_teardown():
    torch.manual_seed(17)
    model = _frozen_transformer()
    model.enable_gradient_checkpointing(keep_last=1)
    reference_input = torch.randn(2, 4)
    reference = model(reference_input).detach()
    with mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
        return_value=(8 * 1024**3, 12 * 1024**3),
    ), mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget."
        "auto_physical_vram_headroom_gib",
        return_value=1.0,
    ):
        config = ArenaOffloadConfig(enabled=True, compile_blocks=False)
        config = replace(
            config,
            _policy=replace(config._policy, checkpoint_keep_last=1),
        )
        runtime = prepare_arena_offload(
            model,
            device="cpu",
            block_names=("blocks",),
            config=config,
        )
    installed = []
    adapters = []
    for block in model.blocks:
        saved = block.forward
        block.adapter_gain = torch.nn.Parameter(torch.zeros(()))

        def installed_forward(self, value, _saved=saved):
            return _saved(value) + self.adapter_gain * value

        bound = MethodType(installed_forward, block)
        block.forward = bound
        installed.append(bound)
        adapters.append(block.adapter_gain)

    runtime.finalize()
    diagnostics = runtime.diagnostics()
    accounting = diagnostics["accounting"]
    assert diagnostics["checkpoint_owner"] == "model"
    assert diagnostics["state_audit"]["managed_entries"] == 6
    assert accounting["payload_reconciled"]
    assert accounting["canonical_payload_bytes"] == (
        accounting["canonical_resident_payload_bytes"]
        + accounting["streamed_payload_bytes"]
    )
    assert accounting["protected_training_blocks"] == ("blocks.2",)
    assert accounting["protected_training_blocks_resident"]
    with pytest.raises(RuntimeError, match="outside_transformer_execution"):
        model.blocks[0](torch.randn(2, 4))

    value = reference_input.detach().clone().requires_grad_(True)
    with runtime.training_step(shape_key=(2, 4), step_num=1):
        output = model(value)
        output.sum().backward()
    torch.testing.assert_close(output.detach(), reference)
    assert value.grad is not None
    assert all(parameter.grad is not None for parameter in adapters)

    protected = runtime._executor.protected_training_leaf_keys
    assert any(block == "blocks.2" for block, _leaf in protected)
    with pytest.raises(RuntimeError, match="protected_training_block"):
        runtime.transition_training_block("blocks.2", resident=False)

    close_arena_offload(model)
    assert all(
        block.forward is saved
        for block, saved in zip(model.blocks, installed, strict=True)
    )
    assert get_memory_runtime(model) is None
    assert model._arena_offload_disposed
    assert active_process_owner() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_streamed_compiled_train_sample_train():
    import toolkit.memory_management.arena_offload.transfer as transfer

    torch.manual_seed(23)
    device = torch.device("cuda")
    model = _frozen_transformer().to(device)
    model.enable_gradient_checkpointing(keep_last=1)
    config = ArenaOffloadConfig(
        enabled=True,
        compile_blocks=True,
        _compile_dynamic=False,
    )
    config = replace(
        config,
        _policy=replace(
            config._policy,
            working_reserve_gib=0.0,
            physical_vram_headroom_gib=0.0,
            wddm_hard_gib=1.0,
            checkpoint_keep_last=1,
            prefetch_depth=1,
        ),
    )
    with mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
        # Force a mixed plan: two blocks fit, all three do not.
        return_value=(700, 12 * 1024**3),
    ):
        runtime = prepare_arena_offload(
            model,
            device=device,
            block_names=("blocks",),
            config=config,
        )
    adapters = []
    for block in model.blocks:
        saved = block.forward
        block.adapter_gain = torch.nn.Parameter(torch.zeros((), device=device))

        def installed_forward(self, value, _saved=saved):
            return _saved(value) + self.adapter_gain * value

        block.forward = MethodType(installed_forward, block)
        adapters.append(block.adapter_gain)
    runtime.finalize()
    diagnostics = runtime.diagnostics()
    accounting = diagnostics["accounting"]
    assert accounting["payload_reconciled"]
    assert accounting["mixed_residency"]
    assert accounting["resident_blocks"] >= 1
    assert accounting["streamed_blocks"] >= 1
    assert accounting["planned_training_h2d_bytes"] == (
        2 * accounting["planned_forward_h2d_bytes"]
    )
    assert accounting["protected_training_blocks_resident"]
    streamed = [
        runtime._executor.source(index).transfer is not None
        for index in range(runtime.block_count)
    ]
    assert any(streamed)
    assert not streamed[-1]
    streamed_keys = {
        f"blocks.{index}" for index, is_streamed in enumerate(streamed)
        if is_streamed
    }
    pinned_keys = set(diagnostics["canonical_pinned_block_keys"])
    assert streamed_keys <= pinned_keys
    assert len(pinned_keys - streamed_keys) <= 2

    def train_once(step, *, input_requires_grad=True):
        transfer_before = transfer.lifetime_fetch_stats()["bytes"]
        planned = runtime.diagnostics()["accounting"][
            "planned_training_h2d_bytes"
        ]
        value = torch.randn(
            2, 4, device=device, requires_grad=input_requires_grad
        )
        with runtime.training_step(shape_key=(2, 4), step_num=step):
            output = model(value)
            output.square().mean().backward()
        if input_requires_grad:
            assert value.grad is not None
        assert all(parameter.grad is not None for parameter in adapters)
        for parameter in adapters:
            parameter.grad = None
        assert transfer.lifetime_fetch_stats()["bytes"] - transfer_before == planned
        return output.detach()

    first = train_once(1)
    sample_plan = ResidencyPlan.build("sample", ())
    runtime._executor.activate(runtime._executor.SAMPLE, sample_plan)
    assert runtime._arena.pinned_block_keys() == frozenset(
        f"blocks.{index}" for index in range(runtime.block_count)
    )
    sample_accounting = runtime.diagnostics()["accounting"]
    sample_transfer_before = transfer.lifetime_fetch_stats()["bytes"]
    with torch.no_grad(), runtime._executor.execution(runtime._executor.SAMPLE):
        sampled = model(torch.randn(2, 4, device=device))
    assert transfer.lifetime_fetch_stats()["bytes"] - sample_transfer_before == (
        sample_accounting["planned_forward_h2d_bytes"]
    )
    assert torch.isfinite(sampled).all()
    runtime._executor.activate(runtime._executor.TRAIN, runtime._training_plan)
    restored_pins = runtime._arena.pinned_block_keys()
    assert streamed_keys <= restored_pins
    assert len(restored_pins - streamed_keys) <= 2
    second = train_once(2)
    # ZImage enters some checkpointed block families through frozen setup
    # layers, so their first streamed block has no gradient-bearing input.
    # Two consecutive steps prove recompute returns those ring slots instead
    # of relying on a backward hook that autograd cannot schedule.
    third = train_once(3, input_requires_grad=False)
    fourth = train_once(4, input_requires_grad=False)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.isfinite(third).all()
    assert torch.isfinite(fourth).all()
    close_arena_offload(model)
    assert active_process_owner() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("qtype", "native_fp8"),
    (("qfloat8", True), ("orbit4", False)),
)
def test_cuda_retained_quantization_backends_stream_through_arena(
    qtype, native_fp8
):
    device = torch.device("cuda")
    model = _fp8_transformer(device, qtype=qtype)
    runtime = _fp8_runtime(
        model,
        device,
        forward=native_fp8,
        backward=native_fp8,
        compile_blocks=False,
    )
    try:
        runtime.finalize()
        output = _fp8_train_once(model, runtime, device, 1)
        assert torch.isfinite(output).all()
    finally:
        close_arena_offload(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fp8_gates_select_distinct_canonical_arena_paths():
    from toolkit.quantization import fp8_linear

    device = torch.device("cuda")
    real_scaled_mm = torch._scaled_mm
    real_grad_input = fp8_linear._grad_input_compute
    results = {}

    for name, forward, backward in (
        ("baseline", False, False),
        ("forward", True, False),
        ("forward_backward", True, True),
    ):
        scaled_calls = []
        grad_input_calls = []

        def counted_scaled_mm(*args, **kwargs):
            scaled_calls.append(1)
            return real_scaled_mm(*args, **kwargs)

        def counted_grad_input(*args, **kwargs):
            grad_input_calls.append(1)
            return real_grad_input(*args, **kwargs)

        model = _fp8_transformer(device)
        runtime = _fp8_runtime(
            model,
            device,
            forward=forward,
            backward=backward,
            compile_blocks=False,
        )
        try:
            with mock.patch.object(torch, "_scaled_mm", counted_scaled_mm), mock.patch.object(
                fp8_linear, "_grad_input_compute", counted_grad_input
            ):
                runtime.finalize()
                _fp8_train_once(model, runtime, device, 1)
            diagnostics = runtime.diagnostics()
            results[name] = {
                "scaled": len(scaled_calls),
                "grad_input": len(grad_input_calls),
                "canonical": diagnostics["training_fp8_canonical"],
                "singletons": diagnostics["training_fp8_singletons"],
            }
        finally:
            close_arena_offload(model)

    assert results["baseline"] == {
        "scaled": 0,
        "grad_input": 0,
        "canonical": 0,
        "singletons": 0,
    }
    assert results["forward"]["scaled"] > 0
    assert results["forward"]["grad_input"] == 0
    assert results["forward"]["canonical"] == 3
    assert results["forward"]["singletons"] == 1
    assert results["forward_backward"]["scaled"] > 0
    assert results["forward_backward"]["grad_input"] > 0
    assert results["forward_backward"]["canonical"] == 3
    assert results["forward_backward"]["singletons"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_compiled_fp8_canonical_arena_emits_scaled_mm():
    device = torch.device("cuda")
    model = _fp8_transformer(device)
    runtime = _fp8_runtime(
        model,
        device,
        forward=True,
        backward=True,
        compile_blocks=True,
    )
    network = _apply_lora(model, device, torch.bfloat16)
    try:
        runtime.finalize(network)
        _fp8_train_once(model, runtime, device, 1, network)
        for parameter in network.parameters():
            parameter.grad = None
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            _fp8_train_once(model, runtime, device, 2, network)
        scaled_mm = [
            event
            for event in profiler.key_averages()
            if "_scaled_mm" in event.key
        ]
        assert sum(event.count for event in scaled_mm) > 0
        assert runtime.diagnostics()["training_fp8_canonical"] == 3
        assert runtime.diagnostics()["training_fp8_singletons"] == 1
    finally:
        close_arena_offload(model)
        torch.cuda.empty_cache()
