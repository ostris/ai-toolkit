import os

import pytest
import torch

from toolkit.compile_utils import configure_cuda_only_inductor
from toolkit.lora_special import LoRAModule
from toolkit.models.DoRA import DoRAModule
from toolkit.models.lokr import LokrModule


class _Network:
    network_type = "lora"


def _linear():
    return torch.nn.Linear(8, 8, bias=False)


def test_cuda_only_inductor_keeps_compile_errors_visible():
    configure_cuda_only_inductor()

    assert torch._dynamo.config.suppress_errors is False
    if os.name == "nt":
        from torch._inductor import config as inductor_config

        assert inductor_config.cpp.vec_isa_ok is False


def test_cuda_compile_prefers_aot_eager_then_compile(monkeypatch):
    stances = []
    monkeypatch.setattr(
        torch.compiler,
        "set_stance",
        lambda stance: stances.append(stance),
    )

    configure_cuda_only_inductor()

    assert stances == ["aot_eager_then_compile"]


def test_lora_tensor_alpha_uses_device_owned_runtime_scale():
    module = LoRAModule(
        "compile_scalar",
        _linear(),
        lora_dim=4,
        alpha=torch.tensor(8, dtype=torch.bfloat16),
        network=_Network(),
    )

    assert type(module.scale) is float
    assert module.scale == 2.0
    assert module._runtime_scale.item() == 2.0
    assert "_runtime_scale" not in module.state_dict()
    assert not hasattr(module, "scalar")


def test_dora_tensor_alpha_uses_device_owned_runtime_scale():
    module = DoRAModule(
        "compile_scalar_dora",
        _linear(),
        lora_dim=4,
        alpha=torch.tensor(8, dtype=torch.bfloat16),
        network=_Network(),
    )

    assert type(module.scale) is float
    assert module.scale == 2.0
    assert module._runtime_scale.item() == 2.0
    assert "_runtime_scale" not in module.state_dict()
    assert not hasattr(module, "scalar")


def test_lokr_tensor_alpha_uses_device_owned_runtime_scale():
    module = LokrModule(
        "compile_scalar_lokr",
        _linear(),
        lora_dim=2,
        alpha=torch.tensor(4, dtype=torch.bfloat16),
        network=_Network(),
    )

    assert type(module.scale) is float
    assert module.scale == 1.0
    assert module._runtime_scale.item() == 1.0
    assert "_runtime_scale" not in module.state_dict()
    assert module.get_weight().device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_functional_call_compiles_lora_without_cpu_codegen():
    configure_cuda_only_inductor()
    network = _Network()
    network.is_lorm = False
    network.is_active = True
    network.is_merged_in = False
    network._multiplier = 1.0
    network.torch_multiplier = torch.ones(1, device="cuda")
    original = torch.nn.Linear(
        8, 8, bias=False, device="cuda", dtype=torch.bfloat16
    )
    module = LoRAModule(
        "compile_scalar_cuda",
        original,
        lora_dim=4,
        alpha=torch.tensor(8, dtype=torch.bfloat16),
        network=network,
    ).to("cuda")
    module.org_forward = original.forward
    state = dict(module.named_parameters())
    state.update(module.named_buffers())

    def kernel(value):
        return torch.func.functional_call(
            module,
            state,
            (value,),
            strict=False,
            tie_weights=False,
        )

    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    value = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
    result = compiled(value)
    result.square().mean().backward()
    torch.cuda.synchronize()

    assert module._runtime_scale.device.type == "cuda"
    assert result.device.type == "cuda"
    assert module.lora_down.weight.grad is not None
    assert module.lora_up.weight.grad is not None
