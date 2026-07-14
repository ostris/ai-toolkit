"""Opaque physical storage plus quantization-owned state substitution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class TensorStorageBinding:
    name: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class StateSubstitutionBinding:
    """One module-state target reconstructed from declared physical tensors."""

    name: str
    tensor_indices: tuple[int, ...]
    reconstruct: Callable[[tuple[torch.Tensor, ...]], torch.Tensor]


@dataclass(frozen=True)
class _TensorSubclassReconstruction:
    cls: type
    names: tuple[str, ...]
    context: object
    size: tuple[int, ...]
    stride: tuple[int, ...]

    def __call__(self, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.cls.__tensor_unflatten__(
            dict(zip(self.names, tensors, strict=True)),
            self.context,
            torch.Size(self.size),
            self.stride,
        )


def _identity(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(tensors) != 1:
        raise ValueError("storage_identity_requires_one_tensor")
    return tensors[0]


@dataclass(frozen=True)
class LayerStorageBinding:
    tensors: tuple[TensorStorageBinding, ...]
    execution_key: tuple
    weight_leaf_count: int
    weight_template: torch.Tensor | None
    substitutions: tuple[StateSubstitutionBinding, ...]


def _flatten_named(value, prefix):
    try:
        names, _context = value.__tensor_flatten__()
    except Exception:
        return [TensorStorageBinding(prefix, value)]
    out = []
    for name in names:
        child = getattr(value, name, None)
        if child is not None:
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            out.extend(_flatten_named(child, child_prefix))
    return out


def named_tensor_storage(value, prefix: str = "") -> tuple[TensorStorageBinding, ...]:
    """Return physical tensor leaves with stable dotted source names.

    State-dict consumers can pass the state key as ``prefix`` so tensor
    subclasses such as ``weight._data`` / ``weight._scale`` are addressable
    without retaining a second flattened value table.
    """
    return tuple(_flatten_named(value, prefix))


def linear_storage_binding(weight, bias=None) -> LayerStorageBinding:
    """Describe physical storage without exposing its meaning to the mover."""
    weight_value = weight.data if isinstance(weight, torch.nn.Parameter) else weight
    bias_value = bias.data if isinstance(bias, torch.nn.Parameter) else bias
    from .fp8_linear import declare_fp8_linear, fp8_execution_key

    fp8 = declare_fp8_linear(weight_value)
    if fp8 is None:
        weight_tensors = _flatten_named(weight_value, "weight")
        execution_key = (
            type(weight_value).__module__,
            type(weight_value).__qualname__,
            tuple(
                (item.name, tuple(item.tensor.shape), str(item.tensor.dtype))
                for item in weight_tensors
            ),
        )
        weight_template = weight_value
    else:
        weight_tensors = [
            TensorStorageBinding("qdata", fp8.qdata),
            TensorStorageBinding("scale", fp8.scale),
        ]
        execution_key = fp8_execution_key(fp8.spec)
        weight_template = weight_value
    bias_tensors = [] if bias_value is None else _flatten_named(bias_value, "bias")
    tensors = tuple(weight_tensors + bias_tensors)
    if len(weight_tensors) == 1:
        weight_reconstruct = _identity
    else:
        names, context = weight_value.__tensor_flatten__()
        weight_reconstruct = _TensorSubclassReconstruction(
            cls=type(weight_value),
            names=tuple(names),
            context=context,
            size=tuple(weight_value.shape),
            stride=tuple(weight_value.stride()),
        )
    substitutions = [
        StateSubstitutionBinding(
            name="weight",
            tensor_indices=tuple(range(len(weight_tensors))),
            reconstruct=weight_reconstruct,
        )
    ]
    if bias_value is not None:
        substitutions.append(
            StateSubstitutionBinding(
                name="bias",
                tensor_indices=tuple(range(len(weight_tensors), len(tensors))),
                reconstruct=_identity,
            )
        )
    return LayerStorageBinding(
        tensors=tensors,
        execution_key=execution_key,
        weight_leaf_count=len(weight_tensors),
        weight_template=weight_template,
        substitutions=tuple(substitutions),
    )


def module_storage_binding(module: torch.nn.Module) -> LayerStorageBinding:
    """Declare storage for a supported Linear without materializing weights."""
    from toolkit.util.ostris_quant import OstrisLinear

    if not isinstance(module, OstrisLinear):
        weight = module._parameters.get("weight")
        if weight is None:
            raise ValueError(
                f"unsupported_managed_module:{type(module).__module__}."
                f"{type(module).__qualname__}"
            )
        return linear_storage_binding(weight, module._parameters.get("bias"))

    tensors = []
    substitutions = []
    for name, value in module._buffers.items():
        if value is None:
            continue
        # Ostris quantizers intentionally keep packed execution buffers out of
        # state_dict and serialize them through their own cache format. Buffer
        # persistence is a serialization policy, not an execution-storage
        # policy: every live quantizer buffer is canonical arena state.
        index = len(tensors)
        tensors.append(TensorStorageBinding(f"buffer.{name}", value))
        substitutions.append(
            StateSubstitutionBinding(name, (index,), _identity)
        )
    bias = module._parameters.get("bias")
    if bias is not None:
        index = len(tensors)
        tensors.append(TensorStorageBinding("bias", bias.data))
        substitutions.append(
            StateSubstitutionBinding("bias", (index,), _identity)
        )
    if not tensors:
        raise ValueError("ostris_linear_has_no_declared_storage")
    quantizer = module.ostris_quantizer
    execution_key = (
        "ostris",
        type(quantizer).__module__,
        type(quantizer).__qualname__,
        str(getattr(quantizer, "qtype", None)),
        tuple(
            (item.name, tuple(item.tensor.shape), str(item.tensor.dtype))
            for item in tensors
        ),
    )
    return LayerStorageBinding(
        tensors=tuple(tensors),
        execution_key=execution_key,
        weight_leaf_count=0,
        weight_template=None,
        substitutions=tuple(substitutions),
    )


def temporary_materialization_bytes(value, dtype=torch.bfloat16) -> int:
    """Return scratch bytes needed when wrapped storage must be materialized."""
    value = value.data if isinstance(value, torch.nn.Parameter) else value
    binding = linear_storage_binding(value)
    if binding.weight_leaf_count <= 1:
        return 0
    return int(value.numel() * torch.empty((), dtype=dtype).element_size())
