"""Arena integration for the neutral row-wise FP8 execution policy."""

from __future__ import annotations

import torch

from toolkit.quantization.fp8_linear import (
    bind_parameter_operation,
    declare_fp8_linear,
)


LINEAR_MODULES = {"Linear", "LoRACompatibleLinear", "QLinear"}


def _container(child):
    container, attribute = child, "forward"
    owner_ref = getattr(child, "ara_lora_ref", None)
    owner = owner_ref() if callable(owner_ref) else None
    if owner is None:
        candidate = getattr(getattr(child, "forward", None), "__self__", None)
        if candidate is not None and candidate is not child:
            owner = candidate
    if owner is not None and hasattr(owner, "org_forward"):
        container, attribute = owner, "org_forward"
    return container, attribute


def _live_tensors(child, operation):
    """Read the state currently installed by ``functional_call``.

    Canonical arena modules are repointed to host storage between calls. The
    generic dispatcher replaces their weight/bias state with resident or
    freshly-streamed CUDA tensors for the duration of a block call, so a
    compiled FP8 forward must read the module state here instead of closing
    over the host tensors seen during setup.
    """
    declaration = declare_fp8_linear(child.weight)
    if declaration is None:
        raise RuntimeError("arena_fp8_live_weight_lost_declaration")
    tensors = [declaration.qdata, declaration.scale]
    if operation.bias_index is not None:
        tensors.append(child.bias)
    return tuple(tensors)


def enable(
    model,
    *,
    include_ids=None,
    live_ids=None,
    training: bool,
    device=None,
):
    """Install bound FP8 operations without owning their execution policy.

    ``live_ids`` identifies arena-managed modules whose tensors may change
    device or storage after setup. Their wrappers read the current module
    state at call time instead of retaining stale setup-time tensors.
    """
    restores = []
    include_ids = None if include_ids is None else set(include_ids)
    live_ids = set(live_ids or ())
    for child in model.modules():
        if child.__class__.__name__ not in LINEAR_MODULES:
            continue
        if include_ids is not None and id(child) not in include_ids:
            continue
        weight = getattr(child, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.requires_grad:
            continue
        bias = getattr(child, "bias", None)
        operation, tensors = bind_parameter_operation(
            weight,
            bias,
            device=weight.device if device is None else device,
        )
        if operation.format_key != "rowwise_fp8" or not operation.native:
            continue
        container, attribute = _container(child)
        original = getattr(container, attribute)
        live = id(child) in live_ids

        def installed(
            x,
            *args,
            _child=child,
            _live=live,
            _tensors=tensors,
            _operation=operation,
            _original=original,
            **kwargs,
        ):
            if args or kwargs:
                return _original(x, *args, **kwargs)
            tensors_now = (
                _live_tensors(_child, _operation) if _live else _tensors
            )
            if training:
                return _operation.forward_train(x, tensors_now)
            return _operation.forward_sample(x, tensors_now)

        setattr(container, attribute, installed)
        restores.append((container, attribute, original, installed, id(child)))
    return restores


def disable(restores) -> None:
    for restore in reversed(restores):
        container, attribute, original, installed = restore[:4]
        if getattr(container, attribute, None) is installed:
            setattr(container, attribute, original)
