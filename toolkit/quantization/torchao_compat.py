"""TorchAO API compatibility and optional arena-FP8 capability checks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import re

import torch

from torchao.quantization.quant_api import (
    Float8WeightOnlyConfig,
    Int8WeightOnlyConfig,
    quantize_ as torchao_quantize_,
)

try:
    from torchao.quantization.quant_api import _is_linear as torchao_is_linear
except ImportError:
    def torchao_is_linear(module: torch.nn.Module, _fqn: str) -> bool:
        return isinstance(module, torch.nn.Linear)

try:
    from torchao.quantization import Float8Tensor as _Float8Tensor
except ImportError:
    _Float8Tensor = None

try:
    from torchao.quantization.quant_api import IntxWeightOnlyConfig as _IntxConfig
except ImportError:
    _IntxConfig = None

try:
    from torchao.quantization.quant_api import UIntXWeightOnlyConfig as _UIntXConfig
except ImportError:
    _UIntXConfig = None


TORCHAO_ARENA_FP8_MIN_VERSION = "0.17.0"
_TORCHAO_ARENA_FP8_MIN_RELEASE = (0, 17, 0)


def _release_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value))
    if match is None:
        return ()
    return tuple(int(component) for component in match.group(1).split("."))


try:
    TORCHAO_VERSION = version("torchao")
except PackageNotFoundError:
    TORCHAO_VERSION = "unknown"


def torchao_arena_fp8_supported(
    installed_version: str | None = None,
    *,
    float8_tensor_available: bool | None = None,
) -> bool:
    """Whether the tested TorchAO Float8Tensor arena adapter is available."""
    release = _release_tuple(
        TORCHAO_VERSION if installed_version is None else installed_version
    )
    has_tensor = (
        _Float8Tensor is not None
        if float8_tensor_available is None
        else bool(float8_tensor_available)
    )
    return bool(has_tensor and release >= _TORCHAO_ARENA_FP8_MIN_RELEASE)


def torchao_is_float8_tensor(value) -> bool:
    return bool(_Float8Tensor is not None and isinstance(value, _Float8Tensor))


def intx_weight_only_config(bits: int):
    """Build the low-bit config using either the current or TorchAO 0.10 API."""
    bits = int(bits)
    if _IntxConfig is not None:
        return _IntxConfig(getattr(torch, f"int{bits}"))
    if _UIntXConfig is not None:
        return _UIntXConfig(getattr(torch, f"uint{bits}"))
    raise RuntimeError("installed TorchAO has no supported IntX weight-only config")


__all__ = [
    "Float8WeightOnlyConfig",
    "Int8WeightOnlyConfig",
    "TORCHAO_ARENA_FP8_MIN_VERSION",
    "TORCHAO_VERSION",
    "intx_weight_only_config",
    "torchao_arena_fp8_supported",
    "torchao_is_float8_tensor",
    "torchao_is_linear",
    "torchao_quantize_",
]
