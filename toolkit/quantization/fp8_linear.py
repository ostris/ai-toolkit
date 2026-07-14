"""Row-wise FP8 Linear execution independent of storage and residency.

This module owns native qualification, activation quantization, scaled-matmul
execution, materializing fallback, and training grad-input policy. Callers
supply explicit qdata, row scale, and bias tensors; no arena or memory-manager
state is inspected here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fp8_transpose import column_major


FP8_STATS = {
    "enabled": False,
    "training_enabled": False,
    "kernel_calls": 0,
    "fallback_calls": 0,
}

FP8_LINEAR_EXECUTION_KEY = "toolkit.quantization.fp8_linear"
DEFAULT_ACTIVATION_FP8_DTYPE = torch.float8_e4m3fn
NATIVE_SCALED_MM_FP8_DTYPE = torch.float8_e4m3fn


@dataclass(frozen=True)
class Fp8LinearSpec:
    """Backend-neutral semantic description of an FP8 Linear weight."""

    weight_dtype: torch.dtype
    activation_dtype: torch.dtype
    scale_granularity: str
    scale_dtype: torch.dtype
    has_zero_point: bool
    weight_layout: str
    execution_variant: str


@dataclass(frozen=True)
class Fp8LinearDeclaration:
    spec: Fp8LinearSpec
    qdata: torch.Tensor
    scale: torch.Tensor

    @property
    def tensors(self):
        return (self.qdata, self.scale)


def fp8_execution_key(spec: Fp8LinearSpec) -> tuple:
    return (FP8_LINEAR_EXECUTION_KEY, spec)


def fp8_spec_from_execution_key(execution_key) -> Fp8LinearSpec | None:
    if (
        isinstance(execution_key, tuple)
        and len(execution_key) == 2
        and execution_key[0] == FP8_LINEAR_EXECUTION_KEY
        and isinstance(execution_key[1], Fp8LinearSpec)
    ):
        return execution_key[1]
    return None


def _spec_for_payload(qdata, scale, granularity) -> Fp8LinearSpec:
    return Fp8LinearSpec(
        weight_dtype=qdata.dtype,
        activation_dtype=DEFAULT_ACTIVATION_FP8_DTYPE,
        scale_granularity=granularity,
        scale_dtype=scale.dtype,
        has_zero_point=False,
        weight_layout="out_in",
        execution_variant="scaled_mm_dynamic_activation",
    )


def _adapt_quanto_fp8(value) -> Fp8LinearDeclaration | None:
    try:
        from optimum.quanto.tensor.qbytes import QBytesTensor
    except ImportError:
        return None
    if not isinstance(value, QBytesTensor):
        return None
    qtype = value.qtype
    if not qtype.is_floating_point or qtype.bits != 8:
        return None
    axis = value.axis
    if axis == 0:
        granularity = "output_row"
    elif axis is None:
        granularity = "per_tensor"
    elif axis == -1:
        granularity = "input_column"
    else:
        return None
    qdata, scale = value._data, value._scale
    return Fp8LinearDeclaration(
        _spec_for_payload(qdata, scale, granularity),
        qdata,
        scale,
    )


def _adapt_torchao_fp8(value) -> Fp8LinearDeclaration | None:
    try:
        from torchao.quantization import Float8Tensor
    except ImportError:
        return None
    if not isinstance(value, Float8Tensor):
        return None
    qdata, scale = value.qdata, value.scale
    block_size = tuple(value.block_size or ())
    if qdata.ndim == 2 and block_size == (1, qdata.shape[1]):
        granularity = "output_row"
    elif scale.numel() == 1 and block_size in ((), tuple(qdata.shape)):
        granularity = "per_tensor"
    else:
        return None
    return Fp8LinearDeclaration(
        _spec_for_payload(qdata, scale, granularity),
        qdata,
        scale,
    )


def declare_fp8_linear(value) -> Fp8LinearDeclaration | None:
    """Normalize supported backend wrappers without leaking them to callers."""
    value = value.data if isinstance(value, torch.nn.Parameter) else value
    return _adapt_torchao_fp8(value) or _adapt_quanto_fp8(value)

_FP8_GRAD_INPUT = os.environ.get("AI_TOOLKIT_FP8_GRAD_INPUT", "0").lower() not in (
    "0", "false", "no", "off", "",
)
_FP8_GRAD_VERIFIED = None
_REUSE_DEQUANT = os.environ.get("AI_TOOLKIT_REUSE_DEQUANT", "1").lower() not in (
    "0", "false", "no", "off", "",
)
_REUSE_VERIFIED = None


def set_fp8_grad_input_enabled(enabled: bool) -> None:
    global _FP8_GRAD_INPUT, _FP8_GRAD_VERIFIED
    if bool(enabled) and not _FP8_GRAD_INPUT:
        _FP8_GRAD_VERIFIED = None
    _FP8_GRAD_INPUT = bool(enabled)


def fp8_grad_input_enabled() -> bool:
    """Return the Toolkit-owned process policy for FP8 input gradients."""
    return bool(_FP8_GRAD_INPUT)


def reference_dequantize_to(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    try:
        return tensor.dequantize(output_dtype=dtype)
    except TypeError:
        value = tensor.dequantize()
        return value if value.dtype == dtype else value.to(dtype=dtype)


def _scale_view_shape(spec, qdata, scale):
    if spec.scale_granularity == "output_row":
        if scale.numel() != qdata.shape[0]:
            raise ValueError("invalid_output_row_scale")
        return [qdata.shape[0]] + [1] * (qdata.ndim - 1)
    if spec.scale_granularity == "input_column":
        if scale.numel() != qdata.shape[-1]:
            raise ValueError("invalid_input_column_scale")
        return [1] * (qdata.ndim - 1) + [qdata.shape[-1]]
    if spec.scale_granularity == "per_tensor":
        if scale.numel() != 1:
            raise ValueError("invalid_per_tensor_scale")
        return [1] * qdata.ndim
    raise ValueError("unsupported_fp8_scale_granularity")


def materialize_fp8_weight(spec, qdata, scale, dtype):
    if spec.has_zero_point or spec.weight_layout != "out_in":
        raise ValueError("unsupported_fp8_materialization")
    view_shape = _scale_view_shape(spec, qdata, scale)
    return qdata.to(dtype) * scale.reshape(view_shape).to(dtype)


def dequantize_rowwise(qdata, scale, dtype):
    spec = _spec_for_payload(qdata, scale, "output_row")
    return materialize_fp8_weight(spec, qdata, scale, dtype)


def fast_dequantize_into(qweight, dest):
    declaration = declare_fp8_linear(qweight)
    if declaration is None or dest is None:
        return None
    spec = declaration.spec
    qdata, scale = declaration.qdata, declaration.scale
    if qdata.shape != dest.shape or spec.has_zero_point:
        return None
    try:
        view_shape = _scale_view_shape(spec, qdata, scale)
    except ValueError:
        return None
    dest.copy_(qdata)
    dest.mul_(scale.reshape(view_shape).to(dest.dtype))
    return dest


def fast_dequantize(qweight, dtype):
    global _REUSE_VERIFIED
    if not _REUSE_DEQUANT or _REUSE_VERIFIED is False:
        return None
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    declaration = declare_fp8_linear(qweight)
    if declaration is None:
        return None
    qdata = declaration.qdata
    dest = torch.empty(qdata.shape, dtype=dtype, device=qdata.device)
    fast = fast_dequantize_into(qweight, dest)
    if fast is None:
        return None
    if _REUSE_VERIFIED is None:
        try:
            reference = reference_dequantize_to(qweight, dtype)
            ok = reference.shape == fast.shape and torch.allclose(
                fast, reference, rtol=1e-2, atol=1e-2
            )
        except Exception:
            ok = False
        _REUSE_VERIFIED = bool(ok)
        if not ok:
            return None
    return fast


def dequantize_to(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    fast = fast_dequantize(tensor, dtype)
    return fast if fast is not None else reference_dequantize_to(tensor, dtype)


def dequantize_into(qweight, dest):
    global _REUSE_VERIFIED
    if not _REUSE_DEQUANT or _REUSE_VERIFIED is False or dest is None:
        return None
    fast = fast_dequantize_into(qweight, dest)
    if fast is None:
        return None
    if _REUSE_VERIFIED is None:
        try:
            reference = reference_dequantize_to(qweight, dest.dtype)
            ok = reference.shape == fast.shape and torch.allclose(
                fast, reference, rtol=1e-2, atol=1e-2
            )
        except Exception:
            ok = False
        _REUSE_VERIFIED = bool(ok)
        if not ok:
            return None
    return fast


def native_variant_supported(spec) -> bool:
    return bool(
        spec.weight_dtype == NATIVE_SCALED_MM_FP8_DTYPE
        and spec.activation_dtype == NATIVE_SCALED_MM_FP8_DTYPE
        and spec.scale_granularity == "output_row"
        and not spec.has_zero_point
        and spec.weight_layout == "out_in"
        and spec.execution_variant == "scaled_mm_dynamic_activation"
    )


def supports_native_scaled_mm(spec, qdata, scale, *, device=None) -> bool:
    target = qdata.device if device is None else torch.device(device)
    if not native_device_supported(target):
        return False
    return bool(
        native_variant_supported(spec)
        and qdata.dtype == spec.weight_dtype
        and scale.dtype == spec.scale_dtype
        and qdata.ndim == 2
        and scale.is_floating_point()
        and scale.numel() == qdata.shape[0]
        and qdata.shape[0] % 16 == 0
        and qdata.shape[1] % 16 == 0
    )


def native_rowwise_qualifies(qdata, scale, *, device=None) -> bool:
    spec = _spec_for_payload(qdata, scale, "output_row")
    return supports_native_scaled_mm(spec, qdata, scale, device=device)


def native_device_supported(device) -> bool:
    target = torch.device(device)
    if target.type != "cuda" or not hasattr(torch, "_scaled_mm"):
        return False
    try:
        return torch.cuda.get_device_capability(target) >= (8, 9)
    except Exception:
        return False


def weight_format_key(weight) -> str:
    value = weight.data if isinstance(weight, torch.nn.Parameter) else weight
    declaration = declare_fp8_linear(value)
    if declaration is None:
        return "other"
    return (
        "rowwise_fp8"
        if native_variant_supported(declaration.spec)
        else "fp8"
    )


def fp8_sampling_qualifies(weight, *, device=None) -> bool:
    declaration = declare_fp8_linear(weight)
    return bool(
        declaration is not None
        and supports_native_scaled_mm(
            declaration.spec,
            declaration.qdata,
            declaration.scale,
            device=device,
        )
    )


def native_linear(
    x,
    qdata_t,
    scale_row,
    bias,
    activation_dtype=DEFAULT_ACTIVATION_FP8_DTYPE,
):
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    info = torch.finfo(activation_dtype)
    scale_x = torch.clamp(
        x_2d.abs().amax().float() / info.max,
        min=torch.finfo(torch.float32).tiny,
    )
    x_fp8 = torch.clamp(
        x_2d / scale_x.to(x_2d.dtype), min=info.min, max=info.max
    ).to(activation_dtype)
    one = torch.ones((), device=x.device, dtype=torch.float32)
    out = torch._scaled_mm(
        x_fp8,
        qdata_t,
        scale_a=scale_x,
        scale_b=one,
        out_dtype=x.dtype,
        use_fast_accum=True,
    )
    out = out * scale_row.reshape(1, -1).to(out.dtype)
    if bias is not None:
        out = out + bias.to(device=x.device, dtype=out.dtype)
    return out.reshape(*shape[:-1], scale_row.shape[0])


def materialized_linear(x, spec, qdata, scale, bias):
    return F.linear(x, materialize_fp8_weight(spec, qdata, scale, x.dtype), bias)


def _grad_input_compute(
    grad_out,
    qdata,
    scale,
    target_dtype,
    activation_dtype=DEFAULT_ACTIVATION_FP8_DTYPE,
):
    try:
        shape = grad_out.shape
        grad = grad_out.reshape(-1, shape[-1]).to(torch.float32)
        grad = grad * scale.reshape(1, -1).to(torch.float32)
        info = torch.finfo(activation_dtype)
        scale_grad = torch.clamp(
            grad.abs().amax() / info.max,
            min=torch.finfo(torch.float32).tiny,
        )
        grad_fp8 = torch.clamp(
            grad / scale_grad, min=info.min, max=info.max
        ).to(activation_dtype)
        one = torch.ones((), device=grad_out.device, dtype=torch.float32)
        result = torch._scaled_mm(
            grad_fp8,
            column_major(qdata),
            scale_a=scale_grad,
            scale_b=one,
            out_dtype=target_dtype,
            use_fast_accum=True,
        )
        return result.reshape(*shape[:-1], qdata.shape[1])
    except Exception:
        return None


def grad_input_supported(qdata, scale, grad_out, spec=None) -> bool:
    if not _FP8_GRAD_INPUT or _FP8_GRAD_VERIFIED is False:
        return False
    spec = spec or _spec_for_payload(qdata, scale, "output_row")
    return bool(
        supports_native_scaled_mm(spec, qdata, scale, device=grad_out.device)
        and grad_out.dtype in (torch.bfloat16, torch.float16)
        and grad_out.shape[-1] == qdata.shape[0]
    )


def grad_input_supported_weight(qweight, grad_out) -> bool:
    declaration = declare_fp8_linear(qweight)
    return bool(
        declaration is not None
        and grad_input_supported(
            declaration.qdata,
            declaration.scale,
            grad_out,
            declaration.spec,
        )
    )


def grad_input(grad_out, qweight, target_dtype):
    declaration = declare_fp8_linear(qweight)
    if declaration is None:
        return None
    spec = declaration.spec
    qdata, scale = declaration.qdata, declaration.scale
    global _FP8_GRAD_VERIFIED
    out = (
        _grad_input_compute(
            grad_out,
            qdata,
            scale,
            target_dtype,
            spec.activation_dtype,
        )
        if grad_input_supported(qdata, scale, grad_out, spec)
        else None
    )
    if out is not None and _FP8_GRAD_VERIFIED is None:
        try:
            reference = grad_out.to(target_dtype) @ dequantize_to(qweight, target_dtype)
            _FP8_GRAD_VERIFIED = bool(
                torch.allclose(out, reference, rtol=2e-2, atol=2e-2)
            )
        except Exception:
            _FP8_GRAD_VERIFIED = False
    if out is not None and _FP8_GRAD_VERIFIED:
        return out
    return grad_out.to(target_dtype) @ dequantize_to(qweight, target_dtype)


class _NativeTrainingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qdata_t, scale_row, bias, activation_dtype):
        ctx.save_for_backward(qdata_t, scale_row)
        ctx.input_dtype = x.dtype
        ctx.fp8_grad_input = bool(_FP8_GRAD_INPUT)
        ctx.activation_dtype = activation_dtype
        return native_linear(
            x,
            qdata_t,
            scale_row,
            bias,
            activation_dtype,
        )

    @staticmethod
    def backward(ctx, grad_out):
        qdata_t, scale_row = ctx.saved_tensors
        qdata = qdata_t.t()
        result = None
        if ctx.fp8_grad_input:
            result = _grad_input_compute(
                grad_out,
                qdata,
                scale_row,
                ctx.input_dtype,
                ctx.activation_dtype,
            )
        if result is None:
            weight = dequantize_rowwise(qdata, scale_row, ctx.input_dtype)
            result = grad_out.to(ctx.input_dtype) @ weight
        return result.to(grad_out.dtype), None, None, None, None


def native_linear_training(
    x,
    qdata_t,
    scale_row,
    bias,
    activation_dtype=DEFAULT_ACTIVATION_FP8_DTYPE,
):
    return _NativeTrainingFn.apply(
        x,
        qdata_t,
        scale_row,
        bias,
        activation_dtype,
    )


def fp8_linear_inference(x, weight, bias):
    declaration = declare_fp8_linear(weight)
    if (
        declaration is None
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.numel() == 0
    ):
        FP8_STATS["fallback_calls"] += int(
            FP8_STATS["enabled"] or FP8_STATS["training_enabled"]
        )
        return None
    spec = declaration.spec
    qdata, scale = declaration.qdata, declaration.scale
    if (
        not supports_native_scaled_mm(spec, qdata, scale, device=x.device)
        or qdata.device != x.device
        or scale.device != x.device
        or (bias is not None and bias.device != x.device)
        or x.shape[-1] != qdata.shape[1]
    ):
        FP8_STATS["fallback_calls"] += int(
            FP8_STATS["enabled"] or FP8_STATS["training_enabled"]
        )
        return None
    try:
        out = native_linear(
            x,
            qdata.t(),
            scale,
            bias,
            spec.activation_dtype,
        )
    except RuntimeError:
        FP8_STATS["fallback_calls"] += int(
            FP8_STATS["enabled"] or FP8_STATS["training_enabled"]
        )
        return None
    FP8_STATS["kernel_calls"] += int(
        FP8_STATS["enabled"] or FP8_STATS["training_enabled"]
    )
    return out


@dataclass(frozen=True)
class BoundFp8LinearOperation:
    spec: Fp8LinearSpec
    native: bool
    bias_index: int | None = 2
    @property
    def format_key(self):
        return (
            "rowwise_fp8"
            if native_variant_supported(self.spec)
            else "fp8"
        )

    def explicit_tensors(self, weight, bias, scale):
        values = [weight, scale]
        if self.bias_index is not None:
            values.append(bias)
        return tuple(values)

    def _unpack(self, tensors):
        qdata, scale = tensors[0], tensors[1]
        bias = None if self.bias_index is None else tensors[self.bias_index]
        return qdata, scale, bias

    def functional_components(self, tensors):
        qdata, scale, bias = self._unpack(tensors)
        return qdata, bias, scale

    def forward_sample(self, x, tensors):
        qdata, scale, bias = self._unpack(tensors)
        if self.native:
            return native_linear(
                x,
                qdata.t(),
                scale.reshape(-1),
                bias,
                self.spec.activation_dtype,
            )
        return materialized_linear(x, self.spec, qdata, scale, bias)

    def forward_train(self, x, tensors):
        qdata, scale, bias = self._unpack(tensors)
        if self.native:
            return native_linear_training(
                x,
                qdata.t(),
                scale.reshape(-1),
                bias,
                self.spec.activation_dtype,
            )
        return materialized_linear(x, self.spec, qdata, scale, bias)

    def forward_explicit(self, x, weight, bias, scale, *, training):
        """Execute either the original FP8 tuple or a dense replacement."""
        if scale is None:
            return BoundDenseLinearOperation()._forward(x, weight, bias)
        tensors = self.explicit_tensors(weight, bias, scale)
        forward = self.forward_train if training else self.forward_sample
        return forward(x, tensors)

    def materialize(self, tensors, dtype=torch.bfloat16):
        qdata, scale, _bias = self._unpack(tensors)
        return materialize_fp8_weight(self.spec, qdata, scale, dtype)


def bind_fp8_linear(
    spec,
    qdata,
    scale,
    *,
    device,
    has_bias=True,
) -> BoundFp8LinearOperation:
    return BoundFp8LinearOperation(
        spec=spec,
        native=supports_native_scaled_mm(spec, qdata, scale, device=device),
        bias_index=2 if has_bias else None,
    )


def bind_rowwise_fp8(qdata, scale, *, device, has_bias=True) -> BoundFp8LinearOperation:
    """Compatibility binder for an explicitly declared rowwise payload."""
    spec = _spec_for_payload(qdata, scale, "output_row")
    return bind_fp8_linear(
        spec,
        qdata,
        scale,
        device=device,
        has_bias=has_bias,
    )


@dataclass(frozen=True)
class BoundDenseLinearOperation:
    format_key = "dense"
    def _forward(self, x, weight, bias):
        if weight.dtype != x.dtype and weight.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    bias_index: int | None = 1

    def explicit_tensors(self, weight, bias, _scale):
        return (weight,) if self.bias_index is None else (weight, bias)

    def functional_components(self, tensors):
        weight = tensors[0]
        bias = None if self.bias_index is None else tensors[self.bias_index]
        return weight, bias, None

    def forward_sample(self, x, tensors):
        weight = tensors[0]
        bias = None if self.bias_index is None else tensors[self.bias_index]
        return self._forward(x, weight, bias)

    def forward_train(self, x, tensors):
        weight = tensors[0]
        bias = None if self.bias_index is None else tensors[self.bias_index]
        return self._forward(x, weight, bias)

    def forward_explicit(self, x, weight, bias, scale, *, training):
        del scale, training
        return self._forward(x, weight, bias)

    def materialize(self, tensors, dtype=None):
        weight = tensors[0]
        return weight if dtype is None or weight.dtype == dtype else weight.to(dtype=dtype)


def bind_linear_operation(weight, bias=None, *, device):
    value = weight.data if isinstance(weight, torch.nn.Parameter) else weight
    declaration = declare_fp8_linear(value)
    if declaration is None:
        try:
            value.__tensor_flatten__()
        except Exception:
            pass
        else:
            raise ValueError("unsupported_quantized_linear_operation")
        return BoundDenseLinearOperation(bias_index=1 if bias is not None else None)
    return bind_fp8_linear(
        declaration.spec,
        declaration.qdata,
        declaration.scale,
        device=device,
        has_bias=bias is not None,
    )


def bind_parameter_operation(weight, bias=None, *, device):
    """Bind an operation and snapshot its explicit ordered tensor tuple."""
    from .storage import linear_storage_binding

    binding = linear_storage_binding(weight, bias)
    tensors = tuple(item.tensor for item in binding.tensors)
    operation = bind_storage_operation(
        tensors,
        device=device,
        weight_leaf_count=binding.weight_leaf_count,
        execution_key=binding.execution_key,
    )
    return operation, tensors


def bind_storage_operation(
    tensors,
    *,
    device,
    weight_leaf_count: int,
    execution_key,
):
    """Bind execution from an opaque tuple outside the memory manager."""
    tensors = tuple(tensors)
    spec = fp8_spec_from_execution_key(execution_key)
    if spec is not None:
        if int(weight_leaf_count) != 2 or len(tensors) not in (2, 3):
            raise ValueError("invalid_fp8_linear_storage")
        return bind_fp8_linear(
            spec,
            tensors[0],
            tensors[1],
            device=device,
            has_bias=len(tensors) > 2,
        )
    dense_declaration = False
    try:
        declared_weight_leaves = tuple(execution_key[2])
        dense_declaration = (
            len(declared_weight_leaves) == 1
            and declared_weight_leaves[0][0] == "weight"
        )
    except (IndexError, TypeError):
        pass
    if int(weight_leaf_count) == 1 and dense_declaration:
        return BoundDenseLinearOperation(bias_index=1 if len(tensors) > 1 else None)
    raise ValueError("unsupported_linear_storage_operation")


# Transitional names for callers migrating from manager_modules.
_fp8_linear_compiled = native_linear
_fp8_linear_training = native_linear_training
_fp8_grad_input_compute = _grad_input_compute
