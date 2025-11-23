"""
This code was heavily inspired by the work of Lodestone-Rock, pretty much all credit goes
to them. The original code can be found here:
https://github.com/lodestone-rock/RamTorch/blob/main/ramtorch/modules/linear.py

I simply modified it to work with a memory management model and with AI Toolkit's models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple
from torch.overrides import has_torch_function_unary  # (ADD) torchao detection

if TYPE_CHECKING:
    from .manager import MemoryManager

# --- Per-device global state registry ---
_DEVICE_STATE = {}


def _get_device_state(device: torch.device):
    """Get or initialize per-device state."""
    if isinstance(device, str):
        device = torch.device(device)

    # CPU path needs no CUDA state
    if device.type != "cuda":
        if device not in _DEVICE_STATE:
            _DEVICE_STATE[device] = {}
        return _DEVICE_STATE[device]

    if device not in _DEVICE_STATE:
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                # streams & events
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                "transfer_forward_finished_event": torch.cuda.Event(),
                "compute_forward_start_event": torch.cuda.Event(),
                "transfer_backward_finished_event": torch.cuda.Event(),
                "transfer_weight_backward_finished_event": torch.cuda.Event(),
                "compute_backward_start_event": torch.cuda.Event(),
                "compute_backward_finished_event": torch.cuda.Event(),
                # ping-pong buffers
                "w_buffers": [None, None],
                "b_buffers": [None, None],
                "w_bwd_buffers": [None, None],
                # device-side staging for grads to be sent to CPU
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE[device]


# (ADD) detect torchao wrapper tensors
def _is_ao_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    try:
        if has_torch_function_unary(t):
            return t.__class__.__module__.startswith("torchao.")
    except Exception:
        pass
    for attr in (
        "_scale",
        "_scales",
        "_zero_point",
        "_zp",
        "_block_size",
        "_group_size",
        "_pack_dim",
    ):
        if hasattr(t, attr):
            return True
    return False


def _is_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    # torch quantized tensors
    try:
        if torch.is_quantized(t):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    # (ADD) torchao quantized wrappers
    if _is_ao_quantized_tensor(t):
        return True
    # packed/int formats (weight-only)
    return not t.dtype.is_floating_point


def _ensure_cpu_pinned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.device.type != "cpu":
        try:
            t = t.to("cpu", copy=True)
        except Exception:
            t = t.to("cpu")
    # Don't attempt to pin quantized tensors; many backends don't support it
    if _is_quantized_tensor(t):
        return t
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t


def _move_params_to_cpu_and_pin(module: nn.Module):
    """Force parameters to CPU (+pinned) so we can 'bounce' them per forward/backward."""
    with torch.no_grad():
        if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            module.weight.data = _ensure_cpu_pinned(module.weight.data).detach()
        if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
            if module.bias is not None:
                module.bias.data = _ensure_cpu_pinned(module.bias.data).detach()


# ==========================
# Autograd functions (CUDA)
# ==========================


class _BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device: torch.device):
        # choose compute dtype to match activations
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_linear_weight(cpu_w, dev):
            if _is_quantized_tensor(cpu_w):
                # move quantized wrapper to GPU -> dequantize on GPU -> cast on GPU
                w_q_gpu = cpu_w.to(dev, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w_gpu = cpu_w.to(dev, non_blocking=True)
            return w_gpu

        if device.type != "cuda":
            out = F.linear(
                x.to("cpu"),
                _materialize_linear_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.device = torch.device("cpu")
            return out.to(x.device)

        state = _get_device_state(device)
        ts = state["transfer_stream"]
        w_bufs, b_bufs = state["w_buffers"], state["b_buffers"]
        ev_tx_f = state["transfer_forward_finished_event"]
        ev_cu_s = state["compute_forward_start_event"]
        idx = state["forward_clk"]

        with torch.cuda.stream(ts):
            ts.wait_event(ev_cu_s)
            w_bufs[idx] = _materialize_linear_weight(weight_cpu, device)
            b_bufs[idx] = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )
            state["forward_clk"] ^= 1
            ev_tx_f.record()

        torch.cuda.current_stream().wait_event(ev_tx_f)
        ev_cu_s.record()
        out = F.linear(x, w_bufs[idx], b_bufs[idx])

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.target_dtype = target_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        target_dtype = getattr(ctx, "target_dtype", grad_out.dtype)

        if device.type != "cuda":
            go_cpu = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_mat = (
                weight_cpu.dequantize()
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_mat.dtype != target_dtype and target_dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                w_mat = w_mat.to(target_dtype)
            grad_input = go_cpu @ w_mat
            grad_weight = (
                go_cpu.flatten(0, -2).T @ x_cpu.flatten(0, -2)
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go_cpu.sum(dim=tuple(range(go_cpu.ndim - 1)))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return grad_input.to(grad_out.device), grad_weight, grad_bias, None

        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]

        w_bwd_buffers = state["w_bwd_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]

        ev_tx_b = state["transfer_backward_finished_event"]
        ev_tx_w_bwd_done = state["transfer_weight_backward_finished_event"]
        ev_cu_b_start = state["compute_backward_start_event"]
        ev_cu_b_finish = state["compute_backward_finished_event"]

        idx = state["backward_clk"]

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_for_bwd(cpu_w):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(device, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w = cpu_w.to(device, non_blocking=True)
            return w

        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(ev_cu_b_start)
            w_bwd_buffers[idx] = _materialize_for_bwd(weight_cpu)
            state["backward_clk"] ^= 1
            ev_tx_b.record()

        torch.cuda.current_stream().wait_event(ev_tx_b)
        ev_cu_b_start.record()

        # grad wrt input (GPU)
        grad_input = grad_out.to(dtype=target_dtype) @ w_bwd_buffers[idx]

        # ensure previous grad-to-CPU transfer that used this slot finished
        torch.cuda.current_stream().wait_event(ev_tx_w_bwd_done)

        # compute grads if float masters exist
        grad_weight = None
        grad_bias = None
        if (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        ):
            w_grad_buffers[idx] = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
            reduce_dims = tuple(range(grad_out.ndim - 1))
            b_grad_buffers[idx] = grad_out.sum(dim=reduce_dims)

        ev_cu_b_finish.record()

        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(ev_cu_b_finish)
            if (
                getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
            ):
                grad_weight = w_grad_buffers[idx].to("cpu", non_blocking=True)
            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = b_grad_buffers[idx].to("cpu", non_blocking=True)
            state["transfer_weight_backward_finished_event"].record()

        return grad_input.to(dtype=grad_out.dtype), grad_weight, grad_bias, None


class _BouncingConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        device: torch.device,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ):
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_conv_weight(cpu_w, dev):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(dev, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w_gpu = cpu_w.to(dev, non_blocking=True)
            return w_gpu

        if device.type != "cuda":
            out = F.conv2d(
                x.to("cpu"),
                _materialize_conv_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
                stride,
                padding,
                dilation,
                groups,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.meta = ("cpu", stride, padding, dilation, groups, target_dtype)
            return out.to(x.device)

        state = _get_device_state(device)
        ts = state["transfer_stream"]
        w_bufs, b_bufs = state["w_buffers"], state["b_buffers"]
        ev_tx_f = state["transfer_forward_finished_event"]
        ev_cu_s = state["compute_forward_start_event"]
        idx = state["forward_clk"]

        with torch.cuda.stream(ts):
            ts.wait_event(ev_cu_s)
            w_bufs[idx] = _materialize_conv_weight(weight_cpu, device)
            b_bufs[idx] = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )
            state["forward_clk"] ^= 1
            ev_tx_f.record()

        torch.cuda.current_stream().wait_event(ev_tx_f)
        ev_cu_s.record()
        out = F.conv2d(x, w_bufs[idx], b_bufs[idx], stride, padding, dilation, groups)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.meta = (device, stride, padding, dilation, groups, target_dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device, stride, padding, dilation, groups, target_dtype = ctx.meta

        if (
            isinstance(device, torch.device) and device.type != "cuda"
        ) or device == "cpu":
            go = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_cpu = (
                weight_cpu.dequantize()
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_cpu.dtype != target_dtype and target_dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                w_cpu = w_cpu.to(target_dtype)
            from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore

            grad_input = conv2d_input(
                x_cpu.shape,
                w_cpu,
                go,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            grad_weight = (
                conv2d_weight(
                    x_cpu,
                    w_cpu.shape,
                    go,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go.sum(dim=(0, 2, 3))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return (
                grad_input.to(grad_out.device),
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
            )

        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]

        w_bwd_buffers = state["w_bwd_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]

        ev_tx_b = state["transfer_backward_finished_event"]
        ev_tx_w_bwd_done = state["transfer_weight_backward_finished_event"]
        ev_cu_b_start = state["compute_backward_start_event"]
        ev_cu_b_finish = state["compute_backward_finished_event"]

        idx = state["backward_clk"]

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_for_bwd(cpu_w):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(device, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w = cpu_w.to(device, non_blocking=True)
            return w

        # Stage weights for input-grad compute
        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(ev_cu_b_start)
            w_bwd_buffers[idx] = _materialize_for_bwd(weight_cpu)
            state["backward_clk"] ^= 1
            ev_tx_b.record()

        torch.cuda.current_stream().wait_event(ev_tx_b)
        ev_cu_b_start.record()

        from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore

        grad_input = conv2d_input(
            x.shape,
            w_bwd_buffers[idx],
            grad_out.to(dtype=target_dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        # Ensure previous grad transfer that used this slot is done
        torch.cuda.current_stream().wait_event(ev_tx_w_bwd_done)

        # Compute heavy grads on GPU into staging buffers
        grad_weight = None
        grad_bias = None
        if (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        ):
            w_grad_buffers[idx] = conv2d_weight(
                x,
                weight_cpu.shape,
                grad_out,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
            b_grad_buffers[idx] = grad_out.sum(dim=(0, 2, 3))

        ev_cu_b_finish.record()

        # Launch CPU copies on the dedicated grad stream (overlaps with next H2D)
        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(ev_cu_b_finish)
            if (
                getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
            ):
                grad_weight = w_grad_buffers[idx].to("cpu", non_blocking=True)
            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
                grad_bias = b_grad_buffers[idx].to("cpu", non_blocking=True)
            state["transfer_weight_backward_finished_event"].record()

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


class BaseLayerMemoryManager:
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        self.module: nn.Module = module
        self.manager: "MemoryManager" = manager

    @classmethod
    def attach(cls, module: nn.Module, manager: "MemoryManager"):
        if hasattr(module, "_layer_memory_manager"):
            return
        module._layer_memory_manager = cls(module, manager)

        # mark parameters as memory managed
        for param in module.parameters(recurse=False):
            param._is_memory_managed = True


class LinearLayerMemoryManager(BaseLayerMemoryManager):
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # 2) Hijack forward
        if hasattr(self.module, "ara_lora_ref"):
            # ARA, we need to replace the lora forward
            self._original_forward = getattr(self.module.ara_lora_ref(), "org_forward")
        else:
            self._original_forward = getattr(self.module, "forward")

        def _mm_forward(x, *args, **kwargs):
            # ensure we only use expected signature (Linear: x)
            if args or kwargs:
                # fall back to original if a custom signature is used
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            # NOTE: do NOT move params to device here; autograd fn streams & bounces them
            return _BouncingLinearFn.apply(x, weight_cpu, bias_cpu, device)

        if hasattr(self.module, "ara_lora_ref"):
            self.module.ara_lora_ref().org_forward = _mm_forward
        else:
            self.module.forward = _mm_forward
        
        self.module._memory_management_device = self.manager.process_device


class ConvLayerMemoryManager(BaseLayerMemoryManager):
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # Cache static conv attributes from the module
        stride = (
            self.module.stride
            if isinstance(self.module.stride, tuple)
            else (self.module.stride, self.module.stride)
        )
        padding = (
            self.module.padding
            if isinstance(self.module.padding, tuple)
            else (self.module.padding, self.module.padding)
        )
        dilation = (
            self.module.dilation
            if isinstance(self.module.dilation, tuple)
            else (self.module.dilation, self.module.dilation)
        )
        groups = self.module.groups

        # 2) Hijack forward
        if hasattr(self.module, "ara_lora_ref"):
            # ARA, we need to replace the lora forward
            self._original_forward = getattr(self.module.ara_lora_ref(), "org_forward")
        else:
            self._original_forward = getattr(self.module, "forward")

        def _mm_forward(x, *args, **kwargs):
            # Support the typical Conv2d(x) call; if user passes uncommon extras, fallback.
            if args or kwargs:
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            return _BouncingConv2dFn.apply(
                x, weight_cpu, bias_cpu, device, stride, padding, dilation, groups
            )

        if hasattr(self.module, "ara_lora_ref"):
            self.module.ara_lora_ref().org_forward = _mm_forward
        else:
            self.module.forward = _mm_forward
        
        self.module._memory_management_device = self.manager.process_device
