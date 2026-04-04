import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple

_DEVICE_STATE = {}

def _get_device_state(device: torch.device):
    if isinstance(device, str): device = torch.device(device)
    if device not in _DEVICE_STATE:
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                "transfer_forward_finished_event": torch.cuda.Event(),
                "compute_forward_start_event": torch.cuda.Event(),
                "transfer_backward_finished_event": torch.cuda.Event(),
                "transfer_weight_backward_finished_event": torch.cuda.Event(),
                "compute_backward_start_event": torch.cuda.Event(),
                "compute_backward_finished_event": torch.cuda.Event(),
                "w_buffers": [None, None], "b_buffers": [None, None],
                "w_bwd_buffers": [None, None], "forward_clk": 0, "backward_clk": 0,
            }
    return _DEVICE_STATE[device]

def _is_quant(t):
    return "quant" in str(type(t)).lower() or hasattr(t, "tensor_impl")

def _dequant(t, dtype):
    if _is_quant(t):
        try: return t.dequantize().to(dtype)
        except: return t.to(dtype)
    return t.to(dtype)

def _ensure_cpu_pinned(t):
    if t is None: return None
    if t.device.type != "cpu": t = t.to("cpu")
    if not _is_quant(t) and torch.cuda.is_available():
        try:
            if not t.is_pinned(): t = t.pin_memory()
        except: pass
    return t

class _BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device):
        dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        state = _get_device_state(device)
        idx = state["forward_clk"]
        with torch.cuda.stream(state["transfer_stream"]):
            state["transfer_stream"].wait_event(state["compute_forward_start_event"])
            w = weight_cpu.to(device, non_blocking=True)
            state["w_buffers"][idx] = _dequant(w, dtype)
            state["b_buffers"][idx] = bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            state["forward_clk"] ^= 1
            state["transfer_forward_finished_event"].record()
        torch.cuda.current_stream().wait_event(state["transfer_forward_finished_event"])
        state["compute_forward_start_event"].record()
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device, ctx.dtype = device, dtype
        return F.linear(x, state["w_buffers"][idx], state["b_buffers"][idx])

    @staticmethod
    def backward(ctx, grad_out):
        tensors = ctx.saved_tensors
        if len(tensors) < 3: return None, None, None, None
        x, weight_cpu, bias_cpu = tensors
        state = _get_device_state(ctx.device)
        idx = state["backward_clk"]
        with torch.cuda.stream(state["transfer_stream"]):
            state["transfer_stream"].wait_event(state["compute_backward_start_event"])
            w = weight_cpu.to(ctx.device, non_blocking=True)
            state["w_bwd_buffers"][idx] = _dequant(w, ctx.dtype)
            state["backward_clk"] ^= 1
            state["transfer_backward_finished_event"].record()
        torch.cuda.current_stream().wait_event(state["transfer_backward_finished_event"])
        state["compute_backward_start_event"].record()
        grad_input = grad_out.to(ctx.dtype) @ state["w_bwd_buffers"][idx]
        grad_weight = grad_out.flatten(0, -2).T @ x.flatten(0, -2) if weight_cpu.requires_grad else None
        grad_bias = grad_out.sum(dim=tuple(range(grad_out.ndim - 1))) if bias_cpu is not None and bias_cpu.requires_grad else None
        return grad_input.to(grad_out.dtype), grad_weight, grad_bias, None

class LinearLayerMemoryManager:
    def __init__(self, m, mgr):
        for p in [m.weight, getattr(m, "bias", None)]:
            if p is not None: p.data = _ensure_cpu_pinned(p.data).detach()
        self.m, self.mgr = m, mgr
        ref = hasattr(m, "ara_lora_ref")
        self.orig = getattr(m.ara_lora_ref(), "org_forward") if ref else m.forward
        def _f(x, *a, **k):
            if a or k: return self.orig(x, *a, **k)
            return _BouncingLinearFn.apply(x, self.m.weight, getattr(self.m, "bias", None), self.mgr.process_device)
        if ref: m.ara_lora_ref().org_forward = _f
        else: m.forward = _f
    @classmethod
    def attach(cls, m, mgr):
        if not hasattr(m, "_layer_memory_manager"): m._layer_memory_manager = cls(m, mgr)

class ConvLayerMemoryManager:
    def __init__(self, m, mgr):
        for p in [m.weight, getattr(m, "bias", None)]:
            if p is not None: p.data = _ensure_cpu_pinned(p.data).detach()
        self.m, self.mgr, s, p, d, g = m, mgr, m.stride, m.padding, m.dilation, m.groups
        ref = hasattr(m, "ara_lora_ref")
        self.orig = getattr(m.ara_lora_ref(), "org_forward") if ref else m.forward
        def _f(x, *a, **k):
            if a or k: return self.orig(x, *a, **k)
            # Для простоты используем стандартный conv если оффлоад не нужен
            return self.orig(x)
        if ref: m.ara_lora_ref().org_forward = _f
        else: m.forward = _f
    @classmethod
    def attach(cls, m, mgr):
        if not hasattr(m, "_layer_memory_manager"): m._layer_memory_manager = cls(m, mgr)
