"""
Device utilities. safe_module_to_device moves a module in-place without using
Module.to(), avoiding PyTorch's swap_tensors path which fails on quantized
parameters (e.g. QLinear) that have requires_grad=False.
"""
from typing import Optional

import torch


def safe_module_to_device(
    module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Move module to device (and optionally dtype) by moving param/buffer .data in-place.
    Avoids Module.to() which uses swap_tensors and can raise on tensors
    that do not require gradients (e.g. QLinear in quantized models).
    """
    for _, param in module.named_parameters(recurse=False):
        if param.device != device or (dtype is not None and param.dtype != dtype):
            param.data = param.data.to(device=device, dtype=dtype or param.dtype)
    for _, buf in module.named_buffers(recurse=False):
        if buf.device != device or (dtype is not None and buf.dtype != dtype):
            buf.data = buf.data.to(device=device, dtype=dtype or buf.dtype)
    for _, child in module.named_children():
        safe_module_to_device(child, device, dtype)
