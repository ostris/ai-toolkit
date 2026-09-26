"""Unified memory (integrated GPU) support, e.g. NVIDIA DGX Spark (GB10).

On these devices the CPU and GPU share the same physical memory, so moving a
module to the CPU frees nothing. Worse, device -> host copies are extremely
slow (~100MB/s on GB10), so every "offload to cpu" of a large transformer or
text encoder stalls for minutes. When a unified memory device is detected we
make module level moves to the CPU a no-op (dtype casts still apply). Tensor
level .cpu()/.to('cpu') is left alone since saving, numpy, etc. need it.

Set AITK_DISABLE_UNIFIED_MEMORY=1 to turn this off.
"""
import os
import sys

import torch

_patched = False


def is_unified_memory() -> bool:
    if os.environ.get("AITK_DISABLE_UNIFIED_MEMORY", "0") == "1":
        return False
    # linux only (DGX Spark). The Windows RTX Spark stack does not behave the same way
    if not sys.platform.startswith("linux"):
        return False
    # NVIDIA only. ROCm APUs report integrated too, but may have a dedicated VRAM carveout
    if torch.version.hip is not None:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        return bool(torch.cuda.get_device_properties(0).is_integrated)
    except Exception:
        return False


def apply_unified_memory_patches():
    global _patched
    if _patched or not is_unified_memory():
        return
    _patched = True

    orig_to = torch.nn.Module.to

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is None or device.type != "cpu":
            return orig_to(self, *args, **kwargs)
        # drop the cpu move, keep any dtype / memory format conversion
        new_kwargs = {}
        if dtype is not None:
            new_kwargs["dtype"] = dtype
        if memory_format is not None:
            new_kwargs["memory_format"] = memory_format
        if not new_kwargs:
            return self
        return orig_to(self, non_blocking=non_blocking, **new_kwargs)

    def cpu(self):
        return self

    torch.nn.Module.to = to
    torch.nn.Module.cpu = cpu
    print(f"Unified memory device detected ({torch.cuda.get_device_name(0)}): module moves to cpu are disabled")
