import gc
from contextlib import nullcontext
from typing import Optional, Union

import torch


def _as_torch_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return get_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def get_device() -> torch.device:
    """
    Returns the best available device.
    Prioritizes CUDA, then MPS, then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def is_mps_available() -> bool:
    return torch.backends.mps.is_available()


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def empty_cache(device: Optional[Union[str, torch.device]] = None):
    """
    Empties the cache for the selected device.
    """
    target_device = _as_torch_device(device)
    gc.collect()
    if target_device.type == "cuda" and is_cuda_available():
        torch.cuda.empty_cache()
    elif target_device.type == "mps" and is_mps_available():
        torch.mps.empty_cache()


def manual_seed(seed: int, device: Optional[Union[str, torch.device]] = None):
    """
    Sets global seed and device-specific seed when supported.
    """
    target_device = _as_torch_device(device)
    torch.manual_seed(seed)
    if target_device.type == "cuda" and is_cuda_available():
        torch.cuda.manual_seed(seed)
    elif target_device.type == "mps" and is_mps_available():
        torch.mps.manual_seed(seed)


def get_device_name(device: Optional[Union[str, torch.device]] = None) -> str:
    return _as_torch_device(device).type


def autocast(device: Optional[Union[str, torch.device]] = None):
    target_device = _as_torch_device(device)
    if target_device.type in {"cuda", "mps", "cpu"}:
        return torch.autocast(device_type=target_device.type)
    return nullcontext()
