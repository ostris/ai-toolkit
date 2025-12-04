"""
Device utilities for managing different accelerator types (CUDA, MPS, CPU).
"""
import torch
import warnings
from typing import Optional, Literal, Union


def get_optimal_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the optimal device for computation, with preference order: CUDA > MPS > CPU.

    Args:
        device: Optional device preference. If provided, will be validated and returned.

    Returns:
        torch.device: The selected device
    """
    if device is not None:
        # If device is explicitly provided, validate and return it
        if isinstance(device, str):
            device = torch.device(device)

        # Validate device availability
        if device.type == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available. Falling back to CPU.", UserWarning)
            return torch.device('cpu')
        elif device.type == 'mps' and not _is_mps_available():
            warnings.warn("MPS requested but not available. Falling back to CPU.", UserWarning)
            return torch.device('cpu')

        return device

    # Auto-detect best available device
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif _is_mps_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def _is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    Returns:
        bool: True if MPS is available
    """
    try:
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except AttributeError:
        return False


def get_device_name(device: Optional[Union[str, torch.device]] = None) -> str:
    """
    Get a human-readable device name.

    Args:
        device: Device to check. If None, uses optimal device.

    Returns:
        str: Human-readable device name
    """
    device = get_optimal_device(device)

    if device.type == 'cuda':
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(device.index)})"
        return "CUDA (unavailable)"
    elif device.type == 'mps':
        if _is_mps_available():
            return "Apple Silicon GPU (MPS)"
        return "MPS (unavailable)"
    else:
        return "CPU"


def empty_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Empty the cache for the specified device type.

    Args:
        device: Device to empty cache for. If None, empties cache for optimal device.
    """
    device = get_optimal_device(device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # MPS doesn't have explicit cache emptying, but PyTorch 2.0+ supports this
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    # Force garbage collection as additional cleanup
    import gc
    gc.collect()


def get_device_memory_info(device: Optional[Union[str, torch.device]] = None) -> dict:
    """
    Get memory information for the specified device.

    Args:
        device: Device to check. If None, uses optimal device.

    Returns:
        dict: Memory information with 'allocated' and 'cached' keys (in GB)
    """
    device = get_optimal_device(device)

    if device.type == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        cached = torch.cuda.memory_reserved(device) / (1024**3)
        return {
            'allocated': allocated,
            'cached': cached,
            'total': torch.cuda.get_device_properties(device).total_memory / (1024**3)
        }
    elif device.type == 'mps' and _is_mps_available():
        # MPS doesn't provide detailed memory info, return what we can
        return {
            'allocated': 'N/A (MPS limitation)',
            'cached': 'N/A (MPS limitation)',
            'total': 'N/A (MPS limitation)'
        }
    else:
        return {
            'allocated': 'N/A (CPU)',
            'cached': 'N/A (CPU)',
            'total': 'N/A (CPU)'
        }


def is_compatible_with_accelerator(model_type: str = "diffusion") -> bool:
    """
    Check if the current setup is compatible with hardware acceleration.

    Args:
        model_type: Type of model (diffusion, transformer, etc.)

    Returns:
        bool: True if acceleration is available
    """
    return torch.cuda.is_available() or _is_mps_available()


def normalize_device_string(device_str: str) -> str:
    """
    Normalize device string to a valid PyTorch device format.

    Args:
        device_str: Device string (e.g., 'cuda', 'mps', 'cpu', 'cuda:0')

    Returns:
        str: Normalized device string
    """
    device_str = device_str.lower().strip()

    # Handle common variations
    if device_str in ['gpu', 'cuda']:
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    elif device_str in ['mps', 'metal', 'apple', 'apple_silicon', 'silicon']:
        if _is_mps_available():
            return 'mps'
        else:
            return 'cpu'
    elif device_str in ['cpu']:
        return 'cpu'

    # Handle specific device indices
    if device_str.startswith('cuda:'):
        if torch.cuda.is_available():
            return device_str
        else:
            return 'cpu'

    return device_str  # Return as-is if we can't normalize


# Convenience aliases
def get_device() -> torch.device:
    """Alias for get_optimal_device() with no arguments."""
    return get_optimal_device()


def clear_cache() -> None:
    """Alias for empty_cache()."""
    empty_cache()


def get_dataloader_kwargs(device: Optional[Union[str, torch.device]] = None) -> dict:
    """
    Get appropriate DataLoader kwargs for the given device.

    Args:
        device: The device to get kwargs for. If None, uses optimal device.

    Returns:
        Dictionary of DataLoader kwargs.
    """
    device = get_optimal_device(device)

    kwargs = {}

    if device.type == 'mps':
        # MPS has limited multiprocessing support and issues with tensor sharing
        # Force single-process dataloader to avoid _share_filename_ errors
        kwargs['num_workers'] = 0
        kwargs['prefetch_factor'] = None
        kwargs['persistent_workers'] = False
    elif device.type == 'cuda':
        # CUDA can handle more workers
        kwargs['num_workers'] = 4
        kwargs['prefetch_factor'] = 2
        kwargs['pin_memory'] = True
        kwargs['persistent_workers'] = True
    else:
        # CPU defaults
        kwargs['num_workers'] = 2
        kwargs['prefetch_factor'] = 2
        kwargs['persistent_workers'] = True

    return kwargs