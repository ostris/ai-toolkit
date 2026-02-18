"""
Debug utilities. memory_debug context manager measures GPU/RAM around a block;
enabled via set_debug_config(config) with config.debug. Extensible to RAM later.
"""
import contextlib
from typing import Callable

import torch

_debug_config = None


def set_debug_config(config) -> None:
    """Register the config object used to decide if memory debug is enabled (config.debug)."""
    global _debug_config
    _debug_config = config


def is_debug_enabled() -> bool:
    """Return True if debug logging is enabled (config.debug). Used for optional debug messages."""
    if _debug_config is None:
        return False
    return bool(getattr(_debug_config, "debug", False))


def _is_enabled_for_cuda() -> bool:
    if _debug_config is None:
        return False
    if not getattr(_debug_config, "debug", False):
        return False
    return torch.cuda.is_available()


def _cuda_snapshot_mb():
    """Return (allocated_mb, max_allocated_mb)."""
    return (
        torch.cuda.memory_allocated() / 2**20,
        torch.cuda.max_memory_allocated() / 2**20,
    )


def _format_cuda_diff(label: str, before: tuple, after: tuple) -> list:
    mem_before, max_before = before
    mem_after, max_after = after
    delta = mem_before - mem_after
    delta_str = f"(freed {delta:.1f} MB)" if delta >= 0 else f"(+{-delta:.1f} MB)"
    return [
        f"[DEBUG {label}] CUDA allocated: {mem_before:.1f} MB -> {mem_after:.1f} MB {delta_str}",
        f"[DEBUG {label}] CUDA max:       {max_before:.1f} MB -> {max_after:.1f} MB",
    ]


@contextlib.contextmanager
def memory_debug(
    print_fn: Callable[[str], None],
    label: str,
    kind: str = "cuda",
):
    """
    Context manager: measure memory around the block and log if debug is enabled.
    enabled is read from the config set via set_debug_config(); no need to pass it.
    kind="cuda" measures CUDA allocated/max; other kinds (e.g. "ram") are stubs for now.
    """
    if kind != "cuda":
        yield
        return
    if not _is_enabled_for_cuda():
        yield
        return
    before = _cuda_snapshot_mb()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        after = _cuda_snapshot_mb()
        for line in _format_cuda_diff(label, before, after):
            print_fn(line)


def cuda_memory_debug(print_fn: Callable[[str], None], label: str):
    """Alias for memory_debug(print_fn, label, kind="cuda")."""
    return memory_debug(print_fn, label, kind="cuda")
