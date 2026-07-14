"""Minimal process-global ownership for the arena transfer runtime."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ArenaOwnerToken:
    value: object
    device: torch.device


_LOCK = threading.Lock()
_ACTIVE: ArenaOwnerToken | None = None


def normalize_device(device) -> torch.device:
    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        normalized = torch.device("cuda", index)
    return normalized


def acquire_process_owner(device) -> ArenaOwnerToken:
    global _ACTIVE
    requested = normalize_device(device)
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError(
                "arena_runtime_already_active:\n"
                f"active_device={_ACTIVE.device} requested_device={requested}"
            )
        token = ArenaOwnerToken(object(), requested)
        _ACTIVE = token
        return token


def validate_process_owner(token: ArenaOwnerToken | None) -> ArenaOwnerToken:
    with _LOCK:
        if token is None or _ACTIVE is not token:
            raise RuntimeError("arena_runtime_owner_mismatch")
        return token


def release_process_owner(token: ArenaOwnerToken | None) -> None:
    global _ACTIVE
    with _LOCK:
        if token is None:
            return
        if _ACTIVE is not token:
            raise RuntimeError("arena_runtime_owner_mismatch")
        _ACTIVE = None


def active_process_owner() -> ArenaOwnerToken | None:
    with _LOCK:
        return _ACTIVE
