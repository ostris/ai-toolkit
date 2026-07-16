"""Persistent torch.compile artifacts for the Arena block dispatcher."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import torch


_CACHE_SCHEMA = "aitk-arena-megacache-v1"


def _dynamo_frame_count() -> int:
    try:
        return int(torch._dynamo.utils.counters["frames"].get("total", 0))
    except Exception:
        return 0


def _default_cache_root() -> Path:
    from torch._inductor.runtime.runtime_utils import cache_dir

    return Path(cache_dir()) / "aitk_arena_megacache"


def arena_compile_cache_key(model, config, executor) -> str:
    """Hash the coarse identity of one cumulative Arena compile cache.

    Torch still validates every artifact's graph and guards. This outer key
    separates compiler/runtime policies and immutable dispatcher ABIs while
    allowing train, sample, shape, and residency variants to accumulate.
    """
    from .dispatcher import DISPATCHER_GENERATION

    model_type = type(model)
    programs = getattr(executor, "_programs", {})
    identity = {
        "schema": _CACHE_SCHEMA,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_type": f"{model_type.__module__}.{model_type.__qualname__}",
        "dispatcher": DISPATCHER_GENERATION,
        "programs": sorted(
            (str(mode), str(program.fingerprint))
            for mode, program in programs.items()
        ),
        "compile": {
            "fullgraph": bool(getattr(executor, "compile_fullgraph", False)),
            "dynamic": getattr(executor, "compile_dynamic", True),
            "dynamic_hints": tuple(
                getattr(executor, "compile_dynamic_hints", ()) or ()
            ),
        },
        "fp8": {
            "forward": bool(getattr(config, "fp8_forward", False)),
            "backward": bool(getattr(config, "fp8_backward", False)),
            "sampling": bool(getattr(config, "fp8_sampling", False)),
        },
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ArenaCompileCacheSession:
    """Best-effort load/save around the Arena's lazy compile lifecycle."""

    def __init__(self, path: Path | None, key: str | None):
        self.path = path
        self.key = key
        self.enabled = path is not None
        self.load_attempted = False
        self.loaded = False
        self.saved = False
        self.error: str | None = None
        self.byte_count = 0
        self._last_saved_frames = _dynamo_frame_count()

    @classmethod
    def for_runtime(cls, model, config, executor, *, cache_root=None):
        supported = all(
            callable(getattr(torch.compiler, name, None))
            for name in ("load_cache_artifacts", "save_cache_artifacts")
        )
        if not bool(getattr(config, "compile_blocks", False)) or not supported:
            return cls(None, None)
        key = arena_compile_cache_key(model, config, executor)
        root = Path(cache_root) if cache_root is not None else _default_cache_root()
        return cls(root / f"{key}.torchcompile", key)

    def _warn(self, operation: str, error: BaseException) -> None:
        self.error = f"{type(error).__name__}: {error}"
        warnings.warn(
            f"Arena MegaCache {operation} failed; continuing without it: "
            f"{self.error}",
            RuntimeWarning,
            stacklevel=2,
        )

    def load(self) -> bool:
        if not self.enabled or self.load_attempted:
            return False
        self.load_attempted = True
        if not self.path.is_file():
            return False
        try:
            artifacts = self.path.read_bytes()
            info = torch.compiler.load_cache_artifacts(artifacts)
        except Exception as error:
            self._warn("load", error)
            return False
        self.loaded = info is not None
        self.byte_count = len(artifacts) if self.loaded else 0
        self._last_saved_frames = _dynamo_frame_count()
        return self.loaded

    def save(self, *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        frames = _dynamo_frame_count()
        if not force and frames <= self._last_saved_frames:
            return False
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            result = torch.compiler.save_cache_artifacts()
            if result is None:
                self._last_saved_frames = frames
                return False
            artifacts, _info = result
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(artifacts)
            os.replace(temporary, self.path)
        except Exception as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            self._last_saved_frames = frames
            self._warn("save", error)
            return False
        self.saved = True
        self.error = None
        self.byte_count = len(artifacts)
        self._last_saved_frames = frames
        return True

    def diagnostics(self) -> dict:
        return {
            "enabled": self.enabled,
            "key": self.key,
            "path": None if self.path is None else str(self.path),
            "load_attempted": self.load_attempted,
            "loaded": self.loaded,
            "saved": self.saved,
            "byte_count": self.byte_count,
            "error": self.error,
        }
