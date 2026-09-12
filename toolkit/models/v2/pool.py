"""Process-wide component pool for the live inference server.

While a pool is active (`ComponentPool.current`), `OstrisModelMixin.load_model`
returns an already-resident module for an identical source request instead
of loading it again, and `aitk_post_load` becomes a no-op when the module
already carries the requested policy (qtype / offload / device). Holders
keep calling the normal load API; component reuse across model switches
(same TE or VAE under two archs) falls out of that.

Lifecycle per request: `begin_request()` -> holder loads (hits mark entries
as touched) -> `release_unused()` frees every entry the request did not
touch. `clear()` drops everything.
"""

import gc
import json
import threading
import time
from typing import Dict, Optional

import torch


def _stable(value):
    """JSON-safe, deterministic rendering of a key part."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _stable(to_dict())
        except Exception:
            pass
    if hasattr(value, "__dataclass_fields__"):
        return _stable({k: getattr(value, k) for k in value.__dataclass_fields__})
    return repr(value)


def module_bytes(module: torch.nn.Module) -> int:
    total = 0
    seen = set()
    for t in list(module.parameters()) + list(module.buffers()):
        try:
            ptr = t.untyped_storage().data_ptr()
        except Exception:
            ptr = id(t)
        if ptr in seen or t.device.type == "meta":
            continue
        seen.add(ptr)
        total += t.numel() * t.element_size()
    return total


class PoolEntry:
    def __init__(self, key: str, module: torch.nn.Module, generation: int):
        self.key = key
        self.module = module
        self.cls_name = type(module).__name__
        self.created_at = time.time()
        self.last_used = self.created_at
        self.uses = 0
        self.touched_gen = generation
        self._bytes: Optional[int] = None

    @property
    def bytes(self) -> int:
        if self._bytes is None:
            try:
                self._bytes = module_bytes(self.module)
            except Exception:
                self._bytes = 0
        return self._bytes

    def device(self) -> str:
        try:
            p = next(self.module.parameters())
            return str(p.device)
        except StopIteration:
            return "none"

    def info(self) -> dict:
        try:
            parsed = json.loads(self.key)
        except Exception:
            parsed = {"key": self.key}
        return {
            "cls": self.cls_name,
            "src": parsed.get("src"),
            "subfolder": parsed.get("subfolder"),
            "dtype": parsed.get("dtype"),
            "policy": getattr(self.module, "_aitk_policy", None),
            "bytes": self.bytes,
            "device": self.device(),
            "uses": self.uses,
            "last_used": self.last_used,
        }


class ComponentPool:
    current: Optional["ComponentPool"] = None

    def __init__(self):
        self.entries: Dict[str, PoolEntry] = {}
        self.generation = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.RLock()

    # ---- keys ----
    @staticmethod
    def make_key(cls, name_or_path: str, **parts) -> str:
        key = {"cls": f"{cls.__module__}.{cls.__qualname__}", "src": name_or_path}
        key.update({k: _stable(v) for k, v in parts.items()})
        return json.dumps(key, sort_keys=True, default=repr)

    # ---- request lifecycle ----
    def begin_request(self):
        with self._lock:
            self.generation += 1

    def get(self, key: str) -> Optional[torch.nn.Module]:
        with self._lock:
            entry = self.entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            entry.uses += 1
            entry.last_used = time.time()
            entry.touched_gen = self.generation
            return entry.module

    def put(self, key: str, module: torch.nn.Module):
        with self._lock:
            entry = PoolEntry(key, module, self.generation)
            entry.uses = 1
            self.entries[key] = entry

    def touch_module(self, module: torch.nn.Module):
        with self._lock:
            for entry in self.entries.values():
                if entry.module is module:
                    entry.touched_gen = self.generation
                    entry.last_used = time.time()

    def release_unused(self) -> int:
        """Free every entry not touched during the current request."""
        with self._lock:
            stale = [e for e in self.entries.values() if e.touched_gen != self.generation]
            freed = 0
            for entry in stale:
                freed += entry.bytes
                self._free(entry)
            return freed

    def evict_where(self, predicate) -> int:
        """Free every entry whose module satisfies predicate(module)."""
        with self._lock:
            victims = [e for e in self.entries.values() if predicate(e.module)]
            freed = 0
            for entry in victims:
                freed += entry.bytes
                self._free(entry)
            return freed

    def clear(self):
        with self._lock:
            for entry in list(self.entries.values()):
                self._free(entry)

    def _free(self, entry: PoolEntry):
        from toolkit.memory_management import MemoryManager

        self.entries.pop(entry.key, None)
        self.evictions += 1
        try:
            MemoryManager.free(entry.module)
        except Exception:
            try:
                torch.nn.Module.to(entry.module, "meta")
            except Exception:
                pass
        entry.module = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "resident": [e.info() for e in self.entries.values()],
                "resident_bytes": sum(e.bytes for e in self.entries.values()),
                "untouched": len([e for e in self.entries.values() if e.touched_gen != self.generation]),
            }
