"""Persist torch.compile's Mega-Cache (inductor + autotune + AOTAutograd artifacts) to a single
portable file, so a compiled unet can skip cold-start recompilation across runs / container
migrations / cloud restarts.

torch's default `fx_graph_cache` already persists the inductor cache to /tmp, so on a *persistent*
local machine the 2nd run is automatically faster. This module is for the case that breaks: an
ephemeral / cloud / fresh-container run where /tmp is wiped, or a 1-GPU warm-up whose result you
want to carry into a multi-GPU run. Carry `compile_cache.bin` alongside the job output and the
migrated process loads it instead of recompiling every triton kernel (the several-minute
"first forward will hang" stall).

Everything here is best-effort: any failure is logged and swallowed so a missing / stale / partial
cache never breaks training. Mismatched entries (different GPU arch, torch version, dtype, shape)
are content-hash keyed inside the blob -> they simply miss and recompile, never load a wrong kernel.

Requires the target GPU arch to be present to *build* kernels (autotune + ptxas), so a cache can
only be produced on a real GPU of the same arch you will run on -- a CPU/no-GPU process can't
pre-bake the kernels. 1-GPU -> N-GPU is fine (every DDP rank compiles the same graph and can load
the same blob).
"""
import os
import torch


def _artifact_count(info):
    try:
        return sum(len(v) for v in info.artifacts.values())
    except Exception:
        return "?"


def load_compile_cache(path, log=print):
    """Load a previously saved mega-cache blob into torch.compile's in-memory caches.

    Must be called BEFORE the first forward triggers compilation. Returns True on a successful load.
    """
    fn = getattr(torch.compiler, "load_cache_artifacts", None)
    if fn is None:
        log("torch.compiler.load_cache_artifacts unavailable (torch too old); skipping compile cache")
        return False
    if not path or not os.path.exists(path):
        log(f"No torch.compile mega-cache at {path}; will compile from scratch")
        return False
    try:
        with open(path, "rb") as fp:
            blob = fp.read()
        info = fn(blob)
        log(f"Loaded torch.compile mega-cache from {path} "
            f"({len(blob) / 1e6:.1f} MB, {_artifact_count(info)} artifacts)")
        return True
    except Exception as e:
        log(f"Could not load torch.compile mega-cache ({type(e).__name__}: {e}); will recompile as needed")
        return False


def save_compile_cache(path, log=print):
    """Serialize the current in-memory torch.compile caches to `path` (atomic replace).

    Safe to call repeatedly (e.g. at every checkpoint): each call captures everything compiled so
    far, so the file converges to a complete cache once all bucket shapes have been seen. The
    artifact count in the log lets you watch that convergence. Returns True on a successful write.
    """
    fn = getattr(torch.compiler, "save_cache_artifacts", None)
    if fn is None:
        return False
    try:
        art = fn()
        if art is None:
            return False  # nothing has been compiled yet -- nothing to persist
        blob, info = art
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fp:
            fp.write(blob)
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves a corrupt cache
        log(f"Saved torch.compile mega-cache to {path} "
            f"({len(blob) / 1e6:.1f} MB, {_artifact_count(info)} artifacts)")
        return True
    except Exception as e:
        log(f"Could not save torch.compile mega-cache ({type(e).__name__}: {e})")
        return False
