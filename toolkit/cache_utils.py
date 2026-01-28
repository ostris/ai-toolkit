"""Utilities for content-aware cache filenames and atomic writes.

Small, dependency-free helpers used by cache invalidation and atomic writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute full SHA-256 hex digest of file at `path` using chunked reads."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_param_digest(params: Mapping[str, Any], length: int = 12) -> str:
    """Compute a stable SHA-256 hex digest of params and return truncated prefix.

    Uses deterministic JSON serialization (sorted keys, compact separators).
    """
    canonical = json.dumps(params, sort_keys=True, separators=(',', ':')).encode('utf-8')
    h = hashlib.sha256(canonical).hexdigest()
    return h[:length]


def compute_combined_hash(paths: Iterable[Path], chunk_size: int = 1 << 20) -> str:
    """Compute combined SHA-256 over the bytes of the files in *deterministic* order.

    Implementation: compute each file's SHA-256, concatenate the hex digests in order
    and hash that concatenated string to produce a combined digest.
    """
    digests = []
    for p in paths:
        if not p.exists():
            digests.append('')
        else:
            digests.append(compute_file_sha256(p, chunk_size=chunk_size))
    combined = ''.join(digests).encode('utf-8')
    return hashlib.sha256(combined).hexdigest()


def cache_filename(base: str, param_digest: str, content_hex: str, ext: str) -> str:
    if not ext.startswith('.'):
        ext = '.' + ext
    return f"{base}_{param_digest}_{content_hex}{ext}"


def atomic_write(target: Path, write_fn: Callable[[Path], None], fsync: bool = True) -> None:
    """Atomically write a file to `target` using a temporary file in the same directory.

    write_fn(tmp_path) should write to the supplied Path. On success, the temporary
    file will be moved into place using os.replace(). On failure the tmp file will
    be removed.
    """
    target = Path(target)
    tmp_dir = target.parent
    tmp_file = None
    # create a unique tmp filename in same dir to ensure os.replace is atomic
    fd = None
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.tmp.", dir=str(tmp_dir))
        tmp_path = Path(tmp_path)
        os.close(fd)  # we'll let write_fn open the path
        # call user-provided writer
        write_fn(tmp_path)
        if fsync:
            # fsync file to ensure data durability. Use best-effort fsync and tolerate
            # platforms where a particular fsync call may fail.
            try:
                with tmp_path.open('r+b') as f:
                    os.fsync(f.fileno())
            except Exception:
                # best-effort: ignore fsync failure on some platforms/handles
                pass
            try:
                dir_fd = os.open(str(tmp_dir), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                # best-effort: ok if fsync dir fails on some platforms
                pass
        os.replace(str(tmp_path), str(target))
    except Exception:
        # cleanup tmp file on any failure
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def find_cached_file(expected_path: Path, legacy_fallback: bool = True) -> Optional[Path]:
    """Return a Path to the cached file.

    If `expected_path` exists return it. Otherwise, if `legacy_fallback` is True,
    attempt to find a legacy-named cache file by searching for files that start with
    the same base (prefix before first underscore) and return the best candidate
    (most recent modification time) or None.
    """
    expected_path = Path(expected_path)
    if expected_path.exists():
        return expected_path
    if not legacy_fallback:
        return None
    parent = expected_path.parent
    if not parent.exists():
        return None
    # prefix before first underscore in the expected filename
    base_name = expected_path.name.split('_')[0]
    # get the expected file extension
    expected_ext = expected_path.suffix.lower()
    candidates = []
    try:
        for p in parent.iterdir():
            if not p.is_file():
                continue
            # skip temporary files (atomic_write creates .tmp.* files)
            if p.name.startswith('.') and '.tmp.' in p.name:
                continue
            # only consider files with matching extension to avoid picking up temp files
            if expected_ext and p.suffix.lower() != expected_ext:
                continue
            if p.name.startswith(base_name + '_'):
                candidates.append(p)
    except Exception:
        return None
    if not candidates:
        return None
    # pick most recently modified candidate as heuristic
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def wait_for_cached_file(expected_path: Path, timeout: float = 5.0, poll_interval: float = 0.1, legacy_fallback: bool = True) -> Optional[Path]:
    """Wait up to `timeout` seconds for a cached file to appear and be stable.

    Returns the Path if found and stable, otherwise None.
    Stability heuristic: file exists and its size is unchanged across two polls.
    """
    import time

    deadline = time.time() + float(timeout)
    last_size = None
    while time.time() < deadline:
        candidate = find_cached_file(Path(expected_path), legacy_fallback=legacy_fallback)
        if candidate is None:
            time.sleep(poll_interval)
            continue
        try:
            stat = candidate.stat()
            size = stat.st_size
            # If size is 0, it's still possibly being written; wait for stability
            if last_size is None:
                last_size = size
                time.sleep(poll_interval)
                continue
            if size == last_size:
                return candidate
            last_size = size
        except FileNotFoundError:
            # race: file disappeared between discovery and stat — retry
            time.sleep(poll_interval)
            continue
        except Exception:
            time.sleep(poll_interval)
            continue
    return None
