from types import SimpleNamespace
from unittest import mock

import torch

from toolkit.memory_management.arena_offload.compile_cache import (
    ArenaCompileCacheSession,
    arena_compile_cache_key,
)


class _Transformer(torch.nn.Module):
    pass


def _config(**overrides):
    values = {
        "compile_blocks": True,
        "fp8_forward": True,
        "fp8_backward": False,
        "fp8_sampling": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _executor(**overrides):
    values = {
        "compile_fullgraph": True,
        "compile_dynamic": False,
        "compile_dynamic_hints": ((1, 64, 4096),),
        "_programs": {
            "train": SimpleNamespace(fingerprint="train-abi"),
            "sample": SimpleNamespace(fingerprint="sample-abi"),
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cache_key_is_stable_and_covers_compile_policy():
    model = _Transformer()
    config = _config()
    executor = _executor()

    first = arena_compile_cache_key(model, config, executor)
    second = arena_compile_cache_key(model, config, executor)
    changed = arena_compile_cache_key(
        model, config, _executor(compile_fullgraph=False)
    )

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_cache_session_saves_atomically_and_loads(tmp_path):
    session = ArenaCompileCacheSession.for_runtime(
        _Transformer(), _config(), _executor(), cache_root=tmp_path
    )
    info = SimpleNamespace(artifacts={})

    with mock.patch.object(
        torch.compiler,
        "save_cache_artifacts",
        return_value=(b"arena-artifacts", info),
    ) as save_artifacts:
        assert session.save(force=True)

    assert session.path.read_bytes() == b"arena-artifacts"
    assert not list(tmp_path.glob("*.tmp"))
    save_artifacts.assert_called_once_with()

    restored = ArenaCompileCacheSession(session.path, session.key)
    with mock.patch.object(
        torch.compiler, "load_cache_artifacts", return_value=info
    ) as load_artifacts:
        assert restored.load()

    load_artifacts.assert_called_once_with(b"arena-artifacts")
    assert restored.diagnostics()["byte_count"] == len(b"arena-artifacts")


def test_cache_io_failure_is_non_fatal(tmp_path):
    session = ArenaCompileCacheSession.for_runtime(
        _Transformer(), _config(), _executor(), cache_root=tmp_path
    )
    with mock.patch.object(
        torch.compiler,
        "save_cache_artifacts",
        side_effect=OSError("synthetic write failure"),
    ), mock.patch("toolkit.memory_management.arena_offload.compile_cache.warnings.warn"):
        assert not session.save(force=True)

    assert "synthetic write failure" in session.diagnostics()["error"]
