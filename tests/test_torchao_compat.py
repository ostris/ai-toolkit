import torch

import toolkit.quantization.torchao_compat as torchao_compat
from toolkit.quantization.torchao_compat import (
    _release_tuple,
    intx_weight_only_config,
    torchao_arena_fp8_supported,
)


def test_release_tuple_ignores_local_and_prerelease_suffixes():
    assert _release_tuple("0.17.0+cu132") == (0, 17, 0)
    assert _release_tuple("0.17.0.dev20260715") == (0, 17, 0)
    assert _release_tuple("unknown") == ()


def test_arena_fp8_requires_tested_version_and_tensor_format():
    assert not torchao_arena_fp8_supported(
        "0.10.0", float8_tensor_available=True
    )
    assert not torchao_arena_fp8_supported(
        "0.17.0", float8_tensor_available=False
    )
    assert torchao_arena_fp8_supported(
        "0.17.0", float8_tensor_available=True
    )


def test_current_intx_config_factory_is_available():
    assert intx_weight_only_config(4) is not None


def test_torchao_010_uintx_config_factory_remains_supported(monkeypatch):
    class LegacyUIntXConfig:
        def __init__(self, dtype):
            self.dtype = dtype

    monkeypatch.setattr(torchao_compat, "_IntxConfig", None)
    monkeypatch.setattr(torchao_compat, "_UIntXConfig", LegacyUIntXConfig)

    config = intx_weight_only_config(4)
    assert isinstance(config, LegacyUIntXConfig)
    assert config.dtype is torch.uint4
