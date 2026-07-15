from types import SimpleNamespace

import torch

from toolkit.compile_utils import configure_quantized_compile_tuning


def test_quantized_compile_tuning_preserves_torchao_default_when_unset():
    config = SimpleNamespace(compile=True, quantize=True)
    original_tuning = torch._inductor.config.coordinate_descent_tuning
    original_directions = (
        torch._inductor.config.coordinate_descent_check_all_directions
    )
    try:
        result = configure_quantized_compile_tuning(config)
        assert result is None
        assert torch._inductor.config.coordinate_descent_tuning == original_tuning
        assert (
            torch._inductor.config.coordinate_descent_check_all_directions
            == original_directions
        )
    finally:
        torch._inductor.config.coordinate_descent_tuning = original_tuning
        torch._inductor.config.coordinate_descent_check_all_directions = (
            original_directions
        )


def test_quantized_compile_tuning_can_disable_torchao_search():
    config = SimpleNamespace(
        compile=True,
        quantize=True,
        compile_coordinate_descent=False,
    )
    original_tuning = torch._inductor.config.coordinate_descent_tuning
    original_directions = (
        torch._inductor.config.coordinate_descent_check_all_directions
    )
    try:
        assert configure_quantized_compile_tuning(config) is False
        assert torch._inductor.config.coordinate_descent_tuning is False
        assert (
            torch._inductor.config.coordinate_descent_check_all_directions is False
        )
    finally:
        torch._inductor.config.coordinate_descent_tuning = original_tuning
        torch._inductor.config.coordinate_descent_check_all_directions = (
            original_directions
        )
