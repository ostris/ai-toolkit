import os

import torch


def configure_cuda_only_inductor() -> None:
    """Configure CUDA compilation without requiring a CPU toolchain."""
    torch._dynamo.config.suppress_errors = False
    if os.name == "nt":
        # Inductor otherwise dry-compiles a CPU vector-ISA probe even when the
        # requested graph is CUDA-only. This does not enable a CPU fallback.
        from torch._inductor import config as inductor_config

        inductor_config.cpp.vec_isa_ok = False


def configure_quantized_compile_tuning(model_config) -> bool | None:
    """Apply an explicit coordinate-descent policy after TorchAO quantization."""
    if not getattr(model_config, "compile", False):
        return None
    if not getattr(model_config, "quantize", False):
        return None

    requested = getattr(model_config, "compile_coordinate_descent", None)
    if requested is None:
        return None

    enabled = bool(requested)
    torch._inductor.config.coordinate_descent_tuning = enabled
    torch._inductor.config.coordinate_descent_check_all_directions = enabled
    return enabled
