import os

import torch


def configure_cuda_only_inductor() -> None:
    """Configure CUDA compilation without requiring a CPU toolchain."""
    torch._dynamo.config.suppress_errors = False
    set_stance = getattr(torch.compiler, "set_stance", None)
    if set_stance is not None:
        try:
            # The first AOT-eager pass preserves checkpointing's memory
            # behavior and gives Dynamo real shape evidence before compiling.
            set_stance("aot_eager_then_compile")
        except (RuntimeError, ValueError):
            # Older supported PyTorch builds do not expose this stance.
            pass
    if os.name == "nt":
        # Inductor otherwise dry-compiles a CPU vector-ISA probe even when the
        # requested graph is CUDA-only. This does not enable a CPU fallback.
        from torch._inductor import config as inductor_config

        inductor_config.cpp.vec_isa_ok = False
