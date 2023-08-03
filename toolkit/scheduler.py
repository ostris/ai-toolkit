import torch
from typing import Optional


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        max_iterations: Optional[int],
        lr_min: Optional[float],
        **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, end_factor=0.5, total_iters=max_iterations, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )
