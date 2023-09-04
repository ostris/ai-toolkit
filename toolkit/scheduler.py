import torch
from typing import Optional


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,  **kwargs
        )
    elif name == "cosine_with_restarts":
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )
    elif name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif name == "constant":
        if 'facor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )
