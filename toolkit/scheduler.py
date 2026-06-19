import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        import math

        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')

        total_steps = kwargs.get('T_max')
        warmup_steps = kwargs.pop('num_warmup_steps', 0)
        eta_min = kwargs.get('eta_min', 0.0)

        if warmup_steps > 0:
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                else:
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    base_ratio = eta_min / max(1e-8, 1.0)
                    return base_ratio + (1.0 - base_ratio) * cosine_factor

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

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
        if 'factor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    elif name == 'constant_with_warmup':
        # see if num_warmup_steps is in kwargs
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        del kwargs['total_iters']
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        # try to use a diffusers scheduler
        print(f"Trying to use diffusers scheduler {name}")
        try:
            name = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )
