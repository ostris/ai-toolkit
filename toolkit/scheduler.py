import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
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
    elif name == "polynomial":
        # --- 1. Obtain base values ---
        lr_start = optimizer.param_groups[0]['lr']
        # Defult steps equals to 3000
        total_steps = kwargs.get('total_iters', 3000) 
        
        # --- 2. Intelligent Processing of Terminal Learning Rate ---
        # Prioritize reading lr_end from the config file. If not present, default to 10% of the initial value. 
        # even if you later change the initial LR, it will automatically recalculate the endpoint proportionally.
        default_lr_end = lr_start * (5e-5 / 5e-4) 
        lr_end = kwargs.pop('lr_end', default_lr_end)
        # ---3.Power Calculation: Smooth Deceleration---
        power = kwargs.get('power', 0.8)
        # ---4.Calculation---
        # Preventing logical errors where the start point is less than or equal to the end point
        if lr_start > lr_end:
            ratio = lr_end / lr_start
            # Calculate the total virtual steps required for the curve to precisely land at lr_end after total_steps
            # Fomula：T_{max} = frac{TotalSteps}{1 - (LR_{end} / LR_{start})^{1/power}}
            t_max = total_steps / (1 - pow(ratio, 1/power))
            kwargs['total_iters'] = int(t_max)
        else:
            # If the configuration is incorrect (the endpoint is higher than the starting point), it will downgrade to Constant mode to prevent crashes
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        return torch.optim.lr_scheduler.PolynomialLR(
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
