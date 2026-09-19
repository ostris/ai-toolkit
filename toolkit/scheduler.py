import math
import torch
from typing import Optional
from diffusers.optimization import (
    SchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup as diffusers_get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name is not None:
        name = name.lower().replace("-", "_")

    if name == "cosine":
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
        )
    elif name in ["cosine_with_warmup", "cosine_warmup"]:
        num_training_steps = kwargs.pop('num_training_steps', kwargs.pop('total_iters', None))
        if num_training_steps is None:
            raise ValueError(
                "cosine_with_warmup requires total_iters or num_training_steps to be specified"
            )
        num_training_steps = int(num_training_steps)

        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        num_warmup_steps = int(kwargs.pop('num_warmup_steps'))

        num_cycles = kwargs.pop('num_cycles', 0.5)
        last_epoch = kwargs.pop('last_epoch', -1)

        return diffusers_get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )
    elif name in ["cosine_with_restarts", "cosine_with_hard_restarts"]:
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )
    elif name in ["cosine_with_hard_restarts_with_warmup", "cosine_with_restarts_with_warmup"]:
        num_training_steps = kwargs.pop('num_training_steps', kwargs.pop('total_iters', None))
        if num_training_steps is None:
            raise ValueError(
                "cosine_with_hard_restarts_with_warmup requires total_iters or num_training_steps to be specified"
            )
        num_training_steps = int(num_training_steps)

        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        num_warmup_steps = int(kwargs.pop('num_warmup_steps'))

        num_cycles = kwargs.pop('num_cycles', 1)
        last_epoch = kwargs.pop('last_epoch', -1)

        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )
    elif name == "step":
        kwargs.pop('total_iters', None)
        if 'step_size' not in kwargs:
            kwargs['step_size'] = 1

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
        total_iters = kwargs.pop('total_iters', None)
        if 'num_warmup_steps' not in kwargs and 'warmup_steps' in kwargs:
            kwargs['num_warmup_steps'] = kwargs.pop('warmup_steps')
        if 'warmup_ratio' in kwargs and total_iters is not None:
            kwargs['num_warmup_steps'] = int(total_iters * float(kwargs.pop('warmup_ratio')))
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        # try to use a diffusers scheduler
        print(f"Trying to use diffusers scheduler {name}")
        try:
            scheduler_type = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[scheduler_type]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            f"Scheduler '{name}' is not supported. Supported schedulers: cosine, cosine_with_warmup, cosine_with_restarts, step, linear, constant, constant_with_warmup"
        )
