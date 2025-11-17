import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


class SequentialLRWithStepArg(torch.optim.lr_scheduler.SequentialLR):
    """Wrapper for SequentialLR that handles step arguments for resume compatibility.

    SequentialLR doesn't accept epoch arguments in step(), but the training code
    uses lr_scheduler.step(step_num) to fast-forward the scheduler when resuming.
    This wrapper tracks steps and advances the scheduler appropriately.
    """
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        super().__init__(optimizer, schedulers, milestones, last_epoch)
        self._current_step = 0

    def step(self, epoch=None):
        if epoch is not None and epoch > self._current_step:
            # Fast-forward to the specified step (for resume functionality)
            steps_to_advance = epoch - self._current_step
            for _ in range(steps_to_advance):
                super().step()
                self._current_step += 1
        else:
            # Normal step
            super().step()
            self._current_step += 1


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
        # Handle user-friendly parameters
        total_iters = kwargs.pop('total_iters', None)
        num_cycles = kwargs.pop('num_cycles', None)
        warmup_steps = kwargs.pop('warmup_steps', 0)
        min_lr_ratio = kwargs.pop('min_lr_ratio', None)

        # Calculate T_0 (restart period)
        if 'T_0' not in kwargs:
            if num_cycles is not None and total_iters is not None:
                # T_0 = steps per cycle
                kwargs['T_0'] = max(1, total_iters // num_cycles)
            elif total_iters is not None:
                kwargs['T_0'] = total_iters
            else:
                raise ValueError("cosine_with_restarts requires either T_0 or (num_cycles + total_iters)")

        # Convert min_lr_ratio to eta_min
        if min_lr_ratio is not None and 'eta_min' not in kwargs:
            base_lr = optimizer.param_groups[0]['lr']
            kwargs['eta_min'] = base_lr * min_lr_ratio

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )

        # Wrap with warmup if requested
        if warmup_steps > 0:
            return SequentialLRWithStepArg(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
                    ),
                    cosine_scheduler
                ],
                milestones=[warmup_steps]
            )

        return cosine_scheduler
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
