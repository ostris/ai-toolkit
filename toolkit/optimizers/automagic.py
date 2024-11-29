import math
from typing import List
import torch
from toolkit.optimizers.optimizer_utils import Auto8bitTensor, copy_stochastic, stochastic_grad_accummulation
from optimum.quanto import QBytesTensor
import random


class Automagic(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=None,
        min_lr=1e-7,
        max_lr=1e-2,
        lr_momentum=0.9,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        weight_decay=0.0,
        do_paramiter_swapping=False,
        paramiter_swapping_factor=0.1,
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_momentum = lr_momentum

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [
            lr for group in self.param_groups
        ]

        self.is_stochastic_rounding_accumulation = False

        # setup stochastic grad accum hooks
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

        self.do_paramiter_swapping = do_paramiter_swapping
        self.paramiter_swapping_factor = paramiter_swapping_factor
        self._total_paramiter_size = 0
        # count total paramiters
        for group in self.param_groups:
            for param in group['params']:
                self._total_paramiter_size += torch.numel(param)
        # pretty print total paramiters with comma seperation
        print(f"Total training paramiters: {self._total_paramiter_size:,}")

        # needs to be enabled to count paramiters
        if self.do_paramiter_swapping:
            self.enable_paramiter_swapping(self.paramiter_swapping_factor)

    def enable_paramiter_swapping(self, paramiter_swapping_factor=0.1):
        self.do_paramiter_swapping = True
        self.paramiter_swapping_factor = paramiter_swapping_factor
        # call it an initial time
        self.swap_paramiters()

    def swap_paramiters(self):
        all_params = []
        # deactivate all paramiters
        for group in self.param_groups:
            for param in group['params']:
                param.requires_grad_(False)
                # remove any grad
                param.grad = None
                all_params.append(param)
        # shuffle all paramiters
        random.shuffle(all_params)

        # keep activating paramiters until we are going to go over the target paramiters
        target_paramiters = int(
            self._total_paramiter_size * self.paramiter_swapping_factor)
        total_paramiters = 0
        for param in all_params:
            total_paramiters += torch.numel(param)
            if total_paramiters >= target_paramiters:
                break
            else:
                param.requires_grad_(True)

    @staticmethod
    def _get_lr(param_group, param_state):
        lr = param_group["avg_lr"]
        param_scale = 1.0
        return param_scale * lr

    def _get_group_lr(self, group):
        group_lrs = []
        for p in group["params"]:
            if p.grad is not None:
                group_lrs.append(self._get_lr(group, self.state[p]))
        # return avg
        if len(group_lrs) == 0:
            return self.lr
        return sum(group_lrs) / len(group_lrs)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-
                    1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    # adafactor manages its own lr
    def get_learning_rates(self):

        lrs = [
            self._get_group_lr(group)
            for group in self.param_groups
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs

    def get_avg_learning_rate(self):
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.step_hook()
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.to(torch.float32)
                if grad.is_sparse:
                    raise RuntimeError(
                        "Automagic does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored = len(grad_shape) >= 2
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    # store the lr mask
                    state['lr_mask'] = Auto8bitTensor(torch.ones(
                        p.shape).to(p.device, dtype=torch.float32) * self.lr
                    )
                    state['avg_lr'] = torch.mean(
                        state['lr_mask'].to(torch.float32))
                    state['last_polarity'] = torch.zeros(
                        p.shape, dtype=torch.bool, device=p.device)

                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(
                            grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(
                            grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p

                if isinstance(p_data_fp32, QBytesTensor):
                    p_data_fp32 = p_data_fp32.dequantize()
                if p.dtype != torch.float32:
                    p_data_fp32 = p_data_fp32.clone().float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                # lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                eps = group["eps"]
                if isinstance(eps, tuple) or isinstance(eps, list):
                    eps = eps[0]
                update = (grad**2) + eps
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=(1.0 - beta2t))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                # calculate new lr mask. if the updated param is going in same direction, increase lr, else decrease
                # update the lr mask. self.lr_momentum is < 1.0. If a paramiter is positive and increasing (or negative and decreasing), increase lr,
                # for that single paramiter. If a paramiter is negative and increasing or positive and decreasing, decrease lr for that single paramiter.
                # to decrease lr, multiple by self.lr_momentum, to increase lr, divide by self.lr_momentum.

                # not doing it this way anymore
                # update.mul_(lr)

                # Get signs of current last update and updates
                last_polarity = state['last_polarity']
                current_polarity = (update > 0).to(torch.bool)
                sign_agreement = torch.where(
                    last_polarity == current_polarity, 1, -1)
                state['last_polarity'] = current_polarity

                lr_mask = state['lr_mask'].to(torch.float32)

                # Update learning rate mask based on sign agreement
                new_lr = torch.where(
                    sign_agreement > 0,
                    lr_mask / self.lr_momentum,  # Increase lr
                    lr_mask * self.lr_momentum   # Decrease lr
                )

                # Clip learning rates to bounds
                new_lr = torch.clamp(
                    new_lr,
                    min=self.min_lr,
                    max=self.max_lr
                )

                # Apply the learning rate mask to the update
                update.mul_(new_lr)

                state['lr_mask'] = Auto8bitTensor(new_lr)
                state['avg_lr'] = torch.mean(new_lr)

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=(-group["weight_decay"] * new_lr))

                p_data_fp32.add_(-update)

                if p.dtype != torch.float32:
                    # apply stochastic rounding
                    copy_stochastic(p, p_data_fp32)

        return loss
