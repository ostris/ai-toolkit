"""
NOTE: This is experimental and under active development; expect breaking changes and bugs. Feedback welcome.
"""

import math
import torch

from toolkit.optimizers.automagic3 import Automagic3


class Automagic4(Automagic3):
    """
    Automagic v4: Automagic3 with sensitivity-balanced updates.

    v3 gives every element an Adam-normalized step of about lr, so how far one
    step moves the OUTPUT scales with that weight's sensitivity. A single shared
    lr cannot fit a model whose tensors differ in sensitivity by orders of
    magnitude (a head reading a huge raw residual, a modulation shared by every
    layer): at the lr the bulk needs, the sensitive ones overshoot.

    v4 multiplies each element's (clipped) v3 update by

        balance = min(1, reference / sensitivity)

    sensitivity = sqrt(bias-corrected second moment). For a noise-dominated MSE
    loss (diffusion) gradient power tracks the Gauss-Newton curvature, i.e. how
    far the output moves per unit change of the weight. It comes from the same
    row/col (or 1D) state v3 keeps, so it is free and per element: per column
    for a head whose input dims differ in scale, per row for a modulation.
    Above the reference the step becomes Newton-like (grad / v), which is
    invariant to rescaling the weights -- including coupled Q/K-style
    rescalings, so it cannot seed the fights the shared group lr prevents.

    One-sided on purpose: low gradient power cannot be told apart from a weight
    that is gated off (a LoRA down while its zero-initialized up is still ~0,
    dead units), where Newton would take an unbounded step. Those keep v3's
    normalized step, which is what gets zero-initialized LoRA pairs moving.

    reference: the ``sensitivity_quantile`` (default 0.9) quantile of the
    per-tensor geometric-mean sensitivity, weighted by element count, from the
    previous step. High so a large near-dead mass (LoRA downs at init are about
    half of a LoRA's elements) cannot drag it down and stall everything above
    it; the outliers it targets are a few percent of elements.

    The first step only measures (second moments and the reference): without a
    reference the sensitive tensors would take a full v3 step.

    lr controller: re-centered flip-rate vote. v3's neutral point is a 50% sign
    flip rate, but a weight bouncing on gradient noise at its minimum has
    lag-1 gradient correlation -effective_lr * curvature / 2 at any lr > 0, so
    v3 only balances at lr = 0 and anneals once the signal runs out. v4 votes
    (neutral - observed) flip rate per element, with

        neutral = 0.5 + asin(step_target / 2) / pi

    the flip rate of a stationary weight whose effective_lr * curvature equals
    ``step_target`` (default 0.1; 2 is the stability edge). The lr then has a
    fixed point set by step_target, with or without signal: trends flip less
    and raise it, overshoot flips more and lowers it. Votes are weighted by the
    balanced |update| (output impact). No gain: near the fixed point the flip
    rate moves ~0.16 * step_target per e-fold of lr, so the loop gain over the
    H-step window lag stays well below 1.
    """

    def __init__(
        self,
        params,
        *args,
        sensitivity_quantile: float = 0.9,
        step_target: float = 0.1,
        **kwargs,
    ):
        if not 0.0 < sensitivity_quantile <= 1.0:
            raise ValueError(
                f"sensitivity_quantile ({sensitivity_quantile}) must be in (0, 1]"
            )
        if not 0.0 < step_target < 2.0:
            raise ValueError(f"step_target ({step_target}) must be in (0, 2)")
        self.sensitivity_quantile = float(sensitivity_quantile)
        self.step_target = float(step_target)
        self._neutral_flip = 0.5 + math.asin(self.step_target / 2.0) / math.pi
        super().__init__(params, *args, **kwargs)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _ema_(
        state_t: torch.Tensor, sample: torch.Tensor, beta2: float
    ) -> torch.Tensor:
        # fp32 beta2-EMA; low-precision state is written back
        v = state_t if state_t.dtype == torch.float32 else state_t.to(torch.float32)
        v.mul_(beta2).add_(sample, alpha=1.0 - beta2)
        if v is not state_t:
            state_t.copy_(v)
        return v

    def _rebuild_group_index(self) -> None:
        super()._rebuild_group_index()
        n = len(self.param_groups)
        self._group_ref: list = [None] * n
        self._group_sens_log: list = [[] for _ in range(n)]
        self._group_sens_numel: list = [[] for _ in range(n)]

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        super()._init_state(p, group)
        gi = self._param_group_index.get(p)
        ref = self._group_ref[gi] if gi is not None else None
        if ref is not None:
            self.state[p]["sens_ref"] = ref.to(p.device, copy=True)

    # -------------------------------------------------------------- per-param

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic4 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = grad * grad
        if p.dim() >= 2:
            row = self._ema_(state["exp_avg_sq_row"], sq.mean(dim=-1).add_(eps), beta2)
            col = self._ema_(state["exp_avg_sq_col"], sq.mean(dim=-2).add_(eps), beta2)
            del sq
            # mean log of the factored v = row (x) col / mean(row)
            log_v = (
                row.log().mean(-1) - row.mean(-1).log() + col.log().mean(-1)
            ).mean()
            inv_sqrt_v = self._approx_sq_grad(row, col)
        else:
            v = self._ema_(state["exp_avg_sq"], sq, beta2).add(eps)
            del sq
            log_v = v.log().mean()
            inv_sqrt_v = v.rsqrt_()
        state["step"] += 1
        bias_corr = 1.0 - beta2 ** state["step"]

        gi = self._param_group_index.get(p)
        if gi is not None:
            self._group_sens_log[gi].append(0.5 * (log_v - math.log(bias_corr)))
            self._group_sens_numel[gi].append(p.numel())

        ref = state.get("sens_ref")
        if ref is None:
            # measurement-only first step (no reference yet)
            p.grad = None
            return

        update = grad.mul_(inv_sqrt_v)
        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])
        # balance = min(1, ref / sensitivity), sensitivity = sqrt(v / bias_corr)
        balance = inv_sqrt_v.mul_(ref * math.sqrt(bias_corr)).clamp_(max=1.0)
        update.mul_(balance)
        del balance, inv_sqrt_v

        # flip-rate vote (see class docstring), pooled per group, applied in .step()
        cur_bits = update.gt(0.0)
        hist = state["sign_history"]
        idx = state["hist_idx"]
        H = hist.shape[0]
        lr_t = state["lr"]
        hist[idx].copy_(self._pack_bits(cur_bits))
        state["hist_idx"] = (idx + 1) % H
        fill = min(H, state["hist_fill"] + 1)
        state["hist_fill"] = fill

        if fill == H and gi is not None:
            _, shifts = self._pack_consts(hist.device)
            chron = torch.roll(hist, -state["hist_idx"], dims=0)
            bits = (
                (chron.unsqueeze(-1) >> shifts)
                .bitwise_and_(1)
                .view(H, -1)[:, : update.numel()]
            )
            flips = (bits[1:] ^ bits[:-1]).sum(0, dtype=torch.int16)
            w = update.abs().view(-1)
            den = w.sum()
            num = den * self._neutral_flip - (w * flips).sum() / (H - 1)
            if self._group_num[gi] is None:
                self._group_num[gi] = num
                self._group_den[gi] = den
            else:
                acc = self._group_num[gi]
                if num.device != acc.device:
                    num = num.to(acc.device)
                    den = den.to(acc.device)
                acc.add_(num)
                self._group_den[gi].add_(den)

        wd = group["weight_decay"]
        if p.dtype == torch.float32:
            if wd != 0.0:
                update.add_(p, alpha=wd)
            p.addcmul_(update, lr_t, value=-1.0)
        else:
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.add_(new_p_fp32, alpha=wd)
            new_p_fp32.addcmul_(update, lr_t, value=-1.0)
            self._stochastic_copy_(p, new_p_fp32)

        p.grad = None

    # ----------------------------------------------------------- optimizer API

    def _apply_group_votes(self) -> None:
        self._update_sensitivity_refs()
        super()._apply_group_votes()

    def _update_sensitivity_refs(self) -> None:
        # numel-weighted quantile of this step's per-tensor log sensitivities; no GPU sync
        for gi, group in enumerate(self.param_groups):
            logs = self._group_sens_log[gi]
            if not logs:
                continue
            dev = logs[0].device
            vals = torch.stack([t.to(dev) for t in logs])
            weights = torch.tensor(
                self._group_sens_numel[gi], dtype=torch.float64, device=dev
            )
            order = vals.argsort()
            cum = weights[order].cumsum(0)
            pos = torch.searchsorted(cum, cum[-1:] * self.sensitivity_quantile)
            ref = vals[order][pos.clamp_(max=len(logs) - 1)].squeeze(0).exp_()
            self._group_ref[gi] = ref
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                cur = st.get("sens_ref")
                if cur is None:
                    st["sens_ref"] = ref.to(p.device, copy=True)
                else:
                    cur.copy_(ref)
            self._group_sens_log[gi] = []
            self._group_sens_numel[gi] = []

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # parent cast the reference to param dtype; restore fp32 and the group mirror
        for gi, group in enumerate(self.param_groups):
            for p in group["params"]:
                st = self.state.get(p)
                ref = st.get("sens_ref") if st is not None else None
                if isinstance(ref, torch.Tensor):
                    st["sens_ref"] = ref.to(torch.float32)
                    if self._group_ref[gi] is None:
                        self._group_ref[gi] = st["sens_ref"].clone()
