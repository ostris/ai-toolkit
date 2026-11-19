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
    previous step, over the tensors that had a non-zero gradient. Tensors whose
    gradient is exactly zero (LoRA downs while their ups are still zero) are
    left out: at the eps floor (~1e-15) they would otherwise become the
    reference whenever they are at least ``sensitivity_quantile`` of the
    elements, and every live tensor's balance would go to ~0.

    The first step only measures (second moments and the reference): without a
    reference the sensitive tensors would take a full v3 step.

    lr controller: a zero-mean vote. Per element, over the last H update
    signs, let k be the number of positive signs, m = (2k - H) / H the mean
    sign and f = flips / (H - 1) the sign-flip rate. The vote is

        (m^2 - 1/H) - (f - 1/2)

    weighted by the balanced |update| (output impact) and pooled per group.
    For independent fair signs (gradient noise) E[m^2] = 1/H and E[f] = 1/2
    exactly, so noise votes 0 and the lr holds. A trend (consistent signs)
    has m^2 above 1/H and few flips: it votes up. Overshoot (signs alternate)
    flips more than half the time: it votes down. The pooled vote is capped at
    ``max_vote`` per step, since the window only fully reflects an lr change
    H - 1 steps later.

    Earlier versions voted (neutral - f) with neutral = 0.5 + asin(step_target
    / 2) / pi = 0.516. Noise flips 50% of the time, so every noise-dominated
    element voted +0.016 per step at any lr and the lr doubled every ~44 steps
    until training diverged (issues #1084, #1091). ``step_target`` is still
    accepted so existing configs load, but it no longer has an effect.

    A parameter whose state is created after training started (its first
    gradient arrives late) starts at its group's adapted lr, not the start lr.
    """

    def __init__(
        self,
        params,
        *args,
        sensitivity_quantile: float = 0.9,
        max_vote: float = 0.05,
        step_target: float = 0.1,
        **kwargs,
    ):
        if not 0.0 < sensitivity_quantile <= 1.0:
            raise ValueError(
                f"sensitivity_quantile ({sensitivity_quantile}) must be in (0, 1]"
            )
        if not max_vote > 0.0:
            raise ValueError(f"max_vote ({max_vote}) must be > 0")
        if not 0.0 < step_target < 2.0:
            raise ValueError(f"step_target ({step_target}) must be in (0, 2)")
        self.sensitivity_quantile = float(sensitivity_quantile)
        self.max_vote = float(max_vote)
        self.step_target = float(step_target)  # unused; kept for config compatibility
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
        self._group_sens_weight: list = [[] for _ in range(n)]

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        super()._init_state(p, group)
        state = self.state[p]
        # A param joining late starts at the group's adapted lr: the parent
        # seeds it from group["lr"], which the controller never moves.
        for q in group["params"]:
            st = self.state.get(q)
            if q is not p and st and "lr" in st:
                state["lr"] = (
                    st["lr"]
                    .to(p.device, torch.float32, copy=True)
                    .clamp_(min=group["min_lr"], max=group["max_lr"])
                )
                break
        gi = self._param_group_index.get(p)
        ref = self._group_ref[gi] if gi is not None else None
        if ref is not None:
            state["sens_ref"] = ref.to(p.device, copy=True)

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
        # exactly-zero gradient: excluded from the reference (weight 0); no sync
        alive = sq.sum() > 0
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
            self._group_sens_weight[gi].append(
                alive.to(torch.float64) * float(p.numel())
            )

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

        # zero-mean vote (see class docstring), pooled per group, applied in .step()
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
            k = bits.sum(0, dtype=torch.int16).to(torch.float32)
            flips = (bits[1:] ^ bits[:-1]).sum(0, dtype=torch.int16)
            del bits
            w = update.abs().view(-1)
            den = w.sum()
            # sum w * [((2k - H) / H)^2 - 1/H - flips / (H - 1) + 1/2]
            num = (
                (w * k.square()).sum() * (4.0 / (H * H))
                - (w * k).sum() * (4.0 / H)
                + den * (1.5 - 1.0 / H)
                - (w * flips).sum() / (H - 1)
            )
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
        # cap the pooled vote at +/- max_vote per step (the parent's cap is 1)
        for gi in range(len(self.param_groups)):
            num = self._group_num[gi]
            if num is not None:
                den = self._group_den[gi].clamp(min=1e-30)
                num.copy_((num / den).clamp_(-self.max_vote, self.max_vote) * den)
        super()._apply_group_votes()

    def _update_sensitivity_refs(self) -> None:
        # numel-weighted quantile of this step's per-tensor log sensitivities
        # over the live tensors; no GPU sync once a reference exists
        for gi, group in enumerate(self.param_groups):
            logs = self._group_sens_log[gi]
            if not logs:
                continue
            dev = logs[0].device
            vals = torch.stack([t.to(dev) for t in logs])
            weights = torch.stack([w.to(dev) for w in self._group_sens_weight[gi]])
            self._group_sens_log[gi] = []
            self._group_sens_weight[gi] = []
            order = vals.argsort()
            cum = weights[order].cumsum(0)
            total = cum[-1:]
            pos = torch.searchsorted(cum, total * self.sensitivity_quantile)
            ref = vals[order][pos.clamp_(max=len(logs) - 1)].squeeze(0).exp_()
            prev = self._group_ref[gi]
            if prev is None:
                if not bool(total > 0):
                    continue  # nothing alive yet: keep measuring
            else:
                # a step where every tensor was dead keeps the previous reference
                ref = torch.where(total.squeeze(0) > 0, ref, prev.to(dev))
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
