"""
NOTE: This is experimental and under active development; expect breaking changes and bugs. Feedback welcome.
"""
import math
from typing import List
import torch


class Automagic3(torch.optim.Optimizer):
    """
    Automagic v3.

    A learning rate is kept per row of each parameter: one lr per output
    channel for >=2D weights (e.g. one lr per output neuron of a Linear layer)
    and one lr per element for 1D weights (biases, norms). Each step the lr is
    nudged by how consistent the per-element update direction has been.

    Agreement is window-consistency: over the last ``polarity_history_count``
    update-sign snapshots plus the current sign, the per-element score is the
    fraction sharing the dominant sign (``max(p, 1-p)``, in [0.5, 1]; direction-
    agnostic, and the current step is just one vote so a single flip against a
    long consistent history barely moves it). It is reduced to one value per row.
    ``polarity_history_count`` is min 1 (default 2); a longer window makes the lr
    react to a sustained trend rather than a single noisy step.

    The agreement maps to a direction in [-1, 1] measured relative to the
    window's noise floor ``b`` -- the consistency a pure-noise row shows purely
    by chance, computed automatically from the window size (e.g. 0.75 for a
    window of <=3 snapshots, ~0.69 for 4). Agreement == ``b`` -> 0 (lr steady),
    1.0 -> +1 (lr up), below ``b`` -> down. There is no manual target: holding at
    the noise floor is self-balancing per row -- the lr grows while a row is more
    consistent than chance and settles at whatever lr makes its consistency meet
    ``b``, so noisy and clean layers each find their own operating point without
    tuning, and it neither collapses to min_lr nor runs to max_lr. Measuring
    relative to ``b`` also makes the signal window-size independent. The direction
    scales a multiplicative (geometric) bump ``lr *= exp(direction *
    lr_bump_rate)`` -- a fixed fractional move, uniform across the whole range.
    lr is clamped to [min_lr, max_lr].

    With ``fused=True`` (default) the step is fused into the backward pass via
    ``register_post_accumulate_grad_hook``: each parameter is updated and its
    grad freed as soon as autograd finishes accumulating into it. ``.step()``
    therefore does no real work and peak VRAM stays low. Note this bypasses the
    trainer's grad clipping / nan-skip (they run after backward) and is not
    compatible with multi-backward gradient accumulation.

    With ``fused=False`` it behaves like a traditional optimizer: grads
    accumulate across backward passes and the update happens in ``.step()``.
    Low-precision (bf16/fp16) grads are accumulated with stochastic rounding so
    small per-micro-batch grads aren't lost; fp32 grads accumulate normally.

    Second-moment EMA state is stored in ``p.dtype`` (math runs in fp32 when
    the state is lower precision). Updates to low-precision (e.g. bf16/fp16)
    parameters are applied in fp32 and stochastically rounded on write-back.

    Improvements over v2
    --------------------
    1. Per-row learning rate (was a single scalar per parameter tensor).
       v2 kept one lr for an entire weight matrix; v3 keeps one per output
       channel (per element for 1D params). Plain English: different neurons in
       the same layer can now learn at different speeds instead of being forced
       to share one rate, so a layer where some rows have converged and others
       have not is handled gracefully.

    2. Proportional lr control (was a hard threshold flip). v2 bumped the lr up
       or down by a fixed amount depending on whether agreement crossed a
       threshold, which jitters when agreement hovers near the boundary. v3
       scales the bump by how far consistency is from its hold point relative to
       the noise floor (direction in [-1, 1]). Plain English: the lr nudges gently when the
       signal is weak and firmly when it is strong, and parks itself instead of
       oscillating when gradients are basically noise.

    3. Multiplicative (geometric) lr bump (was additive). v2 added/subtracted a
       fixed absolute amount, so the same bump was a huge relative jump near
       min_lr and a negligible one near max_lr. v3 multiplies by
       ``exp(signal * lr_bump_rate)`` -- a fixed *percentage* step. Plain
       English: the lr moves at the same relative pace whether it is tiny or
       large, traverses its whole range in a predictable number of steps, and a
       full up bump is exactly cancelled by a full down bump (no drift). The
       knob was renamed ``lr_bump`` -> ``lr_bump_rate`` to signal the change.

    4. Stochastic rounding for fp16, not just bf16. v2 only rounded bf16
       write-backs and let fp16 fall back to round-to-nearest, silently
       discarding updates smaller than an fp16 ULP. v3 stochastically rounds
       both (fast bit-trick for bf16/fp16, generic fallback for other low
       precisions). Plain English: fp16 training no longer throws away small
       weight updates, so it actually keeps learning instead of stalling.

    5. Faster hot path, identical math. eps is folded into the small reduced
       row/col vectors instead of the full gradient-square tensor; the lr scale,
       weight decay and parameter update are fused into one ``addcmul_``; and the
       sign-agreement is summed straight off the bool mask with no full-size
       float cast. Plain English: each step issues fewer GPU passes over the
       weights, so it runs faster (notably in bf16/fp16) without changing the
       result.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-8,
        max_lr: float = 1e-2,
        lr_bump_rate: float = 0.1,  # fractional/log step per bump (~10%); see step logic
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        polarity_history_count: int = 3,  # update-sign snapshots kept; current is compared vs all (min 1)
        fused: bool = True,
    ):
        if lr > 1e-3:
            print(f"Warning! Start lr {lr} is very high; forcing to 1e-6.")
            lr = 1e-6
        # Agreement compares the current update sign against the stored history,
        # so at least one snapshot must be kept (1 = compare to previous only).
        polarity_history_count = max(1, int(polarity_history_count))
        defaults = dict(
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            lr_bump_rate=lr_bump_rate,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            polarity_history_count=polarity_history_count,
            # Noise floor of the consistency measure for this window (history+1),
            # subtracted off so the lr signal is window-size independent.
            agreement_floor=self._noise_floor(polarity_history_count + 1),
        )
        super().__init__(params, defaults)

        self.fused = fused
        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.fused:
                    # Fused: update each param the moment its grad is ready.
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                    self._hook_handles.append(handle)
                elif p.dtype != torch.float32:
                    # Non-fused: the actual update happens in .step(); here we
                    # only stochastically accumulate low-precision grads across
                    # micro-batches so repeated round-to-nearest doesn't drop
                    # small grads (fp32 grads accumulate losslessly on their own).
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_accum_hook()
                    )
                    self._hook_handles.append(handle)

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(f"Total training paramiters: {total:,}")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _noise_floor(window: int) -> float:
        # Expected window-consistency max(p, 1-p) of a pure-noise element over a
        # window of `window` independent fair coin flips. This is > 0.5 and grows
        # with smaller windows (0.75 for window<=3, ~0.688 for 4, ...), so it must
        # be subtracted off for the lr signal to behave the same at any window.
        total = sum(math.comb(window, k) * max(k, window - k)
                    for k in range(window + 1))
        return total / (window * (2 ** window))

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    @staticmethod
    def _sr_truncate(v_fp32: torch.Tensor, drop_bits: int) -> torch.Tensor:
        # Fast in-place stochastic rounding for a low-precision float that is a
        # mantissa truncation of fp32: add uniform noise into the dropped low
        # mantissa bits of the fp32 bit pattern, then zero them, so the
        # subsequent narrowing cast is exact and rounds up with probability
        # equal to the truncated fractional part. bf16 drops 16 bits (it is the
        # high half of fp32); fp16 drops 13 bits (23 - 10 mantissa) and is exact
        # within its normal exponent range -- values past fp16's overflow /
        # subnormal limits are rounded at fp32 granularity, which trained
        # weights effectively never reach.
        as_int = v_fp32.view(torch.int32)
        as_int.add_(torch.randint_like(as_int, 1 << drop_bits))
        as_int.bitwise_and_(-(1 << drop_bits))
        return v_fp32

    @staticmethod
    def _stochastic_round(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Generic fp32 -> low-precision stochastic rounding for dtypes that are
        # not a mantissa truncation of fp32 (the bf16/fp16 fast path in
        # _sr_truncate does not apply). Adds uniform noise of +/- half a target
        # ULP and rounds to nearest, so P(round up) equals the fractional
        # distance to the next representable value -> unbiased in expectation.
        # The ULP at |v| is 2**floor(log2|v|) * eps(dtype).
        finfo = torch.finfo(dtype)
        absv = v.abs().clamp_(min=finfo.tiny)
        ulp = torch.exp2(torch.floor(torch.log2(absv))).mul_(finfo.eps)
        noise = torch.rand_like(v).sub_(0.5).mul_(ulp)
        return v.add_(noise).to(dtype)

    @classmethod
    def _stochastic_copy_(cls, dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
        # Stochastically round the fp32 ``src`` into the low-precision ``dst`` in
        # place. Uses the fast mantissa-truncation path for bf16/fp16 and the
        # generic method otherwise. ``src_fp32`` may be mutated (caller owns it).
        if dst.dtype == torch.bfloat16:
            dst.copy_(cls._sr_truncate(src_fp32, 16))
        elif dst.dtype == torch.float16:
            dst.copy_(cls._sr_truncate(src_fp32, 13))
        else:
            dst.copy_(cls._stochastic_round(src_fp32, dst.dtype))

    def _make_accum_hook(self):
        # Non-fused grad accumulation for low-precision params: accumulate the
        # running sum in fp32 then stochastically round it back into the
        # low-precision ``_accum_grad`` buffer, so small per-micro-batch grads
        # are not lost to repeated round-to-nearest. .step() consumes the buffer.
        def _hook(p: torch.Tensor):
            if p.grad is None:
                return
            if hasattr(p, "_accum_grad"):
                acc = p._accum_grad.to(torch.float32).add_(p.grad.to(torch.float32))
                self._stochastic_copy_(p._accum_grad, acc)
            else:
                p._accum_grad = p.grad.clone()
            p.grad = None

        return _hook

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        # Per-row lr: one entry per output channel for >=2D params, per element
        # for 1D params, a scalar for 0D params.
        lr_shape = (p.shape[0],) if p.dim() >= 2 else p.shape
        state["lr"] = torch.full(
            lr_shape, float(group["lr"]), dtype=torch.float32, device=p.device
        )
        # Rolling history of the last polarity_history_count update-sign
        # snapshots (bool); the current step is compared against all of them.
        state["pol_hist"] = []
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1], dtype=p.dtype, device=p.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)

        return _hook

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
            raise RuntimeError("Automagic3 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        # This step is fused into backward, so the trainer's grad clipping and
        # nan/inf-skip run too late to protect us -- the weights are already
        # updated here. A single non-finite gradient would poison the
        # second-moment EMA (NaN*beta2 + ... stays NaN forever) and corrupt the
        # weights, which surfaces as the model "randomly" blowing up. Neutralise
        # non-finite grads in place (we own this fp32 grad) so those elements
        # contribute nothing this step instead of destroying state. Large but
        # finite grads are left alone -- the second-moment normalisation already
        # bounds their effect.
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        beta2 = group["beta2"]
        eps = group["eps"]
        # eps is folded into the reduced row/col (or rsqrt) instead of being
        # added to the full-size sq tensor: mean(g^2 + eps) == mean(g^2) + eps,
        # which saves a full-size kernel pass.
        sq = grad * grad

        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            if row_state.dtype == torch.float32:
                row, col = row_state, col_state
                row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
            else:
                row = row_state.to(torch.float32)
                col = col_state.to(torch.float32)
                row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
                row_state.copy_(row.to(row_state.dtype))
                col_state.copy_(col.to(col_state.dtype))
            update = self._approx_sq_grad(row, col).mul_(grad)
        else:
            v_state = state["exp_avg_sq"]
            if v_state.dtype == torch.float32:
                v = v_state
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            else:
                v = v_state.to(torch.float32)
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
                v_state.copy_(v.to(v_state.dtype))
            update = v.add(eps).rsqrt().mul_(grad)

        # Update-RMS clip (trust region): scale so the update RMS never exceeds
        # clip_threshold. No bias-correction warmup -- LoRA runs are short and a
        # slow ramp wastes steps; for a soft start the user can set a low start
        # lr and let the lr bump up on its own.
        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        # The RMS clip only bounds the aggregate, so a single outlier element can
        # still survive at ~sqrt(numel)*clip_threshold and hit one weight hard,
        # distorting the model. Cap each element to clip_threshold (a true
        # max-norm trust region) so no single weight can take an outsized step.
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])

        # Window-consistency agreement. We keep the last polarity_history_count
        # update-sign snapshots; over the window of those plus the current sign,
        # the per-element agreement is the fraction sharing the *dominant* sign,
        # max(p, 1 - p) == 0.5 + |p - 0.5| where p is the fraction positive. This
        # is direction-agnostic (a consistently-negative element scores as high
        # as a consistently-positive one) and the current step is just one vote,
        # so a single flip against a long consistent history barely moves it
        # (e.g. 20-of-21 the same -> ~0.95). Values lie in [0.5, 1]: 0.5 is a
        # 50/50 (chaotic) element, 1.0 is perfectly consistent. Reduced to one
        # value per output channel for >=2D, per element for 1D.
        cur_polarity = update > 0
        pol_hist = state["pol_hist"]

        lr_t = state["lr"]
        if p.dim() >= 2:
            dims = tuple(range(1, p.dim()))
            lr_b = lr_t.view(lr_t.shape[0], *([1] * (p.dim() - 1)))
        else:
            dims = None
            lr_b = lr_t

        if pol_hist:
            # per-element fraction positive over the window (current + history)
            win = len(pol_hist) + 1
            frac = cur_polarity.to(torch.float32)
            for h in pol_hist:
                frac.add_(h)
            frac.div_(win)
            # dominant-sign fraction in [0.5, 1], then reduce to per-row
            consist = frac.sub_(0.5).abs_().add_(0.5)
            agreement = consist.mean(dim=dims) if dims is not None else consist

            # Map consistency to a direction in [-1, 1], measured relative to the
            # window's noise floor b (the consistency a pure-noise row shows by
            # chance, computed from the window size): agreement == b -> 0 (hold),
            # 1.0 (perfect) -> +1 (lr up), below b -> down. There is no manual
            # target -- holding at b is fully automatic and self-balancing per
            # row: the lr grows while a row is more consistent than chance and
            # settles at whatever lr makes its consistency meet b, so noisy and
            # clean layers each find their own operating point. Measuring
            # relative to b also makes the signal window-size independent.
            b = group["agreement_floor"]
            direction = agreement.sub_(b).div_(1.0 - b).clamp_(-1.0, 1.0)
            # Multiplicative (geometric) bump: lr *= exp(direction * lr_bump_rate).
            # lr_bump_rate is a fractional rate (~lr_bump_rate per step for small
            # values), giving a uniform relative move at every scale across
            # [min_lr, max_lr] instead of a fixed absolute amount.
            lr_t.mul_(torch.exp(direction.mul_(group["lr_bump_rate"]))).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )

        # Record this step's polarity and trim the window to the configured size.
        pol_hist.append(cur_polarity)
        if len(pol_hist) > group["polarity_history_count"]:
            del pol_hist[0]
        state["step"] += 1

        wd = group["weight_decay"]

        if p.dtype == torch.float32:
            # Decoupled weight decay folded in (update += wd*p), then a single
            # fused p -= lr_b * update.
            if wd != 0.0:
                update.add_(p, alpha=wd)
            p.addcmul_(update, lr_b, value=-1.0)
        else:
            # Low precision: apply the update in fp32 then stochastically round
            # back, so tiny updates aren't lost to round-to-nearest. Single
            # bf16/fp16 -> fp32 conversion shared by weight decay and rounding.
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.add_(new_p_fp32, alpha=wd)
            new_p_fp32.addcmul_(update, lr_b, value=-1.0)
            self._stochastic_copy_(p, new_p_fp32)

        p.grad = None

    # ----------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Fused mode already updated every param in the backward hook; nothing
        # left to do. Non-fused mode does the real work here.
        if not self.fused:
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    # Low-precision grads were stochastically accumulated into
                    # _accum_grad; hand it back as the grad to update from.
                    accum = getattr(p, "_accum_grad", None)
                    if accum is not None:
                        p.grad = accum
                        del p._accum_grad
                    if p.grad is None:
                        continue
                    self._update_param(p, group)
        return loss

    def get_learning_rates(self) -> List[float]:
        out = []
        for group in self.param_groups:
            lrs = [
                float(self.state[p]["lr"].mean())
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(sum(lrs) / len(lrs) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        # Parent casts every fp state tensor to param.dtype; force lr back to fp32
        # so subsequent lr bumps aren't rounded away on bf16 weights.
        super().load_state_dict(state_dict)
        # Constructor args always win over whatever was saved in the checkpoint.
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None and isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)
                # Polarity history is transient; rebuild it after load rather
                # than trying to persist/cast a list of bool tensors.
                if st is not None and "pol_hist" in st:
                    st["pol_hist"] = []
