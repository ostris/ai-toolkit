"""
NOTE: This is experimental and under active development; expect breaking changes and bugs. Feedback welcome.
"""

from typing import List
import torch


class Automagic3(torch.optim.Optimizer):
    """
    Automagic v3.

    A single learning rate is kept per parameter tensor (one lr per weight
    matrix / layer). Each step the lr is nudged by whether the per-element update
    direction *flipped* vs the previous step (RProp-style edge-of-stability
    control), aggregated over the whole tensor.

    A sign flip means the step jumped past the local minimum (overshoot) -- the
    one event whose frequency genuinely rises with the lr, so it provides a true
    restoring force. Each element votes: agree with last step -> +``lr_bump_rate``;
    flip -> -``lr_bump_rate`` (symmetric). The votes are reduced to ONE value for
    the whole tensor and EMA-smoothed over ~``lr_smoothing_steps`` steps, then
    applied multiplicatively: ``lr *= exp(nudge)``. Reducing over the whole tensor
    (rather than per row/channel) is deliberate: per-channel lrs let coupled
    channels fight -- one drives its lr up while a neighbour drives its down to
    compensate -- and split to opposite extremes, which visibly wrecks the model;
    one lr per tensor makes opposing channels cancel into a single vote.

    The flip equilibrium is flip fraction == 0.5, the only rate that is both the
    pure-noise point and the edge of stability: a tensor descending cleanly flips
    less than half the time -> its lr grows; once it overshoots it flips more than
    half -> its lr shrinks; pure noise flips ~half -> the lr holds. Symmetry is
    load-bearing (any asymmetry drags a noise-dominated tensor to zero). Elements
    with an exactly-zero update (dead/masked grads, low-precision underflow) carry
    no direction and abstain from the vote.

    On top of the per-tensor flip control, every tensor's lr is gently pulled
    toward the GLOBAL average lr each step (``lr_pull``, geometric / log-space:
    ``lr *= (avg_lr / lr) ** lr_pull``). This mean-reversion is the restoring
    force that replaces hard min/max clamps: layers may settle at their own level,
    but cannot drift apart to opposite extremes (one frozen near zero, one running
    away) -- the failure mode that destroys full finetunes. The target is the
    emergent average across layers, not a fixed number, so it stays fully
    automatic; there are no min/max lr bounds. ``lr_bump_rate`` sets how fast the
    lr moves, not where it lands.

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

    Parameters
    ----------
    lr : float
        Starting learning rate for every layer. The controller adapts away from
        this, so it is a launch point, not a tuned target -- a low value just
        lets the lr ramp up on its own (there is no warmup). Values above 1e-3
        are rejected and forced back to 1e-6. There are no min/max lr clamps;
        the mean-reversion pull keeps the spread bounded instead.
    lr_bump_rate : float
        Fractional, log-space size of each lr nudge (~10% at 0.1). Sets how fast
        the lr moves, NOT where it settles (the flip dynamics fix that); a full
        up-nudge and a full down-nudge cancel exactly.
    lr_pull : float
        Per-step strength of the mean-reversion pulling each layer's lr toward
        the global average (log space; 0 disables). This is the restoring force
        that replaces min/max clamps and stops layers drifting to opposite
        extremes. Small (default 0.05) lets layers keep their own level while
        bounding the spread; larger forces all layers toward a common lr.
    beta2 : float
        EMA decay for the second moment, as in Adam/Adafactor.
    eps : float
        Floor added to the second moment before the rsqrt, to avoid div-by-zero.
    clip_threshold : float
        Trust region on the update: its RMS is scaled to <= this, then every
        element is clamped to +/- this, so no single weight takes an outsized
        step.
    weight_decay : float
        Decoupled (AdamW-style) weight decay; 0 disables it.
    lr_smoothing_steps : int
        How many steps of the flip signal to EMA-average before nudging the lr
        (>=1, default 3). Higher = smoother/slower lr, lower = twitchier/faster;
        it does not change where the lr lands. Held as an EMA, so it costs O(1)
        state per layer regardless of the value.
    fused : bool
        If True (default), each param is updated inside the backward pass the
        moment its grad is ready -- low peak VRAM, but it bypasses the trainer's
        grad clipping / nan-skip and cannot be combined with multi-backward
        gradient accumulation. If False, a normal ``.step()``-time update, with
        low-precision grads accumulated using stochastic rounding.

    Improvements over v2
    --------------------
    1. Per-layer lr with mean-reversion (v2 had one static lr per tensor and no
       coupling between layers). v3 still keeps one lr per tensor, but it is
       adaptive (driven by the flip controller) and every layer's lr is pulled
       toward the global average (``lr_pull``). Plain English: each layer finds
       its own learning rate automatically, but no layer can run away or freeze
       relative to the others -- which is what used to split a full finetune into
       over-cooked and dead layers and destroy it. (An earlier v3 used a separate
       lr per output channel; coupled channels fought and split to opposite
       extremes, so it was reduced back to one lr per tensor.)

    2. Overshoot-based (RProp-style) lr control with a real equilibrium. v2
       bumped the lr from raw direction agreement, which has no upper fixed point
       -- a parameter that is simply still descending keeps agreeing at any lr,
       so the lr ratchets up and eventually runs away on long runs. v3 drives the
       lr from sign *flips* (overshoot) instead, nudging up on agree and down on
       flip symmetrically; the equilibrium is a flip fraction of 0.5, which is
       both the noise point and the edge of stability. Plain English: the lr
       speeds up while a layer is making clean progress, backs off the moment it
       starts overshooting, and simply holds when the gradient is pure noise --
       so it neither climbs without bound on long runs nor collapses to nothing
       on a fresh, noisy LoRA.

    3. Multiplicative (geometric) lr bump (was additive). v2 added/subtracted a
       fixed absolute amount, so the same bump was a huge relative jump when the
       lr was tiny and a negligible one when it was large. v3 multiplies by
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
       row/col vectors instead of the full gradient-square tensor; the lr scale
       and parameter update are fused into one ``addcmul_``; and the per-element
       agree/flip vote is a single int8 sign-product (``cur_sign * prev_sign``)
       instead of several boolean masks plus float casts. Plain English: each
       step issues fewer GPU passes over the weights, so it runs faster (notably
       in bf16/fp16) without changing the result.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        lr_bump_rate: float = 0.1,  # fractional/log step per bump (~10%); see step logic
        lr_pull: float = 0.025,  # per-step pull of each layer's lr toward the global avg (log space)
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        lr_smoothing_steps: int = 3,  # lr-nudge EMA smoothing horizon, in steps (min 1)
        fused: bool = True,
    ):
        if lr > 1e-3:
            print(f"Warning! Start lr {lr} is very high; forcing to 1e-6.")
            lr = 1e-6
        # The lr nudge is EMA-smoothed over ~this many steps; at least 1.
        lr_smoothing_steps = max(1, int(lr_smoothing_steps))
        defaults = dict(
            lr=lr,
            lr_bump_rate=lr_bump_rate,
            lr_pull=max(0.0, float(lr_pull)),
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            lr_smoothing_steps=lr_smoothing_steps,
            # EMA decay for the per-layer lr nudge, derived from the smoothing
            # horizon (n steps -> beta = n/(n+1)).
            dir_beta=lr_smoothing_steps / (lr_smoothing_steps + 1.0),
        )
        super().__init__(params, defaults)

        self.fused = fused
        # Global geometric-mean lr across all layers; the mean-reversion pull
        # targets this. Seeded at the start lr (all layers equal) and refreshed
        # each .step() from the current per-layer lrs.
        self._avg_lr = float(lr)
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
        # Root-mean-square of a tensor; used to size the trust-region clip.
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        # Adafactor's factored second moment (inherited from v2). Rather than
        # store a full RxC tensor of running grad^2, only its per-row and
        # per-col means are kept; this rebuilds the rank-1 approximation of
        # 1/sqrt(v) -- the per-element update scale -- as the outer product
        # rsqrt(row / mean(row)) (x) rsqrt(col). That is the standard HF
        # Adafactor reconstruction and is what keeps optimizer state small.
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
        # ONE lr per parameter tensor (scalar), not per row/channel. Per-channel
        # lrs let coupled channels fight -- one drives up while another drives
        # down to compensate -- so they split to the rails and wreck the model.
        # A single lr per tensor averages the vote over all elements, so opposing
        # channels cancel instead of diverging.
        state["lr"] = torch.tensor(
            float(group["lr"]), dtype=torch.float32, device=p.device
        )
        # Previous update-sign snapshot (int8 {-1, 0, +1}, full param shape); the
        # current sign is compared against it to detect per-element flips. Set on
        # the first step.
        state["prev_sign"] = None
        # EMA of the (scalar) log lr-nudge, smoothing the flip signal over time.
        state["dir_ema"] = torch.zeros((), dtype=torch.float32, device=p.device)
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

        # Second moment: a beta2-EMA of grad^2, then update = grad / sqrt(v),
        # exactly as Adam/Adafactor (this magnitude-normalises the step; only the
        # *sign* of the result drives the lr controller further down). For >=2D
        # params v is Adafactor-factored into row/col means (small state, see
        # _approx_sq_grad); 1D params (biases, norms) keep the full per-element
        # second moment. State lives in p.dtype; when that is low precision the
        # math is done in an fp32 copy and written back.
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

        # RProp-style edge-of-stability lr control. The signal is whether each
        # element's update direction *flipped* vs the previous step, not how
        # steady it has been: a flip means we stepped past the local minimum
        # (overshoot), the one event whose frequency actually rises with the lr,
        # so it gives a true restoring force. Steadiness does not -- a parameter
        # descending monotonically agrees with itself at any non-overshooting lr,
        # which is why a consistency-vs-noise-floor signal has no upper
        # equilibrium and runs away on long tunes.
        # Trinary sign {-1, 0, +1}: zero updates (dead/masked grads, flat
        # activation regions, low-precision underflow) are kept distinct from
        # negatives rather than bucketed with them by a bare ``> 0``.
        cur_sign = update.sign().to(torch.int8)
        prev_sign = state["prev_sign"]
        lr_t = state["lr"]  # scalar (one lr for the whole tensor)

        if prev_sign is not None:
            # Per-element vote via the sign product. With signs in {-1, 0, +1},
            # cur_sign * prev_sign is +1 when the direction held (agree), -1 when
            # it flipped (overshoot), and 0 whenever either step's update was zero
            # -- so a zero update automatically ABSTAINS (contributes nothing and
            # isn't counted), no separate masking needed.
            #
            # Reduced over the WHOLE tensor (not per row) this is
            # bump*(1 - 2*flip_fraction): the lr grows while the layer mostly
            # holds its direction, shrinks once it mostly flips, and holds at the
            # flip_fraction == 0.5 point. Aggregating over all elements means
            # channels pushing opposite ways cancel into one vote instead of
            # splitting to opposite lr extremes.
            bump = group["lr_bump_rate"]
            prod = cur_sign * prev_sign  # int8 {-1, 0, +1} per element
            # Reduce directly on the int8 product (sum -> int64, no full-size
            # float cast; count_nonzero -> valid votes) -- two reductions instead
            # of casting the whole tensor to float twice.
            num = prod.sum()
            den = prod.count_nonzero().clamp_(min=1)
            log_dir = num.float().div_(den.float()).mul_(bump)
            # EMA-smooth the nudge so a single noisy step doesn't swing the lr,
            # then apply it multiplicatively (a fixed fractional move).
            ema = state["dir_ema"]
            beta = group["dir_beta"]
            ema.mul_(beta).add_(log_dir, alpha=1.0 - beta)
            lr_t.mul_(torch.exp(ema))
            # Mean-reversion: pull this layer's lr toward the global average lr
            # (geometric, in log space) -- lr *= (avg/lr)**lr_pull. This is the
            # restoring force that replaces the hard min/max rails: layers can
            # still settle at their own level, but can't drift apart to opposite
            # extremes (one frozen, one runaway) and wreck the model. The pull is
            # toward an emergent average, not a fixed target, so it stays
            # automatic. self._avg_lr is refreshed once per .step().
            pull = group["lr_pull"]
            if pull > 0.0 and self._avg_lr > 0.0:
                lr_t.mul_(lr_t.reciprocal().mul_(self._avg_lr).pow_(pull))

        state["prev_sign"] = cur_sign
        state["step"] += 1

        wd = group["weight_decay"]

        if p.dtype == torch.float32:
            # Decoupled weight decay folded in (update += wd*p), then a single
            # fused p -= lr * update (lr is a scalar, broadcasts).
            if wd != 0.0:
                update.add_(p, alpha=wd)
            p.addcmul_(update, lr_t, value=-1.0)
        else:
            # Low precision: apply the update in fp32 then stochastically round
            # back, so tiny updates aren't lost to round-to-nearest. Single
            # bf16/fp16 -> fp32 conversion shared by weight decay and rounding.
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.add_(new_p_fp32, alpha=wd)
            new_p_fp32.addcmul_(update, lr_t, value=-1.0)
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
        self._refresh_avg_lr()
        return loss

    def _all_lrs(self) -> list:
        # The per-layer lr scalars (0-d tensors), gathered without any device
        # sync. Callers stack these and reduce in one op so there is a single
        # GPU->CPU sync instead of one per layer.
        return [
            st["lr"]
            for group in self.param_groups
            for p in group["params"]
            if (st := self.state.get(p)) is not None and "lr" in st
        ]

    def _refresh_avg_lr(self) -> None:
        # Global geometric-mean lr across all layers, the mean-reversion target.
        # Geometric (mean of log) because the lr is controlled/pulled in log
        # space. One stacked reduction + one sync, refreshed once per step.
        lrs = self._all_lrs()
        if lrs:
            self._avg_lr = float(torch.stack(lrs).log_().mean().exp_())

    def get_learning_rates(self) -> List[float]:
        # Reporting helper: one representative lr per param group -- the mean of
        # the per-layer (per-tensor) lrs. Stacked reduction -> one sync per group.
        out = []
        for group in self.param_groups:
            lrs = [
                self.state[p]["lr"]
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(float(torch.stack(lrs).mean()) if lrs else float(group["lr"]))
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
                # prev_sign / dir_ema are transient; rebuild them after load
                # rather than persisting a sign tensor and an fp32 EMA.
                if st is not None and "prev_sign" in st:
                    st["prev_sign"] = None
                if st is not None and isinstance(st.get("dir_ema"), torch.Tensor):
                    st["dir_ema"] = torch.zeros_like(st["dir_ema"], dtype=torch.float32)
        # Rebuild the global average lr from the restored per-layer lrs.
        self._refresh_avg_lr()
