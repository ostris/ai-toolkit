"""
NOTE: This is experimental and under active development; expect breaking changes and bugs. Feedback welcome.
"""

from typing import List
import torch


class Automagic3(torch.optim.Optimizer):
    """
    Automagic v3.

    A single learning rate is kept per param group (typically: one lr for
    the whole run). The control principle: the lr RISES while elements hold
    a decisively consistent update direction at the current step size, FALLS
    while their signs decisively alternate (the overshoot signature: weights
    hopping across a minimum flip sign step to step -- shrinking the step is
    what makes a trajectory reappear at a finer scale), and HOLDS on
    everything in between, which is treated as noise.

    Each element keeps a window of its last H (= ``polarity_history``,
    default 4) update sign bits ("is the update positive", 1-bit packed) --
    H/8 bytes per element (half a byte at the default), the only
    per-element optimizer state. A short window suffices because verdicts
    are pooled across the whole group: millions of voters make weak
    common-mode evidence visible long before any single element is
    decisive, and the window length is also the controller's reaction lag
    and warmup.

    Vote rule (per element)
    -----------------------
    Only the two perfectly decisive window states vote; everything else is
    noise:

      up    all H signs agree            +1 * |update|  ("step too small")
      down  all H-1 transitions flip     -1 * |update|  ("step too large":
            (perfect alternation)        the overshoot signature)
      else  any imperfect window          0  (noise)

    The two events are exact mirrors with IDENTICAL pure-noise probability
    (2 of the 2^H possible windows each; ~0.8% per element at H=8), so equal
    weights balance exactly -- no correction factors, no tiers. Per element
    the events are rare, but the verdict is pooled over the whole group
    (millions of elements -> tens of thousands of voters per step even
    under pure noise, mean zero), so the pooled signal is smooth and a real
    trend or real overshoot shifts it decisively. A majority being overshot
    always outvotes a persistent minority, which is what anchors the lr's
    absolute level without external rails. Weighting by |update| lets the
    elements actually moving the weights dominate; an exact-zero update
    records as the negative bit, but such dead/masked elements carry zero
    weight anyway. A tensor abstains entirely until its window has filled
    (the first H steps, and again after a history reset on resume).

    ONE learning rate per param group -- not per tensor. Every element of
    every tensor in the group votes into a single pool, and the group lr is
    nudged once per step by the pooled result, applied multiplicatively
    with NO gain knob: ``lr *= exp(vote)`` -- the lr moves at exactly the
    rate the model votes for it. A fully unanimous pool (practically
    unreachable) would move e ~= 2.7x per step; the silent majority dilutes
    the pooled vote, so realistic moves are a few percent per step, and the
    worst-case overshoot past the edge is bounded by the H-step window lag
    before alternation votes answer. There is no
    noise-floor estimation, no smoothing, no significance test: the polarity
    windows are the only indicator. Pooling at group level (rather than per
    tensor, and originally rather than per channel) is the load-bearing
    choice: COUPLED tensors fight per-tensor lrs exactly like coupled
    channels fight per-channel ones. A Q/K pair is the canonical case --
    Q's weights scaling up while K's scale down preserves the attention
    logits, so the gradients reward whichever asymmetry randomly seeded
    first: Q votes "too slow" and climbs while K votes "too fast" and sinks,
    self-reinforcing without bound. One shared lr makes those opposing votes
    cancel in the pool instead of diverging, so only common-mode evidence
    ("the whole group's step is too small / too large") moves the lr.

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
        Starting learning rate for every group. The controller adapts away
        from this in whichever direction the pooled vote points, so it is a
        launch point, not a tuned target. There are no min/max lr clamps
        (only a numerical overflow guard far outside the usable range).
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
    polarity_history : int
        Sign-history window length H (2 to 64, default 4); H/8 bytes of
        state per element. Longer windows make the two vote events rarer
        and more decisive (probability 2^(1-H) each under noise -- a real
        trend's excess grows ~(1+rho)^H), so detection sharpens, at the
        cost of memory, an H-step reaction lag/warmup, and fewer voters
        per step. Changing it on resume resets the histories cleanly (one
        re-warmup of H steps).
    fused : bool
        If True (default), each param is updated inside the backward pass the
        moment its grad is ready -- low peak VRAM, but it bypasses the trainer's
        grad clipping / nan-skip and cannot be combined with multi-backward
        gradient accumulation. If False, a normal ``.step()``-time update, with
        low-precision grads accumulated using stochastic rounding.

    Improvements over v2
    --------------------
    1. One adaptive lr per param group (v2 had one static lr per tensor).
       Plain English: the group finds its learning rate automatically, and no
       layer can run away or freeze relative to the others -- which is what
       used to split a full finetune into over-cooked and dead layers and
       destroy it. (Earlier v3s used a separate lr per output channel, then
       per tensor; each level let coupled units -- channels, then Q/K-style
       tensor pairs -- fight and split to opposite extremes, so the lr was
       pooled one level up each time until the fighting was structurally
       impossible.)

    2. Direction-consistency lr control with a real equilibrium. v2 bumped
       the lr from raw single-step agreement, which has no upper fixed point
       -- a parameter that is simply still descending keeps agreeing at any
       lr, so the lr ratchets up and eventually runs away on long runs. v3
       votes from each element's recent sign window (see the vote rule
       above). Plain English: the lr speeds up while the model holds a
       trajectory, backs off hard when it overshoots, and holds steady on
       pure noise.

    3. Multiplicative (geometric) lr bump (was additive). v2 added/subtracted a
       fixed absolute amount, so the same bump was a huge relative jump when the
       lr was tiny and a negligible one when it was large. v3 multiplies by
       ``exp(vote)`` -- a fixed *percentage* step. Plain
       English: the lr moves at the same relative pace whether it is tiny or
       large, traverses its whole range in a predictable number of steps, and a
       full up bump is exactly cancelled by a full down bump (no drift); the
       gain knob was removed entirely once the vote became a pooled
       fraction with natural log-units.

    4. Stochastic rounding for fp16, not just bf16. v2 only rounded bf16
       write-backs and let fp16 fall back to round-to-nearest, silently
       discarding updates smaller than an fp16 ULP. v3 stochastically rounds
       both (fast bit-trick for bf16/fp16, generic fallback for other low
       precisions). Plain English: fp16 training no longer throws away small
       weight updates, so it actually keeps learning instead of stalling.

    5. Faster hot path, identical math. eps is folded into the small reduced
       row/col vectors instead of the full gradient-square tensor; the lr scale
       and parameter update are fused into one ``addcmul_``; the per-element
       direction and flip sums are recomputed from the 1-bit history planes
       in a single batched unpack and two integer reductions, and scored
       with three boolean compares and weighted sums. Plain English: each
       step issues few GPU passes over the weights, so it runs fast
       (notably in bf16/fp16).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        polarity_history: int = 8,  # sign-history window length (2-64)
        fused: bool = True,
    ):
        if lr > 1e-3:
            # No clamping: a too-high start just oscillates immediately and
            # the controller drives it down.
            print(
                f"Note: start lr {lr} is high; the controller will correct it "
                f"(the pooled vote will walk it down)."
            )
        defaults = dict(
            lr=lr,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            polarity_history=max(2, min(64, int(polarity_history))),
        )
        super().__init__(params, defaults)

        self.fused = fused
        self._rebuild_group_index()
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

    # Per-device cached constants for pack/unpack (avoid re-allocating a tiny
    # tensor on every call).
    _PACK_CONSTS: dict = {}

    @classmethod
    def _pack_consts(cls, device):
        consts = cls._PACK_CONSTS.get(device)
        if consts is None:
            consts = (
                torch.tensor(
                    [1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8
                ),
                torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.uint8
                ),
            )
            cls._PACK_CONSTS[device] = consts
        return consts

    @classmethod
    def _pack_bits(cls, bits: torch.Tensor) -> torch.Tensor:
        # Pack sign bits (bool / {0, 1}) 8 per byte (uint8), as a base-2 dot
        # product per group of 8 (two kernels rather than per-slice shift/or
        # chains).
        weights, _ = cls._pack_consts(bits.device)
        flat = bits.reshape(-1).to(torch.uint8)
        pad = (-flat.numel()) % 8
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        return (flat.view(-1, 8) * weights).sum(-1, dtype=torch.uint8)

    @classmethod
    def _unpack_bits(cls, packed: torch.Tensor, numel: int) -> torch.Tensor:
        # Inverse of _pack_bits: uint8 -> flat uint8 {0, 1} of length numel.
        _, shifts = cls._pack_consts(packed.device)
        vals = (packed.unsqueeze(-1) >> shifts).bitwise_and_(1)
        return vals.view(-1)[:numel]

    def _rebuild_group_index(self) -> None:
        # param -> index of its param group, plus per-group vote accumulators
        # (weighted vote mass and total weight, gathered across every tensor
        # in the group during the step and applied once in .step()). The map
        # exists because the fused hooks cannot rely on group-dict identity:
        # the parent's load_state_dict replaces the group dicts.
        self._param_group_index = {
            p: gi for gi, group in enumerate(self.param_groups) for p in group["params"]
        }
        self._group_num: List = [None] * len(self.param_groups)
        self._group_den: List = [None] * len(self.param_groups)

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
        # The group lr, mirrored per param (every param in a group receives
        # identical multiplicative nudges, so these stay equal; storing per
        # param rides the normal state_dict machinery and tolerates
        # multi-device groups).
        state["lr"] = torch.tensor(
            float(group["lr"]), dtype=torch.float32, device=p.device
        )
        # Ring buffer of per-element update sign bits, one 1-bit-packed
        # plane per step (H/8 bytes per element). Sums are recomputed from
        # the planes each step rather than stored -- the history is the
        # ONLY per-element state.
        H = group["polarity_history"]
        width = (p.numel() + 7) // 8
        state["sign_history"] = torch.zeros(
            (H, width), dtype=torch.uint8, device=p.device
        )
        # Index of the OLDEST plane (the one overwritten next step).
        state["hist_idx"] = 0
        # Number of real sign planes stored so far; the controller is gated
        # until the window is full (there is no per-element abstain state).
        state["hist_fill"] = 0
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

        # In fused mode this runs inside backward, so the trainer's grad
        # clipping and nan/inf-skip come too late to protect us. A single
        # non-finite gradient would poison the second-moment EMA (NaN stays
        # NaN forever) and corrupt the weights, so neutralise non-finite
        # grads in place (we own this fp32 copy); those elements contribute
        # nothing this step. Large but finite grads are left alone -- the
        # second-moment normalisation already bounds their effect.
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

        # Direction-consistency lr control (the vote rule -- see the class
        # docstring). The second-moment scale, RMS clip and clamp are all
        # positive, so the sign bit is exactly sign(grad); an exact-zero
        # update records as the negative bit, harmless because its |update|
        # vote weight is zero.
        cur_bits = update.gt(0.0)
        hist = state["sign_history"]  # (H, numel/8) 1-bit packed uint8
        idx = state["hist_idx"]  # oldest plane (overwritten below)
        H = hist.shape[0]
        lr_t = state["lr"]  # this param's mirror of the shared group lr

        # Slide the window first so the vote sees the freshest H signs.
        hist[idx].copy_(self._pack_bits(cur_bits))
        state["hist_idx"] = (idx + 1) % H
        # The planes hold garbage until H real signs have been stored (fresh
        # start or a history reset on resume): gate the controller, not the
        # parameter update, until the window is full.
        fill = min(H, state["hist_fill"] + 1)
        state["hist_fill"] = fill

        if fill == H:
            # Extremes-only vote (see the class docstring): all H signs
            # agreeing votes up, perfect alternation (all H-1 transitions
            # flipping) votes down -- the two events have identical
            # pure-noise probability (2 of the 2^H windows each), so equal
            # +/-1 weights balance exactly. The planes are rolled into
            # chronological order so adjacent rows are adjacent steps; XOR
            # of neighbour rows marks per-bit flips. The weighted vote mass
            # and total weight are ACCUMULATED into this tensor's group; the
            # single group lr is nudged once per step in .step().
            _, shifts = self._pack_consts(hist.device)
            chron = torch.roll(hist, -state["hist_idx"], dims=0)
            bits = (
                (chron.unsqueeze(-1) >> shifts)
                .bitwise_and_(1)
                .view(H, -1)[:, : update.numel()]
            )
            s1 = bits.sum(0, dtype=torch.int16)
            flips = (bits[1:] ^ bits[:-1]).sum(0, dtype=torch.int16)
            up = s1.eq(H).logical_or_(s1.eq(0))
            down = flips.eq(H - 1)
            w = update.abs().view(-1)
            num = (w * up).sum().sub_((w * down).sum())
            den = w.sum()
            gi = self._param_group_index.get(p)
            if gi is not None:
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
        self._apply_group_votes()
        return loss

    def _apply_group_votes(self) -> None:
        # ONE lr nudge per group per step, from the pooled vote of every
        # element of every tensor in the group (see the class docstring on
        # why pooling at group level is load-bearing). Each param's lr tensor
        # receives the same multiplicative factor, so they stay identical --
        # effectively a single group lr, stored per param only so it rides
        # the normal state_dict machinery. All tensor ops: no GPU sync.
        for gi, group in enumerate(self.param_groups):
            num = self._group_num[gi]
            if num is None:
                continue
            den = self._group_den[gi]
            signal = num.div_(den.clamp_(min=1e-30)).clamp_(-1.0, 1.0)
            factor = torch.exp(signal)
            for p in group["params"]:
                st = self.state.get(p)
                if st is None or "lr" not in st:
                    continue
                lr_t = st["lr"]
                f = factor if factor.device == lr_t.device else factor.to(lr_t.device)
                # Numerical overflow guard only -- NOT a control rail
                # (decades outside the usable range).
                lr_t.mul_(f).clamp_(min=1e-30, max=1e3)
            self._group_num[gi] = None
            self._group_den[gi] = None

    def get_learning_rates(self) -> List[float]:
        # Reporting helper: the (shared) lr of each param group.
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
        # Hyperparameters are NOT loaded from the checkpoint: constructor args
        # always win, so any setting can be changed mid-run just by passing a
        # different value when resuming. Only the adaptive state is restored
        # -- the group lr and the sign history (when its geometry still
        # matches the current config).
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            # One lr per group: unify the restored lrs to their geometric
            # median (they are already identical for checkpoints from this
            # version; older per-tensor checkpoints land on a sane middle).
            lrs = [
                st["lr"]
                for p in group["params"]
                if (st := self.state.get(p)) is not None
                and isinstance(st.get("lr"), torch.Tensor)
            ]
            med = None
            if lrs:
                dev = lrs[0].device
                med = (
                    torch.stack([t.to(torch.float32).to(dev) for t in lrs])
                    .log_()
                    .median()
                    .exp_()
                )
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                if isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)
                    if med is not None:
                        st["lr"].copy_(med.to(st["lr"].device))
                # Sign history: keep it when its geometry matches the current
                # config (the parent cast it to param dtype; recover by shape).
                # On any mismatch (e.g. a checkpoint from an older window
                # layout) -- start fresh.
                numel = p.numel()
                H = group["polarity_history"]
                width = (numel + 7) // 8
                sh = st.get("sign_history")
                hist_ok = (
                    isinstance(sh, torch.Tensor)
                    and sh.shape == (H, width)
                    and isinstance(st.get("hist_idx"), int)
                    and 0 <= st["hist_idx"] < H
                    and isinstance(st.get("hist_fill"), int)
                    and 0 <= st["hist_fill"] <= H
                )
                if hist_ok:
                    st["sign_history"] = sh.to(torch.uint8)
                else:
                    st["sign_history"] = torch.zeros(
                        (H, width), dtype=torch.uint8, device=p.device
                    )
                    st["hist_idx"] = 0
                    st["hist_fill"] = 0
        # The parent rebuilt the group dicts; remap params to groups and
        # reset the vote accumulators.
        self._rebuild_group_index()
