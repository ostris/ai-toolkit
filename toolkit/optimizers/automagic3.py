from typing import List
import torch


class Automagic3(torch.optim.Optimizer):
    """
    Automagic v3.

    A learning rate is kept per row of each parameter: one lr per output
    channel for >=2D weights (e.g. one lr per output neuron of a Linear layer)
    and one lr per element for 1D weights (biases, norms). Each step the lr is
    nudged by how strongly the per-element update direction agrees with the
    previous step. Agreement is the plain fraction of elements that kept their
    sign (every element counts equally, regardless of update magnitude) and is
    used as a proportional signal in [-1, 1] that scales a multiplicative
    (geometric) bump ``lr *= exp(signal * lr_bump_rate)``, so each step is a fixed
    fractional move that behaves uniformly across the whole [min_lr, max_lr]
    range and a full up bump is exactly undone by a full down bump. The lr
    drifts smoothly, self-stabilises near random agreement, and is clamped to
    [min_lr, max_lr].

    The optimizer step is fused into the backward pass via
    ``register_post_accumulate_grad_hook``: each parameter is updated and its
    grad freed as soon as autograd finishes accumulating into it. ``.step()``
    therefore does no real work and peak VRAM stays low.

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
       scales the bump by how strongly the directions agree
       (``2*agreement - 1`` in [-1, 1]). Plain English: the lr nudges gently
       when the signal is weak and firmly when it is strong, and parks itself
       instead of oscillating when gradients are basically noise.

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
    ):
        if lr > 1e-3:
            print(f"Warning! Start lr {lr} is very high; forcing to 1e-6.")
            lr = 1e-6
        defaults = dict(
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            lr_bump_rate=lr_bump_rate,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                    self._hook_handles.append(handle)

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(f"Total training paramiters: {total:,}")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

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

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        # Per-row lr: one entry per output channel for >=2D params, per element
        # for 1D params, a scalar for 0D params.
        lr_shape = (p.shape[0],) if p.dim() >= 2 else p.shape
        state["lr"] = torch.full(
            lr_shape, float(group["lr"]), dtype=torch.float32, device=p.device
        )
        state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
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

        # Per-row sign agreement vs. the previous step: the plain fraction of
        # elements that kept their sign, every element counting equally
        # regardless of update magnitude. For >=2D params this is reduced to one
        # value per output channel; for 1D it is per element. The result drives
        # a proportional lr bump in [-1, 1]. The match is summed straight off the
        # bool tensor (no full-size float cast) into the per-row reduction.
        cur_polarity = update > 0
        eqb = cur_polarity == state["last_polarity"]
        state["last_polarity"] = cur_polarity

        lr_t = state["lr"]
        if p.dim() >= 2:
            dims = tuple(range(1, p.dim()))
            agreement = eqb.sum(dim=dims, dtype=torch.float32).div_(
                eqb.shape[1:].numel()
            )
            lr_b = lr_t.view(lr_t.shape[0], *([1] * (p.dim() - 1)))
        else:
            agreement = eqb.to(torch.float32)
            lr_b = lr_t

        if state["step"] > 0:
            direction = agreement.mul_(2.0).sub_(1.0)
            # Multiplicative (geometric) bump: lr *= exp(direction * lr_bump_rate).
            # A full up step multiplies lr by exp(lr_bump_rate), a full down step by
            # its reciprocal, so up and down are symmetric in log space (a full
            # up bump is exactly undone by a full down bump). lr_bump_rate is thus a
            # fractional rate (~lr_bump_rate per step for small values), giving a
            # uniform relative move at every scale across [min_lr, max_lr]
            # instead of a fixed absolute amount.
            lr_t.mul_(torch.exp(direction.mul_(group["lr_bump_rate"]))).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )
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
            if p.dtype == torch.bfloat16:
                p.copy_(self._sr_truncate(new_p_fp32, 16))
            elif p.dtype == torch.float16:
                p.copy_(self._sr_truncate(new_p_fp32, 13))
            else:
                p.copy_(self._stochastic_round(new_p_fp32, p.dtype))

        p.grad = None

    # ----------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
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
