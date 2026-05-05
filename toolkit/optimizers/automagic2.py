from typing import List
import torch


class Automagic2(torch.optim.Optimizer):
    """
    Automagic v2.

    A single scalar learning rate is kept per parameter (e.g. one lr for the
    full weight matrix of a Linear layer rather than one per element). The lr
    is nudged up when the per-element update direction stays consistent with
    the previous step and nudged down when it flips, clamped to [min_lr, max_lr].

    The optimizer step is fused into the backward pass via
    ``register_post_accumulate_grad_hook``: each parameter is updated and its
    grad freed as soon as autograd finishes accumulating into it. ``.step()``
    therefore does no real work and peak VRAM stays low.

    Second-moment EMA state is stored in ``p.dtype`` (math runs in fp32 when
    the state is lower precision). Stochastic rounding is applied only when
    writing back to a bf16 parameter.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
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
            lr_bump=lr_bump,
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

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.full(
            (), float(group["lr"]), dtype=torch.float32, device=p.device
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
            raise RuntimeError("Automagic2 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = grad * grad + eps

        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            if row_state.dtype == torch.float32:
                row, col = row_state, col_state
                row.mul_(beta2).add_(sq.mean(dim=-1), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2), alpha=1.0 - beta2)
            else:
                row = row_state.to(torch.float32)
                col = col_state.to(torch.float32)
                row.mul_(beta2).add_(sq.mean(dim=-1), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2), alpha=1.0 - beta2)
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
            update = v.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # Per-element sign agreement collapsed to a single bump decision.
        # Kept on-device as a 0-D tensor to avoid a CPU<->GPU sync in the hot path.
        cur_polarity = update > 0
        last_polarity = state["last_polarity"]
        agreement = (cur_polarity == last_polarity).to(torch.float32).mean()
        state["last_polarity"] = cur_polarity

        lr_t = state["lr"]
        if state["step"] > 0:
            direction = (agreement >= 0.5).to(lr_t.dtype) * 2.0 - 1.0
            lr_t.add_(direction, alpha=group["lr_bump"]).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )
        state["step"] += 1

        update.mul_(lr_t)
        if group["weight_decay"] != 0.0:
            p_fp32 = p if p.dtype == torch.float32 else p.to(torch.float32)
            update.addcmul_(p_fp32, lr_t, value=group["weight_decay"])

        if p.dtype == torch.bfloat16:
            # Stochastic rounding fp32 -> bf16: add random noise into the lower
            # 16 mantissa bits, then truncate. Done in place on new_p_fp32 so
            # we don't allocate a separate int32 work buffer.
            new_p_fp32 = p.to(torch.float32).sub_(update)
            as_int = new_p_fp32.view(torch.int32)
            as_int.add_(torch.randint_like(as_int, 1 << 16)).bitwise_and_(-65536)
            p.copy_(new_p_fp32)
        else:
            p.add_(update.to(p.dtype), alpha=-1.0)

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
                float(self.state[p]["lr"])
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
        # so subsequent lr_bump (default 1e-6) isn't rounded away on bf16 weights.
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None and isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)
