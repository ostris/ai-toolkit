from typing import List

import torch
from optimum.quanto import QBytesTensor
from toolkit.optimizers.optimizer_utils import (
    copy_stochastic,
    stochastic_grad_accummulation,
)
from toolkit.util.convrot_quant import (
    largest_pow4_divisor,
    quantize_int8_rows_fused,
    rotate,
)


class _ConvRot8State:
    """
    ConvRot int8 storage for optimizer moment state: values are rotated with
    the block regular-Hadamard transform along the trailing dims (the rotation
    spreads outliers so one symmetric per-row scale is safe -- same scheme as
    the convrot8 weight backend) and stored as int8 codes with one fp32 scale
    per row. Quantization runs through the fused triton kernel
    (quantize_int8_rows_fused): a single pass instead of a chain of eager
    kernels; dequantization is one scale multiply plus the (self-inverse)
    rotation matmul.

    Tensors are viewed as (shape[0], numel // shape[0]) -- per-output-channel
    rows, matching the convrot8 weight layout.
    """

    def __init__(self, source, rot_size: int = 256):
        if isinstance(source, dict):  # constructor from state dict
            self._load_from_state_dict(source)
            return
        self.shape = source.shape
        self.numel = source.numel()
        self.rows = source.shape[0]
        self.K = self.numel // self.rows
        rot = min(rot_size, largest_pow4_divisor(self.K))
        self.rot = rot if rot >= 16 else 1
        self.quantized = None
        self.scale = None
        self.quantize_(source)

    def quantize_(self, values_fp32: torch.Tensor):
        x = rotate(values_fp32.reshape(self.rows, self.K), self.rot)
        self.quantized, self.scale = quantize_int8_rows_fused(x)

    def dequantize(self) -> torch.Tensor:
        w = self.quantized.to(torch.float32).mul_(self.scale.unsqueeze(1))
        return rotate(w, self.rot).reshape(self.shape)  # self-inverse

    def state_dict(self):
        return {
            'quantized': self.quantized,
            'scale': self.scale,
            'shape': tuple(self.shape),
            'numel': self.numel,
            'rot': self.rot,
        }

    def _load_from_state_dict(self, state_dict):
        self.quantized = state_dict['quantized']
        self.scale = state_dict['scale'].to(torch.float32)
        self.shape = torch.Size(state_dict['shape'])
        self.numel = state_dict['numel']
        self.rows = self.shape[0]
        self.K = self.numel // self.rows
        self.rot = state_dict['rot']

    def to_(self, device):
        self.quantized = self.quantized.to(device)
        self.scale = self.scale.to(device)
        return self


class AutomagicEXPERIMENT(torch.optim.Optimizer):
    """
    Automagic3's learning-rate controller with a FULL per-element second
    moment (AdamW-style, stored in ConvRot int8) instead of Adafactor's
    factored approximation. Deliberately the proven v3 design, ported
    faithfully; validated to settle where actual Automagic3 settles.

    NO MOMENTUM -- beta1 must be 0, and this is a theorem of the control
    law, not a preference: v3's overshoot detector is perfect period-2
    gradient alternation, which only exists when the applied update tracks
    the instantaneous gradient. Momentum smooths the step into long-period
    heavy-ball orbits, so an overshoot bounce produces RUNS of same-sign
    gradients that the sign window reads as consistency -- the controller
    then votes the lr UP during a bounce (verified: every acceptance
    scenario exploded with beta1 = 0.9). This is why automagic has always
    been momentum-free.

    The update: v = beta2 EMA of grad^2 (full per-element, bias-corrected),
    update = grad / (sqrt(v_hat) + eps), passed through v3's trust region
    (RMS-scaled then clamped elementwise to clip_threshold -- required with
    8-bit state, whose under-resolved second-moment elements dequantize to
    zero and would otherwise produce exploding elements). Applied step:
    weights -= group_lr * update (+ decoupled weight decay).

    THE CONTROLLER (v3, verbatim): each element records the SIGN of its
    gradient into an H-step 1-bit ring buffer (H = polarity_history). Only
    the two perfectly decisive window states vote -- all H signs agreeing
    (either direction) votes up ("step too small"), all H-1 transitions
    flipping (the period-2 overshoot bounce) votes down ("step too large");
    everything else is noise and votes 0. Two patterns each, identical
    pure-noise probability, exact balance. Votes are weighted by |update|
    and pooled across every element of every tensor in the group; ONE lr per
    group moves as lr *= exp(pooled vote), clamped only by the min_lr/max_lr
    failsafes (parked decades outside any operating range -- if they are
    ever touched, the math failed). Pooling at group level is the
    load-bearing choice: coupled tensors (q/k pairs) fight per-tensor lrs;
    one shared lr makes their opposing votes cancel in the pool. The
    controller is low-gain, integral, and bounded -- there is no estimator
    whose failure can become a training failure, the lesson of five
    destroyed finetune runs on a measurement-based secant controller that
    this file replaces.

    With fused=True (default) the step is fused into the backward pass via
    register_post_accumulate_grad_hook: each parameter is updated and its
    grad freed as soon as autograd finishes accumulating into it, so .step()
    only applies the pooled group votes. Note this bypasses the trainer's
    grad clipping / nan-skip (they run after backward) -- non-finite grads
    are neutralized in the hook instead -- and is not compatible with
    multi-backward gradient accumulation. With fused=False it behaves like a
    traditional optimizer: grads accumulate across backward passes and the
    update happens in .step(); low-precision grads are accumulated with
    stochastic rounding.

    State per element: the second moment in ConvRot int8 for >=2D params
    (stored in SQRT domain -- halves the dynamic range the linear code has
    to cover and is the quantity the update divides by; fp32 for 1D params)
    plus the H/8-byte packed sign history -- ~2 bytes/element total at the
    default H=8. Updates to low-precision parameters are applied in fp32 and
    stochastically rounded on write-back.

    The reported lr (get_avg_learning_rate) is the parameter-count-weighted
    average of the group lrs, in ABSOLUTE units, directly comparable to a
    classic optimizer lr.
    """

    def __init__(
        self,
        params,
        lr=1e-6,  # start lr; the controller adapts away from it
        min_lr=1e-30,  # FAILSAFES only, parked decades outside any operating
        max_lr=1e3,    # range -- if either is ever touched, the math failed
        betas=(0.0, 0.999),  # beta1 MUST be 0: momentum breaks the sign-window
        eps=1e-8,           # overshoot detector (see the class docstring)
        clip_threshold=1.0,
        weight_decay=0.0,
        polarity_history=8,  # sign-window length H (2-64); H/8 bytes/element
        fused=True,
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "weight_decay": weight_decay,
            "polarity_history": max(2, min(64, int(polarity_history))),
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [
            lr for group in self.param_groups
        ]

        self.fused = fused
        self.is_stochastic_rounding_accumulation = False
        self._hook_handles = []
        self._rebuild_group_index()

        for group in self.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    continue
                if self.fused:
                    # Fused: update each param the moment its grad is ready.
                    self._hook_handles.append(
                        param.register_post_accumulate_grad_hook(
                            self._make_backward_hook()
                        )
                    )
                elif param.dtype != torch.float32:
                    # Non-fused: accumulate low-precision grads across
                    # micro-batches with stochastic rounding; the update
                    # happens in .step().
                    self.is_stochastic_rounding_accumulation = True
                    self._hook_handles.append(
                        param.register_post_accumulate_grad_hook(
                            stochastic_grad_accummulation
                        )
                    )

        total = 0
        for group in self.param_groups:
            for param in group['params']:
                total += torch.numel(param)
        print(f"Total training paramiters: {total:,}")

    # ------------------------------------------------------------------ utils

    def _rebuild_group_index(self):
        # param -> index of its param group, plus per-group vote accumulators
        # (gathered across every tensor in the group during the step and
        # applied once in .step()). The map exists because the fused hooks
        # cannot rely on group-dict identity: the parent's load_state_dict
        # replaces the group dicts.
        self._param_group_index = {
            p: gi
            for gi, group in enumerate(self.param_groups)
            for p in group['params']
        }
        self._group_num: List = [None] * len(self.param_groups)
        self._group_den: List = [None] * len(self.param_groups)

    def _make_backward_hook(self):
        def _hook(p: torch.Tensor):
            gi = self._param_group_index.get(p)
            if gi is None:
                self._rebuild_group_index()
                gi = self._param_group_index.get(p, 0)
            self._update_param(p, self.param_groups[gi], gi)
        return _hook

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    # Per-device cached constants for 1-bit pack/unpack (from automagic3).
    _PACK_CONSTS: dict = {}

    @classmethod
    def _pack_consts(cls, device):
        consts = cls._PACK_CONSTS.get(device)
        if consts is None:
            consts = (
                torch.tensor(
                    [1, 2, 4, 8, 16, 32, 64, 128], device=device,
                    dtype=torch.uint8
                ),
                torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.uint8
                ),
            )
            cls._PACK_CONSTS[device] = consts
        return consts

    @classmethod
    def _pack_bits(cls, bits: torch.Tensor) -> torch.Tensor:
        # Pack sign bits (bool / {0, 1}) 8 per byte (uint8).
        weights, _ = cls._pack_consts(bits.device)
        flat = bits.reshape(-1).to(torch.uint8)
        pad = (-flat.numel()) % 8
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        return (flat.view(-1, 8) * weights).sum(-1, dtype=torch.uint8)

    # ------------------------------------------------------------ lr reporting

    @staticmethod
    def _get_lr(param_group, param_state):
        if 'lr' in param_state:
            return param_state['lr']
        return 0.0

    def _get_group_lr(self, group):
        # average weighted by parameter count
        total = 0.0
        count = 0
        for p in group["params"]:
            n = torch.numel(p)
            total = total + self._get_lr(group, self.state[p]) * n
            count += n
        if count == 0:
            return self.lr
        return total / count

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    # automagic manages its own lr
    def get_learning_rates(self):
        lrs = [
            self._get_group_lr(group)
            for group in self.param_groups
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs

    def get_avg_learning_rate(self):
        # weighted by parameter count across all groups
        total = 0.0
        count = 0
        for group in self.param_groups:
            for p in group["params"]:
                n = torch.numel(p)
                total = total + self._get_lr(group, self.state[p]) * n
                count += n
        if count == 0:
            return self.lr
        return total / count

    # -------------------------------------------------------------- per-param

    @torch.no_grad()
    def _update_param(self, p, group, group_index):
        if p.grad is None:
            return

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError(
                "AutomagicAdamW does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        # In fused mode this runs inside backward, so the trainer's grad
        # clipping and nan/inf-skip come too late to protect us. A single
        # non-finite gradient would poison the moment EMAs (NaN stays NaN
        # forever), so neutralize non-finite grads; those elements contribute
        # nothing this step.
        grad = grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        state = self.state[p]
        if len(state) == 0:
            self.initialize_state(p, group)
        if ('exp_avg_sq' not in state
                or 'sign_history' not in state or 'lr' not in state):
            self.initialize_state(p, group)

        state["step"] += 1
        step = state["step"]
        beta1, beta2 = group["betas"]

        bias_correction2 = 1.0 - beta2 ** step

        if beta1 != 0.0:
            raise ValueError(
                "AutomagicAdamW requires beta1 = 0: momentum smooths the "
                "applied step into long-period heavy-ball orbits, so an "
                "overshoot bounce produces RUNS of same-sign gradients "
                "instead of period-2 alternation -- the sign-window "
                "controller then reads the bounce as consistency and votes "
                "the lr UP. v3's control law is only valid when the update "
                "tracks the instantaneous gradient."
            )

        quantized = isinstance(state["exp_avg_sq"], _ConvRot8State)
        if quantized:
            # 8-bit second moment (see _ConvRot8State); the EMA math runs
            # on an fp32 dequantized copy which is requantized right after.
            # The store holds sqrt(v), so square it back for the EMA --
            # clamping first: rotation quant noise can leave tiny negatives,
            # and squaring them would bias v upward.
            v = state["exp_avg_sq"].dequantize().clamp_(min=0.0).square_()
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            v_sqrt = v.sqrt_()  # v is not needed past here; sqrt in place
            state["exp_avg_sq"].quantize_(v_sqrt)
            denom = v_sqrt.div_(bias_correction2 ** 0.5).add_(group["eps"])
        else:
            # 1D params: plain fp32 second moment
            v = state["exp_avg_sq"]
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = (v / bias_correction2).sqrt_().add_(group["eps"])

        update = grad / denom

        # v3's trust region: scale the update so its RMS is <= clip_threshold,
        # then clamp each element to +/-clip_threshold. NOT optional with
        # 8-bit moments: elements whose sqrt(v) falls below the int8
        # resolution dequantize to zero, so their denominator collapses to
        # eps and the raw update explodes by orders of magnitude.
        update.div_(
            (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])

        # ---- the vote bits: v3's rule, verbatim, on GRADIENT signs ----
        # Adafactor's update sign IS the gradient sign (no momentum), so
        # v3's polarity bits were always gradient-sign bits; porting them
        # unchanged keeps every property that made v3 robust. A window of H
        # consistent signs (all ones or all zeros -- direction-agnostic)
        # votes up; perfect alternation (the overshoot bounce: the gradient
        # flips every step regardless of momentum smoothing in the applied
        # update) votes down; anything mixed is noise. Two patterns each,
        # identical pure-noise probability, exact balance. Weighted by
        # |update| and pooled into the group accumulators; the single group
        # lr is nudged once per step in .step(). (A hypergradient bit
        # against the momentum direction was tried here and exploded every
        # scenario: momentum holds the reference direction through a bounce,
        # so the overshoot signature never fires. Gauge immunity comes from
        # POOLING -- opposing votes cancel -- not from the bit definition.)
        H = group["polarity_history"]
        hist = state["sign_history"]  # (H, numel/8) 1-bit packed uint8
        cur_bits = grad > 0
        idx = state["hist_idx"]
        hist[idx].copy_(self._pack_bits(cur_bits))
        state["hist_idx"] = (idx + 1) % H
        fill = min(H, state["hist_fill"] + 1)
        state["hist_fill"] = fill

        if fill == H:
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
            if self._group_num[group_index] is None:
                self._group_num[group_index] = num
                self._group_den[group_index] = den
            else:
                acc = self._group_num[group_index]
                if num.device != acc.device:
                    num = num.to(acc.device)
                    den = den.to(acc.device)
                acc.add_(num)
                self._group_den[group_index].add_(den)

        lr_t = state['lr']  # this param's mirror of the shared group lr

        p_data_fp32 = p
        if isinstance(p_data_fp32, QBytesTensor):
            p_data_fp32 = p_data_fp32.dequantize()
        if p.dtype != torch.float32:
            p_data_fp32 = p_data_fp32.clone().float()

        if group["weight_decay"] != 0:
            # Decoupled weight decay folded into the direction:
            # p -= lr * (update + weight_decay * p)
            update.add_(p_data_fp32, alpha=group["weight_decay"])

        p_data_fp32.addcmul_(update, lr_t, value=-1.0)

        if p.dtype != torch.float32:
            # apply stochastic rounding
            copy_stochastic(p, p_data_fp32)

        p.grad = None

    # ----------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Fused mode already updated every param inside the backward pass;
        # only the pooled group votes remain.
        if not self.fused:
            self.step_hook()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None or not p.requires_grad:
                        continue
                    gidx = self._param_group_index.get(p)
                    if gidx is None:
                        self._rebuild_group_index()
                        gidx = self._param_group_index.get(p, 0)
                    self._update_param(p, group, gidx)

        self._apply_group_votes()
        return loss

    def _apply_group_votes(self):
        # ONE lr nudge per group per step, from the pooled vote of every
        # element of every tensor in the group. Each param's lr tensor
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
                if st is None or 'lr' not in st:
                    continue
                lr_t = st['lr']
                f = factor if factor.device == lr_t.device \
                    else factor.to(lr_t.device)
                lr_t.mul_(f).clamp_(min=self.min_lr, max=self.max_lr)
            self._group_num[gi] = None
            self._group_den[gi] = None

    # ------------------------------------------------------------------- state

    def initialize_state(self, p, group=None):
        if group is None:
            gi = self._param_group_index.get(p, 0)
            group = self.param_groups[gi]
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0

        if 'lr' not in state:
            # the group lr, mirrored per param (identical multiplicative
            # nudges keep them equal)
            state['lr'] = torch.tensor(
                min(max(float(self.lr), self.min_lr), self.max_lr),
                dtype=torch.float32, device=p.device)

        H = group["polarity_history"]
        width = (p.numel() + 7) // 8
        if 'sign_history' not in state:
            state['sign_history'] = torch.zeros(
                (H, width), dtype=torch.uint8, device=p.device)
            state['hist_idx'] = 0
            state['hist_fill'] = 0

        if 'exp_avg_sq' not in state:
            zeros = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            if p.dim() >= 2:
                # holds sqrt(exp_avg_sq) -- see the class docstring
                state["exp_avg_sq"] = _ConvRot8State(zeros)
            else:
                # 1D params: plain fp32 buffer, negligible memory
                state["exp_avg_sq"] = zeros

    # keys stored as quantized-tensor objects, serialized via their own
    # state_dicts and restored by hand in load_state_dict
    _QUANT_KEYS = ('exp_avg_sq',)

    def state_dict(self, *args, **kwargs):
        orig_state_dict = super().state_dict(*args, **kwargs)
        new_save_state = {}
        for p, state in orig_state_dict['state'].items():
            save_state = {k: v for k, v in state.items()
                          if k not in self._QUANT_KEYS}
            for key in self._QUANT_KEYS:
                if key in state:
                    val = state[key]
                    save_state[key] = (
                        val if isinstance(val, torch.Tensor)
                        else val.state_dict()
                    )
            new_save_state[p] = save_state
        orig_state_dict['state'] = new_save_state
        return orig_state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Validate the format; older formats start fresh.
        is_valid_state = False
        if 'state' in state_dict and isinstance(state_dict['state'], dict):
            for param_id, param_state in state_dict['state'].items():
                if isinstance(param_state, dict) and 'sign_history' in param_state:
                    is_valid_state = True
                    break
        if not is_valid_state:
            return

        # Parent load without the quantized buffers (its recursive cast would
        # convert their fp32 scales to the param dtype).
        state_dict_copy = {
            'state': {},
            'param_groups': state_dict['param_groups']
        }
        for param_id, param_state in state_dict['state'].items():
            state_dict_copy['state'][param_id] = {
                k: v for k, v in param_state.items()
                if k not in self._QUANT_KEYS
            }
        super().load_state_dict(state_dict_copy)

        # Hyperparameters are NOT loaded from the checkpoint: constructor
        # args always win (any setting can be changed mid-run by resuming
        # with a different value). Only adaptive state is restored.
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v

        self._rebuild_group_index()

        current_params = [
            p for group in self.param_groups for p in group['params']
        ]
        saved_param_count = sum(
            len(g['params']) for g in state_dict['param_groups'])
        if len(current_params) != saved_param_count:
            print(f"WARNING: Number of parameters doesn't match between saved state ({saved_param_count}) "
                  f"and current model ({len(current_params)}). Optimizer state may not be correctly loaded.")

        # One lr per group: unify restored lrs to their geometric median
        # (identical already for checkpoints from this version).
        for group in self.param_groups:
            lrs = [
                st['lr']
                for p in group['params']
                if (st := self.state.get(p)) is not None
                and isinstance(st.get('lr'), torch.Tensor)
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
            H = group["polarity_history"]
            for p in group['params']:
                st = self.state.get(p)
                if st is None:
                    continue
                if isinstance(st.get('lr'), torch.Tensor):
                    st['lr'] = st['lr'].to(torch.float32)
                    if med is not None:
                        st['lr'].copy_(med.to(st['lr'].device))
                # Sign history: keep when its geometry matches the current
                # config; otherwise start fresh (one re-warmup of H steps).
                numel = p.numel()
                width = (numel + 7) // 8
                sh = st.get('sign_history')
                hist_ok = (
                    isinstance(sh, torch.Tensor)
                    and sh.shape == (H, width)
                    and isinstance(st.get('hist_idx'), int)
                    and 0 <= st['hist_idx'] < H
                    and isinstance(st.get('hist_fill'), int)
                    and 0 <= st['hist_fill'] <= H
                )
                if hist_ok:
                    st['sign_history'] = sh.to(torch.uint8)
                else:
                    st['sign_history'] = torch.zeros(
                        (H, width), dtype=torch.uint8, device=p.device)
                    st['hist_idx'] = 0
                    st['hist_fill'] = 0

        for saved_param_id, saved_state in state_dict['state'].items():
            if 'sign_history' not in saved_state:
                continue
            if not isinstance(saved_param_id, int) or not (0 <= saved_param_id < len(current_params)):
                continue
            i = saved_param_id
            current_param = current_params[i]
            if current_param not in self.state:
                self.initialize_state(current_param)
            current_state = self.state[current_param]

            # Reconstruct the quantized buffers: 8-bit ConvRot dicts for >=2D
            # params, plain fp32 tensors for 1D params
            for key in self._QUANT_KEYS:
                saved_buf = saved_state.get(key)
                restored = None
                if saved_buf is not None:
                    try:
                        if (
                            isinstance(saved_buf, dict)
                            and saved_buf.get('numel') == current_param.numel()
                        ):
                            restored = _ConvRot8State(
                                saved_buf).to_(current_param.device)
                        elif (
                            isinstance(saved_buf, torch.Tensor)
                            and saved_buf.shape == current_param.shape
                        ):
                            restored = saved_buf.to(
                                device=current_param.device,
                                dtype=torch.float32)
                        else:
                            print(f"WARNING: Could not restore {key} for parameter {i}. "
                                  f"Initializing fresh.")
                    except Exception as e:
                        print(f"ERROR: Failed to load {key} for parameter {i}: {e}")
                if restored is None:
                    zeros = torch.zeros(
                        current_param.shape, dtype=torch.float32,
                        device=current_param.device)
                    restored = (
                        _ConvRot8State(zeros)
                        if current_param.dim() >= 2 else zeros
                    )
                current_state[key] = restored
