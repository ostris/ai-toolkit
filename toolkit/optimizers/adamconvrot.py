"""
AdamConvRot: Adam / AdamW with both moments stored in ConvRot n-bit codes.

qtype picks the storage width: "convrot8" or "convrotint2".."convrotint8"
(the convrot_quant names; 8 = plain int8 codes, everything else is the
bit-packed n-bit format of pack_intn_rows). adamw=True is decoupled weight
decay (AdamW), adamw=False is classic L2-in-the-gradient Adam.

Storage. Each moment is viewed flat and cut into 256-element blocks; a block
is rotated with the ConvRot regular-Hadamard transform (rotate(x, 256) from
convrot_quant -- spreads a block's outliers so one symmetric scale covers the
block without clipping the rest) and stored as n-bit symmetric codes with one
fp32 scale per block. The second moment is kept in the SQRT domain (what the
update divides by; half the dynamic range). Cost: bits/4 bytes per element
for both moments plus 1/32 byte of scales -- 2.03 B/elem at 8 bit, 1.03 at 4,
0.53 at 2.

Requantization is NOT dequantize -> EMA -> re-round. That scheme is dead on
arrival for the slow second moment: with beta2 = 0.999 the per-step change
of sqrt(v) is ~0.05%, far below half an int8 step, so nearest rounding
freezes the state (it only ever ratchets up on spikes and never decays), and
per-step stochastic rounding is unbiased but injects one code of noise every
step that the EMA only damps at rate (1 - beta2): ~9 codes rms at 8 bit,
more than the whole 4-bit grid. Instead the uniform decay of an EMA is
applied EXACTLY to the block scale (scale *= beta each step, an fp32
multiply), and only the increment -- (1-beta1)*g for m, the exact
sqrt-domain increment for s -- is rotated and added to the codes with
stochastic rounding, so an element with no gradient decays exactly and the
rounding noise is Poisson in the accumulated increment instead of one code
per step. A block is re-rounded (new scale from its exact fp32 values,
stochastic) only when a code would leave the grid; the new scale leaves
1.25x headroom so growth does not retrigger it every step. A dead block
(scale 0) restarts from its increment on the first non-zero gradient.

The update uses the exact fp32 moments of the current step (before their
requantization), bias-corrected as in torch Adam; parameters are updated in
fp32 and written back with stochastic rounding for bf16 / fp16.

Everything for one parameter -- unpack, 2 rotations back, EMA, update,
parameter write, 2 rotations forward, stochastic requant, pack -- is one
triton kernel launch, one pass over memory, no fp32 temporaries (the
rotation is two batched 16x16 tensor-core dots, exact on integer codes).
CPU / non-triton devices run the identical algorithm in torch. Tensors
below min_quant_numel (or 1D) keep plain fp32 moments.

fused=True updates each parameter inside the backward pass (a
post-accumulate-grad hook) and frees its gradient at once, so the peak never
holds a full set of gradients; .step() then only runs the closure. That
bypasses the trainer's grad clipping / non-finite skip, so non-finite grads
are zeroed in the hook, and it is incompatible with multi-backward gradient
accumulation. Default is the traditional unfused step, where low-precision
(bf16 / fp16) parameters accumulate their micro-batch grads through a
post-accumulate-grad hook that sums in fp32 and stochastically rounds back
into the param's dtype (optimizer_utils.stochastic_grad_accummulation), so
small per-micro-batch grads are not lost to round-to-nearest; .step()
consumes that buffer. Note the trainer's grad clipping does not see it.
"""

import math

import torch

from toolkit.optimizers.optimizer_utils import (
    copy_stochastic,
    stochastic_grad_accummulation,
)
from toolkit.util.convrot_quant import (
    _import_triton,
    _triton_available,
    pack_intn_rows,
    regular_hadamard,
    rotate,
    unpack_intn_rows,
)

BLOCK = 256  # rotation block == scale group
HEADROOM = 1.25  # rescale puts the block max at qmax / HEADROOM
TINY = 1e-30


def _bits_for_qtype(qtype: str) -> int:
    if qtype == "convrot8":
        return 8
    if qtype.startswith("convrotint"):
        try:
            bits = int(qtype[len("convrotint") :])
        except ValueError:
            bits = 0
        if 2 <= bits <= 8:
            return bits
    raise ValueError(
        f"AdamConvRot: unsupported qtype {qtype!r}; use convrot8 or convrotint2..convrotint8"
    )


# ---------------------------------------------------------------- triton path

_kernel = None
_h16_cache = {}


def _h16(device):
    key = str(device)
    h = _h16_cache.get(key)
    if h is None:
        # regular_hadamard(256) == kron(H16, H16): the 256-rotation is H16 over
        # the high base-16 digit and H16 over the low one
        h = regular_hadamard(16, device, torch.float32).contiguous()
        _h16_cache[key] = h
    return h


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    triton, tl = _import_triton()

    @triton.jit
    def _rot256(x, hb, R: tl.constexpr):
        # tf32 dots are exact for integer codes: |c| <= 127 and H16 = +-1/4 keep
        # every partial sum within 11 mantissa bits
        x3 = tl.reshape(x, (R, 16, 16))
        z = tl.dot(x3, hb, input_precision="tf32")
        y3 = tl.dot(hb, z, input_precision="tf32")
        return tl.reshape(y3, (R, 256))

    @triton.jit
    def _unpack256(
        c_ptr,
        b,
        bm,
        BITS: tl.constexpr,
        BPOW: tl.constexpr,
        QMAX: tl.constexpr,
        R: tl.constexpr,
    ):
        # pack_intn_rows layout: 8 codes -> one BITS-byte little-endian word
        gi = tl.arange(0, 32)
        bi = tl.arange(0, BPOW)
        j = tl.arange(0, 8)
        ptrs = (
            c_ptr
            + b[:, None, None] * (32 * BITS)
            + gi[None, :, None] * BITS
            + bi[None, None, :]
        )
        msk = bm[:, None, None] & (bi < BITS)[None, None, :]
        by = tl.load(ptrs, mask=msk, other=0).to(tl.int64)
        word = tl.sum(by << (8 * bi)[None, None, :].to(tl.int64), axis=2)
        v = (word[:, :, None] >> (BITS * j)[None, None, :].to(tl.int64)) & (
            (1 << BITS) - 1
        )
        return tl.reshape((v - QMAX).to(tl.float32), (R, 256))

    @triton.jit
    def _pack256(
        c_ptr,
        b,
        bm,
        q,
        BITS: tl.constexpr,
        BPOW: tl.constexpr,
        QMAX: tl.constexpr,
        R: tl.constexpr,
    ):
        u = tl.reshape(q + QMAX, (R, 32, 8)).to(tl.int64)
        j = tl.arange(0, 8)
        word = tl.sum(u << (BITS * j)[None, None, :].to(tl.int64), axis=2)
        gi = tl.arange(0, 32)
        bi = tl.arange(0, BPOW)
        by = ((word[:, :, None] >> (8 * bi)[None, None, :].to(tl.int64)) & 0xFF).to(
            tl.uint8
        )
        ptrs = (
            c_ptr
            + b[:, None, None] * (32 * BITS)
            + gi[None, :, None] * BITS
            + bi[None, None, :]
        )
        msk = bm[:, None, None] & (bi < BITS)[None, None, :]
        tl.store(ptrs, by, mask=msk)

    @triton.jit
    def _requant256(
        c_ptr,
        s_ptr,
        b,
        bm,
        offs,
        codes,
        scale,
        decay,
        inc_rot,
        rqmax,
        tiny,
        seed_a,
        BITS: tl.constexpr,
        BPOW: tl.constexpr,
        QMAX: tl.constexpr,
        SR: tl.constexpr,
        R: tl.constexpr,
    ):
        s_dec = scale * decay
        values = codes * s_dec[:, None] + inc_rot  # exact fp32 block values
        if SR:
            u = tl.rand(seed_a, offs)
        else:
            u = tl.full((R, 256), 0.5, tl.float32)
        q = tl.floor(values / tl.maximum(s_dec, tiny)[:, None] + u)
        over = tl.max(tl.abs(q), axis=1) > QMAX
        # rescale path: new scale with headroom, same random draw (only the
        # triggering element's rounding is conditioned on it)
        s_out = tl.where(over, tl.max(tl.abs(values), axis=1) / rqmax, s_dec)
        q = tl.floor(values / tl.maximum(s_out, tiny)[:, None] + u)
        q = tl.minimum(tl.maximum(q, -1.0 * QMAX), 1.0 * QMAX)
        tl.store(s_ptr + b, s_out, mask=bm)
        _pack256(c_ptr, b, bm, q.to(tl.int32), BITS, BPOW, QMAX, R)
        return s_out

    @triton.jit(do_not_specialize=["numel", "n_blocks", "seed"])
    def adam_convrot_kernel(
        p_ptr,
        g_ptr,
        cm_ptr,
        sm_ptr,
        cs_ptr,
        ss_ptr,
        h_ptr,
        numel,
        n_blocks,
        lr,
        wd,
        beta1,
        beta2,
        eps,
        inv_bc1,
        inv_sqrt_bc2,
        sqrt_beta2,
        rqmax,
        tiny,
        seed,
        BITS: tl.constexpr,
        BPOW: tl.constexpr,
        QMAX: tl.constexpr,
        ADAMW: tl.constexpr,
        P_KIND: tl.constexpr,
        SR: tl.constexpr,
        SANITIZE: tl.constexpr,
        R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid * R + tl.arange(0, R)
        bm = b < n_blocks
        e = tl.arange(0, 256)
        offs = b[:, None] * 256 + e[None, :]
        em = bm[:, None] & (offs < numel)
        r = tl.arange(0, 16)
        hb = tl.load(
            h_ptr
            + tl.zeros((R, 1, 1), tl.int32)
            + r[None, :, None] * 16
            + r[None, None, :]
        )

        g = tl.load(g_ptr + offs, mask=em, other=0.0).to(tl.float32)
        if SANITIZE:
            g = tl.where((g == g) & (tl.abs(g) < 3.0e38), g, 0.0)
        if not ADAMW:
            p0 = tl.load(p_ptr + offs, mask=em, other=0.0).to(tl.float32)
            g = g + wd * p0

        # second moment (sqrt domain) first: only its reciprocal stays live
        ss = tl.load(ss_ptr + b, mask=bm, other=0.0)
        cs = _unpack256(cs_ptr, b, bm, BITS, BPOW, QMAX, R)
        s_old = _rot256(cs, hb, R) * ss[:, None]
        # increment from the rectified value: quant noise can leave s negative,
        # and an increment against a negative s would grow with the noise
        s_pos = tl.maximum(s_old, 0.0)
        s_new = tl.sqrt(beta2 * s_pos * s_pos + (1.0 - beta2) * g * g)
        inc_s = _rot256(s_new - sqrt_beta2 * s_pos, hb, R)
        ss_new = _requant256(
            cs_ptr,
            ss_ptr,
            b,
            bm,
            offs,
            cs,
            ss,
            sqrt_beta2,
            inc_s,
            rqmax,
            tiny,
            seed + 2,
            BITS,
            BPOW,
            QMAX,
            SR,
            R,
        )
        # denominator floored at the block's code resolution -- a stored
        # sqrt(v) below one code step is unresolved, not small
        den = 1.0 / (tl.maximum(s_new, ss_new[:, None]) * inv_sqrt_bc2 + eps)

        # first moment and the parameter update
        sm = tl.load(sm_ptr + b, mask=bm, other=0.0)
        cm = _unpack256(cm_ptr, b, bm, BITS, BPOW, QMAX, R)
        m_new = beta1 * (_rot256(cm, hb, R) * sm[:, None]) + (1.0 - beta1) * g
        upd = m_new * inv_bc1 * den
        p = tl.load(p_ptr + offs, mask=em, other=0.0).to(tl.float32)
        if ADAMW:
            p_new = p - lr * (upd + wd * p)
        else:
            p_new = p - lr * upd
        inc_m = _rot256(g * (1.0 - beta1), hb, R)
        _requant256(
            cm_ptr,
            sm_ptr,
            b,
            bm,
            offs,
            cm,
            sm,
            beta1,
            inc_m,
            rqmax,
            tiny,
            seed,
            BITS,
            BPOW,
            QMAX,
            SR,
            R,
        )

        if P_KIND == 0:
            tl.store(p_ptr + offs, p_new, mask=em)
        elif P_KIND == 1:
            if SR:
                # copy_stochastic_bf16: random 16-bit add, truncate
                xi = p_new.to(tl.int32, bitcast=True)
                r16 = (tl.randint(seed + 4, offs) & 0xFFFF).to(tl.int32)
                xi = (xi + r16) & -65536
                out = xi.to(tl.float32, bitcast=True).to(tl.bfloat16)
            else:
                out = p_new.to(tl.bfloat16)
            tl.store(p_ptr + offs, out, mask=em)
        else:
            if SR:
                # fp16 ulp from the fp32 exponent (subnormals below 2^-14)
                ex = (p_new.to(tl.int32, bitcast=True) >> 23) & 0xFF
                ulp = tl.exp2((tl.maximum(ex - 127, -14) - 10).to(tl.float32))
                u = tl.rand(seed + 4, offs)
                out = (tl.floor(p_new / ulp + u) * ulp).to(tl.float16)
            else:
                out = p_new.to(tl.float16)
            tl.store(p_ptr + offs, out, mask=em)

    _kernel = adam_convrot_kernel
    return _kernel


# ------------------------------------------------------------------ optimizer


class AdamConvRot(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        qtype="convrot8",
        adamw=True,
        min_quant_numel=4096,
        fused=False,
        blocks_per_program=8,  # 8x4 warps: no register spills, ~1.1 ms per 64M elems on a 5090
        num_warps=4,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        self.bits = _bits_for_qtype(qtype)
        self.qtype = qtype
        self.adamw = bool(adamw)
        self.min_quant_numel = int(min_quant_numel)
        self.blocks_per_program = int(blocks_per_program)
        self.num_warps = int(num_warps)
        # off only for validation (nearest rounding, kernel == torch path)
        self.stochastic_rounding = True
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        self.fused = bool(fused)
        self._hook_handles = []
        self._rebuild_group_index()
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.fused:
                    self._hook_handles.append(
                        p.register_post_accumulate_grad_hook(self._fused_hook)
                    )
                elif p.dtype != torch.float32:
                    # stochastic bf16/fp16 grad accumulation across micro-batches
                    self._hook_handles.append(
                        p.register_post_accumulate_grad_hook(
                            stochastic_grad_accummulation
                        )
                    )

    def _rebuild_group_index(self):
        # the hooks cannot rely on group-dict identity: load_state_dict
        # replaces the group dicts
        self._param_group_index = {
            p: gi for gi, group in enumerate(self.param_groups) for p in group["params"]
        }

    @torch.no_grad()
    def _fused_hook(self, p):
        gi = self._param_group_index.get(p)
        if gi is None:
            self._rebuild_group_index()
            gi = self._param_group_index.get(p, 0)
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        self._update_param(p, self.param_groups[gi], seed, sanitize=True)
        p.grad = None

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def rescale_qmax(self) -> float:
        return max(self.qmax / HEADROOM, 1.0)

    def _quantizes(self, p: torch.Tensor) -> bool:
        return p.dim() >= 2 and p.numel() >= self.min_quant_numel

    # ------------------------------------------------------------------ state

    def _init_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        if self._quantizes(p):
            n_blocks = -(-p.numel() // BLOCK)
            zero_codes = pack_intn_rows(
                torch.zeros(n_blocks, BLOCK, dtype=torch.int8, device=p.device),
                self.bits,
            )
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = zero_codes.clone()
                # fp32 scales as a uint8 view: the parent load_state_dict casts
                # floating state to the param dtype
                state[key + "_scale"] = torch.zeros(
                    n_blocks, dtype=torch.float32, device=p.device
                ).view(torch.uint8)
        else:
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

    def _state_ok(self, p: torch.Tensor, state) -> bool:
        if "exp_avg" not in state or "step" not in state:
            return False
        if self._quantizes(p):
            n_blocks = -(-p.numel() // BLOCK)
            return (
                "exp_avg_scale" in state
                and tuple(state["exp_avg"].shape) == (n_blocks, 32 * self.bits)
                and state["exp_avg"].dtype == torch.uint8
            )
        return state["exp_avg"].shape == p.shape

    def load_state_dict(self, state_dict):
        # the parent casts every state tensor to the param dtype, which would
        # turn the uint8 code / scale buffers (and fp32 moments) into bf16
        super().load_state_dict(state_dict)
        saved_ids = [pid for g in state_dict["param_groups"] for pid in g["params"]]
        cur_params = [p for g in self.param_groups for p in g["params"]]
        id_map = dict(zip(saved_ids, cur_params))
        for pid, saved in state_dict["state"].items():
            p = id_map.get(pid)
            if p is None:
                continue
            for k, v in saved.items():
                if k != "step" and torch.is_tensor(v):
                    self.state[p][k] = v.to(p.device, copy=True)
        self._rebuild_group_index()

    # ------------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.fused:
            return loss  # every param was updated inside backward

        step_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        param_index = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_index += 1
                accum = getattr(p, "_accum_grad", None)
                if accum is not None:
                    p.grad = accum
                    del p._accum_grad
                if p.grad is None:
                    continue
                seed = (step_seed ^ (param_index * 0x9E3779B1)) & 0x7FFFFFFF
                self._update_param(p, group, seed, sanitize=False)
        return loss

    def _update_param(self, p, group, seed, sanitize):
        if p.grad.is_sparse:
            raise RuntimeError("AdamConvRot does not support sparse gradients")
        beta1, beta2 = group["betas"]
        state = self.state[p]
        if not self._state_ok(p, state):
            if len(state) > 0:
                print(
                    "WARNING: AdamConvRot state does not match the parameter "
                    "or qtype; re-initializing its moments."
                )
            self._init_state(p)
        state["step"] += 1
        t = state["step"]
        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        if self._quantizes(p):
            self._step_quant(p, state, group, bc1, bc2, seed, sanitize)
        else:
            self._step_fp32(p, state, group, bc1, bc2, sanitize)

    def _step_fp32(self, p, state, group, bc1, bc2, sanitize=False):
        beta1, beta2 = group["betas"]
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        g = p.grad.to(torch.float32)
        if sanitize:
            g = g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        p32 = p.to(torch.float32)
        if not self.adamw and wd != 0.0:
            g = g.add(p32, alpha=wd)
        m, v = state["exp_avg"], state["exp_avg_sq"]
        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        upd = (m / bc1) / ((v / bc2).sqrt_().add_(eps))
        if self.adamw and wd != 0.0:
            upd.add_(p32, alpha=wd)
        if p.dtype == torch.float32:
            p.add_(upd, alpha=-lr)
            return
        p32.add_(upd, alpha=-lr)
        if self.stochastic_rounding and p.is_cuda:
            copy_stochastic(p, p32)
        else:
            p.copy_(p32)

    def _step_quant(self, p, state, group, bc1, bc2, seed, sanitize=False):
        use_triton = (
            p.is_cuda
            and p.grad.is_cuda
            and p.is_contiguous()
            and p.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and _triton_available()
        )
        if use_triton:
            self._step_quant_triton(p, state, group, bc1, bc2, seed, sanitize)
        else:
            self._step_quant_torch(p, state, group, bc1, bc2, sanitize)

    def _step_quant_triton(self, p, state, group, bc1, bc2, seed, sanitize=False):
        beta1, beta2 = group["betas"]
        g = p.grad
        if not g.is_contiguous():
            g = g.contiguous()
        n_blocks = state["exp_avg"].shape[0]
        R = self.blocks_per_program
        kernel = _get_kernel()
        p_kind = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[p.dtype]
        kernel[(-(-n_blocks // R),)](
            p,
            g,
            state["exp_avg"],
            state["exp_avg_scale"].view(torch.float32),
            state["exp_avg_sq"],
            state["exp_avg_sq_scale"].view(torch.float32),
            _h16(p.device),
            p.numel(),
            n_blocks,
            float(group["lr"]),
            float(group["weight_decay"]),
            float(beta1),
            float(beta2),
            float(group["eps"]),
            1.0 / bc1,
            1.0 / math.sqrt(bc2),
            math.sqrt(beta2),
            float(self.rescale_qmax),
            TINY,
            int(seed),
            BITS=self.bits,
            BPOW=max(2, 1 << (self.bits - 1).bit_length()),
            QMAX=self.qmax,
            ADAMW=self.adamw,
            P_KIND=p_kind,
            SR=self.stochastic_rounding,
            SANITIZE=bool(sanitize),
            R=R,
            num_warps=self.num_warps,
        )

    # ------------------------------------------------------------ torch path

    def _dequant(self, codes, scale_u8, n_blocks):
        c = unpack_intn_rows(codes, self.bits, n_blocks, BLOCK).to(torch.float32)
        return (rotate(c, BLOCK) * scale_u8.view(torch.float32)[:, None]).reshape(-1)

    def _requant(self, codes, scale_u8, decay, inc_rot):
        qmax = self.qmax
        scale = scale_u8.view(torch.float32)
        n_blocks = codes.shape[0]
        c = unpack_intn_rows(codes, self.bits, n_blocks, BLOCK).to(torch.float32)
        s_dec = scale * decay
        values = c * s_dec[:, None] + inc_rot
        u = torch.rand_like(inc_rot) if self.stochastic_rounding else 0.5
        q = torch.floor(values / s_dec.clamp_min(TINY)[:, None] + u)
        over = q.abs().amax(dim=1) > qmax
        s_out = torch.where(over, values.abs().amax(dim=1) / self.rescale_qmax, s_dec)
        q = torch.floor(values / s_out.clamp_min(TINY)[:, None] + u).clamp_(-qmax, qmax)
        codes.copy_(pack_intn_rows(q.to(torch.int8), self.bits))
        scale.copy_(s_out)
        return scale

    def _step_quant_torch(self, p, state, group, bc1, bc2, sanitize=False):
        beta1, beta2 = group["betas"]
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        numel = p.numel()
        n_blocks = state["exp_avg"].shape[0]
        n_pad = n_blocks * BLOCK
        g = torch.zeros(n_pad, dtype=torch.float32, device=p.device)
        g[:numel] = p.grad.reshape(-1).to(torch.float32)
        if sanitize:
            g.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        p32 = torch.zeros(n_pad, dtype=torch.float32, device=p.device)
        p32[:numel] = p.reshape(-1).to(torch.float32)
        if not self.adamw:
            g.add_(p32, alpha=wd)

        m_old = self._dequant(state["exp_avg"], state["exp_avg_scale"], n_blocks)
        m_new = beta1 * m_old + (1.0 - beta1) * g
        self._requant(
            state["exp_avg"],
            state["exp_avg_scale"],
            beta1,
            rotate((g * (1.0 - beta1)).view(n_blocks, BLOCK), BLOCK),
        )
        s_old = self._dequant(state["exp_avg_sq"], state["exp_avg_sq_scale"], n_blocks)
        s_pos = s_old.clamp_min(0.0)
        s_new = torch.sqrt(beta2 * s_pos * s_pos + (1.0 - beta2) * g * g)
        sqrt_beta2 = math.sqrt(beta2)
        ss_new = self._requant(
            state["exp_avg_sq"],
            state["exp_avg_sq_scale"],
            sqrt_beta2,
            rotate((s_new - sqrt_beta2 * s_pos).view(n_blocks, BLOCK), BLOCK),
        )

        # denominator floored at the block's code resolution (see the kernel)
        s_eff = torch.maximum(s_new, ss_new.repeat_interleave(BLOCK))
        upd = (m_new / bc1) / (s_eff / math.sqrt(bc2) + eps)
        if self.adamw:
            upd.add_(p32, alpha=wd)
        p32.add_(upd, alpha=-lr)
        new = p32[:numel].view(p.shape)
        if p.dtype == torch.float32:
            p.copy_(new)
        elif self.stochastic_rounding and p.is_cuda:
            copy_stochastic(p, new)
        else:
            p.copy_(new)
