"""
ConvRot W4A8 codebook backend (qtype ``convrotw4a8``): ComfyUI's ``asym_w4a8_int8``
storage (comfy_kitchen AsymW4A8Int8Layout) as an OstrisQuantizer.

Weights are ConvRot-rotated (block regular Hadamard, default 256), every group of
``group_size`` (16) columns gets an fp8-e4m3 relative scale, and each value is the
nearest of 16 Lloyd-Max codebook levels — an int4 code, two per byte (even column =
low nibble). One fp32 scale per output channel maps the group grid onto the int8 grid
the GEMM consumes:

    w_int8[n, k] = clamp(rint(codebook[code[n, k]] * s_rel[n, k // 16]), -127, 127)
    W_rot[n, k]  = w_int8[n, k] * s_channel[n]

After decoding the codes to that int8 grid the layer IS a convrot8 layer: per-token
int8 activations in the rotated basis against per-channel int8 weights through
torch._int_mm, straight-through backward on the dequantized weight. This backend
subclasses ConvRotInt8Quantizer and owns only the storage:

  - prefill / training (any M): a Triton kernel decodes int4 -> int8 grid per
    forward (a transient the size of an int8 weight), then the parent's _int_mm path.
    The training op saves only the packed codes and decodes again in the backward.
  - decode (M <= 8): a fused Triton GEMV reads the packed nibbles, fp8 group scales
    and codebook directly (int32 multiply-accumulate, no tensor cores), so weight
    traffic per token is 4.5 bits/weight — half of convrot8. Bit-identical output.

Buffers on the module (persistent=False):
  cw4_qdata     int8  [out, in/2]        packed codes
  cw4_srel      uint8 [out, in/group]    fp8 e4m3 group scales (byte view)
  cw4_scales    uint8 [out*4]            fp32 per-channel scales (byte view)
  cw4_codebook  uint8 [16*4]             fp32 codebook (byte view)
  cw4_rot_size, cw4_group_size           ints
"""

from typing import Optional

import torch

from toolkit.util.convrot_quant import (
    ConvRotInt8Quantizer,
    _import_triton,
    _int8_act_quant_padded,
    _int8_epilogue,
    _triton_available,
    rotate,
)

# Lloyd-Max-optimal 16 levels for a group-normalized Gaussian (comfy_kitchen's fixed
# table). ConvRot makes the rotated groups Gaussian, so this table fits every layer.
FIXED_CODEBOOK = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)
# ALS group-scale refinement passes (comfy_kitchen: iteration 1 captures nearly all the gain)
_ALS_ITERS = 2
# rows quantized per chunk: caps the fp32 working set of the level grid ([N, groups, 16])
_QUANT_ROW_ELEM_BUDGET = 1 << 22


# ---------------------------------------------------------------- eager reference

# int8 grid rounding in both kernels: one saturating cvt.rni.sat.s8 (round-nearest-even,
# saturate to [-128, 127]) instead of rint + clamp(-127, 127). The two differ only for a
# code whose group value rounds to <= -127.5, which the fp8 scale rounding makes possible
# but never happened on any Qwen3.8 tensor measured; prefill and decode share the grid.
W4A8_SAT = True


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """unsigned int codes [N, K] in 0..15 -> int8 [N, K/2], even column = low nibble."""
    c = codes.to(torch.int32)
    return ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).to(torch.uint8).view(torch.int8).contiguous()


def _unpack_codes(qdata: torch.Tensor) -> torch.Tensor:
    """int8 [N, K/2] packed -> int64 codes [N, K]."""
    b = qdata.view(torch.uint8).to(torch.int64)
    n, kh = b.shape
    codes = torch.empty(n, kh * 2, dtype=torch.int64, device=qdata.device)
    codes[:, 0::2] = b & 0xF
    codes[:, 1::2] = (b >> 4) & 0xF
    return codes


def unpack_w4a8_eager(qdata: torch.Tensor, srel: torch.Tensor, codebook: torch.Tensor, group_size: int) -> torch.Tensor:
    """Reference decode of packed codes to the int8 grid (comfy_kitchen eager math)."""
    n = qdata.shape[0]
    k = srel.shape[1] * group_size
    values = codebook.float().to(qdata.device)[_unpack_codes(qdata)]
    values = values.view(n, k // group_size, group_size) * srel.float().unsqueeze(-1)
    lo = -128.0 if W4A8_SAT else -127.0
    return values.view(n, k).round().clamp_(lo, 127).to(torch.int8)


def _assign_grid(target: torch.Tensor, levels: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """Nearest level per weight. target [N, G, g] = w_rot / s_channel; levels [N, G, 16]
    non-decreasing. stochastic: pick the upper bracket level with probability
    proportional to the position inside the bracket (unbiased; a merged LoRA delta
    below the int4 step survives instead of rounding away)."""
    n, groups, gsize = target.shape
    lv = levels.reshape(n * groups, -1).contiguous()
    last = lv.shape[1] - 1
    tg = target.reshape(n * groups, gsize).contiguous()
    pos = torch.searchsorted(lv, tg)
    if stochastic:
        lo = (pos - 1).clamp_(0, last - 1)
        level_lo = torch.gather(lv, 1, lo)
        width = torch.gather(lv, 1, lo + 1) - level_lo
        thr = torch.rand_like(tg) * width + level_lo
        return (lo + (tg > thr)).clamp_(0, last).view(n, groups, gsize)
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = (tg - torch.gather(lv, 1, lo)).abs_()
    dhi = (tg - torch.gather(lv, 1, hi)).abs_()
    return torch.where(dhi < dlo, hi, lo).view(n, groups, gsize)


def _assign_codes(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest codebook index (sorted codebook, binary search + neighbour compare)."""
    last = codebook.numel() - 1
    pos = torch.searchsorted(codebook, normalized.contiguous())
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = (normalized - codebook[lo]).abs_()
    dhi = (normalized - codebook[hi]).abs_()
    return torch.where(dhi < dlo, hi, lo)


def grid_levels(codebook: torch.Tensor, srel: torch.Tensor) -> torch.Tensor:
    """The int8 value each codebook level decodes to, per group: [N, G, 16]."""
    return (codebook.float().view(1, 1, -1) * srel.float().unsqueeze(-1)).round_().clamp_(-127, 127)


@torch.no_grad()
def _quantize_rotated_rows(w_rot: torch.Tensor, codebook: torch.Tensor, group_size: int):
    """One row chunk of comfy_kitchen's symmetric codebook W4A8 quantization.
    Returns (packed int8 [n, K/2], s_rel fp8 [n, G], s_channel fp32 [n])."""
    n, k = w_rot.shape
    groups = k // group_size
    gw = w_rot.float().view(n, groups, group_size)
    gs = gw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q = _assign_codes(gw / gs, codebook)
    for _ in range(_ALS_ITERS):
        lv = codebook[q]
        gs = ((gw * lv).sum(-1, keepdim=True) / (lv * lv).sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        q = _assign_codes(gw / gs, codebook)
    shifted = codebook[q] * gs
    s_channel = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    srel = (gs.squeeze(-1) / s_channel.unsqueeze(1)).to(torch.float8_e4m3fn).contiguous()
    # final assignment through the int8 grid the kernels decode to
    codes = _assign_grid(gw / s_channel.view(-1, 1, 1), grid_levels(codebook, srel))
    return _pack_codes(codes.view(n, k)), srel, s_channel.float().contiguous()


@torch.no_grad()
def quantize_w4a8_rotated(w_rot: torch.Tensor, group_size: int = 16, codebook: Optional[torch.Tensor] = None):
    """Quantize an already-rotated fp weight [N, K] into W4A8 storage, row-chunked.
    Returns (packed, s_rel fp8, s_channel fp32, codebook fp32)."""
    n, k = w_rot.shape
    if k % group_size or k % 16:
        raise ValueError(f"W4A8 needs K divisible by 16 and group_size={group_size}, got K={k}")
    if codebook is None:
        codebook = torch.tensor(FIXED_CODEBOOK, dtype=torch.float32, device=w_rot.device)
    codebook = codebook.to(device=w_rot.device, dtype=torch.float32).contiguous()
    block = max(1, _QUANT_ROW_ELEM_BUDGET // max(k, 1))
    parts = [_quantize_rotated_rows(w_rot[r0 : r0 + block], codebook, group_size) for r0 in range(0, n, block)]
    packed = torch.cat([p[0] for p in parts], dim=0)
    srel = torch.cat([p[1] for p in parts], dim=0)
    sch = torch.cat([p[2] for p in parts], dim=0)
    return packed, srel, sch, codebook


# ---------------------------------------------------------------- triton kernels

_kernels = None


def _get_kernels():
    global _kernels
    if _kernels is not None:
        return _kernels
    triton, tl, libdevice = _import_triton()

    @triton.jit
    def _fp8_rep8(s_u8):
        # fp8 e4m3 byte -> fp32, then each group scale repeated over its 8 packed bytes
        s = s_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        s = tl.interleave(s, s)
        s = tl.interleave(s, s)
        return tl.interleave(s, s)

    @triton.jit
    def _sat_s8(v):
        # round-to-nearest-even + saturate to int8 in one PTX cvt (-128 reachable; see SAT)
        return tl.inline_asm_elementwise(
            "cvt.rni.sat.s8.f32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=True, pack=1
        )

    @triton.jit
    def w4a8_unpack_kernel(
        q_ptr, s_ptr, cb_ptr, o_ptr,
        N, KH, G,
        BLOCK_N: tl.constexpr, BLOCK_KH: tl.constexpr, SAT: tl.constexpr,
    ):
        # one [BLOCK_N, BLOCK_KH] tile of packed bytes -> [BLOCK_N, 2*BLOCK_KH] int8 grid.
        # group_size is 16 (8 bytes per group): BLOCK_KH is a multiple of 8.
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bcols = pid_k * BLOCK_KH + tl.arange(0, BLOCK_KH)
        gcols = pid_k * (BLOCK_KH // 8) + tl.arange(0, BLOCK_KH // 8)
        mask_r = rows < N
        m = mask_r[:, None] & (bcols[None, :] < KH)
        byte = tl.load(q_ptr + rows[:, None] * KH + bcols[None, :], mask=m, other=0).to(tl.int32) & 0xFF
        low = byte & 0xF
        high = (byte >> 4) & 0xF
        s_u8 = tl.load(s_ptr + rows[:, None] * G + gcols[None, :], mask=mask_r[:, None] & (gcols[None, :] < G), other=0)
        s8 = _fp8_rep8(s_u8)
        if SAT:
            q_low = _sat_s8(tl.load(cb_ptr + low) * s8).to(tl.int8)
            q_high = _sat_s8(tl.load(cb_ptr + high) * s8).to(tl.int8)
        else:
            q_low = tl.clamp(libdevice.rint(tl.load(cb_ptr + low) * s8), -127.0, 127.0).to(tl.int8)
            q_high = tl.clamp(libdevice.rint(tl.load(cb_ptr + high) * s8), -127.0, 127.0).to(tl.int8)
        K = KH * 2
        tl.store(o_ptr + rows[:, None] * K + (2 * bcols)[None, :], q_low, mask=m)
        tl.store(o_ptr + rows[:, None] * K + (2 * bcols + 1)[None, :], q_high, mask=m)

    @triton.jit
    def w4a8_gemv_kernel(
        xq_ptr, xs_ptr, q_ptr, s_ptr, cb_ptr, ws_ptr, b_ptr, o_ptr,
        K, N,
        MP: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SAT: tl.constexpr,
    ):
        # decode GEMV without tensor cores: the int8-grid weight tile is decoded in
        # registers and multiply-accumulated (int32 IMAD, exact and order independent)
        # against the pre-quantized int8 activation rows, so the output is bit-identical
        # to the _int_mm path. Feeding tl.dot instead costs a shared-memory layout
        # round trip per tile and ran 3x slower on Blackwell.
        # xq [MP, K] int8 (rows >= M are zero), xs [MP] fp32 row scales, out [MP, N].
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_m = tl.arange(0, MP)
        KH = K // 2
        G = K // 16
        # the products accumulate over K in a [MP, BN, BK/2] register tile; one cross-lane
        # reduction at the end instead of one per K tile (~25% faster)
        acc_t = tl.zeros((MP, BLOCK_N, BLOCK_K // 2), tl.int32)
        for k0 in range(0, K, BLOCK_K):
            offs_b = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            m_b = mask_n[:, None] & (offs_b[None, :] < KH)
            byte = tl.load(q_ptr + offs_n[:, None] * KH + offs_b[None, :], mask=m_b, other=0).to(tl.int32) & 0xFF
            low = byte & 0xF
            high = (byte >> 4) & 0xF
            # group_size 16 = 8 packed bytes per fp8 group scale
            s_u8 = tl.load(s_ptr + offs_n[:, None] * G + (offs_b // 8)[None, :], mask=m_b, other=0)
            s = s_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            vl = tl.load(cb_ptr + low) * s
            vh = tl.load(cb_ptr + high) * s
            if SAT:
                ql = _sat_s8(vl)
                qh = _sat_s8(vh)
            else:
                ql = tl.clamp(libdevice.rint(vl), -127.0, 127.0).to(tl.int32)
                qh = tl.clamp(libdevice.rint(vh), -127.0, 127.0).to(tl.int32)
            offs_ke = k0 + 2 * tl.arange(0, BLOCK_K // 2)
            m_x = offs_ke[None, :] < K
            xe = tl.load(xq_ptr + offs_m[:, None] * K + offs_ke[None, :], mask=m_x, other=0).to(tl.int32)
            xo = tl.load(xq_ptr + offs_m[:, None] * K + (offs_ke + 1)[None, :], mask=m_x, other=0).to(tl.int32)
            acc_t += ql[None, :, :] * xe[:, None, :] + qh[None, :, :] * xo[:, None, :]
        acc = tl.sum(acc_t, axis=2)  # [MP, BN]
        ws = tl.load(ws_ptr + offs_n, mask=mask_n, other=0.0)
        xs = tl.load(xs_ptr + offs_m)
        out = acc.to(tl.float32) * (xs[:, None] * ws[None, :])
        if HAS_BIAS:
            out += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        tl.store(o_ptr + offs_m[:, None] * N + offs_n[None, :], out.to(o_ptr.dtype.element_ty), mask=mask_n[None, :])

    @triton.jit
    def w4a8_act_quant_kernel(x_ptr, q_ptr, s_ptr, M, K, QMAX: tl.constexpr, MP: tl.constexpr, BLOCK_K: tl.constexpr):
        # per-row symmetric int8 quant of x [M, K] into exactly MP rows (rows >= M: zeros,
        # scale 1) -- the gemv's padded operand without any fill kernels. Same arithmetic
        # as int8_act_quant_kernel: max is exact under any blocking.
        row = tl.program_id(0)
        valid = row < M
        base = row * K
        acc = tl.zeros((BLOCK_K,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            v = tl.load(x_ptr + base + offs, mask=(offs < K) & valid, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(v))
        amax = tl.max(acc, axis=0)
        scale = tl.where(amax > 0, amax / QMAX, 1.0)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            v = tl.load(x_ptr + base + offs, mask=mask & valid, other=0.0).to(tl.float32)
            q = libdevice.rint(v / scale)
            q = tl.minimum(tl.maximum(q, -1.0 * QMAX), 1.0 * QMAX)
            tl.store(q_ptr + row * K + offs, q.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, scale)

    _kernels = (w4a8_unpack_kernel, w4a8_gemv_kernel, w4a8_act_quant_kernel)
    return _kernels


def _unpack_impl(qdata: torch.Tensor, srel_u8: torch.Tensor, cb_u8: torch.Tensor, group_size: int) -> torch.Tensor:
    n, kh = qdata.shape
    k = kh * 2
    if group_size != 16:
        return unpack_w4a8_eager(qdata, srel_u8.view(torch.float8_e4m3fn), cb_u8.view(torch.float32), group_size)
    out = torch.empty(n, k, dtype=torch.int8, device=qdata.device)
    kernel, _, _ = _get_kernels()
    block_n, block_kh = 32, 128
    grid = (-(-n // block_n), -(-kh // block_kh))
    kernel[grid](qdata, srel_u8, cb_u8.view(torch.float32), out, n, kh, k // 16,
                 BLOCK_N=block_n, BLOCK_KH=block_kh, SAT=W4A8_SAT, num_warps=4)
    return out


@torch.library.custom_op("ostris::convrot_w4a8_unpack", mutates_args=())
def _w4a8_unpack_op(qdata: torch.Tensor, srel_u8: torch.Tensor, cb_u8: torch.Tensor, group_size: int) -> torch.Tensor:
    if not (_triton_available() and qdata.is_cuda):
        return unpack_w4a8_eager(qdata, srel_u8.view(torch.float8_e4m3fn), cb_u8.view(torch.float32), group_size)
    return _unpack_impl(qdata, srel_u8, cb_u8, group_size)


@_w4a8_unpack_op.register_fake
def _w4a8_unpack_fake(qdata, srel_u8, cb_u8, group_size):
    return torch.empty(qdata.shape[0], qdata.shape[1] * 2, dtype=torch.int8, device=qdata.device)


# decode GEMV: M <= W4A8_GEMV_MAX_M rows. the activation rows are int8-quantized by
# the shared convrot act-quant op (bit-identical to the GEMM path) and padded to MP.
W4A8_GEMV_MAX_M = 8


def _gemv_mp(m: int) -> int:
    return 1 << max(0, (m - 1).bit_length())  # 1, 2, 4, 8


@torch.library.custom_op("ostris::convrot_w4a8_act_quant", mutates_args=())
def _w4a8_act_quant_op(x2d: torch.Tensor, act_qmax: int) -> list[torch.Tensor]:
    """Rotated activation rows -> (int8 [MP, K], fp32 scales [MP]), MP = next pow2 of M."""
    m, k = x2d.shape
    mp = _gemv_mp(m)
    x2d = x2d.contiguous()
    q = torch.empty(mp, k, device=x2d.device, dtype=torch.int8)
    s = torch.empty(mp, device=x2d.device, dtype=torch.float32)
    _, _, kernel = _get_kernels()
    block_k = min(2048, 1 << (k - 1).bit_length())
    kernel[(mp,)](x2d, q, s, m, k, QMAX=act_qmax, MP=mp, BLOCK_K=block_k, num_warps=8)
    return [q, s]


@_w4a8_act_quant_op.register_fake
def _w4a8_act_quant_fake(x2d, act_qmax):
    m, k = x2d.shape
    mp = _gemv_mp(m)
    return [torch.empty(mp, k, device=x2d.device, dtype=torch.int8), torch.empty(mp, device=x2d.device, dtype=torch.float32)]


@torch.library.custom_op("ostris::convrot_w4a8_gemv_prepared", mutates_args=())
def _w4a8_gemv_prepared_op(
    xq: torch.Tensor,
    xs: torch.Tensor,
    qdata: torch.Tensor,
    srel_u8: torch.Tensor,
    cb_u8: torch.Tensor,
    scales_u8: torch.Tensor,
    bias: Optional[torch.Tensor],
    m: int,
    out_dtype: str,
) -> torch.Tensor:
    """GEMV on already-quantized activation rows (xq [MP, K] int8, xs [MP] fp32)."""
    mp, k = xq.shape
    n = qdata.shape[0]
    out = torch.empty(mp, n, device=xq.device, dtype=getattr(torch, out_dtype))
    ws = scales_u8.view(torch.float32)
    _, kernel, _ = _get_kernels()
    # swept on RTX PRO 6000 (Blackwell) over the qwen3.8 shapes with HBM-resident weights:
    # 8-row tiles (more programs in flight for the N<=12k projections); the K tile and
    # warp count follow M to keep the [MP, BN, BK/2] accumulator in registers
    if mp <= 2:
        block_n, block_k, warps = 8, 512, 4
    elif mp <= 4:
        block_n, block_k, warps = 8, 256, 4
    else:
        block_n, block_k, warps = 16, 256, 8
    kernel[(-(-n // block_n),)](
        xq, xs, qdata, srel_u8, cb_u8.view(torch.float32), ws,
        bias if bias is not None else ws, out,
        k, n,
        MP=mp, HAS_BIAS=bias is not None, BLOCK_N=block_n, BLOCK_K=block_k, SAT=W4A8_SAT,
        num_warps=warps, num_stages=2,
    )
    return out[:m]


@_w4a8_gemv_prepared_op.register_fake
def _w4a8_gemv_prepared_fake(xq, xs, qdata, srel_u8, cb_u8, scales_u8, bias, m, out_dtype):
    return torch.empty(m, qdata.shape[0], device=xq.device, dtype=getattr(torch, out_dtype))


def _w4a8_gemv_op(x2d, qdata, srel_u8, cb_u8, scales_u8, bias, act_qmax, out_dtype):
    xq, xs = _w4a8_act_quant_op(x2d, act_qmax)
    return _w4a8_gemv_prepared_op(xq, xs, qdata, srel_u8, cb_u8, scales_u8, bias, x2d.shape[0], out_dtype)


# training-path linear: int8 tensor-core forward on the decoded grid (bit-identical to
# inference), straight-through backward on the dequantized weight. autograd keeps only
# the packed codes and decodes again in the backward, so no layer holds an unpacked
# int8 weight between forward and backward.
@torch.library.custom_op("ostris::convrot_w4a8_linear_ste", mutates_args=())
def _w4a8_linear_ste_op(
    x2d: torch.Tensor,
    qdata: torch.Tensor,
    srel_u8: torch.Tensor,
    cb_u8: torch.Tensor,
    scales_u8: torch.Tensor,
    bias: Optional[torch.Tensor],
    group_size: int,
    act_qmax: int,
    out_dtype: str,
) -> torch.Tensor:
    m = x2d.shape[0]
    w8 = _w4a8_unpack_op(qdata, srel_u8, cb_u8, group_size)
    aq, a_s = _int8_act_quant_padded(x2d, act_qmax)
    i32 = torch._int_mm(aq, w8.t())
    return _int8_epilogue(i32[:m], a_s[:m], scales_u8.view(torch.float32), bias, getattr(torch, out_dtype))


@_w4a8_linear_ste_op.register_fake
def _w4a8_linear_ste_fake(x2d, qdata, srel_u8, cb_u8, scales_u8, bias, group_size, act_qmax, out_dtype):
    return torch.empty(x2d.shape[0], qdata.shape[0], device=x2d.device, dtype=getattr(torch, out_dtype))


def _w4a8_linear_ste_setup(ctx, inputs, output):
    x2d, qdata, srel_u8, cb_u8, scales_u8, bias, group_size, act_qmax, out_dtype = inputs
    ctx.group_size = group_size
    ctx.save_for_backward(qdata, srel_u8, cb_u8, scales_u8)


def _w4a8_linear_ste_backward(ctx, grad):
    qdata, srel_u8, cb_u8, scales_u8 = ctx.saved_tensors
    w8 = _w4a8_unpack_op(qdata, srel_u8, cb_u8, ctx.group_size)
    w = w8.to(grad.dtype) * scales_u8.view(torch.float32).to(grad.dtype).unsqueeze(1)
    return grad @ w, None, None, None, None, None, None, None, None


_w4a8_linear_ste_op.register_autograd(_w4a8_linear_ste_backward, setup_context=_w4a8_linear_ste_setup)


# ---------------------------------------------------------------- the backend


class ConvRotW4A8Quantizer(ConvRotInt8Quantizer):
    """ConvRot W4A8 codebook backend: convrot8 numerics on an int4 codebook weight grid.
    One instance per qtype, shareable across modules."""

    wants_fp32_weight = True

    def __init__(self, rot_size: int = 256, group_size: int = 16):
        super().__init__(rot_size=rot_size)
        self.group_size = group_size

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        if module.in_features % self.group_size:
            return False
        return super().can_quantize(module)

    @staticmethod
    def attach_(
        module: torch.nn.Module,
        qdata: torch.Tensor,
        srel: torch.Tensor,
        s_channel: torch.Tensor,
        codebook: Optional[torch.Tensor],
        rot_size: int,
        group_size: int = 16,
    ) -> None:
        """Attach packed W4A8 tensors (comfy_kitchen storage) to a linear."""
        if codebook is None:
            # symmetric uniform codes: value = code - 8
            codebook = torch.arange(-8, 8, dtype=torch.float32)
        if srel.dtype != torch.float8_e4m3fn:
            srel = srel.view(torch.float8_e4m3fn) if srel.dtype == torch.uint8 else srel.to(torch.float8_e4m3fn)
        n, kh = qdata.shape
        if (n, kh * 2 // group_size) != tuple(srel.shape):
            raise ValueError(f"W4A8 scale shape {tuple(srel.shape)} does not match packed weight {tuple(qdata.shape)}")
        module.register_buffer("cw4_qdata", qdata.contiguous(), persistent=False)
        module.register_buffer("cw4_srel", srel.contiguous().view(torch.uint8), persistent=False)
        module.register_buffer(
            "cw4_scales", s_channel.detach().float().reshape(-1).contiguous().view(torch.uint8), persistent=False
        )
        module.register_buffer(
            "cw4_codebook", codebook.detach().float().reshape(-1).contiguous().view(torch.uint8), persistent=False
        )
        module.cw4_rot_size = int(rot_size)
        module.cw4_group_size = int(group_size)

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        rot = self._rot_for(module.in_features)
        packed, srel, sch, cb = quantize_w4a8_rotated(rotate(weight_fp32.float(), rot), self.group_size)
        self.attach_(module, packed, srel, sch, cb, rot, self.group_size)

    # storage accessors used by the parent's forward / dequantize paths
    def _rot(self, module) -> int:
        return module.cw4_rot_size

    def _qdata(self, module) -> torch.Tensor:
        return _w4a8_unpack_op(module.cw4_qdata, module.cw4_srel, module.cw4_codebook, module.cw4_group_size)

    def _scales_u8(self, module) -> torch.Tensor:
        return module.cw4_scales

    def _gemv_args(self, module):
        # the parent's packed gemv does not know the codebook grid; decode runs below
        return None

    def _codebook(self, module) -> torch.Tensor:
        return module.cw4_codebook.view(torch.float32)

    def _srel(self, module) -> torch.Tensor:
        return module.cw4_srel.view(torch.float8_e4m3fn)

    def _linear_ste(self, module, x2d: torch.Tensor, out_dtype: str) -> torch.Tensor:
        return _w4a8_linear_ste_op(
            x2d, module.cw4_qdata, module.cw4_srel, module.cw4_codebook, module.cw4_scales,
            module.bias, module.cw4_group_size, self.act_qmax, out_dtype,
        )

    def fake_quant_rotated_weight(self, module, w_rot: torch.Tensor) -> torch.Tensor:
        """dequant(quant(w_rot)) on the STORED group/channel scales and codebook. fp32."""
        n, k = w_rot.shape
        g = module.cw4_group_size
        sch = self._scales(module)
        levels = grid_levels(self._codebook(module), self._srel(module))
        codes = _assign_grid(w_rot.float().view(n, k // g, g) / sch.view(-1, 1, 1), levels)
        return (torch.gather(levels, 2, codes).view(n, k) * sch.unsqueeze(1))

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.cw4_qdata.device, dtype=torch.float32)
        packed, srel, sch, cb = quantize_w4a8_rotated(
            rotate(w, module.cw4_rot_size), module.cw4_group_size, self._codebook(module)
        )
        # in place: fused views over several modules' storage stay coherent
        module.cw4_qdata.copy_(packed)
        module.cw4_srel.copy_(srel.view(torch.uint8))
        module.cw4_scales.copy_(sch.view(torch.uint8))

    @torch.no_grad()
    def requantize_codes_(self, module, fp_weight: torch.Tensor, stochastic: bool = False) -> None:
        """Re-assign only the codes on the module's STORED scales (the grid a QAT run
        trains against). stochastic: unbiased rounding between the bracketing levels."""
        w = fp_weight.to(device=module.cw4_qdata.device, dtype=torch.float32)
        n, k = w.shape
        g = module.cw4_group_size
        w_rot = rotate(w, module.cw4_rot_size)
        levels = grid_levels(self._codebook(module), self._srel(module))
        codes = _assign_grid(w_rot.view(n, k // g, g) / self._scales(module).view(-1, 1, 1), levels, stochastic)
        module.cw4_qdata.copy_(_pack_codes(codes.view(n, k)))

    def gemv_ok(self, module, x: torch.Tensor) -> bool:
        """Whether the fused decode GEMV handles this (module, input)."""
        return (
            not x.requires_grad
            and x.numel() // module.in_features <= W4A8_GEMV_MAX_M
            and x.is_cuda
            and module.cw4_group_size == 16
            and module.in_features % 16 == 0
            and _triton_available()
        )

    def prepare_act(self, module, x: torch.Tensor):
        """Rotate + int8-quantize the activation rows once; shareable by every W4A8
        linear that consumes the same input (same in_features / rotation)."""
        return _w4a8_act_quant_op(rotate(x, module.cw4_rot_size).reshape(-1, module.in_features), self.act_qmax)

    def gemv_prepared(self, module, xq: torch.Tensor, xs: torch.Tensor, shape, dtype: torch.dtype) -> torch.Tensor:
        m = 1
        for d in shape[:-1]:
            m *= d
        out = _w4a8_gemv_prepared_op(
            xq, xs, module.cw4_qdata, module.cw4_srel, module.cw4_codebook, module.cw4_scales,
            module.bias, m, str(dtype).split(".")[-1],
        )
        return out.reshape(*shape[:-1], module.out_features)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        if self.gemv_ok(module, x):
            xq, xs = self.prepare_act(module, x)
            return self.gemv_prepared(module, xq, xs, x.shape, x.dtype)
        return super().forward(module, x)


def module_is_plain(mod) -> bool:
    """True when calling mod(x) is just its class forward: no instance-level forward
    (training LoRA wrapper) and no forward hooks (inference LoRA sidechains)."""
    return "forward" not in mod.__dict__ and not mod._forward_hooks and not mod._forward_pre_hooks


def module_hooks_only(mod) -> bool:
    """Class forward plus plain forward hooks (what inference LoRA sidechains register):
    the output can be computed by a fast path and the hooks applied by apply_forward_hooks."""
    from torch.nn.modules import module as _m

    return (
        "forward" not in mod.__dict__
        and not mod._forward_pre_hooks
        and not mod._forward_hooks_with_kwargs
        and not _m._global_forward_hooks
        and not _m._global_forward_pre_hooks
    )


def apply_forward_hooks(mod, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Run mod's forward hooks on an output computed outside mod.__call__."""
    for hook in mod._forward_hooks.values():
        r = hook(mod, (x,), out)
        if r is not None:
            out = r
    return out


def w4a8_grouped_linear(modules, x: torch.Tensor):
    """Outputs of several linears fed the same input. W4A8 OstrisLinears that share the
    rotation and are not wrapped (e.g. by an active LoRA) share one rotate+quant of x on
    the decode path; everything else just runs module(x)."""
    outs = [None] * len(modules)
    shared = None
    for i, mod in enumerate(modules):
        q = getattr(mod, "ostris_quantizer", None)
        if (
            isinstance(q, ConvRotW4A8Quantizer)
            and module_hooks_only(mod)
            and q.gemv_ok(mod, x)
        ):
            if shared is None:
                shared = (mod.cw4_rot_size, mod.in_features, q.prepare_act(mod, x))
            if shared[0] == mod.cw4_rot_size and shared[1] == mod.in_features:
                out = q.gemv_prepared(mod, *shared[2], x.shape, x.dtype)
                outs[i] = apply_forward_hooks(mod, x, out) if mod._forward_hooks else out
                continue
        outs[i] = mod(x)
    return outs


def dequantize_w4a8(qdata, srel, s_channel, codebook, rot_size: int, group_size: int = 16, dtype=torch.float32):
    """Full weight in the original basis from comfy_kitchen W4A8 tensors (reference path)."""
    w8 = unpack_w4a8_eager(qdata, srel, codebook, group_size)
    w_rot = w8.float() * s_channel.float().to(w8.device).unsqueeze(1)
    return rotate(w_rot, rot_size).to(dtype)
