"""
ConvRot quantization backends (convrot4 / convrot8 qtypes).

convrot4 is the paper's W4A4 NVFP4 method described below. convrot8 pairs the same
rotation with per-token / per-output-channel symmetric int8 (W8A8) and
torch._int_mm: near-lossless (~1% weight error), and the fast path runs on any int8
tensor-core gpu (Ampere+), not just Blackwell. The rotation is what makes the coarse
per-row scales safe — it spreads outliers so a whole row shares one scale without
clipping damage (the classic SmoothQuant failure mode).

Implements "ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion
Transformers" (arXiv:2512.03673) as an OstrisQuantizer backend, self-contained on
top of torch (no torchao version requirements).

Method: weights and activations are rotated with a block *regular* Hadamard
transform (R4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]/2 Kronecker-powered
to rot_size, a power of 4, default 256). Unlike the standard Hadamard whose all-ones
row concentrates the block mean into one coordinate, the regular Hadamard has
constant row sums, smoothing row-wise and column-wise outliers symmetrically. The
rotation is folded into the weight offline and applied to the activation at runtime,
so it cancels in the matmul. Both sides are then quantized to NVFP4 (fp4 e2m1 values,
fp8 e4m3 scale per 16 elements, plus one fp32 per-tensor scale) and multiplied with
the Blackwell fp4 tensor cores via torch._scaled_mm — a real ~5-6x gemm speedup, ~2x
at the layer level after rotation + activation-quant overhead.

Paths:
  - inference (no grad): rotate -> fused triton nvfp4 activation quant ->
    hardware fp4 gemm. Requires sm_100+ (Blackwell); otherwise falls back to the
    dequantized matmul below.
  - training (grad enabled): rotate -> straight-through fake-quant of the
    activation (so adapters train against the same W4A4 numerics that deployment
    uses) -> bf16 matmul with the dequantized rotated weight. Fully differentiable
    w.r.t. the input.

Everything is deterministic: the rotation is a fixed matrix (no randomness at all)
and quantization is pure rounding.

Quantized state attached to each module:
  cr_qdata    packed e2m1 codes (uint8, out x in/2; low nibble = even element)
  cr_scales   e4m3 block scales (out x in/16)
  cr_scales_blocked  the same scales pre-swizzled for torch._scaled_mm
  cr_pts      fp32 per-tensor scale (scalar)
  cr_rot / module.cr_rot_size  rotation block size
"""

from typing import Optional

import torch
import torch.nn.functional as F

from toolkit.print import print_acc
from toolkit.util.ostris_quant import OstrisQuantizer

CONVROT_QTYPES = ("convrot4", "convrot8", "convrotbitnet", "convrotcomfyw4a4") + tuple(
    f"convrotint{b}" for b in range(2, 9)
)


def get_convrot_quantizer(qtype: str):
    if qtype == "convrot4":
        return ConvRotQuantizer(rot_size=256)
    if qtype == "convrot8":
        return ConvRotInt8Quantizer(rot_size=256)
    if qtype == "convrotbitnet":
        return ConvRotBitNetQuantizer(rot_size=256)
    if qtype == "convrotcomfyw4a4":
        return ConvRotComfyW4A4Quantizer()
    if qtype.startswith("convrotint"):
        bits = int(qtype[len("convrotint"):])
        if 2 <= bits <= 8:
            return ConvRotIntNQuantizer(bits, rot_size=256)
    return None


F4_MAX = 6.0
F8_E4M3_MAX = 448.0
BLOCK = 16  # nvfp4 scale block

_E2M1_EDGES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

_hadamard_cache = {}
_edges_cache = {}
_skip_warned = set()


def _cached(cache, key, build):
    if key not in cache:
        cache[key] = build()
    return cache[key]


def regular_hadamard(rot_size: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """The ConvRot rotation: Kronecker powers of the 4x4 regular Hadamard matrix,
    orthonormal. Symmetric and orthogonal, so it is its own inverse."""
    key = (rot_size, str(device), dtype)

    def build():
        r4 = torch.tensor(
            [[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float64,
        )
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise ValueError(f"rot_size {rot_size} is not a power of 4")
        return (h / rot_size**0.5).to(device=device, dtype=dtype)

    return _cached(_hadamard_cache, key, build)


def largest_pow4_divisor(d: int) -> int:
    h = 1
    while d % (h * 4) == 0:
        h *= 4
    return h


def rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply the block regular-Hadamard rotation along the last dim (self-inverse)."""
    if rot_size == 1:
        return x
    h = regular_hadamard(rot_size, x.device, x.dtype)
    shape = x.shape
    xb = x.reshape(-1, shape[-1] // rot_size, rot_size)
    return torch.matmul(xb, h).reshape(shape)


def to_blocked(m: torch.Tensor) -> torch.Tensor:
    """Rearrange an (R, C) scale matrix into the swizzled layout torch._scaled_mm
    expects for block-scaled fp4 (cublas 128x4-tile layout)."""
    rows, cols = m.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    if (rows, cols) != (rb * 128, cb * 4):
        padded = torch.zeros(rb * 128, cb * 4, device=m.device, dtype=m.dtype)
        padded[:rows, :cols] = m
        m = padded
    blocks = m.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _optimal_nvfp4_scales(
    xb: torch.Tensor, base: torch.Tensor, pts: torch.Tensor
) -> torch.Tensor:
    """MSE-optimal e4m3 block scales: sweep fractions of the amax-derived scale
    (each snapped to e4m3) and keep the per-block argmin of the e2m1
    reconstruction error. ~11% lower weight error than plain amax scaling;
    deterministic, and the storage/GEMM format is unchanged."""
    edges = _cached(
        _edges_cache, str(xb.device), lambda: torch.tensor(_E2M1_EDGES, device=xb.device)
    )
    vals = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=xb.device
    )
    best_s = base.to(torch.float8_e4m3fn)
    best_e = None
    for frac in torch.linspace(0.70, 1.10, 9, dtype=torch.float64):
        s8 = (base * float(frac)).to(torch.float8_e4m3fn)
        denom = (s8.float() * pts).unsqueeze(-1)
        safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        z = (xb / safe).clamp(-F4_MAX, F4_MAX)
        recon = vals[torch.bucketize(z.abs(), edges)] * torch.sign(z) * safe
        e = (xb - recon).square_().sum(-1)
        if best_e is None:
            best_s, best_e = s8, e
        else:
            better = e < best_e
            best_s = torch.where(better, s8, best_s)
            best_e = torch.where(better, e, best_e)
    return best_s


def quantize_nvfp4(
    x: torch.Tensor,
    pts: Optional[torch.Tensor] = None,
    optimize_scales: bool = False,
):
    """Quantize (rows, K) to nvfp4. Returns (packed uint8 (rows, K/2),
    e4m3 scales (rows, K/16), fp32 per-tensor scale). optimize_scales runs the
    MSE-optimal block-scale sweep — weights only; activations stay amax (the
    sweep costs ~9 extra passes)."""
    rows, K = x.shape
    xf = x.float()
    if pts is None:
        pts = xf.abs().amax() / (F4_MAX * F8_E4M3_MAX)
        pts = torch.where(pts > 0, pts, torch.ones_like(pts))
    xb = xf.view(rows, K // BLOCK, BLOCK)
    base = xb.abs().amax(dim=-1) / (F4_MAX * pts)
    if optimize_scales:
        scales = _optimal_nvfp4_scales(xb, base, pts)
    else:
        scales = base.to(torch.float8_e4m3fn)
    denom = (scales.float() * pts).unsqueeze(-1)
    z = (xb / torch.where(denom > 0, denom, torch.ones_like(denom))).clamp(
        -F4_MAX, F4_MAX
    )
    edges = _cached(
        _edges_cache, str(x.device), lambda: torch.tensor(_E2M1_EDGES, device=x.device)
    )
    mag = torch.bucketize(z.abs(), edges).to(torch.uint8)
    codes = (mag | ((z < 0).to(torch.uint8) << 3)).view(rows, K)
    packed = ((codes[:, 1::2] << 4) | codes[:, ::2]).contiguous()
    return packed, scales, pts


def dequantize_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    pts: torch.Tensor,
    rows: int,
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # single-pass triton path when available: the torch chain below is ~7 full-size
    # elementwise passes with fp32 intermediates, which made every convrot4 training
    # backward pay a dequant cost comparable to the gradient matmul itself
    if _triton_available() and packed.is_cuda and dtype in (torch.bfloat16, torch.float16, torch.float32):
        return _fp4_dequant_op(
            packed, scales.view(torch.uint8), pts.reshape(1).view(torch.uint8),
            str(dtype).split(".")[-1],
        )
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).view(rows, K)
    # the lookup table is built inline (NOT module-cached): this function runs inside
    # custom-op backwards, which torch.compile traces with fake tensors where a
    # pre-existing real tensor is illegal; an in-trace constructed constant is fine
    vals = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device
    )
    mag = torch.index_select(vals, 0, (codes & 7).flatten().to(torch.int32)).view(rows, K)
    v = mag * torch.where((codes & 8) > 0, -1.0, 1.0)
    v = v.view(rows, K // BLOCK, BLOCK) * (scales.float() * pts).unsqueeze(-1)
    return v.view(rows, K).to(dtype)


# ---------------- fused triton activation quant ----------------

_triton_ok = None


def _triton_available() -> bool:
    global _triton_ok
    if _triton_ok is None:
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401

            _triton_ok = True
        except Exception:
            _triton_ok = False
            print_acc(
                "ConvRot: triton is not available. The fused activation-quant kernel is "
                "disabled and activations will be quantized with plain torch ops instead "
                "— inference gets slower (most of the fp4 speedup is lost), but quality "
                "and training are unaffected."
            )
    return _triton_ok


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def nvfp4_act_quant_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        pts_ptr,
        K,
        n_col_tiles,
        BLOCK_K: tl.constexpr,
        BLOCKED_SCALES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        pts = tl.load(pts_ptr)
        offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + pid_m * K + offs, mask=mask, other=0.0).to(tl.float32)
        xb = tl.reshape(x, (BLOCK_K // 16, 16))
        amax = tl.max(tl.abs(xb), axis=1)
        # clamp to the e4m3 max so an oversized block can't overflow the fp8
        # conversion (matters now that activations use a fixed pts=1)
        scale8 = tl.minimum(amax / (6.0 * pts), 448.0).to(tl.float8e4nv)
        denom = scale8.to(tl.float32) * pts
        denom = tl.where(denom > 0, denom, 1.0)
        # note: triton fp32 division on this backend is ~1ulp off ieee (even with
        # tl.fdiv ieee_rounding=True), so values landing exactly on a code boundary
        # can round to the adjacent code vs the torch path. ties are equidistant, so
        # this changes nothing quantitatively; activation codes are transient (never
        # stored), and the kernel itself is deterministic.
        z = xb / denom[:, None]
        z = tl.minimum(tl.maximum(z, -6.0), 6.0)
        az = tl.abs(z)
        # strict > so exact midpoints go to the lower code, matching torch.bucketize
        code = (
            (az > 0.25).to(tl.uint8)
            + (az > 0.75).to(tl.uint8)
            + (az > 1.25).to(tl.uint8)
            + (az > 1.75).to(tl.uint8)
            + (az > 2.5).to(tl.uint8)
            + (az > 3.5).to(tl.uint8)
            + (az > 5.0).to(tl.uint8)
        )
        code = code | ((z < 0).to(tl.uint8) << 3)
        lo, hi = tl.split(tl.reshape(code, (BLOCK_K // 2, 2)))
        byte = lo | (hi << 4)
        offs_b = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        tl.store(out_ptr + pid_m * (K // 2) + offs_b, byte, mask=offs_b < K // 2)
        s_idx = pid_k * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
        if BLOCKED_SCALES:
            # store straight into the cublas 128x4-tile swizzle (see to_blocked)
            r_t = pid_m // 128
            r_in = pid_m % 128
            c_t = s_idx // 4
            c = s_idx % 4
            offs_s = (
                ((r_t * n_col_tiles + c_t) * 32 + (r_in % 32)) * 16
                + (r_in // 32) * 4
                + c
            )
        else:
            offs_s = pid_m * (K // 16) + s_idx
        tl.store(scale_ptr + offs_s, scale8, mask=s_idx < K // 16)

    _kernel = nvfp4_act_quant_kernel
    return _kernel


def _launch_nvfp4_kernel(x, packed, scales, pts, blocked_scales: bool):
    rows, K = x.shape
    n_col_tiles = -(-(K // BLOCK) // 4)
    # triton block shapes must be powers of 2; loads/stores are masked on offs < K
    BLOCK_K = min(2048, 1 << (K - 1).bit_length())
    grid = (rows, -(-K // BLOCK_K))
    _get_kernel()[grid](
        x, packed, scales, pts,
        K, n_col_tiles,
        BLOCK_K=BLOCK_K, BLOCKED_SCALES=blocked_scales, num_warps=4,
    )


# registered as a custom op so torch.compile treats the triton launch as an opaque
# node with known output shapes; tracing raw JITFunction calls breaks inductor's
# autotune arg-cloning (seen on wan 2.2 under compile). the op also pads its output
# rows to a multiple of 16 for torch._scaled_mm (callers slice the mm result) so
# the compiled graph never contains a symbolic constant_pad_nd, which inductor
# mis-handles when fused with the op input.
@torch.library.custom_op("ostris::convrot_nvfp4_act_quant", mutates_args=())
def _nvfp4_act_quant_op(x: torch.Tensor) -> list[torch.Tensor]:
    rows, K = x.shape
    rows_pad = -(-rows // 16) * 16
    x = x.contiguous()
    # activations use a FIXED per-tensor scale of 1: their block scales fit the
    # e4m3 range natively (unlike tiny weight magnitudes, which keep dynamic pts),
    # measured quality-neutral on realistic distributions — and it removes a full
    # activation read (global amax) plus a device sync from every forward
    pts = torch.ones((), device=x.device)
    packed = torch.empty(rows_pad, K // 2, device=x.device, dtype=torch.uint8)
    if rows_pad != rows:
        packed[rows:].zero_()
    n_col_tiles = -(-(K // BLOCK) // 4)
    # zero-init: rows are padded to 128-tiles and the pad region must be zero
    scales = torch.zeros(
        (-(-rows_pad // 128)) * 128 * n_col_tiles * 4,
        device=x.device, dtype=torch.float8_e4m3fn,
    )
    _launch_nvfp4_kernel(x, packed, scales, pts, blocked_scales=True)
    return [packed, scales.view(torch.uint8), pts]


@_nvfp4_act_quant_op.register_fake
def _nvfp4_act_quant_fake(x):
    rows, K = x.shape
    rows_pad = -(-rows // 16) * 16
    n_col_tiles = -(-(K // BLOCK) // 4)
    return [
        torch.empty(rows_pad, K // 2, device=x.device, dtype=torch.uint8),
        torch.empty((-(-rows_pad // 128)) * 128 * n_col_tiles * 4, device=x.device, dtype=torch.uint8),
        torch.empty((), device=x.device, dtype=torch.float32),
    ]


def quantize_nvfp4_fused(x: torch.Tensor, blocked_scales: bool = False):
    """Triton path of quantize_nvfp4 for the inference hot loop: one read of x,
    writes packed codes + e4m3 scales (row-major, or directly in the swizzled
    layout torch._scaled_mm wants when blocked_scales=True). Falls back to the
    torch ops (row-major only)."""
    rows, K = x.shape
    if not (_triton_available() and x.is_cuda and K % 16 == 0):
        if blocked_scales:
            # match the custom op: rows padded to a multiple of 16 for _scaled_mm,
            # fixed pts=1 for activations (see _nvfp4_act_quant_op)
            rows_pad = -(-rows // 16) * 16
            if rows_pad != rows:
                x = F.pad(x, (0, 0, 0, rows_pad - rows))
            packed, scales, pts = quantize_nvfp4(x, pts=torch.ones((), device=x.device))
            return packed, to_blocked(scales), pts
        return quantize_nvfp4(x)
    if blocked_scales:
        packed, scales_u8, pts = _nvfp4_act_quant_op(x)
        return packed, scales_u8.view(torch.float8_e4m3fn), pts
    # row-major variant (used by tests/tools, not the compiled hot path)
    pts = x.float().abs().amax() / (F4_MAX * F8_E4M3_MAX)
    pts = torch.where(pts > 0, pts, torch.ones_like(pts))
    x = x.contiguous()
    packed = torch.empty(rows, K // 2, device=x.device, dtype=torch.uint8)
    scales = torch.empty(rows, K // BLOCK, device=x.device, dtype=torch.float8_e4m3fn)
    _launch_nvfp4_kernel(x, packed, scales, pts, blocked_scales=False)
    return packed, scales, pts



# ---------------- fp4 dequant kernel (backward hot path) ----------------

_dequant_kernel = None


def _get_dequant_kernel():
    global _dequant_kernel
    if _dequant_kernel is not None:
        return _dequant_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def nvfp4_dequant_kernel(
        q_ptr, s_ptr, pts_ptr, out_ptr, K,
        BLOCK_B: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_k = tl.program_id(1)
        pts = tl.load(pts_ptr)
        offs_b = pid_k * BLOCK_B + tl.arange(0, BLOCK_B)
        bmask = offs_b < K // 2
        byte = tl.load(q_ptr + row * (K // 2) + offs_b, mask=bmask, other=0)
        codes = tl.interleave(byte & 15, byte >> 4)  # (2*BLOCK_B,), column order
        m = (codes & 7).to(tl.float32)
        # arithmetic e2m1 decode ([0, .5, 1, 1.5, 2, 3, 4, 6]), exact
        mag = tl.where(m < 2, m * 0.5, tl.exp2(tl.floor(m / 2) - 1) * (1 + (m % 2) * 0.5))
        v = tl.where((codes & 8) > 0, -mag, mag)
        n_s: tl.constexpr = (2 * BLOCK_B) // 16
        offs_s = pid_k * n_s + tl.arange(0, n_s)
        s = tl.load(s_ptr + row * (K // 16) + offs_s, mask=offs_s < K // 16, other=0.0)
        vb = tl.reshape(v, (n_s, 16)) * (s.to(tl.float32) * pts)[:, None]
        out = tl.reshape(vb, (2 * BLOCK_B,))
        offs_v = pid_k * (2 * BLOCK_B) + tl.arange(0, 2 * BLOCK_B)
        tl.store(out_ptr + row * K + offs_v, out.to(out_ptr.dtype.element_ty), mask=offs_v < K)

    _dequant_kernel = nvfp4_dequant_kernel
    return _dequant_kernel


# custom op so the kernel stays opaque where it matters most: inside the fp4
# training op's registered backward, which torch.compile traces with fake tensors
@torch.library.custom_op("ostris::convrot_fp4_dequant", mutates_args=())
def _fp4_dequant_op(
    packed: torch.Tensor,
    scales_u8: torch.Tensor,
    pts_u8: torch.Tensor,
    out_dtype: str,
) -> torch.Tensor:
    rows, half = packed.shape
    out = torch.empty(rows, half * 2, device=packed.device, dtype=getattr(torch, out_dtype))
    kernel = _get_dequant_kernel()
    block_b = 1024
    grid = (rows, -(-half // block_b))
    kernel[grid](
        packed.contiguous(),
        scales_u8.view(torch.float8_e4m3fn),
        pts_u8.view(torch.float32),
        out, half * 2,
        BLOCK_B=block_b, num_warps=4,
    )
    return out


@_fp4_dequant_op.register_fake
def _fp4_dequant_fake(packed, scales_u8, pts_u8, out_dtype):
    rows, half = packed.shape
    return torch.empty(rows, half * 2, device=packed.device, dtype=getattr(torch, out_dtype))


# ---------------- backend ----------------


_warned_no_fp4 = False


def _fp4_gemm_supported(device) -> bool:
    global _warned_no_fp4
    device = torch.device(device)
    supported = (
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device)[0] >= 10  # Blackwell
    )
    if not supported and not _warned_no_fp4:
        _warned_no_fp4 = True
        print_acc(
            f"ConvRot: no fp4 tensor-core support on this device ({device}; needs an "
            "NVIDIA Blackwell GPU, sm_100+). Inference falls back to dequantized bf16 "
            "matmuls: correct output but NO speedup, and inference activations stay "
            "unquantized (W4A16 numerics instead of W4A4). The training path is "
            "unaffected (it always simulates W4A4 via fake-quant)."
        )
    return supported


# the fp4 training-path linear: forward VALUE is the real fp4 tensor-core gemm
# (bit-identical to the inference path), gradient is the straight-through estimate
# d y / d x_rot ~= dequant(W'). the backward re-dequantizes the weight from the fp4
# codes instead of saving a bf16 copy per layer (F.linear in the old fake-quant path
# retained a dequantized weight for every layer of the graph — a full bf16 model of
# extra train-step vram) and x is not saved at all.
@torch.library.custom_op("ostris::convrot_fp4_linear_ste", mutates_args=())
def _fp4_linear_ste_op(
    x2d: torch.Tensor,
    qdata: torch.Tensor,
    scales_u8: torch.Tensor,
    scales_blocked_u8: torch.Tensor,
    pts_u8: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: str,
) -> torch.Tensor:
    m = x2d.shape[0]
    aq, a_scales_blocked, a_pts = quantize_nvfp4_fused(x2d, blocked_scales=True)
    out = torch._scaled_mm(
        aq.view(torch.float4_e2m1fn_x2),
        qdata.view(torch.float4_e2m1fn_x2).t(),
        a_scales_blocked.view(torch.float8_e4m3fn),
        scales_blocked_u8.view(torch.float8_e4m3fn),
        out_dtype=getattr(torch, out_dtype),
    )
    if out.shape[0] != m:
        out = out[:m]
    s = (a_pts * pts_u8.view(torch.float32).reshape(())).to(out.dtype)
    if bias is not None:
        return torch.addcmul(bias, out, s)
    return out * s


@_fp4_linear_ste_op.register_fake
def _fp4_linear_ste_fake(x2d, qdata, scales_u8, scales_blocked_u8, pts_u8, bias, out_dtype):
    return torch.empty(
        x2d.shape[0], qdata.shape[0], device=x2d.device, dtype=getattr(torch, out_dtype)
    )


def _fp4_linear_ste_setup(ctx, inputs, output):
    x2d, qdata, scales_u8, scales_blocked_u8, pts_u8, bias, out_dtype = inputs
    ctx.save_for_backward(qdata, scales_u8, pts_u8)


def _fp4_linear_ste_backward(ctx, grad):
    qdata, scales_u8, pts_u8 = ctx.saved_tensors
    out_f, in_half = qdata.shape
    w = dequantize_nvfp4(
        qdata, scales_u8.view(torch.float8_e4m3fn),
        pts_u8.view(torch.float32).reshape(()),
        out_f, in_half * 2, grad.dtype,
    )
    return grad @ w, None, None, None, None, None, None


_fp4_linear_ste_op.register_autograd(
    _fp4_linear_ste_backward, setup_context=_fp4_linear_ste_setup
)


class ConvRotQuantizer(OstrisQuantizer):
    """ConvRot W4A4 backend. One instance per qtype, shareable across modules."""

    def __init__(self, rot_size: int = 256):
        self.rot_size = rot_size

    def _rot_for(self, d: int) -> int:
        return min(self.rot_size, largest_pow4_divisor(d))

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        d = module.in_features
        rot = self._rot_for(d)
        if d % BLOCK != 0 or module.out_features % BLOCK != 0 or rot < 16:
            if d not in _skip_warned:
                _skip_warned.add(d)
                print_acc(
                    f"ConvRot: skipping linears with in_features={d} "
                    f"(needs in/out divisible by 16 and a power-of-4 block >= 16)"
                )
            return False
        return True

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        rot = self._rot_for(module.in_features)
        w_rot = rotate(weight_fp32, rot)
        packed, scales, pts = quantize_nvfp4(w_rot, optimize_scales=True)
        # scales/pts are stored as uint8 byte views: nn.Module._apply dtype-casts
        # every floating buffer (module.to(dtype=...) would silently convert the
        # e4m3 scales to bf16 and fp32 pts to bf16, corrupting them). integer
        # buffers are only ever moved, never cast.
        module.register_buffer("cr_qdata", packed, persistent=False)
        module.register_buffer("cr_scales", scales.view(torch.uint8), persistent=False)
        module.register_buffer(
            "cr_scales_blocked", to_blocked(scales).view(torch.uint8), persistent=False
        )
        module.register_buffer(
            "cr_pts",
            pts.detach().clone().reshape(1).view(torch.uint8),
            persistent=False,
        )
        module.cr_rot_size = rot

    @staticmethod
    def _pts(module) -> torch.Tensor:
        return module.cr_pts.view(torch.float32).reshape(())

    def _rot(self, module) -> int:
        return module.cr_rot_size

    def fake_quant_rotated_weight(self, module, w_rot: torch.Tensor) -> torch.Tensor:
        """dequant(quant(w_rot)) on the deployed e2m1 grid with the module's STORED
        e4m3 block scales (no scale re-optimization) — the value half of the QAT
        straight-through estimator. Returns float32."""
        rows, K = w_rot.shape
        s = module.cr_scales.view(torch.float8_e4m3fn).float() * self._pts(module)
        denom = s.unsqueeze(-1)
        safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        z = (w_rot.float().view(rows, K // BLOCK, BLOCK) / safe).clamp(-F4_MAX, F4_MAX)
        edges = _cached(
            _edges_cache, str(w_rot.device),
            lambda: torch.tensor(_E2M1_EDGES, device=w_rot.device),
        )
        vals = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w_rot.device
        )
        recon = vals[torch.bucketize(z.abs(), edges)] * torch.sign(z) * safe
        return recon.view(rows, K)

    def _dequantize_rotated(self, module, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_nvfp4(
            module.cr_qdata,
            module.cr_scales.view(torch.float8_e4m3fn),
            self._pts(module),
            module.out_features,
            module.in_features,
            dtype,
        )

    def dequantize(self, module) -> torch.Tensor:
        w = self._dequantize_rotated(module, torch.float32)
        return rotate(w, module.cr_rot_size)  # self-inverse

    @torch.no_grad()
    def requantize_codes_(self, module, fp_weight: torch.Tensor) -> None:
        """Re-quantize only the codes on the module's STORED scales — the grid a
        QAT run trains against. Re-optimizing the scales here would re-grid every
        weight and destroy the code adjustments training made."""
        w = fp_weight.to(device=module.cr_qdata.device, dtype=torch.float32)
        w_rot = rotate(w, module.cr_rot_size)
        rows, K = w_rot.shape
        s = module.cr_scales.view(torch.float8_e4m3fn).float() * self._pts(module)
        denom = s.unsqueeze(-1)
        safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        z = (w_rot.view(rows, K // BLOCK, BLOCK) / safe).clamp(-F4_MAX, F4_MAX)
        edges = _cached(
            _edges_cache, str(w_rot.device),
            lambda: torch.tensor(_E2M1_EDGES, device=w_rot.device),
        )
        z = z.reshape(rows, K)
        mag = torch.bucketize(z.abs(), edges).to(torch.uint8)
        codes = mag | ((z < 0).to(torch.uint8) << 3)
        module.cr_qdata = ((codes[:, 1::2] << 4) | codes[:, ::2]).contiguous()

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.cr_qdata.device, dtype=torch.float32)
        w_rot = rotate(w, module.cr_rot_size)
        packed, scales, pts = quantize_nvfp4(w_rot, optimize_scales=True)
        module.cr_qdata = packed
        module.cr_scales = scales.view(torch.uint8)
        module.cr_scales_blocked = to_blocked(scales).view(torch.uint8)
        module.cr_pts = pts.detach().clone().reshape(1).view(torch.uint8)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        rot = module.cr_rot_size
        in_f, out_f = module.in_features, module.out_features
        m = x.numel() // in_f

        if x.requires_grad:
            # training path, gated on requires_grad alone (not is_grad_enabled) so
            # both passes of gradient checkpointing take the same branch
            if _fp4_gemm_supported(x.device):
                # fp4 tensor-core forward (bit-identical to the inference path)
                # with a straight-through analytic backward
                x2d = rotate(x, rot).reshape(-1, in_f)
                out = _fp4_linear_ste_op(
                    x2d, module.cr_qdata, module.cr_scales,
                    module.cr_scales_blocked, module.cr_pts, module.bias,
                    str(x.dtype).split(".")[-1],
                )
                return out.reshape(*x.shape[:-1], out_f)
            # no fp4 hardware: straight-through fake-quant of the activation and a
            # differentiable bf16 matmul against the dequantized rotated weight
            x2d = rotate(x, rot).reshape(-1, in_f)
            with torch.no_grad():
                aq, a_scales, a_pts = quantize_nvfp4(x2d.detach())
                x_dq = dequantize_nvfp4(aq, a_scales, a_pts, m, in_f, x.dtype)
                w = self._dequantize_rotated(module, x.dtype)
            x_ste = x2d + (x_dq - x2d).detach()
            out = F.linear(x_ste, w, module.bias)
            return out.reshape(*x.shape[:-1], out_f)

        if _fp4_gemm_supported(x.device):
            # row padding for _scaled_mm happens inside the act-quant op (compile
            # safety); slice the mm output back to m rows (a contiguous prefix).
            # NOTE: no fused rotate+quant here (unlike convrot8): the e2m1 packing
            # needs tl.reshape/tl.split of dot-derived tensors, which this triton
            # backend miscompiles when the dot sits in/after a loop (element order
            # scrambles). the int8 kernel avoids those primitives and fuses safely.
            aq, a_scales_blocked, a_pts = quantize_nvfp4_fused(
                rotate(x, rot).reshape(-1, in_f), blocked_scales=True
            )
            out = torch._scaled_mm(
                aq.view(torch.float4_e2m1fn_x2),
                module.cr_qdata.view(torch.float4_e2m1fn_x2).t(),
                a_scales_blocked.view(torch.float8_e4m3fn),
                module.cr_scales_blocked.view(torch.float8_e4m3fn),
                out_dtype=x.dtype,
            )
            if out.shape[0] != m:
                out = out[:m]
            s = (a_pts * self._pts(module)).to(x.dtype)
            if module.bias is not None:
                out = torch.addcmul(module.bias, out, s)
            else:
                out = out * s
            return out.reshape(*x.shape[:-1], out_f)

        # no fp4 hardware: dequantized matmul (correct, no speedup)
        w = self._dequantize_rotated(module, x.dtype)
        out = F.linear(rotate(x, rot).reshape(-1, in_f), w, module.bias)
        return out.reshape(*x.shape[:-1], out_f)


# ---------------- convrot8: W8A8 int8 backend ----------------


def quantize_int8_rows(x: torch.Tensor, qmax: int = 127):
    """Symmetric per-row integer quantization to [-qmax, qmax] (int8 storage).
    Returns (int8 (rows, K), fp32 scales (rows,))."""
    xf = x.float()
    scales = xf.abs().amax(dim=1) / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    q = torch.round(xf / scales.unsqueeze(1)).clamp_(-qmax, qmax).to(torch.int8)
    return q, scales


_int8_kernels = None


def _get_int8_kernels():
    global _int8_kernels
    if _int8_kernels is not None:
        return _int8_kernels
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def int8_act_quant_kernel(
        x_ptr, q_ptr, s_ptr, K, QMAX: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        row = tl.program_id(0)
        base = row * K
        acc = tl.zeros((BLOCK_K,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            v = tl.load(x_ptr + base + offs, mask=offs < K, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(v))
        amax = tl.max(acc, axis=0)
        scale = tl.where(amax > 0, amax / QMAX, 1.0)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            # rint = round-half-to-even, matching torch.round in the reference path
            q = libdevice.rint(v / scale)
            q = tl.minimum(tl.maximum(q, -1.0 * QMAX), 1.0 * QMAX)
            tl.store(q_ptr + base + offs, q.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def int8_epilogue_kernel(
        i_ptr,
        as_ptr,
        ws_ptr,
        b_ptr,
        o_ptr,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cb = tl.program_id(1)
        offs = cb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        acc = tl.load(i_ptr + row * N + offs, mask=mask, other=0).to(tl.float32)
        a_s = tl.load(as_ptr + row)
        w_s = tl.load(ws_ptr + offs, mask=mask, other=0.0)
        out = acc * (a_s * w_s)
        if HAS_BIAS:
            out += tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * N + offs, out.to(o_ptr.dtype.element_ty), mask=mask)

    _int8_kernels = (int8_act_quant_kernel, int8_epilogue_kernel)
    return _int8_kernels


# registered as custom ops so torch.compile treats the triton launches as opaque
# nodes with known output shapes (see _nvfp4_act_quant_op). rows are padded to a
# multiple of 32 inside the op for torch._int_mm; callers slice the mm output.
@torch.library.custom_op("ostris::convrot_int8_act_quant", mutates_args=())
def _int8_act_quant_op(x: torch.Tensor, qmax: int) -> list[torch.Tensor]:
    rows, K = x.shape
    rows_pad = -(-rows // 32) * 32
    x = x.contiguous()
    q = torch.empty(rows_pad, K, device=x.device, dtype=torch.int8)
    scales = torch.empty(rows_pad, device=x.device, dtype=torch.float32)
    if rows_pad != rows:
        q[rows:].zero_()
        scales[rows:].fill_(1.0)
    kernel, _ = _get_int8_kernels()
    # triton block shapes must be powers of 2; loads/stores are masked on offs < K
    block_k = min(2048, 1 << (K - 1).bit_length())
    kernel[(rows,)](
        x, q, scales, K, QMAX=qmax, BLOCK_K=block_k, num_warps=8
    )
    return [q, scales]


@_int8_act_quant_op.register_fake
def _int8_act_quant_fake(x, qmax):
    rows, K = x.shape
    rows_pad = -(-rows // 32) * 32
    return [
        torch.empty(rows_pad, K, device=x.device, dtype=torch.int8),
        torch.empty(rows_pad, device=x.device, dtype=torch.float32),
    ]


@torch.library.custom_op("ostris::convrot_int8_epilogue", mutates_args=())
def _int8_epilogue_op(
    i32: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: str,
) -> torch.Tensor:
    m, n = i32.shape
    out = torch.empty(m, n, device=i32.device, dtype=getattr(torch, out_dtype))
    _, kernel = _get_int8_kernels()
    grid = (m, -(-n // 1024))
    kernel[grid](
        i32,
        a_scales,
        w_scales,
        bias if bias is not None else a_scales,
        out,
        n,
        HAS_BIAS=bias is not None,
        BLOCK_N=1024,
        num_warps=4,
    )
    return out


@_int8_epilogue_op.register_fake
def _int8_epilogue_fake(i32, a_scales, w_scales, bias, out_dtype):
    m, n = i32.shape
    return torch.empty(m, n, device=i32.device, dtype=getattr(torch, out_dtype))


def quantize_int8_rows_fused(x: torch.Tensor, qmax: int = 127):
    """Triton path of quantize_int8_rows: one extra read of x instead of the
    multi-kernel torch chain. Falls back to the torch ops."""
    if not (_triton_available() and x.is_cuda):
        return quantize_int8_rows(x, qmax)
    q, scales = _int8_act_quant_op(x, qmax)
    rows = x.shape[0]
    return q[:rows], scales[:rows]


def _int8_act_quant_padded(x: torch.Tensor, qmax: int = 127):
    """Act quant with rows padded to a multiple of 32 for torch._int_mm."""
    if _triton_available() and x.is_cuda:
        q, scales = _int8_act_quant_op(x, qmax)
        return q, scales
    q, scales = quantize_int8_rows(x, qmax)
    rows = q.shape[0]
    rows_pad = -(-rows // 32) * 32
    if rows_pad != rows:
        q = F.pad(q, (0, 0, 0, rows_pad - rows))
        scales = F.pad(scales, (0, rows_pad - rows), value=1.0)
    return q, scales


# the training-path linear: forward VALUE is the real int8 tensor-core gemm (bit
# identical to the inference path), gradient is the straight-through estimate
# d y / d x_rot ~= dequant(W'), registered as a custom-op autograd so it works
# under torch.compile. the backward re-dequantizes the weight from int8 instead of
# saving a bf16 copy, and x is not saved at all — less memory than F.linear.
@torch.library.custom_op("ostris::convrot_int8_linear_ste", mutates_args=())
def _int8_linear_ste_op(
    x2d: torch.Tensor,
    qdata: torch.Tensor,
    w_scales_u8: torch.Tensor,
    bias: Optional[torch.Tensor],
    act_qmax: int,
    out_dtype: str,
) -> torch.Tensor:
    m = x2d.shape[0]
    aq, a_s = _int8_act_quant_padded(x2d, act_qmax)
    i32 = torch._int_mm(aq, qdata.t())
    return _int8_epilogue(
        i32[:m], a_s[:m], w_scales_u8.view(torch.float32), bias,
        getattr(torch, out_dtype),
    )


@_int8_linear_ste_op.register_fake
def _int8_linear_ste_fake(x2d, qdata, w_scales_u8, bias, act_qmax, out_dtype):
    return torch.empty(
        x2d.shape[0], qdata.shape[0], device=x2d.device, dtype=getattr(torch, out_dtype)
    )


def _int8_linear_ste_setup(ctx, inputs, output):
    x2d, qdata, w_scales_u8, bias, act_qmax, out_dtype = inputs
    ctx.save_for_backward(qdata, w_scales_u8)


def _int8_linear_ste_backward(ctx, grad):
    qdata, w_scales_u8 = ctx.saved_tensors
    w_scales = w_scales_u8.view(torch.float32).to(grad.dtype)
    w = qdata.to(grad.dtype) * w_scales.unsqueeze(1)
    return grad @ w, None, None, None, None, None


_int8_linear_ste_op.register_autograd(
    _int8_linear_ste_backward, setup_context=_int8_linear_ste_setup
)


def _int8_epilogue(
    i32: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    bias,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """out = i32 * a_scales[:, None] * w_scales[None, :] (+ bias), in out_dtype."""
    if _triton_available() and i32.is_cuda:
        return _int8_epilogue_op(
            i32, a_scales, w_scales, bias, str(out_dtype).split(".")[-1]
        )
    out = i32.float() * w_scales
    out = out * a_scales.unsqueeze(1)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


_int8_mm_ok = None


def _int8_gemm_supported(device) -> bool:
    global _int8_mm_ok
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        supported = False
    else:
        if _int8_mm_ok is None:
            try:
                a = torch.zeros(32, 64, dtype=torch.int8, device=device)
                b = torch.zeros(64, 32, dtype=torch.int8, device=device)
                torch._int_mm(a, b)
                _int8_mm_ok = True
            except Exception:
                _int8_mm_ok = False
        supported = _int8_mm_ok
    global _warned_no_int8
    if not supported and not _warned_no_int8:
        _warned_no_int8 = True
        print_acc(
            f"ConvRot: int8 matmul (torch._int_mm) is not usable on this device "
            f"({device}). Inference falls back to dequantized bf16 matmuls: correct "
            "output but NO speedup, and inference activations stay unquantized "
            "(W8A16 numerics instead of W8A8). The training path is unaffected "
            "(it always simulates W8A8 via fake-quant)."
        )
    return supported


_warned_no_int8 = False


class ConvRotInt8Quantizer(OstrisQuantizer):
    """ConvRot W8A8 backend: shared regular-Hadamard rotation + per-token /
    per-output-channel symmetric int8 with torch._int_mm. One instance per qtype,
    shareable across modules."""

    # activation quantization range (per-token symmetric [-act_qmax, act_qmax]);
    # the comfy w4a4 subclass narrows this to 7
    act_qmax = 127

    def __init__(self, rot_size: int = 256):
        self.rot_size = rot_size

    def _rot_for(self, d: int) -> int:
        return min(self.rot_size, largest_pow4_divisor(d))

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        d = module.in_features
        if d % BLOCK != 0 or module.out_features % 8 != 0 or self._rot_for(d) < 16:
            if d not in _skip_warned:
                _skip_warned.add(d)
                print_acc(
                    f"ConvRot: skipping linears with in_features={d} "
                    f"(needs in divisible by 16, out by 8, and a power-of-4 block >= 16)"
                )
            return False
        return True

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        rot = self._rot_for(module.in_features)
        q, scales = quantize_int8_rows(rotate(weight_fp32, rot))
        module.register_buffer("cr8_qdata", q, persistent=False)
        # fp32 scales stored as a uint8 byte view (see convrot4: module.to(dtype=...)
        # would otherwise cast them)
        module.register_buffer("cr8_scales", scales.view(torch.uint8), persistent=False)
        module.cr8_rot_size = rot

    # storage accessors: ConvRotIntNQuantizer overrides these to unpack its
    # bitpacked codes; everything below runs off them unchanged
    def _rot(self, module) -> int:
        return module.cr8_rot_size

    def _qdata(self, module) -> torch.Tensor:
        """The int8 (out, in) weight codes in the rotated basis."""
        return module.cr8_qdata

    def _scales_u8(self, module) -> torch.Tensor:
        return module.cr8_scales

    def _scales(self, module) -> torch.Tensor:
        return self._scales_u8(module).view(torch.float32)

    def _linear_ste(self, module, x2d: torch.Tensor, out_dtype: str) -> torch.Tensor:
        """Hardware STE linear for the training path. For int8 the saved qdata is
        the resident buffer itself, so autograd holds only a free reference."""
        return _int8_linear_ste_op(
            x2d, self._qdata(module), self._scales_u8(module), module.bias,
            self.act_qmax, out_dtype,
        )

    def fake_quant_rotated_weight(self, module, w_rot: torch.Tensor) -> torch.Tensor:
        """dequant(quant(w_rot)) on the deployed int8 grid with the module's STORED
        per-row scales — the value half of the QAT straight-through estimator.
        Returns float32."""
        s = self._scales(module).unsqueeze(1)
        return torch.round(w_rot.float() / s).clamp_(-127, 127) * s

    def _dequantize_rotated(self, module, dtype: torch.dtype) -> torch.Tensor:
        w = self._qdata(module).float() * self._scales(module).unsqueeze(1)
        return w.to(dtype)

    def dequantize(self, module) -> torch.Tensor:
        return rotate(
            self._dequantize_rotated(module, torch.float32), self._rot(module)
        )

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.cr8_qdata.device, dtype=torch.float32)
        q, scales = quantize_int8_rows(rotate(w, module.cr8_rot_size))
        module.cr8_qdata = q
        module.cr8_scales = scales.view(torch.uint8)

    @torch.no_grad()
    def requantize_codes_(self, module, fp_weight: torch.Tensor) -> None:
        """Re-quantize only the codes on the module's STORED scales — the grid a
        QAT run trains against (see ConvRotQuantizer.requantize_codes_)."""
        w = fp_weight.to(device=module.cr8_qdata.device, dtype=torch.float32)
        w_rot = rotate(w, self._rot(module))
        s = self._scales(module).unsqueeze(1)
        module.cr8_qdata = torch.round(w_rot / s).clamp_(-127, 127).to(torch.int8)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        rot = self._rot(module)
        in_f, out_f = module.in_features, module.out_features
        m = x.numel() // in_f

        if x.requires_grad:
            # training: gated on requires_grad alone so both gradient-checkpoint
            # passes take the same branch
            if _int8_gemm_supported(x.device):
                # int8 tensor-core forward (bit-identical to the inference path)
                # with a straight-through analytic backward. the rotation stays a
                # cublas matmul: an in-kernel tl.dot rotation was tried and measured
                # SLOWER on every shape (small-tile dots at low tensor-core
                # utilization, computed twice for the amax and quant passes, cost
                # more than the activation round-trips they saved)
                out = self._linear_ste(
                    module, rotate(x, rot).reshape(-1, in_f),
                    str(x.dtype).split(".")[-1],
                )
                return out.reshape(*x.shape[:-1], out_f)
            # no int8 hardware: straight-through fake-quant + bf16 matmul
            x2d = rotate(x, rot).reshape(-1, in_f)
            with torch.no_grad():
                aq, a_s = quantize_int8_rows(x2d.detach(), self.act_qmax)
                x_dq = (aq.float() * a_s.unsqueeze(1)).to(x.dtype)
                w = self._dequantize_rotated(module, x.dtype)
            x_ste = x2d + (x_dq - x2d).detach()
            out = F.linear(x_ste, w, module.bias)
            return out.reshape(*x.shape[:-1], out_f)

        if _int8_gemm_supported(x.device):
            # row padding for _int_mm happens inside the act-quant op (compile
            # safety); slice the mm output back to m rows (a contiguous prefix)
            aq, a_s = _int8_act_quant_padded(
                rotate(x, rot).reshape(-1, in_f), self.act_qmax
            )
            i32 = torch._int_mm(aq, self._qdata(module).t())
            out = _int8_epilogue(i32[:m], a_s[:m], self._scales(module), module.bias, x.dtype)
            return out.reshape(*x.shape[:-1], out_f)

        w = self._dequantize_rotated(module, x.dtype)
        out = F.linear(rotate(x, rot).reshape(-1, in_f), w, module.bias)
        return out.reshape(*x.shape[:-1], out_f)


# ---------------- convrotint2..8: W{n}A8 bitpacked int backend ----------------
#
# The convrot8 pipeline with the weight grid reduced to n bits: symmetric per-row
# codes in [-(2^(n-1)-1), 2^(n-1)-1], stored bitpacked. Weights are unpacked to
# int8 on the fly and run through the exact same per-token/per-channel
# torch._int_mm path (activations stay 8 bit), so speed matches convrot8 minus
# the unpack while storage shrinks to n/8 of int8.
#
# Packing layout: 8 consecutive codes along K occupy exactly n bytes (8*n bits),
# so every bit width 2..8 gets uniform, alignment-free addressing: group g of a
# row lives at bytes [g*n, (g+1)*n), code j at bit offset j*n inside that word.


def quantize_intn_rows(x: torch.Tensor, bits: int):
    """Symmetric per-row n-bit quantization. Returns (int8 codes in
    [-qmax, qmax] (rows, K), fp32 scales (rows,)) with qmax = 2^(bits-1) - 1."""
    qmax = (1 << (bits - 1)) - 1
    xf = x.float()
    scales = xf.abs().amax(dim=1) / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    q = torch.round(xf / scales.unsqueeze(1)).clamp_(-qmax, qmax).to(torch.int8)
    return q, scales


def pack_intn_rows(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack int8 codes in [-qmax, qmax] (rows, K) into a (rows, K//8*bits) uint8
    bitstream (K must be divisible by 8). Codes are stored offset by +qmax.
    int64 words may wrap for bits=8 but the fields are disjoint, so the
    shift/add/mask arithmetic stays bit-exact."""
    qmax = (1 << (bits - 1)) - 1
    rows, K = q.shape
    u = (q.to(torch.int64) + qmax).reshape(rows, K // 8, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.int64) * bits
    word = (u << shifts).sum(-1)
    byte_shifts = torch.arange(bits, device=q.device, dtype=torch.int64) * 8
    b = (word.unsqueeze(-1) >> byte_shifts) & 0xFF
    return b.to(torch.uint8).reshape(rows, K // 8 * bits)


def unpack_intn_rows(packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
    """Inverse of pack_intn_rows: (rows, cols) int8 codes in [-qmax, qmax]."""
    qmax = (1 << (bits - 1)) - 1
    b = packed.reshape(rows, cols // 8, bits).to(torch.int64)
    byte_shifts = torch.arange(bits, device=packed.device, dtype=torch.int64) * 8
    word = (b << byte_shifts).sum(-1)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * bits
    mask = (1 << bits) - 1
    u = (word.unsqueeze(-1) >> shifts) & mask
    return (u - qmax).to(torch.int8).reshape(rows, cols)


# --- 2/3-bit group-scale variant -------------------------------------------
#
# At 2-3 bits a single per-row amax scale is doubly wrong: rotated rows are
# near-gaussian so amax (~4 sigma) leaves the optimal-MSE step (~1.2 / 0.6
# sigma) far behind, and one scale can't adapt locally. So low widths quantize
# with MSE-optimal per-(row, k-group) scales instead. To keep the single
# torch._int_mm + per-row epilogue (no K-split), the unpack folds the group
# scales in by re-expressing each value on the row's int8 grid:
#
#   code8 = rint(code_n * gscale / rscale),  rscale = max_g(gscale_g) * qmax / 127
#
# The int8 snap adds <= 0.4% of row amax - noise next to the 2-bit error. The
# per-(row, group) RATIO gscale/rscale is precomputed in torch and stored, so
# the triton kernel only multiplies (float division in triton is ~1ulp off
# ieee, which would flip rint ties vs the torch fallback).

# bit widths <= this use group scales. 8 bit is excluded on purpose: the group
# codes are re-expressed on the row's int8 grid at unpack, and at 8 bits that
# grid is no finer than the group grids, so the snap noise outweighs the
# optimal-scale gain (measured 0.0087 -> 0.0106 weight err). 7 and below win.
INTN_GROUP_BITS = 7
INTN_GROUP = 128  # k-group size (halved until it divides in_features)


def _optimal_group_quantize(w: torch.Tensor, bits: int, group: int):
    """MSE-optimal symmetric n-bit quantization with per-(row, k-group) scales:
    coarse scale sweep then least-squares refits. Returns (int8 codes in
    [-qmax, qmax] (rows, K), fp32 group scales (rows, K//group))."""
    qmax = (1 << (bits - 1)) - 1
    rows, K = w.shape
    wg = w.float().reshape(rows, K // group, group)
    amax = wg.abs().amax(-1, keepdim=True)
    amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    best_s = amax / qmax
    best_e = None
    for frac in torch.linspace(0.2, 1.0, 16, dtype=torch.float64):
        s = amax * (float(frac) / qmax)
        q = torch.round(wg / s).clamp_(-qmax, qmax)
        e = (wg - q * s).square_().sum(-1, keepdim=True)
        if best_e is None:
            best_s, best_e = s, e
        else:
            better = e < best_e
            best_s = torch.where(better, s, best_s)
            best_e = torch.where(better, e, best_e)
    s = best_s
    for _ in range(2):
        q = torch.round(wg / s).clamp_(-qmax, qmax)
        num = (wg * q).sum(-1, keepdim=True)
        den = (q * q).sum(-1, keepdim=True)
        s = torch.where((den > 0) & (num > 0), num / den.clamp_min(1e-12), s)
    q = torch.round(wg / s).clamp_(-qmax, qmax)
    return q.to(torch.int8).reshape(rows, K), s.reshape(rows, K // group)


def _intn_group_ratio_and_rscales(gscales: torch.Tensor, bits: int):
    """Row int8-grid scales and the per-group unpack ratios for them."""
    qmax = (1 << (bits - 1)) - 1
    rscales = gscales.amax(dim=1) * (qmax / 127.0)
    rscales = torch.where(rscales > 0, rscales, torch.ones_like(rscales))
    return gscales / rscales.unsqueeze(1), rscales


def unpack_intn_rows_grouped(
    packed: torch.Tensor, gratio: torch.Tensor, bits: int, rows: int, cols: int
) -> torch.Tensor:
    """Torch fallback of the grouped unpack: n-bit codes -> row-grid int8."""
    codes = unpack_intn_rows(packed, bits, rows, cols).float()
    group = cols // gratio.shape[1]
    ratio = gratio.repeat_interleave(group, dim=1)
    return torch.round(codes * ratio).clamp_(-127, 127).to(torch.int8)


# --- convrotbitnet: 1.6-bit base-3 storage of the ternary width --------------
#
# BitNet-b1.58 style: the codes and scales are EXACTLY convrotint2's (ternary
# {-1,0,1}, MSE-optimal group scales), only the storage differs — 5 ternary
# codes per byte (3^5 = 243 <= 256) = 1.6 bits/weight instead of 2.0. Rows are
# padded to a multiple of 5 codes; the pad is never read back.


def pack_ternary_rows(q: torch.Tensor) -> torch.Tensor:
    """Pack int8 ternary codes (rows, K) into (rows, ceil(K/5)) uint8 base-3."""
    rows, K = q.shape
    bpr = -(-K // 5)
    u = q.to(torch.int16) + 1
    if bpr * 5 != K:
        u = F.pad(u, (0, bpr * 5 - K))
    w3 = torch.tensor([1, 3, 9, 27, 81], device=q.device, dtype=torch.int16)
    return (u.view(rows, bpr, 5) * w3).sum(-1).to(torch.uint8)


def unpack_ternary_rows_grouped(
    packed: torch.Tensor, gratio: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """Torch fallback: base-3 bytes -> ternary codes -> row-grid int8."""
    bpr = packed.shape[1]
    p3 = torch.tensor([1, 3, 9, 27, 81], device=packed.device, dtype=torch.int16)
    u = (packed.to(torch.int16).unsqueeze(-1) // p3) % 3
    codes = (u - 1).reshape(rows, bpr * 5)[:, :cols].float()
    group = cols // gratio.shape[1]
    ratio = gratio.repeat_interleave(group, dim=1)
    return torch.round(codes * ratio).clamp_(-127, 127).to(torch.int8)


_intn_kernel = None


def _get_intn_kernel():
    global _intn_kernel
    if _intn_kernel is not None:
        return _intn_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def intn_unpack_kernel(
        p_ptr, o_ptr, n_groups,
        BITS: tl.constexpr, BPOW: tl.constexpr, QMAX: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # one row per group of 8 codes; 2d blocks keep the byte loads and int8
        # stores contiguous/coalesced (BPOW = BITS padded to a power of two for
        # tl.arange, extra lanes masked off)
        pid = tl.program_id(0)
        g = pid * BLOCK + tl.arange(0, BLOCK)
        gm = g < n_groups
        bi = tl.arange(0, BPOW)
        b = tl.load(
            p_ptr + g[:, None] * BITS + bi[None, :],
            mask=gm[:, None] & (bi[None, :] < BITS),
            other=0,
        ).to(tl.int64)
        word = tl.sum(b << (8 * bi)[None, :].to(tl.int64), axis=1)
        j = tl.arange(0, 8)
        # mask after the shift kills any sign extension from int64 wrap
        v = ((word[:, None] >> (BITS * j)[None, :].to(tl.int64)) & ((1 << BITS) - 1)) - QMAX
        tl.store(o_ptr + g[:, None] * 8 + j[None, :], v.to(tl.int8), mask=gm[:, None])

    _intn_kernel = intn_unpack_kernel
    return _intn_kernel


_intn_grouped_kernel = None


def _get_intn_grouped_kernel():
    global _intn_grouped_kernel
    if _intn_grouped_kernel is not None:
        return _intn_grouped_kernel
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def intn_unpack_grouped_kernel(
        p_ptr, r_ptr, o_ptr, n_groups, cols8, gdiv8, ngprow,
        BITS: tl.constexpr, BPOW: tl.constexpr, QMAX: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # like intn_unpack_kernel, plus a per-(row, k-group) ratio multiply that
        # re-expresses the group-scaled codes on the row's int8 grid. the ratio
        # is precomputed (multiply only, so it bit-matches the torch fallback).
        pid = tl.program_id(0)
        g = pid * BLOCK + tl.arange(0, BLOCK)
        gm = g < n_groups
        row = g // cols8
        kg = (g - row * cols8) // gdiv8
        ratio = tl.load(r_ptr + row * ngprow + kg, mask=gm, other=1.0)
        bi = tl.arange(0, BPOW)
        b = tl.load(
            p_ptr + g[:, None] * BITS + bi[None, :],
            mask=gm[:, None] & (bi[None, :] < BITS),
            other=0,
        ).to(tl.int64)
        word = tl.sum(b << (8 * bi)[None, :].to(tl.int64), axis=1)
        j = tl.arange(0, 8)
        v = ((word[:, None] >> (BITS * j)[None, :].to(tl.int64)) & ((1 << BITS) - 1)) - QMAX
        vf = libdevice.rint(v.to(tl.float32) * ratio[:, None])
        vf = tl.minimum(tl.maximum(vf, -127.0), 127.0)
        tl.store(o_ptr + g[:, None] * 8 + j[None, :], vf.to(tl.int8), mask=gm[:, None])

    _intn_grouped_kernel = intn_unpack_grouped_kernel
    return _intn_grouped_kernel


_bitnet_kernel = None


def _get_bitnet_kernel():
    global _bitnet_kernel
    if _bitnet_kernel is not None:
        return _bitnet_kernel
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def bitnet_unpack_kernel(
        p_ptr, r_ptr, o_ptr, n_bytes, bpr, K, group, ngprow, BLOCK: tl.constexpr
    ):
        # one thread-lane per packed byte -> 5 output codes. a byte's 5 codes can
        # straddle a k-group boundary (group % 5 != 0), so the scale ratio is
        # gathered per output element rather than per byte.
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < n_bytes
        b = tl.load(p_ptr + i, mask=m, other=0).to(tl.int32)
        row = i // bpr
        bcol = i - row * bpr
        j = tl.arange(0, 8)
        col = bcol[:, None] * 5 + j[None, :]
        cm = m[:, None] & (j[None, :] < 5) & (col < K)
        p3 = tl.where(
            j < 1, 1, tl.where(j < 2, 3, tl.where(j < 3, 9, tl.where(j < 4, 27, 81)))
        )
        code = ((b[:, None] // p3[None, :]) % 3 - 1).to(tl.float32)
        ratio = tl.load(r_ptr + row[:, None] * ngprow + col // group, mask=cm, other=1.0)
        v = libdevice.rint(code * ratio)
        v = tl.minimum(tl.maximum(v, -127.0), 127.0)
        tl.store(o_ptr + row[:, None] * K + col, v.to(tl.int8), mask=cm)

    _bitnet_kernel = bitnet_unpack_kernel
    return _bitnet_kernel


def _unpack_intn_impl(packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
    if _triton_available() and packed.is_cuda:
        out = torch.empty(rows, cols, device=packed.device, dtype=torch.int8)
        n_groups = rows * cols // 8
        BLOCK = 256
        kernel = _get_intn_kernel()
        kernel[(-(-n_groups // BLOCK),)](
            packed, out, n_groups,
            BITS=bits, BPOW=max(2, 1 << (bits - 1).bit_length()),
            QMAX=(1 << (bits - 1)) - 1, BLOCK=BLOCK, num_warps=4,
        )
        return out
    return unpack_intn_rows(packed, bits, rows, cols)


def _unpack_intn_grouped_impl(
    packed: torch.Tensor, gratio: torch.Tensor, bits: int, rows: int, cols: int
) -> torch.Tensor:
    if _triton_available() and packed.is_cuda:
        out = torch.empty(rows, cols, device=packed.device, dtype=torch.int8)
        n_groups = rows * cols // 8
        ngprow = gratio.shape[1]
        group = cols // ngprow
        BLOCK = 256
        kernel = _get_intn_grouped_kernel()
        kernel[(-(-n_groups // BLOCK),)](
            packed, gratio, out, n_groups, cols // 8, group // 8, ngprow,
            BITS=bits, BPOW=max(2, 1 << (bits - 1).bit_length()),
            QMAX=(1 << (bits - 1)) - 1, BLOCK=BLOCK, num_warps=4,
        )
        return out
    return unpack_intn_rows_grouped(packed, gratio, bits, rows, cols)


# registered as custom ops so torch.compile treats the triton launches as opaque
# nodes with known output shapes (see _nvfp4_act_quant_op)
@torch.library.custom_op("ostris::convrot_intn_unpack", mutates_args=())
def _intn_unpack_op(packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
    return _unpack_intn_impl(packed, bits, rows, cols)


@_intn_unpack_op.register_fake
def _intn_unpack_fake(packed, bits, rows, cols):
    return torch.empty(rows, cols, device=packed.device, dtype=torch.int8)


@torch.library.custom_op("ostris::convrot_intn_unpack_grouped", mutates_args=())
def _intn_unpack_grouped_op(
    packed: torch.Tensor, gratio: torch.Tensor, bits: int, rows: int, cols: int
) -> torch.Tensor:
    return _unpack_intn_grouped_impl(packed, gratio, bits, rows, cols)


@_intn_unpack_grouped_op.register_fake
def _intn_unpack_grouped_fake(packed, gratio, bits, rows, cols):
    return torch.empty(rows, cols, device=packed.device, dtype=torch.int8)


# training-path linear for the grouped widths: same STE scheme as
# _int8_linear_ste_op, but autograd saves the PACKED codes and unpacks again in
# the backward — otherwise every layer holds a full int8-size unpacked weight
# until its backward runs, and the n/8 storage advantage disappears whenever
# gradient checkpointing isn't bounding liveness
@torch.library.custom_op("ostris::convrot_intn_linear_ste", mutates_args=())
def _intn_linear_ste_op(
    x2d: torch.Tensor,
    packed: torch.Tensor,
    gratio: torch.Tensor,
    w_scales_u8: torch.Tensor,
    bias: Optional[torch.Tensor],
    bits: int,
    act_qmax: int,
    out_dtype: str,
) -> torch.Tensor:
    m, K = x2d.shape
    rows = w_scales_u8.view(torch.float32).numel()
    qdata = _unpack_intn_grouped_impl(packed, gratio, bits, rows, K)
    aq, a_s = _int8_act_quant_padded(x2d, act_qmax)
    i32 = torch._int_mm(aq, qdata.t())
    return _int8_epilogue(
        i32[:m], a_s[:m], w_scales_u8.view(torch.float32), bias,
        getattr(torch, out_dtype),
    )


@_intn_linear_ste_op.register_fake
def _intn_linear_ste_fake(x2d, packed, gratio, w_scales_u8, bias, bits, act_qmax, out_dtype):
    return torch.empty(
        x2d.shape[0], w_scales_u8.view(torch.float32).numel(),
        device=x2d.device, dtype=getattr(torch, out_dtype),
    )


def _intn_linear_ste_setup(ctx, inputs, output):
    x2d, packed, gratio, w_scales_u8, bias, bits, act_qmax, out_dtype = inputs
    ctx.bits = bits
    ctx.cols = x2d.shape[1]
    ctx.save_for_backward(packed, gratio, w_scales_u8)


def _intn_linear_ste_backward(ctx, grad):
    packed, gratio, w_scales_u8 = ctx.saved_tensors
    w_scales = w_scales_u8.view(torch.float32)
    # through the opaque op, not the raw impl: registered backwards are traced
    # by torch.compile with fake tensors, which cannot reach a triton launch
    qdata = _intn_unpack_grouped_op(
        packed, gratio, ctx.bits, w_scales.numel(), ctx.cols
    )
    w = qdata.to(grad.dtype) * w_scales.to(grad.dtype).unsqueeze(1)
    return grad @ w, None, None, None, None, None, None, None


_intn_linear_ste_op.register_autograd(
    _intn_linear_ste_backward, setup_context=_intn_linear_ste_setup
)


def _unpack_bitnet_impl(
    packed: torch.Tensor, gratio: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    if _triton_available() and packed.is_cuda:
        out = torch.empty(rows, cols, device=packed.device, dtype=torch.int8)
        bpr = packed.shape[1]
        n_bytes = rows * bpr
        ngprow = gratio.shape[1]
        BLOCK = 256
        kernel = _get_bitnet_kernel()
        kernel[(-(-n_bytes // BLOCK),)](
            packed, gratio, out, n_bytes, bpr, cols, cols // ngprow, ngprow,
            BLOCK=BLOCK, num_warps=4,
        )
        return out
    return unpack_ternary_rows_grouped(packed, gratio, rows, cols)


@torch.library.custom_op("ostris::convrot_bitnet_unpack", mutates_args=())
def _bitnet_unpack_op(
    packed: torch.Tensor, gratio: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    return _unpack_bitnet_impl(packed, gratio, rows, cols)


@_bitnet_unpack_op.register_fake
def _bitnet_unpack_fake(packed, gratio, rows, cols):
    return torch.empty(rows, cols, device=packed.device, dtype=torch.int8)


@torch.library.custom_op("ostris::convrot_bitnet_linear_ste", mutates_args=())
def _bitnet_linear_ste_op(
    x2d: torch.Tensor,
    packed: torch.Tensor,
    gratio: torch.Tensor,
    w_scales_u8: torch.Tensor,
    bias: Optional[torch.Tensor],
    act_qmax: int,
    out_dtype: str,
) -> torch.Tensor:
    m, K = x2d.shape
    rows = w_scales_u8.view(torch.float32).numel()
    qdata = _unpack_bitnet_impl(packed, gratio, rows, K)
    aq, a_s = _int8_act_quant_padded(x2d, act_qmax)
    i32 = torch._int_mm(aq, qdata.t())
    return _int8_epilogue(
        i32[:m], a_s[:m], w_scales_u8.view(torch.float32), bias,
        getattr(torch, out_dtype),
    )


@_bitnet_linear_ste_op.register_fake
def _bitnet_linear_ste_fake(x2d, packed, gratio, w_scales_u8, bias, act_qmax, out_dtype):
    return torch.empty(
        x2d.shape[0], w_scales_u8.view(torch.float32).numel(),
        device=x2d.device, dtype=getattr(torch, out_dtype),
    )


def _bitnet_linear_ste_setup(ctx, inputs, output):
    x2d, packed, gratio, w_scales_u8, bias, act_qmax, out_dtype = inputs
    ctx.cols = x2d.shape[1]
    ctx.save_for_backward(packed, gratio, w_scales_u8)


def _bitnet_linear_ste_backward(ctx, grad):
    packed, gratio, w_scales_u8 = ctx.saved_tensors
    w_scales = w_scales_u8.view(torch.float32)
    # opaque op, not the raw impl: compile traces registered backwards with
    # fake tensors
    qdata = _bitnet_unpack_op(packed, gratio, w_scales.numel(), ctx.cols)
    w = qdata.to(grad.dtype) * w_scales.to(grad.dtype).unsqueeze(1)
    return grad @ w, None, None, None, None, None, None


_bitnet_linear_ste_op.register_autograd(
    _bitnet_linear_ste_backward, setup_context=_bitnet_linear_ste_setup
)


class ConvRotIntNQuantizer(ConvRotInt8Quantizer):
    """ConvRot W{n}A8 backend for n in 2..8: convrot8 numerics with an n-bit
    weight grid and bitpacked storage. One instance per bit-width, shareable
    across modules."""

    def __init__(self, bits: int, rot_size: int = 256):
        super().__init__(rot_size=rot_size)
        self.bits = bits

    @staticmethod
    def _group_for(in_features: int) -> int:
        group = INTN_GROUP
        while in_features % group:
            group //= 2
        return group

    def _quantize_rotated(self, w_rot: torch.Tensor, in_features: int):
        """Quantize the rotated weight. Returns (packed, row scales fp32,
        group-ratio fp32 or None)."""
        if self.bits <= INTN_GROUP_BITS:
            group = self._group_for(in_features)
            q, gscales = _optimal_group_quantize(w_rot, self.bits, group)
            gratio, rscales = _intn_group_ratio_and_rscales(gscales, self.bits)
            return pack_intn_rows(q, self.bits), rscales, gratio.contiguous()
        q, scales = quantize_intn_rows(w_rot, self.bits)
        return pack_intn_rows(q, self.bits), scales, None

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        rot = self._rot_for(module.in_features)
        packed, scales, gratio = self._quantize_rotated(
            rotate(weight_fp32, rot), module.in_features
        )
        module.register_buffer("crn_qdata", packed, persistent=False)
        # fp32 scales stored as a uint8 byte view (see convrot4: module.to(dtype=...)
        # would otherwise cast them)
        module.register_buffer("crn_scales", scales.view(torch.uint8), persistent=False)
        if gratio is not None:
            module.register_buffer(
                "crn_gratio", gratio.view(torch.uint8), persistent=False
            )
        module.crn_bits = self.bits
        module.crn_rot_size = rot

    def _rot(self, module) -> int:
        return module.crn_rot_size

    def _qdata(self, module) -> torch.Tensor:
        gratio = getattr(module, "crn_gratio", None)
        if gratio is not None:
            return _intn_unpack_grouped_op(
                module.crn_qdata, gratio.view(torch.float32),
                module.crn_bits, module.out_features, module.in_features,
            )
        return _intn_unpack_op(
            module.crn_qdata, module.crn_bits, module.out_features, module.in_features
        )

    def _scales_u8(self, module) -> torch.Tensor:
        return module.crn_scales

    def _linear_ste(self, module, x2d: torch.Tensor, out_dtype: str) -> torch.Tensor:
        gratio = getattr(module, "crn_gratio", None)
        if gratio is None:
            # ungrouped (8 bit, the convrot8 bit-identity anchor): base path.
            # autograd saves the unpacked int8 transient, which at 8 bit is the
            # same size as the packed buffer anyway
            return super()._linear_ste(module, x2d, out_dtype)
        # grouped: autograd saves only the packed codes (+ ratios), unpacked again
        # in the backward
        return _intn_linear_ste_op(
            x2d, module.crn_qdata, gratio.view(torch.float32),
            module.crn_scales, module.bias, module.crn_bits, self.act_qmax, out_dtype,
        )

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.crn_qdata.device, dtype=torch.float32)
        packed, scales, gratio = self._quantize_rotated(
            rotate(w, module.crn_rot_size), module.in_features
        )
        module.crn_qdata = packed
        module.crn_scales = scales.view(torch.uint8)
        if gratio is not None:
            module.crn_gratio = gratio.view(torch.uint8)

    def _codes_on_stored_scales(self, module, w_rot: torch.Tensor) -> torch.Tensor:
        """n-bit codes of a rotated weight on the module's stored scales (fp32)."""
        qmax = (1 << (module.crn_bits - 1)) - 1
        rs = self._scales(module).unsqueeze(1)
        wf = w_rot.float()
        gratio = getattr(module, "crn_gratio", None)
        if gratio is None:
            return torch.round(wf / rs).clamp_(-qmax, qmax)
        gratio = gratio.view(torch.float32)
        group = module.in_features // gratio.shape[1]
        gs = gratio.repeat_interleave(group, dim=1) * rs
        safe = torch.where(gs > 0, gs, torch.ones_like(gs))
        return torch.round(wf / safe).clamp_(-qmax, qmax)

    def fake_quant_rotated_weight(self, module, w_rot: torch.Tensor) -> torch.Tensor:
        """Like the int8 version but on the n-bit grid: quantize with the stored
        group scales, then re-express on the row int8 grid exactly like the
        deployed unpack does. Returns float32."""
        rs = self._scales(module).unsqueeze(1)
        codes = self._codes_on_stored_scales(module, w_rot)
        gratio = getattr(module, "crn_gratio", None)
        if gratio is None:
            return codes * rs
        gratio = gratio.view(torch.float32)
        group = module.in_features // gratio.shape[1]
        ratio = gratio.repeat_interleave(group, dim=1)
        code8 = torch.round(codes * ratio).clamp_(-127, 127)
        return code8 * rs

    def _pack(self, q: torch.Tensor) -> torch.Tensor:
        return pack_intn_rows(q, self.bits)

    @torch.no_grad()
    def requantize_codes_(self, module, fp_weight: torch.Tensor) -> None:
        """Re-quantize only the codes on the module's STORED scales — the grid a
        QAT run trains against (see ConvRotQuantizer.requantize_codes_)."""
        w = fp_weight.to(device=module.crn_qdata.device, dtype=torch.float32)
        codes = self._codes_on_stored_scales(module, rotate(w, self._rot(module)))
        module.crn_qdata = self._pack(codes.to(torch.int8))


class ConvRotBitNetQuantizer(ConvRotIntNQuantizer):
    """BitNet-b1.58 style W1.58A8: convrotint2's exact ternary codes and
    MSE-optimal group scales, stored base-3 at 5 codes/byte (1.6 bits/weight
    instead of 2.0). Numerically bit-identical to convrotint2."""

    def __init__(self, rot_size: int = 256):
        super().__init__(bits=2, rot_size=rot_size)

    def _quantize_rotated(self, w_rot: torch.Tensor, in_features: int):
        group = self._group_for(in_features)
        q, gscales = _optimal_group_quantize(w_rot, self.bits, group)
        gratio, rscales = _intn_group_ratio_and_rscales(gscales, self.bits)
        return pack_ternary_rows(q), rscales, gratio.contiguous()

    def _pack(self, q: torch.Tensor) -> torch.Tensor:
        return pack_ternary_rows(q)

    def _qdata(self, module) -> torch.Tensor:
        return _bitnet_unpack_op(
            module.crn_qdata, module.crn_gratio.view(torch.float32),
            module.out_features, module.in_features,
        )

    def _linear_ste(self, module, x2d: torch.Tensor, out_dtype: str) -> torch.Tensor:
        return _bitnet_linear_ste_op(
            x2d, module.crn_qdata, module.crn_gratio.view(torch.float32),
            module.crn_scales, module.bias, self.act_qmax, out_dtype,
        )


# ---------------- convrotcomfyw4a4: ComfyUI convrot_w4a4 compatible --------------
#
# Matches comfy_kitchen's TensorCoreConvRotW4A4Layout numerics exactly so a
# trained model exports to a checkpoint ComfyUI loads and runs natively:
#   - regular-Hadamard rotation, group size fixed at 256 (same R4 kronecker
#     matrix as the other backends; comfy rejects other sizes)
#   - weights: symmetric per-row int4, scale = row absmax / 7, RTN clamp [-7, 7]
#   - activations: per-token int4 the same way (act_qmax = 7)
# Storage here stays our bitpacked layout (bits=4); export_comfy_convrot_w4a4
# repacks to comfy's nibble pairs and writes weight/weight_scale/comfy_quant
# keys. Integer GEMM results are identical (int4 values through _int_mm
# accumulate exactly like their int4 MMA).


class ConvRotComfyW4A4Quantizer(ConvRotIntNQuantizer):
    """ComfyUI-compatible ConvRot W4A4 (see block comment above)."""

    act_qmax = 7
    COMFY_GROUPSIZE = 256

    def __init__(self, rot_size: int = 256):
        super().__init__(bits=4, rot_size=self.COMFY_GROUPSIZE)

    def _rot_for(self, d: int) -> int:
        return self.COMFY_GROUPSIZE

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        d = module.in_features
        if d % self.COMFY_GROUPSIZE != 0 or module.out_features % 8 != 0:
            if d not in _skip_warned:
                _skip_warned.add(d)
                print_acc(
                    f"ConvRot(comfy): skipping linears with in_features={d} "
                    f"(comfy convrot_w4a4 needs in divisible by {self.COMFY_GROUPSIZE}, "
                    f"out by 8)"
                )
            return False
        return True

    def _quantize_rotated(self, w_rot: torch.Tensor, in_features: int):
        # their format only fixes the DECODE (w = code * row_scale), not how the
        # codes/scales are chosen — so use MSE-optimal per-row scales (sweep +
        # LS refit with one group spanning the whole row) instead of their
        # amax/7. group scales can't be expressed: folding 4-bit group grids
        # onto the 4-bit row grid has no headroom (same inversion as the 8-bit
        # fold), so per-row optimal is the ceiling of the format
        q, gscales = _optimal_group_quantize(w_rot, self.bits, w_rot.shape[1])
        return pack_intn_rows(q, self.bits), gscales.reshape(-1), None


def export_comfy_convrot_w4a4(module, prefix: str) -> dict:
    """State-dict entries for one convrotcomfyw4a4-quantized OstrisLinear in
    ComfyUI's checkpoint format: packed int4 weight (low nibble = even column),
    fp32 per-row weight_scale, and the comfy_quant JSON marker."""
    import json

    q = module.ostris_quantizer
    if not isinstance(q, ConvRotComfyW4A4Quantizer):
        raise ValueError(
            f"export_comfy_convrot_w4a4 needs a convrotcomfyw4a4 module, got "
            f"qtype '{getattr(q, 'qtype', None)}'"
        )
    codes = _unpack_intn_impl(
        module.crn_qdata, module.crn_bits, module.out_features, module.in_features
    )
    lo = codes[:, 0::2].to(torch.int32) & 0x0F
    hi = codes[:, 1::2].to(torch.int32) & 0x0F
    packed = (lo | (hi << 4)).to(torch.int8)
    conf = {
        "format": "convrot_w4a4",
        "convrot_groupsize": module.crn_rot_size,
    }
    entries = {
        f"{prefix}weight": packed.contiguous(),
        f"{prefix}weight_scale": q._scales(module).contiguous(),
        f"{prefix}comfy_quant": torch.tensor(
            list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8
        ),
    }
    if module.bias is not None:
        entries[f"{prefix}bias"] = module.bias.detach()
    return entries


# ---------------- QAT: train the quantized weights themselves ----------------


def convrot_qat_forward(module, x: torch.Tensor) -> torch.Tensor:
    """Quantization-aware training forward for an OstrisLinear with a convrot
    backend and a trainable full-precision `qat_master` parameter attached.

    The weight enters the matmul through a straight-through fake-quant on the
    DEPLOYED grid using the module's stored scales (refresh them by calling
    module.requantize_(master) periodically), so the forward numerics track what
    the saved quantized model will compute while gradients flow to the master.
    Activations are fake-quanted like the no-hardware training path, also with a
    straight-through estimate, so gradients pass through to the input when it
    carries grad (end-to-end training; layer-local distillation feeds detached
    inputs and the act STE is then value-identical)."""
    q = module.ostris_quantizer
    rot = q._rot(module)
    in_f, out_f = module.in_features, module.out_features
    w_rot = rotate(module.qat_master.to(x.dtype), rot)
    with torch.no_grad():
        w_fq = q.fake_quant_rotated_weight(module, w_rot.detach()).to(x.dtype)
    w_ste = w_rot + (w_fq - w_rot).detach()
    x2d = rotate(x, rot).reshape(-1, in_f)
    with torch.no_grad():
        if isinstance(q, ConvRotQuantizer):
            pts1 = torch.ones((), device=x.device, dtype=torch.float32)
            aq, a_scales, _ = quantize_nvfp4(x2d.detach(), pts=pts1)
            x_dq = dequantize_nvfp4(aq, a_scales, pts1, x2d.shape[0], in_f, x.dtype)
        else:
            aq, a_scales = quantize_int8_rows(x2d.detach(), q.act_qmax)
            x_dq = (aq.float() * a_scales.unsqueeze(1)).to(x.dtype)
    x_ste = x2d + (x_dq - x2d).detach()
    out = F.linear(x_ste, w_ste, module.bias)
    return out.reshape(*x.shape[:-1], out_f)

