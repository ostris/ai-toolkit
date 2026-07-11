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

CONVROT_QTYPES = ("convrot4", "convrot8")


def get_convrot_quantizer(qtype: str):
    if qtype == "convrot4":
        return ConvRotQuantizer(rot_size=256)
    if qtype == "convrot8":
        return ConvRotInt8Quantizer(rot_size=256)
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


def quantize_nvfp4(x: torch.Tensor, pts: Optional[torch.Tensor] = None):
    """Quantize (rows, K) to nvfp4. Returns (packed uint8 (rows, K/2),
    e4m3 scales (rows, K/16), fp32 per-tensor scale)."""
    rows, K = x.shape
    xf = x.float()
    if pts is None:
        pts = xf.abs().amax() / (F4_MAX * F8_E4M3_MAX)
        pts = torch.where(pts > 0, pts, torch.ones_like(pts))
    xb = xf.view(rows, K // BLOCK, BLOCK)
    scales = (xb.abs().amax(dim=-1) / (F4_MAX * pts)).to(torch.float8_e4m3fn)
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
    BLOCK_K = 2048 if K >= 2048 else K
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
        packed, scales, pts = quantize_nvfp4(w_rot)
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

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.cr_qdata.device, dtype=torch.float32)
        w_rot = rotate(w, module.cr_rot_size)
        packed, scales, pts = quantize_nvfp4(w_rot)
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


def quantize_int8_rows(x: torch.Tensor):
    """Symmetric per-row int8 quantization. Returns (int8 (rows, K), fp32 scales (rows,))."""
    xf = x.float()
    scales = xf.abs().amax(dim=1) / 127.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    q = torch.round(xf / scales.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
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
    def int8_act_quant_kernel(x_ptr, q_ptr, s_ptr, K, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        base = row * K
        acc = tl.zeros((BLOCK_K,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            v = tl.load(x_ptr + base + offs, mask=offs < K, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(v))
        amax = tl.max(acc, axis=0)
        scale = tl.where(amax > 0, amax / 127.0, 1.0)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            # rint = round-half-to-even, matching torch.round in the reference path
            q = libdevice.rint(v / scale)
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
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
def _int8_act_quant_op(x: torch.Tensor) -> list[torch.Tensor]:
    rows, K = x.shape
    rows_pad = -(-rows // 32) * 32
    x = x.contiguous()
    q = torch.empty(rows_pad, K, device=x.device, dtype=torch.int8)
    scales = torch.empty(rows_pad, device=x.device, dtype=torch.float32)
    if rows_pad != rows:
        q[rows:].zero_()
        scales[rows:].fill_(1.0)
    kernel, _ = _get_int8_kernels()
    kernel[(rows,)](x, q, scales, K, BLOCK_K=2048 if K >= 2048 else K, num_warps=8)
    return [q, scales]


@_int8_act_quant_op.register_fake
def _int8_act_quant_fake(x):
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


def quantize_int8_rows_fused(x: torch.Tensor):
    """Triton path of quantize_int8_rows: one extra read of x instead of the
    multi-kernel torch chain. Falls back to the torch ops."""
    if not (_triton_available() and x.is_cuda):
        return quantize_int8_rows(x)
    q, scales = _int8_act_quant_op(x)
    rows = x.shape[0]
    return q[:rows], scales[:rows]


def _int8_act_quant_padded(x: torch.Tensor):
    """Act quant with rows padded to a multiple of 32 for torch._int_mm."""
    if _triton_available() and x.is_cuda:
        q, scales = _int8_act_quant_op(x)
        return q, scales
    q, scales = quantize_int8_rows(x)
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
    out_dtype: str,
) -> torch.Tensor:
    m = x2d.shape[0]
    aq, a_s = _int8_act_quant_padded(x2d)
    i32 = torch._int_mm(aq, qdata.t())
    return _int8_epilogue(
        i32[:m], a_s[:m], w_scales_u8.view(torch.float32), bias,
        getattr(torch, out_dtype),
    )


@_int8_linear_ste_op.register_fake
def _int8_linear_ste_fake(x2d, qdata, w_scales_u8, bias, out_dtype):
    return torch.empty(
        x2d.shape[0], qdata.shape[0], device=x2d.device, dtype=getattr(torch, out_dtype)
    )


def _int8_linear_ste_setup(ctx, inputs, output):
    x2d, qdata, w_scales_u8, bias, out_dtype = inputs
    ctx.save_for_backward(qdata, w_scales_u8)


def _int8_linear_ste_backward(ctx, grad):
    qdata, w_scales_u8 = ctx.saved_tensors
    w_scales = w_scales_u8.view(torch.float32).to(grad.dtype)
    w = qdata.to(grad.dtype) * w_scales.unsqueeze(1)
    return grad @ w, None, None, None, None


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

    @staticmethod
    def _scales(module) -> torch.Tensor:
        return module.cr8_scales.view(torch.float32)

    def _dequantize_rotated(self, module, dtype: torch.dtype) -> torch.Tensor:
        w = module.cr8_qdata.float() * self._scales(module).unsqueeze(1)
        return w.to(dtype)

    def dequantize(self, module) -> torch.Tensor:
        return rotate(
            self._dequantize_rotated(module, torch.float32), module.cr8_rot_size
        )

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.cr8_qdata.device, dtype=torch.float32)
        q, scales = quantize_int8_rows(rotate(w, module.cr8_rot_size))
        module.cr8_qdata = q
        module.cr8_scales = scales.view(torch.uint8)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        rot = module.cr8_rot_size
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
                out = _int8_linear_ste_op(
                    rotate(x, rot).reshape(-1, in_f),
                    module.cr8_qdata, module.cr8_scales, module.bias,
                    str(x.dtype).split(".")[-1],
                )
                return out.reshape(*x.shape[:-1], out_f)
            # no int8 hardware: straight-through fake-quant + bf16 matmul
            x2d = rotate(x, rot).reshape(-1, in_f)
            with torch.no_grad():
                aq, a_s = quantize_int8_rows(x2d.detach())
                x_dq = (aq.float() * a_s.unsqueeze(1)).to(x.dtype)
                w = self._dequantize_rotated(module, x.dtype)
            x_ste = x2d + (x_dq - x2d).detach()
            out = F.linear(x_ste, w, module.bias)
            return out.reshape(*x.shape[:-1], out_f)

        if _int8_gemm_supported(x.device):
            # row padding for _int_mm happens inside the act-quant op (compile
            # safety); slice the mm output back to m rows (a contiguous prefix)
            aq, a_s = _int8_act_quant_padded(rotate(x, rot).reshape(-1, in_f))
            i32 = torch._int_mm(aq, module.cr8_qdata.t())
            out = _int8_epilogue(i32[:m], a_s[:m], self._scales(module), module.bias, x.dtype)
            return out.reshape(*x.shape[:-1], out_f)

        w = self._dequantize_rotated(module, x.dtype)
        out = F.linear(rotate(x, rot).reshape(-1, in_f), w, module.bias)
        return out.reshape(*x.shape[:-1], out_f)

