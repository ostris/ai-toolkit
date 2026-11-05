"""MXFP8 (block-scaled FP8) attention forward+backward for sm_120, wired as a
torch.autograd.Function. Forward = flashinfer-style ragged mxfp8 kernel
(prebuilt mxfp8_ragged_mx.so). Backward = dv64/dk64/dq_ws kernels
(fa-blackwell/kernels/attn_sm120/tests/{s3b64,s3b_dq_ws}_kernel.cuh).

Contract: q/k/v [H, L, 128] (batch must be 1), non-causal. L is zero-padded to a
multiple of 128 internally. Saved-for-backward: e4m3 data + ue8m0 SF for
Q/K/V (natural) and Q/K (transposed); dO is quantized in backward.

Enable via AITK_MXFP8_ATTN=1 (see patch point in flux2 model.py attention()).
"""
import importlib.util
import math
import os

import torch
import torch.nn.functional as F

_ENABLED = os.environ.get("AITK_MXFP8_ATTN", "0") == "1"
_DK_WS = os.environ.get("AITK_DK_WS", "1") == "1"
# bwd kernel mode: 1 = dv_ws+dk_ws split (default), 3 = dvdk2 fused (faster)
_DK_WS_MODE = int(os.environ.get("AITK_DK_WS_MODE", "1")) if "AITK_DK_WS_MODE" in os.environ else (1 if _DK_WS else 0)
_FWD_SO = "/root/.cache/torch_extensions/py312_cu130/mxfp8_ragged_mx/mxfp8_ragged_mx.so"
_BWD_SRC = [
    "/root/fa-blackwell/kernels/attn_sm120/tests/csrc/mxfp8_bwd_ext.cpp",
    "/root/fa-blackwell/kernels/attn_sm120/tests/csrc/mxfp8_bwd_kernel.cu",
    "/root/fa-blackwell/kernels/attn_sm120/tests/csrc/mxfp8_ragged_mx_kernel.cu",
]
_BWD_INC = ["/root/fa-blackwell/tmp/cutlass/include", "/root/fa-blackwell/include",
            "/root/fa-blackwell/kernels/attn_sm120/tests"]

_fwd = None
_bwd = None


def enabled() -> bool:
    return _ENABLED


def _load_fwd():
    global _fwd
    if _fwd is None:
        spec = importlib.util.spec_from_file_location("mxfp8_ragged_mx", _FWD_SO)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _fwd = mod
    return _fwd


def _load_bwd():
    global _bwd
    if _bwd is None:
        from torch.utils.cpp_extension import load
        _bwd = load(
            name="mxfp8_bwd_ext" + ("_dbg" if os.environ.get("AITK_DVDK_DEBUG") == "1" else "") + ("_li" if os.environ.get("AITK_LINEINFO") == "1" else ""),
            sources=_BWD_SRC,
            extra_include_paths=_BWD_INC,
            extra_cuda_cflags=[
                "-O2", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "-gencode", "arch=compute_120a,code=sm_120a",
            ] + (["-DDVDK_DEBUG"] if os.environ.get("AITK_DVDK_DEBUG") == "1" else []) + (["-lineinfo"] if os.environ.get("AITK_LINEINFO") == "1" else []),
            extra_cflags=["-O2"],
            verbose=os.environ.get("AITK_MXFP8_ATTN_VERBOSE", "0") == "1",
        )
    return _bwd


def _e8m0(amax: torch.Tensor) -> torch.Tensor:
    """amax [..] fp32 -> ue8m0 byte tensor (uint8), matches kernel e8m0_byte."""
    b = torch.ceil(torch.log2(amax.clamp_min(2.0**-126))) - 8 + 127
    return b.clamp(1, 254).to(torch.uint8)


def _pack_nat(b: torch.Tensor) -> torch.Tensor:
    """raw sf [H, S, 4] -> packed [H, S//128, 512]: off(r,kb) = (r//128)*512 + 16*(r%32) + 4*((r%128)//32) + kb"""
    H, S, KD = b.shape
    sf = b.reshape(H, S // 128, 4, 32, KD)
    return sf.permute(0, 1, 3, 2, 4).reshape(H, S // 128, 512).contiguous()


def _quant_nat(x: torch.Tensor):
    """x [H, S, 128] -> data uint8 [H, S, 128], raw sf uint8 [H, S, 4]."""
    H, S, D = x.shape
    xf = x.float().reshape(H, S, D // 32, 32)
    b = _e8m0(xf.abs().amax(-1))                          # [H, S, 4]
    scale = torch.exp2(b.float() - 127).unsqueeze(-1)     # [H, S, 4, 1]
    q = (xf / scale).to(torch.float8_e4m3fn).view(torch.uint8).reshape(H, S, D).contiguous()
    return q, b


def _pack_trn(b: torch.Tensor) -> torch.Tensor:
    """raw sf [H, 128, S//32] -> packed [H, S//128, 512]: off(d,kb) = (kb//4)*512 + 16*(d%32) + 4*(d//32) + (kb%4)"""
    H, D, KB = b.shape
    sf = b.reshape(H, 4, 32, KB // 4, 4)                  # [H, dhi, dlo, t, kblo]
    return sf.permute(0, 3, 2, 1, 4).reshape(H, KB // 4, 512).contiguous()


def _quant_trn(x: torch.Tensor):
    """x [H, S, 128] -> data uint8 [H, 128, S] (d-major), raw sf uint8 [H, 128, S//32].
    Scales along the token axis (32-token blocks per (feature, block))."""
    H, S, D = x.shape
    xt = x.transpose(1, 2).contiguous().float().reshape(H, D, S // 32, 32)
    b = _e8m0(xt.abs().amax(-1))                          # [H, D, S/32]
    scale = torch.exp2(b.float() - 127).unsqueeze(-1)
    q = (xt / scale).to(torch.float8_e4m3fn).view(torch.uint8).reshape(H, D, S).contiguous()
    return q, b


_META_CACHE = {}


def _ragged_meta(dev, S: int, L: int):
    key = (dev.index, S, L)
    t = _META_CACHE.get(key)
    if t is None:
        indptr = torch.tensor([0, S], dtype=torch.int32, device=dev)  # padded coords
        lens = torch.tensor([L], dtype=torch.int32, device=dev)       # true lengths
        t = (indptr, indptr.clone(), lens, lens.clone())
        _META_CACHE[key] = t
    return t


# Registered as a proper custom op so torch.compile keeps ONE graph (no graph
# break + per-segment guard re-eval that @torch.compiler.disable caused).
# Returns [out bf16 [H,L,D], qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o]
# -- the trailing tensors are the backward saves, kept alive by the graph.
from torch.library import custom_op


@custom_op("aitk::mxfp8_attn", mutates_args=(), device_types="cuda")
def _mx8_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> list[torch.Tensor]:
    bwd = _load_bwd()
    H, L, D = q.shape
    S = (L + 127) // 128 * 128
    if os.environ.get("AITK_MXFP8_SHAPES") == "1":
        print(f"[mxfp8] fwd H={H} L={L} S={S}", flush=True)
    qdT, kdT, vt, rq, rk, rvt, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, sfvt = bwd.fwd_prep(q, k, v)
    o, lse = bwd.fwd_attn(qdT, kdT, vt, sfq, sfk, sfvt, L, float(sm_scale))  # o [S,H,D] bf16 (S15)
    out = o.permute(1, 0, 2)[:, :L].contiguous()  # already bf16; inductor wants contiguous
    return [out, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o]


@_mx8_op.register_fake
def _(q, k, v, sm_scale):
    H, L, D = q.shape
    S = (L + 127) // 128 * 128
    u8 = dict(dtype=torch.uint8, device=q.device)
    f32 = dict(dtype=torch.float32, device=q.device)
    return [q.new_empty(H, L, D),
            q.new_empty(H, S, D, **u8), q.new_empty(H, S, D, **u8), q.new_empty(H, S, D, **u8),
            q.new_empty(H, D, S, **u8), q.new_empty(H, D, S, **u8),
            q.new_empty(H, S // 128, 512, **u8), q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S // 128, 512, **u8), q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S // 128, 512, **u8),
            q.new_empty(H, S, **f32), q.new_empty(S, H, D)]   # lse fp32, o bf16 (S15)


def _mx8_setup(ctx, inputs, output):
    q, k, v, sm_scale = inputs
    ctx.save_for_backward(*output[1:])
    ctx.sm_scale = sm_scale


# Backward body as its own opaque op: AOTAutograd traces the custom-op backward
# into the compiled graph, so the pybind ext must hide behind an op boundary too.
@custom_op("aitk::mxfp8_attn_bwd", mutates_args=(), device_types="cuda")
def _mx8_bwd_op(do: torch.Tensor, qd: torch.Tensor, kd: torch.Tensor, vd: torch.Tensor,
                qt: torch.Tensor, kt: torch.Tensor, sfq: torch.Tensor, sfk: torch.Tensor,
                sfv: torch.Tensor, sfqt: torch.Tensor, sfkt: torch.Tensor,
                lse: torch.Tensor, o: torch.Tensor, sm_scale: float, mode: int) -> list[torch.Tensor]:
    bwd = _load_bwd()
    return bwd.bwd_full(do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt,
                        lse, o, sm_scale, mode)


@_mx8_bwd_op.register_fake
def _(do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o, sm_scale, mode):
    H, L, D = do.shape
    return [do.new_empty(H, L, D), do.new_empty(H, L, D), do.new_empty(H, L, D)]


def _mx8_bwd(ctx, grad_out, *grad_rest):
    if isinstance(grad_out, (list, tuple)):   # multi-output op: grads arrive as one list
        grad_out = grad_out[0]
    qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt, lse, o = ctx.saved_tensors
    do = grad_out.contiguous()
    dq, dk, dv = torch.ops.aitk.mxfp8_attn_bwd(
        do, qd, kd, vd, qt, kt, sfq, sfk, sfv, sfqt, sfkt,
        lse, o, ctx.sm_scale, _DK_WS_MODE)
    return dq, dk, dv, None


_mx8_op.register_autograd(_mx8_bwd, setup_context=_mx8_setup)


def mxfp8_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q/k/v [B=1, H, L, 128] -> [B, H, L, 128], bf16 in/out."""
    B, H, L, D = q.shape
    x = torch.ops.aitk.mxfp8_attn(q[0], k[0], v[0], 1.0 / math.sqrt(D))[0]
    return x.unsqueeze(0)


def available(q: torch.Tensor) -> bool:
    return (_ENABLED and q.is_cuda and q.dim() == 4 and q.shape[0] == 1
            and q.shape[-1] == 128 and q.dtype == torch.bfloat16)
