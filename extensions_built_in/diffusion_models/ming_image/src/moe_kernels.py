"""Grouped GEMMs for the Ling-mini MoE experts.

Tokens arrive sorted by expert with `offs` (num_experts + 1 int32 prefix sums)
marking each expert's row range; every expert's weights sit in one stacked bank
`(E, N, K)`. One launch per projection covers all experts, so a forward is a
handful of kernels instead of one per (expert, linear). Weights are bf16 or
int8 with a float32 per-output-channel scale (weight-only quantization; the
activations stay bf16 and the accumulator is fp32).

`grouped_swiglu(x, gate_up, scale, offs)` -> silu(x W_gate^T) * (x W_up^T)
`grouped_linear(x, w, scale, offs)`       -> x W^T

Both fall back to a per-expert torch loop without triton or off the gpu.
"""

from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _grouped_swiglu_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        offs_ptr,
        y_ptr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_xm,
        stride_ym,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        W_INT8: tl.constexpr,
    ):
        e = tl.program_id(0)
        nb = tl.program_id(1)
        start = tl.load(offs_ptr + e)
        end = tl.load(offs_ptr + e + 1)
        if start == end:
            return
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)
        w_base = w_ptr + e.to(tl.int64) * (2 * N * K)
        g_ptrs = w_base + offs_n[:, None] * K + offs_k[None, :]
        u_ptrs = w_base + (offs_n[:, None] + N) * K + offs_k[None, :]
        if W_INT8:
            s_g = tl.load(s_ptr + e * (2 * N) + offs_n, mask=n_mask, other=0.0)
            s_u = tl.load(s_ptr + e * (2 * N) + N + offs_n, mask=n_mask, other=0.0)
        for m0 in range(start, end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            m_mask = offs_m < end
            acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
            for k0 in range(0, K, BLOCK_K):
                a = tl.load(x_ptrs + k0, mask=m_mask[:, None], other=0.0)
                g = tl.load(g_ptrs + k0, mask=n_mask[:, None], other=0)
                u = tl.load(u_ptrs + k0, mask=n_mask[:, None], other=0)
                if W_INT8:
                    g = g.to(tl.bfloat16)
                    u = u.to(tl.bfloat16)
                acc_g = tl.dot(a, tl.trans(g), acc_g)
                acc_u = tl.dot(a, tl.trans(u), acc_u)
            if W_INT8:
                acc_g = acc_g * s_g[None, :]
                acc_u = acc_u * s_u[None, :]
            h = acc_g * tl.sigmoid(acc_g) * acc_u
            y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
            tl.store(y_ptrs, h.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _grouped_linear_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        offs_ptr,
        y_ptr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_xm,
        stride_ym,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        W_INT8: tl.constexpr,
    ):
        e = tl.program_id(0)
        nb = tl.program_id(1)
        start = tl.load(offs_ptr + e)
        end = tl.load(offs_ptr + e + 1)
        if start == end:
            return
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)
        w_ptrs = w_ptr + e.to(tl.int64) * (N * K) + offs_n[:, None] * K + offs_k[None, :]
        if W_INT8:
            s = tl.load(s_ptr + e * N + offs_n, mask=n_mask, other=0.0)
        for m0 in range(start, end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            m_mask = offs_m < end
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
            for k0 in range(0, K, BLOCK_K):
                a = tl.load(x_ptrs + k0, mask=m_mask[:, None], other=0.0)
                w = tl.load(w_ptrs + k0, mask=n_mask[:, None], other=0)
                if W_INT8:
                    w = w.to(tl.bfloat16)
                acc = tl.dot(a, tl.trans(w), acc)
            if W_INT8:
                acc = acc * s[None, :]
            y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
            tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def _use_triton(x: torch.Tensor, K: int) -> bool:
    # the kernels tile K by 64 without masking
    return HAS_TRITON and x.is_cuda and K % 64 == 0


def _block_m(rows: int, num_experts: int) -> int:
    # average rows per expert decides the M tile; tiny groups waste less with 16
    avg = rows / max(num_experts, 1)
    if avg >= 96:
        return 128
    if avg >= 48:
        return 64
    if avg >= 24:
        return 32
    return 16


def _check(x, w, scale, offs):
    E, N, K = w.shape
    if x.dim() != 2 or x.shape[1] != K:
        raise ValueError(f"expected activations (rows, {K}), got {tuple(x.shape)}")
    if offs.shape != (E + 1,):
        raise ValueError(f"offs must have {E + 1} entries, got {tuple(offs.shape)}")
    if w.dtype == torch.int8:
        if scale is None or scale.shape != (E, N) or scale.dtype != torch.float32:
            raise ValueError("int8 weights need a float32 (E, N) scale")
    elif w.dtype != x.dtype:
        raise ValueError(f"weight dtype {w.dtype} does not match activations {x.dtype}")


def grouped_swiglu(
    x: torch.Tensor, gate_up: torch.Tensor, scale: Optional[torch.Tensor], offs: torch.Tensor
) -> torch.Tensor:
    """`x` (rows, K) sorted by expert; `gate_up` (E, 2*I, K) with the gate rows
    first; returns silu(gate) * up as (rows, I)."""
    _check(x, gate_up, scale, offs)
    E, two_i, K = gate_up.shape
    inter = two_i // 2
    if not _use_triton(x, K):
        return _swiglu_fallback(x, gate_up, scale, offs)
    x = x.contiguous()
    y = torch.empty((x.shape[0], inter), device=x.device, dtype=x.dtype)
    if x.shape[0] == 0:
        return y
    block_m = _block_m(x.shape[0], E)
    block_n = 64
    grid = (E, triton.cdiv(inter, block_n))
    _grouped_swiglu_kernel[grid](
        x,
        gate_up,
        scale if scale is not None else gate_up,
        offs,
        y,
        K,
        inter,
        x.stride(0),
        y.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        W_INT8=gate_up.dtype == torch.int8,
        num_warps=4,
        num_stages=3,
    )
    return y


def grouped_linear(
    x: torch.Tensor, w: torch.Tensor, scale: Optional[torch.Tensor], offs: torch.Tensor
) -> torch.Tensor:
    """`x` (rows, K) sorted by expert; `w` (E, N, K); returns (rows, N)."""
    _check(x, w, scale, offs)
    E, N, K = w.shape
    if not _use_triton(x, K):
        return _linear_fallback(x, w, scale, offs)
    x = x.contiguous()
    y = torch.empty((x.shape[0], N), device=x.device, dtype=x.dtype)
    if x.shape[0] == 0:
        return y
    block_m = _block_m(x.shape[0], E)
    block_n = 64
    grid = (E, triton.cdiv(N, block_n))
    _grouped_linear_kernel[grid](
        x,
        w,
        scale if scale is not None else w,
        offs,
        y,
        K,
        N,
        x.stride(0),
        y.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        W_INT8=w.dtype == torch.int8,
        num_warps=4,
        num_stages=3,
    )
    return y


# ---------------- torch fallbacks (cpu, no triton) ----------------


def _expert_weight(w: torch.Tensor, scale: Optional[torch.Tensor], e: int, dtype: torch.dtype):
    if w.dtype == torch.int8:
        return (w[e].float() * scale[e][:, None]).to(dtype)
    return w[e].to(dtype)


def _swiglu_fallback(x, gate_up, scale, offs):
    inter = gate_up.shape[1] // 2
    y = torch.empty((x.shape[0], inter), device=x.device, dtype=x.dtype)
    bounds = offs.tolist()
    for e in range(gate_up.shape[0]):
        start, end = bounds[e], bounds[e + 1]
        if start == end:
            continue
        w = _expert_weight(gate_up, scale, e, torch.float32)
        h = x[start:end].float() @ w.t()
        y[start:end] = (F.silu(h[:, :inter]) * h[:, inter:]).to(x.dtype)
    return y


def _linear_fallback(x, w, scale, offs):
    y = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)
    bounds = offs.tolist()
    for e in range(w.shape[0]):
        start, end = bounds[e], bounds[e + 1]
        if start == end:
            continue
        we = _expert_weight(w, scale, e, torch.float32)
        y[start:end] = (x[start:end].float() @ we.t()).to(x.dtype)
    return y
