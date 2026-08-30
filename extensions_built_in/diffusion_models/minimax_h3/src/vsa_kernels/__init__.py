"""Autograd glue for the vendored FastVideo Triton block-sparse kernels.

Adapted from FastVideo's ``fastvideo_kernel/block_sparse_attn.py`` (Apache-2.0)
with the compiled sm90/sm100a backends stripped: only the pure-Triton fwd+bwd
pair remains, JIT-compiled on first call. Custom ops live in their own
``aitk_h3_vsa::`` namespace so a real fastvideo-kernel install cannot clash.

Public entry: ``block_sparse_attn(q, k, v, block_map, variable_block_sizes)``
with q/k/v ``[B, H, S_pad, D]`` (S_pad a multiple of 64), block_map a bool
``[B, H, n_tiles, n_tiles]``, variable_block_sizes int32 ``[n_tiles]`` live
rows per tile (live rows at the front of each 64-token tile).
"""

from typing import Tuple

import torch


def _as_int32_contig(t: torch.Tensor, name: str) -> torch.Tensor:
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor, got device={t.device}")
    if t.dtype != torch.int32:
        t = t.to(torch.int32)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


@torch.library.custom_op(
    "aitk_h3_vsa::block_sparse_attn_triton",
    mutates_args=(),
    device_types="cuda",
)
def _block_sparse_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from .block_sparse_attn_triton import triton_block_sparse_attn_forward

    o, M = triton_block_sparse_attn_forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        q2k_idx,
        q2k_num,
        variable_block_sizes,
    )
    return o, M


@torch.library.register_fake("aitk_h3_vsa::block_sparse_attn_triton")
def _block_sparse_attn_triton_fake(q, k, v, q2k_idx, q2k_num, variable_block_sizes):
    o = torch.empty_like(q)
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    return o, M


@torch.library.custom_op(
    "aitk_h3_vsa::block_sparse_attn_backward_triton",
    mutates_args=(),
    device_types="cuda",
)
def _block_sparse_attn_backward_triton(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    M: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .block_sparse_attn_triton import triton_block_sparse_attn_backward
    from .index import invert_indices

    num_kv_blocks = int(variable_block_sizes.numel())
    k2q_idx, k2q_num = invert_indices(q2k_idx, q2k_num, num_kv_blocks=num_kv_blocks)
    # q/k/v are saved from the user-facing inputs and may be non-contiguous;
    # o/M are kernel outputs so are already contiguous.
    dq, dk, dv = triton_block_sparse_attn_backward(
        grad_output.contiguous(),
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        o,
        M,
        q2k_idx,
        q2k_num,
        k2q_idx,
        k2q_num,
        variable_block_sizes,
    )
    return dq, dk, dv


@torch.library.register_fake("aitk_h3_vsa::block_sparse_attn_backward_triton")
def _block_sparse_attn_backward_triton_fake(
    grad_output, q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, q2k_idx, q2k_num, variable_block_sizes = inputs
    o, M = output
    ctx.save_for_backward(q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes)


def _backward(ctx, grad_o, grad_M):
    q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes = ctx.saved_tensors
    dq, dk, dv = _block_sparse_attn_backward_triton(
        grad_o, q, k, v, o, M, q2k_idx, q2k_num, variable_block_sizes
    )
    return dq, dk, dv, None, None, None


_block_sparse_attn_triton.register_autograd(_backward, setup_context=_setup_context)


def block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Block-sparse attention with autograd from a bool tile map. Returns o."""
    from .index import map_to_index

    q2k_idx, q2k_num = map_to_index(block_map)
    q2k_idx = _as_int32_contig(q2k_idx, "q2k_idx")
    q2k_num = _as_int32_contig(q2k_num, "q2k_num")
    variable_block_sizes = _as_int32_contig(
        variable_block_sizes, "variable_block_sizes"
    )
    o, _ = _block_sparse_attn_triton(q, k, v, q2k_idx, q2k_num, variable_block_sizes)
    return o
