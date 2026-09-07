"""Vendored from FastVideo (https://github.com/hao-ai-lab/FastVideo),
``fastvideo-kernel/python/fastvideo_kernel/triton_kernels/index.py``.
Apache-2.0; Copyright the FastVideo team. Vendored unmodified so the
MiniMax-H3 VSA fine stage can run FastVideo's trained-policy Triton
block-sparse kernels (fwd + bwd) without the fastvideo-kernel package."""

## pytorch sdpa version of block sparse ##
from typing import Tuple

import triton
import triton.language as tl
import torch


@triton.jit
def topk_index_to_map_kernel(
    map_ptr,
    index_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    topk,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = (
        index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    )
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    for i in tl.static_range(topk):
        index = tl.load(index_ptr_base + i * index_kv_stride)
        tl.store(map_ptr_base + index * map_kv_stride, 1.0)


@triton.jit
def map_to_index_kernel(
    map_ptr,
    index_ptr,
    index_num_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    index_num_bs_stride,
    index_num_h_stride,
    index_num_q_stride,
    num_kv_blocks,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = (
        index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    )
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    num = 0
    for i in tl.range(num_kv_blocks):
        map_entry = tl.load(map_ptr_base + i * map_kv_stride)
        if map_entry:
            tl.store(index_ptr_base + num * index_kv_stride, i)
            num += 1

    tl.store(
        index_num_ptr
        + b * index_num_bs_stride
        + h * index_num_h_stride
        + q * index_num_q_stride,
        num,
    )


def topk_index_to_map(
    index: torch.Tensor, num_kv_blocks: int, transpose_map: bool = False
):
    """
    Convert topk indices to a map.

    Args:
        index: [bs, h, num_q_blocks, topk]
            The topk indices tensor.
        num_kv_blocks: int
            The number of key-value blocks in the block_map returned
        transpose_map: bool
            If True, the block_map will be transposed on the final two dimensions.

    Returns:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            A binary map where 1 indicates that the q block attends to the kv block.
    """
    bs, h, num_q_blocks, topk = index.shape

    if transpose_map is False:
        block_map = torch.zeros(
            (bs, h, num_q_blocks, num_kv_blocks), dtype=torch.bool, device=index.device
        )
    else:
        block_map = torch.zeros(
            (bs, h, num_kv_blocks, num_q_blocks), dtype=torch.bool, device=index.device
        )
        block_map = block_map.transpose(2, 3)

    grid = (bs, h, num_q_blocks)
    topk_index_to_map_kernel[grid](
        block_map,
        index,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        topk=topk,
    )

    return block_map


def map_to_index(block_map: torch.Tensor):
    """
    Convert a block map to indices and counts.

    Args:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            The block map tensor.

    Returns:
        index: [bs, h, num_q_blocks, num_kv_blocks]
            The indices of the blocks.
        index_num: [bs, h, num_q_blocks]
            The number of blocks for each q block.
    """
    bs, h, num_q_blocks, num_kv_blocks = block_map.shape

    index = torch.full(
        (block_map.shape), -1, dtype=torch.int32, device=block_map.device
    )
    index_num = torch.empty(
        (bs, h, num_q_blocks), dtype=torch.int32, device=block_map.device
    )

    grid = (bs, h, num_q_blocks)
    map_to_index_kernel[grid](
        block_map,
        index,
        index_num,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        index_num.stride(0),
        index_num.stride(1),
        index_num.stride(2),
        num_kv_blocks=num_kv_blocks,
    )

    return index, index_num


@triton.jit
def _invert_indices_kernel(
    q2k_idx_ptr,
    q2k_num_ptr,
    k2q_idx_ptr,
    k2q_num_ptr,
    q2k_idx_b,
    q2k_idx_h,
    q2k_idx_q,
    q2k_idx_k,
    q2k_num_b,
    q2k_num_h,
    q2k_num_q,
    k2q_idx_b,
    k2q_idx_h,
    k2q_idx_k,
    k2q_idx_q,
    k2q_num_b,
    k2q_num_h,
    k2q_num_k,
    MAX_KV_PER_Q: tl.constexpr,
):
    # One program per (b, h, q): reserve a slot in k2q via atomicAdd, write q.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    n = tl.load(q2k_num_ptr + pid_b * q2k_num_b + pid_h * q2k_num_h + pid_q * q2k_num_q)

    q2k_row = q2k_idx_ptr + pid_b * q2k_idx_b + pid_h * q2k_idx_h + pid_q * q2k_idx_q

    for i in tl.range(0, MAX_KV_PER_Q):
        if i < n:
            kv = tl.load(q2k_row + i * q2k_idx_k)
            count_ptr = (
                k2q_num_ptr + pid_b * k2q_num_b + pid_h * k2q_num_h + kv * k2q_num_k
            )
            pos = tl.atomic_add(count_ptr, 1)
            tl.store(
                k2q_idx_ptr
                + pid_b * k2q_idx_b
                + pid_h * k2q_idx_h
                + kv * k2q_idx_k
                + pos * k2q_idx_q,
                pid_q,
            )


def invert_indices(
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    num_kv_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transpose a Q->KV index list into a K->Q one via atomic compaction (GPU)."""
    if q2k_idx.dim() != 4:
        raise ValueError(
            f"q2k_idx must be [B, H, Nq, Mk], got shape={tuple(q2k_idx.shape)}"
        )
    if q2k_num.dim() != 3:
        raise ValueError(
            f"q2k_num must be [B, H, Nq], got shape={tuple(q2k_num.shape)}"
        )
    if not q2k_idx.is_cuda or not q2k_num.is_cuda:
        raise RuntimeError("invert_indices requires CUDA tensors.")

    B, H, Nq, Mk = q2k_idx.shape
    if q2k_num.shape != (B, H, Nq):
        raise ValueError(
            f"q2k_num shape {tuple(q2k_num.shape)} does not match q2k_idx "
            f"[B, H, Nq] = {(B, H, Nq)}"
        )

    q2k_idx = q2k_idx.contiguous()
    q2k_num = q2k_num.contiguous()
    if q2k_idx.dtype != torch.int32:
        q2k_idx = q2k_idx.to(torch.int32)
    if q2k_num.dtype != torch.int32:
        q2k_num = q2k_num.to(torch.int32)

    # Any KV block is attended by at most Nq Q blocks (one per Q row), so
    # `Nq` is a tight upper bound on the compacted K->Q slots.
    k2q_idx = torch.empty(
        (B, H, num_kv_blocks, Nq),
        dtype=torch.int32,
        device=q2k_idx.device,
    )
    k2q_num = torch.zeros(
        (B, H, num_kv_blocks),
        dtype=torch.int32,
        device=q2k_idx.device,
    )

    grid = (B, H, Nq)
    _invert_indices_kernel[grid](
        q2k_idx,
        q2k_num,
        k2q_idx,
        k2q_num,
        q2k_idx.stride(0),
        q2k_idx.stride(1),
        q2k_idx.stride(2),
        q2k_idx.stride(3),
        q2k_num.stride(0),
        q2k_num.stride(1),
        q2k_num.stride(2),
        k2q_idx.stride(0),
        k2q_idx.stride(1),
        k2q_idx.stride(2),
        k2q_idx.stride(3),
        k2q_num.stride(0),
        k2q_num.stride(1),
        k2q_num.stride(2),
        MAX_KV_PER_Q=Mk,
    )

    return k2q_idx, k2q_num
