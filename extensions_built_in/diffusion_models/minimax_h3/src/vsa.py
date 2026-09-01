"""FastVideo VSA (Video Sparse Attention) for MiniMax-H3, on FlexAttention.

Replicates ``fastvideo/attention/backends/video_sparse_attn_h3.py`` without
the fastvideo_kernel CUDA/Triton extensions. The packed
``[text | condition | audio | video]`` sequence is re-tiled so every 64
contiguous rows form one tile: prefix segments chunk in order (tiles never
straddle a segment boundary) and the video tail becomes (4, 4, 4) 3D tiles,
zero-padding ragged edges. Per-tile fp32 mean pooling of q/k scores every
tile pair; each video query tile keeps the top
``ceil((1 - sparsity) * num_video_tiles)`` video tiles ("exempt" mode:
prefix tiles are always visible and prefix queries run dense). The fine
stage is exact attention over the selected tiles via ``flex_attention`` with
a 64-token BlockMask; the compression branch adds
``gate * softmax(scores) @ v_pooled`` per tile, gate from
``to_gate_compress``.

The FastH3 4-step checkpoints were distilled WITH this policy (sparsity 0.9,
tile 64 is the trained geometry), so training and sampling both run it, at
every sequence length.
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

TILE_SHAPE = (4, 4, 4)
TILE = 64

# fine-stage backend: "auto" runs FastVideo's vendored Triton block-sparse
# kernels when eligible (uniform batch, fp16/bf16, default scale) and
# FlexAttention otherwise; "flex" / "fv_triton" force one side.
VSA_BACKEND = os.environ.get("AITK_H3_VSA_BACKEND", "auto")

_flex_compiled = None
_triton_ok: Optional[bool] = None


def vsa_is_available() -> bool:
    """Compiled FlexAttention needs triton (inductor GPU codegen). Without it
    the only flex path is the eager fallback, which materializes the full
    score matrix — unusable at video lengths — so VSA is disabled instead."""
    global _triton_ok
    if _triton_ok is None:
        try:
            import triton  # noqa: F401

            _triton_ok = True
        except Exception:
            _triton_ok = False
            print(
                "WARNING: triton is not installed; VSA sparse attention is "
                "disabled and MiniMax-H3 FastH3 falls back to dense attention "
                "(FastVideo's dense mode, compression gate off). Install "
                "triton to run the checkpoint's trained sparse policy."
            )
    return _triton_ok


def _get_flex(disable_compile: bool = False):
    if disable_compile:
        return flex_attention
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_compiled


@dataclass(frozen=True)
class H3VSAGeometry:
    seq_len: int
    n_tiles: int
    num_prefix_tiles: int
    num_video_tiles: int
    tile_sizes: torch.Tensor  # (n_tiles,) long, live rows per tile
    untile_index: torch.Tensor  # (seq_len,) long, packed row -> padded slot
    slot_valid: torch.Tensor  # (n_tiles * TILE,) bool


@dataclass
class H3VSAContext:
    geometry: H3VSAGeometry
    sparsity: float
    token_valid: Optional[torch.Tensor]  # (B, S) bool, None = all live


def _video_tile_partition(
    grid: Tuple[int, int, int], device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row indices of the (T, H, W) token grid grouped into (4, 4, 4) tiles in
    (t-tile, h-tile, w-tile) raster order, plus each tile's live-row count."""
    t, h, w = grid
    ts, hs, ws = TILE_SHAPE
    indices = torch.arange(t * h * w, device=device, dtype=torch.long).reshape(t, h, w)
    parts, sizes = [], []
    for ti in range(math.ceil(t / ts)):
        for hi in range(math.ceil(h / hs)):
            for wi in range(math.ceil(w / ws)):
                block = indices[
                    ti * ts : min(ti * ts + ts, t),
                    hi * hs : min(hi * hs + hs, h),
                    wi * ws : min(wi * ws + ws, w),
                ].flatten()
                parts.append(block)
                sizes.append(block.numel())
    return torch.cat(parts), torch.tensor(sizes, dtype=torch.long, device=device)


@lru_cache(maxsize=16)
def _geometry_cached(
    prefix_segments: Tuple[int, ...], video_grid: Tuple[int, int, int], device_str: str
) -> H3VSAGeometry:
    device = torch.device(device_str)
    prefix_len = sum(prefix_segments)

    prefix_sizes = []
    for segment in prefix_segments:
        full, rem = divmod(segment, TILE)
        prefix_sizes.extend([TILE] * full)
        if rem:
            prefix_sizes.append(rem)
    num_prefix_tiles = len(prefix_sizes)

    video_partition, video_sizes = _video_tile_partition(video_grid, device)
    num_video_tiles = int(video_sizes.numel())

    partition = torch.cat(
        [
            torch.arange(prefix_len, device=device, dtype=torch.long),
            video_partition + prefix_len,
        ]
    )
    tile_sizes = torch.cat(
        [torch.tensor(prefix_sizes, dtype=torch.long, device=device), video_sizes]
    )
    n_tiles = int(tile_sizes.numel())
    seq_len = int(partition.numel())

    # padded slot of the k-th row of the tile-ordered sequence: tiles occupy
    # TILE slots each, live rows sit at the front of their tile
    starts = torch.arange(n_tiles, device=device, dtype=torch.long) * TILE
    shift = torch.cat([tile_sizes.new_zeros(1), tile_sizes.cumsum(0)[:-1]])
    intra = torch.arange(seq_len, device=device) - shift.repeat_interleave(tile_sizes)
    non_pad = starts.repeat_interleave(tile_sizes) + intra
    untile_index = non_pad[torch.argsort(partition)]

    slot_valid = torch.zeros(n_tiles * TILE, dtype=torch.bool, device=device)
    slot_valid[non_pad] = True
    return H3VSAGeometry(
        seq_len=seq_len,
        n_tiles=n_tiles,
        num_prefix_tiles=num_prefix_tiles,
        num_video_tiles=num_video_tiles,
        tile_sizes=tile_sizes,
        untile_index=untile_index,
        slot_valid=slot_valid,
    )


def build_vsa_context(
    seq_len: int,
    num_text_rows: int,
    video_grid: Tuple[int, int, int],
    token_tags: torch.Tensor,  # (B, S) long, -1 marks pad rows
    sparsity: float,
    device: torch.device,
) -> H3VSAContext:
    """Context for one forward. The video tokens are the sequence tail; the
    prefix splits as (text, everything between text and video) — for the t2v
    packing this is FastVideo's (text, audio) segmentation exactly."""
    video_rows = video_grid[0] * video_grid[1] * video_grid[2]
    prefix_rest = seq_len - num_text_rows - video_rows
    if prefix_rest < 0:
        raise ValueError(
            f"VSA video grid {video_grid} ({video_rows} rows) does not fit the "
            f"packed sequence ({seq_len} rows, {num_text_rows} text)"
        )
    prefix_segments = tuple(s for s in (num_text_rows, prefix_rest) if s > 0)
    geometry = _geometry_cached(prefix_segments, tuple(video_grid), str(device))
    token_valid = None
    is_pad = token_tags < 0
    if bool(is_pad.any()):
        token_valid = ~is_pad
    return H3VSAContext(geometry=geometry, sparsity=sparsity, token_valid=token_valid)


def compute_topk(sparsity: float, num_blocks: int) -> int:
    """Video tiles each video query tile keeps, clamped to [1, num_blocks]."""
    return max(1, min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks))


def vsa_attention(
    q: torch.Tensor,  # (B, S, H, D), post qk-norm and rope
    k: torch.Tensor,
    v: torch.Tensor,
    gate: Optional[torch.Tensor],  # (B, S, H, D) from to_gate_compress, or None
    ctx: H3VSAContext,
    scale: Optional[float] = None,
    disable_compile: bool = False,
    backend: Optional[str] = None,  # None -> VSA_BACKEND
) -> torch.Tensor:
    b, s, heads, d = q.shape
    g = ctx.geometry
    if s != g.seq_len:
        raise ValueError(f"VSA geometry was built for {g.seq_len} rows, got {s}")
    if scale is None:
        scale = d**-0.5
    idx = g.untile_index
    n = g.n_tiles
    s_pad = n * TILE

    token_valid = ctx.token_valid
    if token_valid is not None:
        # dead rows must not enter pooling or act as keys
        live = token_valid[..., None, None].to(q.dtype)
        q, k, v = q * live, k * live, v * live
        slot_valid = torch.zeros(b, s_pad, dtype=torch.bool, device=q.device)
        slot_valid[:, idx] = token_valid
        sizes = slot_valid.view(b, n, TILE).sum(-1).to(torch.float32)  # (B, n)
    else:
        slot_valid = g.slot_valid.unsqueeze(0).expand(b, -1)
        sizes = g.tile_sizes.to(torch.float32).unsqueeze(0)  # (1, n)

    def tile_rows(x):
        buf = x.new_zeros(b, s_pad, heads, d)
        buf[:, idx] = x
        return buf

    qt, kt, vt = tile_rows(q), tile_rows(k), tile_rows(v)

    def pool(x):
        # pad slots are zero, so a plain fp32 sum / live count is a masked mean
        p = x.view(b, n, TILE, heads, d).sum(2, dtype=torch.float32)
        return (p / sizes.clamp(min=1.0)[..., None, None]).permute(0, 2, 1, 3)

    scores = torch.matmul(pool(qt), pool(kt).transpose(-2, -1)) * scale  # (B,H,n,n)

    npre = g.num_prefix_tiles
    k_vid = compute_topk(ctx.sparsity, g.num_video_tiles)
    if k_vid == g.num_video_tiles:
        mask = torch.ones(b, heads, n, n, dtype=torch.bool, device=q.device)
    else:
        mask = torch.zeros(b, heads, n, n, dtype=torch.bool, device=q.device)
        top = scores[..., npre:].topk(k_vid, dim=-1).indices + npre
        mask.scatter_(-1, top, True)
        mask[..., :npre] = True  # prefix tiles always visible ("exempt" mode)
        mask[:, :, :npre, :] = True  # prefix queries run dense

    def finish(out):
        """Shared tail: gated compression branch, then un-tile to packed order."""
        if gate is not None:
            # compression branch: dense attention over the pooled tiles,
            # broadcast to each tile's rows, scaled by the learned gate
            gt = tile_rows(gate)
            dead = sizes <= 0  # a tile of only pad rows must not be attended
            cs = scores
            if bool(dead.any()):
                cs = cs.masked_fill(dead.view(-1, 1, 1, n), torch.finfo(cs.dtype).min)
            oc = torch.matmul(torch.softmax(cs.float(), dim=-1), pool(vt))
            oc = oc.permute(0, 2, 1, 3).to(out.dtype)  # (B, n, H, D)
            out = (
                out.view(b, n, TILE, heads, d)
                + oc.unsqueeze(2) * gt.view(b, n, TILE, heads, d)
            ).view(b, s_pad, heads, d)
        return out[:, idx]

    if backend is None:
        backend = VSA_BACKEND
    use_fv = backend != "flex" and (
        token_valid is None  # per-item pad rows can't be expressed as tile sizes
        and q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and scale == d**-0.5  # the kernel hardcodes 1/sqrt(D)
    )
    if backend == "fv_triton" and not use_fv:
        raise ValueError(
            "fv_triton backend needs a uniform (pad-free) batch, fp16/bf16 "
            "CUDA tensors and the default scale"
        )
    if use_fv:
        from . import vsa_kernels

        out = vsa_kernels.block_sparse_attn(
            qt.permute(0, 2, 1, 3),
            kt.permute(0, 2, 1, 3),
            vt.permute(0, 2, 1, 3),
            mask,
            g.tile_sizes,
        ).permute(0, 2, 1, 3)  # (B, S_pad, H, D)
        return finish(out)

    # fully-live selected tiles skip the mask_mod; ragged/padded tiles keep it
    col_full = (sizes >= TILE).view(-1, 1, 1, n)
    m_full = mask & col_full
    m_part = mask & ~col_full

    def to_blocks(m):
        num = m.sum(-1, dtype=torch.int32)
        order = m.to(torch.uint8).argsort(dim=-1, descending=True, stable=True)
        return num, order.to(torch.int32)

    valid = slot_valid
    tile_mask = mask

    def mask_mod(bi, hi, qi, ki):
        # the FULL mask truth: eager flex ignores the kv block lists and
        # evaluates only mask_mod, so it must carry tile selection too
        return tile_mask[bi, hi, qi // TILE, ki // TILE] & valid[bi, ki]

    part_num, part_idx = to_blocks(m_part)
    full_num, full_idx = to_blocks(m_full)
    block_mask = BlockMask.from_kv_blocks(
        part_num,
        part_idx,
        full_kv_num_blocks=full_num,
        full_kv_indices=full_idx,
        BLOCK_SIZE=TILE,
        mask_mod=mask_mod,
    )

    out = _get_flex(disable_compile)(
        qt.permute(0, 2, 1, 3),
        kt.permute(0, 2, 1, 3),
        vt.permute(0, 2, 1, 3),
        block_mask=block_mask,
        scale=scale,
        # inductor's flex kernels need their tile sizes to divide the 64-token
        # mask blocks; the defaults (128) reject BLOCK_SIZE=64
        kernel_options={
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        },
    ).permute(0, 2, 1, 3)  # (B, S_pad, H, D)

    return finish(out)
