"""Tiled column-major conversion for one-byte FP8 matrix operands."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fp8_transpose_kernel(
        src_ptr,
        dst_ptr,
        m,
        n,
        stride_sm,
        stride_sn,
        stride_dm,
        stride_dn,
        BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK + tl.arange(0, BLOCK)
        rn = pid_n * BLOCK + tl.arange(0, BLOCK)
        mask = (rm[:, None] < m) & (rn[None, :] < n)
        tile = tl.load(
            src_ptr + rm[:, None] * stride_sm + rn[None, :] * stride_sn,
            mask=mask,
            other=0,
        )
        tl.store(
            dst_ptr + rn[:, None] * stride_dm + rm[None, :] * stride_dn,
            tl.trans(tile),
            mask=tl.trans(mask),
        )

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton-less installs only
    _HAVE_TRITON = False


_BLOCK = 64


def _tiled_supported(x: torch.Tensor) -> bool:
    return (
        _HAVE_TRITON
        and x.device.type == "cuda"
        and x.ndim == 2
        and x.element_size() == 1
        and x.numel() > 0
    )


@torch.library.custom_op("mm::transpose_contiguous_1byte", mutates_args=())
def transpose_contiguous_1byte(x: torch.Tensor) -> torch.Tensor:
    """Return the contiguous transpose of a two-dimensional one-byte tensor."""
    if not _tiled_supported(x):
        return x.t().contiguous()
    src = x.view(torch.uint8)
    m, n = src.shape
    dst = torch.empty((n, m), device=src.device, dtype=torch.uint8)
    grid = (triton.cdiv(m, _BLOCK), triton.cdiv(n, _BLOCK))
    _fp8_transpose_kernel[grid](
        src,
        dst,
        m,
        n,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        BLOCK=_BLOCK,
    )
    return dst.view(x.dtype)


@transpose_contiguous_1byte.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device)


def column_major(x: torch.Tensor) -> torch.Tensor:
    """Return ``x`` as a column-major operand for ``torch._scaled_mm``."""
    return torch.ops.mm.transpose_contiguous_1byte(x).t()
