"""Varlen attention shim for Mage-Flow.

The reference implementation (github.com/microsoft/Mage) runs every attention
call through flash-attn's ``flash_attn_varlen_func`` so that several
variable-length samples can be packed into one sequence and kept isolated via
``cu_seqlens``. This shim exposes the same function with the FA2 calling
convention, falling back to a per-sequence ``torch.scaled_dot_product_attention``
loop when flash-attn is not installed (functionally equivalent for the dense /
non-causal path the DiT uses, just slower).
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

_RESOLVED_FN: Callable[..., Any] | None = None


def _sdpa_varlen(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    **_unused: Any,
):
    """FA2-varlen-compatible SDPA fallback: one SDPA dispatch per sequence."""
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    outs = []
    for qs, qe, ks, ke in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
        # (s, h, d) -> (1, h, s, d)
        q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)
        k_i = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_i = v[ks:ke].transpose(0, 1).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        outs.append(out_i.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0).contiguous()


def _resolve() -> Callable[..., Any]:
    global _RESOLVED_FN
    if _RESOLVED_FN is None:
        try:
            from flash_attn import flash_attn_varlen_func as _fn

            _RESOLVED_FN = _fn
        except ImportError:
            _RESOLVED_FN = _sdpa_varlen
    return _RESOLVED_FN


def flash_attn_varlen_func(*args, **kwargs):
    return _resolve()(*args, **kwargs)


__all__ = ["flash_attn_varlen_func"]
