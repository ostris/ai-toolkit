# Vendored from the Boogu-Image repository (boogu/models/transformers/block_lumina2.py).
# Original work: Copyright 2025 BAAI / OmniGen2 / HuggingFace. Apache-2.0.
#
# The optional triton RMSNorm and flash-attn SwiGLU fast paths are dropped here;
# we always use torch.nn.RMSNorm and a plain SwiGLU so the model runs anywhere.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from torch.nn import RMSNorm

from .embeddings import TimestepEmbedding


def swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y


class LuminaRMSNormZero(nn.Module):
    """Adaptive RMS normalization with a zero-initialized modulation projection."""

    def __init__(
        self,
        embedding_dim: int,
        norm_eps: float,
        norm_elementwise_affine: bool,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )

        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(
                embedding_dim, eps=eps, elementwise_affine=elementwise_affine
            )
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        scale = emb
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x


class LuminaFeedForward(nn.Module):
    """A SwiGLU feed-forward layer with a multiple-of-256 inner dim."""

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.swiglu = swiglu

        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = nn.Linear(dim, inner_dim, bias=False)
        self.linear_2 = nn.Linear(inner_dim, dim, bias=False)
        self.linear_3 = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x):
        h1, h2 = self.linear_1(x), self.linear_3(x)
        return self.linear_2(self.swiglu(h1, h2))


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        instruction_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=timestep_scale,
        )

        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )

        self.caption_embedder = nn.Sequential(
            RMSNorm(instruction_feat_dim, eps=norm_eps),
            nn.Linear(instruction_feat_dim, hidden_size, bias=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.caption_embedder[1].weight, std=0.02)
        nn.init.zeros_(self.caption_embedder[1].bias)

    def forward(
        self,
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(instruction_hidden_states)
        return time_embed, caption_embed
