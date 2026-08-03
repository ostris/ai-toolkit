"""Mage-Flow NR-MMDiT (dual-stream, native-resolution packed) for ai-toolkit.

Vendored from the reference implementation
(github.com/microsoft/Mage, mage_flow/models/mage_flow.py + modules/mage_layers.py)
with these deviations:

  - loguru / pydantic / torch._dynamo decorators dropped,
  - gradient checkpointing is gated on ``self.gradient_checkpointing and
    torch.is_grad_enabled()`` (instead of the reference's ``self.training and
    self.checkpoint``) and exposed via ``enable_gradient_checkpointing`` /
    ``disable_gradient_checkpointing`` so ai-toolkit's trainer can toggle it,
  - flash-attn is imported through the local shim (``.attn``) which falls back
    to a per-sequence SDPA loop when flash-attn is unavailable.

State-dict keys are identical to the reference checkpoints
(``transformer/diffusion_pytorch_model.safetensors`` in
microsoft/Mage-Flow-Base and microsoft/Mage-Flow-Edit-Base).

Sequence layout: images and text are packed as variable-length sequences of
batch dim 1 (``[1, sum_len, C]``) with per-sample ``cu_seqlens``; joint text+image
attention runs in a single varlen kernel. 2D multi-scale RoPE is applied to
image tokens only (text tokens are not rotated). ``timesteps`` is the flow
sigma in [0, 1] (1 = pure noise), one entry per packed sample.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.normalization import RMSNorm

from .attn import flash_attn_varlen_func


def apply_rotary_emb_mageflow(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex rotary embeddings to `x` ([S, H, D]) using `freqs_cis`
    (the MageFlowEmbedRope 2D multi-scale RoPE, adjacent-pair complex convention)."""
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(-2)
    return x_out.type_as(x)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings (DDPM convention).

    NOTE: kept vendored (not diffusers') because the frequency table is
    downcast to ``timesteps.dtype`` (bf16) here — the model was trained with
    this exact bf16 rounding, so diffusers' fp32 variant produces a slightly
    different embedding and degrades outputs.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class MageFlowTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_states.dtype)
        )  # (N, D)
        return timesteps_emb


class MageFlowEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list, scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope
        self.video_freq_cache = {}

    def rope_params(self, index, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self, video_fhw, device: torch.device, max_img_len: int = None
    ) -> torch.Tensor:
        """Compute the vision RoPE frequencies for the packed image tokens.
        Text tokens are NOT rotated, so no text RoPE is computed.

        ``video_fhw`` is a list of (frame, height, width) tuples, one per image
        segment in the packed sequence; segment index doubles as the "frame"
        RoPE coordinate (edit reference images land on later frame indices).
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            key = (frame, height, width, idx)
            if key not in self.video_freq_cache:
                self.video_freq_cache[key] = self._compute_video_freqs(
                    frame, height, width, idx
                )
            vid_freqs.append(self.video_freq_cache[key].to(device))

        vid_freqs = torch.cat(vid_freqs, dim=0)

        if max_img_len is not None and vid_freqs.shape[0] < max_img_len:
            pad_len = max_img_len - vid_freqs.shape[0]
            vid_freqs = torch.nn.functional.pad(vid_freqs, (0, 0, 0, pad_len))

        return vid_freqs

    def _compute_video_freqs(
        self, frame: int, height: int, width: int, idx: int = 0
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = (
            freqs_pos[0][idx : idx + frame]
            .view(frame, 1, 1, -1)
            .expand(frame, height, width, -1)
        )
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(
                frame, height, width, -1
            )
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(
                frame, height, width, -1
            )
        else:
            freqs_height = (
                freqs_pos[1][:height]
                .view(1, height, 1, -1)
                .expand(frame, height, width, -1)
            )
            freqs_width = (
                freqs_pos[2][:width]
                .view(1, 1, width, -1)
                .expand(frame, height, width, -1)
            )

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(
            seq_lens, -1
        )
        return freqs.clone().contiguous()


class Attention(nn.Module):
    """Joint text+image attention (reference subset: the double-stream config
    MageFlow actually instantiates — qk rms_norm, added kv projections)."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        processor=None,
        out_dim: int = None,
        out_context_dim: int = None,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.query_dim = query_dim
        self.use_bias = bias
        self.cross_attention_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = (
            out_context_dim if out_context_dim is not None else query_dim
        )
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        # qk_norm is always "rms_norm" for MageFlow.
        self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)

        self.add_k_proj = nn.Linear(
            added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias
        )
        self.add_v_proj = nn.Linear(
            added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias
        )
        self.add_q_proj = nn.Linear(
            added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
        )
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)

        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        txt_cu_lens: torch.Tensor = None,
        img_cu_lens: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        **attention_kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            txt_cu_lens=txt_cu_lens,
            img_cu_lens=img_cu_lens,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )


class MageDoubleStreamAttnProcessor:
    """Attention processor for the Mage double-stream architecture. Implements
    joint attention where the packed text and image streams are concatenated
    per sample (order: [text, image]) and run through one varlen kernel."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream [1, sum_img, D]
        img_cu_lens: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream [1, sum_txt, D]
        txt_cu_lens: torch.LongTensor = None,
        image_rotary_emb: torch.Tensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError(
                "MageDoubleStreamAttnProcessor requires encoder_hidden_states (text stream)"
            )

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if img_query.ndim == 4:
            img_query = img_query.flatten(0, 1)
            img_key = img_key.flatten(0, 1)
            img_value = img_value.flatten(0, 1)

        if txt_query.ndim == 4:
            txt_query = txt_query.flatten(0, 1)
            txt_key = txt_key.flatten(0, 1)
            txt_value = txt_value.flatten(0, 1)

        # Apply QK normalization
        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)
        txt_query = attn.norm_added_q(txt_query)
        txt_key = attn.norm_added_k(txt_key)

        # Apply 2D multi-scale RoPE (MageFlowEmbedRope) to image tokens
        img_freqs = image_rotary_emb
        img_query = apply_rotary_emb_mageflow(img_query, img_freqs)
        img_key = apply_rotary_emb_mageflow(img_key, img_freqs)

        # Calculate lengths
        img_lens = img_cu_lens[1:] - img_cu_lens[:-1]
        txt_lens = txt_cu_lens[1:] - txt_cu_lens[:-1]

        # Calculate joint cu_seqlens
        joint_lens = txt_lens + img_lens
        joint_cu_lens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=joint_lens.device),
                torch.cumsum(joint_lens, dim=0, dtype=torch.int32),
            ],
            dim=0,
        )

        device = joint_lens.device
        batch_size = len(txt_lens)
        sample_indices = torch.arange(batch_size, device=device)

        txt_sample_ids = torch.repeat_interleave(sample_indices, txt_lens)
        img_sample_ids = torch.repeat_interleave(sample_indices, img_lens)

        txt_intra_pos = (
            torch.arange(txt_query.shape[0], device=device)
            - txt_cu_lens[txt_sample_ids]
        )
        img_intra_pos = (
            torch.arange(img_query.shape[0], device=device)
            - img_cu_lens[img_sample_ids]
        )

        txt_dest_indices = joint_cu_lens[txt_sample_ids] + txt_intra_pos
        img_dest_indices = (
            joint_cu_lens[img_sample_ids] + txt_lens[img_sample_ids] + img_intra_pos
        )

        total_tokens = joint_cu_lens[-1]
        joint_query = torch.empty(
            (total_tokens, *txt_query.shape[1:]), dtype=txt_query.dtype, device=device
        )
        joint_key = torch.empty(
            (total_tokens, *txt_key.shape[1:]), dtype=txt_key.dtype, device=device
        )
        joint_value = torch.empty(
            (total_tokens, *txt_value.shape[1:]), dtype=txt_value.dtype, device=device
        )

        joint_query[txt_dest_indices] = txt_query
        joint_query[img_dest_indices] = img_query

        joint_key[txt_dest_indices] = txt_key
        joint_key[img_dest_indices] = img_key

        joint_value[txt_dest_indices] = txt_value
        joint_value[img_dest_indices] = img_value

        max_seqlen = joint_lens.max().item()
        joint_attn_output = flash_attn_varlen_func(
            joint_query,
            joint_key,
            joint_value,
            cu_seqlens_q=joint_cu_lens,
            cu_seqlens_k=joint_cu_lens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )

        txt_attn_output = joint_attn_output[txt_dest_indices]
        img_attn_output = joint_attn_output[img_dest_indices]

        img_attn_output = img_attn_output.flatten(1, 2)  # (N, H, D) -> (N, H*D)
        img_attn_output = img_attn_output.to(joint_query.dtype)

        txt_attn_output = txt_attn_output.flatten(1, 2)  # (N, H, D) -> (N, H*D)
        txt_attn_output = txt_attn_output.to(joint_query.dtype)

        img_attn_output = attn.to_out[0](img_attn_output)
        img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)
        txt_attn_output = txt_attn_output.view(
            encoder_hidden_states.shape[0],
            encoder_hidden_states.shape[1],
            txt_attn_output.shape[-1],
        )

        return img_attn_output, txt_attn_output


class MageFlowTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim, 6 * dim, bias=True
            ),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=MageDoubleStreamAttnProcessor(),
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim, 6 * dim, bias=True
            ),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def _modulate(self, x, mod_params, cu_lens=None):
        """Apply modulation to input tensor"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if cu_lens is not None:
            assert x.shape[0] == 1, "x must be of shape (1, *) when cu_lens is not None"
            x_flattened = x.view(-1, x.shape[-1])
            lengths = cu_lens[1:] - cu_lens[:-1]
            shift_t = shift.repeat_interleave(lengths, dim=0)
            scale_t = scale.repeat_interleave(lengths, dim=0)
            gate_t = gate.repeat_interleave(lengths, dim=0)

            x_flattened = x_flattened * (1 + scale_t) + shift_t
            x = x_flattened.view(x.shape)
            return x, gate_t
        else:
            return x * (1 + scale) + shift, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        txt_cu_lens: torch.Tensor,
        img_cu_lens: torch.Tensor,
        joint_attention_kwargs: dict = None,
    ):
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(
            img_normed, img_mod1, cu_lens=img_cu_lens
        )

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(
            txt_normed, txt_mod1, cu_lens=txt_cu_lens
        )

        # Joint attention: computes QKV for both streams, applies QK norm and
        # RoPE, concatenates per sample and splits the results back.
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            txt_cu_lens=txt_cu_lens,
            img_cu_lens=img_cu_lens,
            **joint_attention_kwargs,
        )

        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(
            img_normed2, img_mod2, cu_lens=img_cu_lens
        )
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(
            txt_normed2, txt_mod2, cu_lens=txt_cu_lens
        )
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class AdaLayerNormContinuous(nn.Module):
    """Adaptive output norm; supports packed sequences via ``cu_seqlens``."""

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        if cu_seqlens is None:
            scale, shift = torch.chunk(emb, 2, dim=-1)
            x = self.norm(x) * (1 + scale) + shift
        else:
            sample_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            flattened_x = x.view(-1, x.shape[-1])
            scale, shift = torch.chunk(emb, 2, dim=-1)
            scale_t = torch.repeat_interleave(scale, sample_lens, dim=0)
            shift_t = torch.repeat_interleave(shift, sample_lens, dim=0)
            flattened_x = self.norm(flattened_x) * (1 + scale_t) + shift_t
            x = flattened_x.view(x.shape)
        return x


@dataclass
class MageFlowParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    num_heads: int
    depth: int
    axes_dim: list
    checkpoint: bool
    patch_size: int = 1


class MageFlow(nn.Module):
    def __init__(self, params: MageFlowParams):
        super().__init__()
        self.params = params
        self.gradient_checkpointing = bool(params.checkpoint)
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.inner_dim = params.hidden_size  # num_attention_heads * attention_head_dim
        self.axes_dim = params.axes_dim
        self.num_attention_heads = params.num_heads
        self.attention_head_dim = self.inner_dim // self.num_attention_heads
        self.patch_size = params.patch_size
        assert sum(self.axes_dim) == self.attention_head_dim

        self.pos_embed = MageFlowEmbedRope(
            theta=10000, axes_dim=self.axes_dim, scale_rope=True
        )
        self.img_in = nn.Linear(self.in_channels, self.inner_dim)
        self.txt_norm = RMSNorm(params.context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(params.context_in_dim, self.inner_dim)

        self.time_text_embed = MageFlowTimestepProjEmbeddings(
            embedding_dim=self.inner_dim
        )

        self.transformer_blocks = nn.ModuleList(
            [
                MageFlowTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                )
                for _ in range(params.depth)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
        )

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def reset_rope(self):
        """Rebuild the RoPE frequency tables. Needed after a meta-device init
        (the tables are plain tensors, not buffers, so ``load_state_dict`` /
        ``assign=True`` never materializes them)."""
        self.pos_embed = MageFlowEmbedRope(
            theta=10000, axes_dim=self.axes_dim, scale_rope=True
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        timesteps: Tensor,
        img_shapes=None,
        img_cu_seqlens: Tensor = None,
        txt_cu_seqlens: Tensor = None,
        attention_kwargs: dict = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # Prepare vision RoPE (msrope); text tokens are not rotated.
        ms_pe = self.pos_embed(img_shapes, device=img.device)

        img = self.img_in(img)
        txt = self.txt_norm(txt)

        timesteps = timesteps.to(img.dtype)
        temb = self.time_text_embed(timesteps, img)

        txt = self.txt_in(txt)
        # The reference adds a zero "text vector" to temb (vec_type is null in
        # the released config); kept for parity with the reference forward.
        txt_vec = torch.zeros(
            txt.shape[0], self.inner_dim, dtype=txt.dtype, device=txt.device
        )
        temb = temb + txt_vec

        attention_kwargs = attention_kwargs or {}

        for block in self.transformer_blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                txt, img = checkpoint(
                    block,
                    img,  # hidden_states
                    txt,  # encoder_hidden_states
                    temb,  # temb
                    ms_pe,  # image_rotary_emb
                    txt_cu_seqlens,  # txt_cu_lens
                    img_cu_seqlens,  # img_cu_lens
                    use_reentrant=False,
                )
            else:
                txt, img = block(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    txt_cu_lens=txt_cu_seqlens,
                    img_cu_lens=img_cu_seqlens,
                    temb=temb,
                    image_rotary_emb=ms_pe,
                    joint_attention_kwargs=attention_kwargs,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        img = self.norm_out(img, temb, cu_seqlens=img_cu_seqlens)
        img = self.proj_out(img)
        return img
