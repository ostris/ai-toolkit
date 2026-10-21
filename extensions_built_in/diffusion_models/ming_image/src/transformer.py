# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ming-Image's diffusion transformer: the Z-Image 6B DiT with Ming's changes.

Vendored from diffusers' `transformer_z_image.py` (Apache 2.0) because the
Ming checkpoint (`_class_name: DiffusionTransformer`) diverges from Z-Image in
ways the diffusers class cannot express:

  - `alignment_padding_mode="zero_masked"`: the reference pads sequences to a
    multiple of 32 with zeros that are masked out of every attention and
    dropped from the output. Those slots cannot influence anything, so this
    port never materializes them: only their share of the position grid
    survives (the image block starts after the padded caption length). With
    nothing to mask, equal-length batches run unmasked (flash) attention.
    There are no learned `x_pad_token` / `cap_pad_token`.
  - a second caption stream, `cap_feats_2`: the direct-VLM condition already
    lives in model width, so it bypasses `cap_embedder` and is appended to the
    embedded query-token captions before the context refiner.
  - `ref_x`: a reference latent concatenated along the frame axis (F=2) for
    editing; with `multi_frame_output=False` only the first frame is returned.

`OstrisModelMixin` gives it the toolkit's universal load/quantize/offload
path. The omni/siglip and controlnet branches of the upstream file are gone.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.torch_utils import maybe_allow_in_graph

from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.models.v2.diffusion_models.z_image import ZImageTransformer2DModel

from .checkpoints import BASE_REPO, COMFY_REPO, COMFY_TRANSFORMER_FILES, comfy_weight_names

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

LEARNED_PADDING = "learned"
ZERO_MASKED_PADDING = "zero_masked"


class RMSNorm(nn.Module):
    """Same parameters as diffusers' RMSNorm, on torch's fused kernel."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.autocast(device_type=t.device.type, enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
                / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        compute_dtype = getattr(self.mlp[0], "compute_dtype", None)
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        elif compute_dtype is not None:
            t_freq = t_freq.to(compute_dtype)
        return self.mlp(t_freq)


class ZSingleStreamAttnProcessor:
    """Single-stream attention with complex RoPE, on the diffusers Attention class."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def apply_rotary_emb(x_in: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=x_in.device.type, enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs = freqs.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # (batch, seq_len) -> (batch, 1, 1, seq_len), broadcast over heads and queries
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):
    def __init__(self, layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )
        self.attention.norm_q = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else None
        self.attention.norm_k = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else None
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            )

    def forward(self, x, attn_mask, freqs_cis, adaln_input=None):
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        return self.linear(x)


class RopeEmbedder:
    def __init__(self, theta=256.0, axes_dims=(16, 56, 56), axes_lens=(64, 128, 128)):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2 and ids.shape[-1] == len(self.axes_dims)
        device = ids.device
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


class MingImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, OstrisModelMixin
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]
    _repeated_blocks = ["ZImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]

    aitk_subfolder = "transformer"
    aitk_config_repo = BASE_REPO
    aitk_comfy_repo = COMFY_REPO
    aitk_comfy_weight_names = comfy_weight_names(COMFY_TRANSFORMER_FILES)

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        siglip_feat_dim=None,  # accepted for config compatibility, unused
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=(32, 48, 48),
        axes_lens=(20480, 512, 512),
        alignment_padding_mode=ZERO_MASKED_PADDING,
        multi_frame_output=False,
        # the trainer's dynamic-shift schedule reads config.patch_size to count
        # image tokens; latents are patchified 2x2 (all_patch_size[0])
        patch_size=2,
    ) -> None:
        super().__init__()
        if alignment_padding_mode not in (LEARNED_PADDING, ZERO_MASKED_PADDING):
            raise ValueError(f"unsupported alignment_padding_mode {alignment_padding_mode!r}")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = tuple(all_patch_size)
        self.all_f_patch_size = tuple(all_f_patch_size)
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.alignment_padding_mode = alignment_padding_mode
        self.multi_frame_output = multi_frame_output
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder, all_final_layer = {}, {}
        for p, fp in zip(all_patch_size, all_f_patch_size):
            all_x_embedder[f"{p}-{fp}"] = nn.Linear(fp * p * p * in_channels, dim, bias=True)
            all_final_layer[f"{p}-{fp}"] = FinalLayer(dim, p * p * fp * self.out_channels)
        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(1000 + i, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True)
                for i in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
                for i in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True)
        )

        if alignment_padding_mode == LEARNED_PADDING:
            self.x_pad_token = nn.Parameter(torch.zeros((1, dim)))
            self.cap_pad_token = nn.Parameter(torch.zeros((1, dim)))
        else:
            self.register_parameter("x_pad_token", None)
            self.register_parameter("cap_pad_token", None)

        self.layers = nn.ModuleList(
            [ZImageTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, qk_norm) for i in range(n_layers)]
        )
        assert dim // n_heads == sum(axes_dims)
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=self.axes_dims, axes_lens=self.axes_lens)

    # ------------------------------------------------------------------
    # toolkit hooks
    # ------------------------------------------------------------------
    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]

    def get_offload_ignore_modules(self):
        return [p for p in (self.x_pad_token, self.cap_pad_token) if p is not None]

    @classmethod
    def get_quantization_exclude_modules(cls):
        # same precision-sensitive set as Z-Image: the timestep embedder feeds
        # every adaLN, the caption / latent projections and the final layers
        return ["t_embedder*", "cap_embedder*", "all_x_embedder*", "all_final_layer*"]

    # ------------------------------------------------------------------
    # single-file layout: the ComfyUI repack's int8 file uses Comfy's Z-Image
    # names (fused attention.qkv, x_embedder / final_layer at the root); its
    # bf16 file and the vendor's are diffusers-keyed and pass through. Saves
    # use the Comfy names so ComfyUI and this class both load the result.
    # ------------------------------------------------------------------
    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        return ZImageTransformer2DModel.convert_state_dict_on_load(state_dict)

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        return ZImageTransformer2DModel.convert_state_dict_on_save(state_dict)

    # ------------------------------------------------------------------
    # sequence assembly
    # ------------------------------------------------------------------
    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def _positions(self, ori_len: int, pad_len: int, grid_size, grid_start, device):
        """Position ids for one sequence: the grid, then `pad_len` pad slots
        at the origin (learned padding only)."""
        pos_ids = self.create_coordinate_grid(size=grid_size, start=grid_start, device=device).flatten(0, 2)
        if pad_len > 0:
            pad_pos = self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
            pos_ids = torch.cat([pos_ids, pad_pos.flatten(0, 2).repeat(pad_len, 1)], dim=0)
        return pos_ids

    def _pad_feat(self, feat: torch.Tensor, pad_len: int) -> torch.Tensor:
        # learned padding repeats the last feature; the pad token replaces it
        if pad_len == 0:
            return feat
        return torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)

    def _apply_pad_token(self, feats: torch.Tensor, pad_lens: List[int], seqlens: List[int], pad_token):
        if pad_token is None or not any(pad_lens):
            return feats
        pad_mask = torch.cat(
            [
                torch.arange(n, device=feats.device) >= n - pad
                for n, pad in zip(seqlens, pad_lens)
            ]
        )
        fill = pad_token.expand(feats.shape[0], -1).to(feats.dtype)
        return torch.where(pad_mask.unsqueeze(1).expand_as(feats), fill, feats)

    @staticmethod
    def _length_mask(seqlens: List[int], device) -> Optional[torch.Tensor]:
        """Key mask for right-padded sequences, None when nothing is padded
        (decided on the host: no device sync)."""
        max_len = max(seqlens)
        if all(n == max_len for n in seqlens):
            return None
        mask = torch.zeros((len(seqlens), max_len), dtype=torch.bool, device=device)
        for i, n in enumerate(seqlens):
            mask[i, :n] = True
        return mask

    def _batch(self, feats: List[torch.Tensor], pos_ids: List[torch.Tensor], device):
        """Right-pad per-item sequences into a batch; returns the batch, its
        rope, a key mask (None when lengths agree) and the item lengths."""
        seqlens = [len(f) for f in feats]
        freqs = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))
        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs = pad_sequence(freqs, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]
        return feats, freqs, self._length_mask(seqlens, device), seqlens

    def unpatchify(self, x: List[torch.Tensor], sizes: List[Tuple[int, int, int]], patch_size, f_patch_size):
        pH = pW = patch_size
        pF = f_patch_size
        out = []
        for item, (frames, height, width) in zip(x, sizes):
            ori_len = (frames // pF) * (height // pH) * (width // pW)
            item = (
                item[:ori_len]
                .view(frames // pF, height // pH, width // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, frames, height, width)
            )
            if not self.multi_frame_output:
                item = item[:, :1]
            out.append(item)
        return out

    def forward(
        self,
        x: List[torch.Tensor],  # per item (C, F, H, W) noisy latents
        t: torch.Tensor,  # (B,) in the model's convention: 0 = noise, 1 = clean
        cap_feats: List[torch.Tensor],  # per item (L1, cap_feat_dim) query-token conditions
        cap_feats_2: Optional[List[torch.Tensor]] = None,  # per item (L2, dim) direct-VLM conditions
        ref_x: Optional[List[Optional[torch.Tensor]]] = None,  # per item (C, 1, H, W) reference latent
        patch_size: int = 2,
        f_patch_size: int = 1,
        return_dict: bool = True,
    ):
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        bsz = len(x)
        device = x[0].device
        if cap_feats_2 is None:
            cap_feats_2 = [None] * bsz
        if ref_x is None:
            ref_x = [None] * bsz

        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])
        learned_pad = self.alignment_padding_mode == LEARNED_PADDING

        # ---- per-item layout: captions first (positions 1..), then the image.
        # The 32-alignment pad is materialized only under learned padding; the
        # zero_masked reference masks those slots out of everything, so here
        # they exist only as a gap in the position grid.
        cap_in, cap_direct, cap_pad_lens, cap_pos_ids = [], [], [], []
        img_feats, img_sizes, img_pos_ids, img_pad_lens = [], [], [], []
        for image, cap, direct, ref in zip(x, cap_feats, cap_feats_2, ref_x):
            cap_len = len(cap) + (len(direct) if direct is not None else 0)
            cap_pad = (-cap_len) % SEQ_MULTI_OF
            n_pad = cap_pad if learned_pad else 0
            cap_in.append(cap)
            cap_direct.append(direct)
            cap_pad_lens.append(n_pad)
            cap_pos_ids.append(self._positions(cap_len, 0, (cap_len + n_pad, 1, 1), (1, 0, 0), device))

            if ref is not None:
                image = torch.cat([image, ref.to(image.dtype)], dim=1)
            C, Fr, H, W = image.shape
            F_t, H_t, W_t = Fr // f_patch_size, H // patch_size, W // patch_size
            patches = (
                image.view(C, F_t, f_patch_size, H_t, patch_size, W_t, patch_size)
                .permute(1, 3, 5, 2, 4, 6, 0)
                .reshape(F_t * H_t * W_t, f_patch_size * patch_size * patch_size * C)
            )
            img_len = patches.shape[0]
            n_pad = (-img_len) % SEQ_MULTI_OF if learned_pad else 0
            img_feats.append(self._pad_feat(patches, n_pad))
            img_sizes.append((Fr, H, W))
            img_pad_lens.append(n_pad)
            img_pos_ids.append(
                self._positions(img_len, n_pad, (F_t, H_t, W_t), (cap_len + cap_pad + 1, 0, 0), device)
            )

        # ---- image tokens: embed, pad, refine
        x_seqlens = [len(f) for f in img_feats]
        x_emb = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(img_feats, dim=0))
        x_emb = self._apply_pad_token(x_emb, img_pad_lens, x_seqlens, self.x_pad_token)
        x_emb, x_freqs, x_mask, _ = self._batch(list(x_emb.split(x_seqlens, dim=0)), img_pos_ids, device)
        for layer in self.noise_refiner:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x_emb = self._gradient_checkpointing_func(layer, x_emb, x_mask, x_freqs, adaln_input)
            else:
                x_emb = layer(x_emb, x_mask, x_freqs, adaln_input)

        # ---- caption tokens: the query stream through cap_embedder, the direct
        # stream appended as-is
        cap_lens_1 = [len(c) for c in cap_in]
        cap_emb = list(self.cap_embedder(torch.cat(cap_in, dim=0)).split(cap_lens_1, dim=0))
        cap_items = []
        for emb, direct, pad in zip(cap_emb, cap_direct, cap_pad_lens):
            if direct is not None:
                emb = torch.cat([emb, direct.to(emb.dtype)], dim=0)
            cap_items.append(self._pad_feat(emb, pad))
        cap_seqlens = [len(c) for c in cap_items]
        cap_cat = self._apply_pad_token(torch.cat(cap_items, dim=0), cap_pad_lens, cap_seqlens, self.cap_pad_token)
        cap_batch, cap_freqs, cap_mask, _ = self._batch(list(cap_cat.split(cap_seqlens, dim=0)), cap_pos_ids, device)
        for layer in self.context_refiner:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                cap_batch = self._gradient_checkpointing_func(layer, cap_batch, cap_mask, cap_freqs)
            else:
                cap_batch = layer(cap_batch, cap_mask, cap_freqs)

        # ---- unified sequence: [image, caption] per item
        unified, unified_freqs = [], []
        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]
            unified.append(torch.cat([x_emb[i][:x_len], cap_batch[i][:cap_len]]))
            unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))
        unified_mask = self._length_mask([len(u) for u in unified], device)
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                unified = self._gradient_checkpointing_func(layer, unified, unified_mask, unified_freqs, adaln_input)
            else:
                unified = layer(unified, unified_mask, unified_freqs, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        out = self.unpatchify(list(unified.unbind(dim=0)), img_sizes, patch_size, f_patch_size)
        if not return_dict:
            return (out,)
        return Transformer2DModelOutput(sample=out)
