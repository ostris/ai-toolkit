# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for AI Toolkit by Ostris
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# https://raw.githubusercontent.com/facebookresearch/sapiens2/refs/heads/main/sapiens/backbones/standalone/sapiens2.py

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------------------
def to_2tuple(x):
    if isinstance(x, (str, bytes)):
        return (x, x)
    if isinstance(x, Sequence):
        x = tuple(x)
        if len(x) == 2:
            return x
        raise ValueError("Expected scalar or length-2 iterable")
    return (x, x)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype or torch.float32  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=self.dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}
        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(
                -self.shift_coords, self.shift_coords
            )
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        )  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2
                * torch.arange(self.D_head // 4, device=device, dtype=dtype)
                / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


# -------------------------------------------------------------------------------
class Tokenizer(nn.Module):
    """Stacked window self‑attention that emits one token per window
    by re‑using TransformerEncoderLayer blocks."""

    def __init__(
        self,
        embed_dims: int,
        window_size: int = 4,
        num_heads: int = 4,
        num_tokenizer_layers: int = 1,
        qkv_bias: bool = True,
        use_qk_norm: bool = False,
        chunk_size: int = 1024,  # max windows per chunk
    ):
        super().__init__()
        self.ws = window_size
        self.chunk_size = chunk_size

        # local absolute positional embeddings for [CLS] + patch tokens
        self.local_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + window_size * window_size, embed_dims)
        )
        trunc_normal_(self.local_pos_embed, std=0.02)

        # build N identical TransformerEncoderLayer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer2(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=embed_dims * 4,  # standard FFN size
                    qkv_bias=qkv_bias,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(num_tokenizer_layers)
            ]
        )

        # shared CLS token for pooling
        self.w_cls = nn.Parameter(torch.zeros(1, 1, embed_dims))
        trunc_normal_(self.w_cls, std=0.02)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Args:
           x  : B, N, C   (N = H*W)
           hw : (H, W) before reduction
        Returns:
           x_ : B, (H/ws)*(W/ws), C
           hw_: (H/ws, W/ws)
        """
        B, N, C = x.shape
        H, W = hw
        ws = self.ws
        assert H % ws == 0 and W % ws == 0, (
            f"Image size {H}×{W} must be divisible by window {ws}."
        )

        # reshape tokens → non‑overlapping windows
        x = x.view(B, H, W, C)

        ph, pw = H // ws, W // ws  ## ints in eager mode
        ph, pw = int(ph), int(pw)  ## ints in scripting mode
        x = x.view(B, ph, ws, pw, ws, C)  # B, H/ws, ws, W/ws, ws, C
        x = x.permute(0, 1, 3, 2, 4, 5)  # B, H/ws, W/ws, ws, ws, C
        x = x.contiguous().view(B * ph * pw, ws * ws, C)  # (B*H/ws*W/ws), ws², C))

        total_windows = x.size(0)
        chunk_size = int(min(self.chunk_size, total_windows))
        token_out = x.new_empty(total_windows, C)

        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing

        def _run_blocks(t: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                t = blk(t)
            return t

        for i in range(0, total_windows, chunk_size):
            chunk = x[i : i + chunk_size]  # (m, ws², C)
            m = chunk.size(0)
            cls = self.w_cls.expand(m, -1, -1)  # (m, 1, C)
            chunk = torch.cat([cls, chunk], dim=1)  # (m, 1+ws², C)
            chunk = chunk + self.local_pos_embed  # add local PE

            if use_ckpt:
                chunk = checkpoint(_run_blocks, chunk, use_reentrant=False)
            else:
                chunk = _run_blocks(chunk)

            token_out[i : i + m] = chunk[:, 0]  # take CLS out

        token = token_out.view(B, ph * pw, C)  # (B, (H/ws)*(W
        return token, (ph, pw)


# -------------------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        num_kv_heads=None,
        input_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
        qk_scale=None,
        proj_bias=True,
        use_qk_norm=True,
        v_shortcut=False,
        layer_scale_init_value=0.0,
    ):
        super().__init__()
        # Core dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            "num_kv_heads must divide num_heads"
        )
        self.head_dim = embed_dims // num_heads
        self.input_dims = input_dims or embed_dims
        # Features
        self.attn_drop = attn_drop
        self.v_shortcut = v_shortcut
        self.use_qk_norm = use_qk_norm

        # Attention operation selection
        if qk_scale is not None:
            scale = qk_scale
        else:
            scale = self.head_dim**-0.5

        assert qk_scale is None, "qk_scale is not supported"
        self.attn_op = F.scaled_dot_product_attention

        # Q/K/V projections
        self.wq = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.wk = nn.Linear(
            self.input_dims, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.wv = nn.Linear(
            self.input_dims, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        # Output projection + dropout
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Optional LayerScale
        if layer_scale_init_value > 0:
            self.gamma = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma = nn.Identity()

    def apply_rope(
        self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]  ## extra tokens
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = self._rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = self._rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def _rope_rotate_half(self, x: Tensor) -> Tensor:
        # x:   [ x0  x1  x2  x3  x4  x5]
        # out: [-x3 -x4 -x5  x0  x1  x2]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rope_apply(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
        # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
        # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
        return (x * cos) + (self._rope_rotate_half(x) * sin)

    def forward(self, x, rope=None):
        B, N, _ = x.shape
        # Q: (B, N, num_heads, head_dim)
        q = self.wq(x).view(B, N, self.num_heads, self.head_dim)
        # K/V: (B, N, num_kv_heads, head_dim)
        k = self.wk(x).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(B, N, self.num_kv_heads, self.head_dim)

        # (B, heads, N, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Repeat KV heads if group ratio >1
        if self.num_kv_heads != self.num_heads:
            factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(factor, dim=1)
            v = v.repeat_interleave(factor, dim=1)

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        # Scaled dot-product attention
        attn_out = self.attn_op(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )  # (B, num_heads, N, head_dim)

        # Merge heads -> (B, N, embed_dims)
        out = attn_out.permute(0, 2, 1, 3).reshape(B, N, self.embed_dims)

        # Output projection + drop + layer scale
        out = self.proj(out)
        out = self.gamma(self.proj_drop(out))

        # Optional V-shortcut (only when MQA)
        if self.v_shortcut and self.num_kv_heads == 1:
            raise NotImplementedError
        return out


# -------------------------------------------------------------------------------
class TransformerEncoderLayer2(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        num_kv_heads=None,
        feedforward_channels=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale_init_value=0.0,
        use_qk_norm=True,
        qkv_bias=True,
    ):
        super(TransformerEncoderLayer2, self).__init__()

        self.embed_dims = embed_dims
        self.ln1 = nn.RMSNorm(self.embed_dims, eps=1e-6)
        self.attn = GroupedQueryAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value,
            use_qk_norm=use_qk_norm,
        )

        self.ln2 = nn.RMSNorm(self.embed_dims, eps=1e-6)
        self.ffn = SwiGLUFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
        )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln1(x), rope=rope)
        x = self.ffn(self.ln2(x), identity=x)
        return x


##-----------------------------------
class Sapiens2(nn.Module):
    arch_zoo = {
        **dict.fromkeys(
            ["sapiens2_0.1b"],
            {
                "embed_dims": 768,
                "num_layers": 12,
                "num_heads": 12,
                "feedforward_channels": 768 * 4,
                "num_tokenizer_layers": 2,
            },
        ),
        **dict.fromkeys(
            ["sapiens2_0.4b"],
            {
                "embed_dims": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "feedforward_channels": 1024 * 4,
                "num_tokenizer_layers": 2,
            },
        ),
        **dict.fromkeys(
            ["sapiens2_0.8b"],
            {
                "embed_dims": 1280,
                "num_layers": 32,
                "num_heads": 16,
                "feedforward_channels": 1280 * 4,
                "num_tokenizer_layers": 3,
            },
        ),
        **dict.fromkeys(
            ["sapiens2_1b"],
            {
                "embed_dims": 1536,
                "num_layers": 40,
                "num_heads": 24,
                "feedforward_channels": 1536 * 4,
                "num_tokenizer_layers": 4,
            },
        ),
        **dict.fromkeys(
            ["sapiens2_5b"],
            {
                "embed_dims": 2432,
                "num_layers": 56,
                "num_heads": 32,
                "feedforward_channels": 2432 * 4,
                "num_tokenizer_layers": 6,
            },
        ),
    }

    num_extra_tokens = 1  # class token
    OUT_TYPES = {"raw", "cls_token", "featmap"}
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        arch="sapiens2_1b",
        img_size=(1024, 768),
        patch_size=16,
        in_channels=3,
        out_indices=-1,
        drop_rate=0.0,
        window_size=4,
        use_tokenizer=False,  ## 4k resolution
        use_qk_norm=True,
        qkv_bias=True,
        final_norm=True,
        out_type="raw",
        with_cls_token=True,
        layer_scale_init_value=1e-4,  ## non zero init to activate layerscale
        frozen_stages=-1,
        patch_cfg=dict(),
        layer_cfgs=dict(),
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        n_storage_tokens: int = 8,
    ):
        super().__init__()

        arch = arch.lower()
        assert arch in set(self.arch_zoo), (
            f"Arch {arch} is not in default archs {set(self.arch_zoo)}"
        )
        self.arch_settings = self.arch_zoo[arch]

        self.embed_dims = self.arch_settings["embed_dims"]
        self.num_layers = self.arch_settings["num_layers"]
        self.patch_size = patch_size

        self.window_size = window_size
        img_size = to_2tuple(img_size)
        encoder_img_size = (
            (img_size[0] // window_size, img_size[1] // window_size)
            if use_tokenizer
            else img_size
        )
        self.img_size = to_2tuple(encoder_img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=self.img_size,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        self.rope_embed = RopePositionEmbedding(
            embed_dim=self.embed_dims,
            num_heads=self.arch_settings["num_heads"],
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=torch.bfloat16 if pos_embed_rope_dtype == "bf16" else torch.float32,
        )

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(
                f"Unsupported `out_type` {out_type}, please "
                f"choose from {self.OUT_TYPES}"
            )
        self.out_type = out_type

        if use_tokenizer == True:
            self.tokenizer = Tokenizer(
                embed_dims=self.embed_dims,
                window_size=self.window_size,
                num_heads=self.arch_settings["num_heads"],
                num_tokenizer_layers=self.arch_settings["num_tokenizer_layers"],
                qkv_bias=True,
                use_qk_norm=False,
            )
        else:
            self.tokenizer = None

        # Set cls + storage tokens
        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        elif out_type != "cls_token":
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError('with_cls_token must be True when `out_type="cls_token"`.')

        ## registers
        self.n_storage_tokens = int(n_storage_tokens)
        self.storage_tokens = (
            nn.Parameter(torch.zeros(1, self.n_storage_tokens, self.embed_dims))
            if self.n_storage_tokens > 0
            else None
        )
        # how many non-patch tokens are at the front
        self.num_extra_tokens = (
            1 if self.cls_token is not None else 0
        ) + self.n_storage_tokens

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), (
            f'"out_indices" must by a sequence or int, get {type(out_indices)} instead.'
        )
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, (
                f"Invalid out_indices {index}"
            )
        self.out_indices = out_indices

        self.blocks = nn.Sequential()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers

        mhsa_early, mhsa_late = 8, 8
        for i in range(self.num_layers):
            if i < mhsa_early or i >= self.num_layers - mhsa_late:
                num_kv_heads = None  ## use MHSA
            else:
                num_kv_heads = self.arch_settings["num_heads"] // 2  # Use GQA

            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings["num_heads"],
                num_kv_heads=num_kv_heads,
                feedforward_channels=self.arch_settings["feedforward_channels"],
                use_qk_norm=use_qk_norm,
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                qkv_bias=qkv_bias,
            )
            _layer_cfg.update(layer_cfgs[i])
            self.blocks.append(TransformerEncoderLayer2(**_layer_cfg))

        self.frozen_stages = frozen_stages

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.RMSNorm(self.embed_dims, eps=1e-6)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

        ## load init weights
        self.init_weights()

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, enable=True):
        self.gradient_checkpointing = enable
        if self.tokenizer is not None:
            self.tokenizer.gradient_checkpointing = enable

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def init_weights(self):
        # Initialize class token and storagr token embeddings
        if self.with_cls_token:
            trunc_normal_(self.cls_token, std=0.02)

        if self.storage_tokens is not None:
            trunc_normal_(self.storage_tokens, std=0.02)

        # Apply custom initialization to all submodules
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use a truncated normal distribution for linear layer weights
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            # Initialize normalization layers to act as an identity function
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            # Initialize conv layer weights like linear layers
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _freeze_stages(self):
        ## freeze tokenizer
        if self.frozen_stages >= 1 and self.tokenizer is not None:
            self.tokenizer.eval()
            for param in self.tokenizer.parameters():
                param.requires_grad = False

        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        if self.storage_tokens is not None:
            self.storage_tokens.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        # freeze the last layer norm
        if self.frozen_stages == len(self.blocks):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]

        x, patch_resolution = self.patch_embed(x)  # (B, 256*256, C)
        if self.tokenizer is not None:
            x, patch_resolution = self.tokenizer(x, patch_resolution)

        # prepend [CLS] and storage tokens
        prepend = []
        if self.cls_token is not None:
            prepend.append(self.cls_token.expand(B, -1, -1))
        if self.storage_tokens is not None:
            prepend.append(self.storage_tokens.expand(B, -1, -1))
        if len(prepend) > 0:
            x = torch.cat(prepend + [x], dim=1)

        rope_sincos = self.rope_embed(H=patch_resolution[0], W=patch_resolution[1])
        outs = []
        for i, layer in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, rope_sincos, use_reentrant=False)
            else:
                x = layer(x, rope=rope_sincos)

            if i == len(self.blocks) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == "raw":
            return x
        if self.out_type == "cls_token":
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens :]
        if self.out_type == "featmap":
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)

    @property
    def norm1(self):
        return self.ln1


# ----------------------------------------------------------------------------
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        inplace: bool = False,
        data_format: str = "channels_last",
        scale: float = 1e-5,
    ):
        super().__init__()
        assert data_format in (
            "channels_last",
            "channels_first",
        ), "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x) -> torch.Tensor:
        if self.data_format == "channels_first":
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)


# ----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        kernel_size=16,
        stride=16,
        padding="corner",
        dilation=1,
        bias=True,
        input_size=None,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        padding = 0
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            h_out = (
                input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
            ) // stride[0] + 1
            w_out = (
                input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
            ) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


# ----------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer.
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """  # noqa

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.0,
        bias: bool = True,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)
        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(dim=embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

        self.add_identity = add_identity

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        out = self.w3(hidden)
        out = self.gamma2(out)

        if self.out_dims != self.embed_dims or not self.add_identity:
            # due to the dimension inconsistence or user setting
            # not to apply residual operation
            return out

        if identity is None:
            identity = x
        return identity + out


# ----------------------------------------------------------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(tensors_0_1: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a (B, C, H, W) RGB tensor in [0, 1]."""
    mean = torch.as_tensor(
        _IMAGENET_MEAN, dtype=tensors_0_1.dtype, device=tensors_0_1.device
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        _IMAGENET_STD, dtype=tensors_0_1.dtype, device=tensors_0_1.device
    ).view(1, 3, 1, 1)
    return (tensors_0_1 - mean) / std
