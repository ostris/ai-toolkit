"""Iris-3B pixel-space diffusion transformer, vendored from
https://github.com/speridlabs/iris-3b (``src/iris3b/models`` + ``src/iris3b/nn``).

Two levels:
  - patch stage: a hybrid trunk of ``dual_depth`` dual-stream MM-DiT blocks
    followed by single-stream blocks over 16x16 patch tokens, text-conditioned
    through a layerwise-attention adapter over stacked Qwen3-VL hidden layers;
  - pixel stage: ``pixel.depth`` PiT blocks refine each patch's pixels,
    conditioned on the patch stage's output tokens.
No VAE: input and output are RGB velocities in pixel space.

Deviations from the reference, none of which change the state-dict layout:
  - SDPA only (no fa3/fa4/cudnn backend switch); REPA feature capture dropped.
  - Shared adaLN cores (``model.modulation = shared_*``) are evaluated ONCE per
    forward in ``IrisDiT.forward`` and handed to the blocks, instead of each
    block holding a hidden Python reference to the core Linear. The reference's
    tuple trick bypasses module replacement (quantization / LoRA swap the
    Linear under ``modulation_cores`` and the blocks would keep calling the old
    one), so the core lives only in ``modulation_cores`` here.
  - gradient checkpointing per block via ``enable_gradient_checkpointing()``.
"""

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from toolkit.models.v2._mixin import OstrisModelMixin


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class IrisPixelConfig:
    enabled: bool = True
    depth: int = 4
    hidden_size: int = 16
    attn_hidden_size: int = 1280
    num_heads: int = 10
    mlp_ratio: float = 4.0
    modulation: str = "post"  # pre | post
    abs_pos_embed: bool = True


@dataclass
class IrisConfig:
    """Mirror of the reference ``ModelConfig`` (defaults = Iris-3B). Keys the
    reference config.yaml carries that do not affect the module (attn_backend,
    repa_layer, adaln_zero_init) are accepted and ignored by ``from_dict``."""

    block: str = "single_stream"  # mmdit | single_stream
    dual_depth: int = 8
    final_block_text: str = "keep"  # keep | drop
    hidden_size: int = 2560
    depth: int = 24
    num_heads: int = 20
    num_kv_heads: Optional[int] = 5
    gated_attention: bool = True
    sandwich_norm: bool = True
    patch_size: int = 16
    in_channels: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_norm: bool = True
    norm_eps: float = 1e-6
    modulation: str = "shared_bias"  # per_block | shared_lowrank | shared_bias
    modulation_rank: int = 64
    timestep_max_period: float = 10.0
    rope_theta: float = 10000.0
    rope_scale: float = 16.0
    rope_aspect: str = "isotropic"  # square | isotropic
    rope_frame_pairs: int = 0
    rope_frame_theta: float = 10.0
    text_rope: bool = True
    text_rope_theta: float = 10000.0
    text_abs_pos_embed: bool = True
    text_dim: int = 2560
    text_len: int = 300
    text_adapter: str = "lap_blocks2"  # linear | blocks2 | lap_blocks2
    text_lap_num_layers: int = 12
    text_lap_num_heads: int = 32
    text_lap_mlp_ratio: float = 1.3
    pixel: IrisPixelConfig = field(default_factory=IrisPixelConfig)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "IrisConfig":
        raw = dict(raw or {})
        pixel_raw = raw.pop("pixel", None) or {}
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        pixel_known = {f.name for f in fields(IrisPixelConfig)}
        kwargs["pixel"] = IrisPixelConfig(
            **{k: v for k, v in pixel_raw.items() if k in pixel_known}
        )
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # BaseModel reads unet.config['in_channels'] / unet.config.patch_size
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def validate(self) -> None:
        if not 0 <= self.dual_depth <= self.depth:
            raise ValueError(
                f"dual_depth must be in [0, depth={self.depth}], got {self.dual_depth}"
            )
        if self.final_block_text not in ("keep", "drop"):
            raise ValueError(f"unknown final_block_text '{self.final_block_text}'")
        if self.block not in ("mmdit", "single_stream"):
            raise ValueError(f"unknown block '{self.block}'")
        if self.modulation not in ("per_block", "shared_lowrank", "shared_bias"):
            raise ValueError(f"unknown modulation '{self.modulation}'")
        if self.text_adapter not in ("linear", "blocks2", "lap_blocks2"):
            raise ValueError(f"unknown text_adapter '{self.text_adapter}'")


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMS normalization computed in fp32, learned gain applied in input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normed.to(x.dtype)


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward with the 2/3 width rule, bias-free."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(2 * int(dim * mlp_ratio) / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeluMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


# ---- rope ----


def rope_2d(
    head_dim: int,
    height: int,
    width: int,
    theta: float = 10000.0,
    scale: float = 16.0,
    aspect: str = "square",
    frame_pairs: int = 0,
    frame_theta: float = 10.0,
    frame_index: int = 0,
) -> torch.Tensor:
    """Complex rotation factors ``[height*width, head_dim//2]`` for a row-major
    grid. Coordinates are normalized to a fixed span ``[0, scale]`` so any grid
    size covers the same angular range (``isotropic`` shares one step across
    both axes so aspect ratio survives)."""
    n_pairs = head_dim // 4
    if not 0 <= frame_pairs < n_pairs:
        raise ValueError(f"frame_pairs must be in [0, {n_pairs}), got {frame_pairs}")
    xy_pairs = n_pairs - frame_pairs
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 4)[:n_pairs].float() / head_dim)
    freqs = freqs[:xy_pairs]
    if aspect == "square":
        x_pos = torch.linspace(0, scale, width)
        y_pos = torch.linspace(0, scale, height)
    elif aspect == "isotropic":
        step = scale / max(max(height, width) - 1, 1)
        x_pos = torch.arange(width).float() * step
        y_pos = torch.arange(height).float() * step
    else:
        raise ValueError(f"unknown rope aspect mode '{aspect}'")
    x_ang = torch.outer(x_pos, freqs)
    y_ang = torch.outer(y_pos, freqs)
    x_cis = torch.polar(torch.ones_like(x_ang), x_ang)
    y_cis = torch.polar(torch.ones_like(y_ang), y_ang)
    x_grid = x_cis[None, :, :].expand(height, width, -1)
    y_grid = y_cis[:, None, :].expand(height, width, -1)
    cis = torch.stack([x_grid, y_grid], dim=-1).reshape(height * width, 2 * xy_pairs)
    if frame_pairs:
        n_slots = 2 * frame_pairs
        f_freqs = 1.0 / frame_theta ** (torch.arange(n_slots).float() / n_slots)
        f_cis = torch.polar(torch.ones(n_slots), float(frame_index) * f_freqs)
        cis = torch.cat([cis, f_cis[None, :].expand(height * width, -1)], dim=-1)
    return cis


def rope_1d(head_dim: int, length: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim)
    ang = torch.outer(torch.arange(length).float(), freqs)
    return torch.polar(torch.ones_like(ang), ang)


def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    """Rotate ``q``/``k`` of shape ``[B, N, H, head_dim]``; complex math in fp32."""
    q_c = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_c = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    fc = freqs_cis[None, :, None, :]
    q_out = torch.view_as_real(q_c * fc).flatten(3)
    k_out = torch.view_as_real(k_c * fc).flatten(3)
    return q_out.to(q.dtype), k_out.to(k.dtype)


# ---- attention ----


def scaled_dot_product(q, k, v, attn_mask=None, enable_gqa: bool = False):
    """Attention over ``[B, heads, N, head_dim]`` tensors (SDPA)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)


class SelfAttention(nn.Module):
    """Multi-head self-attention with per-head QK RMSNorm and RoPE."""

    def __init__(self, dim, num_heads, qkv_bias=False, qk_norm=True, norm_eps=1e-6):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, rope=None, attn_mask=None):
        batch, n, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, n, 3, self.num_heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, k, rope)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = scaled_dot_product(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(batch, n, dim))


class JointAttention(nn.Module):
    """MM-DiT joint attention: separate stream projections, one softmax over
    the text-first concatenation. GQA unfuses QKV into Q/K/V linears."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_norm=True,
        norm_eps=1e-6,
        num_kv_heads=None,
        text_out=True,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by heads {num_heads}")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError(f"query heads {num_heads} not divisible by KV heads {num_kv_heads}")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if num_kv_heads == num_heads:
            self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.q_proj_x = self.k_proj_x = self.v_proj_x = None
            self.q_proj_y = self.k_proj_y = self.v_proj_y = None
        else:
            kv_dim = num_kv_heads * self.head_dim
            self.qkv_x = self.qkv_y = None
            self.q_proj_x = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj_x = nn.Linear(dim, kv_dim, bias=qkv_bias)
            self.v_proj_x = nn.Linear(dim, kv_dim, bias=qkv_bias)
            self.q_proj_y = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj_y = nn.Linear(dim, kv_dim, bias=qkv_bias)
            self.v_proj_y = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.q_norm_x = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm_x = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.q_norm_y = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm_y = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.proj_x = nn.Linear(dim, dim, bias=True)
        self.proj_y = nn.Linear(dim, dim, bias=True) if text_out else None

    def _project(self, tokens, stream):
        batch, n_tokens, _ = tokens.shape
        fused = self.qkv_x if stream == "x" else self.qkv_y
        if fused is not None:
            return fused(tokens).reshape(batch, n_tokens, 3, self.num_heads, self.head_dim).unbind(2)
        if stream == "x":
            q_proj, k_proj, v_proj = self.q_proj_x, self.k_proj_x, self.v_proj_x
        else:
            q_proj, k_proj, v_proj = self.q_proj_y, self.k_proj_y, self.v_proj_y
        q = q_proj(tokens).reshape(batch, n_tokens, self.num_heads, self.head_dim)
        k = k_proj(tokens).reshape(batch, n_tokens, self.num_kv_heads, self.head_dim)
        v = v_proj(tokens).reshape(batch, n_tokens, self.num_kv_heads, self.head_dim)
        return q, k, v

    def forward(self, x, y, rope_img, rope_txt=None, gate_x=None, gate_y=None):
        batch, n_img, dim = x.shape
        n_txt = y.shape[1]
        qx, kx, vx = self._project(x, "x")
        qy, ky, vy = self._project(y, "y")
        qx, kx = self.q_norm_x(qx), self.k_norm_x(kx)
        qy, ky = self.q_norm_y(qy), self.k_norm_y(ky)
        qx, kx = apply_rope(qx, kx, rope_img)
        if rope_txt is not None:
            qy, ky = apply_rope(qy, ky, rope_txt)
        q = torch.cat([qy, qx], dim=1).transpose(1, 2)
        k = torch.cat([ky, kx], dim=1).transpose(1, 2)
        v = torch.cat([vy, vx], dim=1).transpose(1, 2)
        out = scaled_dot_product(q, k, v, enable_gqa=self.num_kv_heads != self.num_heads)
        out = out.transpose(1, 2).reshape(batch, n_txt + n_img, dim)
        out_x = out[:, n_txt:]
        if gate_x is not None:
            out_x = out_x * gate_x
        if self.proj_y is None:
            return self.proj_x(out_x), None
        out_y = out[:, :n_txt]
        if gate_y is not None:
            out_y = out_y * gate_y
        return self.proj_x(out_x), self.proj_y(out_y)


# ---- embedders ----


class TimestepEmbedder(nn.Module):
    """Sinusoidal features + 2-layer SiLU MLP. ``max_period`` is 10 (not
    10000): model time is the shifted flow level scaled to [0, 1000]."""

    def __init__(self, hidden_size: int, freq_dim: int = 256, max_period: float = 10.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        n = self.freq_dim // 2
        k = torch.arange(n, dtype=torch.float32, device=t.device)
        phase = torch.outer(t.float(), torch.exp(-math.log(self.max_period) * k / n))
        return torch.cat([phase.cos(), phase.sin()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t``: [B] model time -> [B, 1, hidden]."""
        batch = t.shape[0]
        emb = self.timestep_embedding(t.reshape(-1))
        emb = self.mlp(emb.to(self.mlp[0].weight.dtype))
        return emb.reshape(batch, -1, emb.shape[-1])


class PatchEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, norm: bool = False, norm_eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size, eps=norm_eps) if norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class TextEmbedder(nn.Module):
    """Linear + RMSNorm text adapter (``text_adapter = linear``)."""

    def __init__(self, text_dim: int, hidden_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, y):
        return self.norm(self.proj(y))


class TextAdapterBlock(nn.Module):
    """Unconditioned pre-norm block: ``x + attn(norm(x))``, ``x + swiglu(norm(x))``."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=True, norm_eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, mlp_ratio)

    def forward(self, y, attn_mask=None):
        y = y + self.attn(self.norm1(y), attn_mask=attn_mask)
        return y + self.mlp(self.norm2(y))


class LayerwiseAttentionBlock(nn.Module):
    """Pre-norm block over one token's stack of encoder-layer states."""

    def __init__(self, dim, num_heads, mlp_ratio=1.3, norm_eps=1e-6):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, qkv_bias=False, qk_norm=False, norm_eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=True),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TransformerTextEmbedder(nn.Module):
    """``Linear`` -> N unconditioned blocks -> RMSNorm. No RoPE (text positions
    come from ``y_pos_embedding`` + the trunk's text RoPE). Pad KEYS are masked
    inside its blocks; every query keeps its own position (mask OR diagonal)
    so an all-pad row cannot produce NaN."""

    def __init__(
        self,
        text_dim,
        hidden_size,
        num_blocks=2,
        num_heads=24,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=True,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.blocks = nn.ModuleList(
            [
                TextAdapterBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_norm=qk_norm, norm_eps=norm_eps,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, y, mask):
        if mask.shape != y.shape[:2]:
            raise ValueError(f"text mask must be {tuple(y.shape[:2])}, got {tuple(mask.shape)}")
        keep = mask.bool()[:, None, None, :]
        eye = torch.eye(y.shape[1], dtype=torch.bool, device=y.device)
        attn_mask = keep | eye
        y = self.proj(y)
        for block in self.blocks:
            y = block(y, attn_mask=attn_mask)
        return self.norm(y)


class LayerwiseTextEmbedder(nn.Module):
    """Aggregate the stacked frozen-encoder layers per token (2 attention blocks
    over the layer axis + learned pooling), then refine across tokens."""

    def __init__(
        self,
        text_dim,
        hidden_size,
        num_layers,
        layer_num_heads,
        layer_mlp_ratio=1.3,
        refiner_num_heads=24,
        refiner_mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=True,
        norm_eps=1e-6,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"text_lap_num_layers must be positive, got {num_layers}")
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.layer_blocks = nn.ModuleList(
            [
                LayerwiseAttentionBlock(text_dim, layer_num_heads, mlp_ratio=layer_mlp_ratio, norm_eps=norm_eps)
                for _ in range(2)
            ]
        )
        self.layer_pool = nn.Linear(num_layers, 1, bias=True)
        self.refiner = TransformerTextEmbedder(
            text_dim,
            hidden_size,
            num_blocks=2,
            num_heads=refiner_num_heads,
            mlp_ratio=refiner_mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )

    def forward(self, y, mask):
        expected = (*mask.shape, self.num_layers, self.text_dim)
        if tuple(y.shape) != expected:
            raise ValueError(f"layerwise text states must be {expected}, got {tuple(y.shape)}")
        batch, tokens, layers, dim = y.shape
        y = y.reshape(batch * tokens, layers, dim)
        for block in self.layer_blocks:
            y = block(y)
        y = self.layer_pool(y.transpose(1, 2)).squeeze(-1)
        y = y.reshape(batch, tokens, dim)
        return self.refiner(y, mask)


def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000.0**omega
    out = torch.outer(pos.reshape(-1).double(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def sincos_pos_embed_2d(embed_dim: int, height: int, width: int) -> torch.Tensor:
    """Fixed 2D sincos table ``[height*width, embed_dim]`` (MAE/DiT grid convention)."""
    grid_h = torch.arange(height, dtype=torch.float32)
    grid_w = torch.arange(width, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, 1, height, width)
    emb = torch.cat([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], dim=1)
    return emb.float()


class PixelEmbedder(nn.Module):
    """Per-pixel linear embedding grouped into per-patch sequences:
    ``[B, C, H, W] -> [B*(H/p)*(W/p), p*p, hidden]``."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int, abs_pos_embed: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.abs_pos_embed = abs_pos_embed
        self.proj = nn.Linear(in_channels, hidden_size, bias=True)
        self._pos_cache: Dict[tuple, torch.Tensor] = {}

    def _pos(self, height, width, device, dtype):
        key = (height, width, str(device), dtype)
        table = self._pos_cache.get(key)
        if table is None:
            table = sincos_pos_embed_2d(self.proj.out_features, height, width)
            table = table.reshape(height, width, -1).to(device=device, dtype=dtype)
            self._pos_cache[key] = table
        return table

    def forward(self, x):
        batch, _, height, width = x.shape
        p = self.patch_size
        h_patches, w_patches = height // p, width // p
        tokens = self.proj(x.permute(0, 2, 3, 1))
        if self.abs_pos_embed:
            tokens = tokens + self._pos(height, width, tokens.device, tokens.dtype)
        tokens = tokens.reshape(batch, h_patches, p, w_patches, p, -1)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5)
        return tokens.reshape(batch * h_patches * w_patches, p * p, -1)


# ---- modulation ----


class BlockModulation(nn.Module):
    """One stream's adaLN projection for a patch block, same parameter names
    as the reference for every mode:

    - ``per_block``: own ``weight``/``bias`` (the reference's ``nn.Linear(D, 6D)``)
    - ``shared_bias``: ``bias`` only; output = shared core output + bias
    - ``shared_lowrank``: ``down`` / ``adaln_up``; output = core + U(V(cond))

    For the shared modes the core output ``shared`` ([B, 1, 6D]) is computed
    once per forward by ``IrisDiT`` from ``modulation_cores`` and passed in.
    """

    def __init__(self, dim: int, mode: str, rank: int = 64):
        super().__init__()
        self.mode = mode
        if mode == "per_block":
            self.weight = nn.Parameter(torch.empty(6 * dim, dim))
            self.bias = nn.Parameter(torch.zeros(6 * dim))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif mode == "shared_bias":
            self.bias = nn.Parameter(torch.zeros(6 * dim))
        elif mode == "shared_lowrank":
            if rank <= 0:
                raise ValueError(f"modulation_rank must be positive, got {rank}")
            self.down = nn.Linear(dim, rank, bias=False)
            self.adaln_up = nn.Linear(rank, 6 * dim, bias=False)
        else:
            raise ValueError(f"unknown modulation mode '{mode}'")

    def forward(self, cond: torch.Tensor, shared: Optional[torch.Tensor]) -> torch.Tensor:
        if self.mode == "per_block":
            return F.linear(cond, self.weight, self.bias)
        if shared is None:
            raise ValueError(f"{self.mode} modulation needs the shared core output")
        if self.mode == "shared_bias":
            return shared + self.bias
        return shared + self.adaln_up(self.down(cond))


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


class MMDiTBlock(nn.Module):
    """Dual-stream block: separate text/image weights, joint attention."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=True,
        norm_eps=1e-6,
        modulation="per_block",
        modulation_rank=64,
        num_kv_heads=None,
        gated_attention=False,
        sandwich_norm=False,
        text_out=True,
    ):
        super().__init__()
        self.norm_x1 = RMSNorm(dim, eps=norm_eps)
        self.norm_x2 = RMSNorm(dim, eps=norm_eps)
        self.norm_y1 = RMSNorm(dim, eps=norm_eps)
        self.norm_y2 = RMSNorm(dim, eps=norm_eps) if text_out else None
        self.attn = JointAttention(
            dim, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_eps=norm_eps,
            num_kv_heads=num_kv_heads, text_out=text_out,
        )
        self.attn_gate_x = nn.Linear(dim, dim, bias=False) if gated_attention else None
        self.attn_gate_y = nn.Linear(dim, dim, bias=False) if gated_attention and text_out else None
        self.attn_post_norm_x = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.attn_post_norm_y = (
            (RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()) if text_out else None
        )
        self.mlp_x = SwiGLU(dim, mlp_ratio)
        self.mlp_y = SwiGLU(dim, mlp_ratio) if text_out else None
        self.mlp_post_norm_x = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.mlp_post_norm_y = (
            (RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()) if text_out else None
        )
        self.adaln_img = BlockModulation(dim, modulation, modulation_rank)
        self.adaln_txt = BlockModulation(dim, modulation, modulation_rank)
        self.text_out = text_out

    def forward(self, x, y, cond, rope_img, rope_txt=None, mod_img=None, mod_txt=None):
        (xs1, xc1, xg1, xs2, xc2, xg2) = self.adaln_img(cond, mod_img).chunk(6, dim=-1)
        (ys1, yc1, yg1, ys2, yc2, yg2) = self.adaln_txt(cond, mod_txt).chunk(6, dim=-1)
        hx = modulate(self.norm_x1(x), xs1, xc1)
        hy = modulate(self.norm_y1(y), ys1, yc1)
        attn_x, attn_y = self.attn(
            hx,
            hy,
            rope_img,
            rope_txt,
            gate_x=None if self.attn_gate_x is None else torch.sigmoid(self.attn_gate_x(hx)),
            gate_y=None if self.attn_gate_y is None else torch.sigmoid(self.attn_gate_y(hy)),
        )
        x = x + xg1 * self.attn_post_norm_x(attn_x)
        x = x + xg2 * self.mlp_post_norm_x(self.mlp_x(modulate(self.norm_x2(x), xs2, xc2)))
        if not self.text_out:
            return x, y
        y = y + yg1 * self.attn_post_norm_y(attn_y)
        y = y + yg2 * self.mlp_post_norm_y(self.mlp_y(modulate(self.norm_y2(y), ys2, yc2)))
        return x, y


class SingleStreamBlock(nn.Module):
    """Single-stream block: text and image share one set of weights."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=True,
        norm_eps=1e-6,
        modulation="per_block",
        modulation_rank=64,
        num_kv_heads=None,
        gated_attention=False,
        sandwich_norm=False,
        text_out=True,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by query heads {num_heads}")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError(f"query heads {num_heads} not divisible by KV heads {num_kv_heads}")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        if num_kv_heads == num_heads:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            kv_dim = num_kv_heads * self.head_dim
            self.qkv = None
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, kv_dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.attn_gate = nn.Linear(dim, dim, bias=False) if gated_attention else None
        self.attn_proj = nn.Linear(dim, dim, bias=True)
        self.attn_post_norm = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.mlp = SwiGLU(dim, mlp_ratio)
        self.mlp_post_norm = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.adaln = BlockModulation(dim, modulation, modulation_rank)
        self.text_out = text_out

    def _qkv(self, h, batch, n):
        if self.qkv is not None:
            return self.qkv(h).reshape(batch, n, 3, self.num_heads, self.head_dim).unbind(2)
        q = self.q_proj(h).reshape(batch, n, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(batch, n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).reshape(batch, n, self.num_kv_heads, self.head_dim)
        return q, k, v

    def forward(self, x, y, cond, rope_img, rope_txt=None, mod=None):
        """Text and image rows share every weight, so each per-token op (norm,
        modulation, projections, gate, MLP) runs per stream and only Q/K/V are
        concatenated for the joint attention: same math as the reference's
        cat-first layout, but no kernel iterates over a cat output, which
        inductor cannot range-split under dynamic shapes (CantSplit)."""
        batch, n_img, dim = x.shape
        n_txt = y.shape[1]
        (s1, c1, g1, s2, c2, g2) = self.adaln(cond, mod).chunk(6, dim=-1)

        hx = modulate(self.norm1(x), s1, c1)
        hy = modulate(self.norm1(y), s1, c1)
        qx, kx, vx = self._qkv(hx, batch, n_img)
        qy, ky, vy = self._qkv(hy, batch, n_txt)
        qx, kx = self.q_norm(qx), self.k_norm(kx)
        qy, ky = self.q_norm(qy), self.k_norm(ky)
        qx, kx = apply_rope(qx, kx, rope_img)
        if rope_txt is not None:
            qy, ky = apply_rope(qy, ky, rope_txt)
        q = torch.cat([qy, qx], dim=1).transpose(1, 2)
        k = torch.cat([ky, kx], dim=1).transpose(1, 2)
        v = torch.cat([vy, vx], dim=1).transpose(1, 2)
        attn = scaled_dot_product(q, k, v, enable_gqa=self.num_kv_heads != self.num_heads)
        attn = attn.transpose(1, 2)  # [B, n_txt + n_img, H, D]
        attn_x = attn[:, n_txt:].reshape(batch, n_img, dim)
        attn_y = attn[:, :n_txt].reshape(batch, n_txt, dim) if self.text_out else None

        def finish(tokens, h, a):
            if self.attn_gate is not None:
                a = a * torch.sigmoid(self.attn_gate(h))
            tokens = tokens + g1 * self.attn_post_norm(self.attn_proj(a))
            mlp = self.mlp(modulate(self.norm2(tokens), s2, c2))
            return tokens + g2 * self.mlp_post_norm(mlp)

        x = finish(x, hx, attn_x)
        if not self.text_out:
            # text supplied keys/values above; its rows are discarded downstream
            return x, y
        return x, finish(y, hy, attn_y)


class PiTBlock(nn.Module):
    """Pixel-level transformer block over per-patch pixel sequences
    ``[B*L, p*p, d_pix]`` conditioned on the patch's (timestep-fused) token.
    ``post`` modulation: affine on branch outputs, chunk order (scale, shift)."""

    def __init__(
        self,
        hidden_size,
        cond_dim,
        pixels_per_patch,
        attn_hidden_size,
        num_heads,
        mlp_ratio=4.0,
        modulation="pre",
        qk_norm=True,
        norm_eps=1e-6,
    ):
        super().__init__()
        if modulation not in ("pre", "post"):
            raise ValueError(f"unknown PiT modulation '{modulation}'")
        self.modulation = modulation
        self.pixels_per_patch = pixels_per_patch
        n_mod = 6 if modulation == "pre" else 4
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.adaln = nn.Linear(cond_dim, n_mod * hidden_size * pixels_per_patch, bias=True)
        self.compress = nn.Linear(pixels_per_patch * hidden_size, attn_hidden_size, bias=True)
        self.expand = nn.Linear(attn_hidden_size, pixels_per_patch * hidden_size, bias=True)
        self.attn = SelfAttention(attn_hidden_size, num_heads, qkv_bias=False, qk_norm=qk_norm, norm_eps=norm_eps)
        self.mlp = GeluMLP(hidden_size, mlp_ratio)

    def _global_attn(self, pixels, rope, grid):
        n_patches = grid[0] * grid[1]
        batch = pixels.shape[0] // n_patches
        compact = self.compress(pixels.reshape(batch * n_patches, -1))
        attn = self.attn(compact.reshape(batch, n_patches, -1), rope=rope)
        return self.expand(attn.reshape(batch * n_patches, -1)).reshape_as(pixels)

    def forward(self, x, cond, rope, grid):
        mods = self.adaln(cond).reshape(x.shape[0], self.pixels_per_patch, -1)
        if self.modulation == "pre":
            shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)
            h = modulate(self.norm1(x), shift1, scale1)
            x = x + gate1 * self._global_attn(h, rope, grid)
            x = x + gate2 * self.mlp(modulate(self.norm2(x), shift2, scale2))
        else:
            scale1, shift1, scale2, shift2 = mods.chunk(4, dim=-1)
            x = x + modulate(self._global_attn(self.norm1(x), rope, grid), shift1, scale1)
            x = x + modulate(self.mlp(self.norm2(x)), shift2, scale2)
        return x


class FinalLayer(nn.Module):
    """Unmodulated head: RMSNorm + Linear."""

    def __init__(self, hidden_size: int, out_dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

    def forward(self, x):
        return self.linear(self.norm(x))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class IrisDiT(nn.Module, OstrisModelMixin):
    """Iris pixel-space DiT. ``forward(x, t, y, y_mask)`` -> velocity [B, C, H, W]."""

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = IrisConfig.from_dict(config)
        config.validate()
        self.config = config
        cfg = config
        self.gradient_checkpointing = False
        p = cfg.patch_size
        dim = cfg.hidden_size

        self.s_embedder = PatchEmbedder(p * p * cfg.in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim, max_period=cfg.timestep_max_period)
        if cfg.text_adapter == "linear":
            self.y_embedder = TextEmbedder(cfg.text_dim, dim, norm_eps=cfg.norm_eps)
        elif cfg.text_adapter == "blocks2":
            self.y_embedder = TransformerTextEmbedder(
                cfg.text_dim, dim, num_blocks=2, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias, qk_norm=cfg.qk_norm, norm_eps=cfg.norm_eps,
            )
        else:
            self.y_embedder = LayerwiseTextEmbedder(
                cfg.text_dim,
                dim,
                num_layers=cfg.text_lap_num_layers,
                layer_num_heads=cfg.text_lap_num_heads,
                layer_mlp_ratio=cfg.text_lap_mlp_ratio,
                refiner_num_heads=cfg.num_heads,
                refiner_mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                norm_eps=cfg.norm_eps,
            )
        self._adapter_needs_mask = cfg.text_adapter != "linear"
        self.y_pos_embedding = (
            nn.Parameter(torch.randn(1, cfg.text_len, dim)) if cfg.text_abs_pos_embed else None
        )

        # shared adaLN cores, one Linear(D, 6D) per stream, evaluated once per
        # forward; the single-stream half of a hybrid trunk rides the dual
        # half's image core (reference ``stream_aliases={"shared": "img"}``)
        self.modulation_cores = nn.ModuleDict()
        self._shared_modulation = cfg.modulation in ("shared_lowrank", "shared_bias")
        self._stream_alias = {"shared": "img"} if cfg.dual_depth else {}
        if self._shared_modulation:
            streams = []
            if cfg.dual_depth:
                streams += ["img", "txt"]
            if cfg.dual_depth < cfg.depth:
                streams.append(self._stream_alias.get("shared", "shared"))
            for stream in dict.fromkeys(streams):
                self.modulation_cores[f"adaln_{stream}"] = nn.Linear(dim, 6 * dim, bias=True)

        block_kwargs = dict(
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_norm=cfg.qk_norm,
            norm_eps=cfg.norm_eps,
            modulation=cfg.modulation,
            modulation_rank=cfg.modulation_rank,
            num_kv_heads=cfg.num_kv_heads,
            gated_attention=cfg.gated_attention,
            sandwich_norm=cfg.sandwich_norm,
        )
        tail_cls = MMDiTBlock if cfg.block == "mmdit" else SingleStreamBlock
        final_text_out = cfg.final_block_text == "keep"
        block_classes = [MMDiTBlock if i < cfg.dual_depth else tail_cls for i in range(cfg.depth)]
        self.blocks = nn.ModuleList(
            [
                block_cls(dim, cfg.num_heads, text_out=final_text_out or i < cfg.depth - 1, **block_kwargs)
                for i, block_cls in enumerate(block_classes)
            ]
        )
        # recorded here rather than isinstance() at forward time: the trainer
        # may replace entries with torch.compile wrappers (OptimizedModule)
        self._block_is_dual = [block_cls is MMDiTBlock for block_cls in block_classes]

        if not cfg.pixel.enabled:
            self.pixel_embedder = None
            self.pixel_blocks = None
            self.final_layer = FinalLayer(dim, p * p * cfg.in_channels, norm_eps=cfg.norm_eps)
        else:
            self.pixel_embedder = PixelEmbedder(
                cfg.in_channels, cfg.pixel.hidden_size, p, abs_pos_embed=cfg.pixel.abs_pos_embed
            )
            self.pixel_blocks = nn.ModuleList(
                [
                    PiTBlock(
                        cfg.pixel.hidden_size,
                        cond_dim=dim,
                        pixels_per_patch=p * p,
                        attn_hidden_size=cfg.pixel.attn_hidden_size,
                        num_heads=cfg.pixel.num_heads,
                        mlp_ratio=cfg.pixel.mlp_ratio,
                        modulation=cfg.pixel.modulation,
                        qk_norm=cfg.qk_norm,
                        norm_eps=cfg.norm_eps,
                    )
                    for _ in range(cfg.pixel.depth)
                ]
            )
            self.final_layer = FinalLayer(cfg.pixel.hidden_size, cfg.in_channels, norm_eps=cfg.norm_eps)

        self._rope_img: Dict[tuple, torch.Tensor] = {}
        self._rope_txt: Dict[tuple, torch.Tensor] = {}
        self._rope_pix: Dict[tuple, torch.Tensor] = {}

    # -- toolkit hooks ---------------------------------------------------------
    @property
    def in_channels(self) -> int:
        return self.config.in_channels

    @property
    def device(self):
        return self.s_embedder.proj.weight.device

    @property
    def dtype(self):
        return self.s_embedder.proj.weight.dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    @classmethod
    def get_transformer_block_names(cls) -> Optional[List[str]]:
        # streamed quantization over the trunk; the text adapter and pixel
        # stage go through the whole-module "extras" pass (full names, so the
        # exclude patterns below apply to them)
        return ["blocks"]

    @classmethod
    def get_quantization_exclude_modules(cls) -> Optional[List[str]]:
        # embedders / heads / modulation cores feed every block or are tiny;
        # the pixel stage is 16-wide per pixel and sets the final RGB values
        return [
            "s_embedder*",
            "t_embedder*",
            "final_layer*",
            "modulation_cores*",
            "pixel_embedder*",
            "pixel_blocks*",
            "y_embedder.layer_pool*",
            "adaln*",
            "*.adaln*",
        ]

    def get_offload_ignore_modules(self):
        # tiny live state every block reads: keep resident
        return [self.t_embedder, self.modulation_cores, self.s_embedder, self.final_layer]

    # -- rope caches -----------------------------------------------------------
    def _fetch_rope_img(self, grid, device):
        key = (grid, str(device))
        if key not in self._rope_img:
            cfg = self.config
            head_dim = cfg.hidden_size // cfg.num_heads
            self._rope_img[key] = rope_2d(
                head_dim, grid[0], grid[1], theta=cfg.rope_theta, scale=cfg.rope_scale,
                aspect=cfg.rope_aspect, frame_pairs=cfg.rope_frame_pairs, frame_theta=cfg.rope_frame_theta,
            ).to(device)
        return self._rope_img[key]

    def _fetch_rope_txt(self, length, device):
        if not self.config.text_rope:
            return None
        key = (length, str(device))
        if key not in self._rope_txt:
            head_dim = self.config.hidden_size // self.config.num_heads
            self._rope_txt[key] = rope_1d(head_dim, length, theta=self.config.text_rope_theta).to(device)
        return self._rope_txt[key]

    def _fetch_rope_pix(self, grid, device):
        key = (grid, str(device))
        if key not in self._rope_pix:
            cfg = self.config
            head_dim = cfg.pixel.attn_hidden_size // cfg.pixel.num_heads
            self._rope_pix[key] = rope_2d(
                head_dim, grid[0], grid[1], theta=cfg.rope_theta, scale=cfg.rope_scale, aspect=cfg.rope_aspect,
            ).to(device)
        return self._rope_pix[key]

    def _run_block(self, block, *args):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: noisy image [B, C, H, W], H and W divisible by patch_size.
        t: model time in [0, 1000] (shifted flow level x 1000), [B].
        y: text states [B, T, text_dim]; for ``lap_blocks2`` the selected layer
           stack [B, T, L, text_dim], or its flattened form [B, T, L*text_dim].
        y_mask: encoder attention mask [B, T] (1 = real token); required by the
           masked adapters.
        """
        cfg = self.config
        p = cfg.patch_size
        batch, _, height, width = x.shape
        if height % p or width % p:
            raise ValueError(f"input {height}x{width} is not divisible by patch_size {p}")
        grid = (height // p, width // p)
        n_patches = grid[0] * grid[1]

        if cfg.text_adapter == "lap_blocks2" and y.ndim == 3:
            y = y.reshape(y.shape[0], y.shape[1], cfg.text_lap_num_layers, cfg.text_dim)

        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)  # [B, L, p*p*C]
        s = self.s_embedder(patches)
        t_emb = self.t_embedder(t)  # [B, 1, D]
        cond = F.silu(t_emb)

        y = y[:, : cfg.text_len]
        if self._adapter_needs_mask:
            if y_mask is None:
                raise ValueError(
                    f"text_adapter='{cfg.text_adapter}' masks pad text positions and requires y_mask"
                )
            y = self.y_embedder(y, y_mask[:, : cfg.text_len])
        else:
            y = self.y_embedder(y)
        if self.y_pos_embedding is not None:
            y = y + self.y_pos_embedding[:, : y.shape[1]].to(y.dtype)

        rope_img = self._fetch_rope_img(grid, x.device)
        rope_txt = self._fetch_rope_txt(y.shape[1], x.device)

        shared = {}
        if self._shared_modulation:
            shared = {key: core(cond) for key, core in self.modulation_cores.items()}

        def mod_for(stream):
            if not self._shared_modulation:
                return None
            return shared[f"adaln_{self._stream_alias.get(stream, stream)}"]

        for block, is_dual in zip(self.blocks, self._block_is_dual):
            if is_dual:
                s, y = self._run_block(block, s, y, cond, rope_img, rope_txt, mod_for("img"), mod_for("txt"))
            else:
                s, y = self._run_block(block, s, y, cond, rope_img, rope_txt, mod_for("shared"))

        s = F.silu(t_emb + s)  # timestep re-fused into every patch token

        if self.pixel_blocks is None:
            out = self.final_layer(s)  # [B, L, p*p*C]
            folded = out.transpose(1, 2)
        else:
            s_cond = s.reshape(batch * n_patches, -1)
            pixels = self.pixel_embedder(x)  # [B*L, p*p, d_pix]
            rope_pix = self._fetch_rope_pix(grid, x.device)
            for block in self.pixel_blocks:
                pixels = self._run_block(block, pixels, s_cond, rope_pix, grid)
            out = self.final_layer(pixels)  # [B*L, p*p, C]
            folded = out.reshape(batch, n_patches, p * p, -1).permute(0, 3, 2, 1)
            folded = folded.reshape(batch, -1, n_patches)

        return F.fold(folded, output_size=(height, width), kernel_size=p, stride=p)
