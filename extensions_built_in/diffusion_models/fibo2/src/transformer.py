"""Fibo 2.0 DiT (briaai/Fibo-2.0-dev).

Port of diffusers' ``BriaFibo2Transformer2DModel`` in the Comfy-Org/Fibo-2.0
key layout: self-attention q/k/v fused into ``attention.qkv``, cross-attention
and Perceiver k/v into ``to_kv``, SwiGLU w1/w3 into ``feed_forward.w13`` with
the hidden width zero-padded to a multiple of 256 (9536 -> 9728), and the
input taking FLUX.2's 128-channel packed latents (the diffusers model's 2x2
patchify of the 32-channel latents, same (c, dy, dx) feature order). Comfy
files load as-is and LoRA keys match ComfyUI.

One stream of [gist | image | references] through Z-Image blocks; the gist
tokens come from a Perceiver over the first Qwen3-VL bundle and are clean
(t = 0 modulation) like the references. Five injection blocks also
cross-attend to their own bundle. Batches whose gist counts differ put the
gist last and mask the padding, exactly as diffusers does.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from toolkit.models.v2._mixin import OstrisModelMixin

ADALN_DIM = 256


@dataclass
class Fibo2Params:
    dim: int = 3584
    n_heads: int = 28
    n_layers: int = 44
    n_refiner_layers: int = 2
    injection_layers: Tuple[int, ...] = (6, 14, 22, 30, 38)
    cap_feat_dim: int = 7680
    perceiver_layers: int = 2
    max_gist_tokens: int = 512
    ffn_dim: int = 9728
    in_channels: int = 128
    norm_eps: float = 1e-5
    # fixed by the release (diffusers config defaults)
    min_gist_tokens: int = 256
    gist_step: int = 64
    gist_min_text_len: int = 256
    gist_max_text_len: int = 2048
    rope_theta: float = 256.0
    axes_dims: Tuple[int, ...] = (32, 48, 48)


def gist_counts(lengths: Sequence[int], p: Fibo2Params) -> List[int]:
    """Gist tokens per prompt: min_gist up to min_text_len text tokens, rising
    linearly to max_gist at max_text_len, rounded up to gist_step."""
    span = p.gist_max_text_len - p.gist_min_text_len
    counts = []
    for n in lengths:
        ratio = min(1.0, max(0.0, (float(n) - p.gist_min_text_len) / span))
        raw = p.min_gist_tokens + ratio * (p.max_gist_tokens - p.min_gist_tokens)
        quantized = int(math.ceil(raw / p.gist_step)) * p.gist_step
        counts.append(max(p.min_gist_tokens, min(p.max_gist_tokens, quantized)))
    return counts


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps)
        return y.to(x.dtype) * self.weight


def _apply_rope(x, rope):
    # interleaved (real, imag) pairs rotated in fp32, as diffusers' complex form
    cos, sin = rope
    re, im = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.stack((re * cos - im * sin, re * sin + im * cos), -1).flatten(-2)
    return out.type_as(x)


def _sdpa(q, k, v, mask):
    # (B, L, H, D) in, (B, L, H * D) out; mask: bool (B, 1, 1, S), True = attend
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(
        q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask
    )
    return out.transpose(1, 2).flatten(2)


def _modulate(h, values, span):
    """h * value per token: values = (noisy,) for all tokens, or (noisy, clean)
    with only tokens [span[0]:span[1]] noisy."""
    if span is None:
        return h * values[0][:, None]
    noisy, clean = values
    s, e = span
    return torch.cat(
        (
            h[:, :s] * clean[:, None],
            h[:, s:e] * noisy[:, None],
            h[:, e:] * clean[:, None],
        ),
        1,
    )


class _JointAttention(nn.Module):
    def __init__(self, dim, heads, eps):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = _RMSNorm(dim // heads, eps)
        self.k_norm = _RMSNorm(dim // heads, eps)

    def forward(self, x, rope, mask):
        q, k, v = self.qkv(x).unflatten(-1, (3, self.heads, -1)).unbind(2)
        q = _apply_rope(self.q_norm(q), rope)
        k = _apply_rope(self.k_norm(k), rope)
        return self.out(_sdpa(q, k, v, mask))


class _Attention(nn.Module):
    # no RoPE: the cross-attention to the text, and the Perceiver's self-attention
    def __init__(self, dim, heads, kv_dim, eps):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.norm_q = _RMSNorm(dim // heads, eps)
        self.norm_k = _RMSNorm(dim // heads, eps)

    def forward(self, x, context, mask):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k, v = self.to_kv(context).unflatten(-1, (2, self.heads, -1)).unbind(2)
        return self.to_out(_sdpa(q, self.norm_k(k), v, mask))


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate, up = self.w13(x).chunk(2, -1)
        return self.w2(F.silu(gate) * up)


class _Block(nn.Module):
    """Z-Image's block; an injection block also cross-attends to its bundle."""

    def __init__(self, p: Fibo2Params, injection: bool):
        super().__init__()
        dim, eps = p.dim, p.norm_eps
        self.attention = _JointAttention(dim, p.n_heads, eps)
        self.feed_forward = _FeedForward(dim, p.ffn_dim)
        self.attention_norm1 = _RMSNorm(dim, eps)
        self.ffn_norm1 = _RMSNorm(dim, eps)
        self.attention_norm2 = _RMSNorm(dim, eps)
        self.ffn_norm2 = _RMSNorm(dim, eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(ADALN_DIM, 4 * dim, bias=True))
        self.cross_attention = None
        if injection:
            self.cross_attn_norm1 = _RMSNorm(dim, eps)
            self.cross_attention = _Attention(dim, p.n_heads, p.cap_feat_dim, eps)
            self.cross_attn_norm2 = _RMSNorm(dim, eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, rope, adaln, span, mask=None, context=None, context_mask=None):
        # adaln: (B, 256) for all tokens, or (2B, 256) = [noisy; clean] with span
        mod = self.adaLN_modulation(adaln)
        parts = (mod,) if span is None else mod.chunk(2, 0)
        chunks = [m.chunk(4, -1) for m in parts]
        scale_msa = [1 + c[0] for c in chunks]
        gate_msa = [c[1].tanh() for c in chunks]
        scale_mlp = [1 + c[2] for c in chunks]
        gate_mlp = [c[3].tanh() for c in chunks]

        y = self.attention(
            _modulate(self.attention_norm1(x), scale_msa, span), rope, mask
        )
        x = x + _modulate(self.attention_norm2(y), gate_msa, span)
        if self.cross_attention is not None:
            y = self.cross_attention(self.cross_attn_norm1(x), context, context_mask)
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(y)
        y = self.feed_forward(_modulate(self.ffn_norm1(x), scale_mlp, span))
        return x + _modulate(self.ffn_norm2(y), gate_mlp, span)


class _PerceiverLayer(nn.Module):
    def __init__(self, p: Fibo2Params):
        super().__init__()
        dim, eps = p.dim, p.norm_eps
        self.cross_attn_norm = _RMSNorm(dim, eps)
        self.cross_attention = _Attention(dim, p.n_heads, p.cap_feat_dim, eps)
        self.self_attn_norm = _RMSNorm(dim, eps)
        self.self_attention = _Attention(dim, p.n_heads, dim, eps)
        self.ffn_norm = _RMSNorm(dim, eps)
        self.ffn = _FeedForward(dim, p.ffn_dim)

    def forward(self, x, context, context_mask, gist_mask):
        x = x + self.cross_attention(self.cross_attn_norm(x), context, context_mask)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y, gist_mask)
        return x + self.ffn(self.ffn_norm(x))


class _Perceiver(nn.Module):
    def __init__(self, p: Fibo2Params):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.zeros(p.max_gist_tokens, p.dim))
        self.layers = nn.ModuleList(
            [_PerceiverLayer(p) for _ in range(p.perceiver_layers)]
        )
        self.norm_out = _RMSNorm(p.dim, p.norm_eps)


class _TimestepEmbedder(nn.Module):
    def __init__(self, freq_dim=256, mid_dim=1024, out_dim=ADALN_DIM):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, mid_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mid_dim, out_dim, bias=True),
        )

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat((args.cos(), args.sin()), -1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class _FinalLayer(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(ADALN_DIM, dim, bias=True)
        )

    def forward(self, x, adaln):
        scale = 1 + self.adaLN_modulation(adaln)
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
        return self.linear(x * scale[:, None])


class Fibo2Transformer(nn.Module, OstrisModelMixin):
    @classmethod
    def aitk_config_from_state_dict(cls, state_dict):
        # unquantized tensors only (the block linears may be int8)
        max_gist, dim = state_dict["perceiver.latent_queries"].shape
        n_layers = 1 + max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")
        )
        return Fibo2Params(
            dim=dim,
            n_heads=dim // state_dict["layers.0.attention.q_norm.weight"].shape[0],
            n_layers=n_layers,
            n_refiner_layers=1
            + max(
                int(k.split(".")[1])
                for k in state_dict
                if k.startswith("noise_refiner.")
            ),
            injection_layers=tuple(
                i
                for i in range(n_layers)
                if f"layers.{i}.cross_attn_gate" in state_dict
            ),
            # int8 convrot keeps the (out, in) weight shape
            cap_feat_dim=state_dict[
                "perceiver.layers.0.cross_attention.to_kv.weight"
            ].shape[1],
            perceiver_layers=1
            + max(
                int(k.split(".")[2])
                for k in state_dict
                if k.startswith("perceiver.layers.")
            ),
            max_gist_tokens=max_gist,
            ffn_dim=state_dict["layers.0.feed_forward.w13.weight"].shape[0] // 2,
            in_channels=state_dict["x_embedder.weight"].shape[1],
        )

    @classmethod
    def get_transformer_block_names(cls):
        return ["noise_refiner", "perceiver.layers", "layers"]

    @classmethod
    def get_quantization_exclude_modules(cls):
        # what the Comfy int8 convrot file keeps in bf16
        return ["x_embedder", "t_embedder*", "final_layer*", "*adaLN_modulation*"]

    def __init__(self, params: Optional[Fibo2Params] = None):
        super().__init__()
        p = params or Fibo2Params()
        if p.dim // p.n_heads != 2 * sum(d // 2 for d in p.axes_dims):
            raise ValueError("rope axes must cover the head dim")
        self.params = p
        self.x_embedder = nn.Linear(p.in_channels, p.dim, bias=True)
        self.t_embedder = _TimestepEmbedder()
        self.noise_refiner = nn.ModuleList(
            [_Block(p, False) for _ in range(p.n_refiner_layers)]
        )
        self.perceiver = _Perceiver(p)
        self.layers = nn.ModuleList(
            [_Block(p, i in p.injection_layers) for i in range(p.n_layers)]
        )
        self.final_layer = _FinalLayer(p.dim, p.in_channels)
        self.gradient_checkpointing = False

    @property
    def device(self):
        return self.x_embedder.weight.device

    @property
    def dtype(self):
        # never quantized
        return self.x_embedder.weight.dtype

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _ckpt(self, module, *args):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _rope(self, ids: torch.Tensor):
        """(L, 3) integer positions -> (cos, sin), each (1, L, 1, head_dim / 2)
        fp32; phases in fp64 like diffusers' precomputed tables."""
        angles = []
        for i, d in enumerate(self.params.axes_dims):
            freqs = 1.0 / (
                self.params.rope_theta
                ** (torch.arange(0, d, 2, dtype=torch.float64, device=ids.device) / d)
            )
            angles.append(torch.outer(ids[:, i].double(), freqs).float())
        angles = torch.cat(angles, -1)[None, :, None]
        return angles.cos(), angles.sin()

    @staticmethod
    def _grid(plane, height, width, device):
        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        return torch.stack((torch.full_like(h, plane), h, w), -1).flatten(0, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, 128, H, W) normalized FLUX.2 latents
        timestep: torch.Tensor,  # 0..1000, 1000 = pure noise
        encoder_hidden_states: torch.Tensor,  # (B, T, 6 * cap_feat_dim)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, T) bool
        ref_latents: Optional[List[torch.Tensor]] = None,  # each (B, 128, h, w)
    ):
        p = self.params
        batch, channels, height, width = hidden_states.shape
        device, dtype = hidden_states.device, self.dtype
        num_image = height * width

        t = (
            torch.as_tensor(timestep, device=device, dtype=torch.float32)
            .reshape(-1)
            .expand(batch)
        )
        adaln_noisy = self.t_embedder(t)
        adaln_both = torch.cat((adaln_noisy, self.t_embedder(torch.zeros_like(t))))

        # the image on rope plane max_gist + 1 whatever the gist count, reference
        # k on the k-th plane after it
        plane = p.max_gist_tokens + 1
        tokens = [hidden_states.flatten(2).transpose(1, 2)]
        ids = [self._grid(plane, height, width, device)]
        for k, ref in enumerate(ref_latents or []):
            tokens.append(ref.flatten(2).transpose(1, 2))
            ids.append(self._grid(plane + 1 + k, ref.shape[-2], ref.shape[-1], device))
        x = self.x_embedder(torch.cat(tokens, 1).to(dtype))
        image_rope = self._rope(torch.cat(ids))
        span = (0, num_image) if ref_latents else None
        for block in self.noise_refiner:
            x = self._ckpt(
                block, x, image_rope, adaln_both if span else adaln_noisy, span
            )

        context = encoder_hidden_states.to(dtype)
        mask = encoder_attention_mask
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)
            lengths = mask.sum(1).tolist()
            context_mask = (
                None
                if all(n == mask.shape[1] for n in lengths)
                else mask[:, None, None]
            )
        else:
            lengths, context_mask = [context.shape[1]] * batch, None
        bundles = context.chunk(1 + len(p.injection_layers), -1)

        counts = gist_counts(lengths, p)
        num_gist = max(counts)
        gist_mask = None
        if min(counts) < num_gist:
            gist_mask = (
                torch.arange(num_gist, device=device)
                < torch.tensor(counts, device=device)[:, None]
            )
        gist = (
            self.perceiver.latent_queries[:num_gist]
            .to(dtype)
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
        for layer in self.perceiver.layers:
            gist = self._ckpt(
                layer,
                gist,
                bundles[0],
                context_mask,
                None if gist_mask is None else gist_mask[:, None, None],
            )
        gist = self.perceiver.norm_out(gist)
        gist_ids = torch.zeros(num_gist, 3, dtype=torch.long, device=device)
        gist_ids[:, 0] = torch.arange(1, num_gist + 1, device=device)
        gist_rope = self._rope(gist_ids)

        # gist first as trained; with padded gists last so the padding ends every row
        if gist_mask is None:
            x = torch.cat((gist, x), 1)
            rope = tuple(torch.cat(r, 1) for r in zip(gist_rope, image_rope))
            span, attn_mask = (num_gist, num_gist + num_image), None
        else:
            attn_mask = torch.cat(
                (
                    torch.ones(batch, x.shape[1], dtype=torch.bool, device=device),
                    gist_mask,
                ),
                1,
            )[:, None, None]
            x = torch.cat((x, gist), 1)
            rope = tuple(torch.cat(r, 1) for r in zip(image_rope, gist_rope))
            span = (0, num_image)

        text = {
            layer_id: bundles[i + 1] for i, layer_id in enumerate(p.injection_layers)
        }
        for index, block in enumerate(self.layers):
            x = self._ckpt(
                block,
                x,
                rope,
                adaln_both,
                span,
                attn_mask,
                text.get(index),
                context_mask,
            )

        x = self.final_layer(x[:, span[0] : span[1]], adaln_noisy)
        return x.transpose(1, 2).reshape(batch, channels, height, width)
