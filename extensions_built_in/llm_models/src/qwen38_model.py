"""Qwen3.8 (Qwen3.5-family hybrid VLM) built for throughput.

Module tree and parameter names mirror transformers' Qwen3_5ForConditionalGeneration
(model.language_model.layers.N.{linear_attn|self_attn,mlp,...}, model.visual.*,
lm_head, mtp.*) so the ComfyUI / HF checkpoints load 1:1 and LoRA keys stay standard.
The vision tower is transformers' own Qwen3_5VisionModel (runs once per image); the
text stack, caches, decode step and speculative loop are this file's.

Text stack: 3 of every 4 layers are Gated DeltaNet (linear attention, fla kernels:
chunked for prefill / training, fused recurrent for decode), the 4th is gated full
attention (GQA, partial interleaved MRoPE). Linears are whatever the loader attached
(W4A8 codebook / int8 convrot OstrisLinear, or plain nn.Linear).

Decode: static KV + recurrent-state caches with a device-side position, so one
decode step is a fixed graph that CUDA-graph replays. With the MTP draft head the
step is draft D tokens -> verify D+1 in one pass -> accept -> commit states ->
refresh the MTP cache from the verified hidden states.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from toolkit.util.convrot_w4a8_quant import apply_forward_hooks, module_hooks_only, w4a8_grouped_linear

from .qwen38_kernels import _conv_step_op, _decode_attn_op, add_rmsnorm, decode_attention_fused_ok, rmsnorm, swiglu

_FLA = None


def _fla():
    """flash-linear-attention, imported on first use so the extension registry loads
    without it; the DeltaNet layers need its kernels at model build time."""
    global _FLA
    if _FLA is None:
        try:
            from fla.modules import FusedRMSNormGated
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        except ImportError as e:
            raise ImportError(
                "qwen38 needs the flash-linear-attention package (pip install flash-linear-attention)"
            ) from e
        _FLA = (FusedRMSNormGated, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
    return _FLA


# ------------------------------------------------------------------ norms / rope


class RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm: fp32 math, (1 + weight) scaling."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps, add_one=True)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(q, k, cos, sin):
    """q [B,H,T,D], k [B,Hk,T,D]; cos/sin [B,T,R] rotate the first R dims."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    r = cos.shape[-1]
    q_rot, q_pass = q[..., :r], q[..., r:]
    k_rot, k_pass = k[..., :r], k[..., r:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class InterleavedMRope(nn.Module):
    """Qwen3.5 interleaved MRoPE for the partial rotary dims: frequency i takes its
    position from axis i % 3 (T, H, W). Text tokens have equal positions on all axes,
    so decode uses a 1-D table."""

    def __init__(self, rotary_dim: int, theta: float, mrope_section):
        super().__init__()
        half = rotary_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # axis per frequency: [T,H,W] interleaved as in apply_interleaved_mrope
        axis = torch.zeros(half, dtype=torch.long)
        for dim, offset in enumerate((1, 2), start=1):
            axis[offset : mrope_section[dim] * 3 : 3] = dim
        self.register_buffer("axis", axis, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        """position_ids [3, B, T] -> cos, sin [B, T, rotary_dim]."""
        pos = position_ids.float()
        freqs = pos[..., None] * self.inv_freq.float()[None, None, None, :]  # [3,B,T,half]
        freqs = torch.gather(freqs, 0, self.axis.view(1, 1, 1, -1).expand(1, *freqs.shape[1:]))[0]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def table(self, max_len: int, dtype: torch.dtype):
        """1-D cos/sin table [max_len, rotary_dim] for text-only positions."""
        pos = torch.arange(max_len, device=self.inv_freq.device).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.forward(pos, dtype)
        return cos[0], sin[0]


# ------------------------------------------------------------------ caches


@dataclass
class DecodeCtx:
    """Per-forward decode context: where the M new tokens live in the caches."""

    pos_idx: torch.Tensor  # [M] int64 device: cache slots of the new tokens
    qpos: torch.Tensor  # [M] int64 device: absolute position of each query (for the causal mask)
    cos: torch.Tensor  # [B, M, R]
    sin: torch.Tensor
    commit: bool = True  # linear layers: write the new recurrent/conv state now
    min_key: int = 0  # keys below this slot are never attended (MTP slot 0 is empty)


class LayerCache:
    """Static per-layer state. Attention layers: k/v [B, Hk, L, D]. Linear layers:
    conv [B, C, ks-1] (last conv inputs) and rec [B, Hv, Dk, Dv] fp32."""

    def __init__(self):
        self.k = self.v = None
        self.conv = self.rec = None
        # speculative verify stash (uncommitted step inputs)
        self.spec = None


class Qwen38Cache:
    def __init__(self, cfg, batch: int, max_len: int, device, dtype, layer_types=None, with_linear=True):
        self.max_len = max_len
        self.batch = batch
        layer_types = cfg.layer_types if layer_types is None else layer_types
        self.layers: List[LayerCache] = []
        for lt in layer_types:
            lc = LayerCache()
            if lt == "full_attention":
                lc.k = torch.zeros(batch, cfg.num_key_value_heads, max_len, cfg.head_dim, device=device, dtype=dtype)
                lc.v = torch.zeros_like(lc.k)
            elif with_linear:
                conv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2 + cfg.linear_num_value_heads * cfg.linear_value_head_dim
                lc.conv = torch.zeros(batch, conv_dim, cfg.linear_conv_kernel_dim - 1, device=device, dtype=dtype)
                lc.rec = torch.zeros(
                    batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim,
                    device=device, dtype=torch.float32,
                )
            self.layers.append(lc)
        # committed length, device-side so the decode graph never syncs
        self.pos = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self):
        for lc in self.layers:
            for t in (lc.k, lc.v, lc.conv, lc.rec):
                if t is not None:
                    t.zero_()
        self.pos.zero_()


# ------------------------------------------------------------------ mixers


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, bias=False, kernel_size=self.conv_kernel_size,
            groups=self.conv_dim, padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = _fla()[0](self.head_v_dim, eps=cfg.rms_norm_eps, activation="silu")
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def _heads(self, mixed: torch.Tensor):
        b, t, _ = mixed.shape
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, t, self.num_k_heads, self.head_k_dim)
        k = k.reshape(b, t, self.num_k_heads, self.head_k_dim)
        v = v.reshape(b, t, self.num_v_heads, self.head_v_dim)
        # fla's kernels take fewer key heads than value heads natively (same result as
        # the repeat_interleave transformers does, measured bit-identical)
        return q, k, v

    def _gates(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        return beta, g

    def _raw_ab(self, x: torch.Tensor):
        """Pre-activation gate inputs (a, b) [B, T, Hv]: one GEMM over the stacked weights
        when the two tiny projections are not wrapped (e.g. by an active LoRA)."""
        pa, pb = self.in_proj_a, self.in_proj_b
        if not (module_hooks_only(pa) and module_hooks_only(pb)):
            return pa(x), pb(x)
        w = getattr(self, "_ab_weight", None)
        if w is None or w.device != x.device or w.dtype != x.dtype:
            w = torch.cat([pa.weight, pb.weight], dim=0).to(x.dtype)
            self._ab_weight = w
        ab = F.linear(x, w)
        a, b = ab[..., : self.num_v_heads], ab[..., self.num_v_heads :]
        if pa._forward_hooks:
            a = apply_forward_hooks(pa, x, a)
        if pb._forward_hooks:
            b = apply_forward_hooks(pb, x, b)
        return a, b

    def _finish(self, core: torch.Tensor, z: torch.Tensor):
        b, t = core.shape[:2]
        out = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(b, t, -1))

    def forward(self, x: torch.Tensor, cache: Optional[LayerCache] = None, ctx: Optional[DecodeCtx] = None):
        if ctx is not None:
            return self.forward_decode(x, cache, ctx)
        return self.forward_prefill(x, cache)

    def forward_prefill(self, x: torch.Tensor, cache: Optional[LayerCache]):
        b, t, _ = x.shape
        mixed, z = w4a8_grouped_linear((self.in_proj_qkv, self.in_proj_z), x)
        beta, g = self._gates(self.in_proj_a(x), self.in_proj_b(x))
        xt = mixed.transpose(1, 2)  # [B, C, T]
        if cache is not None:
            ks = self.conv_kernel_size - 1
            cache.conv.copy_(F.pad(xt, (max(ks - t, 0), 0))[:, :, -ks:])
        conv = F.silu(F.conv1d(xt, self.conv1d.weight, None, padding=self.conv_kernel_size - 1, groups=self.conv_dim)[:, :, :t])
        q, k, v = self._heads(conv.transpose(1, 2))
        core, final = _fla()[1](
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=cache is not None,
            use_qk_l2norm_in_kernel=True,
        )
        if cache is not None:
            cache.rec.copy_(final)
        return self._finish(core, z)

    def forward_decode(self, x: torch.Tensor, cache: LayerCache, ctx: DecodeCtx):
        b, m, _ = x.shape
        mixed, z = w4a8_grouped_linear((self.in_proj_qkv, self.in_proj_z), x)
        a_raw, b_raw = self._raw_ab(x)
        # one kernel: causal depthwise conv over [state | new tokens] + silu, token major
        conv = _conv_step_op(cache.conv, mixed, self.conv1d.weight, ctx.commit)  # [B, M, C]
        q, k, v = self._heads(conv)
        # the decay / beta activations run inside the recurrent kernel
        core, final = _fla()[2](
            q, k, v, g=a_raw.float(), beta=b_raw, initial_state=cache.rec, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(), use_beta_sigmoid_in_kernel=True,
        )
        if ctx.commit:
            cache.rec.copy_(final)
        else:
            cache.spec = (q, k, v, a_raw, b_raw, mixed)
        return self._finish(core, z)

    def commit_speculative(self, cache: LayerCache, keep: torch.Tensor, accepted: torch.Tensor):
        """After verification: advance the state through the first accepted+1 of the
        stashed tokens. Rejected tokens get g=0, beta=0 (no decay, no update), so one
        more recurrent pass from the committed state lands exactly on S_{accepted+1}."""
        q, k, v, a_raw, b_raw, mixed = cache.spec
        full = torch.cat([cache.conv, mixed.transpose(1, 2)], dim=-1)  # [B, C, ks-1+M]
        beta, g = self._gates(a_raw, b_raw)
        keep_h = keep.view(1, -1, 1).to(g.dtype)
        _, final = _fla()[2](
            q, k, v, g=g * keep_h, beta=beta * keep_h.to(beta.dtype), initial_state=cache.rec,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        cache.rec.copy_(final)
        ks = self.conv_kernel_size - 1
        idx = accepted.view(1) + 1 + torch.arange(ks, device=full.device)
        cache.conv.copy_(full.index_select(-1, idx))
        cache.spec = None


def decode_attention(q, k_cache, v_cache, qpos, scale, min_key: int = 0):
    """Single-step GQA attention of M queries against the whole static cache.
    q [B,Hq,M,D]; caches [B,Hk,L,D]; qpos [M] absolute positions. Scores in fp32
    straight from the bf16 GEMM (sink logits overflow a bf16 score tensor)."""
    if decode_attention_fused_ok(q, k_cache):
        return _decode_attn_op(q, k_cache, v_cache, qpos, scale, min_key)
    b, hq, m, d = q.shape
    hk, L = k_cache.shape[1], k_cache.shape[2]
    g = hq // hk
    qg = q.reshape(b, hk, g, m, d).reshape(b * hk, g * m, d)
    kT = k_cache.reshape(b * hk, L, d).transpose(-1, -2)
    scores = torch.bmm(qg, kT, out_dtype=torch.float32).view(b * hk, g, m, L) * scale
    keys = torch.arange(L, device=q.device)
    allowed = keys[None, :] <= qpos[:, None]  # [M, L]
    if min_key > 0:
        allowed = allowed & (keys[None, :] >= min_key)
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(q.dtype).view(b * hk, g * m, L)
    out = torch.bmm(probs, v_cache.reshape(b * hk, L, d))  # [B*hk, g*m, d]
    return out.view(b, hk, g, m, d).reshape(b, hq, m, d)


class GatedAttention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim * 2, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def _qkv(self, x, cos, sin):
        b, t, _ = x.shape
        qg, k, v = w4a8_grouped_linear((self.q_proj, self.k_proj, self.v_proj), x)
        qg = qg.view(b, t, self.num_heads, 2 * self.head_dim)
        q, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        gate = gate.reshape(b, t, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k.view(b, t, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_partial_rope(q, k, cos, sin)
        return q, k, v, gate

    def forward(self, x, cos, sin, cache: Optional[LayerCache] = None, ctx: Optional[DecodeCtx] = None, start: int = 0):
        b, t, _ = x.shape
        q, k, v, gate = self._qkv(x, cos, sin)
        if ctx is None:
            if cache is not None:
                cache.k[:, :, start : start + t].copy_(k)
                cache.v[:, :, start : start + t].copy_(v)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=self.scaling)
        else:
            cache.k.index_copy_(2, ctx.pos_idx, k)
            cache.v.index_copy_(2, ctx.pos_idx, v)
            o = decode_attention(q, cache.k, cache.v, ctx.qpos, self.scaling, ctx.min_key)
        o = o.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        gate, up = w4a8_grouped_linear((self.gate_proj, self.up_proj), x)
        return self.down_proj(swiglu(gate, up))


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int, layer_type: Optional[str] = None):
        super().__init__()
        self.layer_type = layer_type or cfg.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(cfg, layer_idx)
        else:
            self.self_attn = GatedAttention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, h, cos, sin, cache=None, ctx=None, start: int = 0, pending=None):
        """pending: the previous layer's MLP output, not yet added to the residual stream;
        its add is fused with this layer's input norm. Returns (h, mlp_out) the same way."""
        if pending is not None:
            h, x = add_rmsnorm(h, pending, self.input_layernorm.weight, self.input_layernorm.eps)
        else:
            x = self.input_layernorm(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, cache, ctx)
        else:
            x = self.self_attn(x, cos, sin, cache, ctx, start)
        h, x = add_rmsnorm(h, x, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
        return h, self.mlp(x)


# ------------------------------------------------------------------ text model / mtp / top


class Qwen38TextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        rp = cfg.rope_parameters
        self.rotary = InterleavedMRope(
            int(cfg.head_dim * rp.get("partial_rotary_factor", 1.0)), rp["rope_theta"], rp.get("mrope_section", [11, 11, 10])
        )
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, cos, sin, cache: Optional[Qwen38Cache] = None, ctx: Optional[DecodeCtx] = None, start: int = 0):
        h = inputs_embeds
        pending = None
        for i, layer in enumerate(self.layers):
            lc = cache.layers[i] if cache is not None else None
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h, pending = torch.utils.checkpoint.checkpoint(layer, h, cos, sin, lc, ctx, start, pending, use_reentrant=False)
            else:
                h, pending = layer(h, cos, sin, lc, ctx, start, pending)
        _, y = add_rmsnorm(h, pending, self.norm.weight, self.norm.eps)
        return y

    def commit_speculative(self, cache: Qwen38Cache, keep: torch.Tensor, accepted: torch.Tensor):
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "linear_attention":
                layer.linear_attn.commit_speculative(cache.layers[i], keep, accepted)


class Qwen38MTP(nn.Module):
    """Multi-token-prediction draft head: fc(cat[norm(embed(token_p)), norm(hidden_{p-1})])
    -> one full-attention decoder layer with its own KV cache -> norm -> lm_head."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.pre_fc_norm_embedding = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList([DecoderLayer(cfg, 0, layer_type="full_attention")])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, embeds, hidden, cos, sin, cache: LayerCache, ctx: Optional[DecodeCtx] = None, start: int = 0):
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)], dim=-1))
        h, mlp = self.layers[0](x, cos, sin, cache, ctx, start)
        h, y = add_rmsnorm(h, mlp, self.norm.weight, self.norm.eps)
        return y, h


class Qwen38Inner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = Qwen38TextModel(config.text_config)


class Qwen38ForConditionalGeneration(nn.Module):
    def __init__(self, config, with_mtp: bool = True):
        super().__init__()
        self.config = config
        self.model = Qwen38Inner(config)
        tc = config.text_config
        self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)
        self.mtp = Qwen38MTP(tc) if with_mtp else None
        self.generator: Optional["Qwen38Generator"] = None

    @property
    def text_config(self):
        return self.config.text_config

    @property
    def device(self):
        return self.model.language_model.norm.weight.device

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    # -------------------------------------------------- vision

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Merged vision embeddings [N_merged, hidden] for one or more images / videos."""
        visual = self.model.visual
        out = visual(pixel_values.to(visual.patch_embed.proj.weight.dtype), grid_thw=grid_thw, return_dict=True)
        return out.pooler_output

    # -------------------------------------------------- positions

    def get_rope_index(self, input_ids: torch.Tensor, mm_token_type_ids: torch.Tensor, image_grid_thw=None, video_grid_thw=None):
        """Port of transformers' Qwen3_5 get_rope_index for one unpadded sequence.
        input_ids / mm_token_type_ids [T]; returns position_ids [3, 1, T] and the rope
        delta (first text position after the sequence minus its length)."""
        merge = self.config.vision_config.spatial_merge_size
        device = input_ids.device
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        grids = {1: iter(image_grid_thw) if image_grid_thw is not None else None, 2: iter(video_grid_thw) if video_grid_thw is not None else None}
        types = mm_token_type_ids.tolist()
        groups = []
        i = 0
        while i < len(types):
            j = i
            while j < len(types) and types[j] == types[i]:
                j += 1
            groups.append((types[i], i, j))
            i = j
        cur = 0
        parts = []
        for kind, s, e in groups:
            if kind == 0:
                n = e - s
                parts.append(torch.arange(n, device=device).view(1, -1).expand(3, -1) + cur)
                cur += n
            else:
                thw = next(grids[kind])
                t, h, w = int(thw[0]), int(thw[1]) // merge, int(thw[2]) // merge
                n = t * h * w
                pw = torch.arange(cur, cur + w, device=device).repeat(h * t)
                ph = torch.arange(cur, cur + h, device=device).repeat_interleave(w * t)
                pt = torch.full((n,), cur, device=device, dtype=torch.long)
                parts.append(torch.stack([pt, ph, pw], dim=0))
                cur += max(h, w)
        pos = torch.cat(parts, dim=1)
        delta = int(pos.max().item()) + 1 - input_ids.shape[0]
        return pos.view(3, 1, -1), delta

    # -------------------------------------------------- forward (prefill / training)

    def forward_hidden(self, inputs_embeds, position_ids, cache: Optional[Qwen38Cache] = None):
        """Post-norm hidden states [B, T, H] for a full sequence (prefill or training)."""
        lm = self.model.language_model
        cos, sin = lm.rotary(position_ids, inputs_embeds.dtype)
        return lm(inputs_embeds, cos, sin, cache=cache)

    def enable_gradient_checkpointing(self):
        self.model.language_model.gradient_checkpointing = True


# ------------------------------------------------------------------ generation


class Qwen38Generator:
    """Greedy generation with static caches and CUDA-graph decode steps; optional MTP
    speculative decoding (draft depth D). One instance per model; caches and graphs
    are reused across prompts and rebuilt when the length budget grows."""

    def __init__(self, model: Qwen38ForConditionalGeneration, use_cuda_graph: bool = True):
        self.model = model
        self.cfg = model.text_config
        self.use_cuda_graph = use_cuda_graph
        self.cache: Optional[Qwen38Cache] = None
        self.mtp_cache: Optional[Qwen38Cache] = None
        self.max_len = 0
        self.graphs: Dict[tuple, Tuple[torch.cuda.CUDAGraph, dict]] = {}
        self.dtype = model.model.language_model.norm.weight.dtype
        self.device = model.device
        self.rope_table = None
        self.last_stats = {}

    # ---------------- buffers

    def _ensure(self, max_len: int, batch: int = 1):
        if self.cache is not None and self.max_len >= max_len and self.cache.batch == batch:
            return
        self.graphs = {}
        self.max_len = max_len
        self.cache = Qwen38Cache(self.cfg, batch, max_len, self.device, self.dtype)
        if self.model.mtp is not None:
            self.mtp_cache = Qwen38Cache(self.cfg, batch, max_len, self.device, self.dtype, layer_types=["full_attention"], with_linear=False)
        cos, sin = self.model.model.language_model.rotary.table(max_len + 64, self.dtype)
        self.rope_table = (cos, sin)
        self.rope_delta = torch.zeros(1, dtype=torch.long, device=self.device)
        hs = self.cfg.hidden_size
        self.nt_buf = torch.zeros(batch, 1, dtype=torch.long, device=self.device)
        self.h_buf = torch.zeros(batch, 1, hs, dtype=self.dtype, device=self.device)
        self.out_buf = torch.zeros(1 + 16, dtype=torch.long, device=self.device)

    def _cos_sin(self, positions: torch.Tensor):
        cos, sin = self.rope_table
        idx = (positions + self.rope_delta).clamp_(min=0)
        return cos.index_select(0, idx)[None], sin.index_select(0, idx)[None]

    def _embed(self, ids: torch.Tensor):
        return self.model.get_input_embeddings()(ids).to(self.dtype)

    def _lm_head_argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden).argmax(dim=-1)

    # ---------------- step bodies (graph-capturable: no host syncs)

    def _plain_step(self):
        cache = self.cache
        pos = cache.pos
        ctx = DecodeCtx(pos_idx=pos, qpos=pos, cos=None, sin=None, commit=True)
        ctx.cos, ctx.sin = self._cos_sin(pos)
        hidden = self.model.model.language_model(self._embed(self.nt_buf), ctx.cos, ctx.sin, cache=cache, ctx=ctx)
        nt = self._lm_head_argmax(hidden[:, -1:])
        self.out_buf[0].fill_(0)
        self.out_buf[1].copy_(nt.view(-1)[0])
        self.nt_buf.copy_(nt)
        self.h_buf.copy_(hidden[:, -1:])
        pos.add_(1)

    def _mtp_step(self, depth: int):
        model, cache, mtp_cache = self.model, self.cache, self.mtp_cache
        lm = model.model.language_model
        pos = cache.pos
        ar = torch.arange(depth + 1, device=self.device)
        tok_in, hid_in = self.nt_buf, self.h_buf
        drafts = []
        for k in range(depth):
            p = pos + k
            ctx = DecodeCtx(pos_idx=p, qpos=p, cos=None, sin=None, min_key=1)
            ctx.cos, ctx.sin = self._cos_sin(p)
            normed, pre = model.mtp(self._embed(tok_in), hid_in, ctx.cos, ctx.sin, mtp_cache.layers[0], ctx)
            tok_in = self._lm_head_argmax(normed)
            hid_in = pre
            drafts.append(tok_in)
        drafts = torch.cat(drafts, dim=1)  # [B, D]
        tokens = torch.cat([self.nt_buf, drafts], dim=1)  # [B, D+1]
        positions = pos + ar
        ctx = DecodeCtx(pos_idx=positions, qpos=positions, cos=None, sin=None, commit=False)
        ctx.cos, ctx.sin = self._cos_sin(positions)
        hidden = lm(self._embed(tokens), ctx.cos, ctx.sin, cache=cache, ctx=ctx)  # [B, D+1, H]
        pred = self._lm_head_argmax(hidden)  # [B, D+1]
        match = (pred[:, :depth] == drafts).long()
        accepted = match.cumprod(dim=1).sum(dim=1)[0]  # scalar in [0, D]
        keep = (ar <= accepted).long()
        lm.commit_speculative(cache, keep, accepted)
        # refresh the draft cache from the verified hidden states: pair (token_p, hidden_{p-1})
        prev_hidden = torch.cat([self.h_buf, hidden[:, :depth]], dim=1)
        rctx = DecodeCtx(pos_idx=positions, qpos=positions, cos=ctx.cos, sin=ctx.sin, min_key=1)
        model.mtp(self._embed(tokens), prev_hidden, rctx.cos, rctx.sin, mtp_cache.layers[0], rctx)
        # committed tokens: drafts[:a] then pred[a]; report [a, tok_0..tok_D]
        drafts_ext = torch.cat([drafts, pred[:, -1:]], dim=1)[0]
        committed = torch.where(ar < accepted, drafts_ext, pred[0])
        self.out_buf[0].copy_(accepted)
        self.out_buf[1 : depth + 2].copy_(committed)
        a_idx = accepted.view(1, 1)
        self.nt_buf.copy_(pred.gather(1, a_idx))
        self.h_buf.copy_(hidden.gather(1, a_idx.view(1, 1, 1).expand(1, 1, hidden.shape[-1])))
        pos.add_(accepted + 1)

    def _adapter_signature(self):
        """Captured graphs bake in the adapter wrappers' Python-side decisions (active /
        merged / multiplier), so the graph cache is keyed on them."""
        hooks = 0
        for mod in self.model.model.language_model.modules():
            if not isinstance(mod, nn.Linear):
                continue
            hooks += len(mod._forward_hooks) + len(mod._forward_pre_hooks)
            fwd = mod.__dict__.get("forward")
            if fwd is None:
                continue
            lora = getattr(fwd, "__self__", None)
            net_ref = getattr(lora, "network_ref", None)
            net = net_ref() if callable(net_ref) else None
            if net is None:
                return ("wrapped", id(lora))
            return (id(net), bool(getattr(net, "is_active", True)), bool(getattr(net, "is_merged_in", False)), repr(getattr(net, "_multiplier", None)))
        # inference LoRA sidechains are forward hooks; their count keys the graph too
        return ("hooks", hooks) if hooks else None

    def _get_step(self, depth: int):
        key = (("mtp", depth) if depth > 0 else ("plain",)) + (self._adapter_signature(),)
        fn = (lambda: self._mtp_step(depth)) if depth > 0 else self._plain_step
        if not self.use_cuda_graph:
            return fn
        if key not in self.graphs:
            # warm up (triton autotune / allocator) on a side stream, then capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            saved = self._snapshot_state()
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn()
                    self._restore_state(saved)
            torch.cuda.current_stream().wait_stream(s)
            # two instantiations of the same step: a graph cannot be relaunched until its
            # previous replay has finished, so alternating two hides the (ms-scale) CPU
            # launch cost of a ~2000-node graph behind the other one's execution
            graphs = []
            for _ in range(2):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                self._restore_state(saved)
                graphs.append(g)
            self.graphs[key] = graphs
        graphs = self.graphs[key]
        state = {"i": 0}

        def replay():
            graphs[state["i"]].replay()
            state["i"] ^= 1

        return replay

    def _snapshot_state(self):
        c = self.cache
        snap = {"pos": c.pos.clone(), "nt": self.nt_buf.clone(), "h": self.h_buf.clone(), "layers": []}
        for lc in c.layers:
            snap["layers"].append(tuple(t.clone() if t is not None else None for t in (lc.conv, lc.rec)))
        snap["mtp_pos"] = self.mtp_cache.pos.clone() if self.mtp_cache is not None else None
        return snap

    def _restore_state(self, snap):
        c = self.cache
        c.pos.copy_(snap["pos"])
        self.nt_buf.copy_(snap["nt"])
        self.h_buf.copy_(snap["h"])
        for lc, (conv, rec) in zip(c.layers, snap["layers"]):
            if conv is not None:
                lc.conv.copy_(conv)
                lc.rec.copy_(rec)
        if self.mtp_cache is not None and snap["mtp_pos"] is not None:
            self.mtp_cache.pos.copy_(snap["mtp_pos"])

    # ---------------- prefill + loop

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        rope_delta: int,
        max_new_tokens: int,
        eos_ids,
        mtp_depth: int = 0,
        progress=None,
    ) -> List[int]:
        """inputs_embeds [1, T, H] (media already scattered in), input_ids [1, T],
        position_ids [3, 1, T]. Returns the generated token ids (eos included)."""
        model = self.model
        b, t, _ = inputs_embeds.shape
        if b != 1:
            raise ValueError("Qwen38Generator decodes one sequence at a time")
        if mtp_depth > 0 and model.mtp is None:
            mtp_depth = 0
        need = t + max_new_tokens + mtp_depth + 2
        if self.cache is None or self.max_len < need:
            # round up so a slightly longer prompt does not rebuild caches and graphs
            self._ensure(max(need, 2048) if self.cache is None else -(-need // 1024) * 1024)
        cache, mtp_cache = self.cache, self.mtp_cache
        cache.reset()
        if mtp_cache is not None:
            mtp_cache.reset()
        self.rope_delta.fill_(rope_delta)
        eos = set(int(e) for e in eos_ids)

        # prefill (eager): all layers over the prompt, caches written at slots 0..T-1
        lm = model.model.language_model
        cos, sin = lm.rotary(position_ids, inputs_embeds.dtype)
        hidden = lm(inputs_embeds, cos, sin, cache=cache)  # [1, T, H]
        nt = self._lm_head_argmax(hidden[:, -1:])  # [1, 1]
        cache.pos.fill_(t)
        if mtp_depth > 0:
            # draft cache over the prompt: slot p = (token_p, hidden_{p-1}), p = 1..T-1
            model.mtp(self._embed(input_ids[:, 1:]), hidden[:, :-1], cos[:, 1:], sin[:, 1:], mtp_cache.layers[0], None, start=1)
        self.nt_buf.copy_(nt)
        self.h_buf.copy_(hidden[:, -1:])
        out = [int(nt.item())]
        if out[-1] in eos:
            return out
        step = self._get_step(mtp_depth)
        n_steps = 0
        n_accepted = 0
        # pipelined: step k+1 is queued before step k's result is read, so the host-side
        # bookkeeping overlaps the GPU instead of leaving it idle at every sync. The one
        # extra step launched after the stop condition is harmless (its state is discarded).
        host = [torch.empty_like(self.out_buf, device="cpu", pin_memory=True) for _ in range(2)]
        events = [torch.cuda.Event(), torch.cuda.Event()]
        slot = 0
        step()
        host[slot].copy_(self.out_buf, non_blocking=True)
        events[slot].record()
        done = False
        while not done:
            nxt = slot ^ 1
            step()
            host[nxt].copy_(self.out_buf, non_blocking=True)
            events[nxt].record()
            events[slot].synchronize()
            res = host[slot].tolist()
            n_steps += 1
            if mtp_depth > 0:
                a = res[0]
                n_accepted += a
                new = res[1 : a + 2]
            else:
                new = res[1:2]
            for tok in new:
                out.append(tok)
                if tok in eos or len(out) >= max_new_tokens:
                    done = True
                    break
            if progress is not None:
                progress(len(out))
            slot = nxt
        torch.cuda.current_stream().synchronize()
        self.last_stats = {"steps": n_steps, "tokens": len(out), "accepted_drafts": n_accepted}
        return out


def build_sequence_embeds(model: Qwen38ForConditionalGeneration, input_ids: torch.Tensor, media_embeds: Optional[torch.Tensor], placeholder_ids):
    """Token embeddings with the media placeholder positions replaced by the vision
    embeddings. input_ids [T]; media_embeds [N, H] (rows in placeholder order)."""
    emb = model.get_input_embeddings()(input_ids[None])
    if media_embeds is not None:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for pid in placeholder_ids:
            mask |= input_ids == pid
        n = int(mask.sum())
        if n != media_embeds.shape[0]:
            raise ValueError(f"media placeholder count {n} != media embeddings {media_embeds.shape[0]}")
        emb = emb.masked_scatter(mask[None, :, None].expand_as(emb), media_embeds.to(emb.dtype))
    return emb
