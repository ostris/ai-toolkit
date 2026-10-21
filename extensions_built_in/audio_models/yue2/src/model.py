"""YuE2 AR + NAR mixture-of-transformers backbone in the ComfyUI repack layout.

Two 28-layer Qwen3-shaped experts share one architecture:
  - ``YuE2AR``  (``text_encoders.*`` in the repack): token embedding, causal
    layers, final norm and lm head. Writes ABC + semantic codec tokens.
  - ``YuE2NAR`` (``model.diffusion_model.*``): latent in/out projections,
    timestep embedder, sinusoidal frame position table and the same layer
    stack. Predicts flow velocity on VAE latents while attending into the
    AR expert's key/value cache over ``prefix + codec tokens + MUSIC_END``.

Weights use the merged ``qkv_proj`` / ``gate_up_proj`` linears of the
Comfy-Org checkpoint so LoRA keys line up with ComfyUI's loader.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as ckpt

from toolkit.models.v2._mixin import OstrisModelMixin

# vocabulary layout of the AR expert
EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
CONTEXT = 24576
FRAMES_PER_SECOND = 25
LATENT_DIM = 64

INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}


class YuE2Config:
    hidden_size = 2048
    intermediate_size = 6144
    num_hidden_layers = 28
    num_attention_heads = 16
    num_key_value_heads = 8
    head_dim = 128
    vocab_size = 184704
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    max_position_embeddings = CONTEXT


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    """positions [B, L] -> cos, sin [B, L, head_dim // 2] in fp32."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim)
    )
    angles = positions.float().unsqueeze(-1) * inv_freq
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x [B, H, L, D]; cos/sin [B, L, D/2]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class YuE2Attention(nn.Module):
    def __init__(self, cfg: YuE2Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        inner = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_size, inner + 2 * kv, bias=False)
        self.o_proj = nn.Linear(inner, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def project(self, x, cos, sin):
        b, l, _ = x.shape
        inner = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        q, k, v = self.qkv_proj(x).split((inner, kv, kv), dim=-1)
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        return q, k, v

    def attend(self, q, k, v, causal: bool):
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=self.num_heads != self.num_kv_heads
        )
        b, _, l, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(b, l, -1))


class YuE2MLP(nn.Module):
    def __init__(self, cfg: YuE2Config):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class YuE2Layer(nn.Module):
    def __init__(self, cfg: YuE2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = YuE2Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = YuE2MLP(cfg)

    def forward(
        self,
        x,
        cos,
        sin,
        prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal: bool = True,
        return_kv: bool = False,
    ):
        q, k, v = self.self_attn.project(self.input_layernorm(x), cos, sin)
        present = (k, v) if return_kv else None
        if prefix_kv is not None:
            k = torch.cat([prefix_kv[0], k], dim=2)
            v = torch.cat([prefix_kv[1], v], dim=2)
        x = x + self.self_attn.attend(q, k, v, causal)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class YuE2Stack(nn.Module):
    """``model.layers`` + ``model.norm`` of one expert."""

    def __init__(self, cfg: YuE2Config):
        super().__init__()
        self.layers = nn.ModuleList([YuE2Layer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class YuE2Expert(nn.Module):
    """Shared expert base: LoRA targets this class name."""

    def __init__(self, cfg: YuE2Config):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

    def _run_layer(self, layer, *args, **kwargs):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return ckpt.checkpoint(layer, *args, use_reentrant=False, **kwargs)
        return layer(*args, **kwargs)

    @property
    def device(self):
        return self.model.norm.weight.device

    @property
    def dtype(self):
        # norms stay in the activation dtype when the linears are quantized
        return self.model.norm.weight.dtype


class YuE2AR(YuE2Expert):
    def __init__(self, cfg: YuE2Config):
        super().__init__(cfg)
        self.model = YuE2Stack(cfg)
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def embed(self, ids: torch.Tensor):
        # never touch embed_tokens.weight: the int8 checkpoint's table dequantizes the whole vocab on that property
        return self.model.embed_tokens(ids.to(self.device))

    def prefill(self, inputs_embeds: torch.Tensor, return_hidden: bool = False):
        """Causal pass over [B, L, H]. Returns per-layer post-rope (k, v) as
        [B, kv_heads, L, head_dim] plus (optionally) the final normed hidden."""
        b, l, _ = inputs_embeds.shape
        positions = torch.arange(l, device=inputs_embeds.device)[None].expand(b, -1)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta)
        x = inputs_embeds
        cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.model.layers:
            x, kv = self._run_layer(layer, x, cos, sin, None, True, True)
            cache.append(kv)
        hidden = self.model.norm(x) if return_hidden else None
        return cache, hidden

    @torch.no_grad()
    def decode_step(self, token_embeds: torch.Tensor, cache: "KVCache"):
        """One new token per batch row against a preallocated cache."""
        b = token_embeds.shape[0]
        pos = torch.full((b, 1), cache.length, device=token_embeds.device, dtype=torch.long)
        cos, sin = rope_cos_sin(pos, self.cfg.head_dim, self.cfg.rope_theta)
        x = token_embeds
        for i, layer in enumerate(self.model.layers):
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), cos, sin)
            k, v = cache.append(i, k, v)
            x = x + layer.self_attn.attend(q, k, v, causal=False)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        cache.length += 1
        return self.model.lm_head(self.model.norm(x[:, -1]))

    @torch.no_grad()
    def prefill_into_cache(self, inputs_embeds: torch.Tensor, cache: "KVCache"):
        b, l, _ = inputs_embeds.shape
        positions = torch.arange(l, device=inputs_embeds.device)[None].expand(b, -1)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta)
        x = inputs_embeds
        for i, layer in enumerate(self.model.layers):
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), cos, sin)
            cache.write(i, 0, k, v)
            x = x + layer.self_attn.attend(q, k, v, causal=True)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        cache.length = l
        return self.model.lm_head(self.model.norm(x[:, -1]))


class KVCache:
    """Preallocated [B, kv_heads, capacity, head_dim] per layer for AR decode."""

    def __init__(self, cfg: YuE2Config, batch: int, capacity: int, device, dtype):
        shape = (batch, cfg.num_key_value_heads, capacity, cfg.head_dim)
        self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.capacity = capacity
        self.length = 0

    def write(self, layer, start, k, v):
        n = k.shape[2]
        self.k[layer][:, :, start : start + n] = k
        self.v[layer][:, :, start : start + n] = v

    def append(self, layer, k, v):
        end = self.length + k.shape[2]
        if end > self.capacity:
            raise ValueError("YuE2 KV cache capacity exceeded")
        self.write(layer, self.length, k, v)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.freq_size = freq_size

    def forward(self, t: torch.Tensor):
        half = self.freq_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class AudioPositionEmbedding(nn.Module):
    def __init__(self, max_frames, hidden_size):
        super().__init__()
        self.register_buffer("pe", torch.zeros(max_frames, hidden_size))

    def forward(self, length: int):
        return self.pe[:length]


class YuE2NAR(YuE2Expert):
    def __init__(self, cfg: YuE2Config):
        super().__init__(cfg)
        self.model = YuE2Stack(cfg)
        self.vae2llm = nn.Linear(LATENT_DIM, cfg.hidden_size)
        self.llm2vae = nn.Linear(cfg.hidden_size, LATENT_DIM)
        self.time_embedder = TimestepEmbedder(cfg.hidden_size)
        self.latent_pos_embed = AudioPositionEmbedding(cfg.max_position_embeddings, cfg.hidden_size)

    def forward(
        self,
        noisy_latents: torch.Tensor,  # [B, T, 64]
        t: torch.Tensor,  # [B] in 0..1, 1 = pure noise
        prefix_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_len: int,
    ):
        b, frames, _ = noisy_latents.shape
        n = frames + 2
        x = F.pad(noisy_latents, (0, 0, 1, 1))  # START / END slots
        x = self.vae2llm(x.to(self.vae2llm.weight.dtype))
        x = x + self.time_embedder(t)[:, None]
        x = x + self.latent_pos_embed(n)[None].to(x.dtype)
        positions = torch.arange(prefix_len, prefix_len + n, device=x.device)[None].expand(b, -1)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta)
        for layer, kv in zip(self.model.layers, prefix_cache):
            x, _ = self._run_layer(layer, x, cos, sin, kv, False, False)
        return self.llm2vae(self.model.norm(x))[:, 1:-1]


class YuE2Model(nn.Module, OstrisModelMixin):
    """Both experts under one module so a single LoRA network covers them."""

    aitk_comfy_repo = "Comfy-Org/YuE2"

    def __init__(self, cfg: Optional[YuE2Config] = None):
        super().__init__()
        cfg = cfg or YuE2Config()
        self.cfg = cfg
        self.ar = YuE2AR(cfg)
        self.nar = YuE2NAR(cfg)

    @classmethod
    def get_transformer_block_names(cls):
        return ["ar.model.layers", "nar.model.layers"]

    @classmethod
    def get_quantization_exclude_modules(cls):
        return ["ar.model.embed_tokens", "ar.model.lm_head", "nar.vae2llm", "nar.llm2vae", "nar.time_embedder*"]

    @property
    def device(self):
        return self.nar.device

    @property
    def dtype(self):
        return self.nar.dtype

    def enable_gradient_checkpointing(self):
        self.ar.gradient_checkpointing = True
        self.nar.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.ar.gradient_checkpointing = False
        self.nar.gradient_checkpointing = False

    @classmethod
    def load_from_state_dict(cls, sd: dict, dtype=torch.bfloat16):
        """``sd`` is a Comfy-Org all-in-one checkpoint (raw keys): bf16, or the int8 convrot repack whose
        ``comfy_quant`` markers attach the shipped quantization directly (no requantization)."""
        from toolkit.util.comfy_quant_import import Int8Embedding, import_comfy_quantized_layers
        from toolkit.util.ostris_quant import OstrisLinear

        model = cls()
        nar_sd = {k[len("model.diffusion_model.") :]: v for k, v in sd.items() if k.startswith("model.diffusion_model.")}
        ar_sd = {k[len("text_encoders.") :]: v for k, v in sd.items() if k.startswith("text_encoders.") and k != "text_encoders.yue2_tokenizer_json"}
        prequantized = any(k.endswith(".comfy_quant") for k in sd)
        for name, module, part in (("NAR", model.nar, nar_sd), ("AR", model.ar, ar_sd)):
            allowed_missing = set()
            if prequantized:
                quantized_paths = {k[: -len(".comfy_quant")] for k in part if k.endswith(".comfy_quant")}
                part, n = import_comfy_quantized_layers(module, part, orig_dtype=dtype)
                for mod_name, m in module.named_modules():
                    if isinstance(m, OstrisLinear):
                        allowed_missing.add(f"{mod_name}.weight")
                        if m.bias is not None:
                            allowed_missing.add(f"{mod_name}.bias")
                            m.bias.data = m.bias.data.to(dtype)
                    elif isinstance(m, (Int8Embedding, nn.Embedding)) and mod_name in quantized_paths:
                        allowed_missing.add(f"{mod_name}.weight")
            part = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in part.items()}
            missing, unexpected = module.load_state_dict(part, strict=False)
            missing = [k for k in missing if k not in allowed_missing]
            if missing:
                raise ValueError(f"YuE2 {name} missing keys: {missing[:5]} (+{max(0, len(missing) - 5)})")
            if unexpected:
                print(f"    YuE2 {name} unexpected: {len(unexpected)} (first 3: {unexpected[:3]})")
        if not prequantized:
            return model.to(dtype)
        # cast the remaining float tensors (norms, embeddings, buffers) without touching the quantized layers
        for m in model.modules():
            if isinstance(m, (OstrisLinear, Int8Embedding)):
                continue
            for pname, prm in list(m.named_parameters(recurse=False)):
                if prm.is_floating_point():
                    prm.data = prm.data.to(dtype)
            for bname, buf in list(m.named_buffers(recurse=False)):
                if buf is not None and buf.is_floating_point():
                    setattr(m, bname, buf.to(dtype))
        model.aitk_is_quantized = True  # aitk_post_load keeps the shipped convrot8 layers when convrot8 is requested
        return model


@torch.no_grad()
def _nar_lora_from_safetensors(path: str) -> dict:
    """The ``.safetensors`` releases of the community NAR adapter name every tensor
    (``layers.N.nar_self_attn.q_proj.lora_A`` ..., ``vae2llm.*``, ``llm2vae.*``); rebuild the
    ``.pt`` layout (``lora``: layer-major A/B list, ``io``: full projection weights) that the merge walks.
    The ``_comfyui`` files use ComfyUI's fused layout and are not handled here."""
    from safetensors.torch import load_file

    tensors = load_file(path, device="cpu")
    if not any(".nar_self_attn." in k and k.endswith(".lora_A") for k in tensors):
        raise ValueError(f"{path}: not an unmerged NAR LoRA release (the _comfyui layout is not supported for merging)")
    modules = [("nar_self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
    modules += [("nar_mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]
    lora = []
    for layer in sorted({int(k.split(".")[1]) for k in tensors if k.startswith("layers.")}):
        for block, proj in modules:
            lora += [tensors[f"layers.{layer}.{block}.{proj}.lora_A"], tensors[f"layers.{layer}.{block}.{proj}.lora_B"]]
    io = {m: {k.split(".", 1)[1]: v for k, v in tensors.items() if k.startswith(m + ".")} for m in ("vae2llm", "llm2vae")}
    return {"lora": lora, "io": {m: sd for m, sd in io.items() if sd}}


def merge_nar_lora(model: YuE2Model, ckpt_path: str, scale: float = 1.0):
    """Fold the community NAR adapter (unmerged q/k/v + gate/up LoRA pairs,
    plus full vae2llm/llm2vae weights) into the merged-projection base."""
    if str(ckpt_path).endswith(".safetensors"):
        ck = _nar_lora_from_safetensors(ckpt_path)
    else:
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if getattr(model, "aitk_is_quantized", False):
        raise ValueError("merge_nar_lora needs the bf16 checkpoint; the int8 convrot repack cannot take merged weights")
    tensors = iter(ck["lora"])
    cfg = model.cfg
    inner = cfg.num_attention_heads * cfg.head_dim
    kv = cfg.num_key_value_heads * cfg.head_dim
    qkv_rows = {"q_proj": (0, inner), "k_proj": (inner, inner + kv), "v_proj": (inner + kv, inner + 2 * kv)}
    gu_rows = {"gate_proj": (0, cfg.intermediate_size), "up_proj": (cfg.intermediate_size, 2 * cfg.intermediate_size)}
    for layer in model.nar.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            A = next(tensors).float()
            B = next(tensors).float()
            delta = (B @ A) * scale
            if name == "o_proj":
                w = layer.self_attn.o_proj.weight
                w.add_(delta.to(w.device, w.dtype))
            else:
                r0, r1 = qkv_rows[name]
                w = layer.self_attn.qkv_proj.weight
                w[r0:r1].add_(delta.to(w.device, w.dtype))
        for name in ("gate_proj", "up_proj", "down_proj"):
            A = next(tensors).float()
            B = next(tensors).float()
            delta = (B @ A) * scale
            if name == "down_proj":
                w = layer.mlp.down_proj.weight
                w.add_(delta.to(w.device, w.dtype))
            else:
                r0, r1 = gu_rows[name]
                w = layer.mlp.gate_up_proj.weight
                w[r0:r1].add_(delta.to(w.device, w.dtype))
    io = ck.get("io", {})
    for name in ("vae2llm", "llm2vae"):
        if name in io:
            mod = getattr(model.nar, name)
            for pname, value in io[name].items():
                getattr(mod, pname).copy_(value.to(mod.weight.device, mod.weight.dtype))
    return model
