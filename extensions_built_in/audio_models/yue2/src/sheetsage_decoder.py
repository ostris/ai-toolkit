"""SheetSage2's score decoder (BART-style, post-LN, learned positions with offset 2) in plain
torch, so the m-a-p/SheetSage2 remote code runs on current transformers without BartDecoder.
Parameter names follow the released checkpoint (decoder.embed_positions, decoder.layers.N.self_attn.*,
encoder_attn.*, fc1/fc2, *_layer_norm, layernorm_embedding). KV cache = per-layer tuples with the
batch dimension first, which the upstream generation loop can index."""

from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

POSITION_OFFSET = 2  # BART learned positional embedding convention


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split(self, x):
        b, l, _ = x.shape
        return x.view(b, l, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x, kv_source=None, mask=None, past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, cache_static=False):
        q = self._split(self.q_proj(x))
        if cache_static and past is not None:
            k, v = past  # cross attention: memory keys never change
        else:
            src = x if kv_source is None else kv_source
            k, v = self._split(self.k_proj(src)), self._split(self.v_proj(src))
            if past is not None and not cache_static:
                k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b, _, l, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, l, -1)), (k, v)

    # static-cache path: explicit casts to the projection dtype (no-ops under autocast with fp32 weights)
    def kv(self, src):
        src = src.to(self.k_proj.weight.dtype)
        return self._split(self.k_proj(src)), self._split(self.v_proj(src))

    def attend(self, x, k, v, mask):
        q = self._split(self.q_proj(x.to(self.q_proj.weight.dtype))).to(k.dtype)
        if mask is not None and q.shape[2] == 1:
            # single-query + mask sends SDPA to the mem-efficient kernel, which tiles over every cached key (~0.3 ms); by hand it is ~20 us
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (self.head_dim**-0.5)
            out = torch.matmul(scores.masked_fill(~mask, float("-inf")).softmax(-1).to(v.dtype), v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b, _, l, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, l, -1).to(self.out_proj.weight.dtype))


class _Layer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int):
        super().__init__()
        self.self_attn = _Attention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.encoder_attn = _Attention(dim, heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x, memory, self_mask, past):
        self_past = None if past is None else past[:2]
        cross_past = None if past is None else past[2:]
        h, self_kv = self.self_attn(x, mask=self_mask, past=self_past)
        x = self.self_attn_layer_norm(x + h)
        h, cross_kv = self.encoder_attn(x, kv_source=memory, past=cross_past, cache_static=True)
        x = self.encoder_attn_layer_norm(x + h)
        x = self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))
        return x, self_kv + cross_kv

    def step_static(self, x, cache, i, write_idx, mask):
        k, v = self.self_attn.kv(x)
        cache.k[i].index_copy_(2, write_idx, k.to(cache.k.dtype))
        cache.v[i].index_copy_(2, write_idx, v.to(cache.v.dtype))
        x = self.self_attn_layer_norm(x + self.self_attn.attend(x, cache.k[i], cache.v[i], mask))
        x = self.encoder_attn_layer_norm(x + self.encoder_attn.attend(x, cache.ck[i], cache.cv[i], None))
        return self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x.to(self.fc1.weight.dtype)))))


class StaticCache:
    """Preallocated batch-1 KV buffers so a decode step has no growing tensors and can be CUDA-graph replayed.
    ``pos`` (device tensor) is the number of cached tokens; the captured step reads and bumps it in place."""

    def __init__(self, decoder: "ScoreDecoder", max_len: int, memory_len: int, device, dtype):
        attn = decoder.layers[0].self_attn
        n, h, d = len(decoder.layers), attn.heads, attn.head_dim
        self.max_len = max_len
        self.k = torch.zeros(n, 1, h, max_len, d, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.ck = torch.zeros(n, 1, h, memory_len, d, device=device, dtype=dtype)
        self.cv = torch.zeros_like(self.ck)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.arange = torch.arange(max_len, device=device)

    def fill_cross(self, decoder: "ScoreDecoder", memory):
        for i, layer in enumerate(decoder.layers):
            k, v = layer.encoder_attn.kv(memory)
            self.ck[i].copy_(k)
            self.cv[i].copy_(v)
        self.pos.zero_()


class ScoreDecoder(nn.Module):
    """Drop-in for the BartDecoder the upstream model builds: ``ScoreDecoder(bart_config, embed_tokens=...)``."""

    def __init__(self, config, embed_tokens: nn.Embedding):
        super().__init__()
        dim, heads = config.d_model, config.decoder_attention_heads
        self.embed_tokens = embed_tokens
        self.embed_positions = nn.Embedding(config.max_position_embeddings + POSITION_OFFSET, dim)
        self.layernorm_embedding = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([_Layer(dim, heads, config.decoder_ffn_dim) for _ in range(config.decoder_layers)])
        self.pad_token_id = config.pad_token_id

    def gradient_checkpointing_disable(self):
        pass

    def prefill_static(self, input_ids, cache: StaticCache):
        """[1, P] into an empty cache; returns hidden [1, P, dim]."""
        p = input_ids.shape[1]
        idx = torch.arange(p, device=input_ids.device)
        x = self.layernorm_embedding(self.embed_tokens(input_ids) + self.embed_positions(idx + POSITION_OFFSET)[None])
        mask = (cache.arange[None, :] <= idx[:, None])[None, None]
        for i, layer in enumerate(self.layers):
            x = layer.step_static(x, cache, i, idx, mask)
        cache.pos.fill_(p)
        return x

    def step_static(self, input_ids, cache: StaticCache):
        """[1, 1] at position ``cache.pos``; graph-capturable (no host reads)."""
        x = self.layernorm_embedding(self.embed_tokens(input_ids) + self.embed_positions(cache.pos + POSITION_OFFSET)[None])
        mask = (cache.arange <= cache.pos)[None, None, None, :]
        for i, layer in enumerate(self.layers):
            x = layer.step_static(x, cache, i, cache.pos, mask)
        cache.pos += 1
        return x

    def forward(self, input_ids, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=False, output_hidden_states=False, return_dict=True, **_):
        b, l = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        positions = torch.arange(past_len, past_len + l, device=input_ids.device) + POSITION_OFFSET
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)[None]
        x = self.layernorm_embedding(x)
        mask = None
        if l > 1:
            causal = torch.ones(l, past_len + l, dtype=torch.bool, device=x.device).tril(past_len)
            mask = causal[None, None]
            if attention_mask is not None:
                mask = mask & attention_mask.bool()[:, None, None, :]
        elif attention_mask is not None and past_len == 0:
            mask = attention_mask.bool()[:, None, None, :]
        if mask is not None:
            mask = torch.zeros(mask.shape, dtype=x.dtype, device=x.device).masked_fill(~mask, float("-inf"))
        hidden_states = [x] if output_hidden_states else None
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, encoder_hidden_states, mask, None if past_key_values is None else past_key_values[i])
            new_cache.append(kv)
            if hidden_states is not None:
                hidden_states.append(x)
        return SimpleNamespace(
            last_hidden_state=x,
            past_key_values=tuple(new_cache) if use_cache else None,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )
