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
