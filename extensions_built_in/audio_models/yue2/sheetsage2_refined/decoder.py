"""BART-style token decoder (post-LayerNorm, learned positions) with a KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int = 31_678
    width: int = 512
    layers: int = 6
    heads: int = 8
    ffn_width: int = 2048
    max_tokens: int = 5120
    dropout: float = 0.1
    pad_token: int = 0


# BART reserves the first two position rows, so position p uses row p + 2.
POSITION_OFFSET = 2


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def split(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        return x.view(batch, length, self.heads, width // self.heads).transpose(1, 2)

    def forward(
        self, x: Tensor, key: Tensor, value: Tensor, *, mask: Tensor | None = None, causal: bool = False
    ) -> Tensor:
        """``key``/``value`` are already split into heads: ``[B, heads, S, head_dim]``."""

        y = F.scaled_dot_product_attention(
            self.split(self.q(x)),
            key,
            value,
            attn_mask=mask,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        batch, _, length, _ = y.shape
        return self.out(y.transpose(1, 2).reshape(batch, length, -1))


class DecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        width, self.dropout = config.width, config.dropout
        self.self_attn = Attention(width, config.heads, config.dropout)
        self.self_attn_norm = nn.LayerNorm(width)
        self.cross_attn = Attention(width, config.heads, config.dropout)
        self.cross_attn_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, config.ffn_width)
        self.fc2 = nn.Linear(config.ffn_width, width)
        self.final_norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, cache: dict, *, mask: Tensor | None, causal: bool) -> Tensor:
        drop = lambda y: F.dropout(y, self.dropout, self.training)  # noqa: E731

        key, value = self.self_attn.split(self.self_attn.k(x)), self.self_attn.split(self.self_attn.v(x))
        if "self_key" in cache:
            key = torch.cat((cache["self_key"], key), dim=2)
            value = torch.cat((cache["self_value"], value), dim=2)
        cache["self_key"], cache["self_value"] = key, value
        x = self.self_attn_norm(x + drop(self.self_attn(x, key, value, mask=mask, causal=causal)))

        cross = self.cross_attn(x, cache["cross_key"], cache["cross_value"], mask=cache.get("cross_mask"))
        x = self.cross_attn_norm(x + drop(cross))

        y = F.dropout(F.gelu(self.fc1(x)), self.dropout, self.training)
        return self.final_norm(x + drop(self.fc2(y)))


class TokenDecoder(nn.Module):
    """Causal decoder over symbolic tokens that cross-attends to encoder memory."""

    def __init__(self, config: DecoderConfig = DecoderConfig()):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=config.pad_token)
        self.position_embedding = nn.Embedding(config.max_tokens + POSITION_OFFSET, config.width)
        self.embedding_norm = nn.LayerNorm(config.width)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.layers)])

    def new_cache(self, memory: Tensor, frames: Tensor | None = None) -> list[dict]:
        """Project the encoder memory once per layer for cross-attention.

        ``frames`` ([B], optional) is the number of real (non-padding) memory
        frames per sequence; later frames are hidden from cross-attention.
        """

        mask = None
        if frames is not None:
            mask = (torch.arange(memory.shape[1], device=memory.device)[None] < frames[:, None])[:, None, None]
        return [
            {
                "cross_key": layer.cross_attn.split(layer.cross_attn.k(memory)),
                "cross_value": layer.cross_attn.split(layer.cross_attn.v(memory)),
                "cross_mask": mask,
                "length": 0,
            }
            for layer in self.layers
        ]

    def forward(self, tokens: Tensor, cache: list[dict]) -> Tensor:
        """Return logits ``[B, T, vocab]`` for ``tokens`` appended to ``cache``."""

        return self.logits(self.hidden(tokens, cache))

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.token_embedding.weight)  # output projection tied to the embedding

    def hidden(self, tokens: Tensor, cache: list[dict]) -> Tensor:
        """Final hidden states ``[B, T, width]`` for ``tokens`` appended to ``cache``.

        Without earlier tokens in the cache, attention is causal and padding
        tokens are masked as keys (right-padded teacher forcing).  With a cache,
        ``tokens`` are new positions that may see every cached position.
        """

        start = cache[0]["length"]
        positions = torch.arange(start, start + tokens.shape[1], device=tokens.device) + POSITION_OFFSET
        x = self.embedding_norm(self.token_embedding(tokens) + self.position_embedding(positions))
        x = F.dropout(x, self.config.dropout, self.training)

        mask, causal = None, False
        if start == 0 and tokens.shape[1] > 1:
            padding = tokens == self.config.pad_token
            if padding.any():
                length = tokens.shape[1]
                allowed = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
                mask = (allowed[None] & ~padding[:, None, :])[:, None]
            else:
                causal = True
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, layer_cache, mask=mask, causal=causal)
            layer_cache["length"] = start + tokens.shape[1]
        return x
