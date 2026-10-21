"""Qwen3-VL prompt encoding for Iris-3B, ported from the reference
``iris3b/text/qwen3_vl.py``.

The conditioning is a fixed-length (``text_len`` = 300) stack of 12 Qwen3-VL
hidden layers per token. The prompt is wrapped in a chat template: the system
prefix is fed through the encoder as context but its hidden states are sliced
off; the caption is truncated to ``text_len - len(suffix)`` tokens BEFORE the
assistant-turn suffix is appended, so the suffix can never be the part that is
cut; the rest is right-padded to ``text_len`` and the pad positions are zeroed.

Fixed-length padding is load-bearing, not a convenience: the trunk's joint
attention sees every text position unmasked (only the text adapter masks pad
keys), so the number of pad tokens is part of the distribution the model was
trained on. Always encode to the full ``text_len``.
"""

from typing import List, Tuple

import torch

# Chat template around every prompt (identical to the reference and to Krea 2).
PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# 1-based decoder layers whose hidden states are stacked (hidden_states[i] is
# the output of layer i; index 0 is the embedding output).
DEFAULT_HIDDEN_LAYERS: Tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
DEFAULT_TEXT_LEN = 300


class IrisPromptEncoder:
    """Tokenizes with the Iris template and runs the frozen Qwen3-VL language
    stack. ``__call__`` returns ``(features, mask)`` with features
    ``(B, text_len, num_layers * hidden)`` (layer axis flattened so the toolkit
    treats it as an ordinary (B, L, D) embedding) and ``mask`` ``(B, text_len)``
    bool, True at real tokens."""

    def __init__(
        self,
        tokenizer,
        text_len: int = DEFAULT_TEXT_LEN,
        hidden_layers: Tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
    ):
        self.tokenizer = tokenizer
        self.text_len = int(text_len)
        self.hidden_layers = tuple(int(i) for i in hidden_layers)
        prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
        suffix_ids = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)
        if not prefix_ids or not suffix_ids:
            raise ValueError("Iris prompt template produced an empty token prefix or suffix")
        self.prefix_ids = torch.tensor(prefix_ids, dtype=torch.long)
        self.suffix_ids = torch.tensor(suffix_ids, dtype=torch.long)
        self.caption_budget = self.text_len - len(suffix_ids)
        if self.caption_budget < 1:
            raise ValueError(f"text_len={self.text_len} leaves no room for a caption")
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer exposes neither pad_token_id nor eos_token_id")
        self.pad_id = int(pad_id)

    def tokenize(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(input_ids, attention_mask)`` of shape ``(B, len(prefix) + text_len)``."""
        caption_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        start = len(self.prefix_ids)
        stop = start + self.text_len
        input_ids = torch.full((len(prompts), stop), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), stop), dtype=torch.long)
        input_ids[:, :start] = self.prefix_ids
        n_suffix = len(self.suffix_ids)
        for row, ids in enumerate(caption_ids):
            n = min(len(ids), self.caption_budget)
            if n:
                input_ids[row, start : start + n] = torch.tensor(ids[:n], dtype=torch.long)
            input_ids[row, start + n : start + n + n_suffix] = self.suffix_ids
            attention_mask[row, : start + n + n_suffix] = 1
        return input_ids, attention_mask

    @torch.no_grad()
    def __call__(self, text_encoder, prompts: List[str], dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # the language stack alone: Qwen3VLForConditionalGeneration.model is the
        # inner Qwen3VLModel (skips the lm_head logits over 150k vocab)
        decoder = getattr(text_encoder, "model", text_encoder)
        device = text_encoder.device
        input_ids, attention_mask = self.tokenize(prompts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) <= self.hidden_layers[-1]:
            raise RuntimeError("Qwen3-VL decoder did not return the requested hidden layers")
        start = len(self.prefix_ids)
        stop = start + self.text_len
        # (B, T, L, D), pads zeroed; with layer offloading the parameters'
        # device is not the compute device, so follow the hidden states
        embeds = torch.stack([hidden_states[layer][:, start:stop] for layer in self.hidden_layers], dim=2)
        mask = attention_mask[:, start:stop].to(embeds.device)
        embeds = embeds * mask[:, :, None, None].to(embeds.dtype)
        if dtype is not None:
            embeds = embeds.to(dtype)
        embeds = embeds.reshape(embeds.shape[0], embeds.shape[1], -1)
        return embeds, mask.bool()
