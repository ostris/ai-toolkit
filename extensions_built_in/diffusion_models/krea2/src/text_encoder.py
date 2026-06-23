"""Qwen3-VL text conditioning for Krea 2.

Vendored / adapted from the reference ``encoder.py``. Krea 2 conditions on a
*stack* of hidden states pulled from several layers of Qwen3-VL-4B-Instruct
(``SELECT_LAYERS``), wrapped in a fixed instruction template. The MMDiT's
``TextFusionTransformer`` later collapses that layer axis down to one.

The reference encodes a whole batch padded to ``max_length``; here we encode one
prompt at a time at its natural length (the ai-toolkit pattern -- caches stay
small, any prompts can share a batch, and per-sample padding is deferred to the
model call). The fixed instruction prefix is fed through the model as context but
its hidden states are sliced off the returned features, exactly like the
reference.
"""

import torch
from torch import Tensor


# Layers of Qwen3-VL whose hidden states are stacked and fed to the MMDiT (12).
SELECT_LAYERS: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Fixed instruction template wrapped around every prompt. The prefix is fed
# through the model as context but its hidden states are dropped from the output
# (the assistant only ever sees the prompt + suffix tokens as conditioning).
PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# Number of leading tokens (the system prefix) sliced off the encoded features.
PROMPT_TEMPLATE_ENCODE_START_IDX = 34


@torch.no_grad()
def encode_krea_prompt(
    qwen,
    tokenizer,
    processor,
    prompt: str,
    max_length: int = 512,
    select_layers: tuple[int, ...] = SELECT_LAYERS,
    prefix_idx: int = PROMPT_TEMPLATE_ENCODE_START_IDX,
) -> Tensor:
    """Encode a single prompt into stacked Qwen3-VL hidden states.

    Returns a ``(L, num_select_layers, hidden)`` float tensor (in the encoder's
    dtype) holding the prompt + suffix token features -- the system prefix has
    been sliced off. ``L`` is the natural (unpadded) length so the caller stores
    one tensor per prompt and pads to the batch max at the model call.
    """
    device = qwen.device

    # The suffix ("...assistant\n") is tokenized without the BOS/template extras
    # the main tokenizer adds, matching the reference's separate processor pass.
    suffix_inputs = processor(
        text=[PROMPT_TEMPLATE_ENCODE_SUFFIX], return_tensors="pt"
    ).to(device, non_blocking=True)
    suffix_ids = suffix_inputs["input_ids"]
    suffix_mask = suffix_inputs["attention_mask"].bool()

    # Prefix + prompt at natural length (no padding); truncate very long prompts.
    text = PROMPT_TEMPLATE_ENCODE_PREFIX + prompt
    inputs = tokenizer(
        [text],
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        max_length=max_length + prefix_idx,
        return_tensors="pt",
    ).to(device, non_blocking=True)

    input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
    mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)

    states = qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

    # (1, L, num_layers, hidden)
    hiddens = torch.stack([states.hidden_states[i] for i in select_layers], dim=2)
    # Drop the system-prefix tokens; what remains is prompt + suffix conditioning.
    hiddens = hiddens[:, prefix_idx:]
    return hiddens[0]
