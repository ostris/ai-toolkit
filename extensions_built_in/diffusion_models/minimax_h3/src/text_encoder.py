"""Qwen3-VL conditioning for MiniMax-H3.

MiniMax-H3 conditions on the **unnormalized** ``hidden_states[50]`` of its
Qwen3-VL-32B conditioner (``hidden_states[0]`` is the embedding output, so
this is the output of decoder layer 49, before the final norm). The LM head
and layers 50..63 are never used, which lets the loader truncate the stack.

The presentation is raw tokens — no chat template, no special tokens:

  - t2va: the verbatim prompt.
  - fl2va: per keyframe, a ``"<Picture i>: "`` label plus a vision block
    (``<|vision_start|>`` + one ``<|image_pad|>`` per merged vision patch +
    ``<|vision_end|>``), then the verbatim prompt. Vision-block rows are
    tagged as *video* (0) rather than text (1) — the transformer's AdaLN
    modality selection keys off these tags.
"""

from typing import List, Optional

import torch

from .packing import TEXT_TAG, VIDEO_TAG

TEXT_ENCODER_LAYER = 50


@torch.no_grad()
def encode_minimax_h3_prompt(
    text_encoder,  # transformers Qwen3VLForConditionalGeneration
    tokenizer,  # Qwen2TokenizerFast
    processor,  # Qwen3VLProcessor (needed only when keyframes are present)
    prompt: str,
    keyframes: Optional[List] = None,  # PIL images already on the target canvas
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_length: Optional[int] = None,  # cap on PROMPT tokens (vision blocks are never cut)
):
    """Encode ONE prompt (with optional keyframes) into MiniMax-H3 conditioning.

    Returns (embeds (L, 5120), token_tags (L,) long). The embeds come from
    ``hidden_states[50]`` unnormalized. A stack truncated to exactly 50 layers
    also works ONLY if the final ``model.norm`` has been replaced with an
    Identity (transformers applies the final norm to the last entry of
    ``hidden_states``); the loader in minimax_h3.py does exactly that.
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers < TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 needs at least {TEXT_ENCODER_LAYER} Qwen3-VL decoder "
            f"layers to read hidden_states[{TEXT_ENCODER_LAYER}], got {num_layers}"
        )
    if device is None:
        device = text_encoder.device

    pixel_values, image_grid_thw = None, None
    token_ids: List[int] = []
    token_tags: List[int] = []
    if keyframes:
        vision = processor.image_processor(images=keyframes, return_tensors="pt")
        pixel_values = vision["pixel_values"]
        image_grid_thw = vision["image_grid_thw"]
        merge = processor.image_processor.merge_size**2
        vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        for i in range(len(keyframes)):
            num_image_tokens = int(image_grid_thw[i].prod()) // merge
            label_ids = tokenizer(f"<Picture {i + 1}>: ", add_special_tokens=False)[
                "input_ids"
            ]
            vision_ids = [vision_start] + [image_pad] * num_image_tokens + [vision_end]
            token_ids += label_ids + vision_ids
            token_tags += [TEXT_TAG] * len(label_ids) + [VIDEO_TAG] * len(vision_ids)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if max_length is not None and max_length > 0:
        # the cap applies to the caption only; a keyframe's vision block is
        # structural conditioning and cannot be truncated without corrupting it
        prompt_ids = prompt_ids[:max_length]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    if len(token_ids) == 0:
        # empty (unconditional) prompt: a single pad token keeps the sequence
        # non-degenerate; the model was not trained with CFG so this is only
        # ever a fallback
        token_ids = [tokenizer.pad_token_id or 0]
        token_tags = [TEXT_TAG]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
    )

    # call the inner .model directly: the LM head's vocab projection is dead
    # weight here and hidden_states[50] is all that is consumed
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None
        if pixel_values is None
        else pixel_values.to(device, text_encoder.dtype),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
        use_cache=False,
        output_hidden_states=True,
    )
    layer = min(TEXT_ENCODER_LAYER, len(outputs.hidden_states) - 1)
    embeds = outputs.hidden_states[layer][0]
    if dtype is not None:
        embeds = embeds.to(dtype)
    return embeds, torch.tensor(token_tags, dtype=torch.long)
