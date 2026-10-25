"""Fibo 2.0 text conditioning: Qwen3-VL-4B hidden states in six bundles.

The prompt goes in Qwen3-VL's user turn; 15 hidden layers are each scaled to
unit RMS in fp32, and every bundle concatenates three of them: the
Perceiver's, then one per injection block. Stored per prompt as
(L, 6 * 7680), the ComfyUI conditioning layout.

- Generation: a structured JSON caption, cleaned and compacted the way Fibo
  2.0 was trained, or plain text as it is; an empty prompt is the null
  caption (CFG negative, caption dropout).
- Edit: one JSON object, a vision marker per reference (``image`` or
  ``image_1``..), the result's caption when the prompt has one, then the edit
  instruction; Qwen3-VL sees each reference in place of its marker.

Not done: the images of a caption's text_render entries (diffusers' LoRA
trainers encode without them).
"""

import json
import math
from typing import List, Optional, Sequence

import torch
from PIL import Image

from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLModelEncoder

from .checkpoints import QWEN3_VL_4B_REPO

PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
IMAGE_PAD = "<|image_pad|>"
VISION_MARKER = f"<|vision_start|>{IMAGE_PAD}<|vision_end|>"

# the null caption Fibo 2.0 was trained with for classifier-free guidance
NULL_CAPTION = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)

# Qwen3-VL hidden_states indices (0 = embeddings) behind the six bundles
BUNDLE_LAYERS = (
    (9, 20, 31),
    (5, 16, 27),
    (7, 18, 29),
    (9, 20, 31),
    (11, 22, 33),
    (13, 24, 35),
)
MAX_TEXT_TOKENS = 4096
SCORES = ("aesthetic_score", "preference_score")


def _dumps(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _empty(value) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def _clean(value):
    if isinstance(value, dict):
        value = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in value.items() if not _empty(v)}
    if isinstance(value, list):
        return [v for v in map(_clean, value) if not _empty(v)]
    return value


def parse_prompt(prompt: str):
    """(caption, edit instruction) as diffusers' BriaFibo2Pipeline parses
    them; a prompt that isn't a JSON object comes back as it is."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction
    keys = set(caption)
    caption = {
        key: {k: v for k, v in value.items() if k not in SCORES}
        if isinstance(value, dict)
        else value
        for key, value in _clean(caption).items()
        if key not in SCORES
    }
    caption = {key: value for key, value in caption.items() if value != {}}
    # exactly a short_description and a text_render list keeps both, the list
    # even when empty: the image has no text
    if keys == {"short_description", "text_render"}:
        caption = {
            "short_description": caption.get("short_description") or "",
            "text_render": caption.get("text_render") or [],
        }
    return caption or None, instruction


def generation_text(prompt: str) -> str:
    if prompt.strip() == "":
        return NULL_CAPTION
    caption, _ = parse_prompt(prompt)
    return prompt if isinstance(caption, str) else _dumps(caption or {})


def edit_text(prompt: str, num_images: int) -> str:
    caption, instruction = parse_prompt(prompt)
    if instruction is None:
        caption, instruction = None, prompt
    if num_images == 1:
        edit = {"image": VISION_MARKER}
    else:
        edit = {f"image_{k}": VISION_MARKER for k in range(1, num_images + 1)}
    if caption:
        edit["structured_caption"] = caption
    edit["edit_instruction"] = instruction
    return _dumps(edit)


class Fibo2TextEncoder(Qwen3VLModelEncoder):
    """Qwen3-VL-4B (language + vision tower) from the Comfy single file, int8
    convrot or bf16; the tied lm_head is dropped on load."""

    aitk_config_repo = QWEN3_VL_4B_REPO
    aitk_tokenizer_repo = QWEN3_VL_4B_REPO
    aitk_tokenizer_subfolder = ""

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        return {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in state_dict.items()
            if not key.startswith("lm_head.")
        }

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        return {f"model.{k}": v for k, v in state_dict.items()}


def _bundles(hidden_states, dtype) -> torch.Tensor:
    normalized = {}
    for layer in sorted({layer for bundle in BUNDLE_LAYERS for layer in bundle}):
        states = hidden_states[layer].float()
        normalized[layer] = (
            states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + 1e-6)
        ).to(dtype)
    return torch.cat(
        [normalized[layer] for bundle in BUNDLE_LAYERS for layer in bundle], -1
    )


@torch.no_grad()
def encode_prompts(
    text_encoder,
    tokenizer,
    image_processor,
    prompts: List[str],
    images: Sequence[Optional[List[Image.Image]]],
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = MAX_TEXT_TOKENS,
) -> List[torch.Tensor]:
    """Prompts (+ per-prompt Qwen3-VL reference images) -> per-prompt
    (L_i, 46080) bundles. Text-only prompts share one right-padded forward
    without a mask: causal attention keeps the pads from reaching any real
    token. Prompts with references run one forward each."""
    out: List[Optional[torch.Tensor]] = [None] * len(prompts)

    text_only = [
        i for i, imgs in enumerate(images) if not imgs or prompts[i].strip() == ""
    ]
    if text_only:
        enc = tokenizer(
            [PROMPT_TEMPLATE.format(generation_text(prompts[i])) for i in text_only],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            truncation=True,
            max_length=max_length,
        )
        lengths = enc["attention_mask"].sum(1).tolist()
        hidden_states = text_encoder(
            input_ids=enc["input_ids"].to(device),
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        context = _bundles(hidden_states, dtype)
        for row, (i, n) in enumerate(zip(text_only, lengths)):
            out[i] = context[row, :n]

    image_token_id = text_encoder.config.image_token_id
    merge = image_processor.merge_size**2
    for i, imgs in enumerate(images):
        if out[i] is not None:
            continue
        pixels = image_processor(images=imgs, do_resize=False, return_tensors="pt")
        grid = pixels["image_grid_thw"]
        text = PROMPT_TEMPLATE.format(edit_text(prompts[i], len(imgs)))
        # each marker's single pad expands to its image's token count; no
        # truncation, the pad runs must stay whole
        parts = text.split(IMAGE_PAD)
        counts = (grid.prod(-1) // merge).tolist()
        text = parts[0] + "".join(
            IMAGE_PAD * n + part for n, part in zip(counts, parts[1:])
        )
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        mm_token_type_ids = (input_ids == image_token_id).int()
        if int(mm_token_type_ids.sum()) != sum(counts):
            raise ValueError("image placeholder count does not match the image grids")
        hidden_states = text_encoder(
            input_ids=input_ids,
            pixel_values=pixels["pixel_values"].to(device, dtype),
            image_grid_thw=grid.to(device),
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        out[i] = _bundles(hidden_states, dtype)[0]
    return out
