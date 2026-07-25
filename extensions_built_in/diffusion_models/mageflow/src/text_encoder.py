"""Qwen3-VL prompt encoding for Mage-Flow.

The reference implementation (mage_flow/models/modules/text_encoder.py) packs
many prompts into one varlen forward; per-sequence the math reduces to a plain
causal forward with positions ``arange(L)`` replicated on every mrope axis and
the leading system-prompt tokens dropped from the output. ai-toolkit encodes
prompts one at a time (embeds are cached per caption), so that per-sequence
form is implemented here directly on the stock HF Qwen3-VL module.

The DiT consumes the final hidden states ``[L - drop_idx, 2560]``; the pooled
"vec" of the reference is unused by the released checkpoints (vec_type null —
the transformer adds a zero vector), so it is not computed.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


# Prompt templates from the reference (mage_flow/models/utils.py). ``start_idx``
# is the number of leading (system prompt) tokens dropped from the encoder
# output before it conditions the DiT.
PROMPT_TEMPLATE = {
    "mage-flow": {
        "template": (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "start_idx": 34,
    },
    "mage-flow-edit": {
        "template": (
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture,"
            " objects, background), then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency with the original "
            "input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "start_idx": 64,
    },
}

# Fixed image placeholder used at edit training time (one per reference image).
EDIT_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def edit_prompt_body(instruction: str, num_refs: int) -> str:
    """Training-time multi-reference prompt body:
    ``Image 1: <ph>Image 2: <ph>…{instruction}``."""
    prefix = "".join(
        f"Image {j}: {EDIT_IMAGE_PLACEHOLDER}" for j in range(1, num_refs + 1)
    )
    return prefix + instruction


def patch_qwen_vl_patch_embed(model):
    """Qwen-VL's vision patch_embed is a Conv3d whose kernel == stride, i.e. a plain
    linear projection of each flattened patch. bf16 Conv3d has no fast cuDNN kernel and
    falls back to a slow, GPU-underutilizing path. Swap it for the equivalent F.linear
    (a GEMM). The weight is read lazily so this survives later .to(device)/dtype moves.
    Returns the number of patch_embed modules patched. (Same patch as the krea2
    extension / Qwen3VLCaptioner.)"""
    patched = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(
            proj.stride
        ):

            def fast_forward(hidden_states, _proj=proj):
                w = _proj.weight.reshape(_proj.weight.shape[0], -1)
                x = hidden_states.view(-1, w.shape[1]).to(w.dtype)
                return F.linear(x, w, _proj.bias)

            module.forward = fast_forward
            patched += 1
    return patched


def resize_vl_images(
    images: List[torch.Tensor], max_long_edge: int = 384
) -> List["PIL.Image.Image"]:
    """Prepare reference images for the Qwen3-VL conditioning pass.

    Matches the reference's ``_resize_long_edge`` (cap the long edge at
    ``max_long_edge``, preserving aspect ratio, BICUBIC, never upscale) — the
    MLLM only needs a coarse view of the reference; full-resolution detail
    flows through the VAE reference latents. Input tensors are ``(C, H, W)``
    or ``(1, C, H, W)`` in [0, 1]; output is PIL for the Qwen processor.
    """
    from torchvision.transforms.functional import to_pil_image

    out = []
    for img in images:
        if img.dim() == 4:
            img = img[0]
        img = img.float().clamp(0, 1).cpu()
        pil = to_pil_image(img)
        if max_long_edge and max_long_edge > 0:
            w, h = pil.size
            long_edge = max(w, h)
            if long_edge > max_long_edge:
                scale = max_long_edge / long_edge
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                pil = pil.resize((new_w, new_h), resample=3)  # PIL.Image.BICUBIC
        out.append(pil)
    return out


@torch.no_grad()
def encode_mageflow_prompt(
    text_encoder,  # Qwen3VLForConditionalGeneration
    tokenizer,
    prompt: str,
    template_name: str = "mage-flow",
    max_length: int = 2048,
    images: Optional[list] = None,  # list of PIL images (already VL-resized)
    processor=None,  # AutoProcessor, required when images are given
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Encode one prompt (optionally with reference images) to DiT conditioning.

    Returns ``(L - start_idx, 2560)`` final hidden states with the system
    prompt dropped, matching the reference ``TextEncoder.forward`` txt output.
    """
    info = PROMPT_TEMPLATE[template_name]
    template = info["template"]
    drop_idx = int(info["start_idx"])
    device = next(text_encoder.parameters()).device

    if images:
        if processor is None:
            raise ValueError(
                "encoding reference images requires the Qwen3-VL AutoProcessor"
            )
        formatted = template.format(edit_prompt_body(prompt, len(images)))
        vl = processor(text=[formatted], images=list(images), return_tensors="pt")
        input_ids = vl["input_ids"].to(device)
        extra = {}
        for key in ("pixel_values", "image_grid_thw"):
            if vl.get(key) is not None:
                extra[key] = vl[key].to(device)
    else:
        formatted = template.format(prompt)
        input_ids = tokenizer(
            formatted,
            max_length=max_length + drop_idx,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        extra = {}

    # The reference packed encoder gives every sequence positions arange(L)
    # replicated on all mrope axes (even for image tokens); a 2D position_ids
    # is expanded to exactly that by the HF text model.
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    out = text_encoder.model(
        input_ids=input_ids,
        position_ids=position_ids,
        **extra,
    )
    hidden = out.last_hidden_state[0]  # (L, D)
    return hidden[drop_idx:].to(dtype)
