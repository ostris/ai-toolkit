"""Fibo 2.0 edit references: sizing, crops, and Qwen3-VL's copies.

Bria's preprocessing (diffusers BriaFibo2ImageProcessor): a reference is
scaled to cover its size (Lanczos, ceil) and center cropped; the VAE encodes
the crop, Qwen3-VL sees it thumbnailed to at most 768 px a side on the
processor's 32 px grid. Vendor sizes: a single reference crops to the output
size (the edit lines up with it pixel for pixel), several references each to
the ~1 MP size Fibo 2.0 was trained at for their own aspect ratio.

match_target_res (default on): a single reference crops to the target size,
several each to the target's area at their own aspect. Off: every reference
at its own trained ~1 MP size, whatever the target.
"""

import math
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image

TRAINED_AREA = 1024 * 1024
VLM_MAX_SIDE = 768
# Qwen3-VL processor grid for Fibo 2.0 (ComfyUI's port of Bria's processor config)
VLM_GRID = 32
VLM_MIN_PIXELS = 3136
VLM_MAX_PIXELS = 1605632


@lru_cache(maxsize=256)
def trained_size(aspect_ratio: float, area: int = TRAINED_AREA) -> Tuple[int, int]:
    """(width, height): of the multiples of 16 (>= 64 a side) within 10% of
    area, the least cropping for this aspect, then the closest to area."""
    sizes = [
        (w, h)
        for w in range(64, 2 * area // 64, 16)
        for h in range(
            max(64, 16 * math.ceil(0.9 * area / w / 16)),
            16 * math.floor(1.1 * area / w / 16) + 1,
            16,
        )
    ]
    return min(
        sizes,
        key=lambda s: (
            round(1 - min(s[0] / s[1] / aspect_ratio, aspect_ratio * s[1] / s[0]), 9),
            abs(s[0] * s[1] - area),
            -s[0],
        ),
    )


def reference_sizes(
    images: Sequence[Image.Image],
    target_size: Optional[Tuple[int, int]],
    match_target_res: bool,
) -> List[Tuple[int, int]]:
    """(width, height) each reference is cropped to."""
    if match_target_res and target_size is not None:
        width, height = target_size
        if len(images) == 1:
            return [(width, height)]
        return [trained_size(img.width / img.height, width * height) for img in images]
    return [trained_size(img.width / img.height) for img in images]


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """(C, H, W) / (1, C, H, W) in [0, 1] -> RGB PIL."""
    if image.dim() == 4:
        image = image[0]
    array = (image[:3].float().clamp(0, 1) * 255).round().to(torch.uint8)
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy())


def resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(height / image.height, width / image.width)
    resized_width, resized_height = (
        math.ceil(image.width * scale),
        math.ceil(image.height * scale),
    )
    image = image.resize((resized_width, resized_height), resample=Image.LANCZOS)
    left, top = (resized_width - width) // 2, (resized_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def vlm_image(crop: Image.Image) -> Image.Image:
    """Qwen3-VL's copy of a reference crop: thumbnailed, then on the 32 px
    grid (pixel budget clamped like the processor's smart resize)."""
    image = crop.copy()
    image.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE))
    h, w = image.height, image.width
    height, width = round(h / VLM_GRID) * VLM_GRID, round(w / VLM_GRID) * VLM_GRID
    if height * width > VLM_MAX_PIXELS:
        beta = math.sqrt(h * w / VLM_MAX_PIXELS)
        height = max(VLM_GRID, math.floor(h / beta / VLM_GRID) * VLM_GRID)
        width = max(VLM_GRID, math.floor(w / beta / VLM_GRID) * VLM_GRID)
    elif height * width < VLM_MIN_PIXELS:
        beta = math.sqrt(VLM_MIN_PIXELS / (h * w))
        height = math.ceil(h * beta / VLM_GRID) * VLM_GRID
        width = math.ceil(w * beta / VLM_GRID) * VLM_GRID
    if (width, height) != (w, h):
        image = image.resize((width, height), resample=Image.BICUBIC)
    return image


def prepare_references(
    images: Sequence[torch.Tensor],
    target_size: Optional[Tuple[int, int]],
    match_target_res: bool,
) -> List[Image.Image]:
    """[0, 1] reference tensors -> the crops the VAE and Qwen3-VL see."""
    pils = [tensor_to_pil(image) for image in images]
    sizes = reference_sizes(pils, target_size, match_target_res)
    return [resize_and_crop(img, w, h) for img, (w, h) in zip(pils, sizes)]


def drop_repeats(images):
    """ctrl_img_1 defaults to ctrl_img, so a single reference can arrive twice."""
    kept = []
    for image in images:
        previous = kept[-1] if kept else None
        if (
            previous is not None
            and previous.shape == image.shape
            and torch.equal(previous, image)
        ):
            continue
        kept.append(image)
    return kept
