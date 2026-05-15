"""
HiDream-O1 reference-image pipeline helpers for training.

Architecture recap (from the technical report and inference code):
  - The model has NO VAE; it works on raw 32×32 RGB patches (PATCH_SIZE=32).
  - Special tokens used during reference-image conditioning:
        <|bor_token|>  – beginning-of-reference block  (one per ref image)
        <|eor_token|>  – end-of-reference block         (one per ref image)
        <|bot_token|>  – beginning-of-target            (single, after all refs)
        <|tms_token|>  – timestep placeholder           (one per sample)
        <|boi_token|>  – beginning-of-image (generation target)
  - Sequence layout for a sample with N reference images:
        [text_tokens] [bor] [REF_PATCHES_1] [eor]
                      [bor] [REF_PATCHES_2] [eor]
                      ...
                      [bot]
                      [tms]
                      [boi] [TARGET_PATCHES…]
  - Reference patches are AR (causal) tokens; target patches are generative tokens.
  - All image pixels are encoded through model.x_embedder (BottleneckPatchEmbed),
    the same linear that handles target patches.

This module provides:
  - `patchify_image`            – convert a PIL/tensor image to flat patches
  - `build_ref_input_ids`       – build the text+reference-token prefix input_ids
  - `build_ref_conditioning`    – full conditioning dict for one training sample
  - `collate_ref_batch`         – batch-collate with left-padding for variable refs
  - `encode_ref_prompt`         – pipeline-level helper (wraps HiDreamO1Pipeline)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import einops
import torch
import torchvision.transforms.v2 as T
from PIL import Image

# The canonical patch size used by HiDream-O1
PATCH_SIZE: int = 32
TIMESTEP_TOKEN_NUM: int = 1

# ---------------------------------------------------------------------------
# Image → patch tensor helpers
# ---------------------------------------------------------------------------

_IMG_TRANSFORM = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.5], [0.5]),  # → [-1, 1]
    ]
)


def round_to_patch(dim: int, patch: int = PATCH_SIZE) -> int:
    """Round *dim* down to the nearest multiple of *patch* (minimum = *patch*)."""
    return max(patch, int(dim // patch * patch))


def resize_image_to_patches(
    image: Image.Image,
    max_patches_per_side: int = 16,
    patch_size: int = PATCH_SIZE,
) -> Image.Image:
    """
    Resize *image* so that both dimensions are multiples of *patch_size* and
    neither side exceeds *max_patches_per_side * patch_size* pixels.
    Aspect ratio is preserved (shrink-only, bilinear).
    """
    max_px = max_patches_per_side * patch_size
    w, h = image.size
    scale = min(max_px / w, max_px / h, 1.0)
    nw = round_to_patch(int(w * scale), patch_size)
    nh = round_to_patch(int(h * scale), patch_size)
    if (nw, nh) != (w, h):
        image = image.resize((nw, nh), Image.BILINEAR)
    return image


def patchify_image(
    image: Union[Image.Image, torch.Tensor],
    patch_size: int = PATCH_SIZE,
    max_patches_per_side: int = 16,
) -> torch.Tensor:
    """
    Convert *image* (PIL or CHW float tensor in [-1,1]) to a flat patch tensor
    of shape ``(H/p * W/p, C*p*p)``.

    This matches the layout expected by ``model.x_embedder`` (BottleneckPatchEmbed)
    and the target-patch patchification in the forward pass.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = resize_image_to_patches(image, max_patches_per_side, patch_size)
        img_t: torch.Tensor = _IMG_TRANSFORM(image)  # (3, H, W)
    else:
        # Assume CHW tensor already in [-1,1]; snap dimensions to patch grid
        c, h, w = image.shape
        nh = round_to_patch(h, patch_size)
        nw = round_to_patch(w, patch_size)
        if (nh, nw) != (h, w):
            img_t = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
            ).squeeze(0)
        else:
            img_t = image

    # (C, H, W) → (H/p * W/p, C*p*p)
    patches = einops.rearrange(
        img_t,
        "C (H p1) (W p2) -> (H W) (C p1 p2)",
        p1=patch_size,
        p2=patch_size,
    )
    return patches  # (num_patches, C*p*p)


# ---------------------------------------------------------------------------
# Token-sequence builders
# ---------------------------------------------------------------------------

def _get_token_id(tokenizer, attr: str) -> int:
    """Safely retrieve a special-token id from the tokenizer."""
    token = getattr(tokenizer, attr, None)
    if token is None:
        raise AttributeError(
            f"Tokenizer is missing attribute '{attr}'. "
            "Call add_special_tokens(tokenizer) before building conditioning."
        )
    ids = tokenizer.encode(token, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Token '{token}' not found in tokenizer vocabulary.")
    return ids[0]


def build_ref_input_ids(
    text_input_ids: torch.Tensor,
    ref_patch_counts: List[int],
    tokenizer,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build the full prefix input_ids tensor for a sample that has reference images.

    Layout:
        [text_tokens] [bor] [REF_1_image_tokens] [eor]
                      [bor] [REF_2_image_tokens] [eor]
                      ...
                      [bot]
                      [tms * TIMESTEP_TOKEN_NUM]
                      [boi] [TARGET image_tokens – appended by pipeline later]

    Parameters
    ----------
    text_input_ids : (1, txt_len) or (txt_len,)
        Text tokens already tokenized (with chat template applied and boi/tms suffix
        stripped – i.e. just the plain chat-templated text part).
    ref_patch_counts : list[int]
        Number of patches for each reference image (``H/p * W/p``).
    tokenizer : any
        Tokenizer with ``bor_token``, ``eor_token``, ``bot_token``, ``tms_token``,
        ``boi_token`` attributes attached.

    Returns
    -------
    input_ids : (1, total_len) int64 tensor
    """
    if text_input_ids.dim() == 1:
        text_input_ids = text_input_ids.unsqueeze(0)
    dev = device or text_input_ids.device

    # Retrieve special token ids
    bor_id = _get_token_id(tokenizer, "bor_token")
    eor_id = _get_token_id(tokenizer, "eor_token")
    bot_id = _get_token_id(tokenizer, "bot_token")
    tms_id = _get_token_id(tokenizer, "tms_token")
    boi_id = _get_token_id(tokenizer, "boi_token")
    # image_token_id is used as the placeholder inside ref/target windows
    img_id = tokenizer.encode(
        "<|image_pad|>", add_special_tokens=False
    )
    # Fallback: use image_token_id from model config if available
    if not img_id:
        img_id = [tokenizer.encode("<|vision_pad|>", add_special_tokens=False)]
    img_id = img_id[0] if img_id else 151655  # default from qwen3-vl config

    # Build the reference blocks: [bor, img×N, eor] for each ref
    ref_blocks: List[torch.Tensor] = []
    for n_patches in ref_patch_counts:
        block = torch.tensor(
            [bor_id] + [img_id] * n_patches + [eor_id],
            dtype=torch.long,
            device=dev,
        ).unsqueeze(0)  # (1, n_patches+2)
        ref_blocks.append(block)

    # [bot] [tms×TIMESTEP_TOKEN_NUM]
    control_tokens = torch.tensor(
        [bot_id] + [tms_id] * TIMESTEP_TOKEN_NUM,
        dtype=torch.long,
        device=dev,
    ).unsqueeze(0)  # (1, 1+TIMESTEP_TOKEN_NUM)

    parts = [text_input_ids.to(dev)] + ref_blocks + [control_tokens]
    return torch.cat(parts, dim=1)  # (1, total_prefix_len)


# ---------------------------------------------------------------------------
# Full conditioning builder (mirrors _build_t2i_sample_from_input_ids in pipeline.py
# but extended to handle reference image pixel tokens)
# ---------------------------------------------------------------------------

from .pipeline import (  # noqa: E402  (relative import – same package)
    _get_rope_index_t2i,
    PATCH_SIZE as _P,
    TIMESTEP_TOKEN_NUM as _TMN,
)


def build_ref_conditioning(
    text_input_ids: torch.Tensor,
    ref_patches: List[torch.Tensor],
    height: int,
    width: int,
    model_config,
    tokenizer,
    *,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Build the complete per-sample conditioning dict for training with reference images.

    This is the training analogue of HiDreamO1Pipeline.build_conditioning_sample,
    extended to inject reference-image pixel patches into the sequence.

    Parameters
    ----------
    text_input_ids : (1, txt_len) int64
        Plain text tokens (chat-template applied, **no** boi/tms suffix yet).
    ref_patches : list of (n_patches_i, C*p*p) float tensors
        Patchified reference images (from ``patchify_image``).
    height, width : int
        Target image pixel dimensions (must be multiples of PATCH_SIZE).
    model_config : Qwen3VLConfig (or compatible)
        The model's config; needs image_token_id, video_token_id,
        vision_start_token_id.
    tokenizer
        Tokenizer with the HiDream special tokens attached.

    Returns
    -------
    dict with keys:
        input_ids         (1, prefix_len)      int64
        position_ids      (3, 1, total_len)    int64
        token_types       (1, total_len)        int64  0=AR, 1=gen
        vinput_mask       (1, total_len)        bool   True=generation patches
        ref_vinput_mask   (1, total_len)        bool   True=reference patches
        n_ref_patches     int   total reference patch count
        n_tgt_patches     int   target patch count  (h//p * w//p)
    """
    if text_input_ids.dim() == 1:
        text_input_ids = text_input_ids.unsqueeze(0)

    dev = device or text_input_ids.device
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id

    # Token ids
    bor_id = _get_token_id(tokenizer, "bor_token")
    eor_id = _get_token_id(tokenizer, "eor_token")
    bot_id = _get_token_id(tokenizer, "bot_token")
    tms_id = _get_token_id(tokenizer, "tms_token")
    boi_id = _get_token_id(tokenizer, "boi_token")

    # -----------------------------------------------------------------------
    # Patch counts
    # -----------------------------------------------------------------------
    ref_patch_counts = [p.shape[0] for p in ref_patches]
    n_ref_patches_total = sum(ref_patch_counts)
    h_patches = height // PATCH_SIZE
    w_patches = width // PATCH_SIZE
    n_tgt_patches = h_patches * w_patches

    # -----------------------------------------------------------------------
    # Build prefix input_ids  [text | bor img×n eor | … | bot tms×1]
    # -----------------------------------------------------------------------
    prefix_ids = build_ref_input_ids(
        text_input_ids,
        ref_patch_counts,
        tokenizer,
        device=dev,
    )  # (1, prefix_len)

    # -----------------------------------------------------------------------
    # Append generation image tokens  [boi | img×n_tgt]
    # (mirrors what _build_t2i_sample_from_input_ids does for the target)
    # -----------------------------------------------------------------------
    gen_tokens = torch.zeros(
        (1, n_tgt_patches), dtype=torch.long, device=dev
    ) + image_token_id
    gen_tokens[0, 0] = vision_start_token_id  # first token is vision_start per pipeline

    full_ids = torch.cat([prefix_ids, gen_tokens], dim=1)  # (1, total_len)
    total_len = full_ids.shape[1]

    # -----------------------------------------------------------------------
    # Compute image_grid_thw for the TARGET (used by rope indexing)
    # -----------------------------------------------------------------------
    image_grid_thw = torch.tensor(
        [1, h_patches, w_patches], dtype=torch.int64, device=dev
    ).unsqueeze(0)  # (1, 3)

    # -----------------------------------------------------------------------
    # Position IDs (3D RoPE as in _get_rope_index_t2i from pipeline.py)
    # We only pass the TARGET vision tokens to the rope helper (reference
    # tokens use text-style sequential positions).
    # -----------------------------------------------------------------------
    # We need to build position ids manually here because we have reference
    # image tokens in-between which the standard helper doesn't know about.
    # Strategy:
    #   1. Compute positions for [text | ref_blocks | bot+tms] as pure text (sequential).
    #   2. For the target vision window, use the 2D grid positions from rope helper.
    position_ids = _build_position_ids_with_refs(
        full_ids=full_ids,
        prefix_ids=prefix_ids,
        ref_patch_counts=ref_patch_counts,
        image_grid_thw=image_grid_thw,
        model_config=model_config,
        vision_start_token_id=vision_start_token_id,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )  # (3, 1, total_len)

    # -----------------------------------------------------------------------
    # token_types: 0 = AR (text + reference patches), 1 = generation patches
    # -----------------------------------------------------------------------
    token_types = torch.zeros((1, total_len), dtype=torch.long, device=dev)
    prefix_len = prefix_ids.shape[1]
    # Generation window starts right after the prefix
    token_types[0, prefix_len:] = 1  # target patches

    # -----------------------------------------------------------------------
    # vinput_mask: True where generation target patches live
    # -----------------------------------------------------------------------
    vinput_mask = token_types == 1  # (1, total_len)

    # -----------------------------------------------------------------------
    # ref_vinput_mask: True where reference patches live inside the prefix
    # -----------------------------------------------------------------------
    ref_vinput_mask = torch.zeros((1, total_len), dtype=torch.bool, device=dev)
    # Walk through prefix_ids to locate reference patch positions
    ids_flat = prefix_ids[0].tolist()
    ref_mask_list = []
    i = 0
    while i < len(ids_flat):
        if ids_flat[i] == bor_id:
            # skip bor token itself
            i += 1
            start = i
            while i < len(ids_flat) and ids_flat[i] != eor_id:
                i += 1
            # positions [start, i) are reference patch slots
            ref_mask_list.extend(range(start, i))
            i += 1  # skip eor
        else:
            i += 1
    for pos in ref_mask_list:
        ref_vinput_mask[0, pos] = True

    # -----------------------------------------------------------------------
    # binary token_types (0/1) already computed; pass as token_types_bin
    # -----------------------------------------------------------------------
    return {
        "input_ids": prefix_ids,           # (1, prefix_len)
        "position_ids": position_ids,      # (3, 1, total_len)
        "token_types": token_types,        # (1, total_len)
        "vinput_mask": vinput_mask,        # (1, total_len) bool
        "ref_vinput_mask": ref_vinput_mask,  # (1, total_len) bool
        "n_ref_patches": n_ref_patches_total,
        "n_tgt_patches": n_tgt_patches,
    }


def _build_position_ids_with_refs(
    full_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    ref_patch_counts: List[int],
    image_grid_thw: torch.Tensor,
    model_config,
    vision_start_token_id: int,
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    """
    Build 3D RoPE position ids for the full sequence including reference tokens.

    Reference patch tokens are assigned sequential text-like positions so the
    model can attend to them with causal attention.  The target image window
    gets proper 2D H×W grid positions as in the standard pipeline.
    """
    dev = full_ids.device
    bs = full_ids.shape[0]
    total_len = full_ids.shape[1]
    prefix_len = prefix_ids.shape[1]

    position_ids = torch.ones(
        3, bs, total_len, dtype=torch.long, device=dev
    )

    for b in range(bs):
        ids_row = full_ids[b]

        # --- Assign sequential positions to the full prefix (text + ref tokens)
        seq_pos = torch.arange(prefix_len, device=dev)
        position_ids[:, b, :prefix_len] = seq_pos.unsqueeze(0).expand(3, -1)

        # --- Assign grid positions to the target window
        h_patches = image_grid_thw[0, 1].item()
        w_patches = image_grid_thw[0, 2].item()
        n_tgt = h_patches * w_patches
        spatial_merge = getattr(model_config.vision_config, "spatial_merge_size", 1)
        llm_h = int(h_patches // spatial_merge)
        llm_w = int(w_patches // spatial_merge)

        # t_index is always 0 for single-frame images
        h_index = (
            torch.arange(llm_h, device=dev)
            .view(1, -1, 1)
            .expand(1, llm_h, llm_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_w, device=dev)
            .view(1, 1, -1)
            .expand(1, llm_h, llm_w)
            .flatten()
        )
        t_index = torch.zeros_like(h_index)

        # Base position for the target: continue from the last prefix position
        base = prefix_len
        position_ids[0, b, prefix_len : prefix_len + n_tgt] = base + t_index
        position_ids[1, b, prefix_len : prefix_len + n_tgt] = base + h_index
        position_ids[2, b, prefix_len : prefix_len + n_tgt] = base + w_index

    return position_ids


# ---------------------------------------------------------------------------
# Batch collation for variable-length reference sets
# ---------------------------------------------------------------------------

def collate_ref_batch(
    samples: List[dict],
    tokenizer,
    pad_token_id: int = 0,
) -> dict:
    """
    Collate a list of per-sample conditioning dicts (from build_ref_conditioning)
    into a batch with left-padding on the text/prefix dimension.

    Each sample dict must have:
        input_ids       (1, prefix_len_i)
        position_ids    (3, 1, total_len_i)
        token_types     (1, total_len_i)
        vinput_mask     (1, total_len_i)
        ref_vinput_mask (1, total_len_i)
        ref_patches     list[tensor(n_i, dim)]   – pixel patches per ref image
        target_patches  tensor(n_tgt, dim)        – target image pixel patches

    Returns a dict ready for HidreamO1Model.get_noise_prediction / forward.
    """
    bs = len(samples)
    device = samples[0]["input_ids"].device

    # -----------------------------------------------------------------------
    # Find max sequence lengths
    # -----------------------------------------------------------------------
    max_prefix_len = max(s["input_ids"].shape[1] for s in samples)
    max_total_len = max(s["token_types"].shape[1] for s in samples)

    input_ids_out = torch.full(
        (bs, max_prefix_len), pad_token_id, dtype=torch.long, device=device
    )
    position_ids_out = torch.ones(
        (3, bs, max_total_len), dtype=torch.long, device=device
    )
    token_types_out = torch.zeros(
        (bs, max_total_len), dtype=torch.long, device=device
    )
    vinput_mask_out = torch.zeros(
        (bs, max_total_len), dtype=torch.bool, device=device
    )
    ref_vinput_mask_out = torch.zeros(
        (bs, max_total_len), dtype=torch.bool, device=device
    )
    attention_mask_out = torch.zeros(
        (bs, max_total_len), dtype=torch.long, device=device
    )

    for i, s in enumerate(samples):
        pl = s["input_ids"].shape[1]
        tl = s["token_types"].shape[1]
        pad_len = max_prefix_len - pl
        total_pad = max_total_len - tl

        # Left-pad the prefix
        input_ids_out[i, pad_len:] = s["input_ids"][0]
        # Left-pad positions
        position_ids_out[:, i, total_pad:] = s["position_ids"][:, 0, :]
        # Left-pad masks
        token_types_out[i, total_pad:] = s["token_types"][0]
        vinput_mask_out[i, total_pad:] = s["vinput_mask"][0]
        ref_vinput_mask_out[i, total_pad:] = s["ref_vinput_mask"][0]
        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask_out[i, total_pad:] = 1

    return {
        "input_ids": input_ids_out,
        "position_ids": position_ids_out,
        "token_types": token_types_out,
        "vinput_mask": vinput_mask_out,
        "ref_vinput_mask": ref_vinput_mask_out,
        "attention_mask": attention_mask_out,
    }


# ---------------------------------------------------------------------------
# AdvancedPromptEmbeds helper for ref-image training
# ---------------------------------------------------------------------------

def encode_ref_prompt(
    pipeline,
    prompt: Union[str, List[str]],
    ref_images: List[Union[Image.Image, torch.Tensor]],
    height: int,
    width: int,
    max_patches_per_side: int = 16,
) -> Tuple["AdvancedPromptEmbeds", List[torch.Tensor]]:  # noqa: F821
    """
    Encode a prompt + list of reference images into the AdvancedPromptEmbeds
    format used by HidreamO1Model, and return the patchified reference tensors.

    Parameters
    ----------
    pipeline : HiDreamO1Pipeline
        The pipeline (with tokenizer and model attached).
    prompt : str or list[str]
        Text prompt(s).  If a list, batch size must match len(ref_images) grouping.
    ref_images : list of PIL.Image or CHW float tensors
        Reference images for **one** training sample.
    height, width : int
        Target output dimensions (multiples of PATCH_SIZE).
    max_patches_per_side : int
        Maximum patches per side when resizing reference images.

    Returns
    -------
    embeds : AdvancedPromptEmbeds
        Token-id embeds as expected by HidreamO1Model.get_prompt_embeds.
    ref_patches_list : list[Tensor]
        One (n_i, C*p*p) float32 tensor per reference image, ready to be
        passed to _forward_generation as additional vinputs context.
    """
    from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds

    if isinstance(prompt, str):
        prompt = [prompt]

    tokenizer = pipeline.tokenizer
    ref_patches_list = [
        patchify_image(img, patch_size=PATCH_SIZE, max_patches_per_side=max_patches_per_side)
        for img in ref_images
    ]

    token_list = []
    for p in prompt:
        ids = pipeline.encode_prompt(p)  # (1, seq_len) – text + boi + tms
        token_list.append(ids)

    pe = AdvancedPromptEmbeds(text_embeds=token_list)
    pe._frozen_dtype_keys = ["text_embeds"]
    return pe, ref_patches_list
