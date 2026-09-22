"""Prompt encoding, sequence packing and a preview sampler for Qwen-Image 2.1.

Qwen-Image 2.1 puts text and images in ONE sequence. The Qwen3-VL encoder is
handed the prompt with a `<|image_pad|>` slot per condition image; the
transformer then throws the vision embeddings away and drops the VAE latents
into those slots, four latent tokens per slot (a vision token covers 32 px, a
latent token 16 px). Everything here exists to keep those two views lined up:
`encode_prompt` returns the per-slot mask alongside the embeddings, and
`run_transformer` appends one slot per 2x2 group of target latents before the
call.

Condition images are therefore resized ONCE, to a multiple of 32, and that same
resize feeds both the Qwen3-VL processor and the VAE -- a different size on
either side changes the slot/latent-token counts and the transformer rejects
the sequence.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPT_TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# a Qwen3-VL vision token covers 32 px (patch 16 x merge 2), a latent token 16 px
VISION_TOKEN_PIXELS = 32
VAE_SCALE_FACTOR = 16
LATENT_TOKENS_PER_SLOT = (VISION_TOKEN_PIXELS // VAE_SCALE_FACTOR) ** 2


def _snap(size: float) -> int:
    """Round to the 32 px grid the vision tower and the VAE share."""
    return max(
        VISION_TOKEN_PIXELS, round(size / VISION_TOKEN_PIXELS) * VISION_TOKEN_PIXELS
    )


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """Width/height with the given aspect ratio covering `target_area`, both on
    the 32 px grid."""
    width = math.sqrt(target_area * ratio)
    return _snap(width), _snap(width / ratio)


def prepare_condition_image(
    image: torch.Tensor, max_pixels: int, match: bool = False
) -> torch.Tensor:
    """Put one `(1, C, H, W)` condition image on the 32 px grid, shrinking it
    first if it is over the pixel budget. With `match`, scale it to the budget's
    area in both directions (own aspect kept) instead of only shrinking.

    Deterministic in the image and budget alone, on purpose: the Qwen3-VL pass
    and the VAE pass happen in different calls (`get_prompt_embeds` and
    `get_noise_prediction`) and must agree on the size to the pixel, or the slot
    count and the latent-token count disagree and the sequence is rejected.
    """
    height, width = image.shape[2], image.shape[3]
    if match or height * width > max_pixels:
        new_width, new_height = calculate_dimensions(max_pixels, width / height)
    else:
        new_width, new_height = _snap(width), _snap(height)
    if (new_height, new_width) == (height, width):
        return image
    return F.interpolate(
        image.float(), size=(new_height, new_width), mode="bicubic", antialias=True
    ).clamp(0, 1)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """`(1, C, H, W)` or `(C, H, W)` in [0, 1] -> RGB PIL for the vision tower.

    An alpha channel is composited over white, which is what the checkpoint was
    trained with; the VAE still reads all four channels.
    """
    if image.dim() == 4:
        image = image[0]
    image = image.detach().float().clamp(0, 1).cpu()
    if image.shape[0] == 4:
        alpha = image[3:4]
        image = image[:3] * alpha + (1.0 - alpha)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class QwenImage21PromptEncoder:
    """Turns prompts (+ optional condition images) into the transformer's text
    stream, its validity mask, and the image-slot mask."""

    def __init__(self, text_encoder, processor):
        self.text_encoder = text_encoder
        self.processor = processor
        # How many leading tokens the system turn takes, so it can be dropped
        # from the hidden states. Derived from the processor, not hardcoded.
        system_message = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]
        system_tokens = processor.apply_chat_template(
            system_message, tokenize=True, return_dict=False
        )
        self.drop_index = len(system_tokens[0])
        self.image_token_id = processor.tokenizer.encode("<|image_pad|>")[0]

    def _template(self, num_images: int) -> str:
        if num_images == 0:
            return PROMPT_TEMPLATE
        refs = " ".join(f"<image{i + 1}>{VISION_BLOCK}" for i in range(num_images))
        return PROMPT_TEMPLATE.replace("{}", refs + "{}", 1)

    @torch.no_grad()
    def encode(
        self,
        prompts: Sequence[str],
        images: Optional[Sequence[Sequence[Image.Image]]] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Encode a batch of prompts.

        `images[i]` are the condition images for `prompts[i]`, already resized.
        Returns three per-sample lists, each entry at its own natural length:
        the `(L, dim)` embeddings, an `(L,)` validity mask, and an `(L,)` bool
        mask marking the image slots.
        """
        device = device or self.text_encoder.device
        # Qwen has no bos token, so an empty prompt leaves the encoder nothing to read
        prompts = [p if p else " " for p in prompts]
        images = list(images) if images is not None else [[] for _ in prompts]

        texts = [
            self._template(len(sample_images)).format(prompt)
            for prompt, sample_images in zip(prompts, images)
        ]
        flat_images = [image for sample_images in images for image in sample_images]

        processor_kwargs = {
            # left padding, as the checkpoint was trained with: the padding is
            # dropped either way, but the side decides the positions the
            # encoder sees across prompts of different lengths
            "text": texts,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if flat_images:
            processor_kwargs["images"] = flat_images
        model_inputs = self.processor(**processor_kwargs).to(device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
        }
        if flat_images and hasattr(model_inputs, "pixel_values"):
            forward_kwargs["pixel_values"] = model_inputs.pixel_values
            forward_kwargs["image_grid_thw"] = model_inputs.image_grid_thw
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        # The transformer was trained on the last decoder layer's output BEFORE
        # the encoder's final RMSNorm. transformers 4.x returned that as
        # hidden_states[-1]; 5.x ties that entry to last_hidden_state, so it
        # comes back normalized instead. A forward hook returning the norm's
        # input neutralizes it for this call on either version.
        language_model = getattr(
            self.text_encoder.model, "language_model", self.text_encoder.model
        )
        handle = language_model.norm.register_forward_hook(
            lambda module, args, output: args[0]
        )
        try:
            outputs = self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()
        hidden_states = outputs.hidden_states[-1]

        embeds, masks, slot_masks = [], [], []
        for sample_ids, sample_mask, sample_hidden in zip(
            model_inputs.input_ids, model_inputs.attention_mask, hidden_states
        ):
            valid = sample_mask.bool()
            sample_hidden = sample_hidden[valid][self.drop_index :]
            slots = (sample_ids[valid] == self.image_token_id)[self.drop_index :]
            embeds.append(sample_hidden)
            masks.append(
                torch.ones(sample_hidden.shape[0], dtype=torch.long, device=device)
            )
            slot_masks.append(slots)
        return embeds, masks, slot_masks


def pad_prompt_batch(
    embeds: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    slot_masks: Sequence[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad per-sample prompt tensors into one batch.

    The transformer reads the image-slot layout from row 0 and applies it to
    every row, which is sound because the template puts the slots before the
    prompt text: the prefix is identical across samples and only the padded
    tail differs.
    """
    max_len = max(e.shape[0] for e in embeds)
    batch_size, dim = len(embeds), embeds[0].shape[-1]

    out_embeds = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    out_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    out_slots = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i, (embed, mask, slots) in enumerate(zip(embeds, masks, slot_masks)):
        length = embed.shape[0]
        out_embeds[i, :length] = embed.to(device, dtype)
        out_mask[i, :length] = mask.to(device).bool()
        out_slots[i, :length] = slots.to(device).bool()
    return out_embeds, out_mask, out_slots


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """`(B, C, H, W)` -> `(B, H * W, C)`. 2.1 consumes latents unpatched."""
    batch_size, channels, height, width = latents.shape
    return latents.view(batch_size, channels, height * width).transpose(1, 2)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """`(B, H * W, C)` -> `(B, C, H, W)`."""
    batch_size, _, channels = latents.shape
    return latents.transpose(1, 2).reshape(batch_size, channels, height, width)


def run_transformer(
    transformer,
    latents: torch.Tensor,  # (B, C, h, w) noisy target latents
    timestep: torch.Tensor,  # (B,) in [0, 1]
    prompt_embeds: torch.Tensor,  # (B, L, dim)
    prompt_mask: torch.Tensor,  # (B, L) bool
    image_slot_mask: torch.Tensor,  # (B, L) bool
    condition_latents: Optional[torch.Tensor] = None,  # (B, N, C) packed, in slot order
    condition_shapes: Sequence[tuple[int, int]] = (),  # per condition image (h, w)
    **kwargs,
) -> torch.Tensor:
    """Run the DiT on the joint sequence and return the target velocity as
    `(B, C, h, w)`."""
    batch_size, channels, height, width = latents.shape
    target_tokens = height * width

    hidden_states = pack_latents(latents)
    if condition_latents is not None:
        hidden_states = torch.cat([condition_latents, hidden_states], dim=1)

    # The prompt reserved one slot per 2x2 group of each reference's latent
    # tokens. Embeddings encoded against different references than the ones
    # passed here (a shared blank/unconditional embedding, say) would silently
    # mis-slot the sequence, so check it before the DiT does.
    expected_slots = sum(h * w for h, w in condition_shapes) // LATENT_TOKENS_PER_SLOT
    if int(image_slot_mask[0].sum()) != expected_slots:
        raise ValueError(
            f"the prompt embeddings hold {int(image_slot_mask[0].sum())} reference "
            f"image slots but {len(condition_shapes)} reference image(s) were passed, "
            f"worth {expected_slots} slots. They must be encoded together -- an "
            "embedding cached against other references (caption dropout, a blank "
            "unconditional embedding) cannot be reused here."
        )

    img_shapes = [
        [*[(1, h, w) for h, w in condition_shapes], (1, height, width)]
    ] * batch_size

    # one image slot per 2x2 group of target latent tokens
    img_mask = torch.cat(
        [
            image_slot_mask,
            image_slot_mask.new_ones(
                batch_size, target_tokens // LATENT_TOKENS_PER_SLOT
            ),
        ],
        dim=1,
    )
    # a mask with nothing masked costs the attention backends that reject one
    encoder_mask = None if bool(prompt_mask.all()) else prompt_mask

    output = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=encoder_mask,
        timestep=timestep,
        img_shapes=img_shapes,
        img_mask=img_mask,
        return_dict=False,
        **kwargs,
    )[0]
    return unpack_latents(output[:, -target_tokens:], height, width)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


class QwenImage21Pipeline:
    """Minimal flow-matching sampler for ai-toolkit's preview generation."""

    def __init__(self, model):
        # `model` is the QwenImage2Model, so its encode/decode and config are reused
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds=None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        condition_images: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device, dtype = model.device_torch, model.torch_dtype
        transformer = model.transformer

        latent_height = height // VAE_SCALE_FACTOR
        latent_width = width // VAE_SCALE_FACTOR
        channels = transformer.config.in_channels

        condition_latents, condition_shapes = model.encode_condition_images(
            condition_images
        )

        if latents is None:
            # randn_tensor, not torch.randn: the caller's generator may be a cpu one
            latents = randn_tensor(
                (1, channels, latent_height, latent_width),
                generator=generator,
                device=torch.device(device),
                dtype=torch.float32,
            )
        latents = latents.to(device, dtype=dtype)

        scheduler = model.get_train_scheduler()
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            latent_height * latent_width,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 8192),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 0.9),
        )
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        scheduler.set_begin_index(0)

        # 2.1 is meant to be sampled without guidance; a scale of 1 skips it
        do_cfg = guidance_scale > 1.0 and unconditional_embeds is not None
        cond = model.pad_prompt_embeds(conditional_embeds)
        uncond = model.pad_prompt_embeds(unconditional_embeds) if do_cfg else None

        for timestep in scheduler.timesteps:
            t = timestep.expand(latents.shape[0]).to(device, dtype=dtype) / 1000
            noise_pred = run_transformer(
                transformer,
                latents,
                t,
                *cond,
                condition_latents=condition_latents,
                condition_shapes=condition_shapes,
            )
            if do_cfg:
                uncond_pred = run_transformer(
                    transformer,
                    latents,
                    t,
                    *uncond,
                    condition_latents=condition_latents,
                    condition_shapes=condition_shapes,
                )
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)

            latents = scheduler.step(
                noise_pred.to(torch.float32),
                timestep,
                latents.to(torch.float32),
                return_dict=False,
            )[0].to(dtype)

        return model.decode_to_images(latents)
