"""Fibo 2.0 sampling (training previews / inference engine).

Matches diffusers' BriaFibo2Pipeline: sigmas linspace(1, 1 / steps) under
FLUX.2's resolution-dependent exponential shift (mu 0.5 at 256 tokens to 1.15
at 4096), fp32 Euler on fp32 noise, CFG 5 against the null caption as a
separate pass. Edit references condition both passes as clean latents. Flow
convention is the toolkit's: sigma 1 = noise, the DiT predicts noise - clean.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_GUIDANCE = 5.0
DEFAULT_STEPS = 30
NUM_TRAIN_TIMESTEPS = 1000

# FLUX.2's scheduler config, which Fibo 2.0 shares
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


def fibo2_sigmas(num_steps: int, image_seq_len: int) -> torch.Tensor:
    """Shifted sigmas with the terminal 0 appended."""
    c = scheduler_config
    m = (c["max_shift"] - c["base_shift"]) / (
        c["max_image_seq_len"] - c["base_image_seq_len"]
    )
    mu = image_seq_len * m + c["base_shift"] - m * c["base_image_seq_len"]
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
    return torch.cat([torch.from_numpy(sigmas).float(), torch.zeros(1)])


def stack_text_embeds(
    embeds: List[torch.Tensor], device: torch.device, dtype: torch.dtype
):
    """Per-prompt (L_i, D) bundles -> right-padded (B, S, D) + (B, S) mask,
    the mask None when nothing is padded."""
    length = max(e.shape[0] for e in embeds)
    stacked = torch.stack(
        [F.pad(e.to(device, dtype), (0, 0, 0, length - e.shape[0])) for e in embeds]
    )
    if all(e.shape[0] == length for e in embeds):
        return stacked, None
    mask = torch.stack(
        [torch.arange(length, device=device) < e.shape[0] for e in embeds]
    )
    return stacked, mask


class Fibo2Pipeline:
    """Lightweight sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # the Fibo2Model, for its transformer + decode
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
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        ref_latents: Optional[List[torch.Tensor]] = None,  # each (1, 128, h, w)
        **kwargs,
    ):
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer

        do_cfg = unconditional_embeds is not None and guidance_scale > 1.0
        cond, cond_mask = stack_text_embeds(
            conditional_embeds.text_embeds, device, dtype
        )
        if do_cfg:
            uncond, uncond_mask = stack_text_embeds(
                unconditional_embeds.text_embeds, device, dtype
            )

        shape = (1, transformer.params.in_channels, height // 16, width // 16)
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=torch.float32
            )
        latents = latents.to(device, torch.float32)

        sigmas = fibo2_sigmas(num_inference_steps, shape[2] * shape[3]).to(device)
        for step_index in range(num_inference_steps):
            sigma, sigma_next = sigmas[step_index], sigmas[step_index + 1]
            timestep = (sigma * NUM_TRAIN_TIMESTEPS).reshape(1)
            x = latents.to(dtype)
            pred = transformer(x, timestep, cond, cond_mask, ref_latents).float()
            if do_cfg:
                pred_uncond = transformer(
                    x, timestep, uncond, uncond_mask, ref_latents
                ).float()
                pred = pred_uncond + guidance_scale * (pred - pred_uncond)
            if getattr(model, "sample_step_hook", None) is not None:
                # inference engine preview: x0 estimate
                model._emit_sample_step(
                    latents - sigma * pred, step_index, num_inference_steps
                )
            latents = latents + (sigma_next - sigma) * pred

        images = model.decode_latents(latents, device=device, dtype=torch.float32)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
