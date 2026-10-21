"""MOSS-Image sampling (training previews / inference engine).

Matches the vendor pipeline: shift-5 sigma grid snapped to the nearest of the
1000 training sigmas, fp32 Euler on fp32 noise, CFG 3 whose negative branch
skips DiT block 22 (two forwards per step), empty negative prompt. Flow
convention is the toolkit's: sigma 1 = noise, the DiT predicts noise - clean.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_SHIFT = 5.0
DEFAULT_GUIDANCE = 3.0
DEFAULT_STEPS = 30
DEFAULT_UNCOND_SKIP_LAYERS = (22,)
NUM_TRAIN_TIMESTEPS = 1000


def moss_sigmas(
    num_steps: int,
    shift: float = DEFAULT_SHIFT,
    training_shift: float = 1.0,
    total: int = NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    """The vendor scheduler's sigmas (MOSSImageScheduler.set_timesteps) with
    the terminal 0 appended: a shifted grid snapped to the training grid,
    consecutive duplicates removed (so fewer than num_steps are possible)."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def stack_text_embeds(
    embeds: List[torch.Tensor], device: torch.device, dtype: torch.dtype
):
    """Per-item (L_i, D) prompt states -> right-padded (B, S, D) + (B, S) mask."""
    length = max(e.shape[0] for e in embeds)
    stacked = torch.stack(
        [F.pad(e.to(device, dtype), (0, 0, 0, length - e.shape[0])) for e in embeds]
    )
    mask = torch.stack(
        [
            torch.arange(length, device=device) < e.shape[0]
            for e in embeds
        ]
    )
    return stacked, mask


class MossImagePipeline:
    """Lightweight sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # the MossImageModel, for its transformer + encode/decode
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
        shift: float = DEFAULT_SHIFT,
        uncond_skip_layers: Sequence[int] = DEFAULT_UNCOND_SKIP_LAYERS,
        ref_latents: Optional[List[List[torch.Tensor]]] = None,
        isolate_refs: bool = False,
        **kwargs,
    ):
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer

        do_cfg = unconditional_embeds is not None and guidance_scale != 1.0
        cond, cond_mask = stack_text_embeds(conditional_embeds.text_embeds, device, dtype)
        if do_cfg:
            uncond, uncond_mask = stack_text_embeds(
                unconditional_embeds.text_embeds, device, dtype
            )

        shape = (1, transformer.params.latent_channels, height // 16, width // 16)
        if latents is None:
            # fp32 noise: bf16 noise promoted afterwards cannot regain precision
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents.to(device, torch.float32)

        # edit: references ride in both branches. Isolated references get
        # their K/V computed on the first step and reused after; one cache per
        # branch (the reference RoPE follows the prompt length, and the
        # negative branch skips blocks)
        has_refs = ref_latents is not None and any(len(r) > 0 for r in ref_latents)
        ref_kwargs = {}
        cond_cache = uncond_cache = None
        if has_refs:
            ref_kwargs = {"ref_latents": ref_latents, "isolate_refs": isolate_refs}
            if isolate_refs:
                cond_cache, uncond_cache = {}, {}

        sigmas = moss_sigmas(num_inference_steps, shift=shift).to(device)
        num_steps = len(sigmas) - 1
        for step_index in range(num_steps):
            sigma, sigma_next = sigmas[step_index], sigmas[step_index + 1]
            timestep = (sigma * NUM_TRAIN_TIMESTEPS).reshape(1)
            # CFG combines fp32 predictions even though the DiT runs in bf16
            pred = transformer(
                latents, timestep, cond, cond_mask, ref_kv_cache=cond_cache, **ref_kwargs
            ).float()
            if do_cfg:
                pred_uncond = transformer(
                    latents,
                    timestep,
                    uncond,
                    uncond_mask,
                    skip_layers=uncond_skip_layers,
                    ref_kv_cache=uncond_cache,
                    **ref_kwargs,
                ).float()
                pred = pred_uncond + guidance_scale * (pred - pred_uncond)
            if getattr(model, "sample_step_hook", None) is not None:
                # inference engine preview: x0 estimate
                model._emit_sample_step(latents - sigma * pred, step_index, num_steps)
            latents = latents + (sigma_next - sigma) * pred

        images = model.decode_latents(latents, device=device, dtype=torch.float32)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
