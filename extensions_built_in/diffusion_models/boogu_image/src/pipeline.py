"""Packing / sampling helpers for Boogu-Image (base T2I).

This module glues the Qwen3-VL instruction features and the image latents into
the call the Boogu transformer expects, and provides a minimal flow-matching
sampler used to render preview images during training.

Time convention
---------------
Boogu's native flow time is ``t in [0, 1]`` with ``t=0`` pure noise and ``t=1``
clean; the transformer predicts ``clean - noise``. ai-toolkit's scheduler uses
the opposite convention (``t=1`` noise, velocity ``noise - clean``). The
conversion lives in ``BooguImageModel.get_noise_prediction``; this sampler runs
entirely in Boogu's native domain via :func:`run_boogu_transformer`.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from .transformer import BooguImageTransformer2DModel


# ---------------------------------------------------------------------------
# Instruction feature padding.
# ---------------------------------------------------------------------------


def pad_instruction_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad per-sample ``(L_i, D)`` instruction features into a batch.

    Captions are stored per-sample at their natural length and only padded to the
    batch max here, right before the model call. Returns ``(features (B, L, D),
    attention_mask (B, L))`` with the mask 1 for real tokens, 0 for padding.
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, f in enumerate(features_list):
        n = f.shape[0]
        features[i, :n] = f.to(device, dtype)
        mask[i, :n] = 1
    return features, mask


# ---------------------------------------------------------------------------
# Time-shift schedule (mirrors the released Boogu base scheduler: v1 shift).
# ---------------------------------------------------------------------------


def _lin_shift(
    num_tokens: float,
    x1: float = 256.0,
    y1: float = 0.5,
    x2: float = 4096.0,
    y2: float = 1.15,
) -> float:
    """Linear token-count -> mu mapping (Boogu base_shift/max_shift defaults)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * num_tokens + b


def boogu_time_schedule(
    num_steps: int,
    num_patch_tokens: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Boogu native-domain timesteps (0=noise .. 1=clean) with v1 time shift.

    Returns a length ``num_steps + 1`` tensor; the trailing ``1.0`` is the clean
    endpoint, matching the ``_timesteps`` tail in the reference scheduler.
    """
    t_arr = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)[:-1]

    mu = _lin_shift(max(1, int(num_patch_tokens)))
    eps = 1e-8
    t1 = np.clip(1.0 - t_arr, eps, 1.0 - eps)
    num = math.exp(mu)
    denom = num + (1.0 / t1 - 1.0)
    t_arr = (1.0 - num / denom).astype(np.float32)

    times = np.concatenate([t_arr, np.ones(1, dtype=np.float32)])
    return torch.from_numpy(times).to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Transformer call (Boogu native time domain).
# ---------------------------------------------------------------------------


def run_boogu_transformer(
    transformer: BooguImageTransformer2DModel,
    latents: torch.Tensor,  # (B, 16, H, W)
    boogu_t: torch.Tensor,  # (B,) in [0, 1], 0=noise, 1=clean
    instruction_features: torch.Tensor,  # (B, L, instruction_feat_dim)
    instruction_mask: torch.Tensor,  # (B, L) 1 for real tokens
    freqs_cis,  # precomputed per-axis rotary tables
    ref_image_hidden_states=None,  # edit/TI2I: List[List[(16, H, W)]] per batch item
) -> torch.Tensor:
    """Run the transformer and return the raw model velocity (``clean - noise``).

    Shapes pass straight through: the prediction comes back as ``(B, 16, H, W)``
    in the same latent layout as ``latents``. ``ref_image_hidden_states`` stays
    ``None`` for the base T2I model and carries reference-image VAE latents for
    the edit (TI2I) model.
    """
    out = transformer(
        hidden_states=latents,
        timestep=boogu_t,
        instruction_hidden_states=instruction_features,
        freqs_cis=freqs_cis,
        instruction_attention_mask=instruction_mask,
        ref_image_hidden_states=ref_image_hidden_states,
        return_dict=False,
    )
    return out


# ---------------------------------------------------------------------------
# Minimal sampling pipeline (for training previews).
# ---------------------------------------------------------------------------


class BooguImagePipeline:
    """Lightweight flow-matching sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # ``model`` is the BooguImageModel so we can reuse its encode/decode and
        # latent helpers without duplicating state.
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        ref_latents=None,  # edit/TI2I: List[List[(16, H, W)]] reference VAE latents
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer
        patch = model.patch_size
        ae_scale = model.vae_scale_factor  # 8

        latent_channels = transformer.config.in_channels
        h_lat = height // ae_scale
        w_lat = width // ae_scale
        num_patch_tokens = (h_lat // patch) * (w_lat // patch)

        freqs_cis = model.get_freqs_cis()

        do_cfg = guidance_scale > 1.0

        if latents is None:
            shape = (1, latent_channels, h_lat, w_lat)
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=torch.float32
            )
        # In Boogu's domain t=0 is pure noise, so the initial latent IS the noise.
        latents = latents.to(device, dtype=torch.float32)

        cond_feats, cond_mask = pad_instruction_features(
            conditional_embeds.text_embeds, device, dtype
        )
        if do_cfg:
            uncond_feats, uncond_mask = pad_instruction_features(
                unconditional_embeds.text_embeds, device, dtype
            )

        times = boogu_time_schedule(num_inference_steps, num_patch_tokens, device)

        for t, t_next in zip(times[:-1], times[1:]):
            boogu_t = t.expand(latents.shape[0])
            v_cond = run_boogu_transformer(
                transformer,
                latents.to(dtype),
                boogu_t,
                cond_feats,
                cond_mask,
                freqs_cis,
                ref_image_hidden_states=ref_latents,
            )
            if do_cfg:
                v_uncond = run_boogu_transformer(
                    transformer,
                    latents.to(dtype),
                    boogu_t,
                    uncond_feats,
                    uncond_mask,
                    freqs_cis,
                    ref_image_hidden_states=ref_latents,
                )
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = latents + v.to(torch.float32) * (t_next - t)

        images = model.decode_latents(latents, device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
