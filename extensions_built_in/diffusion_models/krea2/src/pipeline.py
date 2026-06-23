"""Packing / sampling helpers for Krea 2.

Turns image latents + stacked Qwen3-VL text features into the single sequence the
``SingleStreamDiT`` consumes, and provides a minimal flow-matching sampler used to
render preview images during training.

Time convention: Krea 2 is a plain flow-matching model whose time runs ``t=1``
(pure noise) -> ``t=0`` (clean), the velocity it predicts is ``noise - clean``,
and ``x_t = (1 - t) * clean + t * noise``. This is *identical* to ai-toolkit's
convention, so unlike ideogram4 there is no flipping or negation -- the toolkit
``timestep / 1000`` flows straight through as ``t``.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from einops import rearrange, repeat
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from .mmdit import SingleStreamDiT


# ---------------------------------------------------------------------------
# Text feature padding.
# ---------------------------------------------------------------------------


def pad_text_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of per-sample ``(Lt_i, F)`` features into a batch.

    Each caption is stored 2D at its natural length -- the 12 stacked Qwen3-VL
    hidden-state layers are flattened into the feature axis ``F = n * d`` so the
    ai-toolkit batching machinery treats the list length as the batch size (it
    only special-cases 2D per-sample tensors). The layer axis is restored in
    ``predict_velocity`` right before the MMDiT call. Padding to the batch max is
    deferred to here. Returns ``(features (B, Lt, F), mask (B, Lt))``; the mask is
    1 for real text tokens and 0 for padding.
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, f in enumerate(features_list):
        ln = f.shape[0]
        features[i, :ln] = f.to(device, dtype)
        mask[i, :ln] = 1
    return features, mask


# ---------------------------------------------------------------------------
# Latent <-> token packing and combined position / mask construction.
# ---------------------------------------------------------------------------


def prepare(
    img: torch.Tensor, txtlen: int, patch: int, txtmask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patchify the latent and build the combined text+image position / mask.

    in:  img      (B, C, h, w) image latent
         txtlen   number of text tokens
         patch    transformer patch size
         txtmask  (B, txtlen) long/bool mask, 1 for real text tokens
    out: (img_tokens (B, h/p*w/p, C*p*p), pos (B, txtlen+imglen, 3),
          mask (B, txtlen+imglen))
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask.to(img.device).bool(), imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def predict_velocity(
    model: SingleStreamDiT,
    latents: torch.Tensor,  # (B, C, h, w)
    t: torch.Tensor,  # (B,) flow time in [0, 1] (1 = pure noise)
    context: torch.Tensor,  # (B, Lt, n*d) flattened stacked Qwen3-VL features
    text_mask: torch.Tensor,  # (B, Lt) 1 for real text tokens
) -> torch.Tensor:
    """Run the MMDiT on the packed [text | image] sequence.

    ``latents`` stay in the unpacked ``(B, C, h, w)`` latent layout; image-token
    packing is internal to this function. ``context`` arrives 2D-per-sample
    flattened ``(B, Lt, n*d)`` and is restored to ``(B, Lt, n, d)`` for the MMDiT.
    Returns the velocity ``noise - clean`` reshaped back to ``(B, C, h, w)``. No
    time flip / negation: Krea's convention matches toolkit's.
    """
    patch = model.config.patch
    b, c, h, w = latents.shape

    # Restore the stacked-layer axis flattened in pad_text_features: F -> (n, d).
    n = model.config.txtlayers
    context = context.reshape(
        context.shape[0], context.shape[1], n, context.shape[-1] // n
    )

    img_tokens, pos, mask = prepare(latents, context.shape[1], patch, text_mask)

    out = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)

    # (B, imglen, c*p*p) -> (B, c, h, w)
    velocity = rearrange(
        out,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=h // patch,
        w=w // patch,
    )
    return velocity


# ---------------------------------------------------------------------------
# Resolution-aware flow-matching timestep schedule.
# ---------------------------------------------------------------------------


def timesteps(
    seq_len: int,
    steps: int,
    x1: float,
    x2: float,
    y1: float = 0.5,
    y2: float = 1.15,
    sigma: float = 1.0,
    mu: Optional[float] = None,
) -> List[float]:
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0).

    ``mu`` is interpolated linearly in image-sequence length between (x1, y1) and
    (x2, y2), then used to time-shift a uniform 1->0 grid. Pass an explicit ``mu``
    to pin a constant shift regardless of resolution (the distilled turbo
    checkpoint was trained at a fixed mu=1.15).
    """
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


# ---------------------------------------------------------------------------
# Minimal sampling pipeline (for training previews).
# ---------------------------------------------------------------------------


class Krea2Pipeline:
    """Lightweight flow-matching sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # ``model`` is the Krea2Model so we can reuse its encode/decode and config.
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
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer: SingleStreamDiT = model.transformer
        patch = model.patch_size
        ae_scale = model.vae_scale_factor  # 8

        mkw = model.model_config.model_kwargs
        y1 = float(mkw.get("schedule_y1", 0.5))
        y2 = float(mkw.get("schedule_y2", 1.15))
        minres = int(mkw.get("schedule_min_res", 256))
        maxres = int(mkw.get("schedule_max_res", 1280))
        mu = mkw.get("schedule_mu", None)
        mu = float(mu) if mu is not None else None

        do_cfg = guidance_scale > 0 and unconditional_embeds is not None

        gh = height // (ae_scale * patch)
        gw = width // (ae_scale * patch)
        latent_channels = transformer.config.channels

        # Starting gaussian noise in the (B, C, h8, w8) latent layout.
        if latents is None:
            shape = (1, latent_channels, height // ae_scale, width // ae_scale)
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=torch.float32
            )
        latents = latents.to(device, dtype=torch.float32)

        cond_feats, cond_mask = pad_text_features(
            conditional_embeds.text_embeds, device, dtype
        )
        if do_cfg:
            uncond_feats, uncond_mask = pad_text_features(
                unconditional_embeds.text_embeds, device, dtype
            )

        # min_res / max_res define the (x1,y1)-(x2,y2) interpolation endpoints for mu.
        align = ae_scale * patch
        x1 = (minres // align) ** 2
        x2 = (maxres // align) ** 2
        ts = timesteps(gh * gw, num_inference_steps, x1, x2, y1=y1, y2=y2, mu=mu)

        # Euler integration of the flow ODE (with optional CFG).
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((latents.shape[0],), tcurr, dtype=dtype, device=device)
            v_cond = predict_velocity(
                transformer, latents.to(dtype), t, cond_feats, cond_mask
            )
            if do_cfg:
                v_uncond = predict_velocity(
                    transformer, latents.to(dtype), t, uncond_feats, uncond_mask
                )
                v = v_cond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = latents + (tprev - tcurr) * v.to(torch.float32)

        images = model.decode_latents(latents, device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
