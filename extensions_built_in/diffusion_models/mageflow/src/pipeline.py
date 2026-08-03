"""Packing / sampling helpers for Mage-Flow.

Turns latents + Qwen3-VL text features into the packed variable-length
sequences the ``MageFlow`` DiT consumes (batch dim 1, per-sample
``cu_seqlens``, exactly mirroring the reference training/inference packing),
and provides a minimal flow-matching sampler for ai-toolkit preview images.

Time convention: rectified flow with ``x_t = (1 - sigma) * clean + sigma *
noise``, the model predicts the velocity ``noise - clean``, and sampling
integrates sigma from 1 (pure noise) down to 0 (clean). This is identical to
ai-toolkit's convention, so the toolkit ``timestep / 1000`` flows straight
through as sigma. The inference sigma schedule is the reference's static
shift: ``shifted = shift * s / (1 + (shift - 1) * s)`` over ``s =
linspace(1, 1/steps, steps)`` with a terminal 0 (shift 6.0).

Edit conditioning: each sample's sequence is ``[target, ref_1, …, ref_N]`` —
clean reference latents ride along after the noisy target tokens, sharing the
sample's timestep modulation. RoPE gives every image segment in the pack its
own "frame" index (segment order), so references land on later frame
coordinates than their target, as in the reference ``generate_edits``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from einops import rearrange
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from .transformer import MageFlow


def lens_to_cu(lens: List[int], device) -> torch.Tensor:
    """Sequence lengths -> cumulative cu_seqlens [0, l0, l0+l1, ...] (int32)."""
    t = torch.tensor(lens, device=device, dtype=torch.int32)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(t, dim=0, dtype=torch.int32),
        ]
    )


def pack_text_features(
    features_list: List[torch.Tensor], device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-sample ``(L_i, D)`` text features into the packed
    ``[1, sum L, D]`` text stream + its ``cu_seqlens``. No padding — varlen
    attention isolates the samples."""
    feats = [f.to(device, dtype) for f in features_list]
    txt = torch.cat(feats, dim=0).unsqueeze(0)
    txt_cu = lens_to_cu([f.shape[0] for f in feats], device)
    return txt, txt_cu


def predict_velocity(
    model: MageFlow,
    latents: torch.Tensor,  # (B, C, h, w) noisy target latents
    t: torch.Tensor,  # (B,) flow sigma in [0, 1] (1 = pure noise)
    text_embeds: List[torch.Tensor],  # per-sample (L_i, D) Qwen3-VL features
    ref_latents: Optional[
        List[List[torch.Tensor]]
    ] = None,  # per-sample clean (C, hr, wr) refs
) -> torch.Tensor:
    """Run the DiT on the packed [text | target(+refs)] sequences.

    Packs the batch into one varlen forward (mirroring reference packing:
    per-sample cu_seqlens isolate the samples inside the attention kernel).
    Reference latents, when given, are appended clean after each sample's
    target tokens; the returned velocity covers only the target tokens,
    reshaped back to ``(B, C, h, w)``. No time flip / negation: Mage-Flow's
    convention matches ai-toolkit's.
    """
    device = latents.device
    dtype = latents.dtype
    b, c, h, w = latents.shape

    if ref_latents is not None and not any(len(r) > 0 for r in ref_latents):
        ref_latents = None

    img_parts, samp_lens, shape_seq, target_idx_parts = [], [], [], []
    off = 0
    for i in range(b):
        tgt = rearrange(latents[i], "c h w -> (h w) c")
        img_parts.append(tgt)
        shape_seq.append((1, h, w))
        target_idx_parts.append(torch.arange(off, off + h * w, device=device))
        samp_len = h * w
        off += h * w
        if ref_latents is not None:
            for ref in ref_latents[i]:
                ref = ref.to(device, dtype)
                _, rh, rw = ref.shape
                img_parts.append(rearrange(ref, "c h w -> (h w) c"))
                shape_seq.append((1, rh, rw))
                samp_len += rh * rw
                off += rh * rw
        samp_lens.append(samp_len)

    img = torch.cat(img_parts, dim=0).unsqueeze(0)  # [1, sum, C]
    img_cu = lens_to_cu(samp_lens, device)
    img_shapes = [shape_seq]
    target_idx = torch.cat(target_idx_parts)

    txt, txt_cu = pack_text_features(text_embeds, device, dtype)

    out = model(
        img=img,
        txt=txt,
        timesteps=t.to(device),
        img_shapes=img_shapes,
        img_cu_seqlens=img_cu,
        txt_cu_seqlens=txt_cu,
    )  # [1, sum, C_out]

    pred = out[:, target_idx, :]  # [1, B*h*w, C]
    return rearrange(pred, "1 (b h w) c -> b c h w", b=b, h=h, w=w)


def build_shifted_sigmas(
    num_steps: int, shift: float = 6.0, device=None
) -> torch.Tensor:
    """The reference inference sigma schedule: ``linspace(1, 1/steps, steps)``
    run through the static shift ``shift*s/(1+(shift-1)*s)`` with a terminal 0."""
    s = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float64)
    s = shift * s / (1 + (shift - 1) * s)
    sigmas = torch.cat([s, torch.zeros(1, dtype=torch.float64)])
    return sigmas.to(device=device, dtype=torch.float32)


class MageFlowPipeline:
    """Lightweight flow-matching sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # ``model`` is the MageFlowModel (BaseModel) so we can reuse its
        # encode/decode and config.
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
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        ref_latents: Optional[List[List[torch.Tensor]]] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer: MageFlow = model.transformer

        shift = float(model.model_config.model_kwargs.get("static_shift", 6.0))
        renorm = bool(model.model_config.model_kwargs.get("cfg_renormalization", False))

        do_cfg = guidance_scale > 1.0 and unconditional_embeds is not None

        gh = height // model.vae_scale_factor
        gw = width // model.vae_scale_factor
        latent_channels = transformer.in_channels

        if latents is None:
            latents = randn_tensor(
                (1, latent_channels, gh, gw),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        latents = latents.to(device, dtype=torch.float32)

        cond_feats = [f.to(device, dtype) for f in conditional_embeds.text_embeds]
        if do_cfg:
            uncond_feats = [
                f.to(device, dtype) for f in unconditional_embeds.text_embeds
            ]

        sigmas = build_shifted_sigmas(num_inference_steps, shift=shift, device=device)

        # Euler integration of the flow ODE, sigma 1 -> 0 (with optional CFG:
        # v = uncond + cfg * (cond - uncond), reference convention).
        for i in range(num_inference_steps):
            s_cur = sigmas[i].item()
            s_next = sigmas[i + 1].item()
            t = torch.full((latents.shape[0],), s_cur, dtype=dtype, device=device)
            v_cond = predict_velocity(
                transformer, latents.to(dtype), t, cond_feats, ref_latents=ref_latents
            )
            if do_cfg:
                v_uncond = predict_velocity(
                    transformer,
                    latents.to(dtype),
                    t,
                    uncond_feats,
                    ref_latents=ref_latents,
                )
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
                if renorm:
                    # Rescale the guided velocity per token back to the
                    # conditional velocity's norm (reduces oversaturation).
                    v = v * (
                        torch.norm(v_cond, dim=1, keepdim=True)
                        / (torch.norm(v, dim=1, keepdim=True) + 1e-6)
                    )
            else:
                v = v_cond
            latents = latents + (s_next - s_cur) * v.to(torch.float32)

        images = model.decode_latents(latents, device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
