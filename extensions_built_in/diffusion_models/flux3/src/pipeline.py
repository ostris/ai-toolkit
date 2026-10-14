"""FLUX.3 sampling pipeline + shared latent/conditioning packers.

Covers t2i and i2i (reference images). All conditioning arrives as prompt
*embeds* (never raw text) plus optional control-image tensors.

Flow convention: time runs t=1 (pure noise) -> t=0 (clean); the DiT
predicts the velocity ``noise - clean`` -- identical to ai-toolkit's
convention. The inference schedule is a timeshifted uniform grid
(``timeshift(alpha, t)``); the package default alpha is 6.93.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from .positional import batched_prc_img, prc_img, prc_txt

IMAGE_ROPE_SCALE = 0.5
I2I_COND_L = -1
DEFAULT_ALPHA = 6.93
DEFAULT_GUIDANCE = 3.5


def timeshift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def build_timesteps(n_steps: int, alpha: float) -> torch.Tensor:
    return timeshift(alpha, torch.linspace(1.0, 0.0, n_steps + 1))


# ---------------------------------------------------------------------------
# Text conditioning
# ---------------------------------------------------------------------------


def build_text_conditioning(
    text_embeds_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
    """Per-sample (L, 20480) ctx tensors -> batched DiT text kwargs.

    Embeds are padded to a multiple of 80 tokens at encode time, so items
    encoded separately (TE cache, cond/uncond pairs) can differ in length;
    shorter ones are zero-padded to the batch's longest. Returns
    ``(ctx (B, L, D), ctx_ids (B, L, 4), timesteps_ctx (B, L))``.
    """
    max_len = max(e.shape[0] for e in text_embeds_list)
    ctx = torch.stack(
        [
            F.pad(e.to(device, dtype), (0, 0, 0, max_len - e.shape[0]))
            for e in text_embeds_list
        ]
    )
    _, ctx_ids = prc_txt(ctx[0])
    ctx_ids = ctx_ids.unsqueeze(0).expand(ctx.shape[0], -1, -1).to(device)
    timesteps_ctx = torch.zeros(ctx.shape[:2], device=device, dtype=dtype)
    return ctx, ctx_ids, timesteps_ctx


# ---------------------------------------------------------------------------
# Token packing (training + sampling share these)
# ---------------------------------------------------------------------------


def pack_image_latents(latents: torch.Tensor):
    """(B, 128, h, w) image latents -> ((B, h*w, 128) tokens, float ids with
    the image RoPE scale applied to the h/w axes)."""
    tokens, ids = batched_prc_img(latents)
    ids = ids.to(dtype=torch.float32)
    ids[..., 1:3] = ids[..., 1:3] * IMAGE_ROPE_SCALE
    return tokens, ids


def pack_i2i_image_condition(
    latents_per_sample: List[List[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
):
    """Reference-image latents for i2i editing.

    ``latents_per_sample`` is one list of (128, h, w) latents per batch item
    (every item must have the same reference geometry). Reference i placed at
    ``t = 0, l = -1 - i``; conditioning tokens run at timestep 0.
    Returns ``(tokens, ids, timesteps)``.
    """
    all_tokens, all_ids = [], []
    for refs in latents_per_sample:
        parts, id_parts = [], []
        for i_ref, latent in enumerate(refs):
            x_ref, x_ref_ids = prc_img(
                latent,
                t_coord=torch.tensor([0], device=latent.device, dtype=torch.long),
                l_coord=torch.tensor(
                    [I2I_COND_L - i_ref], device=latent.device, dtype=torch.long
                ),
            )
            parts.append(x_ref)
            id_parts.append(x_ref_ids)
        all_tokens.append(torch.cat(parts, dim=0))
        all_ids.append(torch.cat(id_parts, dim=0))
    tokens = torch.stack(all_tokens).to(device, dtype)
    ids = torch.stack(all_ids).to(device=device, dtype=torch.float32)
    ids[..., 1:3] = ids[..., 1:3] * IMAGE_ROPE_SCALE
    timesteps = torch.zeros(tokens.shape[:2], device=device, dtype=dtype)
    return tokens, ids, timesteps


# ---------------------------------------------------------------------------
# Sampling pipeline
# ---------------------------------------------------------------------------


class Flux3Pipeline:
    """Lightweight sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # ``model`` is the Flux3Model so we can reuse its encode/decode + config.
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
        num_inference_steps: int = 50,
        guidance_scale: float = DEFAULT_GUIDANCE,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        ctrl_imgs: Optional[List[torch.Tensor]] = None,  # each (1, 3, H, W) in [0, 1]
        alpha: Optional[float] = None,
        **kwargs,
    ):
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer

        do_cfg = unconditional_embeds is not None and guidance_scale > 1.0

        if alpha is None:
            alpha = DEFAULT_ALPHA
        timesteps = build_timesteps(num_inference_steps, alpha=alpha).to(device)

        # text conditioning, CFG-doubled as [uncond, cond]
        cond_list = conditional_embeds.text_embeds
        if do_cfg:
            embeds_list = unconditional_embeds.text_embeds + cond_list
        else:
            embeds_list = cond_list
        ctx, ctx_ids, ts_ctx = build_text_conditioning(embeds_list, device, dtype)
        cfg_batch = ctx.shape[0]

        # initial noise tokens
        h_lat, w_lat = height // 16, width // 16
        if latents is None:
            latents = randn_tensor(
                (1, 128, h_lat, w_lat), generator=generator, dtype=torch.float32
            )
        latents = latents.to(device)
        x, ids = pack_image_latents(latents)
        x = x.float()
        ids = ids.repeat(cfg_batch, 1, 1).to(device)

        # reference-image conditioning: reference i rides at l = -1 - i
        cond_kwargs = {}
        if ctrl_imgs:
            ref_latents = []
            for ctrl_img in ctrl_imgs:
                ctrl = ctrl_img.to(device, dtype) * 2 - 1
                ref_latents.append(
                    model.encode_images(list(ctrl), device=device, dtype=dtype)[0]
                )
            xc, xc_ids, xc_ts = pack_i2i_image_condition([ref_latents], device, dtype)
            cond_kwargs = {
                "x_image_cond": xc.repeat(cfg_batch, 1, 1),
                "x_image_cond_ids": xc_ids.repeat(cfg_batch, 1, 1),
                "x_image_cond_timesteps": xc_ts.repeat(cfg_batch, 1),
            }

        # Euler integration of the flow ODE with CFG
        for step_index, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            x_in = x.to(dtype).repeat(cfg_batch, 1, 1)
            pred = transformer(
                ctx=ctx,
                ctx_ids=ctx_ids,
                timesteps_ctx=ts_ctx,
                x_image=x_in,
                x_image_ids=ids,
                x_image_timesteps=t_curr
                * torch.ones(x_in.shape[:2], device=device, dtype=dtype),
                **cond_kwargs,
            )
            v = pred["x_image"].float()
            if do_cfg:
                v_uc, v_c = v.chunk(2)
                v = v_uc + guidance_scale * (v_c - v_uc)
            if getattr(model, "sample_step_hook", None) is not None:
                # inference engine preview: x0 estimate
                x0 = (x - t_curr * v).reshape(1, h_lat, w_lat, -1).permute(0, 3, 1, 2)
                model._emit_sample_step(x0, step_index, len(timesteps) - 1)
            x = x + (t_prev - t_curr) * v

        img_latent = x.reshape(1, h_lat, w_lat, -1).permute(0, 3, 1, 2)
        images = model.decode_latents(img_latent, device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
