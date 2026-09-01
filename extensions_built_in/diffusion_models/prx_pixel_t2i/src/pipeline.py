"""Minimal preview sampler for the pixel-space PRX (PRXPixel) model.

ai-toolkit only uses this pipeline to render preview/sample images during
training (BaseModel.generate_images -> PRXPixelT2IModel.generate_single_image).
It does not need to be a diffusers DiffusionPipeline, and ai-toolkit always
encodes the prompts itself, so the pipeline only ever receives already-encoded
``PromptEmbeds`` (text features + attention mask), never raw text.

PRXPixel specifics this sampler bakes in (see ../prx_pixel_t2i.py for the why):
  - **pixel space**: there is no VAE. The model's "latents" are the RGB image
    itself in [-1, 1]; the final "decode" is just a clamp + uint8 cast.
  - **x-prediction**: the transformer predicts the clean image x0, not the
    flow-matching velocity. Each step converts it to velocity
    ``v = (x_t - x0) / t`` (t clamped for stability) before the scheduler step,
    exactly like the diffusers PRXPixelPipeline.
  - **noise_scale**: PRXPixel trains with a non-unit initial-noise std, so the
    starting noise is ``randn * noise_scale`` (2.0 for the released model).
  - **CFG** is applied on the x0 prediction (before the velocity conversion).
"""

from typing import List, Optional

import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor


# Minimum normalized timestep used when converting an x0 prediction to a flow
# velocity ``v = (x_t - x0) / t``. Mirrors the 0.05 clamp in the diffusers
# PRXPixelPipeline; without it the division blows up as t -> 0.
X_PRED_T_MIN = 0.05


class PRXPixelPipeline:
    """Lightweight pixel-space flow-matching sampler used for training previews."""

    def __init__(self, model):
        # ``model`` is the PRXPixelT2IModel (a BaseModel subclass), giving us
        # access to model.transformer, model.decode_latents, device/dtype, the
        # scheduler factory and the noise scale.
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        # BaseModel.generate_images may call pipeline.to(device); we manage
        # devices through the model itself, so this is a no-op.
        return self

    def set_progress_bar_config(self, **kwargs):
        # called by the sampler harness (inside a try/except, so optional)
        pass

    def _embeds_and_mask(self, embeds, device, dtype):
        """Pull (features, attention_mask) out of a PromptEmbeds onto device/dtype.

        ``text_embeds`` is (B, L, D). The mask is (B, L) bool (1 = real token);
        it is kept as long for the transformer's boolean masking and never cast
        to the model dtype.
        """
        feats = embeds.text_embeds.to(device, dtype=dtype)
        mask = getattr(embeds, "attention_mask", None)
        if mask is not None:
            mask = mask.to(device)
        return feats, mask

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,  # PromptEmbeds: .text_embeds (B,L,D) + .attention_mask
        unconditional_embeds,  # PromptEmbeds or None (negative prompt)
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 5.0,
        latents: Optional[torch.Tensor] = None,  # pre-made noise, usually None
        generator: Optional[
            torch.Generator
        ] = None,  # seeded RNG for reproducible samples
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer

        # Always sample with a FRESH scheduler -- the training scheduler is
        # stateful and mutating it mid-training would corrupt the train step.
        scheduler = model.get_train_scheduler()
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # 1000 -> 0 scale

        do_cfg = unconditional_embeds is not None and guidance_scale != 1.0

        # 1. starting noise -- pixel space, so channels = transformer.in_channels (3),
        #    spatial size = the requested pixels (no VAE downsample). Scaled by the
        #    model's noise_scale to match the learned flow-matching trajectory.
        if latents is None:
            shape = (1, transformer.in_channels, height, width)
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=torch.float32
            )
            latents = latents * model.noise_scale
        latents = latents.to(device, dtype=torch.float32)

        # 2. text features + masks
        cond_feats, cond_mask = self._embeds_and_mask(conditional_embeds, device, dtype)
        if do_cfg:
            uncond_feats, uncond_mask = self._embeds_and_mask(
                unconditional_embeds, device, dtype
            )

        # 3. denoising loop
        for t in timesteps:
            # scheduler timesteps are 0-1000; the transformer wants [0, 1]
            t01 = (t / 1000.0).to(device).float().expand(latents.shape[0])

            x0_cond = transformer(
                hidden_states=latents.to(dtype),
                timestep=t01,
                encoder_hidden_states=cond_feats,
                attention_mask=cond_mask,
                return_dict=False,
            )[0]
            if do_cfg:
                x0_uncond = transformer(
                    hidden_states=latents.to(dtype),
                    timestep=t01,
                    encoder_hidden_states=uncond_feats,
                    attention_mask=uncond_mask,
                    return_dict=False,
                )[0]
                # classifier-free guidance is applied on the x0 prediction
                x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                x0 = x0_cond

            # convert the x0 (clean-image) prediction to the flow velocity the
            # scheduler consumes: v = (x_t - x0) / t, t clamped for stability.
            t_x = torch.clamp(t01.to(torch.float32), min=X_PRED_T_MIN).view(-1, 1, 1, 1)
            v = (latents - x0.to(torch.float32)) / t_x

            latents = scheduler.step(v, t, latents, return_dict=False)[0]

        # 4. pixel space: the denoised latents ARE the image in [-1, 1].
        #    decode_latents is an identity (FakeVAE) but we route through it so
        #    any future latent normalization stays in one place.
        images = model.decode_latents(latents, device=device, dtype=torch.float32)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
