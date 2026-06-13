"""A minimal sampling pipeline for the example model.

ai-toolkit only uses your pipeline to render preview/sample images during
training (see BaseModel.generate_images -> ExampleModel.generate_single_image).
It does NOT need to be a diffusers DiffusionPipeline, and because ai-toolkit
always encodes the prompts itself (so it can cache embeds, apply trigger words,
run adapters, etc.) the pipeline never sees raw prompt strings -- only
already-encoded ``AdvancedPromptEmbeds``.

So all a pipeline has to do is:

  1. make starting noise
  2. loop the scheduler over timesteps, calling the transformer
  3. apply classifier-free guidance (cond vs uncond prediction)
  4. decode the final latents with the VAE and return PIL images

The pattern of passing the whole BaseModel instance into the pipeline (rather
than individual components) is borrowed from ../../ideogram4/src/pipeline.py.
It keeps the pipeline tiny because it can reuse the model's scheduler factory,
``decode_latents`` and device/dtype bookkeeping.
"""

from typing import List, Optional

import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor


def pad_prompt_embeds(
    embeds_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
    """Right-pad a list of per-sample text features into one batch tensor.

    in:  embeds_list  list (len B) of (L_i, D) tensors -- this is exactly what
                      ``AdvancedPromptEmbeds.text_embeds`` holds: one tensor per
                      batch item, each at its own natural length.
    out: features     (B, L_max, D) zero-padded on the right
         mask         (B, L_max) long, 1 = real token, 0 = padding

    Storing embeds unpadded per item and only padding at the model call is the
    preferred pattern: cached embeds stay small, and items of very different
    prompt lengths can share a batch.
    """
    lengths = [e.shape[0] for e in embeds_list]
    max_len = max(lengths)
    dim = embeds_list[0].shape[-1]
    batch_size = len(embeds_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, e in enumerate(embeds_list):
        n = e.shape[0]
        features[i, :n] = e.to(device, dtype)
        mask[i, :n] = 1
    return features, mask


class ExamplePipeline:
    """Lightweight flow-matching sampler used for training previews."""

    def __init__(self, model):
        # ``model`` is the ExampleModel (a BaseModel subclass), giving us
        # access to model.transformer, model.vae, model.decode_latents, etc.
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        # BaseModel.generate_images may call pipeline.to(device); we manage
        # devices through the model itself, so this is a no-op.
        return self

    def set_progress_bar_config(self, **kwargs):
        # called by the sampler harness (inside a try/except, so optional);
        # diffusers pipelines use it to silence tqdm. Nothing to do here.
        pass

    @torch.no_grad()
    def __call__(
        self,
        # AdvancedPromptEmbeds with key ``text_embeds`` (list of (L, D) tensors)
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        latents: Optional[torch.Tensor] = None,   # pre-made noise, usually None
        generator: Optional[torch.Generator] = None,  # seeded RNG for reproducible samples
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer

        # Always sample with a FRESH scheduler. The training scheduler is
        # stateful; mutating it mid-training would corrupt the train step.
        scheduler = model.get_train_scheduler()
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # 1000 -> 0 scale

        # pixel size -> latent size (VAE downsample only; the transformer
        # patchifies internally so latents stay unpacked here)
        gh = height // model.vae_scale_factor
        gw = width // model.vae_scale_factor

        do_cfg = unconditional_embeds is not None and guidance_scale != 1.0

        # 1. starting noise (keep it float32; cast per model call)
        if latents is None:
            shape = (1, transformer.in_channels, gh, gw)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents.to(device, dtype=torch.float32)

        # 2. pad the per-item embed lists into batch tensors once, up front
        cond_feats, cond_mask = pad_prompt_embeds(conditional_embeds.text_embeds, device, dtype)
        if do_cfg:
            uncond_feats, uncond_mask = pad_prompt_embeds(unconditional_embeds.text_embeds, device, dtype)

        # 3. denoising loop
        for t in timesteps:
            # scheduler timesteps are on a 0-1000 scale; the transformer wants
            # flow time in [0, 1] with 1 = pure noise
            t01 = (t / 1000.0).to(device).expand(latents.shape[0])

            v_cond = transformer(
                hidden_states=latents.to(dtype),
                timestep=t01,
                encoder_hidden_states=cond_feats,
                attention_mask=cond_mask,
            )
            if do_cfg:
                v_uncond = transformer(
                    hidden_states=latents.to(dtype),
                    timestep=t01,
                    encoder_hidden_states=uncond_feats,
                    attention_mask=uncond_mask,
                )
                # classifier-free guidance: push the prediction away from the
                # unconditional (negative prompt) direction
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            latents = scheduler.step(v.to(torch.float32), t, latents, return_dict=False)[0]

        # 4. decode latents -> images in [-1, 1] -> uint8 PIL
        images = model.decode_latents(latents, device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
