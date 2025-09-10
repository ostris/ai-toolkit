from typing import Union, List, Optional, Dict, Any, Callable

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def prepare_latent_image_ids(batch_size, height, width, patch_size=2, max_offset=0):
    """
    Generates positional embeddings for a latent image.

    Args:
        batch_size (int): The number of images in the batch.
        height (int): The height of the image.
        width (int): The width of the image.
        patch_size (int, optional): The size of the patches. Defaults to 2.
        max_offset (int, optional): The maximum random offset to apply. Defaults to 0.

    Returns:
        torch.Tensor: A tensor containing the positional embeddings.
    """
    # the random pos embedding helps generalize to larger res without training at large res
    # pos embedding for rope, 2d pos embedding, corner embedding and not center based
    latent_image_ids = torch.zeros(height // patch_size, width // patch_size, 3)

    # Add positional encodings
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // patch_size)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // patch_size)[None, :]
    )

    # Add random offset if specified
    if max_offset > 0:
        offset_y = torch.randint(0, max_offset + 1, (1,)).item()
        offset_x = torch.randint(0, max_offset + 1, (1,)).item()
        latent_image_ids[..., 1] += offset_y
        latent_image_ids[..., 2] += offset_x


    (
        latent_image_id_height,
        latent_image_id_width,
        latent_image_id_channels,
    ) = latent_image_ids.shape

    # Reshape for batch
    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids


class ChromaPipeline(FluxPipeline):
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        image_encoder = None,
        feature_extractor = None,
        is_radiance: bool = False,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.is_radiance = is_radiance
        self.vae_scale_factor = 8 if not is_radiance else 1
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = prepare_latent_image_ids(
                batch_size, 
                height, 
                width, 
                patch_size=2 if not self.is_radiance else 16
            ).to(device=device, dtype=dtype)
            # latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        
        if not self.is_radiance:
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        # latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        latent_image_ids = prepare_latent_image_ids(
            batch_size, 
            height, 
            width, 
            patch_size=2 if not self.is_radiance else 16
        ).to(device=device, dtype=dtype)

        return latents, latent_image_ids
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attn_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attn_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[
            int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if isinstance(device, str):
            device = torch.device(device)

        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=torch.bfloat16)
        if guidance_scale > 1.00001:
            negative_text_ids = torch.zeros(batch_size, negative_prompt_embeds.shape[1], 3).to(device=device, dtype=torch.bfloat16)

        # 4. Prepare latent variables
        num_channels_latents = 64 // 4
        if self.is_radiance:
            num_channels_latents = 3
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # extend img ids to match batch size
        # latent_image_ids = latent_image_ids.unsqueeze(0)
        # latent_image_ids = torch.cat([latent_image_ids] * batch_size, dim=0)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        guidance = torch.full([1], 0, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance

                noise_pred_text = self.transformer(
                    img=latents,
                    img_ids=latent_image_ids,
                    txt=prompt_embeds,
                    txt_ids=text_ids,
                    txt_mask=prompt_attn_mask, # todo add this
                    timesteps=timestep / 1000,
                    guidance=guidance
                )

                if guidance_scale > 1.00001:
                    noise_pred_uncond = self.transformer(
                        img=latents,
                        img_ids=latent_image_ids,
                        txt=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        txt_mask=negative_prompt_attn_mask, # todo add this
                        timesteps=timestep / 1000,
                        guidance=guidance
                    )

                    noise_pred = noise_pred_uncond + self.guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                else:
                    noise_pred = noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            if not self.is_radiance:
                latents = self._unpack_latents(
                    latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + \
                self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(
                image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
