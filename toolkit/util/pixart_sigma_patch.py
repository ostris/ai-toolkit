import torch
from diffusers import ImagePipelineOutput, PixArtAlphaPipeline, AutoencoderKL, Transformer2DModel, \
    DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import PixArtAlphaTextProjection, PatchEmbed
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps
from typing import Callable, List, Optional, Tuple, Union

from diffusers.utils import deprecate
from torch import nn
from transformers import T5Tokenizer, T5EncoderModel

ASPECT_RATIO_2048_BIN = {
    "0.25": [1024.0, 4096.0],
    "0.26": [1024.0, 3968.0],
    "0.27": [1024.0, 3840.0],
    "0.28": [1024.0, 3712.0],
    "0.32": [1152.0, 3584.0],
    "0.33": [1152.0, 3456.0],
    "0.35": [1152.0, 3328.0],
    "0.4": [1280.0, 3200.0],
    "0.42": [1280.0, 3072.0],
    "0.48": [1408.0, 2944.0],
    "0.5": [1408.0, 2816.0],
    "0.52": [1408.0, 2688.0],
    "0.57": [1536.0, 2688.0],
    "0.6": [1536.0, 2560.0],
    "0.68": [1664.0, 2432.0],
    "0.72": [1664.0, 2304.0],
    "0.78": [1792.0, 2304.0],
    "0.82": [1792.0, 2176.0],
    "0.88": [1920.0, 2176.0],
    "0.94": [1920.0, 2048.0],
    "1.0": [2048.0, 2048.0],
    "1.07": [2048.0, 1920.0],
    "1.13": [2176.0, 1920.0],
    "1.21": [2176.0, 1792.0],
    "1.29": [2304.0, 1792.0],
    "1.38": [2304.0, 1664.0],
    "1.46": [2432.0, 1664.0],
    "1.67": [2560.0, 1536.0],
    "1.75": [2688.0, 1536.0],
    "2.0": [2816.0, 1408.0],
    "2.09": [2944.0, 1408.0],
    "2.4": [3072.0, 1280.0],
    "2.5": [3200.0, 1280.0],
    "2.89": [3328.0, 1152.0],
    "3.0": [3456.0, 1152.0],
    "3.11": [3584.0, 1152.0],
    "3.62": [3712.0, 1024.0],
    "3.75": [3840.0, 1024.0],
    "3.88": [3968.0, 1024.0],
    "4.0": [4096.0, 1024.0]
}

ASPECT_RATIO_256_BIN = {
    "0.25": [128.0, 512.0],
    "0.28": [128.0, 464.0],
    "0.32": [144.0, 448.0],
    "0.33": [144.0, 432.0],
    "0.35": [144.0, 416.0],
    "0.4": [160.0, 400.0],
    "0.42": [160.0, 384.0],
    "0.48": [176.0, 368.0],
    "0.5": [176.0, 352.0],
    "0.52": [176.0, 336.0],
    "0.57": [192.0, 336.0],
    "0.6": [192.0, 320.0],
    "0.68": [208.0, 304.0],
    "0.72": [208.0, 288.0],
    "0.78": [224.0, 288.0],
    "0.82": [224.0, 272.0],
    "0.88": [240.0, 272.0],
    "0.94": [240.0, 256.0],
    "1.0": [256.0, 256.0],
    "1.07": [256.0, 240.0],
    "1.13": [272.0, 240.0],
    "1.21": [272.0, 224.0],
    "1.29": [288.0, 224.0],
    "1.38": [288.0, 208.0],
    "1.46": [304.0, 208.0],
    "1.67": [320.0, 192.0],
    "1.75": [336.0, 192.0],
    "2.0": [352.0, 176.0],
    "2.09": [368.0, 176.0],
    "2.4": [384.0, 160.0],
    "2.5": [400.0, 160.0],
    "3.0": [432.0, 144.0],
    "4.0": [512.0, 128.0]
}

ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}


def pipeline_pixart_alpha_call(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        **kwargs,
) -> Union[ImagePipelineOutput, Tuple]:
    """
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_inference_steps (`int`, *optional*, defaults to 100):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
            timesteps are used. Must be in descending order.
        guidance_scale (`float`, *optional*, defaults to 4.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        height (`int`, *optional*, defaults to self.unet.config.sample_size):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size):
            The width in pixels of the generated image.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
            provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
        negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
            Pre-generated attention mask for negative text embeddings.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        clean_caption (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
            be installed. If the dependencies are not installed, the embeddings will be created from the raw
            prompt.
        use_resolution_binning (`bool` defaults to `True`):
            If set to `True`, the requested height and width are first mapped to the closest resolutions using
            `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
            the requested resolution. Useful for generating non-square images.

    Examples:

    Returns:
        [`~pipelines.ImagePipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
            returned where the first element is a list with the generated images
    """
    if "mask_feature" in kwargs:
        deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
        deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
    # 1. Check inputs. Raise error if not correct
    height = height or self.transformer.config.sample_size * self.vae_scale_factor
    width = width or self.transformer.config.sample_size * self.vae_scale_factor
    if use_resolution_binning:
        if self.transformer.config.sample_size == 32:
            aspect_ratio_bin = ASPECT_RATIO_256_BIN
        elif self.transformer.config.sample_size == 64:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif self.transformer.config.sample_size == 128:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        elif self.transformer.config.sample_size == 256:
            aspect_ratio_bin = ASPECT_RATIO_2048_BIN
        else:
            raise ValueError("Invalid sample size")
        orig_height, orig_width = height, width
        height, width = self.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_steps,
        prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
    )

    # 2. Default height and width to transformer
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = self.encode_prompt(
        prompt,
        do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        clean_caption=clean_caption,
        max_sequence_length=max_sequence_length,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latents.
    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        latent_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Prepare micro-conditions.
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    if self.transformer.config.sample_size == 128:
        resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
        aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
        resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

        if do_classifier_free_guidance:
            resolution = torch.cat([resolution, resolution], dim=0)
            aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

    # 7. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(current_timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # compute previous image: x_t -> x_t-1
            if num_inference_steps == 1:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        if use_resolution_binning:
            image = self.resize_and_crop_tensor(image, orig_width, orig_height)
    else:
        image = latents

    if not output_type == "latent":
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return ImagePipelineOutput(images=image)


class PixArtSigmaPipeline(PixArtAlphaPipeline):
    r"""
    tmp Pipeline for text-to-image generation using PixArt-Sigma.
    """

    def __init__(
            self,
            tokenizer: T5Tokenizer,
            text_encoder: T5EncoderModel,
            vae: AutoencoderKL,
            transformer: Transformer2DModel,
            scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)


def pixart_sigma_init_patched_inputs(self, norm_type):
    assert self.config.sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

    self.height = self.config.sample_size
    self.width = self.config.sample_size

    self.patch_size = self.config.patch_size
    interpolation_scale = (
        self.config.interpolation_scale
        if self.config.interpolation_scale is not None
        else max(self.config.sample_size // 64, 1)
    )
    self.pos_embed = PatchEmbed(
        height=self.config.sample_size,
        width=self.config.sample_size,
        patch_size=self.config.patch_size,
        in_channels=self.in_channels,
        embed_dim=self.inner_dim,
        interpolation_scale=interpolation_scale,
    )

    self.transformer_blocks = nn.ModuleList(
        [
            BasicTransformerBlock(
                self.inner_dim,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
                dropout=self.config.dropout,
                cross_attention_dim=self.config.cross_attention_dim,
                activation_fn=self.config.activation_fn,
                num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                attention_bias=self.config.attention_bias,
                only_cross_attention=self.config.only_cross_attention,
                double_self_attention=self.config.double_self_attention,
                upcast_attention=self.config.upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps,
                attention_type=self.config.attention_type,
            )
            for _ in range(self.config.num_layers)
        ]
    )

    if self.config.norm_type != "ada_norm_single":
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels
        )
    elif self.config.norm_type == "ada_norm_single":
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim ** 0.5)
        self.proj_out = nn.Linear(
            self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels
        )

    # PixArt-Sigma blocks.
    self.adaln_single = None
    self.use_additional_conditions = False
    if self.config.norm_type == "ada_norm_single":
        # TODO(Sayak, PVP) clean this, PixArt-Sigma doesn't use additional_conditions anymore
        # additional conditions until we find better name
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )

    self.caption_projection = None
    if self.caption_channels is not None:
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=self.caption_channels, hidden_size=self.inner_dim
        )
