# originally from https://github.com/fal-ai/f-lite/blob/main/f_lite/pipeline.py  but modified slightly
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch import FloatTensor
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5TokenizerFast



logger = logging.getLogger(__name__)


@dataclass
class APGConfig:
    """APG (Augmented Parallel Guidance) configuration"""

    enabled: bool = True
    orthogonal_threshold: float = 0.03


@dataclass
class FLitePipelineOutput(BaseOutput):
    """
    Output class for FLitePipeline pipeline.
    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[Image.Image], np.ndarray]


class FLitePipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using F-Lite model.
    This model inherits from [`DiffusionPipeline`].
    """

    model_cpu_offload_seq = "text_encoder->dit_model->vae" 

    dit_model: torch.nn.Module
    vae: AutoencoderKL
    text_encoder: T5EncoderModel
    tokenizer: T5TokenizerFast
    _progress_bar_config: Dict[str, Any]

    def __init__(
        self, dit_model: torch.nn.Module, vae: AutoencoderKL, text_encoder: T5EncoderModel, tokenizer: T5TokenizerFast
    ):
        super().__init__()
        # Register all modules for the pipeline
        # Access DiffusionPipeline's register_modules directly to avoid mypy error
        DiffusionPipeline.register_modules(
            self, dit_model=dit_model, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer
        )

        # Move models to channels last for better performance
        # AutoencoderKL inherits from torch.nn.Module which has these methods
        if hasattr(self.vae, "to"):
            self.vae.to(memory_format=torch.channels_last)
        if hasattr(self.vae, "requires_grad_"):
            self.vae.requires_grad_(False)
        if hasattr(self.text_encoder, "requires_grad_"):
            self.text_encoder.requires_grad_(False)

        # Constants
        self.vae_scale_factor = 8
        self.return_index = -8  # T5 hidden state index to use

    def enable_vae_slicing(self):
        """Enable VAE slicing for memory efficiency."""
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()

    def enable_vae_tiling(self):
        """Enable VAE tiling for memory efficiency."""
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()

    def set_progress_bar_config(self, **kwargs):
        """Set progress bar configuration."""
        self._progress_bar_config = kwargs

    def progress_bar(self, iterable=None, **kwargs):
        """Create progress bar for iterations."""
        self._progress_bar_config = getattr(self, "_progress_bar_config", None) or {}
        config = {**self._progress_bar_config, **kwargs}
        return tqdm(iterable, **config)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 512,
        return_index: int = -8,
    ) -> Tuple[FloatTensor, FloatTensor]:
        """Encodes the prompt and negative prompt."""
        if isinstance(prompt, str):
            prompt = [prompt]
        device = device or self.text_encoder.device
        # Text encoder forward pass
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = self.text_encoder(text_input_ids, return_dict=True, output_hidden_states=True)
        prompt_embeds_tensor = prompt_embeds.hidden_states[return_index]
        if return_index != -1:
            prompt_embeds_tensor = self.text_encoder.encoder.final_layer_norm(prompt_embeds_tensor)
            prompt_embeds_tensor = self.text_encoder.encoder.dropout(prompt_embeds_tensor)

        dtype = dtype or next(self.text_encoder.parameters()).dtype
        prompt_embeds_tensor = prompt_embeds_tensor.to(dtype=dtype, device=device)

        # Handle negative prompts
        if negative_prompt is None:
            negative_embeds = torch.zeros_like(prompt_embeds_tensor)
        else:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            negative_result = self.encode_prompt(
                prompt=negative_prompt, device=device, dtype=dtype, return_index=return_index
            )
            negative_embeds = negative_result[0]

        # Explicitly cast both tensors to FloatTensor for mypy
        from typing import cast

        prompt_tensor = cast(FloatTensor, prompt_embeds_tensor.to(dtype=dtype))
        negative_tensor = cast(FloatTensor, negative_embeds.to(dtype=dtype))
        return (prompt_tensor, negative_tensor)

    def to(self, torch_device=None, torch_dtype=None, silence_dtype_warnings=False):
        """Move pipeline components to specified device and dtype."""
        if hasattr(self, "vae"):
            self.vae.to(device=torch_device, dtype=torch_dtype)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(device=torch_device, dtype=torch_dtype)
        if hasattr(self, "dit_model"):
            self.dit_model.to(device=torch_device, dtype=torch_dtype)
        return self

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]]=None,
        prompt_embeds: Optional[FloatTensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[FloatTensor] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        alpha: Optional[float] = None,
        apg_config: Optional[APGConfig] = None,
        **kwargs,
    ):
        """Generate images from text prompt."""
        # Ensure height and width are not None for calculation
        if height is None:
            height = 1024
        if width is None:
            width = 1024

        dtype = dtype or next(self.dit_model.parameters()).dtype
        apg_config = apg_config or APGConfig(enabled=False)

        device = self._execution_device

        # 2. Encode prompts
        prompt_batch_size = len(prompt) if isinstance(prompt, list) else 1
        batch_size = prompt_batch_size * num_images_per_prompt

        if prompt_embeds is None or negative_prompt_embeds is None:
            prompt_embeds, negative_embeds = self.encode_prompt(
                prompt=prompt, negative_prompt=negative_prompt, device=self.text_encoder.device, dtype=dtype,
                return_index=self.return_index,
            )
        else:
            negative_embeds = negative_prompt_embeds

        # Repeat embeddings for num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        negative_embeds = negative_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        # 3. Initialize latents
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(f"Got {len(generator)} generators for {batch_size} samples")

        latents = randn_tensor((batch_size, 16, latent_height, latent_width), generator=generator, device=device, dtype=dtype)
        acc_latents = latents.clone()

        # 4. Calculate alpha if not provided
        if alpha is None:
            image_token_size = latent_height * latent_width
            alpha = 2 * math.sqrt(image_token_size / (64 * 64))

        # 6. Sampling loop
        self.dit_model.eval()
        
        # Check if guidance is needed
        do_classifier_free_guidance = guidance_scale >= 1.0
        
        for i in self.progress_bar(range(num_inference_steps, 0, -1)):
            # Calculate timesteps
            t = i / num_inference_steps
            t_next = (i - 1) / num_inference_steps
            # Scale timesteps according to alpha
            t = t * alpha / (1 + (alpha - 1) * t)
            t_next = t_next * alpha / (1 + (alpha - 1) * t_next)
            dt = t - t_next
            
            # Create tensor with proper device
            t_tensor = torch.tensor([t] * batch_size, device=device, dtype=dtype)
            
            if do_classifier_free_guidance:
                # Duplicate latents for both conditional and unconditional inputs
                latents_input = torch.cat([latents] * 2)
                # Concatenate negative and positive prompt embeddings
                context_input = torch.cat([negative_embeds, prompt_embeds])
                # Duplicate timesteps for the batch
                t_input = torch.cat([t_tensor] * 2)
                
                # Get model predictions in a single pass
                model_outputs = self.dit_model(latents_input, context_input, t_input)
                
                # Split outputs back into unconditional and conditional predictions
                uncond_output, cond_output = model_outputs.chunk(2)
                
                if apg_config.enabled:
                    # Augmented Parallel Guidance
                    dy = cond_output
                    dd = cond_output - uncond_output
                    # Find parallel direction
                    parallel_direction = (dy * dd).sum() / (dy * dy).sum() * dy
                    orthogonal_direction = dd - parallel_direction
                    # Scale orthogonal component
                    orthogonal_std = orthogonal_direction.std()
                    orthogonal_scale = min(1, apg_config.orthogonal_threshold / orthogonal_std)
                    orthogonal_direction = orthogonal_direction * orthogonal_scale
                    model_output = dy + (guidance_scale - 1) * orthogonal_direction
                else:
                    # Standard classifier-free guidance
                    model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
            else:
                # If no guidance needed, just run the model normally
                model_output = self.dit_model(latents, prompt_embeds, t_tensor)
            
            # Update latents
            acc_latents = acc_latents + dt * model_output.to(device)
            latents = acc_latents.clone()

        # 7. Decode latents
        # These checks handle the case where mypy doesn't recognize these attributes
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215) if hasattr(self.vae, "config") else 0.18215
        shift_factor = getattr(self.vae.config, "shift_factor", 0) if hasattr(self.vae, "config") else 0

        latents = latents / scaling_factor + shift_factor

        vae_dtype = self.vae.dtype if hasattr(self.vae, "dtype") else dtype
        decoded_images = self.vae.decode(latents.to(vae_dtype)).sample if hasattr(self.vae, "decode") else latents

        # Offload all models
        try:
            self.maybe_free_model_hooks()
        except AttributeError as e:
            if "OptimizedModule" in str(e):
                import warnings
                warnings.warn(
                    "Encountered 'OptimizedModule' error when offloading models. "
                    "This issue might be fixed in the future by: "
                    "https://github.com/huggingface/diffusers/pull/10730"
                )
            else:
                raise
            
        # 8. Post-process images
        images = (decoded_images / 2 + 0.5).clamp(0, 1)
        # Convert to PIL Images
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu()
        pil_images = [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in images]

        return FLitePipelineOutput(
            images=pil_images,
        ) 