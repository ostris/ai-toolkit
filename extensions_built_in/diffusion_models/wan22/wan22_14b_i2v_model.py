import torch
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
import torch
from toolkit.config_modules import GenerateImageConfig
from .wan22_pipeline import Wan22Pipeline

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from diffusers import WanImageToVideoPipeline
from torchvision.transforms import functional as TF

from .wan22_14b_model import Wan2214bModel

class Wan2214bI2VModel(Wan2214bModel):
    arch = "wan22_14b_i2v"
    
    
    def generate_single_image(
        self,
        pipeline: Wan22Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        
        # todo 
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)

        num_frames = (
            (gen_config.num_frames - 1) // 4
        ) * 4 + 1  # make sure it is divisible by 4 + 1
        gen_config.num_frames = num_frames

        height = gen_config.height
        width = gen_config.width
        
        d = self.get_bucket_divisibility()
        
        # make sure they are divisible by d
        height = height // d * d
        width = width // d * d
        
        # 5. Prepare latent variables
        # num_channels_latents = self.transformer.config.in_channels
        num_channels_latents = 16
        latents = pipeline.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            gen_config.num_frames,
            torch.float32,
            self.device_torch,
            generator,
            None,
        ).to(self.torch_dtype)
        
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img).convert("RGB")
            
            # resize the control image
            control_img = control_img.resize((width, height), Image.LANCZOS)

            first_frame_n1p1 = (
                TF.to_tensor(control_img)
                .unsqueeze(0)
                .to(self.device_torch, dtype=self.torch_dtype)
                * 2.0
                - 1.0
            )  # normalize to [-1, 1]
        else:
            # Generate dummy first frame when no control image is provided
            # Use a solid gray color (0.0 in [-1, 1] range) instead of random noise
            # This allows baseline sampling to work without requiring a control image
            # and produces a more reasonable starting point than pure noise
            # Create on CPU first (safer) then move to device
            first_frame_n1p1 = torch.zeros(
                1, 3, height, width,
                dtype=self.torch_dtype
            ).to(self.device_torch)  # Gray color (0.0 in [-1, 1] range)
        
        # Add conditioning using the standalone function
        gen_config.latents = add_first_frame_conditioning(
            latent_model_input=latents,
            first_frame=first_frame_n1p1,
            vae=self.vae
        )

        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            height=height,
            width=width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra,
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img
    
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs
    ):
        # videos come in (bs, num_frames, channels, height, width)
        # images come in (bs, channels, height, width)
        with torch.no_grad():
            # Check if we have frames/tensor data for i2v conditioning
            # If batch.tensor is None (e.g., when latents are cached), try to use cached first frames
            if batch.tensor is not None and batch.dataset_config.do_i2v:
                frames = batch.tensor
                if len(frames.shape) == 4:
                    first_frames = frames
                elif len(frames.shape) == 5:
                    first_frames = frames[:, 0]
                else:
                    raise ValueError(f"Unknown frame shape {frames.shape}")
                
                # Ensure VAE is on the correct device before encoding
                target_device = latent_model_input.device
                vae_was_on_cpu = next(self.vae.parameters()).device.type == 'cpu'
                if vae_was_on_cpu:
                    self.vae.to(target_device)
                
                # Add conditioning using the standalone function
                conditioned_latent = add_first_frame_conditioning(
                    latent_model_input=latent_model_input,
                    first_frame=first_frames,
                    vae=self.vae
                )
                
                # Move VAE back to CPU if it was there before (to save memory)
                if vae_was_on_cpu:
                    self.vae.to('cpu')
            elif batch.first_frame_tensor is not None and batch.dataset_config.do_i2v:
                # Use cached first frames when batch.tensor is None (latents are cached)
                first_frames = batch.first_frame_tensor
                
                # Ensure VAE is on the correct device before encoding
                target_device = latent_model_input.device
                vae_was_on_cpu = next(self.vae.parameters()).device.type == 'cpu'
                if vae_was_on_cpu:
                    self.vae.to(target_device)
                
                # Add conditioning using the standalone function
                conditioned_latent = add_first_frame_conditioning(
                    latent_model_input=latent_model_input,
                    first_frame=first_frames,
                    vae=self.vae
                )
                
                # Move VAE back to CPU if it was there before (to save memory)
                if vae_was_on_cpu:
                    self.vae.to('cpu')
            elif batch.dataset_config.do_i2v:
                # i2v is enabled but no frames are available - this is an error
                raise ValueError(
                    "i2v conditioning requires either batch.tensor or batch.first_frame_tensor, "
                    "but both are None. This usually means:\n"
                    "1. Latents were cached before first frame caching was implemented, OR\n"
                    "2. First frames failed to cache. Please re-cache latents to also cache first frames.\n"
                    "To fix: Delete the _latent_cache folder or set cache_latents_to_disk: false temporarily, "
                    "then re-enable caching to regenerate caches with first frames included."
                )
            else:
                # i2v not enabled - use latent_model_input as-is
                conditioned_latent = latent_model_input
        
        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred