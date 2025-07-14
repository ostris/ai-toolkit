# DONT USE THIS!. IT DOES NOT WORK YET!
# Will revisit this when they release more info on how it was trained. 

import weakref
from diffusers import CogView4Pipeline
import torch
import yaml

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds

import os
import copy
from toolkit.config_modules import ModelConfig, GenerateImageConfig, ModelArch
import torch
import diffusers
from diffusers import AutoencoderKL, CogView4Transformer2DModel, CogView4Pipeline
from optimum.quanto import freeze, qfloat8, QTensor, qint4
from toolkit.util.quantize import quantize, get_qtype
from transformers import GlmModel, AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from typing import TYPE_CHECKING
from toolkit.accelerator import unwrap_model
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork

# remove this after a bug is fixed in diffusers code. This is a workaround.


class FakeModel:
    def __init__(self, model):
        self.model_ref = weakref.ref(model)
        pass

    @property
    def device(self):
        return self.model_ref().device


scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.25,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 0.75,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "time_shift_type": "linear",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False
}


class CogView4(BaseModel):
    arch = 'cogview4'
    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(device, model_config, dtype,
                         custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['CogView4Transformer2DModel']

        # cache for holding noise
        self.effective_noise = None

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def load_model(self):
        dtype = self.torch_dtype
        base_model_path = "THUDM/CogView4-6B"
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading CogView4 model")
        # base_model_path = "black-forest-labs/FLUX.1-schnell"
        base_model_path = self.model_config.name_or_path_original
        subfolder = 'transformer'
        transformer_path = model_path
        if os.path.exists(transformer_path):
            subfolder = None
            transformer_path = os.path.join(transformer_path, 'transformer')
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, 'text_encoder')
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        self.print_and_status_update("Loading GlmModel")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder = GlmModel.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype)

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing GlmModel")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        # hack to fix diffusers bug workaround
        text_encoder.model = FakeModel(text_encoder)

        self.print_and_status_update("Loading transformer")
        transformer = CogView4Transformer2DModel.from_pretrained(
            transformer_path,
            subfolder=subfolder,
            torch_dtype=dtype,
        )

        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for CogViewModels models")

        transformer.to(self.quantize_device, dtype=dtype)
        flush()

        if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
            raise ValueError(
                "Assistant LoRA is not supported for CogViewModels models currently")

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for CogViewModels models currently")

        flush()

        if self.model_config.quantize:
            quantization_args = self.model_config.quantize_kwargs
            if 'exclude' not in quantization_args:
                quantization_args['exclude'] = []
            if 'include' not in quantization_args:
                quantization_args['include'] = []

            # Be more specific with the include pattern to exactly match transformer blocks
            quantization_args['include'] += ["transformer_blocks.*"]

            # Exclude all LayerNorm layers within transformer blocks
            quantization_args['exclude'] += [
                "transformer_blocks.*.norm1",
                "transformer_blocks.*.norm2",
                "transformer_blocks.*.norm2_context",
                "transformer_blocks.*.attn1.norm_q",
                "transformer_blocks.*.attn1.norm_k"
            ]

            # patch the state dict method
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update("Quantizing transformer")
            quantize(transformer, weights=quantization_type, **quantization_args)
            freeze(transformer)
            transformer.to(self.device_torch)
        else:
            transformer.to(self.device_torch, dtype=dtype)

        flush()

        scheduler = CogView4.get_train_scheduler()
        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype)
        flush()

        self.print_and_status_update("Making pipe")
        pipe: CogView4Pipeline = CogView4Pipeline(
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        text_encoder.to(self.device_torch)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()
        self.pipeline = pipe
        self.model = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def get_generation_pipeline(self):
        scheduler = CogView4.get_train_scheduler()
        pipeline = CogView4Pipeline(
            vae=self.vae,
            transformer=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=scheduler,
        )
        return pipeline

    def generate_single_image(
        self,
        pipeline: CogView4Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        # target_size = (height, width)
        target_size = latent_model_input.shape[-2:]
        # multiply by 8
        target_size = (target_size[0] * 8, target_size[1] * 8)
        crops_coords_top_left = torch.tensor(
            [(0, 0)], dtype=self.torch_dtype, device=self.device_torch)

        original_size = torch.tensor(
            [target_size], dtype=self.torch_dtype, device=self.device_torch)
        target_size = original_size.clone()
        noise_pred_cond = self.model(
            hidden_states=latent_model_input,
            encoder_hidden_states=text_embeddings.text_embeds,
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crops_coords_top_left,
            return_dict=False,
        )[0]
        return noise_pred_cond

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )
        return PromptEmbeds(prompt_embeds)

    def get_model_has_grad(self):
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        return self.text_encoder.layers[0].mlp.down_proj.weight.requires_grad

    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: CogView4Transformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w', encoding='utf-8') as f:
            yaml.dump(meta, f, allow_unicode=True)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        effective_noise = self.effective_noise
        batch = kwargs.get('batch')
        if batch is None:
            raise ValueError("Batch is not provided")
        if noise is None:
            raise ValueError("Noise is not provided")
        # return batch.latents
        # return (batch.latents - noise).detach()
        return (noise - batch.latents).detach()
        # return (batch.latents).detach()
        # return (effective_noise - batch.latents).detach()

    def _get_low_res_latents(self, latents):
        # todo prevent needing to do this and grab the tensor another way.
        with torch.no_grad():
            # Decode latents to image space
            images = self.decode_latents(
                latents, device=latents.device, dtype=latents.dtype)

            # Downsample by a factor of 2 using bilinear interpolation
            B, C, H, W = images.shape
            low_res_images = torch.nn.functional.interpolate(
                images,
                size=(H // 2, W // 2),
                mode="bilinear",
                align_corners=False
            )

            # Upsample back to original resolution to match expected VAE input dimensions
            upsampled_low_res_images = torch.nn.functional.interpolate(
                low_res_images,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

            # Encode the low-resolution images back to latent space
            low_res_latents = self.encode_images(
                upsampled_low_res_images, device=latents.device, dtype=latents.dtype)
            return low_res_latents

    # def add_noise(
    #         self,
    #         original_samples: torch.FloatTensor,
    #         noise: torch.FloatTensor,
    #         timesteps: torch.IntTensor,
    #         **kwargs,
    # ) -> torch.FloatTensor:
    #     relay_start_point = 500

    #     # Store original samples for loss calculation
    #     self.original_samples = original_samples

    #     # Prepare chunks for batch processing
    #     original_samples_chunks = torch.chunk(
    #         original_samples, original_samples.shape[0], dim=0)
    #     noise_chunks = torch.chunk(noise, noise.shape[0], dim=0)
    #     timesteps_chunks = torch.chunk(timesteps, timesteps.shape[0], dim=0)

    #     # Get the low res latents only if needed
    #     low_res_latents_chunks = None

    #     # Handle case where timesteps is a single value for all samples
    #     if len(timesteps_chunks) == 1 and len(timesteps_chunks) != len(original_samples_chunks):
    #         timesteps_chunks = [timesteps_chunks[0]] * len(original_samples_chunks)

    #     noisy_latents_chunks = []
    #     effective_noise_chunks = []  # Store the effective noise for each sample

    #     for idx in range(original_samples.shape[0]):
    #         t = timesteps_chunks[idx]
    #         t_01 = (t / 1000).to(original_samples_chunks[idx].device)

    #         # Flowmatching interpolation between original and noise
    #         if t > relay_start_point:
    #             # Standard flowmatching - direct linear interpolation
    #             noisy_latents = (1 - t_01) * original_samples_chunks[idx] + t_01 * noise_chunks[idx]
    #             effective_noise_chunks.append(noise_chunks[idx])  # Effective noise is just the noise
    #         else:
    #             # Relay flowmatching case - only compute low_res_latents if needed
    #             if low_res_latents_chunks is None:
    #                 low_res_latents = self._get_low_res_latents(original_samples)
    #                 low_res_latents_chunks = torch.chunk(low_res_latents, low_res_latents.shape[0], dim=0)

    #             # Calculate the relay ratio (0 to 1)
    #             t_ratio = t.float() / relay_start_point
    #             t_ratio = torch.clamp(t_ratio, 0.0, 1.0)

    #             # First blend between original and low-res based on t_ratio
    #             z0_t = (1 - t_ratio) * original_samples_chunks[idx] + t_ratio * low_res_latents_chunks[idx]

    #             added_lor_res_noise =  z0_t - original_samples_chunks[idx]

    #             # Then apply flowmatching interpolation between this blended state and noise
    #             noisy_latents = (1 - t_01) * z0_t + t_01 * noise_chunks[idx]

    #             # For prediction target, we need to store the effective "source"
    #             effective_noise_chunks.append(noise_chunks[idx] + added_lor_res_noise)

    #         noisy_latents_chunks.append(noisy_latents)

    #     noisy_latents = torch.cat(noisy_latents_chunks, dim=0)
    #     self.effective_noise = torch.cat(effective_noise_chunks, dim=0)  # Store for loss calculation

    #     return noisy_latents

    # def add_noise(
    #         self,
    #         original_samples: torch.FloatTensor,
    #         noise: torch.FloatTensor,
    #         timesteps: torch.IntTensor,
    #         **kwargs,
    # ) -> torch.FloatTensor:
    #     relay_start_point = 500

    #     # Store original samples for loss calculation
    #     self.original_samples = original_samples

    #     # Prepare chunks for batch processing
    #     original_samples_chunks = torch.chunk(
    #         original_samples, original_samples.shape[0], dim=0)
    #     noise_chunks = torch.chunk(noise, noise.shape[0], dim=0)
    #     timesteps_chunks = torch.chunk(timesteps, timesteps.shape[0], dim=0)

    #     # Get the low res latents only if needed
    #     low_res_latents = self._get_low_res_latents(original_samples)
    #     low_res_latents_chunks = torch.chunk(low_res_latents, low_res_latents.shape[0], dim=0)

    #     # Handle case where timesteps is a single value for all samples
    #     if len(timesteps_chunks) == 1 and len(timesteps_chunks) != len(original_samples_chunks):
    #         timesteps_chunks = [timesteps_chunks[0]] * len(original_samples_chunks)

    #     noisy_latents_chunks = []
    #     effective_noise_chunks = []  # Store the effective noise for each sample

    #     for idx in range(original_samples.shape[0]):
    #         t = timesteps_chunks[idx]
    #         t_01 = (t / 1000).to(original_samples_chunks[idx].device)

    #         lrln = low_res_latents_chunks[idx] - original_samples_chunks[idx]
    #         # lrln = lrln * (1 - t_01)

    #         # make the noise an interpolation between noise and low_res_latents with
    #         # being noise at t_01=1 and low_res_latents at t_01=0
    #         new_noise = t_01 * noise_chunks[idx] + (1 - t_01) * lrln
    #         # new_noise = noise_chunks[idx] + lrln
    #         # new_noise = noise_chunks[idx] + lrln

    #         # Then apply flowmatching interpolation between this blended state and noise
    #         noisy_latents = (1 - t_01) * original_samples + t_01 * new_noise

    #         # For prediction target, we need to store the effective "source"
    #         effective_noise_chunks.append(new_noise)

    #     noisy_latents_chunks.append(noisy_latents)

    #     noisy_latents = torch.cat(noisy_latents_chunks, dim=0)
    #     self.effective_noise = torch.cat(effective_noise_chunks, dim=0)  # Store for loss calculation

    #     return noisy_latents
