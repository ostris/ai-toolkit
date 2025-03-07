# WIP, coming soon ish
import torch
import yaml
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.paths import REPOS_ROOT
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
import os
import sys

import weakref
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
from optimum.quanto import freeze, qfloat8, QTensor, qint4
from toolkit.util.quantize import quantize
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from typing import TYPE_CHECKING, List
from toolkit.accelerator import unwrap_model
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from torchvision.transforms import Resize, ToPILImage

# for generation only?
scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.33.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False
}

# for training. I think it is right
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False
}


class Wan21(BaseModel):
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
        self.target_lora_modules = ['WanTransformer3DModel']

        # cache for holding noise
        self.effective_noise = None

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def load_model(self):
        dtype = self.torch_dtype
        # todo , will this work with other wan models?
        base_model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading Wan2.1 model")
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

        self.print_and_status_update("Loading UMT5EncoderModel")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder = UMT5EncoderModel.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype)

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing UMT5EncoderModel")
            quantize(text_encoder, weights=qfloat8)
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading transformer")
        transformer = WanTransformer3DModel.from_pretrained(
            transformer_path,
            subfolder=subfolder,
            torch_dtype=dtype,
        )

        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.1 models")

        transformer.to(self.quantize_device, dtype=dtype)
        flush()

        if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
            raise ValueError(
                "Assistant LoRA is not supported for Wan2.1 models currently")

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for Wan2.1 models currently")

        flush()

        if self.model_config.quantize:
            quantization_args = self.model_config.quantize_kwargs
            if 'exclude' not in quantization_args:
                quantization_args['exclude'] = []
            # patch the state dict method
            patch_dequantization_on_save(transformer)
            quantization_type = qfloat8
            self.print_and_status_update("Quantizing transformer")
            quantize(transformer, weights=quantization_type,
                     **quantization_args)
            freeze(transformer)
            transformer.to(self.device_torch)
        else:
            transformer.to(self.device_torch, dtype=dtype)

        flush()

        scheduler = Wan21.get_train_scheduler()
        self.print_and_status_update("Loading VAE")
        # todo, example does float 32? check if quality suffers
        vae = AutoencoderKLWan.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype)
        flush()

        self.print_and_status_update("Making pipe")
        pipe: WanPipeline = WanPipeline(
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
        scheduler = UniPCMultistepScheduler(**scheduler_configUniPC)
        pipeline = WanPipeline(
            vae=self.vae,
            transformer=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=scheduler,
        )
        return pipeline

    def generate_single_image(
        self,
        pipeline: WanPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # todo, figure out how to do video
        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        # vae_scale_factor_spatial = 8
        # vae_scale_factor_temporal = 4
        # num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # shape = (
        #     batch_size,
        #     num_channels_latents, # 16
        #     num_latent_frames,  # 81
        #     int(height) // self.vae_scale_factor_spatial,
        #     int(width) // self.vae_scale_factor_spatial,
        # )

        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=512,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )
        return PromptEmbeds(prompt_embeds)

    @torch.no_grad()
    def encode_images(
            self,
            image_list: List[torch.Tensor],
            device=None,
            dtype=None
    ):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        latent_list = []
        # Move to vae to device if on cpu
        if self.vae.device == 'cpu':
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        # move to device and dtype
        image_list = [image.to(device, dtype=dtype) for image in image_list]

        VAE_SCALE_FACTOR = 8

        # resize images if not divisible by 8
        for i in range(len(image_list)):
            image = image_list[i]
            if image.shape[1] % VAE_SCALE_FACTOR != 0 or image.shape[2] % VAE_SCALE_FACTOR != 0:
                image_list[i] = Resize((image.shape[1] // VAE_SCALE_FACTOR * VAE_SCALE_FACTOR,
                                        image.shape[2] // VAE_SCALE_FACTOR * VAE_SCALE_FACTOR))(image)

        images = torch.stack(image_list)
        images = images.unsqueeze(2)
        latents = self.vae.encode(images).latent_dist.sample()
        
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std
        
        latents = latents.to(device, dtype=dtype)

        return latents

    def get_model_has_grad(self):
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        return self.text_encoder.encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad

    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: Wan21 = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        if batch is None:
            raise ValueError("Batch is not provided")
        if noise is None:
            raise ValueError("Noise is not provided")
        return (noise - batch.latents).detach()
