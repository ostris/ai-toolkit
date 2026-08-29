import os
from typing import TYPE_CHECKING

import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.text_encoders.t5 import T5TextEncoder
from toolkit.models.v2.vae.autoencoder_kl import KLVAE
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.accelerator import unwrap_model
from optimum.quanto import QTensor

from .src import FLitePipeline, DiT

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True
}


class FLiteModel(BaseModel):
    arch = "f-lite"

    def get_transformer_block_names(self):
        return ["blocks"]

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['DiT']

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
    
    def get_bucket_divisibility(self):
        # return the bucket divisibility for the model
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        
        extras_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")

        transformer = DiT.load(
            model_path, **self.component_load_kwargs("transformer")
        )

        flush()

        self.print_and_status_update("Loading T5")
        tokenizer = T5TextEncoder.load_tokenizer(extras_path, subfolder="tokenizer")
        text_encoder = T5TextEncoder.load(
            extras_path, subfolder="text_encoder", **self.component_load_kwargs("te")
        )

        self.noise_scheduler = FLiteModel.get_train_scheduler()
        
        self.print_and_status_update("Loading VAE")
        vae = KLVAE.load_model(extras_path, dtype=dtype, device=self.device_torch)

        self.print_and_status_update("Making pipe")

        pipe: FLitePipeline = FLitePipeline(
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            dit_model=None,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.dit_model = transformer
        pipe.transformer = transformer
        pipe.scheduler = self.noise_scheduler,

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # low_vram: the text encoder stays on cpu; get_prompt_embeds moves it
        # to the gpu on demand
        if not self.low_vram:
            text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = FLiteModel.get_train_scheduler()
        # it has built in scheduler. Basically euler flowmatching
        pipeline = FLitePipeline(
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            dit_model=unwrap_model(self.transformer)
        )
        pipeline.transformer = pipeline.dit_model
        pipeline.scheduler = scheduler

        return pipeline

    def generate_single_image(
        self,
        pipeline: FLitePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):

        extra['negative_prompt_embeds'] = unconditional_embeds.text_embeds
        
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        cast_dtype = self.unet.dtype

        noise_pred = self.unet(
            latent_model_input.to(
                self.device_torch, cast_dtype
            ),
            text_embeddings.text_embeds.to(
                self.device_torch, cast_dtype
            ),
            timestep / 1000,
        )

        if isinstance(noise_pred, QTensor):
            noise_pred = noise_pred.dequantize()
        
        return noise_pred
    
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, negative_embeds = self.pipeline.encode_prompt(
            prompt=prompts, 
            negative_prompt=None, 
            device=self.text_encoder[0].device, 
            dtype=self.torch_dtype,
        )
        
        pe = PromptEmbeds(prompt_embeds)
        
        return pe
    
    def get_model_has_grad(self):
        # return from a weight if it has grad
        return False

    def get_te_has_grad(self):
        # return from a weight if it has grad
        return False
    
    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: DiT = unwrap_model(self.model)
        # diffusers
        # only save the unet
        transformer: DiT = unwrap_model(self.transformer)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'dit_model'),
            safe_serialization=True,
        )
        # save out meta config
        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        # return (noise - batch.latents).detach()
        return (batch.latents - noise).detach()
    
    lora_keys_use_comfy_prefix = True

    
    def get_base_model_version(self):
        return "f-lite"
    
    def get_stepped_pred(self, pred, noise):
        # just used for DFE support
        latents = pred + noise
        return latents
