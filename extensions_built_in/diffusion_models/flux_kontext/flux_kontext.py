import os
from typing import TYPE_CHECKING, List

import torch
import torchvision
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.text_encoders.t5 import T5TextEncoder
from toolkit.models.v2.text_encoders.clip import CLIPTextEncoder
from toolkit.models.v2.vae.autoencoder_kl import KLVAE
from diffusers import FluxKontextPipeline
from toolkit.models.v2.diffusion_models.flux import FluxTransformer2DModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.models.flux import add_model_gpu_splitter_to_flux, bypass_flux_guidance, restore_flux_guidance
from toolkit.accelerator import get_accelerator, unwrap_model
from optimum.quanto import QTensor
from toolkit.util.mask import generate_random_mask, random_dialate_mask

from einops import rearrange, repeat
import random
import torch.nn.functional as F

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

class FluxKontextModel(BaseModel):
    arch = "flux_kontext"

    def get_transformer_block_names(self):
        return ["transformer_blocks", "single_transformer_blocks"]

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
        self.target_lora_modules = ['FluxTransformer2DModel']

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Flux Kontext model")
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        # this is the original path put in the model directory
        # it is here because for finetuning we only save the transformer usually
        # so we need this for the VAE, te, etc
        base_model_path = self.model_config.extras_name_or_path

        if os.path.exists(model_path):
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, 'text_encoder')
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        self.print_and_status_update("Loading transformer")
        transformer = FluxTransformer2DModel.load(
            model_path, **self.component_load_kwargs("transformer")
        )

        flush()

        self.print_and_status_update("Loading T5")
        tokenizer_2 = T5TextEncoder.load_tokenizer(base_model_path)
        text_encoder_2 = T5TextEncoder.load(
            base_model_path, **self.component_load_kwargs("te")
        )

        self.print_and_status_update("Loading CLIP")
        text_encoder = CLIPTextEncoder.load_model(
            base_model_path, dtype=dtype, device=self.device_torch
        )
        tokenizer = CLIPTextEncoder.load_tokenizer(base_model_path, use_fast=False)

        self.print_and_status_update("Loading VAE")
        vae = KLVAE.load_model(base_model_path, dtype=dtype)

        self.noise_scheduler = FluxKontextModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: FluxKontextPipeline = FluxKontextPipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder_2 = text_encoder_2
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder, pipe.text_encoder_2]
        tokenizer = [pipe.tokenizer, pipe.tokenizer_2]

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # low_vram: text encoders stay on cpu; get_prompt_embeds moves them
        # to the gpu on demand
        if not self.low_vram:
            text_encoder[0].to(self.device_torch)
            text_encoder[1].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        text_encoder[1].requires_grad_(False)
        text_encoder[1].eval()
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
        scheduler = FluxKontextModel.get_train_scheduler()

        pipeline: FluxKontextPipeline = FluxKontextPipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            text_encoder_2=unwrap_model(self.text_encoder[1]),
            tokenizer_2=self.tokenizer[1],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer)
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: FluxKontextPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if gen_config.ctrl_img is None:
            raise ValueError(
                "Control image is required for Flux Kontext model generation."
            )
        else:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            # resize to width and height
            if control_img.size != (gen_config.width, gen_config.height):
                control_img = control_img.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )
        gen_config.width = int(gen_config.width  // 16 * 16)
        gen_config.height = int(gen_config.height // 16 * 16)
        img = pipeline(
            image=control_img,
            prompt_embeds=conditional_embeds.text_embeds,
            pooled_prompt_embeds=conditional_embeds.pooled_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            max_area=gen_config.height * gen_config.width,
            _auto_resize=False,
            **extra
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        guidance_embedding_scale: float,
        bypass_guidance_embedding: bool,
        **kwargs
    ):
        with torch.no_grad():
            bs, c, h, w = latent_model_input.shape
            # if we have a control on the channel dimension, put it on the batch for packing
            has_control = False
            if latent_model_input.shape[1] == 32:
                # chunk it and stack it on batch dimension
                # dont update batch size for img_its
                lat, control = torch.chunk(latent_model_input, 2, dim=1)
                latent_model_input = torch.cat([lat, control], dim=0)
                has_control = True

            latent_model_input_packed = rearrange(
                latent_model_input,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=2,
                pw=2
            )

            img_ids = torch.zeros(h // 2, w // 2, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c",
                             b=bs).to(self.device_torch)
            
            # handle control image ids
            if has_control:
                ctrl_ids = img_ids.clone()
                ctrl_ids[..., 0] = 1
                img_ids = torch.cat([img_ids, ctrl_ids], dim=1)
                

            txt_ids = torch.zeros(
                bs, text_embeddings.text_embeds.shape[1], 3).to(self.device_torch)

            # # handle guidance
            if self.unet_unwrapped.config.guidance_embeds:
                if isinstance(guidance_embedding_scale, list):
                    guidance = torch.tensor(
                        guidance_embedding_scale, device=self.device_torch)
                else:
                    guidance = torch.tensor(
                        [guidance_embedding_scale], device=self.device_torch)
                    # Expand guidance to match original batch_size
                    guidance = guidance.expand(bs)
            else:
                guidance = None

        if bypass_guidance_embedding:
            bypass_flux_guidance(self.unet)

        cast_dtype = self.unet.dtype
        # changes from orig implementation
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        
        latent_size = latent_model_input_packed.shape[1]
        # move the kontext channels. We have them on batch dimension to here, but need to put them on the latent dimension
        if has_control:
            latent, control = torch.chunk(latent_model_input_packed, 2, dim=0)
            latent_model_input_packed = torch.cat(
                [latent, control], dim=1
            )
            latent_size = latent.shape[1]

        noise_pred = self.unet(
            hidden_states=latent_model_input_packed.to(
                self.device_torch, cast_dtype),
            timestep=timestep / 1000,
            encoder_hidden_states=text_embeddings.text_embeds.to(
                self.device_torch, cast_dtype),
            pooled_projections=text_embeddings.pooled_embeds.to(
                self.device_torch, cast_dtype),
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=guidance,
            return_dict=False,
            **kwargs,
        )[0]
        
        # remove kontext image conditioning
        noise_pred = noise_pred[:, :latent_size]

        if isinstance(noise_pred, QTensor):
            noise_pred = noise_pred.dequantize()

        noise_pred = rearrange(
            noise_pred,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=latent_model_input.shape[2] // 2,
            w=latent_model_input.shape[3] // 2,
            ph=2,
            pw=2,
            c=self.vae.config.latent_channels
        )

        if bypass_guidance_embedding:
            restore_flux_guidance(self.unet)
        
        return noise_pred
    
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        prompt_embeds, pooled_prompt_embeds = train_tools.encode_prompts_flux(
            self.tokenizer,
            self.text_encoder,
            prompt,
            max_length=512,
        )
        pe = PromptEmbeds(
            prompt_embeds
        )
        pe.pooled_embeds = pooled_prompt_embeds
        return pe
    
    def get_model_has_grad(self):
        # return from a weight if it has grad
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        # return from a weight if it has grad
        return self.text_encoder[1].encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad
    
    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: FluxTransformer2DModel = unwrap_model(self.model)
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
        return (noise - batch.latents).detach()
    
    def condition_noisy_latents(self, latents: torch.Tensor, batch:'DataLoaderBatchDTO'):
        with torch.no_grad():
            control_tensor = batch.control_tensor
            if control_tensor is not None:
                self.vae.to(self.device_torch)
                # we are not packed here, so we just need to pass them so we can pack them later
                control_tensor = control_tensor * 2 - 1
                control_tensor = control_tensor.to(self.vae_device_torch, dtype=self.torch_dtype)
                
                # if it is not the size of batch.tensor, (bs,ch,h,w) then we need to resize it
                if batch.tensor is not None:
                    target_h, target_w = batch.tensor.shape[2], batch.tensor.shape[3]
                else:
                    # When caching latents, batch.tensor is None. We get the size from the file_items instead.
                    target_h = batch.file_items[0].crop_height
                    target_w = batch.file_items[0].crop_width

                if control_tensor.shape[2] != target_h or control_tensor.shape[3] != target_w:
                    control_tensor = F.interpolate(control_tensor, size=(target_h, target_w), mode='bilinear')
                    
                control_latent = self.encode_images(control_tensor).to(latents.device, latents.dtype)
                latents = torch.cat((latents, control_latent), dim=1)

        return latents.detach() 

    def get_base_model_version(self):
        return "flux.1_kontext"