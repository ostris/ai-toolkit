import os
from typing import TYPE_CHECKING, List, Optional

import einops
import torch
import torchvision
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from diffusers import AutoencoderKL, TorchAoConfig
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.models.flux import add_model_gpu_splitter_to_flux, bypass_flux_guidance, restore_flux_guidance
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.accelerator import get_accelerator, unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.mask import generate_random_mask, random_dialate_mask
from toolkit.util.quantize import quantize, get_qtype
from transformers import T5TokenizerFast, T5EncoderModel, CLIPTextModel, CLIPTokenizer, TorchAoConfig as TorchAoConfigTransformers
from .src.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from .src.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from .src.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from einops import rearrange, repeat
import random
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
  "num_train_timesteps": 1000,
  "shift": 3.0
}

# LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-Instruct"
BASE_MODEL_PATH = "HiDream-ai/HiDream-I1-Full"


class HidreamModel(BaseModel):
    arch = "hidream"

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
        self.target_lora_modules = ['HiDreamImageTransformer2DModel']

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        # HiDream-ai/HiDream-I1-Full
        self.print_and_status_update("Loading HiDream model")
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        extras_path = self.model_config.extras_name_or_path
        
        llama_model_path = self.model_config.model_kwargs.get('llama_model_path', LLAMA_MODEL_PATH)
        
        scheduler = HidreamModel.get_train_scheduler()
        
        self.print_and_status_update("Loading llama 8b model")
        
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
            llama_model_path,
            use_fast=False
        )
        
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            llama_model_path,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16,
        )
        text_encoder_4.to(self.device_torch, dtype=dtype)
        
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing llama 8b model")
            quantization_type = get_qtype(self.model_config.qtype_te)
            quantize(text_encoder_4, weights=quantization_type)
            freeze(text_encoder_4)
        
        if self.low_vram:
            # unload it for now
            text_encoder_4.to('cpu')
            
        flush()
        
        self.print_and_status_update("Loading transformer")
            
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            model_path, 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16
        )
        
        if not self.low_vram:
            transformer.to(self.device_torch, dtype=dtype)
        
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing transformer")
            quantization_type = get_qtype(self.model_config.qtype)
            if self.low_vram:
                # move and quantize only certain pieces at a time.
                all_blocks = list(transformer.double_stream_blocks) + list(transformer.single_stream_blocks)
                self.print_and_status_update(" - quantizing transformer blocks")
                for block in tqdm(all_blocks):
                    block.to(self.device_torch, dtype=dtype)
                    quantize(block, weights=quantization_type)
                    freeze(block)
                    block.to('cpu')
                    # flush()
                
                self.print_and_status_update(" - quantizing extras")
                transformer.to(self.device_torch, dtype=dtype)
                quantize(transformer, weights=quantization_type)
                freeze(transformer)
            else: 
                quantize(transformer, weights=quantization_type)
                freeze(transformer)
            
        if self.low_vram:
            # unload it for now
            transformer.to('cpu')
        
        flush()
        
        self.print_and_status_update("Loading vae")
        
        vae = AutoencoderKL.from_pretrained(
            extras_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to(self.device_torch, dtype=dtype)
        
        
        self.print_and_status_update("Loading clip encoders")
        
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            extras_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16
        ).to(self.device_torch, dtype=dtype)
        
        tokenizer = CLIPTokenizer.from_pretrained(
            extras_path,
            subfolder="tokenizer"
        )
        
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            extras_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16
        ).to(self.device_torch, dtype=dtype)
        
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            extras_path,
            subfolder="tokenizer_2"
        )
        
        flush()
        self.print_and_status_update("Loading T5 encoders")
        
        text_encoder_3 = T5EncoderModel.from_pretrained(
            extras_path,
            subfolder="text_encoder_3",
            torch_dtype=torch.bfloat16
        ).to(self.device_torch, dtype=dtype)
        
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing T5")
            quantization_type = get_qtype(self.model_config.qtype_te)
            quantize(text_encoder_3, weights=quantization_type)
            freeze(text_encoder_3)
            flush()
        
        tokenizer_3 = T5Tokenizer.from_pretrained(
            extras_path,
            subfolder="tokenizer_3"
        )
        flush()
        
        if self.low_vram:
            self.print_and_status_update("Moving ecerything to device")
            # move it all back
            transformer.to(self.device_torch, dtype=dtype)
            vae.to(self.device_torch, dtype=dtype)
            text_encoder.to(self.device_torch, dtype=dtype)
            text_encoder_2.to(self.device_torch, dtype=dtype)
            text_encoder_4.to(self.device_torch, dtype=dtype)
            text_encoder_3.to(self.device_torch, dtype=dtype)
            
        # set to eval mode
        # transformer.eval()
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        text_encoder_4.eval()
        text_encoder_3.eval()

        pipe = HiDreamImagePipeline(
            scheduler=scheduler,
            vae=vae, 
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
            text_encoder_4=text_encoder_4,
            tokenizer_4=tokenizer_4,
            transformer=transformer,
        )

        flush()
        
        text_encoder_list = [text_encoder, text_encoder_2, text_encoder_3, text_encoder_4]
        tokenizer_list = [tokenizer, tokenizer_2, tokenizer_3, tokenizer_4]
        
        for te in text_encoder_list:
            # set the dtype
            te.to(self.device_torch, dtype=dtype)
            # freeze the model
            freeze(te)
            # set to eval mode
            te.eval()
            # set the requires grad to false
            te.requires_grad_(False)
        
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder_list  # list of text encoders
        self.tokenizer = tokenizer_list  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, 
            shift=3.0, 
            use_dynamic_shifting=False
        )
        
        pipeline: HiDreamImagePipeline = HiDreamImagePipeline(
            scheduler=scheduler,
            vae=self.vae, 
            text_encoder=self.text_encoder[0],
            tokenizer=self.tokenizer[0],
            text_encoder_2=self.text_encoder[1],
            tokenizer_2=self.tokenizer[1],
            text_encoder_3=self.text_encoder[2],
            tokenizer_3=self.tokenizer[2],
            text_encoder_4=self.text_encoder[3],
            tokenizer_4=self.tokenizer[3],
            transformer=unwrap_model(self.model),
            aggressive_unloading=self.low_vram
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: HiDreamImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            pooled_prompt_embeds=conditional_embeds.pooled_embeds,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_pooled_prompt_embeds=unconditional_embeds.pooled_embeds,
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
        batch_size = latent_model_input.shape[0]
        with torch.no_grad():
            if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
                B, C, H, W = latent_model_input.shape
                pH, pW = H // self.model.config.patch_size, W // self.model.config.patch_size

                img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
                img_ids = torch.zeros(pH, pW, 3)
                img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
                img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
                img_ids = img_ids.reshape(pH * pW, -1)
                img_ids_pad = torch.zeros(self.transformer.max_seq, 3)
                img_ids_pad[:pH*pW, :] = img_ids

                img_sizes = img_sizes.unsqueeze(0).to(latent_model_input.device)
                img_sizes = torch.cat([img_sizes] * batch_size, dim=0)
                img_ids = img_ids_pad.unsqueeze(0).to(latent_model_input.device)
                img_ids = torch.cat([img_ids] * batch_size, dim=0)
            else:
                img_sizes = img_ids = None

        dtype = self.model.dtype
        device = self.device_torch
        
        # Pack the latent
        if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
            B, C, H, W = latent_model_input.shape
            patch_size = self.transformer.config.patch_size
            pH, pW = H // patch_size, W // patch_size
            out = torch.zeros(
                (B, C, self.transformer.max_seq, patch_size * patch_size), 
                dtype=latent_model_input.dtype, 
                device=latent_model_input.device
            )
            latent_model_input = einops.rearrange(latent_model_input, 'B C (H p1) (W p2) -> B C (H W) (p1 p2)', p1=patch_size, p2=patch_size)
            out[:, :, 0:pH*pW] = latent_model_input 
            latent_model_input = out

        text_embeds = text_embeddings.text_embeds
        # run the to for the list
        text_embeds = [te.to(device, dtype=dtype) for te in text_embeds]
        
        noise_pred = self.transformer(
            hidden_states = latent_model_input,
            timesteps = timestep,
            encoder_hidden_states = text_embeds,
            pooled_embeds = text_embeddings.pooled_embeds.to(device, dtype=dtype),
            img_sizes = img_sizes,
            img_ids = img_ids,
            return_dict = False,
        )[0]
        noise_pred = -noise_pred

        return noise_pred
    
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        self.text_encoder_to(self.device_torch, dtype=self.torch_dtype)
        max_sequence_length = 128
        prompt_embeds, pooled_prompt_embeds = self.pipeline._encode_prompt(
            prompt = prompt,
            prompt_2 = prompt,
            prompt_3 = prompt,
            prompt_4 = prompt,
            device = self.device_torch,
            dtype = self.torch_dtype,
            num_images_per_prompt = 1,
            max_sequence_length = max_sequence_length,
        )
        pe = PromptEmbeds(
            [prompt_embeds, pooled_prompt_embeds]
        )
        return pe
    
    def get_model_has_grad(self):
        # return from a weight if it has grad
        return self.model.double_stream_blocks[0].block.attn1.to_q.weight.requires_grad

    def get_te_has_grad(self):
        # assume no one wants to finetune 4 text encoders.
        return False
    
    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: HiDreamImageTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w', encoding='utf-8') as f:
            yaml.dump(meta, f, allow_unicode=True)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        return (noise - batch.latents).detach()
    
    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ['double_stream_blocks', 'single_stream_blocks']

    def convert_lora_weights_before_save(self, state_dict):
        # currently starte with transformer. but needs to start with diffusion_model. for comfyui
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        # saved as diffusion_model. but needs to be transformer. for ai-toolkit
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
    
    def get_base_model_version(self):
        return "hidream_i1"
    
