import inspect
import os
from typing import TYPE_CHECKING, List, Optional

import einops
import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from diffusers import AutoencoderKL
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from .src.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from .src.models.transformers import OmniGen2Transformer2DModel
from .src.models.transformers.repo import OmniGen2RotaryPosEmbed
from .src.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler as OmniFlowMatchEuler
from transformers import CLIPProcessor, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
  "num_train_timesteps": 1000
}

BASE_MODEL_PATH = "OmniGen2/OmniGen2"


class OmniGen2Model(BaseModel):
    arch = "omnigen2"

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
        self.target_lora_modules = ['OmniGen2Transformer2DModel']

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        # HiDream-ai/HiDream-I1-Full
        self.print_and_status_update("Loading OmniGen2 model")
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        extras_path = self.model_config.extras_name_or_path
        
        scheduler = OmniGen2Model.get_train_scheduler()
        
        self.print_and_status_update("Loading Qwen2.5 VL")
        processor = CLIPProcessor.from_pretrained(
            extras_path,
            subfolder="processor",
            use_fast=True
        )
        
        mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            extras_path,
            subfolder="mllm",
            torch_dtype=torch.bfloat16
        )
        
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen2.5 VL model")
            quantization_type = get_qtype(self.model_config.qtype_te)
            quantize(mllm, weights=quantization_type)
            freeze(mllm)
        
        if self.low_vram:
            # unload it for now
            mllm.to('cpu')
            
        flush()
        
        self.print_and_status_update("Loading transformer")
            
        transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_path, 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16
        )
        
        if not self.low_vram:
            transformer.to(self.device_torch, dtype=dtype)
        
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing transformer")
            quantization_type = get_qtype(self.model_config.qtype)
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
        
        
        flush()
        self.print_and_status_update("Loading Qwen2.5 VLProcessor")
        
        flush()
        
        if self.low_vram:
            self.print_and_status_update("Moving everything to device")
            # move it all back
            transformer.to(self.device_torch, dtype=dtype)
            vae.to(self.device_torch, dtype=dtype)
            mllm.to(self.device_torch, dtype=dtype)
            
        # set to eval mode
        # transformer.eval()
        vae.eval()
        mllm.eval()
        mllm.requires_grad_(False)

        pipe: OmniGen2Pipeline = OmniGen2Pipeline(
            transformer=transformer,
            vae=vae, 
            scheduler=scheduler,
            mllm=mllm,
            processor=processor,
        )

        # pipe: OmniGen2Pipeline = OmniGen2Pipeline.from_pretrained(
        #     model_path,
        #     transformer=transformer,
        #     vae=vae, 
        #     scheduler=scheduler,
        #     mllm=mllm,
        #     trust_remote_code=True,
        # )
        # processor = pipe.processor

        flush()
        
        text_encoder_list = [mllm]
        tokenizer_list = [processor]
        
        
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder_list  # list of text encoders
        self.tokenizer = tokenizer_list  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        
        self.freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            transformer.config.axes_dim_rope,
            transformer.config.axes_lens,
            theta=10000,
        )
        
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = OmniFlowMatchEuler(
            dynamic_time_shift=True,
            num_train_timesteps=1000
        )
        
        pipeline: OmniGen2Pipeline = OmniGen2Pipeline(
            transformer=self.model,
            vae=self.vae,
            scheduler=scheduler,
            mllm=self.text_encoder[0],
            processor=self.tokenizer[0],
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: OmniGen2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_attention_mask=conditional_embeds.attention_mask,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_prompt_attention_mask=unconditional_embeds.attention_mask,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            text_guidance_scale=gen_config.guidance_scale,
            image_guidance_scale=1.0, # reference image guidance scale. Add this for controls
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
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timestep.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
        
        # optional_kwargs = {}
        # if 'ref_image_hidden_states' in set(inspect.signature(self.model.forward).parameters.keys()):
        #     optional_kwargs['ref_image_hidden_states'] = ref_image_hidden_states
        
        timesteps = timestep / 1000  # convert to 0 to 1 scale
        # timestep for model starts at 0 instead of 1. So we need to reverse them
        timestep = 1 - timesteps
        model_pred = self.model(
            latent_model_input,
            timestep,
            text_embeddings.text_embeds,
            self.freqs_cis,
            text_embeddings.attention_mask,
            ref_image_hidden_states=None, # todo add ref latent ability
        )

        return model_pred
    
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [self.pipeline._apply_chat_template(_prompt) for _prompt in prompt]
        self.text_encoder_to(self.device_torch, dtype=self.torch_dtype)
        max_sequence_length = 256
        prompt_embeds, prompt_attention_mask, _, _ = self.pipeline.encode_prompt(
            prompt = prompt,
            do_classifier_free_guidance=False,
            device=self.device_torch,
            max_sequence_length=max_sequence_length,
        )
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_attention_mask
        return pe
    
    def get_model_has_grad(self):
        # return from a weight if it has grad
        return False

    def get_te_has_grad(self):
        # assume no one wants to finetune 4 text encoders.
        return False
    
    def save_model(self, output_path, meta, save_dtype):
        # only save the transformer
        transformer: OmniGen2Transformer2DModel = unwrap_model(self.model)
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
        # return (noise - batch.latents).detach()
        return (batch.latents - noise).detach()
    
    def get_transformer_block_names(self) -> Optional[List[str]]:
        # omnigen2 had a few blocks for things like noise_refiner, ref_image_refiner, context_refiner, and layers.
        # lets do all but image refiner until we add it
        return ['noise_refiner', 'context_refiner', 'layers']
        # return ['layers']

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
        return "omnigen2"
    
