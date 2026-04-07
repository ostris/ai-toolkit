import json

import torch

from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds


class BaseAudioModel(BaseModel):
    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_audio_model = True

    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # This is called on the base model. We override it to make it make more sense for audio models.
        return self.generate_single_audio(
            pipeline,
            gen_config,
            conditional_embeds,
            unconditional_embeds,
            generator,
            extra,
        )

    def generate_single_audio(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # This is called on the base model. We override it to make it make more sense for audio models.
        raise NotImplementedError(
            "generate_single_audio is not implemented for this model"
        )
    
    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False
    
    def save_model(self, output_path, meta, save_dtype):
        # we need to save the model, vae, text encoder, and tokenizer together since they are all trained together and depend on each other
        raise NotImplementedError("save_model is not implemented for this model. Use the pipeline directly instead.")
    
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
