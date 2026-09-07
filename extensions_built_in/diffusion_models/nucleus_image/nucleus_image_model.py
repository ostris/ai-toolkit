import itertools
import os
from typing import List, Optional

import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.vae.qwen_image import QwenImageVAE, QwenImageVAEHolderMixin
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
from toolkit.basic import flush
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model

from transformers import Qwen3VLProcessor
import torch.nn.functional as F

try:
    from diffusers import NucleusMoEImagePipeline, AutoencoderKLQwenImage
    from toolkit.models.v2.diffusion_models.nucleus_image import (
        NucleusMoEImageTransformer2DModel,
    )
    from diffusers.models.transformers.transformer_nucleusmoe_image import SwiGLUExperts
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


scheduler_config = {
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "invert_sigmas": False,
  "max_image_seq_len": 4096,
  "max_shift": 1.15,
  "num_train_timesteps": 1000,
  "shift": 1.0,
  "shift_terminal": None,
  "stochastic_sampling": False,
  "time_shift_type": "exponential",
  "use_beta_sigmas": False,
  "use_dynamic_shifting": False,
  "use_exponential_sigmas": False,
  "use_karras_sigmas": False
}


class NucleusImageModel(QwenImageVAEHolderMixin, BaseModel):
    arch = "nucleus_image"

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
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["NucleusMoEImageTransformer2DModel"]

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # 16 for the VAE, 2 for patch size

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Nucleus model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")

        if os.path.exists(model_path):
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, "text_encoder")
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        transformer = NucleusMoEImageTransformer2DModel.load_model(model_path, dtype=dtype)
        
        # handle versions of pytorch that don't have grouped mm, by disabling it in the SwiGLUExperts
        if not hasattr(torch.nn.functional, "grouped_mm"):
            for m in transformer.modules():
                if isinstance(m, SwiGLUExperts):
                    m.use_grouped_mm = False

        # quantize + offload + placement, all driven by model_config
        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = Qwen3VLProcessor.from_pretrained(
            base_model_path, subfolder="processor", torch_dtype=dtype
        )
        text_encoder = Qwen3VLTextEncoder.load(
            base_model_path, **self.component_load_kwargs("te")
        )

        self.print_and_status_update("Loading VAE")
        vae = QwenImageVAE.load_model(
            base_model_path, dtype=dtype, device=self.device_torch
        )

        self.noise_scheduler = NucleusImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        pipe: NucleusMoEImagePipeline = NucleusMoEImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            processor=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.processor]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # low_vram: the text encoder stays on cpu; get_prompt_embeds moves it
        # to the gpu on demand
        if not self.low_vram:
            text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = NucleusImageModel.get_train_scheduler()

        pipeline: NucleusMoEImagePipeline = NucleusMoEImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            processor=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: NucleusMoEImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
            if self.model_config.layer_offloading:
                parameters_and_buffers = itertools.chain(self.model.parameters(), self.model.buffers())
                next(parameters_and_buffers).to(self.device_torch)
                

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds[0].unsqueeze(0),
            prompt_embeds_mask=conditional_embeds.attention_mask[0].unsqueeze(0),
            negative_prompt_embeds=unconditional_embeds.text_embeds[0].unsqueeze(0),
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask[0].unsqueeze(0),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
            if self.model_config.layer_offloading:
                parameters_and_buffers = itertools.chain(self.model.parameters(), self.model.buffers())
                next(parameters_and_buffers).to(self.device_torch)
        
        with torch.no_grad():
            patch_size = self.pipeline.transformer.config.patch_size
            
            img_shape = (1, latent_model_input.shape[2] // patch_size, latent_model_input.shape[3] // patch_size)
            img_shapes = [
                img_shape for _ in range(latent_model_input.shape[0])
            ]
            latent_height = latent_model_input.shape[2]
            latent_width = latent_model_input.shape[3]
            
            pixel_height = latent_model_input.shape[2] * self.pipeline.vae_scale_factor
            pixel_width = latent_model_input.shape[3] * self.pipeline.vae_scale_factor
        
        latent_model_input = self.pipeline._pack_latents(
            latents=latent_model_input, 
            batch_size=latent_model_input.shape[0],
            num_channels_latents=self.pipeline.transformer.config.in_channels // 4,
            height=latent_height, 
            width=latent_width, 
            patch_size=patch_size,
        )

        pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            encoder_hidden_states=torch.stack(text_embeddings.text_embeds, dim=0),
            encoder_hidden_states_mask=torch.stack(text_embeddings.attention_mask, dim=0),
            img_shapes=img_shapes,
            return_dict=False,
        )[0]
        
        # invert it
        pred = -pred
        
        pred = self.pipeline._unpack_latents(
            latents=pred,
            height=pixel_height,
            width=pixel_width,
            patch_size=patch_size,
            vae_scale_factor=self.pipeline.vae_scale_factor
        )

        pred = pred.squeeze(2)  # remove frame dimension [B, C, 1, H, W] -> [B, C, H, W]

        return pred

    def get_prompt_embeds(self, prompt: str) -> AdvancedPromptEmbeds:
        if self.pipeline.text_encoder.device == torch.device("cpu"):
            self.pipeline.text_encoder.to(self.device_torch)

        if isinstance(prompt, str):
            prompt = [prompt]
        
        return_index = self.pipeline.default_return_index
        device = self.device_torch

        formatted = [self.pipeline._format_prompt(p) for p in prompt]

        inputs = self.pipeline.processor(
            text=formatted,
            padding="longest",
            pad_to_multiple_of=8,
            max_length=1024,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device=device)

        prompt_embeds_mask = inputs.attention_mask

        outputs = self.pipeline.text_encoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[return_index]
        prompt_embeds = prompt_embeds.to(dtype=self.pipeline.text_encoder.dtype, device=device)

        pe = AdvancedPromptEmbeds(
            text_embeds=[x for x in prompt_embeds],
            attention_mask=[x for x in prompt_embeds_mask],
        )
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: NucleusMoEImageTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    lora_keys_use_comfy_prefix = True

