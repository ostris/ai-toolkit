import os
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.accelerator import get_accelerator, unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.quantize import quantize, get_qtype, quantize_model
import torch.nn.functional as F

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from tqdm import tqdm

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "invert_sigmas": False,
  "max_image_seq_len": 8192,
  "max_shift": 0.9,
  "num_train_timesteps": 1000,
  "shift": 1.0,
  "shift_terminal": 0.02,
  "stochastic_sampling": False,
  "time_shift_type": "exponential",
  "use_beta_sigmas": False,
  "use_dynamic_shifting": True,
  "use_exponential_sigmas": False,
  "use_karras_sigmas": False
}



class QwenImageModel(BaseModel):
    arch = "qwen_image"

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
        self.target_lora_modules = ['QwenImageTransformer2DModel']

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2 # 16 for the VAE, 2 for patch size

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen Image model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        transformer_path = model_path
        transformer_subfolder = 'transformer'
        if os.path.exists(transformer_path):
            transformer_subfolder = None
            transformer_path = os.path.join(transformer_path, 'transformer')
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, 'text_encoder')
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        self.print_and_status_update("Loading transformer")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            transformer_path,
            subfolder=transformer_subfolder,
            torch_dtype=dtype
        )

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to('cpu')

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        
        # remove the visual model as it is not needed for image generation
        text_encoder.model.visual = None
        
        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(
                self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKLQwenImage.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype)

        self.noise_scheduler = QwenImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: QwenImagePipeline = QwenImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # just to make sure everything is on the right device and dtype
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
        scheduler = QwenImageModel.get_train_scheduler()

        pipeline: QwenImagePipeline = QwenImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer)
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: QwenImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        control_img = None
        if gen_config.ctrl_img is not None:
            raise NotImplementedError(
                "Control image generation is not supported in Qwen Image model... yet"
            )
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            # resize to width and height
            if control_img.size != (gen_config.width, gen_config.height):
                control_img = control_img.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )
        
        # flush for low vram if we are doing that
        flush_between_steps = self.model_config.low_vram
            # Fix a bug in diffusers/torch
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if flush_between_steps:
                flush()
            latents = callback_kwargs["latents"]
            
            return {"latents": latents}
        
        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width  // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_embeds_mask=conditional_embeds.attention_mask.to(self.device_torch, dtype=torch.int64),
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask.to(self.device_torch, dtype=torch.int64),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            true_cfg_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
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
        batch_size, num_channels_latents, height, width = latent_model_input.shape

        # pack image tokens
        latent_model_input = latent_model_input.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
        latent_model_input = latent_model_input.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        # clamp text length to RoPE capacity for this image size
        # img_shapes passed to the model
        img_h2, img_w2 = height // 2, width // 2
        img_shapes = [(1, img_h2, img_w2)] * batch_size

        # QwenEmbedRope logic:
        max_vid_index = max(img_h2 // 2, img_w2 // 2)

        rope_cap = 1024 - max_vid_index  # available text positions in RoPE cache
        seq_len_actual = text_embeddings.text_embeds.shape[1]
        use_len = min(seq_len_actual, rope_cap)

        enc_hs = text_embeddings.text_embeds[:, :use_len].to(self.device_torch, self.torch_dtype)
        prompt_embeds_mask = text_embeddings.attention_mask.to(self.device_torch, dtype=torch.int64)[:, :use_len]
        txt_seq_lens = [use_len] * batch_size

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(self.device_torch, self.torch_dtype),
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
            **kwargs,
        )[0]

        # unpack
        noise_pred = noise_pred.view(batch_size, height // 2, width // 2, num_channels_latents, 2, 2)
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred
    
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        
        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt,
            device=self.device_torch,
            num_images_per_prompt=1,
        )
        pe = PromptEmbeds(
            prompt_embeds
        )
        pe.attention_mask = prompt_embeds_mask
        return pe
    
    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False
    
    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: QwenImageTransformer2DModel = unwrap_model(self.model)
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


    def get_base_model_version(self):
        return "qwen_image"
    
    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ['transformer_blocks']
    
    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
    
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

        # Move to vae to device if on cpu
        if self.vae.device == 'cpu':
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        # move to device and dtype
        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)
        # it uses wan vae, so add dim for frame count

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
        
        
        latents = latents.squeeze(2)  # remove the frame count dimension

        return latents