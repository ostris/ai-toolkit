import os
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.vae.qwen_image import QwenImageVAE, QwenImageVAEHolderMixin
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import get_accelerator, unwrap_model
import torch.nn.functional as F
from toolkit.memory_management import MemoryManager
from toolkit.metadata import get_meta_for_safetensors
from safetensors.torch import load_file

from diffusers import (
    QwenImagePipeline,
    AutoencoderKLQwenImage,
)
from transformers import Qwen2VLProcessor
from toolkit.models.v2.diffusion_models.qwen_image import QwenImageTransformer2DModel
from toolkit.models.v2.text_encoders.qwen25_vl import Qwen25VLTextEncoder
from tqdm import tqdm
from toolkit.util.qwen_vae_gradient_checkpointing import patch_qwen_vae_gradient_checkpointing

patch_qwen_vae_gradient_checkpointing()

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
    "use_karras_sigmas": False,
}


class QwenImageModel(QwenImageVAEHolderMixin, BaseModel):
    arch = "qwen_image"
    _qwen_image_keep_visual = False
    _qwen_pipeline = QwenImagePipeline

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
        self.target_lora_modules = ["QwenImageTransformer2DModel"]

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # 16 for the VAE, 2 for patch size

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen Image model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path
        model_dtype = dtype

        if base_model_path.endswith(".safetensors"):
            # use the repo for extras
            base_model_path = "Qwen/Qwen-Image"

        self.print_and_status_update("Loading transformer")

        if not model_path.endswith(".safetensors") and os.path.exists(model_path):
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, "text_encoder")
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        # load + quantize + offload + placement, all driven by model_config
        transformer = QwenImageTransformer2DModel.load(
            model_path, **self.component_load_kwargs("transformer")
        )
        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = Qwen25VLTextEncoder.load_tokenizer(base_model_path, use_fast=False)
        text_encoder = Qwen25VLTextEncoder.load_model(base_model_path, dtype=dtype)

        # remove the visual model as it is not needed for image generation
        # (before quantization, so the dead tower is never quantized)
        self.processor = None
        if not self._qwen_image_keep_visual:
            text_encoder.drop_vision_tower()

        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))

        self.print_and_status_update("Loading VAE")
        vae = QwenImageVAE.load_model(base_model_path, dtype=dtype)

        self.noise_scheduler = QwenImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        if self._qwen_image_keep_visual:
            try:
                self.processor = Qwen2VLProcessor.from_pretrained(
                    model_path, subfolder="processor"
                )
            except OSError:
                self.processor = Qwen2VLProcessor.from_pretrained(
                    base_model_path, subfolder="processor"
                )
            kwargs["processor"] = self.processor

        pipe: QwenImagePipeline = self._qwen_pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
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
        scheduler = QwenImageModel.get_train_scheduler()

        pipeline: QwenImagePipeline = QwenImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
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
        self.model.to(self.device_torch)

        # flush for low vram if we are doing that
        flush_between_steps = self.model_config.low_vram

        # Fix a bug in diffusers/torch
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if flush_between_steps:
                flush()
            latents = callback_kwargs["latents"]

            return {"latents": latents}

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        if self.model_config.low_vram:
            # set vae to tile decode
            pipeline.vae.enable_tiling()

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_embeds_mask=conditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.int64
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.int64
            ),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            true_cfg_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            **extra,
        ).images[0]

        if self.model_config.low_vram:
            # restore no tiling
            pipeline.vae.disable_tiling()

        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)
        batch_size, num_channels_latents, height, width = latent_model_input.shape

        ps = self.transformer.config.patch_size

        # pack image tokens
        latent_model_input = latent_model_input.view(
            batch_size, num_channels_latents, height // ps, ps, width // ps, ps
        )
        latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
        latent_model_input = latent_model_input.reshape(
            batch_size, (height // ps) * (width // ps), num_channels_latents * (ps * ps)
        )

        # img_shapes passed to the model
        img_h2, img_w2 = height // ps, width // ps
        img_shapes = [[(1, img_h2, img_w2)]] * batch_size

        enc_hs = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        prompt_embeds_mask = text_embeddings.attention_mask.to(
            self.device_torch, dtype=torch.int64
        )

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(
                self.device_torch, self.torch_dtype
            ).detach(),
            timestep=(timestep / 1000).detach(),
            guidance=None,
            encoder_hidden_states=enc_hs.detach(),
            encoder_hidden_states_mask=prompt_embeds_mask.detach(),
            img_shapes=img_shapes,
            return_dict=False,
            **kwargs,
        )[0]

        # unpack
        noise_pred = noise_pred.view(
            batch_size, height // ps, width // ps, num_channels_latents, ps, ps
        )
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
        # diffusers >=0.37 returns None when all tokens are valid (no padding)
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.int64
            )
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_embeds_mask
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        # comfy-format single-file save (diffusers keys ARE the comfy layout
        # for qwen image); prequantized layers keep their quantized storage
        transformer: QwenImageTransformer2DModel = unwrap_model(self.model)
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
        )

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "qwen_image"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    lora_keys_use_comfy_prefix = True

