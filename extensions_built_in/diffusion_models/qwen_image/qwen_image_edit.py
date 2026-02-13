import math
import torch
from .qwen_image import QwenImageModel
import os
from typing import TYPE_CHECKING, List, Optional
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import get_accelerator, unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.util.device import safe_module_to_device
import torch.nn.functional as F

from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    AutoencoderKLQwenImage,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from tqdm import tqdm

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

try:
    from diffusers import QwenImageEditPipeline
except ImportError:
    raise ImportError(
        "QwenImageEditPipeline not found. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


class QwenImageEditModel(QwenImageModel):
    arch = "qwen_image_edit"
    _qwen_image_keep_visual = True
    _qwen_pipeline = QwenImageEditPipeline

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

        # set true for models that encode control image into text embeddings
        self.encode_control_in_text_embeddings = True

    def load_model(self):
        super().load_model()

    def get_generation_pipeline(self):
        scheduler = QwenImageModel.get_train_scheduler()

        pipeline: QwenImageEditPipeline = QwenImageEditPipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            processor=self.processor,
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: QwenImageEditPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model_config.low_vram:
            safe_module_to_device(self.model, self.device_torch, self.torch_dtype)
        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        control_img = None
        if gen_config.ctrl_img is not None:
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

        img = pipeline(
            image=control_img,
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
        return img

    def condition_noisy_latents(
        self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"
    ):
        with torch.no_grad():
            control_tensor = batch.control_tensor
            if control_tensor is not None:
                self.vae.to(self.device_torch)
                # we are not packed here, so we just need to pass them so we can pack them later
                control_tensor = control_tensor * 2 - 1
                control_tensor = control_tensor.to(
                    self.vae_device_torch, dtype=self.torch_dtype
                )

                # if it is not the size of batch.tensor, (bs,ch,h,w) then we need to resize it
                if batch.tensor is not None:
                    target_h, target_w = batch.tensor.shape[2], batch.tensor.shape[3]
                else:
                    # When caching latents, batch.tensor is None. We get the size from the file_items instead.
                    target_h = batch.file_items[0].crop_height
                    target_w = batch.file_items[0].crop_width

                if (
                    control_tensor.shape[2] != target_h
                    or control_tensor.shape[3] != target_w
                ):
                    control_tensor = F.interpolate(
                        control_tensor, size=(target_h, target_w), mode="bilinear"
                    )

                control_latent = self.encode_images(control_tensor).to(
                    latents.device, latents.dtype
                )
                latents = torch.cat((latents, control_latent), dim=1)

        return latents.detach()

    def get_prompt_embeds(self, prompt: str, control_images=None) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        if control_images is not None:
            # control images are 0 - 1 scale, shape (bs, ch, height, width)
            # images are always run through at 1MP, based on diffusers inference code.
            target_area = 1024 * 1024
            ratio = control_images.shape[2] / control_images.shape[3]
            width = math.sqrt(target_area * ratio)
            height = width / ratio

            width = round(width / 32) * 32
            height = round(height / 32) * 32

            control_images = F.interpolate(
                control_images, size=(height, width), mode="bilinear"
            )

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt,
            image=control_images,
            device=self.device_torch,
            num_images_per_prompt=1,
        )
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_embeds_mask
        return pe

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        # control is stacked on channels, move it to the batch dimension for packing
        latent_model_input, control = torch.chunk(latent_model_input, 2, 1)

        batch_size, num_channels_latents, height, width = latent_model_input.shape
        (
            control_batch_size,
            control_num_channels_latents,
            control_height,
            control_width,
        ) = control.shape

        # pack image tokens
        latent_model_input = latent_model_input.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
        latent_model_input = latent_model_input.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        # pack control
        control = control.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        control = control.permute(0, 2, 4, 1, 3, 5)
        control = control.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        img_h2, img_w2 = height // 2, width // 2
        control_img_h2, control_img_w2 = control_height // 2, control_width // 2
        
        img_shapes = [[(1, img_h2, img_w2), (1, control_img_h2, control_img_w2)]] * batch_size

        latents = latent_model_input
        latent_model_input = torch.cat([latent_model_input, control], dim=1)
        batch_size = latent_model_input.shape[0]

        prompt_embeds_mask = text_embeddings.attention_mask.to(
            self.device_torch, dtype=torch.int64
        )
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        enc_hs = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        prompt_embeds_mask = text_embeddings.attention_mask.to(self.device_torch, dtype=torch.int64)

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

        noise_pred = noise_pred[:, : latents.size(1)]

        # unpack
        noise_pred = noise_pred.view(
            batch_size, height // 2, width // 2, num_channels_latents, 2, 2
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred
