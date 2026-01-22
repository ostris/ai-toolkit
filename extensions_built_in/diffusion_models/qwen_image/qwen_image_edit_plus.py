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
import torch.nn.functional as F

from diffusers import (
    QwenImageTransformer2DModel,
    AutoencoderKLQwenImage,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from tqdm import tqdm


if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

try:
    from .qwen_image_pipelines import QwenImageEditPlusCustomPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        VAE_IMAGE_SIZE,
    )
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing 'pip uninstall diffusers' and then 'pip install -r requirements.txt'"
    )


class QwenImageEditPlusModel(QwenImageModel):
    arch = "qwen_image_edit_plus"
    _qwen_image_keep_visual = True
    _qwen_pipeline = QwenImageEditPlusCustomPipeline

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
        # control images will come in as a list for encoding some things if true
        self.has_multiple_control_images = True
        # do not resize control images
        self.use_raw_control_images = True

    def load_model(self):
        super().load_model()

    def get_generation_pipeline(self):
        scheduler = QwenImageModel.get_train_scheduler()

        pipeline: QwenImageEditPlusCustomPipeline = QwenImageEditPlusCustomPipeline(
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
        pipeline: QwenImageEditPlusCustomPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        control_img_list = []
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        elif gen_config.ctrl_img_1 is not None:
            control_img = Image.open(gen_config.ctrl_img_1)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)

        if gen_config.ctrl_img_2 is not None:
            control_img = Image.open(gen_config.ctrl_img_2)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_3 is not None:
            control_img = Image.open(gen_config.ctrl_img_3)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)

        # flush for low vram if we are doing that
        # flush_between_steps = self.model_config.low_vram
        flush_between_steps = False

        # Fix a bug in diffusers/torch
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if flush_between_steps:
                flush()
            latents = callback_kwargs["latents"]

            return {"latents": latents}

        img = pipeline(
            image=control_img_list,
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
            do_cfg_norm=gen_config.do_cfg_norm,
            **extra,
        ).images[0]
        return img

    def condition_noisy_latents(
        self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"
    ):
        # we get the control image from the batch
        return latents.detach()

    def get_prompt_embeds(self, prompt: str, control_images=None) -> PromptEmbeds:
        # todo handle not caching text encoder
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
            
        if control_images is None:
            raise ValueError("Missing control images for QwenImageEditPlusModel")
        
        if not isinstance(control_images, List):
            control_images = [control_images]

        if control_images is not None and len(control_images) > 0:
            for i in range(len(control_images)):
                # control images are 0 - 1 scale, shape (bs, ch, height, width)
                ratio = control_images[i].shape[2] / control_images[i].shape[3]
                width = math.sqrt(CONDITION_IMAGE_SIZE * ratio)
                height = width / ratio

                width = round(width / 32) * 32
                height = round(height / 32) * 32

                control_images[i] = F.interpolate(
                    control_images[i], size=(height, width), mode="bilinear"
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
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        with torch.no_grad():
            batch_size, num_channels_latents, height, width = latent_model_input.shape
            if self.vae.device != self.device_torch:
                self.vae.to(self.device_torch)
            
            control_image_res = VAE_IMAGE_SIZE
            if self.model_config.model_kwargs.get("match_target_res", False):
                # use the current target size to set the control image res
                control_image_res = height * self.pipeline.vae_scale_factor * width * self.pipeline.vae_scale_factor

            # pack image tokens
            latent_model_input = latent_model_input.view(
                batch_size, num_channels_latents, height // 2, 2, width // 2, 2
            )
            latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
            latent_model_input = latent_model_input.reshape(
                batch_size, (height // 2) * (width // 2), num_channels_latents * 4
            )

            raw_packed_latents = latent_model_input

            img_h2, img_w2 = height // 2, width // 2

            # build distinct instances per batch item, per mamad8
            img_shapes = [[(1, img_h2, img_w2)] for _ in range(batch_size)]

            # pack controls
            if batch is None:
                raise ValueError("Batch is required for QwenImageEditPlusModel")

            # split the latents into batch items so we can concat the controls
            packed_latents_list = torch.chunk(latent_model_input, batch_size, dim=0)
            packed_latents_with_controls_list = []
            
            batch_control_tensor_list = batch.control_tensor_list
            if batch_control_tensor_list is None and batch.control_tensor is not None:
                batch_control_tensor_list = []
                for b in range(batch_size):
                    batch_control_tensor_list.append(batch.control_tensor[b : b + 1])

            if batch_control_tensor_list is not None:
                b = 0
                for control_tensor_list in batch_control_tensor_list:
                    # control tensor list is a list of tensors for this batch item
                    controls = []
                    # pack control
                    for control_img in control_tensor_list:
                        # control images are 0 - 1 scale, shape (1, ch, height, width)
                        control_img = control_img.to(
                            self.device_torch, dtype=self.torch_dtype
                        )
                        # if it is only 3 dim, add batch dim
                        if len(control_img.shape) == 3:
                            control_img = control_img.unsqueeze(0)
                        ratio = control_img.shape[2] / control_img.shape[3]
                        c_width = math.sqrt(control_image_res * ratio)
                        c_height = c_width / ratio

                        c_width = round(c_width / 32) * 32
                        c_height = round(c_height / 32) * 32

                        control_img = F.interpolate(
                            control_img, size=(c_height, c_width), mode="bilinear"
                        )

                        # scale to -1 to 1
                        control_img = control_img * 2 - 1

                        control_latent = self.encode_images(
                            control_img,
                            device=self.device_torch,
                            dtype=self.torch_dtype,
                        )

                        clb, cl_num_channels_latents, cl_height, cl_width = (
                            control_latent.shape
                        )

                        control = control_latent.view(
                            1,
                            cl_num_channels_latents,
                            cl_height // 2,
                            2,
                            cl_width // 2,
                            2,
                        )
                        control = control.permute(0, 2, 4, 1, 3, 5)
                        control = control.reshape(
                            1,
                            (cl_height // 2) * (cl_width // 2),
                            num_channels_latents * 4,
                        )

                        img_shapes[b].append((1, cl_height // 2, cl_width // 2))
                        controls.append(control)

                    # stack controls on dim 1
                    control = torch.cat(controls, dim=1).to(
                        packed_latents_list[b].device,
                        dtype=packed_latents_list[b].dtype,
                    )
                    # concat with latents
                    packed_latents_with_control = torch.cat(
                        [packed_latents_list[b], control], dim=1
                    )

                    packed_latents_with_controls_list.append(
                        packed_latents_with_control
                    )

                    b += 1

                latent_model_input = torch.cat(packed_latents_with_controls_list, dim=0)

            prompt_embeds_mask = text_embeddings.attention_mask.to(
                self.device_torch, dtype=torch.int64
            )
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
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
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
            **kwargs,
        )[0]

        noise_pred = noise_pred[:, : raw_packed_latents.size(1)]

        # unpack
        noise_pred = noise_pred.view(
            batch_size, height // 2, width // 2, num_channels_latents, 2, 2
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred
