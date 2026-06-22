from .hidream_model import HidreamModel
from .src.pipelines.hidream_image.pipeline_hidream_image_editing import (
    HiDreamImageEditingPipeline,
)
from .src.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from toolkit.accelerator import unwrap_model
import torch
from toolkit.prompt_utils import PromptEmbeds
from toolkit.config_modules import GenerateImageConfig
from diffusers.models import HiDreamImageTransformer2DModel

import torch.nn.functional as F
from PIL import Image
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


class HidreamE1Model(HidreamModel):
    arch = "hidream_e1"
    hidream_transformer_class = HiDreamImageTransformer2DModel
    hidream_pipeline_class = HiDreamImageEditingPipeline

    def get_generation_pipeline(self):
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False
        )

        pipeline: HiDreamImageEditingPipeline = HiDreamImageEditingPipeline(
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
            aggressive_unloading=self.low_vram,
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: HiDreamImageEditingPipeline,
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
        img = pipeline(
            prompt_embeds_t5=conditional_embeds.text_embeds[0],
            prompt_embeds_llama3=conditional_embeds.text_embeds[1],
            pooled_prompt_embeds=conditional_embeds.pooled_embeds,
            negative_prompt_embeds_t5=unconditional_embeds.text_embeds[0],
            negative_prompt_embeds_llama3=unconditional_embeds.text_embeds[1],
            negative_pooled_prompt_embeds=unconditional_embeds.pooled_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            image=control_img,
            **extra,
        ).images[0]
        return img

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        self.text_encoder_to(self.device_torch, dtype=self.torch_dtype)
        max_sequence_length = 128
        (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            prompt_4=prompt,
            device=self.device_torch,
            dtype=self.torch_dtype,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = [prompt_embeds_t5, prompt_embeds_llama3]
        pe = PromptEmbeds([prompt_embeds, pooled_prompt_embeds])
        return pe

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

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        with torch.no_grad():
            # make sure config is set
            self.model.config.force_inference_output = True
            has_control = False
            lat_size = latent_model_input.shape[-1]
            if latent_model_input.shape[1] == 32:
                # chunk it and stack it on batch dimension
                # dont update batch size for img_its
                lat, control = torch.chunk(latent_model_input, 2, dim=1)
                latent_model_input = torch.cat([lat, control], dim=-1)
                has_control = True

        dtype = self.model.dtype
        device = self.device_torch

        text_embeds = text_embeddings.text_embeds
        # run the to for the list
        text_embeds = [te.to(device, dtype=dtype) for te in text_embeds]

        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timesteps=timestep,
            encoder_hidden_states_t5=text_embeds[0],
            encoder_hidden_states_llama3=text_embeds[1],
            pooled_embeds=text_embeddings.pooled_embeds.to(device, dtype=dtype),
            return_dict=False,
        )[0]

        if has_control:
            noise_pred = -1.0 * noise_pred[..., :lat_size]
        else:
            noise_pred = -1.0 * noise_pred

        return noise_pred
