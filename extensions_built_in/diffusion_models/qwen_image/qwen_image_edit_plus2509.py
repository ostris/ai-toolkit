import math
import torch
from .qwen_image import QwenImageModel
from typing import TYPE_CHECKING
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.accelerator import unwrap_model
import torch.nn.functional as F


if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    raise ImportError(
        "QwenImageEditPlusPipeline not found. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


class QwenImageEditPlus2509Model(QwenImageModel):
    arch = "qwen_image_edit_plus2509"
    _qwen_image_keep_visual = True
    _qwen_pipeline = QwenImageEditPlusPipeline

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

        pipeline: QwenImageEditPlusPipeline = QwenImageEditPlusPipeline(
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
        pipeline: QwenImageEditPlusPipeline,
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

        # Support multiple control images
        control_images = []
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            # resize to width and height
            if control_img.size != (gen_config.width, gen_config.height):
                control_img = control_img.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )
            control_images.append(control_img)

        # Support second control image
        if hasattr(gen_config, "ctrl_img2") and gen_config.ctrl_img2 is not None:
            control_img2 = Image.open(gen_config.ctrl_img2)
            control_img2 = control_img2.convert("RGB")
            # resize to width and height
            if control_img2.size != (gen_config.width, gen_config.height):
                control_img2 = control_img2.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )
            control_images.append(control_img2)

        # Support third control image if needed
        if hasattr(gen_config, "ctrl_img3") and gen_config.ctrl_img3 is not None:
            control_img3 = Image.open(gen_config.ctrl_img3)
            control_img3 = control_img3.convert("RGB")
            # resize to width and height
            if control_img3.size != (gen_config.width, gen_config.height):
                control_img3 = control_img3.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )
            control_images.append(control_img3)

        # Use the list of images or None if no control images
        control_input = control_images if control_images else None

        # flush for low vram if we are doing that
        flush_between_steps = self.model_config.low_vram

        # Fix a bug in diffusers/torch
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if flush_between_steps:
                flush()
            latents = callback_kwargs["latents"]

            return {"latents": latents}

        img = pipeline(
            image=control_input,
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
            # Support multiple control tensors
            control_tensors = []

            # Primary control tensor
            control_tensor = batch.control_tensor
            if control_tensor is not None:
                control_tensors.append(control_tensor)

            # Secondary control tensor
            if hasattr(batch, "control_tensor2") and batch.control_tensor2 is not None:
                control_tensors.append(batch.control_tensor2)

            # Third control tensor
            if hasattr(batch, "control_tensor3") and batch.control_tensor3 is not None:
                control_tensors.append(batch.control_tensor3)

            if not control_tensors:
                return latents

            self.vae.to(self.device_torch)

            # Process all control tensors
            control_latents = []
            for control_tensor in control_tensors:
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
                control_latents.append(control_latent)

            # Concatenate all control latents along the channel dimension
            if control_latents:
                all_control_latents = torch.cat(control_latents, dim=1)
                latents = torch.cat((latents, all_control_latents), dim=1)

        return latents.detach()

    def get_prompt_embeds(self, prompt: str, control_images=None) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        # Handle multiple control images
        processed_control_images = None
        if control_images is not None:
            # control images are 0 - 1 scale, shape (bs, ch, height, width)
            # For multiple images, we need to process each one
            if isinstance(control_images, list):
                processed_control_images = []
                for img in control_images:
                    # images are always run through at ~384x384 for conditioning (Edit Plus uses smaller res for VL)
                    target_area = 384 * 384
                    ratio = img.shape[2] / img.shape[3]
                    width = math.sqrt(target_area * ratio)
                    height = width / ratio

                    width = round(width / 32) * 32
                    height = round(height / 32) * 32

                    processed_img = F.interpolate(
                        img, size=(int(height), int(width)), mode="bilinear"
                    )
                    processed_control_images.append(processed_img)
            else:
                # Single image case (backward compatibility)
                target_area = (
                    384 * 384
                )  # Edit Plus uses smaller conditioning resolution
                ratio = control_images.shape[2] / control_images.shape[3]
                width = math.sqrt(target_area * ratio)
                height = width / ratio

                width = round(width / 32) * 32
                height = round(height / 32) * 32

                processed_control_images = F.interpolate(
                    control_images, size=(int(height), int(width)), mode="bilinear"
                )

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt,
            image=processed_control_images,
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
        # For Edit Plus, we need to handle multiple control images in the latent concatenation
        # The latents are packed as [main_latents, control1_latents, control2_latents, ...]

        # Split the latents - first part is main latents, rest are control latents
        main_channels = 16  # Qwen Image uses 16 channels for main latents
        main_latents = latent_model_input[:, :main_channels, :, :]

        # The rest are control latents - we need to split them by number of control images
        if latent_model_input.shape[1] > main_channels:
            control_latents = latent_model_input[:, main_channels:, :, :]
            # Assuming each control image contributes 16 channels
            num_control_images = control_latents.shape[1] // main_channels
            control_list = torch.chunk(control_latents, num_control_images, dim=1)
        else:
            control_list = []

        batch_size, num_channels_latents, height, width = main_latents.shape

        # pack main image tokens
        main_latents_packed = main_latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        main_latents_packed = main_latents_packed.permute(0, 2, 4, 1, 3, 5)
        main_latents_packed = main_latents_packed.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        # pack control image tokens
        control_packed_list = []
        img_shapes_per_batch = [(1, height // 2, width // 2)]  # Main image shape

        for control in control_list:
            control_height, control_width = control.shape[2], control.shape[3]
            control_packed = control.view(
                batch_size,
                num_channels_latents,
                control_height // 2,
                2,
                control_width // 2,
                2,
            )
            control_packed = control_packed.permute(0, 2, 4, 1, 3, 5)
            control_packed = control_packed.reshape(
                batch_size,
                (control_height // 2) * (control_width // 2),
                num_channels_latents * 4,
            )
            control_packed_list.append(control_packed)
            img_shapes_per_batch.append((1, control_height // 2, control_width // 2))

        # Concatenate all latents (main + controls) along sequence dimension
        if control_packed_list:
            latent_model_input = torch.cat(
                [main_latents_packed] + control_packed_list, dim=1
            )
        else:
            latent_model_input = main_latents_packed

        # img_shapes for transformer - repeated for each batch item
        img_shapes = [img_shapes_per_batch] * batch_size

        prompt_embeds_mask = text_embeddings.attention_mask.to(
            self.device_torch, dtype=torch.int64
        )
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        enc_hs = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        prompt_embeds_mask = text_embeddings.attention_mask.to(
            self.device_torch, dtype=torch.int64
        )

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

        # Extract only the main latent predictions (first part of the sequence)
        main_seq_len = (height // 2) * (width // 2)
        noise_pred = noise_pred[:, :main_seq_len]

        # unpack main latents only
        noise_pred = noise_pred.view(
            batch_size, height // 2, width // 2, num_channels_latents, 2, 2
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred
