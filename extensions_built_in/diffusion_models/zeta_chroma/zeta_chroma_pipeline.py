from diffusers.pipelines.z_image.pipeline_z_image import (
    ZImagePipeline,
    calculate_shift,
    retrieve_timesteps,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
import torch
from diffusers.utils import logging, replace_example_docstring
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput
from extensions_built_in.diffusion_models.zeta_chroma.zeta_chroma_transformer import (
    get_schedule,
    get_low_step_schedule,
    prepare_latent_image_ids,
    make_text_position_ids,
    vae_unflatten,
)


class ZetaChromaPipeline(ZImagePipeline):
    need_something_here = True
    patch_size = 32
    max_sequence_length = 512

    @torch.no_grad()
    def _encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of prompts with the Qwen3 chat template."""
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            formatted.append(text)

        inputs = self.tokenizer(
            formatted,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.text_encoder.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )

        # Second-to-last hidden state (same as training)
        embeddings = outputs.hidden_states[-2]
        mask = inputs.attention_mask.bool()
        return embeddings, mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        prompt_embeds_mask: Optional[torch.BoolTensor] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds_mask: Optional[torch.BoolTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        low_step_schedule: bool = False,
    ):
        device = self._execution_device

        batch_size = len(prompt_embeds)
        device = self._execution_device
        patch_size = self.patch_size
        in_channels = patch_size * patch_size * 3

        h_patches = height // patch_size
        w_patches = width // patch_size
        num_patches = h_patches * w_patches

        pos_embeds, pos_mask = prompt_embeds, prompt_embeds_mask
        neg_embeds, neg_mask = negative_prompt_embeds, negative_prompt_embeds_mask

        # --- Build position IDs ---
        pos_lengths = pos_mask.sum(1)
        neg_lengths = neg_mask.sum(1)
        offset = torch.maximum(pos_lengths, neg_lengths)

        image_pos_ids = prepare_latent_image_ids(
            offset, h_patches, w_patches, patch_size=1
        ).to(device)
        pos_text_ids = make_text_position_ids(pos_lengths, max_sequence_length).to(
            device
        )
        neg_text_ids = make_text_position_ids(neg_lengths, max_sequence_length).to(
            device
        )

        # --- Initial noise ---
        noise = randn_tensor(
            (batch_size, num_patches, in_channels),
            generator=generator,
            device=device,
            dtype=self.transformer.dtype,
        )

        # --- Timestep schedule ---
        if low_step_schedule:
            timesteps = get_low_step_schedule(num_inference_steps)
        else:
            timesteps = get_schedule(num_inference_steps, num_patches)

        # --- Denoising loop (CFG) ---
        img = noise
        img_mask = torch.ones(
            (batch_size, num_patches), device=device, dtype=torch.bool
        )
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):

                t_vec = torch.full(
                    (batch_size,), t_curr, dtype=self.dtype, device=device
                )

                pred = self.transformer(
                    img=img,
                    img_ids=image_pos_ids,
                    img_mask=img_mask,
                    txt=pos_embeds,
                    txt_ids=pos_text_ids,
                    txt_mask=pos_mask,
                    timesteps=t_vec,
                )

                if guidance_scale > 1.0:
                    pred_neg = self.transformer(
                        img=img,
                        img_ids=image_pos_ids,
                        img_mask=img_mask,
                        txt=neg_embeds,
                        txt_ids=neg_text_ids,
                        txt_mask=neg_mask,
                        timesteps=t_vec,
                    )
                    pred = pred_neg + guidance_scale * (pred - pred_neg)

                img = img + (t_prev - t_curr) * pred

                progress_bar.update()

        if output_type == "latent":
            image = img

        else:
            # --- Unpatchify: [B, num_patches, C*P*P] -> [B, 3, H, W] ---
            pixel_shape = (batch_size, 3, height, width)
            image = vae_unflatten(img.float(), pixel_shape, patch_size=patch_size)
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ZImagePipelineOutput(images=image)
