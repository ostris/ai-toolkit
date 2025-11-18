from typing import List, Optional, Union

import numpy as np
import torch
import PIL.Image
from dataclasses import dataclass
from typing import List, Union

from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffusers.utils import BaseOutput

from .autoencoder import AutoEncoder
from .model import Flux2

from einops import rearrange

from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from .sampling import (
    get_schedule,
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    scatter_ids,
)


@dataclass
class Flux2ImagePipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation."""
OUTPUT_LAYERS = [10, 20, 30]
MAX_LENGTH = 512


class Flux2Pipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoEncoder,
        text_encoder: Mistral3ForConditionalGeneration,
        tokenizer: AutoProcessor,
        transformer: Flux2,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 16  # 8x plus 2x pixel shuffle
        self.num_channels_latents = 128
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = 64

    def format_input(
        self,
        txt: list[str],
    ) -> list[list[dict]]:
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in txt]

        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]

    def _get_mistral_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 512,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        if not isinstance(prompt, list):
            prompt = [prompt]

        # Format input messages
        messages_batch = self.format_input(txt=prompt)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages_batch,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
        except ValueError as e:
            print(
                f"Error processing input: {e}, your max length is probably too short, when you have images in the input."
            )
            raise e

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass through the model
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
        prompt_embeds = rearrange(out, "b c l d -> b l (c d)")

        # they don't return attention mask, so we create it here
        return prompt_embeds, None

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_mistral_prompt_embeds(
                prompt, device, max_sequence_length=max_sequence_length
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds, prompt_embeds_mask

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        control_img_list: Optional[List[PIL.Image.Image]] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode the prompt

        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        txt, txt_ids = batched_prc_txt(prompt_embeds)

        # 4. Prepare latent variables\
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        packed_latents, img_ids = batched_prc_img(latents)

        timesteps = get_schedule(num_inference_steps, packed_latents.shape[1])

        self._num_timesteps = len(timesteps)

        guidance_vec = torch.full(
            (packed_latents.shape[0],),
            guidance_scale,
            device=packed_latents.device,
            dtype=packed_latents.dtype,
        )

        if control_img_list is not None and len(control_img_list) > 0:
            img_cond_seq, img_cond_seq_ids = encode_image_refs(
                self.vae, control_img_list
            )
        else:
            img_cond_seq, img_cond_seq_ids = None, None

        # 6. Denoising loop
        i = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                if self.interrupt:
                    continue
                t_vec = torch.full(
                    (packed_latents.shape[0],),
                    t_curr,
                    dtype=packed_latents.dtype,
                    device=packed_latents.device,
                )

                self._current_timestep = t_curr
                img_input = packed_latents
                img_input_ids = img_ids

                if img_cond_seq is not None:
                    assert img_cond_seq_ids is not None, (
                        "You need to provide either both or neither of the sequence conditioning"
                    )
                    img_input = torch.cat((img_input, img_cond_seq), dim=1)
                    img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

                pred = self.transformer(
                    x=img_input,
                    x_ids=img_input_ids,
                    timesteps=t_vec,
                    ctx=txt,
                    ctx_ids=txt_ids,
                    guidance=guidance_vec,
                )

                if img_cond_seq is not None:
                    pred = pred[:, : packed_latents.shape[1]]

                packed_latents = packed_latents + (t_prev - t_curr) * pred
                i += 1
                progress_bar.update(1)

        self._current_timestep = None

        # 7. Post-processing
        latents = torch.cat(scatter_ids(packed_latents, img_ids)).squeeze(2)

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            image = self.vae.decode(latents).float()

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2ImagePipelineOutput(images=image)
