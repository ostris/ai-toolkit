"""PRXPixelT2IModel -- the Photoroom pixel-space PRX-7B text-to-image model
(https://huggingface.co/Photoroom/prxpixel-t2i) wired into ai-toolkit.

This is implemented from scratch (the diffusers support is an unmerged PR,
https://github.com/huggingface/diffusers/pull/13928) so ai-toolkit does not
depend on it: the transformer architecture is vendored in ``src/transformer_prx.py``
and a minimal sampler lives in ``src/pipeline.py``.

What makes PRXPixel unusual (and what each override below is doing about it):

  - **Pixel space, no VAE.** The transformer denoises raw RGB directly
    (``in_channels=3``, ``patch_size=16``). We use a ``FakeVAE`` (identity,
    scaling_factor=1) so BaseModel's encode_images/decode_latents become no-ops
    and the "latents" everywhere in the toolkit are just the image in [-1, 1].
    Same trick as ``../chroma/chroma_radiance_model.py`` and
    ``extensions/z_image_pixel``.

  - **x-prediction.** The model predicts the CLEAN image x0, not the
    flow-matching velocity. ai-toolkit's MSE compares ``get_noise_prediction``
    against ``get_loss_target``; we set BOTH to the x0 space (prediction = the
    model's x0 output, target = the clean latents), which is PRXPixel's native
    training objective ("Back to Basics: Let Denoising Generative Models
    Denoise", https://arxiv.org/abs/2511.13720). The x0->velocity conversion
    only happens at sampling time, inside the pipeline.

  - **noise_scale = 2.0.** PRXPixel trains with a non-unit initial-noise std,
    so the noise mixed into the latents (training) and the starting noise
    (sampling) are ``randn * noise_scale``. We override
    ``get_latent_noise_from_latents`` for the training side; the pipeline
    handles the sampling side.

  - **Qwen3-VL text tower.** Prompts are encoded by ``Qwen3VLTextModel``
    (hidden size 2048 -> the transformer's ``context_in_dim``). We keep the
    per-token ``last_hidden_state`` plus an attention mask.

See ../example_model/README.md for the generic lifecycle/registration guide.
"""

import os
from typing import List, Optional

import torch
import yaml

from transformers import AutoTokenizer, Qwen3VLTextModel
from optimum.quanto import freeze

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.models.FakeVAE import FakeVAE
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.train_tools import apply_noise_offset
from toolkit.util.quantize import quantize, get_qtype, quantize_model

from .src.transformer_prx import PRXTransformer2DModel
from .src.pipeline import PRXPixelPipeline


# Flow-matching scheduler config, matching the released model's
# scheduler/scheduler_config.json (shift 3.0, 1000 train timesteps).
scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}

# Number of text tokens PRXPixel was trained with (the Qwen tokenizer's own
# model_max_length is far larger). Matches PRX_PIXEL_DEFAULT_MAX_TOKENS.
PROMPT_MAX_TOKENS = 256
# Initial-noise std PRXPixel trains/samples with.
NOISE_SCALE = 2.0


class PRXPixelT2IModel(BaseModel):
    # ``model.arch: "prx_pixel"`` in the training config YAML selects this class.
    arch = "prx_pixel"

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
        # Matched against type(module).__name__ to place LoRA layers.
        self.target_lora_modules = ["PRXTransformer2DModel"]

        # used by our overrides below
        self.patch_size = 16
        self.vae_scale_factor = 1  # pixel space: no VAE downsampling
        self.max_text_length = PROMPT_MAX_TOKENS
        self.noise_scale = NOISE_SCALE

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # pixels must be divisible by patch_size (no VAE downsample): 1 * 16 = 16
        return self.vae_scale_factor * self.patch_size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading PRXPixel model")
        # Expected diffusers-style layout under name_or_path:
        #   transformer/ (PRXTransformer2DModel)  text_encoder/ (Qwen3VLTextModel)
        #   tokenizer/                            scheduler/
        model_path = self.model_config.name_or_path

        # --- transformer (vendored PRXTransformer2DModel) ---
        self.print_and_status_update("Loading transformer")
        # from_pretrained reads config.json (bottleneck_size, resolution_embeds,
        # in_channels=3, ...) and the safetensors in one shot.
        transformer = PRXTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=dtype
        )
        transformer.to(dtype=dtype)
        flush()

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing transformer")
            quantize_model(self, transformer)
            flush()

        if self.model_config.low_vram:
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        # --- text encoder + tokenizer (Qwen3-VL text tower) ---
        self.print_and_status_update("Loading text encoder")
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = Qwen3VLTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        text_encoder.to(self.te_device_torch)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing text encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        # --- "VAE": identity, since PRXPixel is pixel space ---
        self.print_and_status_update("Preparing pixel-space VAE (identity)")
        vae = FakeVAE(scaling_factor=1.0)
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)

        # --- scheduler + store everything ---
        self.noise_scheduler = PRXPixelT2IModel.get_train_scheduler()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = transformer  # aliased as self.transformer / self.unet
        self.pipeline = PRXPixelPipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Sampling (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return PRXPixelPipeline(self)

    def generate_single_image(
        self,
        pipeline: PRXPixelPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # snap requested size to the model's divisibility (16px)
        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        img = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
        )[0]
        return img

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0..1000 scale, 1000 = pure noise
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        """Forward pass. Returns the model's predicted CLEAN image x0.

        PRXPixel is an x-prediction model, so the raw transformer output is the
        prediction we compare against ``get_loss_target`` (the clean latents).
        No velocity conversion happens here -- that is a sampling-time concern.
        """
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # toolkit timestep (0..1000) -> PRX flow time in [0, 1].
        t01 = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0

        feats = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        mask = getattr(text_embeddings, "attention_mask", None)
        if mask is not None:
            mask = mask.to(self.device_torch)

        x0_pred = self.model(
            hidden_states=latent_model_input.to(self.device_torch, self.torch_dtype),
            timestep=t01,
            encoder_hidden_states=feats,
            attention_mask=mask,
            return_dict=False,
        )[0]
        return x0_pred

    def get_prompt_embeds(self, prompt) -> PromptEmbeds:
        """Encode prompt text with the Qwen3-VL text tower.

        Returns a PromptEmbeds whose ``text_embeds`` is (B, L, 2048) and whose
        ``attention_mask`` is (B, L). Prompts are padded to a fixed length
        (PROMPT_MAX_TOKENS) so cached embeds are concatenatable for CFG.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        attention_mask = text_inputs.attention_mask.to(self.text_encoder.device)

        with torch.no_grad():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        embeds = output["last_hidden_state"].to(self.torch_dtype)

        pe = PromptEmbeds(embeds)
        # keep the mask as bool/long; it must not be dtype-cast with the embeds
        pe.attention_mask = attention_mask.bool()
        return pe

    def get_loss_target(self, *args, **kwargs):
        """x-prediction target: the clean image (x0).

        PRXPixel predicts x0 directly, so the MSE target is simply the clean
        latents (the pixel image in [-1, 1]), not the flow velocity.
        """
        batch = kwargs.get("batch")
        return batch.latents.detach()

    def get_latent_noise_from_latents(self, latents: torch.Tensor, noise_offset=0.0):
        """Noise for the forward flow, scaled by the model's noise_scale.

        PRXPixel trains with a non-unit initial-noise std: the noise mixed into
        the latents is ``randn * noise_scale``. The scheduler's add_noise then
        forms ``x_t = (1 - t) * clean + t * noise``.
        """
        noise = torch.randn_like(latents) * self.noise_scale
        if noise_offset is not None and noise_offset != 0.0:
            noise = apply_noise_offset(noise, noise_offset)
        return noise

    def condition_noisy_latents(self, latents: torch.Tensor, batch):
        # plain text-to-image: nothing to inject
        return latents

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        """Full fine-tune save: write the transformer back in diffusers layout."""
        transformer: PRXTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        with open(os.path.join(output_path, "aitk_meta.yaml"), "w") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        return "prx_pixel"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        return {
            k.replace("transformer.", "diffusion_model."): v
            for k, v in state_dict.items()
        }

    def convert_lora_weights_before_load(self, state_dict):
        return {
            k.replace("diffusion_model.", "transformer."): v
            for k, v in state_dict.items()
        }
