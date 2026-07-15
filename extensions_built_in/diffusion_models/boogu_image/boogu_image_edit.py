"""Boogu-Image edit (TI2I) integration for ai-toolkit.

The edit model is the same Lumina2-style transformer + Qwen3-VL encoder as the
base T2I model, with reference-image conditioning. A reference image feeds the
model in TWO places:

  1. Into the Qwen3-VL instruction encoder as image content alongside the edit
     instruction (so the *text embeddings* already encode the reference image).
     This is why ``encode_control_in_text_embeddings = True``.
  2. Into the transformer as reference-image VAE latents
     (``ref_image_hidden_states``), which the ref-image refiner + double-stream
     blocks attend to.

Everything else (transformer, VAE, scheduler, time/velocity convention, saving)
is inherited from ``BooguImageModel`` -- this file only overrides the pieces
that change for TI2I.
"""

import math
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import GenerateImageConfig, ModelConfig

from .boogu_image import BooguImageModel
from .src.pipeline import (
    BooguImagePipeline,
    pad_instruction_features,
    run_boogu_transformer,
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


# Edit release (clean bf16 safetensors); same layout as the base repo.
BOOGU_EDIT_PATH = "Boogu/Boogu-Image-0.1-Edit"

# System prompt the edit model was trained with (SYSTEM_PROMPT_4_TI2I upstream).
SYSTEM_PROMPT_TI2I = (
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate."
)


class BooguImageEditModel(BooguImageModel):
    arch = "boogu_image_edit"
    default_repo = BOOGU_EDIT_PATH

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
        # The reference image is encoded into the Qwen3-VL instruction features,
        # so get_prompt_embeds receives the control image(s).
        self.encode_control_in_text_embeddings = True
        # Boogu supports up to 5 reference images -> they arrive as a list.
        self.has_multiple_control_images = True
        # Reference images keep their own aspect/size (not resized to the target).
        self.use_raw_control_images = True

    @property
    def text_embedding_space_version(self):
        # Distinct from the base T2I cache: the edit features fold in the ref image.
        return self.arch + "_v1"

    # ------------------------------------------------------------------
    # Reference-image helpers
    # ------------------------------------------------------------------
    def _vlm_resize_hw(self, h, w, max_pixels, max_side, factor=16):
        """Boogu's VLM image downscale (BooguImageProcessor.get_new_height_width).

        Scale down (never up) to fit BOTH ``max_pixels`` (area) and
        ``max_side_length``, then round each dim down to a multiple of ``factor``
        (the image processor's ``vae_scale_factor`` = 16 for this model). The Qwen
        processor's own smart_resize runs afterwards, exactly as upstream.
        """
        longest = h if h > w else w
        ratio_side = max_side / longest
        ratio_pixels = (max_pixels / (h * w)) ** 0.5
        ratio = min(ratio_pixels, ratio_side, 1.0)
        nh = max(factor, int(h * ratio) // factor * factor)
        nw = max(factor, int(w * ratio) // factor * factor)
        return nh, nw

    def _ref_target_pixels(self, target_pixels: Optional[int]) -> int:
        """Decide the pixel budget each reference image is resized to fit within.

        - default: ``control_image_max_pixels`` model_kwarg (1 MP) -- a hard cap so
          raw, full-size control images don't blow up the token count / VRAM.
        - ``match_target_res`` model_kwarg: use the target generation area instead,
          matching Boogu's recommendation of ``max_input_image_pixels ~= H*W``.
        """
        max_pixels = int(
            self.model_config.model_kwargs.get("control_image_max_pixels", 1024 * 1024)
        )
        if (
            self.model_config.model_kwargs.get("match_target_res", False)
            and target_pixels
        ):
            return int(target_pixels)
        return max_pixels

    def _encode_ref_latents(
        self, control_tensors, target_pixels: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Encode ``[0, 1]`` reference image tensors to VAE latents.

        Returns a list of ``(16, h, w)`` latents (one per reference image). Each
        control image is resized so its area fits within the pixel budget (see
        ``_ref_target_pixels``) -- preserving aspect ratio -- then snapped so the
        latent grid is divisible by the patch size. ``control_tensors`` is a list
        of ``(C, H, W)`` or ``(1, C, H, W)`` tensors in ``[0, 1]``.
        """
        sc = self.get_bucket_divisibility()  # 16: VAE(8) * patch(2)
        budget = self._ref_target_pixels(target_pixels)
        match = self.model_config.model_kwargs.get("match_target_res", False)

        latents = []
        for img in control_tensors:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(self.device_torch, dtype=self.torch_dtype)

            h, w = img.shape[2], img.shape[3]
            # match_target_res: scale area *to* the budget; otherwise only scale
            # *down* when the image is larger than the budget.
            area = h * w
            if match or area > budget:
                ratio = h / w
                new_h = math.sqrt(budget * ratio)
                new_w = new_h / ratio
            else:
                new_h, new_w = float(h), float(w)

            # snap to a multiple of the bucket divisibility so the VAE latent grid
            # is patchifiable (the transformer rearranges 2x2 latent patches).
            new_h = max(sc, int(round(new_h / sc)) * sc)
            new_w = max(sc, int(round(new_w / sc)) * sc)
            if (new_h, new_w) != (h, w):
                img = F.interpolate(img, size=(new_h, new_w), mode="bilinear")

            # encode_images expects [-1, 1]; control tensors arrive in [0, 1].
            latent = self.encode_images(
                img * 2 - 1, device=self.device_torch, dtype=self.torch_dtype
            )
            latents.append(latent[0])  # drop batch dim -> (16, h, w)
        return latents

    def _batch_ref_latents_from_batch(
        self,
        batch: "DataLoaderBatchDTO",
        batch_size: int,
        target_pixels: Optional[int] = None,
    ) -> Optional[List[List[torch.Tensor]]]:
        """Build the transformer's ``ref_image_hidden_states`` from a train batch."""
        control_list = batch.control_tensor_list
        if control_list is None and batch.control_tensor is not None:
            control_list = [batch.control_tensor[b : b + 1] for b in range(batch_size)]
        if control_list is None:
            return None
        if len(control_list) != batch_size:
            raise ValueError("Control tensor list length does not match batch size")
        return [
            self._encode_ref_latents(controls, target_pixels=target_pixels)
            for controls in control_list
        ]

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]

        if control_images is None:
            raise ValueError("BooguImageEditModel requires control (reference) images")

        # Normalize to List[List[Tensor]] (per-prompt list of reference images), the
        # same convention qwen_image_edit_plus uses.
        if not isinstance(control_images, list):
            control_images = [control_images]
        if not isinstance(control_images[0], list):
            control_images = [control_images]
        if len(prompt) != len(control_images):
            raise ValueError(
                "Number of prompts must match number of control image sets"
            )

        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        device = self.text_encoder.device

        features_list = []
        for p, ctrl in zip(prompt, control_images):
            # Keep reference images as tensors the whole way (no GPU->CPU->PIL
            # round-trip). Match Boogu's VLM preprocessing: downscale each control
            # image to fit max_pixels (384^2) AND max_side_length (768) -- the MLLM
            # only needs a coarse understanding of the reference (high-res detail
            # flows through the VAE ref latents), and this keeps the instruction
            # sequence well under the transformer rope axes_lens (~144 tokens/ref).
            max_pixels = int(
                self.model_config.model_kwargs.get("vlm_max_pixels", 384 * 384)
            )
            max_side = int(
                self.model_config.model_kwargs.get("vlm_max_side_length", 768)
            )
            images = []
            for img in ctrl:
                if img.dim() == 4:
                    img = img[0]
                img = img.to(device)
                nh, nw = self._vlm_resize_hw(
                    img.shape[1], img.shape[2], max_pixels, max_side
                )
                if (nh, nw) != (img.shape[1], img.shape[2]):
                    img = (
                        F.interpolate(
                            img.unsqueeze(0),
                            size=(nh, nw),
                            mode="bicubic",
                            antialias=True,
                        )
                        .squeeze(0)
                        .clamp(0, 1)
                    )
                images.append(img)

            # Build just the text template with image placeholders (tokenize=False),
            # then let the processor expand the image tokens from the real grid size.
            user_content = [{"type": "image"} for _ in images]
            user_content.append({"type": "text", "text": p})
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_TI2I}],
                },
                {"role": "user", "content": user_content},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # do_rescale=False: control tensors are already [0, 1] (the image
            # normalizer maps them to [-1, 1]). No size override -- the images are
            # already at Boogu's target size, the processor just snaps to its grid.
            inputs = self.tokenizer(
                text=[text],
                images=images,
                return_tensors="pt",
                do_rescale=False,
            )
            model_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    # cast image pixels to the encoder dtype; leave ids/masks as ints
                    if v.is_floating_point():
                        v = v.to(self.torch_dtype)
                model_inputs[k] = v

            with torch.no_grad():
                output = self.text_encoder(**model_inputs)
            features_list.append(output.last_hidden_state[0].to(self.torch_dtype))

        return AdvancedPromptEmbeds(text_embeds=features_list)

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 16, h, w)
        timestep: torch.Tensor,  # 0..1000 scale (1000 = pure noise)
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        with torch.no_grad():
            # target pixel area from the noise latents (h, w are VAE-downsampled)
            _, _, lh, lw = latent_model_input.shape
            target_pixels = (lh * self.vae_scale_factor) * (lw * self.vae_scale_factor)
            ref_latents = (
                self._batch_ref_latents_from_batch(
                    batch, latent_model_input.shape[0], target_pixels=target_pixels
                )
                if batch is not None
                else None
            )

        # toolkit timestep (0..1000, 1000=noise) -> Boogu native time (0=noise, 1=clean)
        t01 = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
        if t01.dim() == 0:
            t01 = t01.unsqueeze(0)
        if t01.shape[0] != latent_model_input.shape[0]:
            t01 = t01.expand(latent_model_input.shape[0])
        boogu_t = 1.0 - t01

        instr_feats, instr_mask = pad_instruction_features(
            text_embeddings.text_embeds, self.device_torch, self.torch_dtype
        )

        # Model predicts clean - noise; negate to return the toolkit velocity.
        raw_velocity = run_boogu_transformer(
            self.transformer,
            latent_model_input.to(self.device_torch, self.torch_dtype),
            boogu_t,
            instr_feats,
            instr_mask,
            self.get_freqs_cis(),
            ref_image_hidden_states=ref_latents,
        )
        return -raw_velocity

    # ------------------------------------------------------------------
    # Sampling previews
    # ------------------------------------------------------------------
    def generate_single_image(
        self,
        pipeline: BooguImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        # Load the reference image(s) for the transformer ref latents. The MLLM
        # side already saw them (baked into conditional/unconditional embeds).
        ctrl_paths = [
            p
            for p in (
                gen_config.ctrl_img,
                gen_config.ctrl_img_1,
                gen_config.ctrl_img_2,
                gen_config.ctrl_img_3,
            )
            if p is not None
        ]
        ref_latents = None
        if ctrl_paths:
            ctrl_tensors = [
                to_tensor(Image.open(path).convert("RGB")) for path in ctrl_paths
            ]
            target_pixels = gen_config.width * gen_config.height
            # one batch item (preview batch size is 1) -> List[List[(16, h, w)]]
            ref_latents = [
                self._encode_ref_latents(ctrl_tensors, target_pixels=target_pixels)
            ]

        img = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            ref_latents=ref_latents,
        )[0]
        return img

    def get_base_model_version(self):
        return "boogu_image_edit.0.1"
