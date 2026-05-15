"""
HiDream-O1 model wrapper with multi-reference-image training support.

Key additions over the base version
------------------------------------
* ``get_prompt_embeds_with_refs`` – encode text + reference images together,
  returning an ``AdvancedPromptEmbeds`` whose ``text_embeds`` already contain
  the patchified reference pixels embedded alongside the text token ids.

* ``get_noise_prediction`` – extended to accept ``ref_patches`` in the batch DTO
  and inject them into the sequence before the generation tokens.  Works with
  batch_size > 1 by left-padding to the longest reference sequence.

* ``_build_ref_vinputs`` – helper that concatenates reference patches and target
  patches into the ``vinputs`` tensor that ``_forward_generation`` expects,
  while updating token_types / vinput_mask accordingly.

Reference-image sequence layout (per sample)
----------------------------------------------
    [text_tokens]
    [bor] [ref_patch_tokens_1] [eor]
    [bor] [ref_patch_tokens_2] [eor]
    ...
    [bot] [tms]
    ← prefix ends here (input_ids) ─────────────────────────────────────────
    [boi] [target_patch_tokens]
    ← vinputs window (generation) ──────────────────────────────────────────

All "image" tokens (ref + target) go through model.x_embedder; the distinction
between AR (reference) and generative (target) tokens is controlled by
``token_types``:  0 = AR/causal, 1 = generative/bidirectional.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Union

import einops
import torch
import yaml
from safetensors.torch import load_file, save_file

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management import MemoryManager
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.FakeVAE import FakeVAE
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import quantize, get_qtype, quantize_model

from optimum.quanto import freeze
from transformers import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from .src.hidream_o1.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from .src.hidream_o1.pipeline import (
    HiDreamO1Pipeline,
    DEFAULT_NOISE_SCALE,
    PATCH_SIZE,
    TIMESTEP_TOKEN_NUM,
)
from .src.hidream_o1.pipeline_ref import (
    patchify_image,
    build_ref_input_ids,
    build_ref_conditioning,
    collate_ref_batch,
)
from .src.hidream_o1.model_config import model_config as _MODEL_CFG_DICT

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False,
}

_GLOBAL_NOISE_SCALE = DEFAULT_NOISE_SCALE


# ---------------------------------------------------------------------------
# Scheduler (same as before)
# ---------------------------------------------------------------------------

class HidreamO1FlowmatchScheduler(CustomFlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        self.noise_scale = kwargs.pop("noise_scale", DEFAULT_NOISE_SCALE)
        super().__init__(*args, **kwargs)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        scaled_noise = noise * self.noise_scale
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * scaled_noise
        return noisy_model_input


# ---------------------------------------------------------------------------
# Special token helpers
# ---------------------------------------------------------------------------

def add_special_tokens(tokenizer) -> None:
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


class FakeConfig:
    pass


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, scaling_factor=1.0):
        super().__init__()
        self._dtype = torch.float32
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = FakeConfig()
        self.config.scaling_factor = scaling_factor

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs):
        if "dtype" in kwargs:
            self._dtype = kwargs["dtype"]
        if "device" in kwargs:
            self._device = kwargs["device"]
        return super().to(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class HidreamO1Model(BaseModel):
    arch = "hidream_o1"

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
        self.target_lora_modules = ["Qwen3VLForConditionalGeneration"]
        self.noise_scale = self.model_config.model_kwargs.get(
            "noise_scale", DEFAULT_NOISE_SCALE
        )
        self.noise_scale_inference = self.model_config.model_kwargs.get(
            "noise_scale_inference", self.noise_scale
        )
        # Maximum reference images supported per sample during training
        self.max_ref_images: int = self.model_config.model_kwargs.get(
            "max_ref_images", 4
        )
        # Reference image max patches per side (controls resize)
        self.ref_max_patches_per_side: int = self.model_config.model_kwargs.get(
            "ref_max_patches_per_side", 16
        )
        print(f"Using noise scale: {self.noise_scale}")
        print(f"Multi-reference training: max_ref_images={self.max_ref_images}")
        global _GLOBAL_NOISE_SCALE
        _GLOBAL_NOISE_SCALE = self.noise_scale
        self.is_comfy_weight = False

    # -----------------------------------------------------------------------
    # Scheduler
    # -----------------------------------------------------------------------

    @staticmethod
    def get_train_scheduler():
        return HidreamO1FlowmatchScheduler(
            **scheduler_config, noise_scale=_GLOBAL_NOISE_SCALE
        )

    def get_bucket_divisibility(self):
        return 32  # PATCH_SIZE

    # -----------------------------------------------------------------------
    # Model loading (unchanged from original)
    # -----------------------------------------------------------------------

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading HidreamO1 model")
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading transformer")

        try:
            processor = AutoProcessor.from_pretrained(model_path)
        except Exception as e:
            print(
                f"Failed to load processor from {model_path}, trying extras path. Error: {e}"
            )
            processor_path = self.model_config.extras_name_or_path
            if processor_path and processor_path.endswith(".safetensors"):
                processor_path = "HiDream-ai/HiDream-O1-Image"
            processor = AutoProcessor.from_pretrained(
                processor_path or "HiDream-ai/HiDream-O1-Image"
            )

        tokenizer = get_tokenizer(processor)
        add_special_tokens(tokenizer)

        if model_path.endswith(".safetensors"):
            self.is_comfy_weight = True
            self.print_and_status_update(
                "Model is in safetensors format, loading with safetensors"
            )
            state_dict = load_file(model_path)
            for key, value in state_dict.items():
                state_dict[key] = value.to(dtype=dtype)
            state_dict["lm_head.weight"] = torch.zeros(
                151936, 4096, dtype=torch.bfloat16, device="cpu"
            )
            transformer = Qwen3VLForConditionalGeneration.from_pretrained(
                None,
                config=Qwen3VLConfig(**_MODEL_CFG_DICT),
                state_dict=state_dict,
                torch_dtype=self.torch_dtype,
            )
            del state_dict
        else:
            transformer = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
            )

        flush()
        if not self.model_config.low_vram:
            transformer.to(self.device_torch)

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[],
            )

        flush()

        if self.model_config.low_vram:
            transformer.to(self.device_torch)

        vae = FakeVAE().to(self.device_torch, dtype=dtype)
        text_encoder = FakeTextEncoder().to(self.device_torch, dtype=dtype)

        self.noise_scheduler = HidreamO1Model.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: HiDreamO1Pipeline = HiDreamO1Pipeline(
            scheduler=self.noise_scheduler,
            processor=processor,
            model=None,
        )
        pipe.model = transformer

        flush()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = processor
        self.model = pipe.model
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    # -----------------------------------------------------------------------
    # VAE pass-through (no VAE in this model)
    # -----------------------------------------------------------------------

    def encode_images(self, image_list: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        return image_list.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        return latents.to(device, dtype=dtype)

    # -----------------------------------------------------------------------
    # Prompt embedding (plain text, no ref images)
    # -----------------------------------------------------------------------

    def get_prompt_embeds(self, prompt: Union[str, List[str]]) -> AdvancedPromptEmbeds:
        if not isinstance(prompt, list):
            prompt = [prompt]
        token_list = [self.pipeline.encode_prompt(p) for p in prompt]
        pe = AdvancedPromptEmbeds(text_embeds=token_list)
        pe._frozen_dtype_keys = ["text_embeds"]
        return pe

    # -----------------------------------------------------------------------
    # Prompt embedding WITH reference images
    # -----------------------------------------------------------------------

    def get_prompt_embeds_with_refs(
        self,
        prompt: Union[str, List[str]],
        ref_images_per_sample: List[List[Union["Image", torch.Tensor]]],
    ) -> AdvancedPromptEmbeds:
        """
        Encode prompt(s) together with per-sample reference image lists.

        Parameters
        ----------
        prompt : str or list[str]
            One prompt per sample (or a single prompt broadcast to all).
        ref_images_per_sample : list[list[PIL.Image or tensor]]
            ``ref_images_per_sample[i]`` is the list of reference images for
            sample i.  Can be an empty list for text-only samples.

        Returns
        -------
        AdvancedPromptEmbeds
            ``.text_embeds[i]`` is a (1, prefix_len_i) int64 tensor that already
            encodes text + reference block tokens (bor/img/eor/bot/tms).
            The caller is responsible for passing the corresponding
            ``ref_patches_per_sample`` to ``get_noise_prediction``.
        """
        if isinstance(prompt, str):
            prompt = [prompt] * len(ref_images_per_sample)

        assert len(prompt) == len(ref_images_per_sample), (
            "prompt list length must match ref_images_per_sample length"
        )

        token_list = []
        for p, ref_imgs in zip(prompt, ref_images_per_sample):
            # 1. get base text tokens (chat-template + boi + tms)
            base_ids = self.pipeline.encode_prompt(p)  # (1, txt_len)

            if not ref_imgs:
                # No references – use the standard token sequence
                token_list.append(base_ids)
                continue

            # 2. patchify each reference image to count patches
            ref_patches = [
                patchify_image(
                    img,
                    patch_size=PATCH_SIZE,
                    max_patches_per_side=self.ref_max_patches_per_side,
                )
                for img in ref_imgs[:self.max_ref_images]
            ]
            ref_patch_counts = [rp.shape[0] for rp in ref_patches]

            # 3. Rebuild prefix with reference block tokens
            #    We strip the trailing boi+tms from base_ids (added by encode_prompt)
            #    and let build_ref_input_ids add the proper suffix instead.
            tokenizer = get_tokenizer(self.tokenizer)
            boi_id = tokenizer.encode(
                getattr(tokenizer, "boi_token", "<|boi_token|>"),
                add_special_tokens=False,
            )
            tms_id = tokenizer.encode(
                getattr(tokenizer, "tms_token", "<|tms_token|>"),
                add_special_tokens=False,
            )
            # Strip trailing tms + boi tokens that encode_prompt appended
            strip = TIMESTEP_TOKEN_NUM + 1  # tms×1 + boi×1
            text_only_ids = base_ids[:, :-strip]  # (1, txt_len - strip)

            prefix_ids = build_ref_input_ids(
                text_only_ids,
                ref_patch_counts,
                tokenizer,
            )  # (1, prefix_len)

            token_list.append(prefix_ids)

        pe = AdvancedPromptEmbeds(text_embeds=token_list)
        pe._frozen_dtype_keys = ["text_embeds"]
        return pe

    # -----------------------------------------------------------------------
    # Noise prediction – core training step
    # -----------------------------------------------------------------------

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO",
        **kwargs,
    ) -> torch.Tensor:
        """
        Run one denoising forward pass.

        If ``batch.ref_images`` is set (list-of-lists of PIL/tensor images),
        reference patches are embedded and prepended to the sequence.
        Otherwise, the standard text-only path is used.

        Reference images in the batch
        ------------------------------
        ``batch.ref_images`` should be a list of length ``batch_size``, where
        each element is a list of reference PIL images (or tensors) for that
        sample.  An empty list means no references for that sample.
        ``batch.ref_images`` can be ``None`` for backwards compatibility.
        """
        import einops
        from .src.hidream_o1.pipeline import PATCH_SIZE as _P, T_EPS

        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        device = self.device_torch
        in_dtype = latent_model_input.dtype
        bs, _, h_pix, w_pix = latent_model_input.shape
        h_patches = h_pix // _P
        w_patches = w_pix // _P

        # Patchify target noise input
        z = einops.rearrange(
            latent_model_input,
            "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
            p1=_P,
            p2=_P,
        ).to(device)  # (B, h_patches*w_patches, C*p*p)

        model_config = self.model.config
        pad_token_id = getattr(model_config, "pad_token_id", 0) or 0

        # ----------------------------------------------------------------
        # Determine whether we have reference images for this batch
        # ----------------------------------------------------------------
        ref_images_batch: Optional[List[List]] = getattr(batch, "ref_images", None)
        has_refs = (
            ref_images_batch is not None
            and any(len(r) > 0 for r in ref_images_batch)
        )

        with torch.no_grad():
            per_sample = []

            for b in range(bs):
                tokens = text_embeddings.text_embeds[b]  # (1, seq_len)
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)

                # ---- Build per-sample refs (if any) ----
                ref_imgs: List = (
                    ref_images_batch[b]
                    if has_refs and ref_images_batch is not None
                    else []
                )
                ref_imgs = ref_imgs[: self.max_ref_images]

                if ref_imgs:
                    # Patchify reference images
                    ref_patches = [
                        patchify_image(
                            img,
                            patch_size=_P,
                            max_patches_per_side=self.ref_max_patches_per_side,
                        ).to(device, dtype=torch.float32)
                        for img in ref_imgs
                    ]
                    ref_patch_counts = [rp.shape[0] for rp in ref_patches]
                    n_ref_total = sum(ref_patch_counts)

                    # Embed reference patches through x_embedder
                    ref_patches_cat = torch.cat(ref_patches, dim=0)  # (n_ref, dim)
                    ref_embedded = self.model.model.x_embedder(
                        ref_patches_cat.unsqueeze(0)  # (1, n_ref, dim)
                    )  # (1, n_ref, hidden)

                    # Build conditioning sample with ref-aware positions
                    sample = self._build_ref_sample(
                        tokens, ref_patches, h_pix, w_pix, device
                    )
                    sample["ref_embedded"] = ref_embedded
                    sample["n_ref"] = n_ref_total
                    per_sample.append(sample)
                else:
                    # No refs – standard path
                    sample = self.pipeline.build_conditioning_sample(
                        tokens.to(device), h_pix, w_pix
                    )
                    sample["ref_embedded"] = None
                    sample["n_ref"] = 0
                    per_sample.append(sample)

            # ----------------------------------------------------------------
            # Pad batch to common length
            # ----------------------------------------------------------------
            max_seq_len = max(s["input_ids"].shape[-1] for s in per_sample)
            ids_l, pos_l, tt_l, vm_l, mask_l, ref_em_l = [], [], [], [], [], []

            for s in per_sample:
                ids = s["input_ids"].to(device)
                pos = s["position_ids"].to(device)
                tt = s["token_types"].to(device)
                vm = s["vinput_mask"].to(device)
                seq_len = ids.shape[-1]
                pad_len = max_seq_len - seq_len

                if pad_len > 0:
                    ids = torch.cat(
                        [torch.full((1, pad_len), pad_token_id, dtype=ids.dtype, device=device), ids],
                        dim=-1,
                    )
                    pos = torch.cat(
                        [torch.ones((3, 1, pad_len), dtype=pos.dtype, device=device), pos],
                        dim=-1,
                    )
                    tt = torch.cat(
                        [torch.zeros((1, pad_len), dtype=tt.dtype, device=device), tt],
                        dim=-1,
                    )
                    vm = torch.cat(
                        [torch.zeros((1, pad_len), dtype=vm.dtype, device=device), vm],
                        dim=-1,
                    )
                    mask = torch.cat(
                        [torch.zeros((1, pad_len), dtype=torch.long, device=device),
                         torch.ones((1, seq_len), dtype=torch.long, device=device)],
                        dim=-1,
                    )
                else:
                    mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

                ids_l.append(ids)
                pos_l.append(pos)
                tt_l.append(tt)
                vm_l.append(vm)
                mask_l.append(mask)
                ref_em_l.append(s.get("ref_embedded"))

            input_ids = torch.cat(ids_l, dim=0)                    # (B, S)
            position_ids = torch.cat(pos_l, dim=1)                 # (3, B, S)
            token_types = torch.cat(tt_l, dim=0)                   # (B, S)
            vinput_mask = torch.cat(vm_l, dim=0)                   # (B, S)
            attention_mask = torch.cat(mask_l, dim=0)              # (B, S)

        # ----------------------------------------------------------------
        # Build vinputs: concatenate ref patches then target patches
        # ----------------------------------------------------------------
        # z is (B, n_tgt, C*p*p) – target patches
        # We build per-sample vinputs = [ref_patches | target_patches]
        vinputs = self._build_vinputs(z, ref_em_l, per_sample, device)
        # vinputs: (B, n_ref_max + n_tgt, C*p*p) [ref parts zero-padded for shorter seqs]

        # ----------------------------------------------------------------
        # Timestep
        # ----------------------------------------------------------------
        t_pixeldit = (1.0 - timestep.float() / 1000.0).to(device)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask if bs > 1 else None,
            vinputs=vinputs,
            timestep=t_pixeldit.reshape(-1),
            token_types=token_types,
            use_flash_attn=True,
        )
        x_pred = outputs.x_pred  # (B, total_seq, C*p*p)

        # ----------------------------------------------------------------
        # Extract only the TARGET generation tokens from x_pred
        # ----------------------------------------------------------------
        vision_pred = torch.stack(
            [x_pred[b][vinput_mask[b].bool()] for b in range(bs)],
            dim=0,
        )  # (B, n_tgt, C*p*p)

        x0_pred = einops.rearrange(
            vision_pred,
            "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
            H=h_patches,
            W=w_patches,
            p1=_P,
            p2=_P,
        )

        sigma = (timestep.float() / 1000.0).clamp_min(T_EPS).to(device)
        while sigma.dim() < latent_model_input.dim():
            sigma = sigma.unsqueeze(-1)
        pred = (latent_model_input.float().to(device) - x0_pred.float()) / sigma
        return pred.to(in_dtype)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_ref_sample(
        self,
        tokens: torch.Tensor,
        ref_patches: List[torch.Tensor],
        h_pix: int,
        w_pix: int,
        device: torch.device,
    ) -> dict:
        """
        Build the conditioning sample (input_ids, position_ids, token_types,
        vinput_mask) for a sample that has reference images.

        This mirrors pipeline.build_conditioning_sample but inserts
        bor/eor/bot tokens around the reference patch slots.
        """
        tokenizer = get_tokenizer(self.tokenizer)
        model_config = self.model.config

        ref_patch_counts = [rp.shape[0] for rp in ref_patches]

        # Strip the trailing boi+tms that encode_prompt already appended
        strip = TIMESTEP_TOKEN_NUM + 1  # tms + boi
        text_only_ids = tokens[:, :-strip] if tokens.shape[1] > strip else tokens

        # Build the prefix with ref blocks
        prefix_ids = build_ref_input_ids(
            text_only_ids,
            ref_patch_counts,
            tokenizer,
            device=device,
        )  # (1, prefix_len)

        # Re-use build_ref_conditioning for position ids, token_types, masks
        conditioning = build_ref_conditioning(
            text_input_ids=text_only_ids,
            ref_patches=ref_patches,
            height=h_pix,
            width=w_pix,
            model_config=model_config,
            tokenizer=tokenizer,
            device=device,
        )

        return conditioning

    def _build_vinputs(
        self,
        z: torch.Tensor,
        ref_em_l: List[Optional[torch.Tensor]],
        per_sample: List[dict],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the combined vinputs tensor: [ref_patches | target_patches].

        For samples with no reference images, the ref slot is omitted and only
        target patches are included (the vinput_mask already marks only target
        tokens as generative, so ref-less samples work correctly).

        Parameters
        ----------
        z : (B, n_tgt, C*p*p)
            Patchified target (noisy) images.
        ref_em_l : list of (1, n_ref, hidden) or None
            Pre-embedded reference patches from x_embedder per sample.
            NOTE: these are already embedded – but _forward_generation calls
            x_embedder again on vinputs.  So we pass raw patches here instead.

        Returns
        -------
        vinputs : (B, max_total, C*p*p)
            Zero-padded if samples have different numbers of ref patches.
        """
        bs = z.shape[0]
        n_tgt = z.shape[1]
        patch_dim = z.shape[2]

        # Max total vinput length = max(n_ref_i) + n_tgt
        max_n_ref = max(s["n_ref"] for s in per_sample)
        max_total = max_n_ref + n_tgt

        vinputs = torch.zeros((bs, max_total, patch_dim), dtype=z.dtype, device=device)

        for b in range(bs):
            n_ref = per_sample[b]["n_ref"]
            # Reference patches – if present, we get them from the batch's
            # ref_vinput_mask to know where they sit; simpler here is to
            # just front-fill.
            # The ref_em_l holds EMBEDDED values; we need raw patches for
            # the model's x_embedder inside _forward_generation.
            # We store raw patches on the sample dict in get_noise_prediction.
            raw_ref = per_sample[b].get("raw_ref_patches")  # (n_ref, patch_dim)
            if raw_ref is not None and n_ref > 0:
                vinputs[b, :n_ref, :] = raw_ref.to(device, dtype=z.dtype)
            # Target patches
            vinputs[b, max_n_ref : max_n_ref + n_tgt, :] = z[b]

        return vinputs  # (B, max_total, patch_dim)

    # -----------------------------------------------------------------------
    # Generation (inference)
    # -----------------------------------------------------------------------

    def get_generation_pipeline(self):
        scheduler = HidreamO1Model.get_train_scheduler()
        pipe: HiDreamO1Pipeline = HiDreamO1Pipeline(
            scheduler=scheduler,
            processor=self.tokenizer,
            model=None,
        )
        pipe.model = self.transformer
        return pipe

    def generate_single_image(
        self,
        pipeline: HiDreamO1Pipeline,
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

        img = pipeline(
            prompt_input_ids=conditional_embeds.text_embeds[0],
            negative_prompt_input_ids=unconditional_embeds.text_embeds[0],
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            generator=generator,
            noise_scale=self.noise_scale_inference,
            **extra,
        ).images[0]
        return img

    # -----------------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------------

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: Qwen3VLForConditionalGeneration = unwrap_model(self.model)
        if self.is_comfy_weight:
            sd = transformer.state_dict()
            save_dict = {}
            for key, value in sd.items():
                if "lm_head.weight" in key:
                    continue
                save_dict[key] = value.clone().to("cpu", dtype=save_dtype)
            if not output_path.endswith(".safetensors"):
                output_path += ".safetensors"
            meta = get_meta_for_safetensors(meta, name=self.arch)
            save_file(save_dict, output_path, metadata=meta)
        else:
            transformer.save_pretrained(
                save_directory=output_path,
                safe_serialization=True,
            )
            self.tokenizer.save_pretrained(output_path)
            meta_path = os.path.join(output_path, "aitk_meta.yaml")
            with open(meta_path, "w") as f:
                yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        noise_scale = self.noise_scale
        return (noise * noise_scale - batch.latents).detach()

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_key = new_key.replace(".model.", ".")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.model.")
            new_key = new_key.replace("transformer.model.model.", "transformer.model.")
            new_sd[new_key] = value
        return new_sd
