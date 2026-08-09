"""Mage-Flow (microsoft/Mage) for ai-toolkit.

Two archs backed by the same NR-MMDiT stack:
  - ``mageflow``       text-to-image  (microsoft/Mage-Flow-Base)
  - ``mageflow_edit``  instruction edit (microsoft/Mage-Flow-Edit-Base)

Components (diffusers-style repo layout):
  - transformer: ``MageFlow`` dual-stream DiT (src/transformer.py) — packed
    variable-length [text | image(+refs)] sequences, per-sample 2D RoPE,
    joint varlen attention. 4B params, hidden 3072, 12 blocks.
  - text encoder: Qwen3-VL (ships inside the repo under ``text_encoder/``);
    conditioning is the final hidden states with the templated system prompt
    dropped (34 tokens for t2i, 64 for edit).
  - autoencoder: ``MageVAE`` (src/vae.py) — 128-channel, 16x downsample
    one-step diffusion codec, no latent normalization.

Flow matching convention matches ai-toolkit exactly (sigma=1 noise -> sigma=0
clean, target = noise - clean), static sigma shift 6.0. Edit mode feeds the
reference images in two places, as in the reference implementation: through
the Qwen3-VL encoder alongside the instruction (long edge capped at 384) and
as clean VAE latents sequence-appended after the noisy target tokens (at the
target resolution).
"""

import json
import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from safetensors.torch import load_file, save_file

import huggingface_hub
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from optimum.quanto import freeze

from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from toolkit.metadata import get_meta_for_safetensors
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager

from .src.transformer import MageFlow, MageFlowParams
from .src.vae import MageVAE
from .src.text_encoder import (
    encode_mageflow_prompt,
    patch_qwen_vl_patch_embed,
    resize_vl_images,
)
from .src.pipeline import MageFlowPipeline, predict_velocity

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


# Training/sampling timestep distribution: the released checkpoints use a
# static sigma shift of 6.0 (transformer/config.json + scheduler config).
scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 6.0,
}

# Keys of the checkpoint transformer/config.json that are NOT MageFlowParams
# constructor args (legacy/unused fields) — same filter as the reference
# ``load_from_repo``. Everything else becomes the DiT structure.
_CONFIG_META_KEYS = {
    "_class_name",
    "txt_max_length",
    "max_sequence_length",
    "param_dtype",
    "packing",
    "schedule_mode",
    "static_shift",
    "use_time_shift",
    "rope_type",
    "apply_text_rotary_emb",
    "mlp_ratio",
    "depth_single_blocks",
    "theta",
    "qkv_bias",
    "guidance_embed",
    "vec_in_dim",
    "vec_type",
    "time_type",
    "double_block_type",
}

HF_TOKEN = os.getenv("HF_TOKEN", None)


class MageFlowModel(BaseModel):
    arch = "mageflow"
    is_edit = False
    use_old_lokr_format = False

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
        self.target_lora_modules = ["MageFlow"]

        # MageVAE is 16x downsampling; the DiT patch size is 1 (one token per
        # latent pixel), so pixel sizes must be multiples of 16.
        self.patch_size = 1
        self.vae_scale_factor = 16
        # Safety cap on prompt token length (truncation only); embeds are stored
        # per-sample at natural length and packed varlen at the model call.
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", 2048)
        )
        # Qwen3-VL AutoProcessor for encoding reference images into the prompt
        # (edit only).
        self.vl_processor = None

        if self.is_edit:
            # Reference images feed the model in two places: through the
            # Qwen3-VL encoder alongside the instruction, and as clean VAE
            # latents sequence-appended after the noisy target tokens.
            self.encode_control_in_text_embeddings = True
            self.has_multiple_control_images = True
            # References keep their own aspect/size in the dataloader; they are
            # resized to the target resolution here (reference behavior).
            self.use_raw_control_images = True

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # 16 for the VAE downsample, patch size is 1.
        return self.vae_scale_factor * self.patch_size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _get_model_file(self, relpath: str) -> str:
        """Resolve a repo-relative file from a local directory or the HF hub."""
        name_or_path = self.model_config.name_or_path
        if os.path.isdir(name_or_path):
            path = os.path.join(name_or_path, relpath)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Could not find {relpath!r} in {name_or_path!r}"
                )
            return path
        return huggingface_hub.hf_hub_download(
            repo_id=name_or_path, filename=relpath, token=HF_TOKEN
        )

    def _load_transformer(self) -> MageFlow:
        dtype = self.torch_dtype
        self.print_and_status_update("Loading transformer (MageFlow NR-MMDiT)")

        with open(self._get_model_file("transformer/config.json")) as f:
            tcfg = json.load(f)
        structure = {k: v for k, v in tcfg.items() if k not in _CONFIG_META_KEYS}
        structure.update(self.model_config.model_kwargs.get("transformer_config", {}))
        params = MageFlowParams(**structure)

        # Build on meta, then materialize straight from the checkpoint.
        with torch.device("meta"):
            transformer = MageFlow(params)

        self.print_and_status_update("  - fetching transformer weights")
        state_dict = load_file(
            self._get_model_file("transformer/diffusion_pytorch_model.safetensors")
        )
        state_dict = {
            k: (v.to(dtype) if v.is_floating_point() else v)
            for k, v in state_dict.items()
        }
        self.print_and_status_update("  - loading transformer state dict")
        transformer.load_state_dict(state_dict, strict=True, assign=True)
        # The RoPE tables are plain tensors (not buffers/params), so the meta
        # init left them unmaterialized — rebuild them for real.
        transformer.reset_rope()
        del state_dict
        flush()
        return transformer

    def _load_text_encoder(self):
        dtype = self.torch_dtype
        te_path = self.model_config.model_kwargs.get("text_encoder_path", None)
        if te_path is not None:
            te_kwargs = {}
        else:
            te_path = self.model_config.name_or_path
            te_kwargs = {"subfolder": "text_encoder"}
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")

        tokenizer = AutoTokenizer.from_pretrained(te_path, token=HF_TOKEN, **te_kwargs)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            te_path, torch_dtype=dtype, token=HF_TOKEN, **te_kwargs
        )
        vl_processor = None
        if self.is_edit:
            # Edit mode: reference images are encoded into the text embeddings,
            # so the vision tower stays. Swap its Conv3d patch_embed for an
            # equivalent GEMM (bf16 Conv3d has no fast cuDNN kernel).
            vl_processor = AutoProcessor.from_pretrained(
                te_path, token=HF_TOKEN, **te_kwargs
            )
            patch_qwen_vl_patch_embed(text_encoder)
        else:
            # We only ever encode text, so the vision tower is dead weight --
            # drop it to free VRAM.
            if getattr(text_encoder.model, "visual", None) is not None:
                text_encoder.model.visual = None
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return tokenizer, vl_processor, text_encoder

    def _load_vae(self) -> MageVAE:
        self.print_and_status_update("Loading MageVAE")
        vae_path = self.model_config.model_kwargs.get("vae_path", None)
        if vae_path is None:
            vae_path = self._get_model_file("vae/diffusion_pytorch_model.safetensors")
        vae = MageVAE(
            ckpt_path=vae_path,
            sample_posterior=bool(
                self.model_config.model_kwargs.get("vae_sample_posterior", True)
            ),
        )
        vae.eval()
        vae.requires_grad_(False)
        return vae

    def get_quantization_exclude_modules(self):
        # sensitive modules kept in full precision (fnmatch patterns on module
        # names within MageFlow):
        #   img_in / txt_in / txt_norm - input projections
        #   time_text_embed*           - timestep embedder feeding every
        #                                block's modulation
        #   norm_out* / proj_out       - final adaptive norm / output projection
        return [
            "img_in",
            "txt_in",
            "txt_norm",
            "time_text_embed*",
            "norm_out*",
            "proj_out",
        ]

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Mage-Flow model")

        transformer = self._load_transformer()

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing transformer")
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
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        tokenizer, vl_processor, text_encoder = self._load_text_encoder()
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing text encoder")
            text_encoder.to(self.device_torch)
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving text encoder to CPU")
            text_encoder.to("cpu")
        else:
            text_encoder.to(self.device_torch)
        flush()

        vae = self._load_vae()
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)

        self.noise_scheduler = MageFlowModel.get_train_scheduler()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vl_processor = vl_processor
        self.model = transformer
        self.pipeline = MageFlowPipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return MageFlowPipeline(self)

    def generate_single_image(
        self,
        pipeline: MageFlowPipeline,
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

        # Reference image(s) -> clean VAE latents appended to the sequence.
        # The Qwen3-VL side already saw them (baked into the prompt embeds).
        # ctrl_img_1 mirrors ctrl_img when unset, so use one or the other.
        ctrl_paths = []
        if self.is_edit:
            if gen_config.ctrl_img is not None:
                ctrl_paths.append(gen_config.ctrl_img)
            elif gen_config.ctrl_img_1 is not None:
                ctrl_paths.append(gen_config.ctrl_img_1)
            if gen_config.ctrl_img_2 is not None:
                ctrl_paths.append(gen_config.ctrl_img_2)
            if gen_config.ctrl_img_3 is not None:
                ctrl_paths.append(gen_config.ctrl_img_3)

        ref_latents = None
        if ctrl_paths:
            ctrl_tensors = [
                to_tensor(Image.open(path).convert("RGB")) for path in ctrl_paths
            ]
            # one batch item (preview batch size is 1) -> List[List[(128, h, w)]]
            ref_latents = [
                self._encode_ref_latents(
                    ctrl_tensors, gen_config.height, gen_config.width
                )
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

    # ------------------------------------------------------------------
    # Reference-image helpers (edit)
    # ------------------------------------------------------------------
    def _encode_ref_latents(
        self, control_tensors, target_height: int, target_width: int
    ) -> List[torch.Tensor]:
        """Encode ``[0, 1]`` reference image tensors to clean VAE latents.

        Sizing (deliberate deviation from the reference ``generate_edits``,
        which resizes every ref to exactly the target resolution and squishes
        mismatched aspect ratios): each reference is resized to the TARGET's
        total pixel count while keeping its OWN aspect ratio, snapped to the
        16px divisibility. A ref that shares the target's aspect ratio still
        lands on exactly the target size (identical to the reference
        behavior); one with a different aspect ratio keeps its shape instead
        of being distorted. Returns a list of ``(128, h, w)`` latents (one per
        reference image). ``control_tensors`` is a list of ``(C, H, W)`` or
        ``(1, C, H, W)`` tensors in ``[0, 1]``.
        """
        sc = self.get_bucket_divisibility()  # 16
        target_area = target_height * target_width
        latents = []
        for img in control_tensors:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(self.device_torch, dtype=torch.float32)

            h, w = img.shape[2], img.shape[3]
            ratio = h / w
            new_h = math.sqrt(target_area * ratio)
            new_w = new_h / ratio
            new_h = max(sc, int(round(new_h / sc)) * sc)
            new_w = max(sc, int(round(new_w / sc)) * sc)
            if (new_h, new_w) != (h, w):
                img = F.interpolate(
                    img,
                    size=(new_h, new_w),
                    mode="bicubic",
                    antialias=True,
                ).clamp(0, 1)
            # encode_images expects [-1, 1]; control tensors arrive in [0, 1].
            latent = self.encode_images(
                img * 2 - 1, device=self.device_torch, dtype=self.torch_dtype
            )
            latents.append(latent[0])  # drop batch dim -> (128, h, w)
        return latents

    def _batch_ref_latents_from_batch(
        self,
        batch: "DataLoaderBatchDTO",
        batch_size: int,
        target_height: int,
        target_width: int,
    ) -> Optional[List[List[torch.Tensor]]]:
        """Build predict_velocity's ``ref_latents`` from a train batch."""
        control_list = batch.control_tensor_list
        if control_list is None and batch.control_tensor is not None:
            control_list = [batch.control_tensor[b : b + 1] for b in range(batch_size)]
        if control_list is None:
            return None
        if len(control_list) != batch_size:
            raise ValueError("Control tensor list length does not match batch size")
        ref_latents = []
        for controls in control_list:
            if isinstance(controls, torch.Tensor):
                controls = [controls]
            ref_latents.append(
                self._encode_ref_latents(controls, target_height, target_width)
            )
        return ref_latents

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 128, h, w)
        timestep: torch.Tensor,  # 0..1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # Clean reference latents from the batch's control images (if any);
        # they ride along in the sequence, never noised.
        ref_latents = None
        if batch is not None and self.is_edit:
            with torch.no_grad():
                _, _, lh, lw = latent_model_input.shape
                ref_latents = self._batch_ref_latents_from_batch(
                    batch,
                    latent_model_input.shape[0],
                    target_height=lh * self.vae_scale_factor,
                    target_width=lw * self.vae_scale_factor,
                )

        # toolkit timestep (0..1000, 1000 = pure noise) -> flow sigma in
        # [0, 1] with 1 = pure noise. Same convention -> straight divide.
        t = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != latent_model_input.shape[0]:
            t = t.expand(latent_model_input.shape[0])

        pred = predict_velocity(
            self.transformer,
            latent_model_input.to(self.device_torch, self.torch_dtype),
            t,
            text_embeddings.text_embeds,
            ref_latents=ref_latents,
        )
        return pred

    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        # Normalize control images to a per-prompt list (List[List[Tensor]]).
        # They arrive as a (B, C, H, W) batch tensor (control_tensor), a list of
        # per-sample lists (control_tensor_list), or a flat list of (1, C, H, W)
        # tensors for a single prompt (sampling / blank-embed caching).
        if control_images is not None:
            if isinstance(control_images, torch.Tensor):
                control_images = [
                    [control_images[i]] for i in range(control_images.shape[0])
                ]
            elif len(control_images) > 0 and not isinstance(control_images[0], list):
                control_images = [control_images]
            if len(control_images) == 1 and len(prompt) > 1:
                control_images = control_images * len(prompt)
            if len(control_images) != len(prompt):
                raise ValueError(
                    "Number of prompts must match number of control image sets"
                )
        else:
            control_images = [None] * len(prompt)

        template_name = "mage-flow-edit" if self.is_edit else "mage-flow"
        vl_long_edge = int(self.model_config.model_kwargs.get("vl_cond_long_edge", 384))

        # Encode each prompt at its natural length and store one (L, 2560)
        # tensor per batch item. Padding is never needed: the packed varlen
        # model call consumes the per-sample lengths directly.
        features_list = []
        for p, ctrl in zip(prompt, control_images):
            images = None
            if self.is_edit and ctrl is not None and len(ctrl) > 0:
                images = resize_vl_images(ctrl, max_long_edge=vl_long_edge)
            features = encode_mageflow_prompt(
                self.text_encoder,
                self.tokenizer,
                p,
                template_name=template_name,
                max_length=self.max_text_length,
                images=images,
                processor=self.vl_processor,
                dtype=self.torch_dtype,
            )
            features_list.append(features)

        return AdvancedPromptEmbeds(text_embeds=features_list)

    def get_loss_target(self, *args, **kwargs):
        # Flow-matching velocity target: noise - clean.
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # VAE (MageVAE -- raw latents, no normalization)
    # ------------------------------------------------------------------
    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()

        if isinstance(image_list, list):
            images = torch.stack(
                [img.squeeze(0) if img.dim() == 4 else img for img in image_list], dim=0
            )
        else:
            images = image_list
        images = images.to(self.vae.device, dtype=self.vae.dtype)

        latents = self.vae.encode(images)  # (B, 128, H/16, W/16)
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(self.vae.device, dtype=self.vae.dtype)
        images = self.vae.decode(latents)  # (B, 3, H, W) in [-1, 1]
        return images.to(device, dtype=dtype)

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        from toolkit.util.quantize import dequantize_if_quantized

        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: MageFlow = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            # dequantize any quantized (e.g. quanto/torchao) weights so we save plain full precision tensors
            save_dict[k] = (
                dequantize_if_quantized(v).clone().to("cpu", dtype=save_dtype)
            )
        meta = get_meta_for_safetensors(meta, name=self.arch)
        save_file(save_dict, output_path, metadata=meta)

    def get_base_model_version(self):
        return "mageflow"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

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


class MageFlowEditModel(MageFlowModel):
    arch = "mageflow_edit"
    is_edit = True

    def get_base_model_version(self):
        return "mageflow_edit"
