"""Krea 2 (K2) for ai-toolkit.

Krea 2 is a single-stream MMDiT text-to-image model:
  - text encoder: Qwen3-VL-4B-Instruct (a stack of 12 hidden-state layers is fed
    in; ``src/text_encoder.py``),
  - autoencoder: the Qwen-Image VAE (f8, 16 latent channels, the same VAE the
    ``qwen_image`` arch uses),
  - denoiser: ``SingleStreamDiT`` (``src/mmdit.py``), which fuses the text layers
    with a small ``TextFusionTransformer`` and runs the packed [text | image]
    sequence through ``SingleStreamBlock`` layers.

Flow-matching convention matches ai-toolkit exactly (t=1 noise -> t=0 clean,
target = noise - clean), so ``get_noise_prediction`` does no time flip / negation.
"""

import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from safetensors.torch import load_file, save_file

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
from diffusers import AutoencoderKLQwenImage
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
)
from optimum.quanto import freeze

from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
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

from .src.mmdit import (
    DoubleSharedModulation,
    SimpleModulation,
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from .src.text_encoder import encode_krea_prompt, SELECT_LAYERS
from .src.pipeline import Krea2Pipeline, pad_text_features, predict_velocity

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


# The reference "single_mmdit_large_wide" architecture (oss_raw / oss_turbo share it).
KREA2_MMDIT_CONFIG = dict(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch_size=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

# Krea 2's mu schedule is exponential time-shifting whose mu is linearly
# interpolated in image-token count between (256-res -> 0.5) and (1280-res ->
# 1.15) -- exactly what CustomFlowMatchEulerDiscreteScheduler's dynamic shifting
# does, so we mirror those endpoints here for the training timestep distribution.
#   x1 = (256  // (8*2))**2 = 256
#   x2 = (1280 // (8*2))**2 = 6400
scheduler_config = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 6400,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
}

# Defaults; both overridable via model.model_kwargs.
QWEN3_VL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_IMAGE_VAE_PATH = "Qwen/Qwen-Image"

HF_TOKEN = os.getenv("HF_TOKEN", None)


def patch_qwen_vl_patch_embed(model):
    """Qwen-VL's vision patch_embed is a Conv3d whose kernel == stride, i.e. a plain
    linear projection of each flattened patch. bf16 Conv3d has no fast cuDNN kernel and
    falls back to a slow, GPU-underutilizing path. Swap it for the equivalent F.linear
    (a GEMM). The weight is read lazily so this survives later .to(device)/dtype moves.
    Returns the number of patch_embed modules patched. (Same patch as the
    Qwen3VLCaptioner extension.)"""
    patched = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(
            proj.stride
        ):

            def fast_forward(hidden_states, _proj=proj):
                w = _proj.weight.reshape(_proj.weight.shape[0], -1)
                x = hidden_states.view(-1, w.shape[1]).to(w.dtype)
                return F.linear(x, w, _proj.bias)

            module.forward = fast_forward
            patched += 1
    return patched


def _load_mmdit_state_dict(name_or_path: str, filename: Optional[str]) -> dict:
    """Load the MMDiT weights from a local safetensors file/dir or the HF hub.

    ``name_or_path`` may be: a ``.safetensors`` file, a directory containing one
    (``filename`` or the lone ``.safetensors`` in it), or a hub repo id (the
    file ``filename`` is downloaded, defaulting to ``model.safetensors``).
    """
    if name_or_path.endswith(".safetensors") and os.path.isfile(name_or_path):
        return load_file(name_or_path)

    if os.path.isdir(name_or_path):
        if filename is not None:
            return load_file(os.path.join(name_or_path, filename))
        candidates = [f for f in os.listdir(name_or_path) if f.endswith(".safetensors")]
        if len(candidates) == 1:
            return load_file(os.path.join(name_or_path, candidates[0]))
        raise FileNotFoundError(
            f"Could not pick an MMDiT checkpoint in {name_or_path}: found "
            f"{candidates}. Set model.model_kwargs.checkpoint_filename."
        )

    # Treat as a hub repo id. When no filename is given, derive it from the repo
    # name's trailing segment (e.g. "krea/Krea-2-Raw" -> "raw.safetensors",
    # "krea/Krea-2-Turbo" -> "turbo.safetensors").
    fname = filename or (
        name_or_path.split("/")[-1].split("-")[-1].lower() + ".safetensors"
    )
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=name_or_path, filename=fname, token=HF_TOKEN
        )
    except EntryNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find {fname!r} in hub repo {name_or_path!r}. Set "
            "model.model_kwargs.checkpoint_filename to the weight file name."
        ) from e
    return load_file(path)


class Krea2Model(BaseModel):
    arch = "krea2"

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
        self.target_lora_modules = ["SingleStreamDiT"]

        self.patch_size = KREA2_MMDIT_CONFIG["patch_size"]
        self.vae_scale_factor = 8  # Qwen-Image VAE is f8
        # Safety cap on prompt token length (truncation only); embeds are stored
        # per-sample at natural length and padded to the batch max at the model call.
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", 512)
        )
        # Qwen2TokenizerFast used to tokenize the assistant suffix (matches the
        # reference's separate processor pass).
        self.processor = None
        # Qwen3-VL AutoProcessor for encoding reference images into the prompt.
        self.vl_processor = None
        self.use_old_lokr_format = False

        # Optional reference-image (edit) conditioning, enabled with
        # model_kwargs.edit = true. Control images feed the model in two places:
        # through the Qwen3-VL encoder alongside the prompt (edit-plus style, so
        # the text embeddings see them) and as clean VAE latents appended to the
        # image sequence at t=0 (ComfyUI Kontext "index_timestep_zero"). Runs in
        # ComfyUI with the ComfyUI-Krea2-Ostris-Edit custom nodes. With edit off
        # (the default) all of it is skipped and this is the plain T2I model.
        self.is_edit = bool(self.model_config.model_kwargs.get("edit", False))
        self.encode_control_in_text_embeddings = self.is_edit
        self.has_multiple_control_images = self.is_edit
        # Reference images keep their own aspect/size (not resized to the target).
        self.use_raw_control_images = self.is_edit
        # model_kwargs.kv_cache = true: train with an asymmetric attention mask
        # where the clean reference tokens attend only to each other (never to
        # text / noisy tokens). Their hidden states then depend only on the
        # refs + t=0 modulation, so at inference their per-layer K/V can be
        # computed once and reused across all denoising steps
        # (OminiControl2-style conditioning feature reuse). Off by default:
        # the base model was trained fully bidirectional, so a LoRA must be
        # trained with kv_cache enabled for kv-cached inference (the ComfyUI
        # node / hub pipeline kv_cache toggles) to work properly.
        self.kv_cache = bool(self.model_config.model_kwargs.get("kv_cache", False))

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # 8 for the VAE downsample, 2 for the patch size.
        return self.vae_scale_factor * self.patch_size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_transformer(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading transformer (SingleStreamDiT)")

        mmdit_kwargs = dict(KREA2_MMDIT_CONFIG)
        mmdit_kwargs.update(self.model_config.model_kwargs.get("mmdit_config", {}))
        config = SingleMMDiTConfig(**mmdit_kwargs)

        # Build on meta, then materialize straight from the checkpoint.
        with torch.device("meta"):
            transformer = SingleStreamDiT(config)

        self.print_and_status_update("  - fetching transformer weights")
        state_dict = _load_mmdit_state_dict(
            self.model_config.name_or_path,
            self.model_config.model_kwargs.get("checkpoint_filename", None),
        )
        state_dict = {
            k: (v.to(dtype) if v.is_floating_point() else v)
            for k, v in state_dict.items()
        }
        self.print_and_status_update("  - loading transformer state dict")
        transformer.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        flush()
        return transformer

    def _load_text_encoder(self):
        dtype = self.torch_dtype
        te_path = self.model_config.model_kwargs.get("text_encoder_path", QWEN3_VL_PATH)
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            te_path, max_length=self.max_text_length, token=HF_TOKEN
        )
        processor = Qwen2TokenizerFast.from_pretrained(
            te_path, max_length=self.max_text_length, token=HF_TOKEN
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            te_path, torch_dtype=dtype, token=HF_TOKEN
        )
        vl_processor = None
        if self.is_edit:
            # Edit mode: reference images are encoded into the text embeddings,
            # so the vision tower stays. Swap its Conv3d patch_embed for an
            # equivalent GEMM (bf16 Conv3d has no fast cuDNN kernel).
            vl_processor = AutoProcessor.from_pretrained(te_path, token=HF_TOKEN)
            patch_qwen_vl_patch_embed(text_encoder)
        else:
            # We only ever encode text, so the vision tower is dead weight -- drop it to
            # free VRAM and skip loading its (bf16-slow) Conv3d patch_embed onto the GPU.
            if getattr(text_encoder.model, "visual", None) is not None:
                text_encoder.model.visual = None
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return tokenizer, processor, vl_processor, text_encoder

    def _load_vae(self):
        vae_path = self.model_config.model_kwargs.get("vae_path", QWEN_IMAGE_VAE_PATH)
        self.print_and_status_update(f"Loading Qwen-Image VAE from {vae_path}")
        vae = AutoencoderKLQwenImage.from_pretrained(
            vae_path, subfolder="vae", torch_dtype=self.vae_torch_dtype, token=HF_TOKEN
        )
        vae.eval()
        vae.requires_grad_(False)
        return vae

    def load_training_adapter(self, transformer: SingleStreamDiT):
        self.print_and_status_update("Loading assistant LoRA")
        lora_path = self.model_config.assistant_lora_path
        if not os.path.exists(lora_path):
            # assume it is a hub path
            lora_splits = lora_path.split("/")
            if len(lora_splits) != 3:
                raise ValueError(
                    f"Assistant LoRA path {lora_path} is not a valid local path or hub path."
                )
            repo_id = "/".join(lora_splits[:2])
            filename = lora_splits[2]
            try:
                lora_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=HF_TOKEN,
                )
                # upgrade path to the local download
                self.model_config.assistant_lora_path = lora_path
            except Exception as e:
                raise ValueError(
                    f"Failed to download assistant LoRA from {lora_path}: {e}"
                )
        # load the adapter and merge it in. We will inference with a -1.0 multiplier so the adapter effects only work during training.
        lora_state_dict = load_file(lora_path)
        # detect the LoRA rank from the first down-projection weight.
        dim_key = next(k for k in lora_state_dict if k.endswith("lora_A.weight"))
        dim = int(lora_state_dict[dim_key].shape[0])

        new_sd = {}
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        lora_state_dict = new_sd

        network_config = {
            "type": "lora",
            "linear": dim,
            "linear_alpha": dim,
            "transformer_only": True,
        }

        network_config = NetworkConfig(**network_config)
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,
            is_ara=True,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        self.print_and_status_update("Merging in assistant LoRA")
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)

        network.merge_in(merge_weight=1.0)

        # mark it as not merged so inference ignores it.
        network.is_merged_in = False

        # add the assistant so sampler will activate it while sampling
        self.assistant_lora: LoRASpecialNetwork = network

        # deactivate lora during training
        self.assistant_lora.multiplier = -1.0
        self.assistant_lora.is_active = False

        # tell the model to invert assistant on inference since we want remove lora effects
        self.invert_assistant_lora = True
    
    def get_quantization_exclude_modules(self):
        # sensitive modules kept in full precision (fnmatch patterns on module
        # names within SingleStreamDiT):
        #   first             - patchified latent input projection
        #   tmlp* / tproj*    - timestep embedder + modulation projection; feed
        #                       every block's DoubleSharedModulation and LastLayer
        #   txtmlp*           - text feature -> model width projection
        #   txtfusion.projector - tiny (num_txt_layers -> 1) encoder-layer mixer
        #   last*             - final norm/modulated output projection
        return [
            "first",
            "tmlp*",
            "tproj*",
            "txtmlp*",
            "txtfusion.projector",
            "last*",
        ]

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Krea 2 model")

        transformer = self._load_transformer()

        # load assistant lora if specified
        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)
            # set qtype to be float8 if it is qfloat8
            if self.model_config.qtype == "qfloat8":
                self.model_config.qtype = "float8"

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
                ignore_modules=[
                    module
                    for module in transformer.modules()
                    if isinstance(module, (SimpleModulation, DoubleSharedModulation))
                ],
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        tokenizer, processor, vl_processor, text_encoder = self._load_text_encoder()
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

        self.noise_scheduler = Krea2Model.get_train_scheduler()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.vl_processor = vl_processor
        self.model = transformer
        self.pipeline = Krea2Pipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Krea2Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Krea2Pipeline,
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

        # Reference image(s) -> clean VAE latents for the t=0 sequence tokens.
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
            target_pixels = gen_config.width * gen_config.height
            # one batch item (preview batch size is 1) -> List[List[(16, h, w)]]
            ref_latents = [
                self._encode_ref_latents(ctrl_tensors, target_pixels=target_pixels)
            ]
        
        # CFG is 0 normalized for this model
        guidance = max(0.0, gen_config.guidance_scale - 1.0)

        img = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=guidance,
            latents=gen_config.latents,
            generator=generator,
            ref_latents=ref_latents,
        )[0]
        return img

    # ------------------------------------------------------------------
    # Reference-image helpers
    # ------------------------------------------------------------------
    def _ref_target_pixels(self, target_pixels: Optional[int]) -> int:
        """Pixel budget each reference image is resized to fit within.

        - default: ``control_image_max_pixels`` model_kwarg (1 MP) -- a hard cap
          so raw, full-size control images don't blow up the token count / VRAM.
        - ``match_target_res`` model_kwarg: use the target generation area instead.
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
        """Build predict_velocity's ``ref_latents`` from a train batch."""
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
    # Training hooks
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 16, h, w)
        timestep: torch.Tensor,  # 0..1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # Clean reference latents from the batch's control images (if any); they
        # ride along in the sequence at t=0 and are never noised.
        ref_latents = None
        if batch is not None and self.is_edit:
            with torch.no_grad():
                _, _, lh, lw = latent_model_input.shape
                target_pixels = (lh * self.vae_scale_factor) * (
                    lw * self.vae_scale_factor
                )
                ref_latents = self._batch_ref_latents_from_batch(
                    batch, latent_model_input.shape[0], target_pixels=target_pixels
                )

        # toolkit timestep (0..1000, 1000 = pure noise) -> Krea flow time t in
        # [0, 1] with t=1 = pure noise. Same convention -> straight divide.
        t = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != latent_model_input.shape[0]:
            t = t.expand(latent_model_input.shape[0])

        context, text_mask = pad_text_features(
            text_embeddings.text_embeds, self.device_torch, self.torch_dtype
        )

        pred = predict_velocity(
            self.transformer,
            latent_model_input.to(self.device_torch, self.torch_dtype),
            t,
            context,
            text_mask,
            ref_latents=ref_latents,
            isolate_refs=self.kv_cache,
        )
        return pred

    def _prep_vlm_images(self, ctrl: List[torch.Tensor]) -> List[torch.Tensor]:
        """Resize reference images for the Qwen3-VL pass.

        Downscaled (aspect-preserved, never upscaled) to fit ``vlm_max_pixels``
        total area (384^2 by default, the boogu_image_edit / ComfyUI
        TextEncodeQwenImageEditPlus budget) -- the MLLM only needs a coarse
        understanding of the reference; high-res detail flows through the VAE
        ref latents.
        """
        target = int(self.model_config.model_kwargs.get("vlm_max_pixels", 384 * 384))
        images = []
        for img in ctrl:
            if img.dim() == 4:
                img = img[0]
            img = img.to(self.device_torch)
            h, w = img.shape[1], img.shape[2]
            scale = min(1.0, math.sqrt(target / (h * w)))
            nh, nw = max(round(h * scale), 28), max(round(w * scale), 28)
            if (nh, nw) != (h, w):
                img = (
                    F.interpolate(
                        img.unsqueeze(0).float(),
                        size=(nh, nw),
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .clamp(0, 1)
                )
            images.append(img.float())
        return images

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

        # Encode each prompt at its natural length and store one (L, 12*2560)
        # tensor per batch item. The (L, 12, 2560) stack is flattened to 2D so the
        # toolkit's batching reads the list length (not the seq length) as the
        # batch size; predict_velocity restores the layer axis. Padding to the
        # batch max is deferred to the model call so caches stay small and any
        # prompts can share a batch.
        features_list = []
        for p, ctrl in zip(prompt, control_images):
            images = self._prep_vlm_images(ctrl) if ctrl is not None else None
            features = encode_krea_prompt(
                self.text_encoder,
                self.tokenizer,
                self.processor,
                p,
                max_length=self.max_text_length,
                select_layers=SELECT_LAYERS,
                images=images,
                vl_processor=self.vl_processor,
                dtype=self.torch_dtype,
            )
            # (L, n, d) -> (L, n*d)
            features = features.reshape(features.shape[0], -1)
            features_list.append(features.to(self.torch_dtype))

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
    # VAE (Qwen-Image AutoencoderKLQwenImage -- same handling as qwen_image arch)
    # ------------------------------------------------------------------
    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)

        # AutoencoderKLQwenImage is a video VAE: add a frame dim.
        images = images.unsqueeze(2)
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.squeeze(2)  # drop frame dim
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)
        latents = latents.unsqueeze(2)  # add frame dim

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        # Full-resolution decode spikes VRAM; tile it when low on VRAM (decode
        # only -- encode stays untiled).
        tiled = self.model_config.low_vram
        if tiled:
            self.vae.enable_tiling()
        try:
            images = self.vae.decode(latents).sample
        finally:
            if tiled:
                self.vae.disable_tiling()
        images = images.squeeze(2)  # drop frame dim
        return images.to(device, dtype=dtype)

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        from toolkit.util.quantize import dequantize_if_quantized

        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: SingleStreamDiT = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            # dequantize any quantized (e.g. quanto/torchao) weights so we save plain full precision tensors
            save_dict[k] = (
                dequantize_if_quantized(v).clone().to("cpu", dtype=save_dtype)
            )
        meta = get_meta_for_safetensors(meta, name="krea2")
        save_file(save_dict, output_path, metadata=meta)

    def get_base_model_version(self):
        return "krea2"

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
