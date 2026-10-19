"""MOSS-Image (OpenMOSS-Team/MOSS-Image-SFT) for ai-toolkit.

  - DiT: 42-block single-stream transformer, 3072 wide, 128 latent channels
    (``src/transformer.py``, Comfy-Org/MOSS-Image key layout)
  - VAE: FLUX.2, 2x2 packed and BatchNorm normalized (the toolkit's BFL
    ``AutoEncoder``; same packing as MOSS). The shared comfy flux2-vae file
    (full decoder) replaces the vendor's small-decoder copy by default
  - text: Qwen3.5-9B final-normed last hidden state of the chat-templated
    prompt (thinking on, generation prompt appended), one unpadded forward per
    prompt; stored per item as (L, 4096) in an ``AdvancedPromptEmbeds``

``model.name_or_path`` defaults to the vendor repo; ``Comfy-Org/MOSS-Image``
(the repack), a local diffusers dir or a ``.safetensors`` path (local or
``org/repo/file``) also load. Comfy files on disk take precedence for the
vendor ids; see ``src/checkpoints.py`` for sourcing. The text encoder is
stock Qwen3.5-9B from a comfy file.

Flow convention matches ai-toolkit (sigma 1 = noise, target = noise - clean).
"""

import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torchvision.transforms.functional import to_tensor
from transformers import AutoImageProcessor, Qwen2Tokenizer

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.text_encoders.qwen35 import (
    Qwen35TextOnlyEncoder,
    Qwen35VLEncoder,
)
from toolkit.models.v2.vae.flux2_kl import AutoEncoder, convert_diffusers_state_dict
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.checkpoints import (
    COMFY_REPO,
    QWEN35_9B_REPO,
    SFT_REPO,
    VENDOR_VAE_FILE,
    comfy_text_encoder_file,
    comfy_transformer_file,
    comfy_vae_file,
    resolve_hub_file,
)
from .src.pipeline import (
    DEFAULT_SHIFT,
    DEFAULT_UNCOND_SKIP_LAYERS,
    MossImagePipeline,
    stack_text_embeds,
)
from .src.transformer import MossImageTransformer

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

HF_TOKEN = os.getenv("HF_TOKEN", None)
MAX_TEXT_TOKENS = 4096

scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
}


def _is_hub_repo(path: str) -> bool:
    return (
        not os.path.exists(path)
        and not path.endswith(".safetensors")
        and path.count("/") == 1
    )


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _hub_tokenizer_folder(repo_id: str) -> str:
    """The repo's tokenizer/ folder in the hub cache, downloaded when any file
    is missing. Not transformers' local_files_only: on an empty cache 5.5.3
    returns a 1-token tokenizer without the template instead of raising."""
    from huggingface_hub import snapshot_download

    patterns = [f"tokenizer/{name}" for name in TOKENIZER_FILES]
    try:
        root = snapshot_download(
            repo_id, allow_patterns=patterns, token=HF_TOKEN, local_files_only=True
        )
        if all(os.path.isfile(os.path.join(root, p)) for p in patterns):
            return os.path.join(root, "tokenizer")
    except Exception:
        pass
    root = snapshot_download(repo_id, allow_patterns=patterns, token=HF_TOKEN)
    return os.path.join(root, "tokenizer")


def _hub_file_folder(repo_id: str, filename: str) -> str:
    """Folder holding one hub file, cache first (loaded from the folder so no
    other file of the repo is resolved)."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id, filename, token=HF_TOKEN, local_files_only=True)
    except Exception:
        path = hf_hub_download(repo_id, filename, token=HF_TOKEN)
    return os.path.dirname(path)


@torch.no_grad()
def encode_text_batch(text_encoder, tokenizer, texts, max_length, device) -> List[torch.Tensor]:
    """Chat-templated prompts -> per-item (L_i, D) final states from ONE forward.
    Right-padded, no attention mask: for the causal model the pads sit after
    each prompt's real tokens, so neither causal attention nor the gated-delta
    recurrence can reach them -- each item's first L_i states equal its
    unpadded forward (the HF mask path only zeroes padded states, which is
    left-padding hygiene, and would also drop attention off its fast path).
    Default arange positions are right for right padding."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        truncation=True,
        max_length=max_length,
    )
    lengths = enc["attention_mask"].sum(1).tolist()
    states = text_encoder(
        input_ids=enc["input_ids"].to(device), use_cache=False
    ).last_hidden_state
    return [states[i, :n] for i, n in enumerate(lengths)]


def _drop_repeats(images):
    """ctrl_img_1 defaults to ctrl_img, so a single reference can arrive twice."""
    kept = []
    for image in images:
        previous = kept[-1] if kept else None
        if (
            previous is not None
            and previous.shape == image.shape
            and torch.equal(previous, image)
        ):
            continue
        kept.append(image)
    return kept


class MossImageModel(BaseModel):
    arch = "moss_image"

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
        self.target_lora_modules = ["MossImageTransformer"]
        self.supports_model_paths = True
        self.use_old_lokr_format = False
        self.lora_keys_use_comfy_prefix = True
        self.latent_space_version = "moss_image"

        kw = self.model_config.model_kwargs
        self.max_text_length = int(kw.get("max_text_length", MAX_TEXT_TOKENS))
        self.sample_shift = float(kw.get("sample_shift", DEFAULT_SHIFT))
        self.uncond_skip_layers = tuple(
            kw.get("uncond_skip_layers", DEFAULT_UNCOND_SKIP_LAYERS) or ()
        )

        # model_kwargs.edit: krea2-style reference conditioning for edit LoRAs
        # (the vendor model is t2i only). References ride in the prompt through
        # the Qwen3.5 vision tower and as clean t=0 latents after the text.
        # kv_cache: train with references attending only to themselves so
        # their K/V are computed once per sample at inference.
        self.is_edit = bool(kw.get("edit", False))
        self.kv_cache = bool(kw.get("kv_cache", False))
        self.encode_control_in_text_embeddings = self.is_edit
        self.has_multiple_control_images = self.is_edit
        self.use_raw_control_images = self.is_edit
        self.image_processor = None

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(
            shift=DEFAULT_SHIFT, **scheduler_config
        )

    def get_bucket_divisibility(self):
        return 16  # f8 conv + 2x2 packing

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_transformer(self) -> MossImageTransformer:
        self.print_and_status_update("Loading transformer (MossImageTransformer)")
        base = self.model_config.name_or_path
        qtype = self.model_config.qtype if self.model_config.quantize else None
        if os.path.isdir(base):
            folder = os.path.join(base, "transformer")
            if not os.path.isdir(folder):
                folder = base
            state_dict = {}
            for name in sorted(os.listdir(folder)):
                if name.endswith(".safetensors"):
                    with safe_open(os.path.join(folder, name), "pt") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
            if not state_dict:
                raise FileNotFoundError(f"no MOSS-Image transformer weights in {folder}")
            return MossImageTransformer.load_from_state_dict(state_dict, self.torch_dtype)
        if base.endswith(".safetensors"):
            path = resolve_hub_file(
                base, hf_token=HF_TOKEN, status_fn=self.print_and_status_update
            )
        else:
            path = comfy_transformer_file(
                base,
                qtype=qtype,
                hf_token=HF_TOKEN,
                status_fn=self.print_and_status_update,
            )
        self.print_and_status_update(f" - transformer weights: {path}")
        return MossImageTransformer.load_model(
            path, dtype=self.torch_dtype, qtype=qtype, quantize_on_load=False
        )

    def _load_text_encoder(self):
        te_path = self.model_config.te_name_or_path
        if te_path is None:
            te_path = self.model_config.model_kwargs.get(
                "text_encoder_path", QWEN35_9B_REPO
            )
        te_kwargs = self.component_load_kwargs("te")
        config_path = None
        if te_path == QWEN35_9B_REPO and te_kwargs["use_comfy_weights"]:
            te_path = comfy_text_encoder_file(
                qtype=te_kwargs["qtype"],
                hf_token=HF_TOKEN,
                status_fn=self.print_and_status_update,
            )
        elif te_path.endswith(".safetensors"):
            te_path = resolve_hub_file(
                te_path, hf_token=HF_TOKEN, status_fn=self.print_and_status_update
            )
        if te_path.endswith(".safetensors"):
            config_path = QWEN35_9B_REPO
        self.print_and_status_update(f"Loading Qwen3.5 text encoder from {te_path}")
        # edit keeps the vision tower for the reference images
        te_cls = Qwen35VLEncoder if self.is_edit else Qwen35TextOnlyEncoder
        text_encoder = te_cls.load_model(
            te_path,
            dtype=te_kwargs["dtype"],
            config_path=config_path,
            subfolder="",
            qtype=te_kwargs["qtype"],  # comfy candidate ranking only
            quantize_on_load=False,
            use_comfy_weights=te_kwargs["use_comfy_weights"],
        )
        if self.is_edit:
            text_encoder.patch_vision_patch_embed()
            # MOSS ships no image processor; Qwen's (same template + vocab)
            self.image_processor = AutoImageProcessor.from_pretrained(
                _hub_file_folder(QWEN35_9B_REPO, "preprocessor_config.json")
            )
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        return text_encoder

    def _load_tokenizer(self):
        base = self.model_config.name_or_path
        source = SFT_REPO
        if os.path.isdir(os.path.join(base, "tokenizer")) or (
            _is_hub_repo(base) and base != COMFY_REPO
        ):
            source = base
        folder = os.path.join(source, "tokenizer")
        if not os.path.isdir(folder):
            folder = _hub_tokenizer_folder(source)
        # the declared class: AutoTokenizer also resolves a config.json the
        # folder lacks, which fails offline
        tokenizer = Qwen2Tokenizer.from_pretrained(folder)
        if tokenizer.chat_template is None or len(tokenizer) < 1000:
            raise ValueError(f"incomplete MOSS-Image tokenizer in {folder}")
        return tokenizer

    def _load_vae(self) -> AutoEncoder:
        self.print_and_status_update("Loading VAE")
        base = self.model_config.name_or_path
        path = self.model_config.vae_path
        if path is None and os.path.isdir(base):
            local = os.path.join(base, VENDOR_VAE_FILE)
            if os.path.exists(local):
                path = local
        if path is None:
            path = comfy_vae_file(
                hf_token=HF_TOKEN, status_fn=self.print_and_status_update
            )
        elif not os.path.exists(path):
            path = resolve_hub_file(
                path, hf_token=HF_TOKEN, status_fn=self.print_and_status_update
            )
        state_dict = load_file(path)
        if any(k.startswith("encoder.down_blocks.") for k in state_dict):
            state_dict = convert_diffusers_state_dict(state_dict)
        # config sniffed from the checkpoint (small decoder)
        vae = AutoEncoder.load_from_state_dict(state_dict, self.vae_torch_dtype)
        return vae.eval().requires_grad_(False)

    def load_model(self):
        self.print_and_status_update("Loading MOSS-Image model")

        transformer = self._load_transformer()
        # quantize + offload + placement, all driven by model_config
        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))
        flush()

        text_encoder = self._load_text_encoder()
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        flush()
        tokenizer = self._load_tokenizer()

        vae = self._load_vae()
        vae.to(self.vae_device_torch)

        self.noise_scheduler = MossImageModel.get_train_scheduler()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = transformer
        self.pipeline = MossImagePipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return MossImagePipeline(self)

    def generate_single_image(
        self,
        pipeline: MossImagePipeline,
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

        # reference latents; the prompt embeds already carry the references
        # through the vision tower (ctrl_img_1 mirrors ctrl_img when unset)
        ref_latents = None
        if self.is_edit:
            ctrl_paths = []
            for path in (
                gen_config.ctrl_img,
                gen_config.ctrl_img_1,
                gen_config.ctrl_img_2,
                gen_config.ctrl_img_3,
            ):
                if path is not None and path not in ctrl_paths:
                    ctrl_paths.append(path)
            if ctrl_paths:
                ref_latents = [
                    self._encode_ref_latents(
                        [to_tensor(Image.open(p).convert("RGB")) for p in ctrl_paths],
                        target_pixels=gen_config.width * gen_config.height,
                    )
                ]

        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            shift=self.sample_shift,
            uncond_skip_layers=self.uncond_skip_layers,
            ref_latents=ref_latents,
            isolate_refs=self.kv_cache,
        )
        return result[0]

    # ------------------------------------------------------------------
    # Reference images (edit)
    # ------------------------------------------------------------------
    def _ref_target_pixels(self, target_pixels: Optional[int]) -> int:
        """Area each reference is fitted to: control_image_max_pixels (1 MP
        cap, only scales down), or the target's area with match_target_res."""
        kw = self.model_config.model_kwargs
        if kw.get("match_target_res", False) and target_pixels:
            return int(target_pixels)
        return int(kw.get("control_image_max_pixels", 1024 * 1024))

    def _encode_ref_latents(
        self, control_tensors, target_pixels: Optional[int] = None
    ) -> List[torch.Tensor]:
        """[0, 1] (C, H, W) / (1, C, H, W) references -> (128, h, w) latents,
        aspect kept, snapped to the 16 px grid."""
        sc = self.get_bucket_divisibility()
        budget = self._ref_target_pixels(target_pixels)
        match = self.model_config.model_kwargs.get("match_target_res", False)
        latents = []
        for img in control_tensors:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(self.device_torch, dtype=torch.float32)
            h, w = img.shape[2], img.shape[3]
            if match or h * w > budget:
                ratio = h / w
                new_h = math.sqrt(budget * ratio)
                new_w = new_h / ratio
            else:
                new_h, new_w = float(h), float(w)
            new_h = max(sc, int(round(new_h / sc)) * sc)
            new_w = max(sc, int(round(new_w / sc)) * sc)
            if (new_h, new_w) != (h, w):
                img = F.interpolate(img, size=(new_h, new_w), mode="bilinear")
            latent = self.encode_images(
                img * 2 - 1, device=self.device_torch, dtype=self.torch_dtype
            )
            latents.append(latent[0])
        return latents

    def _batch_ref_latents(
        self, batch: "DataLoaderBatchDTO", batch_size: int, target_pixels: int
    ) -> Optional[List[List[torch.Tensor]]]:
        control_list = batch.control_tensor_list
        if control_list is None and batch.control_tensor is not None:
            control_list = [batch.control_tensor[b : b + 1] for b in range(batch_size)]
        if control_list is None:
            return None
        if len(control_list) != batch_size:
            raise ValueError("control tensor list length does not match batch size")
        return [
            self._encode_ref_latents(controls, target_pixels=target_pixels)
            for controls in control_list
        ]

    def _prep_vlm_images(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """References for the vision tower: downscaled (never up) to
        vlm_max_pixels (384^2 = 144 tokens); detail comes from the latents."""
        target = int(self.model_config.model_kwargs.get("vlm_max_pixels", 384 * 384))
        out = []
        for img in images:
            if img.dim() == 4:
                img = img[0]
            img = img.to(self.device_torch, dtype=torch.float32)
            h, w = img.shape[1], img.shape[2]
            scale = min(1.0, math.sqrt(target / (h * w)))
            nh, nw = max(round(h * scale), 32), max(round(w * scale), 32)
            if (nh, nw) != (h, w):
                img = F.interpolate(
                    img[None], size=(nh, nw), mode="bicubic", antialias=True
                )[0].clamp(0, 1)
            out.append(img)
        return out

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
        device = self.device_torch
        dtype = self.torch_dtype
        if self.model.device == torch.device("cpu"):
            self.model.to(device)

        batch_size = latent_model_input.shape[0]
        embeds_list = text_embeddings.text_embeds
        if len(embeds_list) == 1 and batch_size > 1:
            embeds_list = embeds_list * batch_size
        with torch.no_grad():
            embeds, mask = stack_text_embeds(embeds_list, device, dtype)
            t = timestep.to(device, dtype=torch.float32).reshape(-1)
            if t.shape[0] != batch_size:
                t = t.expand(batch_size)
            ref_latents = None
            if self.is_edit and batch is not None:
                _, _, lh, lw = latent_model_input.shape
                ref_latents = self._batch_ref_latents(
                    batch, batch_size, target_pixels=lh * lw * 16 * 16
                )

        return self.transformer(
            latent_model_input.to(device, dtype),
            t,
            embeds,
            mask,
            ref_latents=ref_latents,
            isolate_refs=self.kv_cache,
        )

    def _chat(self, content: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _encode_with_images(self, text: str, images: List[torch.Tensor]) -> torch.Tensor:
        """Prompt with references ahead of it (``Picture N: <vision>``, the
        Qwen-Image-Edit-Plus layout) -> (L, 4096) final states."""
        pixels = self.image_processor(
            images=self._prep_vlm_images(images), return_tensors="pt", do_rescale=False
        )
        grid = pixels["image_grid_thw"]
        counts = (grid.prod(-1) // self.image_processor.merge_size**2).tolist()
        content = "".join(
            f"Picture {i + 1}: <|vision_start|>{'<|image_pad|>' * n}<|vision_end|>"
            for i, n in enumerate(counts)
        )
        # no truncation: the image-pad runs must stay whole
        input_ids = self.tokenizer(self._chat(content + text), return_tensors="pt")[
            "input_ids"
        ].to(self.device_torch)
        mm_token_type_ids = (input_ids == self.text_encoder.config.image_token_id).int()
        if int(mm_token_type_ids.sum()) != sum(counts):
            raise ValueError("image placeholder count does not match the image grids")
        out = self.text_encoder(
            input_ids=input_ids,
            pixel_values=pixels["pixel_values"].to(self.device_torch, self.torch_dtype),
            image_grid_thw=grid.to(self.device_torch),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )
        return out.last_hidden_state[0]

    @torch.no_grad()
    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        # control images -> one list per prompt: a (B, C, H, W) batch, a list of
        # per-sample lists, or a flat list for a single prompt
        if control_images is not None and self.is_edit:
            if isinstance(control_images, torch.Tensor):
                control_images = [[control_images[i]] for i in range(control_images.shape[0])]
            elif len(control_images) > 0 and not isinstance(control_images[0], list):
                control_images = [control_images]
            if len(control_images) == 1 and len(prompt) > 1:
                control_images = control_images * len(prompt)
            if len(control_images) != len(prompt):
                raise ValueError("number of prompts must match number of control image sets")
            control_images = [_drop_repeats(c) if c else None for c in control_images]
        else:
            control_images = [None] * len(prompt)

        # text-only prompts go through ONE right-padded forward (exact, see
        # encode_text_batch); prompts with references stay one forward each
        # (their image-pad runs differ per prompt)
        embeds: List[Optional[torch.Tensor]] = [None] * len(prompt)
        text_only = [i for i, images in enumerate(control_images) if not images]
        if text_only:
            states = encode_text_batch(
                self.text_encoder,
                self.tokenizer,
                [self._chat(prompt[i]) for i in text_only],
                self.max_text_length,
                self.device_torch,
            )
            for i, st in zip(text_only, states):
                embeds[i] = st.to(self.torch_dtype)
        for i, (text, images) in enumerate(zip(prompt, control_images)):
            if images:
                embeds[i] = self._encode_with_images(text, images).to(self.torch_dtype)
        return AdvancedPromptEmbeds(text_embeds=embeds)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None):
        """Images (C, H, W) in [-1, 1] -> normalized latents (B, 128, H/16, W/16)."""
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        batch = torch.stack([img.to(device, dtype=self.vae.dtype) for img in image_list])
        return self.vae.encode(batch).to(device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        images = self.vae.decode(latents.to(device, self.vae.dtype))
        return images.to(device, dtype)

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: MossImageTransformer = unwrap_model(self.model)
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
            quantized=False,
        )

    # LoRA keys: the module fuses q/k/v (like the Comfy files), but ComfyUI's
    # MOSS LoRA convention addresses the separate projections (mapped onto
    # fused files by row slices, and the only names unfused files have).
    # Saved as q/k/v, fused back on load.
    _QKV_SPLIT_ROWS = ("lora_B.weight", "lora_up.weight")
    _QKV_SHARED = ("lora_A.weight", "lora_down.weight", "alpha")

    def convert_lora_weights_before_save(self, state_dict):
        state_dict = super().convert_lora_weights_before_save(state_dict)
        out = {}
        for key, value in state_dict.items():
            base, sep, suffix = key.partition(".proj.qkv.")
            if not sep:
                out[key] = value
            elif suffix in self._QKV_SPLIT_ROWS:
                for name, part in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                    out[f"{base}.proj.{name}.{suffix}"] = part.contiguous()
            elif suffix in self._QKV_SHARED:
                for name in ("q", "k", "v"):
                    out[f"{base}.proj.{name}.{suffix}"] = value.clone()
            else:
                out[key] = value  # e.g. LoKr: kept fused (maps on fused files)
        return out

    def convert_lora_weights_before_load(self, state_dict):
        state_dict = super().convert_lora_weights_before_load(state_dict)
        out = dict(state_dict)
        down_keys = [
            k for k in state_dict
            if k.endswith((".proj.q.lora_A.weight", ".proj.q.lora_down.weight"))
        ]
        for q_down in down_keys:
            base, _, down_suffix = q_down.partition(".proj.q.")
            up_suffix = "lora_B.weight" if down_suffix == "lora_A.weight" else "lora_up.weight"
            names = [f"{base}.proj.{p}." for p in ("q", "k", "v")]
            if not all(n + down_suffix in state_dict and n + up_suffix in state_dict for n in names):
                continue
            downs = [out.pop(n + down_suffix) for n in names]
            ups = [out.pop(n + up_suffix) for n in names]
            alphas = [out.pop(n + "alpha", None) for n in names]
            rank = downs[0].shape[0]
            if all(torch.equal(downs[0], d) for d in downs[1:]):
                # shared down projection (our saves): the fused rank-r LoRA
                down, up = downs[0], torch.cat(ups, dim=0)
            else:
                # independent q/k/v LoRAs: exact rank-3r block form
                down = torch.cat(downs, dim=0)
                up = torch.block_diag(*ups)
            out[f"{base}.proj.qkv.{down_suffix}"] = down
            out[f"{base}.proj.qkv.{up_suffix}"] = up
            if alphas[0] is not None:
                # keep alpha / rank, the per-projection scale
                out[f"{base}.proj.qkv.alpha"] = alphas[0] * (down.shape[0] / rank)
        return out

    def get_base_model_version(self):
        return "moss_image"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return MossImageTransformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        return MossImageTransformer.get_quantization_exclude_modules()
