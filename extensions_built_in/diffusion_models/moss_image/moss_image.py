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

import os
from typing import TYPE_CHECKING, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen2Tokenizer

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.text_encoders.qwen35 import Qwen35TextOnlyEncoder
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
        text_encoder = Qwen35TextOnlyEncoder.load_model(
            te_path,
            dtype=te_kwargs["dtype"],
            config_path=config_path,
            subfolder="",
            qtype=te_kwargs["qtype"],  # comfy candidate ranking only
            quantize_on_load=False,
            use_comfy_weights=te_kwargs["use_comfy_weights"],
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
        # the tokenizer's declared class: AutoTokenizer also resolves a
        # config.json the folder lacks, which fails offline
        return Qwen35TextOnlyEncoder._local_first(
            Qwen2Tokenizer.from_pretrained, source, subfolder="tokenizer"
        )

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
        )
        return result[0]

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

        return self.transformer(
            latent_model_input.to(device, dtype), t, embeds, mask
        )

    @torch.no_grad()
    def get_prompt_embeds(self, prompt) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        embeds = []
        # one unpadded forward per prompt, like the vendor (padding would leak
        # into the gated-delta recurrence)
        for text in prompt:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            input_ids = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_text_length,
            )["input_ids"].to(self.device_torch)
            out = self.text_encoder(input_ids=input_ids, use_cache=False)
            embeds.append(out.last_hidden_state[0].to(self.torch_dtype))
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

    def get_base_model_version(self):
        return "moss_image"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return MossImageTransformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        return MossImageTransformer.get_quantization_exclude_modules()
