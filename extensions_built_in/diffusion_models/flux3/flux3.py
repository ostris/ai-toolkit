"""FLUX.3 (small, ~5.5B) for ai-toolkit.

A joint flow-matching DiT (``src/transformer.py``) that trains and samples
t2i and i2i (reference images as clean conditioning tokens).

  - image VAE: FLUX.2 packed AE, f16, 128 latent channels
  - text: Qwen3-VL-4B, 8 hidden layers stacked along the channel dim ->
    ``(B, L, 20480)``; the DiT's ``timesteps_ctx`` are zeros and there is no
    pooled vector. Embeds are stored per-sample as 2D ``(L, 20480)`` tensors
    in an ``AdvancedPromptEmbeds`` (padded to a multiple of 80 tokens -- the
    DiT has no text mask, pad tokens are part of the trained context).

``model.name_or_path`` is the model package (hub repo id or local dir):

    config.json  model.safetensors  ae.safetensors  text_encoder/  tokenizer/

Only the DiT and the image VAE come from the package. The package's
text_encoder/ and tokenizer/ are byte-identical to Qwen/Qwen3-VL-4B-Instruct,
so the text encoder loads from the Comfy-Org single file under
MODELS_PATH/text_encoders/ (downloaded there once when missing) with config
and tokenizer from the public Qwen repo.

Flow convention matches ai-toolkit exactly (t=1 noise -> t=0 clean,
target = noise - clean), so no time flips anywhere.
"""

import json
import os
from typing import TYPE_CHECKING, List, Optional

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision.transforms.functional import to_tensor

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.image_vae import Flux3ImageVAE
from .src.pipeline import (
    DEFAULT_ALPHA,
    Flux3Pipeline,
    build_text_conditioning,
    pack_i2i_image_condition,
    pack_image_latents,
)
from .src.text_encoder import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PAD_MULTIPLE,
    QWEN3_VL_4B_REPO,
    Flux3TextEncoder,
    encode_flux3_prompt,
)
from .src.transformer import Flux3Transformer, Flux3TransformerParams

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


# Static flow-matching timeshift used for the training timestep/noise
# schedule (the package's sampler alpha).
DEFAULT_TIMESTEP_SHIFT = DEFAULT_ALPHA

scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
}


class Flux3Model(BaseModel):
    arch = "flux3"

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
        self.target_lora_modules = ["Flux3Transformer"]
        self.supports_model_paths = True
        self.use_old_lokr_format = False

        self.processor = None  # Qwen3-VL AutoProcessor
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", DEFAULT_MAX_TOKENS)
        )
        self.text_pad_multiple = int(
            self.model_config.model_kwargs.get("text_pad_multiple", DEFAULT_PAD_MULTIPLE)
        )
        self.package_config = None
        self.latent_space_version = "flux3_v1"

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(
            shift=DEFAULT_TIMESTEP_SHIFT, **scheduler_config
        )

    def get_bucket_divisibility(self):
        return 16  # f8 conv + 2x2 packing

    # ------------------------------------------------------------------
    # Package files
    # ------------------------------------------------------------------
    def _package_file(self, filename: str, required: bool = True) -> Optional[str]:
        """A package file: model_kwargs ``<stem>_path`` override, else the file
        inside a local package dir, else downloaded from the hub repo."""
        override = self.model_config.model_kwargs.get(
            filename.split(".")[0] + "_path", None
        )
        if override is not None:
            if not os.path.exists(override):
                raise FileNotFoundError(f"FLUX.3 {filename} override not found: {override}")
            return override
        base = self.model_config.name_or_path
        if os.path.isdir(base):
            path = os.path.join(base, filename)
            if os.path.exists(path):
                return path
            if required:
                raise FileNotFoundError(f"FLUX.3 package file not found: {path}")
            return None
        if os.path.isfile(base) and filename == "model.safetensors":
            return base  # name_or_path points straight at the DiT file
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(
                repo_id=base, filename=filename, token=os.getenv("HF_TOKEN", None)
            )
        except Exception as e:
            if required:
                raise
            self.print_and_status_update(f" - no {filename} in {base} ({type(e).__name__})")
            return None

    def _load_package_config(self) -> Optional[dict]:
        if self.package_config is None:
            path = self._package_file("config.json", required=False)
            if path is not None:
                with open(path, "r") as f:
                    self.package_config = json.load(f)
        return self.package_config

    def _transformer_params(self) -> Flux3TransformerParams:
        cfg = self._load_package_config()
        if cfg is None:
            return Flux3TransformerParams()
        d = cfg["dit"]
        in_channels = {m: c for m, c in d["in_channels"].items() if m != "text"}
        return Flux3TransformerParams(
            in_channels=in_channels,
            sequence={f"x_{m}": m for m in in_channels},
            vec_in_dim=d.get("vec_in_dim", None),
            context_in_dim=d["in_channels"]["text"],
            hidden_size=d["hidden"],
            num_heads=d["heads"],
            depth=d["depth"],
            depth_single_blocks=d["depth_single"],
            axes_dim=list(d["axes_dim"]),
            theta=d["theta"],
            mlp_ratio=d["mlp_ratio"],
            out_modes=list(d["out_modes"]),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_transformer(self) -> Flux3Transformer:
        dtype = self.torch_dtype
        self.print_and_status_update("Loading transformer (Flux3Transformer)")
        dit_path = self._package_file("model.safetensors")
        params = self._transformer_params()
        cfg = self._load_package_config()
        if cfg is not None and cfg.get("quantization", None):
            raise ValueError(
                "FLUX.3 package variants with pre-quantized weights "
                f"({cfg['quantization']}) are not supported; use the bf16 package "
                "and let the toolkit quantize."
            )

        state_dict = load_file(dit_path)
        state_dict = {
            k: (v.to(dtype) if v.is_floating_point() else v)
            for k, v in state_dict.items()
        }
        with torch.device("meta"):
            transformer = Flux3Transformer(params)
        needed = set(transformer.state_dict().keys())
        missing = sorted(needed - set(state_dict.keys()))
        if missing:
            raise ValueError(f"FLUX.3 checkpoint missing required keys: {missing[:8]}")
        result = transformer.load_state_dict(state_dict, strict=False, assign=True)
        if result.unexpected_keys:
            self.print_and_status_update(
                f" - ignoring {len(result.unexpected_keys)} unexpected checkpoint "
                f"keys (e.g. {result.unexpected_keys[:3]})"
            )
        del state_dict
        flush()

        attn_backend = self.model_config.model_kwargs.get("attention_backend", None)
        if attn_backend is not None:
            transformer.set_attention_backend(attn_backend)
        return transformer

    def _load_text_encoder(self):
        te_path = self.model_config.te_name_or_path
        if te_path is None:
            te_path = self.model_config.model_kwargs.get(
                "text_encoder_path", QWEN3_VL_4B_REPO
            )
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")
        te_kwargs = self.component_load_kwargs("te")
        text_encoder = Flux3TextEncoder.load_model(
            te_path,
            dtype=te_kwargs["dtype"],
            subfolder="",
            qtype=te_kwargs["qtype"],  # comfy candidate ranking only
            quantize_on_load=False,
        )
        processor_source = te_path if os.path.isdir(te_path) else None
        processor = Flux3TextEncoder.load_processor(processor_source)
        processor.tokenizer.padding_side = "right"
        # text-only conditioning -- drop the vision tower
        text_encoder.drop_vision_tower()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return processor, text_encoder

    def _load_vae(self) -> Flux3ImageVAE:
        self.print_and_status_update("Loading image VAE")
        vae = Flux3ImageVAE()
        vae.load_state_dict(load_file(self._package_file("ae.safetensors")))
        vae.to(self.vae_torch_dtype).eval().requires_grad_(False)
        return vae

    def load_model(self):
        self.print_and_status_update("Loading FLUX.3 model")

        transformer = self._load_transformer()
        # quantize + offload + placement, all driven by model_config
        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))
        flush()

        processor, text_encoder = self._load_text_encoder()
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        flush()

        vae = self._load_vae()
        vae.to(self.vae_device_torch)

        self.noise_scheduler = Flux3Model.get_train_scheduler()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.model = transformer
        self.pipeline = Flux3Pipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Flux3Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Flux3Pipeline,
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

        # cond and uncond were encoded separately, so their pad-to-80 windows
        # can differ. The DiT attends to real pad-token states, not zeros, so
        # re-encode the shorter prompt at the longer window when the TE is up.
        if unconditional_embeds is not None and self.text_encoder is not None:
            cond_len = conditional_embeds.text_embeds[0].shape[0]
            uncond_len = unconditional_embeds.text_embeds[0].shape[0]
            if cond_len != uncond_len:
                target = max(cond_len, uncond_len)
                if cond_len < target:
                    conditional_embeds = self._encode_at_length(gen_config.prompt, target)
                else:
                    unconditional_embeds = self._encode_at_length(
                        gen_config.negative_prompt or "", target
                    )

        ctrl_img = None
        if gen_config.ctrl_img is not None:
            img = Image.open(gen_config.ctrl_img).convert("RGB")
            img = img.resize((gen_config.width, gen_config.height), Image.LANCZOS)
            ctrl_img = to_tensor(img).unsqueeze(0)

        alpha = self.model_config.model_kwargs.get("sample_alpha", None)
        alpha = float(alpha) if alpha is not None else None

        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            ctrl_img=ctrl_img,
            alpha=alpha,
        )
        return result[0]

    def _encode_at_length(self, prompt: str, length: int) -> AdvancedPromptEmbeds:
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        ctx = encode_flux3_prompt(
            self.text_encoder,
            self.processor,
            [prompt],
            max_length=length,
            dtype=self.torch_dtype,
            pad_multiple=None,
        )
        return AdvancedPromptEmbeds(text_embeds=[ctx[0]])

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

        with torch.no_grad():
            # toolkit timestep (0..1000, 1000 = pure noise) -> flow time in [0, 1]
            t = timestep.to(device, dtype=torch.float32) / 1000.0
            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.shape[0] != batch_size:
                t = t.expand(batch_size)

            ctx, ctx_ids, ts_ctx = build_text_conditioning(
                text_embeddings.text_embeds, device, dtype
            )

            latent_model_input = latent_model_input.to(device, dtype)
            _, _, h_lat, w_lat = latent_model_input.shape
            tokens, ids = pack_image_latents(latent_model_input)
            model_kwargs = {
                "x_image": tokens.to(dtype),
                "x_image_ids": ids,
                "x_image_timesteps": t.to(dtype)[:, None]
                * torch.ones(tokens.shape[:2], device=device, dtype=dtype),
            }

            # i2i: reference image(s) as clean t=0 tokens at l = -1, -2, ...
            control_list = None
            if batch is not None:
                if batch.control_tensor_list is not None:
                    control_list = batch.control_tensor_list
                elif batch.control_tensor is not None:
                    control_list = [[batch.control_tensor[b]] for b in range(batch_size)]
            if control_list is not None:
                ref_latents = []
                for controls in control_list:
                    sample_refs = []
                    for ctrl in controls:
                        if ctrl.ndim == 4:
                            ctrl = ctrl[0]
                        # control tensors arrive in [0, 1]
                        lat = self.encode_images(
                            ctrl.unsqueeze(0) * 2 - 1, device=device, dtype=dtype
                        )
                        sample_refs.append(lat[0])
                    ref_latents.append(sample_refs)
                xc, xc_ids, xc_ts = pack_i2i_image_condition(ref_latents, device, dtype)
                model_kwargs["x_image_cond"] = xc
                model_kwargs["x_image_cond_ids"] = xc_ids
                model_kwargs["x_image_cond_timesteps"] = xc_ts

        pred = self.transformer(
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps_ctx=ts_ctx,
            **model_kwargs,
        )
        out = pred["x_image"]
        return out.reshape(batch_size, h_lat, w_lat, -1).permute(0, 3, 1, 2)

    def get_prompt_embeds(self, prompt) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        ctx = encode_flux3_prompt(
            self.text_encoder,
            self.processor,
            prompt,
            max_length=self.max_text_length,
            dtype=self.torch_dtype,
            pad_multiple=self.text_pad_multiple,
        )
        # one 2D (L, 20480) tensor per batch item
        return AdvancedPromptEmbeds(text_embeds=[ctx[i] for i in range(ctx.shape[0])])

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
        """Images (C, H, W) in [-1, 1] -> latents (B, 128, H/16, W/16).

        Accepts a list of per-item tensors or a stacked batch tensor.
        """
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        vae_dtype = self.vae.dtype
        batch = torch.stack([img.to(device, dtype=vae_dtype) for img in image_list])
        latents = self.vae.encode(batch)
        return latents.to(device, dtype=dtype)

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
        from toolkit.util.quantize import dequantize_if_quantized

        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: Flux3Transformer = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            save_dict[k] = (
                dequantize_if_quantized(v).clone().to("cpu", dtype=save_dtype)
            )
        meta = get_meta_for_safetensors(meta, name=self.arch)
        save_file(save_dict, output_path, metadata=meta)

    def get_base_model_version(self):
        return "flux3"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return Flux3Transformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        # sensitive small modules kept in full precision (fnmatch patterns):
        #   emb_in*/txt_in       - token input projections
        #   time_in*             - timestep embedder; feeds every modulation
        #   *modulations*        - per-stream adaLN modulations
        #   final_layer*         - modulated output projections
        return [
            "emb_in*",
            "txt_in",
            "time_in*",
            "*modulations*",
            "final_layer*",
        ]

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
