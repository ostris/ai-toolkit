"""Boogu-Image base (text-to-image) integration for ai-toolkit.

Boogu-Image is a Lumina2-style mixed double-/single-stream flow-matching DiT
conditioned on Qwen3-VL instruction features. This wires up the base T2I model
for LoRA / fine-tune training and preview sampling.

Only the base text-to-image path is implemented here (no reference-image / edit
conditioning). The architecture lives under ``./src`` (vendored & trimmed from the
upstream Boogu repo); nothing is imported from the original repo.

Weights are pulled from the bf16 release ``Boogu/Boogu-Image-0.1-Base`` (clean
safetensors). The ``-fp8`` sibling ships torchao float8 ``.bin`` weights that
need a matching torchao/cache_dit to deserialize and is not supported here --
use the bf16 repo and set ``quantize: true`` to run the transformer in fp8 via
ai-toolkit's own quantization.
"""

import os
from typing import List, Optional

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file

from transformers import AutoModel, AutoProcessor

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import quantize, get_qtype, quantize_model

from optimum.quanto import freeze, QTensor
from diffusers import AutoencoderKL

from .src.transformer import BooguImageTransformer2DModel
from .src.rope import get_freqs_cis
from .src.pipeline import (
    BooguImagePipeline,
    pad_instruction_features,
    run_boogu_transformer,
)


# ai-toolkit uses CustomFlowMatchEulerDiscreteScheduler for training and (via our
# pipeline) sampling. ``shift`` warps timesteps toward the high-noise end; 3.0 is a
# reasonable high-resolution default and Boogu's own time-shift is applied in the
# preview sampler (see src/pipeline.boogu_time_schedule).
scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}

# Released weights. The "-fp8" sibling ships torchao float8 weights that need
# cache_dit/torchao to deserialize; the plain repo ships clean bf16 safetensors,
# which load directly and let ai-toolkit do its own (optional) quantization.
BOOGU_BASE_PATH = "Boogu/Boogu-Image-0.1-Base"

# System prompt the base T2I model was trained with (SYSTEM_PROMPT_4_T2I upstream).
SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on user "
    "instructions. The instructions are as follows."
)

HF_TOKEN = os.getenv("HF_TOKEN", None)


def patch_qwen_vl_patch_embed(model) -> int:
    """Swap Qwen-VL's vision ``patch_embed`` Conv3d for the equivalent ``F.linear``.

    Qwen-VL's patch_embed is a Conv3d whose kernel == stride, i.e. just a linear
    projection of each flattened patch. bf16 Conv3d has no fast cuDNN kernel and
    falls back to a slow path that effectively locks up image caching for the edit
    (TI2I) model. The weight is read lazily so this survives later ``.to()`` moves.
    Returns the number of patch_embed modules patched. (Vendored from
    extensions_built_in/captioner/Qwen3VLCaptioner.py.)
    """
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


class BooguImageModel(BaseModel):
    arch = "boogu_image"
    # Default HF repo when model.name_or_path is unset (overridden by the edit model).
    default_repo = BOOGU_BASE_PATH
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
        self.target_lora_modules = ["BooguImageTransformer2DModel"]

        self.patch_size = 2
        self.vae_scale_factor = 8
        # Safety cap on instruction token length (truncation only). Each caption is
        # encoded at its natural length and padded to the batch max at the model
        # call, so this is just an upper bound.
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", 1024)
        )

        # Lazily-built, resolution-independent rotary frequency tables.
        self._freqs_cis = None

    @property
    def text_embedding_space_version(self):
        return self.arch + "_v1"

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # 8 for the VAE downsample, 2 for the patch size.
        return self.vae_scale_factor * self.patch_size

    def get_freqs_cis(self):
        """Precompute (once) the per-axis rotary frequency tables for the model."""
        if self._freqs_cis is None:
            cfg = unwrap_model(self.model).config
            self._freqs_cis = get_freqs_cis(
                cfg.axes_dim_rope, cfg.axes_lens, theta=10000
            )
        return self._freqs_cis

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Boogu-Image model")
        base = self.model_config.name_or_path or self.default_repo

        # --- transformer ---
        # Loads the bf16 release (clean safetensors). The "-fp8" sibling ships
        # torchao float8 .bin weights that need a matching torchao/cache_dit to
        # deserialize -- use the bf16 repo and let ai-toolkit quantize if wanted.
        self.print_and_status_update("Loading transformer")
        try:
            transformer = BooguImageTransformer2DModel.from_pretrained(
                base, subfolder="transformer", torch_dtype=dtype, token=HF_TOKEN
            )
        except OSError as e:
            raise OSError(
                f"Could not load Boogu transformer safetensors from '{base}'. The "
                f"'-fp8' release ships torchao float8 .bin weights, which are not "
                f"supported here -- point model.name_or_path at the bf16 repo "
                f"'{BOOGU_BASE_PATH}' instead."
            ) from e
        transformer.eval()
        flush()

        # Attention defaults to torch SDPA ("native"); opt into Flash Attention 2
        # with model_kwargs.attention_backend: "flash" (needs the flash_attn pkg).
        attention_backend = self.model_config.model_kwargs.get(
            "attention_backend", "native"
        )
        if attention_backend != "native":
            transformer.set_attention_backend(attention_backend)

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing transformer")
            quantize_model(self, transformer)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        # --- instruction encoder (Qwen3-VL) + processor ---
        te_path = self.model_config.model_kwargs.get("text_encoder_path", base)
        te_subfolder = self.model_config.model_kwargs.get(
            "text_encoder_subfolder", "mllm"
        )
        self.print_and_status_update("Loading Qwen3-VL instruction encoder")
        processor = AutoProcessor.from_pretrained(
            te_path, subfolder="processor", token=HF_TOKEN
        )
        # AutoModel yields the inner Qwen3VLModel (the ``.model`` of the
        # *ForConditionalGeneration), whose last_hidden_state is exactly the
        # instruction feature the Boogu pipeline consumes.
        text_encoder = AutoModel.from_pretrained(
            te_path, subfolder=te_subfolder, torch_dtype=dtype, token=HF_TOKEN
        )
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        # The vision tower's bf16 Conv3d patch_embed has no fast kernel and stalls
        # image caching for the edit model -- swap it for an equivalent F.linear.
        # No-op for the base T2I model (it never runs the vision tower).
        n_patched = patch_qwen_vl_patch_embed(text_encoder)
        if n_patched:
            self.print_and_status_update(
                f"  - patched {n_patched} Qwen-VL Conv3d patch_embed -> linear"
            )
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing instruction encoder")
            text_encoder.to(self.device_torch)
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving instruction encoder to CPU")
            text_encoder.to("cpu")
        else:
            text_encoder.to(self.device_torch)
        flush()

        # --- VAE (FLUX AutoencoderKL) ---
        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base, subfolder="vae", torch_dtype=self.vae_torch_dtype, token=HF_TOKEN
        )
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)
        vae.eval()
        vae.requires_grad_(False)
        flush()

        self.noise_scheduler = BooguImageModel.get_train_scheduler()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = processor
        self.model = transformer
        self.pipeline = BooguImagePipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return BooguImagePipeline(self)

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
        latent_model_input: torch.Tensor,  # (B, 16, h, w)
        timestep: torch.Tensor,  # 0..1000 scale (1000 = pure noise)
        text_embeddings: AdvancedPromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

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

        # Model predicts clean - noise; negate to return the toolkit velocity
        # (noise - clean), matching get_loss_target / the scheduler.
        raw_velocity = run_boogu_transformer(
            self.transformer,
            latent_model_input.to(self.device_torch, self.torch_dtype),
            boogu_t,
            instr_feats,
            instr_mask,
            self.get_freqs_cis(),
        )
        return -raw_velocity

    def get_prompt_embeds(self, prompt) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        device = self.text_encoder.device

        # Encode each instruction at its natural length (no cross-sample padding);
        # padding to a common length is deferred to the model call. The system
        # prompt + chat template match the base T2I training setup.
        features_list = []
        for p in prompt:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_T2I}],
                },
                {"role": "user", "content": [{"type": "text", "text": p}]},
            ]
            inputs = self.tokenizer.apply_chat_template(
                [messages],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
                truncation=True,
                max_length=self.max_text_length,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                output = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            # (L, D) -- drop the batch dim, one tensor per prompt
            features_list.append(output.last_hidden_state[0].to(self.torch_dtype))

        return AdvancedPromptEmbeds(text_embeds=features_list)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------
    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)

        if isinstance(image_list, list):
            images = torch.stack(image_list, dim=0)
        else:
            images = image_list
        images = images.to(device, dtype=dtype)

        latents = self.vae.encode(images).latent_dist.sample()
        shift = self.vae.config["shift_factor"] or 0
        latents = (latents - shift) * self.vae.config["scaling_factor"]
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(self.vae_device_torch)

        latents = latents.to(device, dtype=dtype)
        shift = self.vae.config["shift_factor"] or 0
        latents = latents / self.vae.config["scaling_factor"] + shift
        return self.vae.decode(latents).sample

    # ------------------------------------------------------------------
    # Saving / misc
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        transformer: BooguImageTransformer2DModel = unwrap_model(self.model)
        transformer_dir = os.path.join(output_path, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)

        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to("cpu", dtype=save_dtype)
        save_file(
            save_dict,
            os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"),
        )
        # config.json so the saved transformer can be reloaded with from_pretrained.
        transformer.save_config(transformer_dir)
        with open(os.path.join(output_path, "aitk_meta.yaml"), "w") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        return "boogu_image.0.1"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["double_stream_layers", "single_stream_layers"]

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
