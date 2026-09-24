"""Ming-Image 0.1 Design for ai-toolkit.

A 6B Z-Image-style DiT (`src/transformer.py`) driven by a Ling-mini-2.0 MoE
multimodal LLM (`src/text_encoder.py`) and an RGBA Qwen-Image VAE. Text and
an optional reference image go through the LLM; the DiT receives two caption
streams (the 256 query-token states through the Qwen2 connector, and the
"direct" LLM states of every prompt token) and, when editing, the reference's
VAE latent as a second frame.

Weights come from what `name_or_path` names (`src/checkpoints.py`): by
default the ComfyUI single-file repack, whose int8_convrot files attach to the
toolkit's convrot8 backend as-is (local copies win over the download), or the
vendor's diffusers-style repo (`transformer/`, `vae/`, `mllm/`, `mlp/`,
`connector/`) / a local checkpoint / a fine-tune loaded exactly as named. A
full fine-tune saves the transformer as one .safetensors in the ComfyUI key
layout, which ComfyUI and this class both load, the repack supplying the rest.

Flow-matching convention matches ai-toolkit (t=1000 noise -> 0 clean, target =
noise - clean); the DiT's own time/prediction convention is handled in
`src/pipeline.py`.
"""

import os
from typing import TYPE_CHECKING, List, Optional

import huggingface_hub
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.checkpoints import BASE_REPO, COMFY_REPO, canonical_repo
from .src.pipeline import (
    PIXELS_PER_TOKEN,
    VAE_SCALE_FACTOR,
    MingImagePipeline,
    ming_shift_params,
    run_transformer,
)
from .src.text_encoder import MingImagePromptEncoder, MingImageTextEncoder
from .src.transformer import MingImageTransformer2DModel
from .src.vae import MingImageVAE

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


# scheduler/scheduler_config.json ships shift=6.0 without dynamic shifting;
# the reference pipeline forces dynamic shifting with the Z-Image defaults
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 6.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "time_shift_type": "exponential",
}


class MingImageFlowMatchScheduler(CustomFlowMatchEulerDiscreteScheduler):
    """The toolkit scheduler with the reference pipeline's shift rule: the
    max-shift endpoint moves with the image size (see `ming_shift_params`),
    so 1024^2 and larger train at the same mu (1.35) they sample at."""

    def set_train_timesteps(self, num_timesteps, device, timestep_type="linear", latents=None, patch_size=1):
        if latents is not None and self.config.use_dynamic_shifting:
            image_seq_len = latents.shape[2] * latents.shape[3] // (patch_size**2)
            max_shift, max_seq_len = ming_shift_params(image_seq_len)
            self.register_to_config(max_shift=max_shift, max_image_seq_len=max_seq_len)
        return super().set_train_timesteps(
            num_timesteps, device, timestep_type=timestep_type, latents=latents, patch_size=patch_size
        )

# decode above this many output pixels goes through the VAE's tiled path
TILE_DECODE_ABOVE_PIXELS = 1024 * 1024


class MingImageModel(BaseModel):
    arch = "ming_image"

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
        self.use_old_lokr_format = False
        self.target_lora_modules = ["MingImageTransformer2DModel"]
        self.vae_scale_factor = VAE_SCALE_FACTOR
        self.prompt_encoder: Optional[MingImagePromptEncoder] = None
        self.image_processor = None

        # Editing is what happens when the dataset has a control path: the one
        # reference image rides into the LLM as vision tokens and its latent
        # into the DiT as a second frame. The dataloader sizes it to the item's
        # bucket, which is also the size the DiT needs it at.
        self.encode_control_in_text_embeddings = True
        self.has_multiple_control_images = False
        self.use_raw_control_images = False

    @staticmethod
    def get_train_scheduler():
        return MingImageFlowMatchScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # 8 for the VAE, 2 for the DiT's 2x2 latent patches
        return PIXELS_PER_TOKEN

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Ming-Image model")
        model_path = canonical_repo(self.model_config.name_or_path)
        extras_path = canonical_repo(self.model_config.extras_name_or_path)
        if extras_path == model_path and model_path.endswith(".safetensors"):
            # a single transformer file (a fine-tune): the rest comes from
            # the repack
            extras_path = COMFY_REPO
        # only the vendor layout carries the configs, tokenizer and processor
        config_path = extras_path if os.path.isdir(extras_path) else BASE_REPO

        self.print_and_status_update("Loading transformer")
        transformer = MingImageTransformer2DModel.load_model(
            model_path,
            dtype=dtype,
            config_path=config_path,
            use_comfy_weights=self.model_config.model_kwargs.get("use_comfy_weights", True),
            qtype=self.model_config.qtype if self.model_config.quantize else None,
            quantize_on_load=False,
        )
        load_kwargs = self.component_load_kwargs("transformer")
        if self.model_config.assistant_lora_path is None:
            transformer.aitk_post_load(**load_kwargs)
        else:
            # the adapter stays a live LoRA on the quantized linears, attached
            # after quantization (it must wrap the quantized forward) and
            # before layer offloading (the offloader routes through it)
            offload = load_kwargs.pop("offload", 0.0)
            transformer.aitk_post_load(offload=0.0, **load_kwargs)
            self.load_training_adapter(transformer)
            if offload and offload > 0:
                from toolkit.memory_management import MemoryManager

                MemoryManager.attach(
                    transformer,
                    torch.device(load_kwargs["quantize_device"]),
                    offload_percent=offload,
                    ignore_modules=list(transformer.get_offload_ignore_modules() or []),
                )
        flush()

        self.print_and_status_update("Loading text encoder (MLLM + connector)")
        tokenizer, image_processor = MingImageTextEncoder.load_tokenizer_and_processor(
            config_path
        )
        text_encoder = MingImageTextEncoder.load(
            extras_path, config_path=config_path, **self.component_load_kwargs("te")
        )
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        flush()

        self.print_and_status_update("Loading VAE")
        vae = MingImageVAE.load(
            extras_path, config_path=config_path, **self.component_load_kwargs("vae")
        )
        vae.requires_grad_(False)
        vae.eval()

        self.noise_scheduler = MingImageModel.get_train_scheduler()

        self.vae = vae
        self.text_encoder = [text_encoder]
        self.tokenizer = [tokenizer]
        self.image_processor = image_processor
        self.model = transformer
        self.prompt_encoder = MingImagePromptEncoder(text_encoder, tokenizer, image_processor)
        self.pipeline = MingImagePipeline(self)
        self.print_and_status_update("Model Loaded")

    def load_training_adapter(self, transformer: MingImageTransformer2DModel):
        """Attach the training adapter (a LoRA on the DiT) as a live assistant:
        on at 1.0 while training, off while sampling so samples show the base
        model. It is never merged: the weights are int8, whose grid swallows a
        delta this small (~1e-4 against a ~6e-3 step) on a requantize."""
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
                lora_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
                self.model_config.assistant_lora_path = lora_path
            except Exception as e:
                raise ValueError(f"Failed to download assistant LoRA from {lora_path}: {e}")
        lora_state_dict = load_file(lora_path)
        dim = int(lora_state_dict["diffusion_model.layers.0.attention.to_k.lora_A.weight"].shape[0])
        lora_state_dict = {
            key.replace("diffusion_model.", "transformer."): value for key, value in lora_state_dict.items()
        }

        network_config = NetworkConfig(type="lora", linear=dim, linear_alpha=dim, transformer_only=True)
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
            # transformer_only filters to the block list only through the holder
            base_model=self,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network.eval()
        network.requires_grad_(False)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)
        self.assistant_lora: LoRASpecialNetwork = network
        self.assistant_lora.multiplier = 1.0
        self.assistant_lora.is_active = True
        self.invert_assistant_lora = False

    # ------------------------------------------------------------------
    # VAE. Four channels in and out: images without alpha get an opaque one on
    # encode, and decode drops it again unless RGBA output is on.
    # ------------------------------------------------------------------
    @property
    def load_rgba(self) -> bool:
        return bool(self.model_config.model_kwargs.get("rgba", False))

    def _to_rgba(self, images: torch.Tensor) -> torch.Tensor:
        """`(B, 3|4, H, W)` in [-1, 1] -> four channels (opaque alpha added)."""
        if images.shape[1] == 3:
            images = torch.cat([images, torch.ones_like(images[:, :1])], dim=1)
        return images

    def encode_images(self, image_list, device=None, dtype=None):
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()

        images = torch.stack([image.to(device, dtype=dtype) for image in image_list])
        images = self._to_rgba(images).unsqueeze(2)  # single-frame dim

        # mode(), not sample(): this VAE's posterior std is ~1 in a latent
        # space whose signal std is ~0.15, so sampling buries the image in noise
        latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents.squeeze(2).to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        images = self._decode_rgba(latents, device=device, dtype=dtype)
        if not self.load_rgba:
            images = images[:, :3]
        return images

    def _decode_rgba(self, latents: torch.Tensor, device=None, dtype=None):
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype).unsqueeze(2)
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor

        # the recommended 2048x2048 output is too much for a one-shot decode
        # on a 32 GB card; tile above 1 MP, and whenever low_vram is set
        pixels = latents.shape[-2] * latents.shape[-1] * self.vae_scale_factor**2
        tiled = self.model_config.low_vram or pixels > TILE_DECODE_ABOVE_PIXELS
        if tiled:
            self.vae.enable_tiling(
                tile_sample_min_height=1024,
                tile_sample_min_width=1024,
                tile_sample_stride_height=768,
                tile_sample_stride_width=768,
            )
        try:
            images = self.vae.decode(latents).sample
        finally:
            if tiled:
                self.vae.disable_tiling()
        return images.squeeze(2).to(device, dtype=dtype)

    def decode_to_images(self, latents: torch.Tensor) -> List[Image.Image]:
        """Decode to PIL, keeping the alpha channel when load_rgba is set."""
        return [
            self.image_tensor_to_pil(image) for image in self.decode_latents(latents)
        ]

    @staticmethod
    def image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
        """`(C, H, W)` in [-1, 1] -> PIL (RGB, or RGBA when C is 4)."""
        array = (image.float().clamp(-1, 1) / 2 + 0.5).permute(1, 2, 0)
        array = (array.cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGBA" if array.shape[2] == 4 else "RGB")

    # ------------------------------------------------------------------
    # Reference images
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_control_images(control_images, batch_size: int) -> List[List[torch.Tensor]]:
        """Any shape the toolkit hands over -> one list (0 or 1 tensor, `(1, C, H, W)`
        in [0, 1]) per batch item."""
        if control_images is None:
            return [[] for _ in range(batch_size)]
        if isinstance(control_images, torch.Tensor):
            if control_images.dim() == 5:
                samples = [list(sample) for sample in control_images]
            elif control_images.dim() == 4:
                samples = [[sample] for sample in control_images]
            else:
                samples = [[control_images]]
        elif len(control_images) > 0 and not isinstance(control_images[0], list):
            samples = [list(control_images)]
        else:
            samples = [list(sample) for sample in control_images]
        samples = [[img.unsqueeze(0) if img.dim() == 3 else img for img in sample] for sample in samples]
        if len(samples) == 1 and batch_size > 1:
            samples = samples * batch_size
        if len(samples) != batch_size:
            raise ValueError(f"got {len(samples)} control image sets for {batch_size} prompts")
        for sample in samples:
            if len(sample) > 1:
                raise ValueError("Ming-Image takes a single reference image per prompt")
        return samples

    @staticmethod
    def tensor_to_pil(image: torch.Tensor) -> Image.Image:
        """`(1, C, H, W)` in [0, 1] -> RGB PIL for the vision tower (alpha dropped)."""
        image = image[0, :3].detach().float().clamp(0, 1).cpu()
        array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def encode_reference_latents(self, images: List[Optional[torch.Tensor]], height: int, width: int):
        """Per-item `(1, C, H, W)` references in [0, 1] (or None) -> per-item
        `(C, h, w)` latents at the target size (or None)."""
        out = []
        for image in images:
            if image is None:
                out.append(None)
                continue
            image = image.to(self.device_torch, dtype=torch.float32)
            if image.shape[-2:] != (height, width):
                image = F.interpolate(image, size=(height, width), mode="bicubic", antialias=True).clamp(0, 1)
            latent = self.encode_images(
                [image[0] * 2 - 1], device=self.device_torch, dtype=self.torch_dtype
            )
            out.append(latent[0])
        return out

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder[0].device != self.device_torch:
            self.text_encoder[0].to(self.device_torch)

        samples = self._normalize_control_images(control_images, len(prompt))
        images = [[self.tensor_to_pil(image) for image in sample] for sample in samples]

        query, direct, has_reference = self.prompt_encoder.encode(prompt, images=images)
        pe = AdvancedPromptEmbeds(
            text_embeds=[q.to(self.torch_dtype) for q in query],
            direct_embeds=[d.to(self.torch_dtype) for d in direct],
            has_reference=[torch.tensor([flag], dtype=torch.bool) for flag in has_reference],
        )
        # bookkeeping, not activations: a .to(dtype) must not turn it into bf16
        pe.frozen_dtype_keys = ["has_reference"]
        return pe

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def condition_noisy_latents(self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"):
        # the reference joins the DiT sequence in get_noise_prediction, clean
        return latents.detach()

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
        batch_size = latent_model_input.shape[0]

        query = list(text_embeddings.text_embeds)
        direct = list(text_embeddings.direct_embeds)
        flags = [bool(f.reshape(-1)[0]) for f in text_embeddings.has_reference]
        if len(flags) == 1 and batch_size > 1:
            flags = flags * batch_size

        # The prompt decides: an embedding encoded with a reference expects the
        # reference frame; one encoded without (plain T2I, a dropped caption)
        # takes none, whatever the batch carries.
        ref_latents = None
        if batch is not None and any(flags):
            control = batch.control_tensor_list
            if control is None:
                control = batch.control_tensor
            if control is None:
                raise ValueError(
                    "the prompt embeddings were encoded with a reference image but the batch has none"
                )
            with torch.no_grad():
                samples = self._normalize_control_images(control, batch_size)
                images = [sample[0] if (flag and sample) else None for sample, flag in zip(samples, flags)]
                ref_latents = self.encode_reference_latents(
                    images,
                    latent_model_input.shape[2] * VAE_SCALE_FACTOR,
                    latent_model_input.shape[3] * VAE_SCALE_FACTOR,
                )

        return run_transformer(
            self.transformer,
            latent_model_input.to(self.device_torch, self.torch_dtype),
            timestep,
            query,
            direct,
            ref_latents=ref_latents,
            **kwargs,
        )

    def get_loss_target(self, *args, **kwargs):
        # flow-matching velocity target: noise - clean
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return MingImagePipeline(self)

    def generate_single_image(
        self,
        pipeline: MingImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        divisor = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // divisor * divisor)
        gen_config.height = int(gen_config.height // divisor * divisor)

        reference_latents = None
        if gen_config.ctrl_img is not None:
            # same channels the dataloader gives training references
            mode = "RGBA" if self.load_rgba else "RGB"
            image = (
                torch.from_numpy(
                    np.array(Image.open(gen_config.ctrl_img).convert(mode), dtype=np.float32) / 255.0
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            reference_latents = self.encode_reference_latents(
                [image], gen_config.height, gen_config.width
            )

        return pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            reference_latents=reference_latents,
        )[0]

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        # one .safetensors in the ComfyUI key layout (fused qkv, root-level
        # x_embedder / final_layer); pre-quantized layers keep their storage,
        # so ComfyUI and this arch both load the file
        transformer: MingImageTransformer2DModel = unwrap_model(self.model)
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
        )

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def get_quantization_exclude_modules(self):
        return MingImageTransformer2DModel.get_quantization_exclude_modules()

    lora_keys_use_comfy_prefix = True
