"""Fibo 2.0 dev (briaai/Fibo-2.0-dev) for ai-toolkit, text-to-image and edit.

  - DiT: 44 Z-Image blocks, 3584 wide, a Perceiver turning the text into
    256-512 gist tokens and five cross-attention injection blocks
    (``src/transformer.py``, Comfy-Org/Fibo-2.0 key layout)
  - VAE: FLUX.2, 2x2 packed and BatchNorm normalized (the toolkit's BFL
    ``AutoEncoder``, the shared comfy flux2-vae file)
  - text: Qwen3-VL-4B, six RMS-normalized bundles of three hidden layers;
    stored per item as (L, 46080) in an ``AdvancedPromptEmbeds``

Editing is not a separate model: control images in the dataset (or sample
ctrl_img) make it an edit. References ride into the prompt through Qwen3-VL's
vision tower and into the DiT as clean t = 0 latents; without them it is
text-to-image. model_kwargs.match_target_res (default true) sizes references
against the target, see ``src/references.py``.

``model.name_or_path`` defaults to the Comfy-Org/Fibo-2.0 repack (int8 convrot
for the default convrot8 qtype, bf16 otherwise); a comfy-layout
``.safetensors`` (local or ``org/repo/file``) loads as named. Prompts are
structured JSON captions (cleaned and compacted as in training) or plain text.

Flow convention matches ai-toolkit (sigma 1 = noise, target = noise - clean).
"""

import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoImageProcessor

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.vae.flux2_kl import AutoEncoder, convert_diffusers_state_dict
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.checkpoints import (
    QWEN3_VL_4B_REPO,
    comfy_text_encoder_file,
    comfy_transformer_file,
    comfy_vae_file,
    resolve_hub_file,
)
from .src.pipeline import Fibo2Pipeline, scheduler_config, stack_text_embeds
from .src.references import drop_repeats, prepare_references, vlm_image
from .src.text_encoder import MAX_TEXT_TOKENS, Fibo2TextEncoder, encode_prompts
from .src.transformer import Fibo2Transformer

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

HF_TOKEN = os.getenv("HF_TOKEN", None)


class Fibo2Model(BaseModel):
    arch = "fibo2"

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
        self.target_lora_modules = ["Fibo2Transformer"]
        self.supports_model_paths = True
        self.use_old_lokr_format = False
        self.lora_keys_use_comfy_prefix = True
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", MAX_TEXT_TOKENS)
        )
        # edit when the dataset has control images, text-to-image otherwise
        self.encode_control_in_text_embeddings = True
        self.has_multiple_control_images = True
        # references are sized in prepare_references, identically for the
        # cached prompt and the training latents
        self.use_raw_control_images = True
        self.image_processor = None

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16  # f8 conv + 2x2 packing

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_transformer(self) -> Fibo2Transformer:
        self.print_and_status_update("Loading transformer (Fibo2Transformer)")
        qtype = self.model_config.qtype if self.model_config.quantize else None
        path = comfy_transformer_file(
            self.model_config.name_or_path,
            qtype=qtype,
            hf_token=HF_TOKEN,
            status_fn=self.print_and_status_update,
        )
        self.print_and_status_update(f" - transformer weights: {path}")
        return Fibo2Transformer.load_model(
            path, dtype=self.torch_dtype, qtype=qtype, quantize_on_load=False
        )

    def _load_text_encoder(self) -> Fibo2TextEncoder:
        te_kwargs = self.component_load_kwargs("te")
        te_path = self.model_config.te_name_or_path
        if te_path is None:
            te_path = comfy_text_encoder_file(
                qtype=te_kwargs["qtype"],
                hf_token=HF_TOKEN,
                status_fn=self.print_and_status_update,
            )
        elif te_path.endswith(".safetensors"):
            te_path = resolve_hub_file(
                te_path, hf_token=HF_TOKEN, status_fn=self.print_and_status_update
            )
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")
        text_encoder = Fibo2TextEncoder.load_model(
            te_path,
            dtype=te_kwargs["dtype"],
            config_path=QWEN3_VL_4B_REPO if te_path.endswith(".safetensors") else None,
            subfolder="",
            qtype=te_kwargs["qtype"],
            quantize_on_load=False,
            use_comfy_weights=False,
        )
        text_encoder.patch_vision_patch_embed()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        return text_encoder

    def _load_vae(self) -> AutoEncoder:
        self.print_and_status_update("Loading VAE")
        path = self.model_config.vae_path
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
        vae = AutoEncoder.load_from_state_dict(state_dict, self.vae_torch_dtype)
        return vae.eval().requires_grad_(False)

    def load_model(self):
        self.print_and_status_update("Loading Fibo 2.0 model")

        transformer = self._load_transformer()
        # quantize + offload + placement, all driven by model_config
        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))
        flush()

        text_encoder = self._load_text_encoder()
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        flush()
        tokenizer = Fibo2TextEncoder.load_tokenizer()
        self.image_processor = Fibo2TextEncoder._local_first(
            AutoImageProcessor.from_pretrained, QWEN3_VL_4B_REPO
        )

        vae = self._load_vae()
        vae.to(self.vae_device_torch)

        self.noise_scheduler = Fibo2Model.get_train_scheduler()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = transformer
        self.pipeline = Fibo2Pipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Fibo2Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Fibo2Pipeline,
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

        # the same references the sampler encoded into the prompt
        paths = []
        for path in (
            gen_config.ctrl_img,
            gen_config.ctrl_img_1,
            gen_config.ctrl_img_2,
            gen_config.ctrl_img_3,
        ):
            if path is not None and path not in paths:
                paths.append(path)
        ref_latents = None
        if paths:
            crops = prepare_references(
                [TF.to_tensor(Image.open(p).convert("RGB")) for p in paths],
                (gen_config.width, gen_config.height),
                self.match_target_res,
            )
            ref_latents = self._encode_crops(crops)
            # the vendor's negative is text only (the references still
            # condition both passes); the sampler encoded it with them
            if gen_config.negative_prompt.strip() and self.sample_prompts_cache is None:
                unconditional_embeds = self.get_prompt_embeds(
                    gen_config.negative_prompt
                )

        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            ref_latents=ref_latents,
        )
        return result[0]

    # ------------------------------------------------------------------
    # References (edit)
    # ------------------------------------------------------------------
    @property
    def match_target_res(self) -> bool:
        return bool(self.model_config.model_kwargs.get("match_target_res", True))

    @property
    def text_embedding_uses_target_size(self) -> bool:
        # the dataloader adds the item's bucket size to the text-embedding cache key
        return self.match_target_res

    def get_text_embedding_space_version(self) -> str:
        # reference sizing changes the vision tokens; keep caches from different rules apart
        rule = "match" if self.match_target_res else "trained"
        return f"{self.text_embedding_space_version}_ref{rule}"

    def _target_size(self, target_size):
        if target_size is None:
            return None
        sc = self.get_bucket_divisibility()
        return (int(target_size[0] // sc * sc), int(target_size[1] // sc * sc))

    @staticmethod
    def _normalize_control_images(
        control_images, batch_size: int
    ) -> List[List[torch.Tensor]]:
        """A (B, C, H, W) / (B, N, C, H, W) tensor, a per-sample list of lists
        or a flat list for one prompt -> one list of [0, 1] tensors per item."""
        if control_images is None:
            return [[] for _ in range(batch_size)]
        if isinstance(control_images, torch.Tensor):
            if control_images.dim() == 5:
                control_images = [list(sample) for sample in control_images]
            else:
                control_images = [[sample] for sample in control_images]
        elif len(control_images) > 0 and not isinstance(control_images[0], list):
            control_images = [list(control_images)]
        control_images = [drop_repeats(sample) for sample in control_images]
        if len(control_images) == 1 and batch_size > 1:
            control_images = control_images * batch_size
        if len(control_images) != batch_size:
            raise ValueError(
                f"got {len(control_images)} control image sets for {batch_size} prompts"
            )
        return control_images

    def _encode_crops(self, crops) -> List[torch.Tensor]:
        """Reference crops -> one (1, 128, h, w) latent each."""
        return [
            self.encode_images(
                [TF.to_tensor(crop) * 2 - 1],
                device=self.device_torch,
                dtype=self.torch_dtype,
            )
            for crop in crops
        ]

    def _batch_ref_latents(
        self, batch, batch_size: int, target_size
    ) -> Optional[List[torch.Tensor]]:
        control = batch.control_tensor_list
        if control is None:
            control = batch.control_tensor
        if control is None:
            return None
        samples = [
            self._encode_crops(
                prepare_references(images, target_size, self.match_target_res)
            )
            for images in self._normalize_control_images(control, batch_size)
        ]
        if not any(samples):
            return None
        shapes = {tuple(tuple(lat.shape) for lat in sample) for sample in samples}
        if len(shapes) > 1:
            raise ValueError(
                "every sample in a batch needs the same number and size of references; "
                f"got {sorted(shapes)}. Use match_target_res or batch size 1."
            )
        return [torch.cat(refs, 0) for refs in zip(*samples)]

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
            if batch is not None:
                # the bucket the prompt was cached against
                _, _, h, w = latent_model_input.shape
                ref_latents = self._batch_ref_latents(
                    batch, batch_size, (w * 16, h * 16)
                )

        return self.transformer(
            latent_model_input.to(device, dtype), t, embeds, mask, ref_latents
        )

    @torch.no_grad()
    def get_prompt_embeds(
        self, prompt, control_images=None, target_size=None
    ) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        target_size = self._target_size(target_size)
        images = [
            [
                vlm_image(crop)
                for crop in prepare_references(
                    sample, target_size, self.match_target_res
                )
            ]
            for sample in self._normalize_control_images(control_images, len(prompt))
        ]
        embeds = encode_prompts(
            self.text_encoder,
            self.tokenizer,
            self.image_processor,
            prompt,
            images,
            self.device_torch,
            self.torch_dtype,
            max_length=self.max_text_length,
        )
        return AdvancedPromptEmbeds(text_embeds=embeds)

    def condition_noisy_latents(
        self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"
    ):
        # references join the sequence in get_noise_prediction, clean
        return latents.detach()

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
        batch = torch.stack(
            [img.to(device, dtype=self.vae.dtype) for img in image_list]
        )
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
        transformer: Fibo2Transformer = unwrap_model(self.model)
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
            quantized=False,
        )

    def get_base_model_version(self):
        return "fibo2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return Fibo2Transformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        return Fibo2Transformer.get_quantization_exclude_modules()
