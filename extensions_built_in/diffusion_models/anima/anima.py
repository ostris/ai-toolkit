import os
from typing import List, Optional

import torch
import yaml
from optimum.quanto import freeze
from safetensors.torch import load_file, save_file

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management import MemoryManager
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.util.quantize import get_qtype, quantize, quantize_model

try:
    from diffusers import AnimaAutoBlocks, AnimaModularPipeline, AnimaTextConditioner
    from diffusers.models import CosmosTransformer3DModel
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.anima.modular_blocks_anima import AnimaCoreDenoiseStep, AnimaDecodeStep
except ImportError as e:
    raise ImportError(
        "Anima requires the Diffusers Anima branch. Run `uv pip install -r requirements.txt`."
    ) from e


scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class AnimaPromptEmbeds(PromptEmbeds):
    def __init__(
        self,
        qwen_prompt_embeds: torch.Tensor,
        t5_input_ids: torch.Tensor,
        qwen_attention_mask: torch.Tensor,
        t5_attention_mask: torch.Tensor,
    ):
        super().__init__(qwen_prompt_embeds, attention_mask=qwen_attention_mask)
        self.t5_input_ids = t5_input_ids
        self.t5_attention_mask = t5_attention_mask

    @staticmethod
    def _device_from_to_args(args, kwargs):
        if "device" in kwargs:
            return kwargs["device"]
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg.device
            if isinstance(arg, (torch.device, str, int)):
                return arg
        return None

    @staticmethod
    def _move_token_tensor(tensor: torch.Tensor, args, kwargs):
        device = AnimaPromptEmbeds._device_from_to_args(args, kwargs)
        if device is None:
            return tensor
        return tensor.to(device=device)

    def to(self, *args, **kwargs):
        self.text_embeds = self.text_embeds.to(*args, **kwargs)
        self.attention_mask = self._move_token_tensor(self.attention_mask, args, kwargs)
        self.t5_input_ids = self._move_token_tensor(self.t5_input_ids, args, kwargs)
        self.t5_attention_mask = self._move_token_tensor(self.t5_attention_mask, args, kwargs)
        return self

    def detach(self):
        return AnimaPromptEmbeds(
            self.text_embeds.detach(),
            self.t5_input_ids.detach(),
            self.attention_mask.detach(),
            self.t5_attention_mask.detach(),
        )

    def clone(self):
        return AnimaPromptEmbeds(
            self.text_embeds.clone(),
            self.t5_input_ids.clone(),
            self.attention_mask.clone(),
            self.t5_attention_mask.clone(),
        )

    def expand_to_batch(self, batch_size):
        if self.text_embeds.shape[0] == batch_size:
            return self.clone()
        if self.text_embeds.shape[0] != 1:
            raise ValueError("Can only expand Anima prompt embeds from batch size 1")
        return AnimaPromptEmbeds(
            self.text_embeds.expand(batch_size, -1, -1).clone(),
            self.t5_input_ids.expand(batch_size, -1).clone(),
            self.attention_mask.expand(batch_size, -1).clone(),
            self.t5_attention_mask.expand(batch_size, -1).clone(),
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(
            {
                "qwen_prompt_embeds": self.text_embeds.cpu(),
                "qwen_attention_mask": self.attention_mask.cpu(),
                "t5_input_ids": self.t5_input_ids.cpu(),
                "t5_attention_mask": self.t5_attention_mask.cpu(),
            },
            path,
            metadata={"class_name": self.__class__.__name__},
        )

    @classmethod
    def load(cls, path: str):
        state_dict = load_file(path, device="cpu")
        return cls(
            qwen_prompt_embeds=state_dict["qwen_prompt_embeds"],
            qwen_attention_mask=state_dict["qwen_attention_mask"],
            t5_input_ids=state_dict["t5_input_ids"],
            t5_attention_mask=state_dict["t5_attention_mask"],
        )

    @staticmethod
    def _pad_2d(tensor: torch.Tensor, max_len: int, padding_side: str, value: int = 0):
        if tensor.shape[1] == max_len:
            return tensor
        pad = torch.full(
            (tensor.shape[0], max_len - tensor.shape[1]),
            value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if padding_side == "left":
            return torch.cat([pad, tensor], dim=1)
        return torch.cat([tensor, pad], dim=1)

    @staticmethod
    def _pad_3d(tensor: torch.Tensor, max_len: int, padding_side: str):
        if tensor.shape[1] == max_len:
            return tensor
        pad = torch.zeros(
            (tensor.shape[0], max_len - tensor.shape[1], tensor.shape[2]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if padding_side == "left":
            return torch.cat([pad, tensor], dim=1)
        return torch.cat([tensor, pad], dim=1)

    @classmethod
    def concat_prompt_embeds(cls, prompt_embeds: list["AnimaPromptEmbeds"], padding_side: str = "right"):
        max_qwen_len = max(prompt.text_embeds.shape[1] for prompt in prompt_embeds)
        max_t5_len = max(prompt.t5_input_ids.shape[1] for prompt in prompt_embeds)
        return cls(
            qwen_prompt_embeds=torch.cat(
                [cls._pad_3d(prompt.text_embeds, max_qwen_len, padding_side) for prompt in prompt_embeds], dim=0
            ),
            qwen_attention_mask=torch.cat(
                [cls._pad_2d(prompt.attention_mask, max_qwen_len, padding_side) for prompt in prompt_embeds], dim=0
            ),
            t5_input_ids=torch.cat(
                [cls._pad_2d(prompt.t5_input_ids, max_t5_len, padding_side) for prompt in prompt_embeds], dim=0
            ),
            t5_attention_mask=torch.cat(
                [cls._pad_2d(prompt.t5_attention_mask, max_t5_len, padding_side) for prompt in prompt_embeds], dim=0
            ),
        )


class AnimaTrainableModel(torch.nn.Module):
    def __init__(self, transformer: CosmosTransformer3DModel, text_conditioner: AnimaTextConditioner):
        super().__init__()
        self.transformer = transformer
        self.text_conditioner = text_conditioner

    @property
    def config(self):
        return self.transformer.config

    @property
    def device(self):
        return self.transformer.device

    @property
    def dtype(self):
        return self.transformer.dtype

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def enable_gradient_checkpointing(self):
        for module in (self.transformer, self.text_conditioner):
            if hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing()
            elif hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
            elif hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True


class AnimaEmbedsToImageBlocks(SequentialPipelineBlocks):
    model_name = "anima"
    block_classes = [AnimaCoreDenoiseStep, AnimaDecodeStep]
    block_names = ["denoise", "decode"]


class AnimaModel(BaseModel):
    arch = "anima"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["AnimaTrainableModel"]
        self.supports_model_paths = True
        self.use_old_lokr_format = False
        self.max_sequence_length = model_config.model_kwargs.get("max_sequence_length", 512)

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2

    @property
    def trainable_model(self) -> AnimaTrainableModel:
        return self.model

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Anima model")

        pipe: AnimaModularPipeline = AnimaAutoBlocks().init_pipeline(self.model_config.name_or_path)
        pipe.load_components(torch_dtype=dtype)
        pipe.update_components(scheduler=self.get_train_scheduler())

        transformer = pipe.transformer
        text_conditioner = pipe.text_conditioner

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

            self.print_and_status_update("Quantizing Text Conditioner")
            quantize(text_conditioner, weights=get_qtype(self.model_config.qtype))
            freeze(text_conditioner)
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

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                pipe.text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )
            MemoryManager.attach(
                text_conditioner,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
            text_conditioner.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)
            text_conditioner.to(self.device_torch, dtype=dtype)

        pipe.text_encoder.to(self.device_torch, dtype=dtype)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.eval()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(pipe.text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(pipe.text_encoder)
            flush()

        if self.model_config.low_vram:
            pipe.text_encoder.to("cpu")
            flush()

        self.noise_scheduler = pipe.scheduler
        self.vae = pipe.vae
        self.text_encoder = [pipe.text_encoder]
        self.tokenizer = [pipe.tokenizer]
        self.t5_tokenizer = pipe.t5_tokenizer
        self.model = AnimaTrainableModel(transformer=transformer, text_conditioner=text_conditioner)
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        trainable_model = unwrap_model(self.trainable_model)
        pipeline = AnimaEmbedsToImageBlocks().init_pipeline()
        pipeline.update_components(
            scheduler=self.get_train_scheduler(),
            transformer=trainable_model.transformer,
            vae=unwrap_model(self.vae),
        )
        pipeline = pipeline.to(self.device_torch)
        return pipeline

    def _offload_text_encoder(self):
        if self.model_config.low_vram and self.pipeline.text_encoder.device != torch.device("cpu"):
            self.pipeline.text_encoder.to("cpu")
            flush()

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        images = image_list
        if isinstance(images, list):
            images = torch.stack([image.to(device, dtype=dtype) for image in images], dim=0)
        else:
            images = images.to(device, dtype=dtype)

        images = images.unsqueeze(2)
        latents = self.vae.encode(images).latent_dist.sample()
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std
        latents = latents.squeeze(2).to(device, dtype=dtype)
        if self.model_config.low_vram:
            self.vae.to("cpu")
            flush()
        return latents

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        latents = latents.to(device, dtype=dtype).unsqueeze(2)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        return self.vae.decode(latents, return_dict=False)[0][:, :, 0]

    def _condition_prompt_embeds(self, text_embeddings: AnimaPromptEmbeds, dtype=None):
        dtype = dtype or self.trainable_model.transformer.dtype
        if self.trainable_model.text_conditioner.device != self.device_torch:
            self.trainable_model.text_conditioner.to(self.device_torch)

        return self.trainable_model.text_conditioner(
            source_hidden_states=text_embeddings.text_embeds.to(self.device_torch, dtype=dtype),
            target_input_ids=text_embeddings.t5_input_ids.to(self.device_torch),
            target_attention_mask=text_embeddings.t5_attention_mask.to(self.device_torch),
            source_attention_mask=text_embeddings.attention_mask.to(self.device_torch),
        )

    def generate_single_image(
        self,
        pipeline: AnimaModularPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AnimaPromptEmbeds,
        unconditional_embeds: AnimaPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        conditional_prompt_embeds = self._condition_prompt_embeds(conditional_embeds, dtype=self.torch_dtype)
        unconditional_prompt_embeds = self._condition_prompt_embeds(unconditional_embeds, dtype=self.torch_dtype)

        if pipeline.vae.device != self.device_torch:
            pipeline.vae.to(self.device_torch, dtype=self.vae_torch_dtype)
        pipeline.guider.guidance_scale = gen_config.guidance_scale

        try:
            return pipeline(
                prompt_embeds=conditional_prompt_embeds,
                negative_prompt_embeds=unconditional_prompt_embeds,
                height=gen_config.height,
                width=gen_config.width,
                num_inference_steps=gen_config.num_inference_steps,
                latents=gen_config.latents,
                generator=generator,
                output="images",
                **extra,
            )[0]
        finally:
            if self.model_config.low_vram:
                pipeline.vae.to("cpu")
                flush()

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: AnimaPromptEmbeds,
        **kwargs,
    ):
        if self.trainable_model.transformer.device != self.device_torch:
            self.trainable_model.transformer.to(self.device_torch)

        latent_model_input = latent_model_input.unsqueeze(2).to(self.device_torch, dtype=self.torch_dtype)
        timestep = (timestep / self.noise_scheduler.config.num_train_timesteps).to(self.device_torch, self.torch_dtype)
        prompt_embeds = self._condition_prompt_embeds(text_embeddings, dtype=self.torch_dtype)
        padding_mask = latent_model_input.new_zeros(
            1,
            1,
            latent_model_input.shape[-2] * 16,
            latent_model_input.shape[-1] * 16,
            dtype=self.torch_dtype,
        )

        noise_pred = self.trainable_model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]

        return noise_pred.squeeze(2)

    @staticmethod
    def _normalize_prompts(prompt: str | List[str | None]) -> List[str]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        return ["" if prompt_item is None else prompt_item for prompt_item in prompt]

    def _get_qwen_prompt_embeds(self, prompt: List[str]):
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device_torch)
        prompt_attention_mask = text_inputs.attention_mask.to(self.device_torch)

        if text_input_ids.shape[1] == 0:
            pad_token_id = self.pipeline.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 151643
            text_input_ids = torch.full(
                (len(prompt), 1),
                pad_token_id,
                dtype=torch.long,
                device=self.device_torch,
            )
            prompt_attention_mask = torch.zeros_like(text_input_ids)

        conditioner_attention_mask = prompt_attention_mask.clone()
        empty_prompt_mask = conditioner_attention_mask.sum(dim=1) == 0
        if empty_prompt_mask.any():
            conditioner_attention_mask[empty_prompt_mask, 0] = 1

        prompt_embeds = self.pipeline.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.torch_dtype, device=self.device_torch)
        prompt_embeds = prompt_embeds * conditioner_attention_mask.to(prompt_embeds).unsqueeze(-1)

        return prompt_embeds, conditioner_attention_mask

    def _get_t5_prompt_ids(self, prompt: List[str]):
        text_inputs = self.t5_tokenizer(
            prompt,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids.to(self.device_torch), text_inputs.attention_mask.to(self.device_torch)

    def get_prompt_embeds(self, prompt: str) -> AnimaPromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        prompt = self._normalize_prompts(prompt)

        try:
            qwen_prompt_embeds, qwen_attention_mask = self._get_qwen_prompt_embeds(prompt)
            t5_input_ids, t5_attention_mask = self._get_t5_prompt_ids(prompt)
            return AnimaPromptEmbeds(
                qwen_prompt_embeds=qwen_prompt_embeds,
                qwen_attention_mask=qwen_attention_mask,
                t5_input_ids=t5_input_ids,
                t5_attention_mask=t5_attention_mask,
            )
        finally:
            self._offload_text_encoder()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        trainable_model = unwrap_model(self.trainable_model)
        trainable_model.transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        trainable_model.text_conditioner.save_pretrained(
            save_directory=os.path.join(output_path, "text_conditioner"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "anima"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks", "text_conditioner"]

    def get_model_to_train(self):
        return self.trainable_model

    @staticmethod
    def _strip_ai_toolkit_wrapper_prefix(key: str) -> str:
        if key.startswith("transformer.transformer."):
            return key.replace("transformer.transformer.", "transformer.", 1)
        if key.startswith("transformer.text_conditioner."):
            return key.replace("transformer.text_conditioner.", "text_conditioner.", 1)
        return key

    @staticmethod
    def _add_ai_toolkit_wrapper_prefix(key: str) -> str:
        if key.startswith("transformer."):
            return key.replace("transformer.", "transformer.transformer.", 1)
        if key.startswith("text_conditioner."):
            return key.replace("text_conditioner.", "transformer.text_conditioner.", 1)
        return key

    def convert_lora_weights_before_save(self, state_dict):
        return {self._strip_ai_toolkit_wrapper_prefix(key): value for key, value in state_dict.items()}

    def convert_lora_weights_before_load(self, state_dict):
        if any(key.startswith("diffusion_model.") for key in state_dict):
            from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_anima_lora_to_diffusers

            state_dict = _convert_non_diffusers_anima_lora_to_diffusers(state_dict)
        return {self._add_ai_toolkit_wrapper_prefix(key): value for key, value in state_dict.items()}
