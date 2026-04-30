from __future__ import annotations

import contextlib
import copy
import inspect
import os
import sqlite3
import time
import types
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from diffusers import ControlNetModel, T2IAdapter
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision.transforms import transforms

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.accelerator import unwrap_model
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GenerateImageConfig
from toolkit.custom_adapter import CustomAdapter
from toolkit.ip_adapter import IPAdapter
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.samplers.custom_flowmatch_sampler import FlowMatchStepWithLogProbScheduler

FLOW_GRPO_NATIVE_SCHEDULER = "flowmatch_step_with_logprob"
SUPPORTED_FLOW_GRPO_SAMPLERS = {FLOW_GRPO_NATIVE_SCHEDULER}
SUPPORTED_FLOW_GRPO_SCHEDULERS = {FLOW_GRPO_NATIVE_SCHEDULER}


@dataclass
class FlowGRPOTrainerConfig:
    clip_range: float = 0.2
    adv_clip_max: float = 5.0
    beta: float = 0.04
    noise_level: float = 0.7
    sde_type: Literal["sde", "cps"] = "sde"
    timestep_fraction: float = 1.0
    group_size: int = 4

    def __init__(self, **kwargs):
        self.clip_range = float(kwargs.get("clip_range", self.clip_range))
        self.adv_clip_max = float(kwargs.get("adv_clip_max", self.adv_clip_max))
        self.beta = float(kwargs.get("beta", self.beta))
        self.noise_level = float(kwargs.get("noise_level", self.noise_level))
        self.sde_type = kwargs.get("sde_type", self.sde_type)
        self.timestep_fraction = float(kwargs.get("timestep_fraction", self.timestep_fraction))
        self.group_size = max(2, int(kwargs.get("group_size", self.group_size)))


@dataclass
class CandidateState:
    task_id: str
    candidate_id: str
    prompt: str
    negative_prompt: str
    seed: int
    guidance_scale: float
    num_inference_steps: int
    sampler: str
    scheduler: str
    conditional_embeds: PromptEmbeds
    unconditional_embeds: Optional[PromptEmbeds]
    timesteps: torch.Tensor
    sigma_current: torch.Tensor
    sigma_next: torch.Tensor
    latents: torch.Tensor
    next_latents: torch.Tensor
    log_probs: torch.Tensor
    reference_images: Optional[torch.Tensor]
    network_multiplier: float = 1.0
    adapter_conditioning: Optional[torch.Tensor] = None
    control_tensor: Any = None
    adapter_conditioning_scale: float = 1.0


@dataclass
class CandidateTrainUnit:
    candidate_index: int
    candidate_state: CandidateState
    advantage: float
    step_index: int
    train_steps: int


@dataclass
class _RolloutSampleConditioning:
    conditional_embeds: PromptEmbeds
    unconditional_embeds: Optional[PromptEmbeds]
    adapter_conditioning: Optional[torch.Tensor] = None
    control_tensor: Any = None


class _RolloutBatch:
    def __init__(
        self,
        *,
        control_tensor: Any = None,
        latents: Optional[torch.Tensor] = None,
        tensor: Optional[torch.Tensor] = None,
    ):
        self.control_tensor = control_tensor
        self.control_tensor_list = None
        self.latents = latents
        self.tensor = tensor
        self.inpaint_tensor = None


def _clone_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone().cpu()


def _clone_prompt_embeds(prompt_embeds: Optional[PromptEmbeds]) -> Optional[PromptEmbeds]:
    if prompt_embeds is None:
        return None
    return prompt_embeds.detach().clone().to("cpu")


def _clone_tensor_tree(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _clone_tensor(value)
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    raise TypeError(f"Unsupported tensor tree value type: {type(value)!r}")


def _tensor_tree_shape(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, list):
        return tuple(_tensor_tree_shape(item) for item in value)
    raise TypeError(f"Unsupported tensor tree value type: {type(value)!r}")


def _to_device_tensor_tree(value, device: torch.device, dtype: torch.dtype):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device, dtype=dtype)
    if isinstance(value, list):
        return [_to_device_tensor_tree(item, device, dtype) for item in value]
    raise TypeError(f"Unsupported tensor tree value type: {type(value)!r}")


def _concat_tensor_tree(values: list[Any], device: torch.device, dtype: torch.dtype):
    first = values[0]
    if first is None:
        return None
    if isinstance(first, torch.Tensor):
        return torch.cat([value.to(device, dtype=dtype) for value in values], dim=0)
    if isinstance(first, list):
        return [
            _concat_tensor_tree([value[idx] for value in values], device, dtype)
            for idx in range(len(first))
        ]
    raise TypeError(f"Unsupported tensor tree value type: {type(first)!r}")


class FlowGRPOTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        train_config = config.setdefault("train", {})
        train_config["disable_sampling"] = True
        train_config["cache_text_embeddings"] = False
        if not train_config.get("noise_scheduler"):
            train_config["noise_scheduler"] = FLOW_GRPO_NATIVE_SCHEDULER
        sample_config = config.setdefault("sample", {})
        sample_config["sample_every"] = 0
        sample_config["samples"] = []
        if not sample_config.get("sampler"):
            sample_config["sampler"] = FLOW_GRPO_NATIVE_SCHEDULER
        super().__init__(process_id, job, config, **kwargs)
        if not self.is_ui_trainer:
            raise ValueError("flow_grpo_trainer requires the UI SQLite runtime.")
        self.flow_grpo_config = FlowGRPOTrainerConfig(**self.config.get("grpo", {}))
        self._prompt_index = 0
        self._task_counter = 0
        self._flow_grpo_root = Path(self.save_root) / "flow_grpo"
        self._candidate_root = self._flow_grpo_root / "candidates"
        self._candidate_root.mkdir(parents=True, exist_ok=True)
        self._validate_flow_grpo_runtime_config()

    def _normalize_control_value(self, value: Optional[str], default: str) -> str:
        normalized = (value or default).strip().lower()
        return normalized or default

    def _validate_flow_grpo_runtime_config(self) -> None:
        train_scheduler = self._normalize_control_value(self.train_config.noise_scheduler, FLOW_GRPO_NATIVE_SCHEDULER)
        sample_sampler = self._normalize_control_value(self.sample_config.sampler, FLOW_GRPO_NATIVE_SCHEDULER)
        self._validate_sampler(sample_sampler)
        self._validate_scheduler(train_scheduler)

    def _validate_sampler(self, sampler: str) -> str:
        sampler = self._normalize_control_value(sampler, FLOW_GRPO_NATIVE_SCHEDULER)
        if sampler not in SUPPORTED_FLOW_GRPO_SAMPLERS:
            supported = ", ".join(sorted(SUPPORTED_FLOW_GRPO_SAMPLERS))
            raise ValueError(f"Unsupported Flow-GRPO sampler '{sampler}'. Supported values: {supported}")
        return sampler

    def _validate_scheduler(self, scheduler: str) -> str:
        scheduler = self._normalize_control_value(scheduler, FLOW_GRPO_NATIVE_SCHEDULER)
        if scheduler not in SUPPORTED_FLOW_GRPO_SCHEDULERS:
            supported = ", ".join(sorted(SUPPORTED_FLOW_GRPO_SCHEDULERS))
            raise ValueError(f"Unsupported Flow-GRPO scheduler '{scheduler}'. Supported values: {supported}")
        return scheduler

    def _build_task_scheduler(self, scheduler: str):
        scheduler = self._validate_scheduler(scheduler)
        source_scheduler = getattr(self.sd, "noise_scheduler", None)
        if source_scheduler is None and hasattr(self.sd, "get_train_scheduler"):
            source_scheduler = self.sd.get_train_scheduler()
        if source_scheduler is None:
            raise ValueError("Flow-GRPO requires the active AITK model to provide a train scheduler.")

        config = copy.deepcopy(dict(getattr(source_scheduler, "config", {})))
        if scheduler == FLOW_GRPO_NATIVE_SCHEDULER:
            if config:
                return FlowMatchStepWithLogProbScheduler.from_config(config)
            return FlowMatchStepWithLogProbScheduler()
        if config and hasattr(source_scheduler.__class__, "from_config"):
            return source_scheduler.__class__.from_config(config)
        return copy.deepcopy(source_scheduler)

    @contextlib.contextmanager
    def _model_device_state_preset(self, preset: str):
        with self._quantized_text_encoder_noop_to_guard():
            self.sd.set_device_state_preset(preset)
            try:
                yield
            finally:
                self.sd.restore_device_state()

    @staticmethod
    def _is_traceable_tensor_subclass(tensor: torch.Tensor) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and type(tensor) is not torch.Tensor
            and hasattr(tensor, "__tensor_flatten__")
            and hasattr(tensor, "__tensor_unflatten__")
        )

    @classmethod
    def _has_traceable_tensor_subclass_param(cls, module: torch.nn.Module) -> bool:
        return any(cls._is_traceable_tensor_subclass(param) for param in module.parameters())

    @staticmethod
    def _module_device(module: torch.nn.Module) -> Optional[torch.device]:
        device = getattr(module, "device", None)
        if device is not None:
            return torch.device(device)
        for tensor in list(module.parameters()) + list(module.buffers()):
            return tensor.device
        return None

    @staticmethod
    def _same_device(current: torch.device, target: torch.device) -> bool:
        return current.type == target.type and (target.index is None or current.index == target.index)

    @contextlib.contextmanager
    def _quantized_text_encoder_noop_to_guard(self):
        text_encoder = getattr(self.sd, "text_encoder", None)
        if text_encoder is None:
            yield
            return

        encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        patched: list[tuple[torch.nn.Module, Any]] = []
        try:
            for encoder in encoders:
                if not self._has_traceable_tensor_subclass_param(encoder):
                    continue
                original_to = encoder.to

                def guarded_to(module, *args, _original_to=original_to, **kwargs):
                    device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
                    current_device = self._module_device(module)
                    if dtype is None and device is not None and current_device is not None:
                        if self._same_device(current_device, torch.device(device)):
                            return module
                    return _original_to(*args, **kwargs)

                patched.append((encoder, original_to))
                encoder.to = types.MethodType(guarded_to, encoder)
            yield
        finally:
            for encoder, original_to in patched:
                encoder.to = original_to

    def _rollout_patch_size(self) -> int:
        for module_name in ("transformer", "unet", "model"):
            module = getattr(self.sd, module_name, None)
            config = getattr(module, "config", None)
            patch_size = getattr(config, "patch_size", None)
            if patch_size is None and isinstance(config, dict):
                patch_size = config.get("patch_size")
            if patch_size is not None:
                if isinstance(patch_size, (tuple, list)):
                    patch_size = patch_size[0]
                return max(1, int(patch_size))
        return 1

    def _get_rollout_initial_latents(self, *, height: int, width: int, batch_size: int) -> torch.Tensor:
        if not hasattr(self.sd, "encode_images") or not hasattr(self.sd, "get_latent_noise_from_latents"):
            raise RuntimeError(
                "Flow-GRPO requires encode_images() and get_latent_noise_from_latents() "
                "to create training-shaped rollout latents."
            )

        vae_device = getattr(self.sd, "vae_device_torch", self.device_torch)
        dtype = self.sd.torch_dtype
        reference_images = [
            torch.zeros((3, int(height), int(width)), device=vae_device, dtype=dtype)
            for _ in range(batch_size)
        ]
        with torch.no_grad():
            reference_latents = self.sd.encode_images(reference_images).to(self.device_torch, dtype=dtype)
            return self.sd.get_latent_noise_from_latents(reference_latents)

    def _db_execute(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        def _op():
            with self._db_connect() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()

        return self._retry_db_operation(_op)

    def _db_execute_write(self, query: str, params: tuple = ()) -> None:
        def _op():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

        self._retry_db_operation(_op)

    def _db_execute_many(self, query: str, rows: list[tuple]) -> None:
        def _op():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, rows)

        self._retry_db_operation(_op)

    def _count_open_tasks(self) -> int:
        rows = self._db_execute(
            "SELECT COUNT(*) AS count FROM FlowGRPOVoteTask WHERE job_id = ? AND status = 'open'",
            (self.job_id,),
        )
        return int(rows[0]["count"]) if rows else 0

    def _count_active_vote_tasks(self) -> int:
        rows = self._db_execute(
            """
            SELECT COUNT(*) AS count
            FROM FlowGRPOVoteTask
            WHERE job_id = ? AND status IN ('generating', 'open')
            """,
            (self.job_id,),
        )
        return int(rows[0]["count"]) if rows else 0

    def _task_dir(self, task_id: str) -> Path:
        task_dir = self._candidate_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _candidate_seed(self, task_row: sqlite3.Row, order_index: int) -> int:
        task_seed = task_row["seed"]
        if task_seed is not None:
            return int(task_seed) + order_index
        return int(self.sample_config.seed) + order_index + self.step_num + (self._task_counter * 1000)

    def _candidate_generation_batch_size(self) -> int:
        return max(1, min(int(self.train_config.batch_size), self.flow_grpo_config.group_size))

    def _prepare_scheduler_run(
        self,
        *,
        sampler: str,
        scheduler_name: str,
        num_inference_steps: int,
        latents: torch.Tensor,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_sampler(sampler)
        scheduler = self._build_task_scheduler(scheduler_name)
        if hasattr(scheduler, "set_train_timesteps"):
            signature = inspect.signature(scheduler.set_train_timesteps)
            kwargs: dict[str, Any] = {
                "num_timesteps": int(num_inference_steps),
                "device": self.device_torch,
            }
            if "timestep_type" in signature.parameters:
                kwargs["timestep_type"] = "shift"
            if "latents" in signature.parameters:
                kwargs["latents"] = latents
            if "patch_size" in signature.parameters:
                kwargs["patch_size"] = self._rollout_patch_size()
            timesteps = scheduler.set_train_timesteps(**kwargs)
        else:
            scheduler.set_timesteps(int(num_inference_steps), device=self.device_torch)
            timesteps = scheduler.timesteps

        if not hasattr(scheduler, "sigmas"):
            raise ValueError("Flow-GRPO requires a scheduler with `sigmas`.")

        timesteps = timesteps.to(self.device_torch, dtype=torch.float32)
        sigmas = scheduler.sigmas.to(self.device_torch, dtype=torch.float32)
        if sigmas.numel() < timesteps.numel() + 1:
            raise ValueError("Flow-GRPO requires one sigma transition per timestep.")

        sigma_current = sigmas[: timesteps.numel()]
        sigma_next = sigmas[1 : timesteps.numel() + 1]
        if sigma_current.numel() > 0 and sigma_current[0] >= 1.0:
            sigma_current = sigma_current.clone()
            adjusted_sigma0 = (sigma_current[0] + sigma_next[0]) * 0.5
            sigma_current[0] = adjusted_sigma0
            scheduler.sigmas = scheduler.sigmas.clone()
            scheduler.sigmas[0] = adjusted_sigma0.to(device=scheduler.sigmas.device, dtype=scheduler.sigmas.dtype)
        scheduler._step_index = None
        return scheduler, timesteps, sigma_current, sigma_next

    def _build_preview_image_config(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
        image_path: Path,
    ) -> GenerateImageConfig:
        return GenerateImageConfig(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            network_multiplier=getattr(self.sample_config, "network_multiplier", 1.0),
            output_path=str(image_path),
            output_ext=image_path.suffix.lstrip("."),
            logger=self.logger,
            num_frames=getattr(self.sample_config, "num_frames", 1),
            fps=getattr(self.sample_config, "fps", 1),
            guidance_rescale=getattr(self.sample_config, "guidance_rescale", 0.0),
            adapter_conditioning_scale=getattr(self.sample_config, "adapter_conditioning_scale", 1.0),
            refiner_start_at=getattr(self.sample_config, "refiner_start_at", 0.5),
            extra_values=getattr(self.sample_config, "extra_values", []),
            ctrl_img=getattr(self.sample_config, "ctrl_img", None),
            ctrl_img_1=getattr(self.sample_config, "ctrl_img_1", None),
            ctrl_img_2=getattr(self.sample_config, "ctrl_img_2", None),
            ctrl_img_3=getattr(self.sample_config, "ctrl_img_3", None),
            ctrl_idx=getattr(self.sample_config, "ctrl_idx", 0),
            do_cfg_norm=getattr(self.sample_config, "do_cfg_norm", False),
        )

    @staticmethod
    def _load_control_image_tensor(path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device, dtype=dtype)

    def _get_text_encoding_control_images(self, gen_config: GenerateImageConfig):
        if not getattr(self.sd, "encode_control_in_text_embeddings", False):
            return None

        ctrl_tensors = [
            self._load_control_image_tensor(path, self.device_torch, self.sd.torch_dtype)
            for path in (gen_config.ctrl_img, gen_config.ctrl_img_1, gen_config.ctrl_img_2, gen_config.ctrl_img_3)
            if path is not None
        ]
        if not ctrl_tensors:
            return None
        if getattr(self.sd, "has_multiple_control_images", False):
            return ctrl_tensors
        return ctrl_tensors[0]

    @staticmethod
    def _tensor_0_1_from_image(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return transforms.ToTensor()(image).unsqueeze(0).to(device, dtype=dtype)

    def _prepare_sample_time_conditioning(
        self,
        gen_config: GenerateImageConfig,
        *,
        sample_index: int = 0,
    ) -> _RolloutSampleConditioning:
        adapter = getattr(self.sd, "adapter", None)
        validation_image = None
        adapter_conditioning = None
        control_tensor = None
        if adapter is not None and gen_config.adapter_image_path is not None:
            validation_image = Image.open(gen_config.adapter_image_path)
            if ".inpaint." not in gen_config.adapter_image_path:
                validation_image = validation_image.convert("RGB")
            elif validation_image.mode != "RGBA":
                raise ValueError("Inpainting images must have an alpha channel")

            if isinstance(adapter, T2IAdapter):
                validation_image = validation_image.resize((gen_config.width * 2, gen_config.height * 2))
                adapter_conditioning = self._tensor_0_1_from_image(
                    validation_image,
                    self.device_torch,
                    self.sd.torch_dtype,
                )
            if isinstance(adapter, ControlNetModel):
                validation_image = validation_image.resize((gen_config.width, gen_config.height))
                adapter_conditioning = self._tensor_0_1_from_image(
                    validation_image,
                    self.device_torch,
                    self.sd.torch_dtype,
                )
            if isinstance(adapter, CustomAdapter) and adapter.control_lora is not None:
                validation_image = validation_image.resize((gen_config.width, gen_config.height))
                control_tensor = self._tensor_0_1_from_image(
                    validation_image,
                    self.device_torch,
                    self.sd.torch_dtype,
                )
            if isinstance(adapter, IPAdapter) or isinstance(adapter, ClipVisionAdapter):
                validation_image = transforms.ToTensor()(validation_image)
            if isinstance(adapter, CustomAdapter):
                validation_image = transforms.ToTensor()(validation_image)
                adapter.num_images = 1
            if isinstance(adapter, ReferenceAdapter):
                validation_image = transforms.ToTensor()(validation_image)
                validation_image = validation_image * 2.0 - 1.0
                validation_image = validation_image.unsqueeze(0)
                adapter.set_reference_images(validation_image)

        if isinstance(adapter, ClipVisionAdapter) and gen_config.adapter_image_path is not None:
            conditional_clip_embeds = adapter.get_clip_image_embeds_from_tensors(validation_image)
            adapter(conditional_clip_embeds)

        if isinstance(adapter, CustomAdapter):
            gen_config.prompt = adapter.condition_prompt(gen_config.prompt, is_unconditional=False)
            gen_config.prompt_2 = gen_config.prompt
            gen_config.negative_prompt = adapter.condition_prompt(gen_config.negative_prompt, is_unconditional=True)
            gen_config.negative_prompt_2 = gen_config.negative_prompt

        if isinstance(adapter, CustomAdapter) and validation_image is not None:
            adapter.trigger_pre_te(
                tensors_0_1=validation_image,
                is_training=False,
                has_been_preprocessed=False,
                quad_count=4,
            )

        sample_prompts_cache = getattr(self.sd, "sample_prompts_cache", None)
        if sample_prompts_cache is not None:
            conditional_embeds = sample_prompts_cache[sample_index]["conditional"].to(
                self.device_torch,
                dtype=self.sd.torch_dtype,
            )
            unconditional_embeds = sample_prompts_cache[sample_index]["unconditional"].to(
                self.device_torch,
                dtype=self.sd.torch_dtype,
            )
        else:
            control_images = self._get_text_encoding_control_images(gen_config)
            if control_tensor is None:
                control_tensor = control_images
            if isinstance(adapter, CustomAdapter):
                adapter.is_unconditional_run = False
            conditional_embeds = self.sd.encode_prompt(
                gen_config.prompt,
                gen_config.prompt_2,
                force_all=True,
                control_images=control_images,
            )

            if isinstance(adapter, CustomAdapter):
                adapter.is_unconditional_run = True
            unconditional_embeds = self.sd.encode_prompt(
                gen_config.negative_prompt,
                gen_config.negative_prompt_2,
                force_all=True,
                control_images=control_images,
            )
            if isinstance(adapter, CustomAdapter):
                adapter.is_unconditional_run = False

        gen_config.post_process_embeddings(conditional_embeds, unconditional_embeds)

        decorator = getattr(self.sd, "decorator", None)
        if decorator is not None:
            conditional_embeds.text_embeds = decorator(conditional_embeds.text_embeds)
            unconditional_embeds.text_embeds = decorator(unconditional_embeds.text_embeds, is_unconditional=True)

        if isinstance(adapter, IPAdapter) and gen_config.adapter_image_path is not None:
            conditional_clip_embeds = adapter.get_clip_image_embeds_from_tensors(validation_image)
            unconditional_clip_embeds = adapter.get_clip_image_embeds_from_tensors(validation_image, True)
            conditional_embeds = adapter(conditional_embeds, conditional_clip_embeds, is_unconditional=False)
            unconditional_embeds = adapter(unconditional_embeds, unconditional_clip_embeds, is_unconditional=True)

        if isinstance(adapter, CustomAdapter):
            conditional_embeds = adapter.condition_encoded_embeds(
                tensors_0_1=validation_image,
                prompt_embeds=conditional_embeds,
                is_training=False,
                has_been_preprocessed=False,
                is_generating_samples=True,
            )
            unconditional_embeds = adapter.condition_encoded_embeds(
                tensors_0_1=validation_image,
                prompt_embeds=unconditional_embeds,
                is_training=False,
                has_been_preprocessed=False,
                is_unconditional=True,
                is_generating_samples=True,
            )

        if isinstance(adapter, CustomAdapter) and len(gen_config.extra_values) > 0:
            extra_values = torch.tensor([gen_config.extra_values], device=self.device_torch, dtype=self.sd.torch_dtype)
            adapter.add_extra_values(extra_values, is_unconditional=False)
            adapter.add_extra_values(torch.zeros_like(extra_values), is_unconditional=True)

        conditional_embeds = conditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype)
        unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype)
        if gen_config.guidance_scale <= 1.0:
            unconditional_embeds = None
        return _RolloutSampleConditioning(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            adapter_conditioning=adapter_conditioning,
            control_tensor=control_tensor,
        )

    @contextlib.contextmanager
    def _rollout_network_state(
        self,
        *,
        network_multiplier: float,
        active: bool = True,
        allow_merge: bool = False,
    ):
        if self.network is None:
            yield
            return

        network = unwrap_model(self.network)
        was_active = bool(getattr(network, "is_active", True))
        was_training = bool(network.training)
        start_multiplier = network.multiplier
        merge_multiplier = float(network_multiplier)
        merged = False
        try:
            network.is_active = bool(active)
            network.eval()
            network.multiplier = float(network_multiplier)
            if active and allow_merge and getattr(network, "can_merge_in", False):
                self.sd.unet.to(self.device_torch)
                network.merge_in(merge_weight=merge_multiplier)
                merged = True
            if active:
                with network:
                    yield
            else:
                yield
        finally:
            if merged and getattr(network, "is_merged_in", False):
                network.merge_out(merge_multiplier)
            network.multiplier = start_multiplier
            network.is_active = was_active
            if was_training:
                network.train()
            else:
                network.eval()

    @contextlib.contextmanager
    def _preserve_rng_state(self):
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        try:
            yield
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

    def _build_rollout_batch(
        self,
        *,
        control_tensor: Any,
        latents: torch.Tensor,
    ) -> _RolloutBatch:
        tensor = None
        if control_tensor is not None:
            tensor_ref = control_tensor[0] if isinstance(control_tensor, list) else control_tensor
            tensor = torch.zeros(
                (
                    tensor_ref.shape[0],
                    tensor_ref.shape[1],
                    tensor_ref.shape[2],
                    tensor_ref.shape[3],
                ),
                device=tensor_ref.device,
                dtype=tensor_ref.dtype,
            )
        return _RolloutBatch(
            control_tensor=control_tensor,
            latents=latents,
            tensor=tensor,
        )

    def _condition_rollout_latents(
        self,
        latents: torch.Tensor,
        batch: _RolloutBatch,
    ) -> torch.Tensor:
        model_latents = self.sd.condition_noisy_latents(latents, batch)
        adapter = getattr(self.sd, "adapter", None)
        if isinstance(adapter, CustomAdapter):
            model_latents = adapter.condition_noisy_latents(model_latents, batch)
        return model_latents

    def _build_adapter_predict_kwargs(
        self,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        adapter_conditioning: Optional[torch.Tensor],
        adapter_conditioning_scale: float,
    ) -> dict[str, Any]:
        adapter = getattr(self.sd, "adapter", None)
        if adapter is None or adapter_conditioning is None:
            return {}

        pred_kwargs: dict[str, Any] = {}
        adapter_conditioning = adapter_conditioning.to(self.device_torch, dtype=self.sd.torch_dtype)
        if isinstance(adapter, T2IAdapter):
            down_block_additional_residuals = adapter(adapter_conditioning)
            pred_kwargs["down_intrablock_additional_residuals"] = [
                sample.to(device=self.device_torch, dtype=self.sd.torch_dtype) * float(adapter_conditioning_scale)
                for sample in down_block_additional_residuals
            ]
        elif isinstance(adapter, ControlNetModel):
            added_cond_kwargs = {}
            if getattr(self.sd, "is_xl", False):
                added_cond_kwargs["text_embeds"] = conditional_embeds.pooled_embeds
                added_cond_kwargs["time_ids"] = self.sd.get_time_ids_from_latents(latents)
            down_block_res_samples, mid_block_res_sample = adapter(
                latents,
                timesteps,
                encoder_hidden_states=conditional_embeds.text_embeds,
                controlnet_cond=adapter_conditioning,
                conditioning_scale=float(adapter_conditioning_scale),
                guess_mode=False,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            pred_kwargs["down_block_additional_residuals"] = down_block_res_samples
            pred_kwargs["mid_block_additional_residual"] = mid_block_res_sample
        return pred_kwargs

    @contextlib.contextmanager
    def _sample_assistant_lora_state(self, *, offload_on_exit: bool = True):
        model_config = getattr(self.sd, "model_config", None)
        assistant_lora = getattr(self.sd, "assistant_lora", None)
        if model_config is None or assistant_lora is None:
            yield
            return

        assistant_lora_path = getattr(model_config, "assistant_lora_path", None)
        inference_lora_path = getattr(model_config, "inference_lora_path", None)
        if assistant_lora_path is None and inference_lora_path is None:
            yield
            return

        assistant_lora_requires_grad = [
            (param, bool(param.requires_grad))
            for param in assistant_lora.parameters()
        ]
        try:
            assistant_lora.requires_grad_(False)
            if assistant_lora_path is not None:
                if getattr(self.sd, "invert_assistant_lora", False):
                    assistant_lora.is_active = True
                    assistant_lora.force_to(self.device_torch, self.sd.torch_dtype)
                else:
                    assistant_lora.is_active = False

            if inference_lora_path is not None:
                assistant_lora.is_active = True
                assistant_lora.force_to(self.device_torch, self.sd.torch_dtype)

            yield
        finally:
            if assistant_lora_path is not None:
                if getattr(self.sd, "invert_assistant_lora", False):
                    assistant_lora.is_active = False
                    if offload_on_exit:
                        assistant_lora.force_to("cpu", self.sd.torch_dtype)
                else:
                    assistant_lora.is_active = True

            if inference_lora_path is not None:
                assistant_lora.is_active = False
                if offload_on_exit:
                    assistant_lora.force_to("cpu", self.sd.torch_dtype)
            for param, requires_grad in assistant_lora_requires_grad:
                param.requires_grad_(requires_grad)

    @staticmethod
    def _decoded_tensor_to_pil(decoded_image: torch.Tensor) -> Image.Image:
        image = decoded_image.detach().to(torch.float32).cpu()
        if image.ndim == 4:
            image = image[0]
        image = (image.clamp(-1.0, 1.0) + 1.0) * 0.5
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        return Image.fromarray(image)

    def _save_rollout_preview_image(
        self,
        *,
        final_latents: torch.Tensor,
        preview_config: GenerateImageConfig,
    ) -> Image.Image:
        with self._model_device_state_preset("generate"):
            with torch.no_grad():
                decoded = self.sd.decode_latents(
                    final_latents.to(self.device_torch, dtype=self.sd.torch_dtype),
                    device=self.device_torch,
                    dtype=self.sd.torch_dtype,
                )
        image = self._decoded_tensor_to_pil(decoded)
        preview_config.save_image(image)
        return image

    def _save_candidate_state(self, state: CandidateState, state_path: Path) -> None:
        torch.save(
            {
                "task_id": state.task_id,
                "candidate_id": state.candidate_id,
                "prompt": state.prompt,
                "negative_prompt": state.negative_prompt,
                "seed": state.seed,
                "guidance_scale": state.guidance_scale,
                "num_inference_steps": state.num_inference_steps,
                "sampler": state.sampler,
                "scheduler": state.scheduler,
                "conditional_embeds": _clone_prompt_embeds(state.conditional_embeds),
                "unconditional_embeds": _clone_prompt_embeds(state.unconditional_embeds),
                "timesteps": _clone_tensor(state.timesteps),
                "sigma_current": _clone_tensor(state.sigma_current),
                "sigma_next": _clone_tensor(state.sigma_next),
                "latents": _clone_tensor(state.latents),
                "next_latents": _clone_tensor(state.next_latents),
                "log_probs": _clone_tensor(state.log_probs),
                "reference_images": _clone_tensor(state.reference_images),
                "network_multiplier": float(state.network_multiplier),
                "adapter_conditioning": _clone_tensor(state.adapter_conditioning),
                "control_tensor": _clone_tensor_tree(state.control_tensor),
                "adapter_conditioning_scale": float(state.adapter_conditioning_scale),
            },
            state_path,
        )

    def _load_candidate_state(self, state_path: str) -> CandidateState:
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        return CandidateState(**payload)

    def _generate_candidate_state(
        self,
        *,
        task_id: str,
        candidate_id: str,
        gen_config: GenerateImageConfig,
        num_inference_steps: int,
        sampler: str,
        scheduler: str,
    ) -> CandidateState:
        return self._generate_candidate_states_batch(
            task_id=task_id,
            candidate_ids=[candidate_id],
            gen_configs=[gen_config],
            num_inference_steps=num_inference_steps,
            sampler=sampler,
            scheduler=scheduler,
            batch_start_index=0,
            total_candidates=1,
        )[0]

    def _generate_candidate_states_batch(
        self,
        *,
        task_id: str,
        candidate_ids: list[str],
        gen_configs: list[GenerateImageConfig],
        num_inference_steps: int,
        sampler: str,
        scheduler: str,
        batch_start_index: int,
        total_candidates: int,
    ) -> list[CandidateState]:
        if not gen_configs:
            return []
        if len(candidate_ids) != len(gen_configs):
            raise ValueError("candidate_ids and gen_configs must have the same length.")

        first_config = gen_configs[0]
        with (
            self._model_device_state_preset("generate"),
            self._sample_assistant_lora_state(),
            self._rollout_network_state(
                network_multiplier=float(first_config.network_multiplier),
                active=True,
                allow_merge=True,
            ),
        ):
            sample_conditioning_list: list[_RolloutSampleConditioning] = []
            latent_list: list[torch.Tensor] = []
            for gen_config in gen_configs:
                torch.manual_seed(gen_config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(gen_config.seed)

                sample_conditioning = self._prepare_sample_time_conditioning(gen_config)
                sample_conditioning_list.append(sample_conditioning)
                latent_list.append(
                    self._get_rollout_initial_latents(
                        height=gen_config.height,
                        width=gen_config.width,
                        batch_size=1,
                    ).to(self.device_torch, dtype=self.sd.torch_dtype)
                )

            latents = torch.cat(latent_list, dim=0)
            conditional_embeds = concat_prompt_embeds(
                [conditioning.conditional_embeds for conditioning in sample_conditioning_list]
            ).to(self.device_torch, dtype=self.sd.torch_dtype)
            unconditional_embeds = None
            if sample_conditioning_list[0].unconditional_embeds is not None:
                unconditional_embeds = concat_prompt_embeds(
                    [conditioning.unconditional_embeds for conditioning in sample_conditioning_list]
                ).to(self.device_torch, dtype=self.sd.torch_dtype)
            adapter_conditioning = None
            if sample_conditioning_list[0].adapter_conditioning is not None:
                adapter_conditioning = _concat_tensor_tree(
                    [conditioning.adapter_conditioning for conditioning in sample_conditioning_list],
                    self.device_torch,
                    self.sd.torch_dtype,
                )
            control_tensor = None
            if sample_conditioning_list[0].control_tensor is not None:
                control_tensor = _concat_tensor_tree(
                    [conditioning.control_tensor for conditioning in sample_conditioning_list],
                    self.device_torch,
                    self.sd.torch_dtype,
                )
            rollout_scheduler, timesteps, sigma_current, sigma_next = self._prepare_scheduler_run(
                sampler=sampler,
                scheduler_name=scheduler,
                num_inference_steps=num_inference_steps,
                latents=latents,
            )

            latents_before: list[torch.Tensor] = []
            latents_after: list[torch.Tensor] = []
            log_probs: list[torch.Tensor] = []

            step_iter = tqdm(
                range(timesteps.numel()),
                desc=(
                    f"Flow-GRPO rollout "
                    f"{batch_start_index + 1}-{batch_start_index + len(gen_configs)}/{total_candidates}"
                ),
                leave=False,
            )
            for idx in step_iter:
                self.maybe_stop()
                timestep = timesteps[idx].reshape(1).repeat(latents.shape[0]).to(self.device_torch, dtype=torch.float32)
                rollout_batch = self._build_rollout_batch(
                    control_tensor=control_tensor,
                    latents=latents,
                )
                model_latents = self._condition_rollout_latents(latents, rollout_batch)
                adapter_predict_kwargs = self._build_adapter_predict_kwargs(
                    latents=latents,
                    timesteps=timestep,
                    conditional_embeds=conditional_embeds,
                    adapter_conditioning=adapter_conditioning,
                    adapter_conditioning_scale=float(first_config.adapter_conditioning_scale),
                )
                with torch.no_grad():
                    noise_pred = self.sd.predict_noise(
                        latents=model_latents,
                        conditional_embeddings=conditional_embeds,
                        unconditional_embeddings=unconditional_embeds,
                        timestep=timestep,
                        guidance_scale=float(first_config.guidance_scale),
                        batch=rollout_batch,
                        **adapter_predict_kwargs,
                    )

                step_output = rollout_scheduler.step(
                    model_output=noise_pred,
                    timestep=timesteps[idx],
                    sample=latents,
                    noise_level=self.flow_grpo_config.noise_level,
                    sde_type=self.flow_grpo_config.sde_type,
                    return_log_prob=True,
                    return_transition=False,
                )
                next_latents = step_output.prev_sample.to(self.device_torch, dtype=self.sd.torch_dtype)
                log_prob = step_output.log_prob
                if log_prob is None:
                    raise RuntimeError("Flow-GRPO scheduler did not return rollout log-probability.")
                latents_before.append(_clone_tensor(latents))
                latents_after.append(_clone_tensor(next_latents))
                log_probs.append(_clone_tensor(log_prob))
                latents = next_latents
                step_iter.set_postfix(step=f"{idx + 1}/{timesteps.numel()}", batch=len(gen_configs))

            latents_stack = torch.stack(latents_before, dim=1)
            next_latents_stack = torch.stack(latents_after, dim=1)
            log_probs_stack = torch.stack(log_probs, dim=1)

            states: list[CandidateState] = []
            for idx, gen_config in enumerate(gen_configs):
                sample_conditioning = sample_conditioning_list[idx]
                states.append(
                    CandidateState(
                        task_id=task_id,
                        candidate_id=candidate_ids[idx],
                        prompt=gen_config.prompt,
                        negative_prompt=gen_config.negative_prompt,
                        seed=int(gen_config.seed),
                        guidance_scale=float(gen_config.guidance_scale),
                        num_inference_steps=int(num_inference_steps),
                        sampler=sampler,
                        scheduler=scheduler,
                        conditional_embeds=_clone_prompt_embeds(sample_conditioning.conditional_embeds),
                        unconditional_embeds=_clone_prompt_embeds(sample_conditioning.unconditional_embeds),
                        timesteps=_clone_tensor(timesteps),
                        sigma_current=_clone_tensor(sigma_current),
                        sigma_next=_clone_tensor(sigma_next),
                        latents=_clone_tensor(latents_stack[idx : idx + 1]),
                        next_latents=_clone_tensor(next_latents_stack[idx : idx + 1]),
                        log_probs=_clone_tensor(log_probs_stack[idx : idx + 1]),
                        reference_images=None,
                        network_multiplier=float(gen_config.network_multiplier),
                        adapter_conditioning=_clone_tensor(sample_conditioning.adapter_conditioning),
                        control_tensor=_clone_tensor_tree(sample_conditioning.control_tensor),
                        adapter_conditioning_scale=float(gen_config.adapter_conditioning_scale),
                    )
                )
            return states

    def _next_requested_task(self) -> Optional[sqlite3.Row]:
        rows = self._db_execute(
            """
            SELECT *
            FROM FlowGRPOVoteTask
            WHERE job_id = ? AND status = 'requested'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (self.job_id,),
        )
        return rows[0] if rows else None

    def _generate_requested_task(self, task_row: sqlite3.Row) -> None:
        task_id = task_row["id"]
        prompt = task_row["prompt"] or ""
        negative_prompt = task_row["negative_prompt"] or ""
        width = int(task_row["width"] or self.sample_config.width)
        height = int(task_row["height"] or self.sample_config.height)
        guidance_scale = float(task_row["guidance_scale"] or self.sample_config.guidance_scale)
        num_inference_steps = int(task_row["num_inference_steps"] or self.sample_config.sample_steps)
        sampler = self._validate_sampler(task_row["sampler"] or self.sample_config.sampler or FLOW_GRPO_NATIVE_SCHEDULER)
        scheduler = self._validate_scheduler(
            task_row["scheduler"] or self.train_config.noise_scheduler or FLOW_GRPO_NATIVE_SCHEDULER
        )
        task_dir = self._task_dir(task_id)

        self._db_execute_write(
            "UPDATE FlowGRPOVoteTask SET status = 'generating' WHERE id = ? AND job_id = ?",
            (task_id, self.job_id),
        )

        candidate_rows: list[tuple] = []
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        try:
            self.update_status("running", "Generating Flow-GRPO rollout group")
            generation_batch_size = self._candidate_generation_batch_size()
            for batch_start in range(0, self.flow_grpo_config.group_size, generation_batch_size):
                batch_end = min(
                    self.flow_grpo_config.group_size,
                    batch_start + generation_batch_size,
                )
                specs: list[dict[str, Any]] = []
                for order_index in range(batch_start, batch_end):
                    self.maybe_stop()
                    candidate_id = str(uuid.uuid4())
                    seed = self._candidate_seed(task_row, order_index)
                    image_path = task_dir / f"{order_index:02d}_{candidate_id}.webp"
                    state_path = task_dir / f"{order_index:02d}_{candidate_id}.pt"
                    preview_config = self._build_preview_image_config(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        image_path=image_path,
                    )
                    specs.append(
                        {
                            "order_index": order_index,
                            "candidate_id": candidate_id,
                            "image_path": image_path,
                            "state_path": state_path,
                            "preview_config": preview_config,
                        }
                    )

                states = self._generate_candidate_states_batch(
                    task_id=task_id,
                    candidate_ids=[spec["candidate_id"] for spec in specs],
                    gen_configs=[spec["preview_config"] for spec in specs],
                    num_inference_steps=specs[0]["preview_config"].num_inference_steps,
                    sampler=sampler,
                    scheduler=scheduler,
                    batch_start_index=batch_start,
                    total_candidates=self.flow_grpo_config.group_size,
                )
                for spec, state in zip(specs, states):
                    order_index = int(spec["order_index"])
                    preview_config = spec["preview_config"]
                    preview_image = self._save_rollout_preview_image(
                        final_latents=state.next_latents[:, -1],
                        preview_config=preview_config,
                    )
                    preview_config.log_image(preview_image, order_index)
                    if hasattr(self.sd, "_after_sample_image"):
                        self.sd._after_sample_image(order_index, self.flow_grpo_config.group_size)
                    self._save_candidate_state(state, spec["state_path"])
                    candidate_rows.append(
                        (
                            state.candidate_id,
                            self.job_id,
                            task_id,
                            order_index,
                            state.prompt,
                            state.negative_prompt,
                            state.seed,
                            state.guidance_scale,
                            state.num_inference_steps,
                            sampler,
                            scheduler,
                            str(spec["image_path"]),
                            str(spec["state_path"]),
                            "open",
                        )
                    )

            self._db_execute_many(
                """
                INSERT INTO FlowGRPOCandidate (
                    id, job_id, vote_task_id, order_index, prompt, negative_prompt, seed,
                    guidance_scale, num_inference_steps, sampler, scheduler, image_path, state_path, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                candidate_rows,
            )
            self._db_execute_write(
                "UPDATE FlowGRPOVoteTask SET status = 'open' WHERE id = ? AND job_id = ?",
                (task_id, self.job_id),
            )
            self._task_counter += 1
        except Exception as exc:
            self._db_execute_write(
                "UPDATE FlowGRPOVoteTask SET status = 'failed', error = ? WHERE id = ? AND job_id = ?",
                (str(exc), task_id, self.job_id),
            )
            self._db_execute_write(
                "UPDATE FlowGRPOCandidate SET status = 'failed' WHERE vote_task_id = ?",
                (task_id,),
            )
            raise
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            adapter = getattr(self.sd, "adapter", None)
            if isinstance(adapter, ReferenceAdapter):
                adapter.clear_memory()

    def _promote_requested_vote_tasks(self) -> None:
        active_count = self._count_active_vote_tasks()
        while active_count < 1:
            task_row = self._next_requested_task()
            if task_row is None:
                return
            self._generate_requested_task(task_row)
            active_count += 1

    def _next_voted_task_group(self) -> Optional[list[str]]:
        rows = self._db_execute(
            """
            SELECT task.*
            FROM FlowGRPOVoteTask task
            WHERE task.job_id = ?
              AND task.status = 'voted'
              AND (
                SELECT COUNT(*)
                FROM FlowGRPOCandidate candidate
                WHERE candidate.vote_task_id = task.id
              ) >= 2
              AND NOT EXISTS (
                SELECT 1
                FROM FlowGRPOCandidate candidate
                WHERE candidate.vote_task_id = task.id
                  AND NOT EXISTS (
                    SELECT 1
                    FROM FlowGRPOVote vote
                    WHERE vote.vote_task_id = task.id
                      AND vote.candidate_id = candidate.id
                      AND vote.processed = 0
                  )
              )
            ORDER BY task.created_at ASC
            LIMIT ?
            """,
            (self.job_id, 1),
        )
        if not rows:
            return None
        return [row["id"] for row in rows]

    def _load_task_rows(self, task_id: str) -> tuple[sqlite3.Row, list[sqlite3.Row], list[sqlite3.Row]]:
        task_rows = self._db_execute(
            "SELECT * FROM FlowGRPOVoteTask WHERE id = ? AND job_id = ? LIMIT 1",
            (task_id, self.job_id),
        )
        if not task_rows:
            raise ValueError(f"Flow-GRPO vote task '{task_id}' was not found.")
        candidate_rows = self._db_execute(
            "SELECT * FROM FlowGRPOCandidate WHERE vote_task_id = ? ORDER BY order_index ASC",
            (task_id,),
        )
        vote_rows = self._db_execute(
            "SELECT * FROM FlowGRPOVote WHERE vote_task_id = ? AND processed = 0 ORDER BY created_at ASC",
            (task_id,),
        )
        return task_rows[0], candidate_rows, vote_rows

    def _mark_task_processed(self, task_id: str, task_status: str, candidate_status: str) -> None:
        self._db_execute_write(
            "UPDATE FlowGRPOVoteTask SET status = ? WHERE id = ?",
            (task_status, task_id),
        )
        self._db_execute_write(
            "UPDATE FlowGRPOCandidate SET status = ? WHERE vote_task_id = ?",
            (candidate_status, task_id),
        )
        self._db_execute_write(
            "UPDATE FlowGRPOVote SET processed = 1 WHERE vote_task_id = ?",
            (task_id,),
        )

    @contextlib.contextmanager
    def _lora_disabled(self):
        if self.network is None:
            yield
            return
        was_active = bool(getattr(self.network, "is_active", True))
        self.network.is_active = False
        try:
            yield
        finally:
            self.network.is_active = was_active

    def _candidate_train_steps(self, candidate_state: CandidateState) -> int:
        total_steps = candidate_state.log_probs.shape[1]
        return max(
            1,
            int(total_steps * max(0.0, min(self.flow_grpo_config.timestep_fraction, 1.0))),
        )

    def _candidate_batch_key(self, candidate_state: CandidateState) -> tuple[Any, ...]:
        return (
            candidate_state.sampler,
            candidate_state.scheduler,
            int(candidate_state.num_inference_steps),
            float(candidate_state.guidance_scale),
            tuple(candidate_state.latents.shape[1:]),
            tuple(candidate_state.next_latents.shape[1:]),
            tuple(candidate_state.timesteps.shape),
            tuple(float(timestep) for timestep in candidate_state.timesteps.detach().cpu().tolist()),
            tuple(candidate_state.log_probs.shape[1:]),
            candidate_state.unconditional_embeds is not None,
            float(candidate_state.network_multiplier),
            tuple(candidate_state.adapter_conditioning.shape) if candidate_state.adapter_conditioning is not None else None,
            _tensor_tree_shape(candidate_state.control_tensor),
            float(candidate_state.adapter_conditioning_scale),
        )

    @staticmethod
    def _expand_like(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = value
        while out.ndim < ref.ndim:
            out = out.unsqueeze(-1)
        return out

    def _compute_batched_unit_logprob(
        self,
        rollout_scheduler: Any,
        train_units: list[CandidateTrainUnit],
        *,
        disable_lora: bool = False,
        manage_assistant_lora: bool = True,
        manage_network_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_state = train_units[0].candidate_state
        latents = torch.cat(
            [
                train_unit.candidate_state.latents[:, train_unit.step_index].to(
                    self.device_torch, dtype=self.sd.torch_dtype
                )
                for train_unit in train_units
            ],
            dim=0,
        )
        timesteps = torch.cat(
            [
                train_unit.candidate_state.timesteps[train_unit.step_index].reshape(1).to(
                    self.device_torch, dtype=torch.float32
                )
                for train_unit in train_units
            ],
            dim=0,
        )
        conditional_embeds = concat_prompt_embeds(
            [
                train_unit.candidate_state.conditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype)
                for train_unit in train_units
            ]
        )
        unconditional_embeds = None
        if first_state.unconditional_embeds is not None:
            unconditional_embeds = concat_prompt_embeds(
                [
                    train_unit.candidate_state.unconditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype)
                    for train_unit in train_units
                ]
            )
        adapter_conditioning = None
        if first_state.adapter_conditioning is not None:
            adapter_conditioning = torch.cat(
                [
                    train_unit.candidate_state.adapter_conditioning.to(self.device_torch, dtype=self.sd.torch_dtype)
                    for train_unit in train_units
                ],
                dim=0,
            )
        control_tensor = None
        if first_state.control_tensor is not None:
            control_tensor = _concat_tensor_tree(
                [train_unit.candidate_state.control_tensor for train_unit in train_units],
                self.device_torch,
                self.sd.torch_dtype,
            )

        with contextlib.ExitStack() as stack:
            if manage_assistant_lora:
                stack.enter_context(self._sample_assistant_lora_state(offload_on_exit=disable_lora))
            if manage_network_state:
                stack.enter_context(
                    self._rollout_network_state(
                        network_multiplier=float(first_state.network_multiplier),
                        active=not disable_lora,
                        allow_merge=False,
                    )
                )
            rollout_batch = self._build_rollout_batch(
                control_tensor=control_tensor,
                latents=latents,
            )
            model_latents = self._condition_rollout_latents(latents, rollout_batch)
            adapter_predict_kwargs = self._build_adapter_predict_kwargs(
                latents=latents,
                timesteps=timesteps,
                conditional_embeds=conditional_embeds,
                adapter_conditioning=adapter_conditioning,
                adapter_conditioning_scale=float(first_state.adapter_conditioning_scale),
            )
            predict_kwargs: dict[str, Any] = {
                "latents": model_latents,
                "conditional_embeddings": conditional_embeds,
                "unconditional_embeddings": unconditional_embeds,
                "timestep": timesteps,
                "guidance_scale": float(first_state.guidance_scale),
                "batch": rollout_batch,
                **adapter_predict_kwargs,
            }
            predict_signature = inspect.signature(self.sd.predict_noise)
            if "requires_grad" in predict_signature.parameters:
                predict_kwargs["requires_grad"] = (not disable_lora)

            noise_pred = self.sd.predict_noise(
                **predict_kwargs,
            )
            if not disable_lora and not noise_pred.requires_grad:
                raise RuntimeError(
                    "Flow-GRPO policy recomputation produced a detached prediction. "
                    "This model's predict_noise/get_noise_prediction path must run the denoiser with gradients enabled."
                )
            prev_sample = torch.cat(
                [
                    train_unit.candidate_state.next_latents[:, train_unit.step_index].to(
                        self.device_torch, dtype=self.sd.torch_dtype
                    )
                    for train_unit in train_units
                ],
                dim=0,
            )

            step_indices = torch.as_tensor(
                [train_unit.step_index for train_unit in train_units],
                device=self.device_torch,
                dtype=torch.long,
            )
            sigmas = rollout_scheduler.sigmas.to(device=self.device_torch, dtype=torch.float32)
            sigma = sigmas[step_indices].to(device=latents.device, dtype=torch.float32)
            sigma_next = sigmas[step_indices + 1].to(device=latents.device, dtype=torch.float32)
            sample_f = latents.to(torch.float32)
            model_output_f = noise_pred.to(torch.float32)
            prev_sample_f = prev_sample.to(torch.float32)
            sigma_e = self._expand_like(sigma, sample_f)
            sigma_next_e = self._expand_like(sigma_next, sample_f)
            eps = 1e-8

            if self.flow_grpo_config.sde_type == "sde":
                dt = sigma_next_e - sigma_e
                sigma_safe = torch.clamp(sigma_e, min=eps)
                one_minus_sigma = torch.clamp(1.0 - sigma_e, min=eps)
                std_dev_t = torch.sqrt(sigma_safe / one_minus_sigma) * float(self.flow_grpo_config.noise_level)
                step_scale = torch.sqrt(torch.clamp(-dt, min=eps))
                prev_mean = (
                    sample_f * (1.0 + ((std_dev_t**2) / (2.0 * sigma_safe)) * dt)
                    + model_output_f * (1.0 + ((std_dev_t**2) * (1.0 - sigma_e) / (2.0 * sigma_safe))) * dt
                )
                transition_std = torch.clamp(std_dev_t * step_scale, min=eps)
                sqrt_2pi = torch.sqrt(torch.as_tensor(2.0 * np.pi, device=sample_f.device, dtype=torch.float32))
                log_prob_tensor = (
                    -((prev_sample_f.detach() - prev_mean) ** 2) / (2.0 * (transition_std**2))
                    - torch.log(transition_std)
                    - torch.log(sqrt_2pi)
                )
            elif self.flow_grpo_config.sde_type == "cps":
                std_dev_t = torch.clamp(
                    sigma_next_e * np.sin(float(self.flow_grpo_config.noise_level) * np.pi / 2.0),
                    min=eps,
                )
                pred_original_sample = sample_f - sigma_e * model_output_f
                noise_estimate = sample_f + model_output_f * (1.0 - sigma_e)
                sqrt_term = torch.sqrt(torch.clamp((sigma_next_e**2) - (std_dev_t**2), min=eps))
                prev_mean = pred_original_sample * (1.0 - sigma_next_e) + noise_estimate * sqrt_term
                transition_std = std_dev_t
                log_prob_tensor = -((prev_sample_f.detach() - prev_mean) ** 2)
            else:
                raise ValueError(f"Unsupported Flow-GRPO sde_type '{self.flow_grpo_config.sde_type}'.")

            log_prob = log_prob_tensor.mean(dim=tuple(range(1, log_prob_tensor.ndim)))
        return log_prob, prev_mean, transition_std

    def _backward_train_unit_batch_loss(
        self,
        train_units: list[CandidateTrainUnit],
        active_candidate_count: int,
        progress_bar: Optional[tqdm] = None,
    ) -> tuple[Optional[float], dict[str, float]]:
        total_loss_value: Optional[float] = None
        policy_losses: list[float] = []
        kl_losses: list[float] = []
        approx_kls: list[float] = []
        clipfracs: list[float] = []
        rollout_seed_latents = torch.cat(
            [
                train_unit.candidate_state.latents[:, 0].to(self.device_torch, dtype=self.sd.torch_dtype)
                for train_unit in train_units
            ],
            dim=0,
        )
        first_state = train_units[0].candidate_state
        policy_scheduler, _, _, _ = self._prepare_scheduler_run(
            sampler=first_state.sampler,
            scheduler_name=first_state.scheduler,
            num_inference_steps=first_state.num_inference_steps,
            latents=rollout_seed_latents,
        )
        reference_scheduler = None
        if self.flow_grpo_config.beta > 0.0:
            reference_scheduler, _, _, _ = self._prepare_scheduler_run(
                sampler=first_state.sampler,
                scheduler_name=first_state.scheduler,
                num_inference_steps=first_state.num_inference_steps,
                latents=rollout_seed_latents,
            )

        with (
            self._sample_assistant_lora_state(offload_on_exit=True),
            self._rollout_network_state(
                network_multiplier=float(first_state.network_multiplier),
                active=True,
                allow_merge=False,
            ),
        ):
            log_prob, prev_mean, transition_std = self._compute_batched_unit_logprob(
                policy_scheduler,
                train_units,
                manage_assistant_lora=False,
                manage_network_state=False,
            )
            old_log_prob = torch.cat(
                [
                    train_unit.candidate_state.log_probs[:, train_unit.step_index].to(
                        self.device_torch, dtype=log_prob.dtype
                    )
                    for train_unit in train_units
                ],
                dim=0,
            )

            advantage_tensor = torch.as_tensor(
                [train_unit.advantage for train_unit in train_units],
                device=self.device_torch,
                dtype=log_prob.dtype,
            ).reshape_as(log_prob)
            advantage_tensor = torch.clamp(
                advantage_tensor,
                -self.flow_grpo_config.adv_clip_max,
                self.flow_grpo_config.adv_clip_max,
            )

            ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0))
            unclipped = -advantage_tensor * ratio
            clipped = -advantage_tensor * torch.clamp(
                ratio,
                1.0 - self.flow_grpo_config.clip_range,
                1.0 + self.flow_grpo_config.clip_range,
            )
            policy_loss_by_sample = torch.maximum(unclipped, clipped)
            loss_by_sample = policy_loss_by_sample

            if self.flow_grpo_config.beta > 0.0:
                if reference_scheduler is None:
                    raise RuntimeError("Flow-GRPO reference scheduler was not initialized.")
                with torch.no_grad():
                    _, ref_prev_mean, _ = self._compute_batched_unit_logprob(
                        reference_scheduler,
                        train_units,
                        disable_lora=True,
                        manage_assistant_lora=False,
                    )
                std_safe = torch.clamp(transition_std, min=1e-6)
                kl_tensor = ((prev_mean - ref_prev_mean) ** 2) / (2.0 * (std_safe**2))
                kl_loss_by_sample = kl_tensor.flatten(start_dim=1).sum(dim=1)
                loss_by_sample = loss_by_sample + self.flow_grpo_config.beta * kl_loss_by_sample
                kl_losses.append(float(kl_loss_by_sample.detach().mean().cpu().item()))

            per_unit_scales = torch.as_tensor(
                [1.0 / float(train_unit.train_steps) for train_unit in train_units],
                device=self.device_torch,
                dtype=loss_by_sample.dtype,
            ).reshape_as(loss_by_sample)
            scaled_loss = (loss_by_sample * per_unit_scales).sum() / float(active_candidate_count)
            total_loss_value = float(scaled_loss.detach().cpu().item())
            self.accelerator.backward(scaled_loss)

        policy_losses.append(float(policy_loss_by_sample.detach().mean().cpu().item()))
        approx_kls.append(float((0.5 * torch.mean((log_prob - old_log_prob) ** 2)).detach().cpu().item()))
        clipfracs.append(
            float(
                torch.mean((torch.abs(ratio - 1.0) > self.flow_grpo_config.clip_range).float())
                .detach()
                .cpu()
                .item()
            )
        )
        if progress_bar is not None:
            step_min = min(train_unit.step_index for train_unit in train_units) + 1
            step_max = max(train_unit.step_index for train_unit in train_units) + 1
            candidate_min = min(train_unit.candidate_index for train_unit in train_units)
            candidate_max = max(train_unit.candidate_index for train_unit in train_units)
            progress_bar.set_postfix_str(
                f"candidates {candidate_min}-{candidate_max} steps {step_min}-{step_max} batch {len(train_units)}"
            )
            progress_bar.update(1)

        metrics = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        }
        del (
            log_prob,
            prev_mean,
            transition_std,
            old_log_prob,
            advantage_tensor,
            ratio,
            unclipped,
            clipped,
            policy_loss_by_sample,
            loss_by_sample,
            per_unit_scales,
            scaled_loss,
        )
        if self.flow_grpo_config.beta > 0.0:
            del ref_prev_mean, std_safe, kl_tensor, kl_loss_by_sample
        return total_loss_value, metrics

    def _compute_group_advantages(
        self,
        candidate_rows: list[sqlite3.Row],
        vote_rows: list[sqlite3.Row],
    ) -> dict[str, float]:
        reward_by_candidate: dict[str, list[float]] = {
            candidate_row["id"]: []
            for candidate_row in candidate_rows
        }
        for vote_row in vote_rows:
            candidate_id = vote_row["candidate_id"]
            if candidate_id in reward_by_candidate:
                reward_by_candidate[candidate_id].append(float(vote_row["reward"]))

        ordered_rewards = np.asarray(
            [
                float(np.mean(reward_by_candidate[candidate_row["id"]]))
                if reward_by_candidate[candidate_row["id"]]
                else 0.0
                for candidate_row in candidate_rows
            ],
            dtype=np.float32,
        )
        if ordered_rewards.size == 0:
            return {}

        reward_mean = float(np.mean(ordered_rewards))
        reward_std = float(np.std(ordered_rewards))
        if reward_std <= 1e-6:
            normalized = np.zeros_like(ordered_rewards)
        else:
            normalized = (ordered_rewards - reward_mean) / reward_std

        return {
            candidate_row["id"]: float(normalized[idx])
            for idx, candidate_row in enumerate(candidate_rows)
        }

    def _process_voted_task_group(self, task_ids: list[str]) -> Optional[dict[str, float]]:
        candidate_rows: list[sqlite3.Row] = []
        vote_rows: list[sqlite3.Row] = []
        for task_id in task_ids:
            _, task_candidate_rows, task_vote_rows = self._load_task_rows(task_id)
            candidate_rows.extend(task_candidate_rows)
            vote_rows.extend(task_vote_rows)

        if not vote_rows:
            return None

        candidate_row_by_id = {row["id"]: row for row in candidate_rows}
        train_votes = [vote_row for vote_row in vote_rows if vote_row["candidate_id"] in candidate_row_by_id]
        if not train_votes:
            for task_id in task_ids:
                self._mark_task_processed(task_id, "processed", "processed")
            return None

        if self.network is None:
            raise ValueError("Flow-GRPO trainer requires a trainable LoRA target.")

        self.network.train()
        self.network.is_active = True
        self.optimizer.zero_grad(set_to_none=True)

        total_loss_value: Optional[float] = None
        metric_accumulator = {
            "policy_loss": [],
            "kl_loss": [],
            "approx_kl": [],
            "clipfrac": [],
        }

        candidate_advantages = self._compute_group_advantages(candidate_rows, train_votes)

        active_advantages = [
            (candidate_id, advantage)
            for candidate_id, advantage in candidate_advantages.items()
            if abs(advantage) > 1e-8
        ]

        train_batch_size = max(1, int(getattr(self.train_config, "batch_size", 1) or 1))
        candidate_groups: OrderedDict[tuple[Any, ...], list[tuple[int, CandidateState, float, int]]] = OrderedDict()
        loaded_candidate_states: list[CandidateState] = []
        for candidate_index, (candidate_id, advantage) in enumerate(active_advantages, start=1):
            candidate_state = self._load_candidate_state(candidate_row_by_id[candidate_id]["state_path"])
            loaded_candidate_states.append(candidate_state)
            candidate_key = self._candidate_batch_key(candidate_state)
            train_steps = self._candidate_train_steps(candidate_state)
            candidate_groups.setdefault(candidate_key, [])
            candidate_groups[candidate_key].append((candidate_index, candidate_state, advantage, train_steps))

        train_unit_batches: list[list[CandidateTrainUnit]] = []
        for candidate_group in candidate_groups.values():
            train_units: list[CandidateTrainUnit] = []
            max_train_steps = max(train_steps for _, _, _, train_steps in candidate_group)
            for step_index in range(max_train_steps):
                for candidate_index, candidate_state, advantage, train_steps in candidate_group:
                    if step_index >= train_steps:
                        continue
                    train_units.append(
                        CandidateTrainUnit(
                            candidate_index=candidate_index,
                            candidate_state=candidate_state,
                            advantage=advantage,
                            step_index=step_index,
                            train_steps=train_steps,
                        )
                    )
            for start_index in range(0, len(train_units), train_batch_size):
                train_unit_batches.append(train_units[start_index : start_index + train_batch_size])

        with tqdm(
            total=len(train_unit_batches),
            desc="Flow-GRPO train batches",
            leave=True,
        ) as progress_bar:
            for train_units in train_unit_batches:
                unit_count = len(train_units)
                candidate_loss_value, metrics = self._backward_train_unit_batch_loss(
                    train_units,
                    active_candidate_count=len(active_advantages),
                    progress_bar=progress_bar,
                )
                if candidate_loss_value is None:
                    continue
                total_loss_value = (
                    candidate_loss_value
                    if total_loss_value is None
                    else total_loss_value + candidate_loss_value
                )
                for key, value in metrics.items():
                    metric_accumulator[key].extend([value] * unit_count)
        train_unit_batches.clear()
        loaded_candidate_states.clear()

        if total_loss_value is None:
            for task_id in task_ids:
                self._mark_task_processed(task_id, "processed", "processed")
            return None

        loss_value = float(total_loss_value)
        if self.params and self.train_config.optimizer != "adafactor":
            if isinstance(self.params[0], dict):
                for param_group in self.params:
                    self.accelerator.clip_grad_norm_(param_group["params"], self.train_config.max_grad_norm)
            else:
                self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
        self.optimizer.step()
        if self.adapter is not None and hasattr(self.adapter, "post_weight_update"):
            self.adapter.post_weight_update()
        if self.ema is not None:
            self.ema.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        for task_id in task_ids:
            self._mark_task_processed(task_id, "processed", "processed")
        metrics = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in metric_accumulator.items()
        }
        metrics["loss"] = loss_value
        return metrics

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self._promote_requested_vote_tasks()

    def hook_train_loop(self, batch):
        while True:
            self.maybe_stop()
            self._promote_requested_vote_tasks()
            task_ids = self._next_voted_task_group()
            if task_ids is None:
                self.update_status("running", "Waiting for Flow-GRPO votes")
                time.sleep(2.0)
                continue

            self.update_status("running", "Applying Flow-GRPO vote group")
            metrics = self._process_voted_task_group(task_ids)
            if metrics is None:
                continue
            self.update_status("running", "Waiting for Flow-GRPO votes")
            return metrics

    def get_training_info(self):
        info = super().get_training_info()
        info["flow_grpo_open_tasks"] = self._count_open_tasks()
        return info
