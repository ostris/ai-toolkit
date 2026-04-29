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
from tqdm import tqdm

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.config_modules import GenerateImageConfig
from toolkit.prompt_utils import PromptEmbeds
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


class _EmptyRolloutBatch:
    control_tensor = None
    control_tensor_list = None


def _clone_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone().cpu()


def _clone_prompt_embeds(prompt_embeds: Optional[PromptEmbeds]) -> Optional[PromptEmbeds]:
    if prompt_embeds is None:
        return None
    return prompt_embeds.detach().clone().to("cpu")


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
        latents: torch.Tensor,
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
            latents=latents.detach().clone(),
            output_path=str(image_path),
            output_ext=image_path.suffix.lstrip("."),
            logger=self.logger,
            num_frames=getattr(self.sample_config, "num_frames", 1),
            fps=getattr(self.sample_config, "fps", 1),
            guidance_rescale=getattr(self.sample_config, "guidance_rescale", 0.0),
            adapter_conditioning_scale=getattr(self.sample_config, "adapter_conditioning_scale", 1.0),
            refiner_start_at=getattr(self.sample_config, "refiner_start_at", 0.5),
            do_cfg_norm=getattr(self.sample_config, "do_cfg_norm", False),
        )

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
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
        sampler: str,
        scheduler: str,
    ) -> CandidateState:
        network_was_active = bool(getattr(self.network, "is_active", True)) if self.network is not None else None
        network_was_training = bool(self.network.training) if self.network is not None else None
        try:
            with self._model_device_state_preset("generate"):
                if self.network is not None:
                    self.network.is_active = True
                    self.network.eval()

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                conditional_embeds = self.sd.encode_prompt(prompt).to(self.device_torch, dtype=self.sd.torch_dtype)
                unconditional_embeds = None
                if guidance_scale > 1.0:
                    unconditional_embeds = self.sd.encode_prompt(negative_prompt or "").to(
                        self.device_torch, dtype=self.sd.torch_dtype
                    )

                latents = self._get_rollout_initial_latents(height=height, width=width, batch_size=1).to(
                    self.device_torch, dtype=self.sd.torch_dtype
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

                for idx in range(timesteps.numel()):
                    self.maybe_stop()
                    timestep = timesteps[idx].reshape(1).repeat(latents.shape[0]).to(self.device_torch, dtype=torch.float32)
                    with torch.no_grad():
                        noise_pred = self.sd.predict_noise(
                            latents=latents,
                            conditional_embeddings=conditional_embeds,
                            unconditional_embeddings=unconditional_embeds,
                            timestep=timestep,
                            guidance_scale=float(guidance_scale),
                            batch=_EmptyRolloutBatch(),
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

                state = CandidateState(
                    task_id=task_id,
                    candidate_id=candidate_id,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=int(seed),
                    guidance_scale=float(guidance_scale),
                    num_inference_steps=int(num_inference_steps),
                    sampler=sampler,
                    scheduler=scheduler,
                    conditional_embeds=_clone_prompt_embeds(conditional_embeds),
                    unconditional_embeds=_clone_prompt_embeds(unconditional_embeds),
                    timesteps=_clone_tensor(timesteps),
                    sigma_current=_clone_tensor(sigma_current),
                    sigma_next=_clone_tensor(sigma_next),
                    latents=torch.stack(latents_before, dim=1),
                    next_latents=torch.stack(latents_after, dim=1),
                    log_probs=torch.stack(log_probs, dim=1),
                    reference_images=None,
                )
                return state
        finally:
            if self.network is not None:
                self.network.is_active = bool(network_was_active)
                if network_was_training:
                    self.network.train()
                else:
                    self.network.eval()

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
        preview_configs: list[GenerateImageConfig] = []
        try:
            self.update_status("running", "Generating Flow-GRPO rollout group")
            for order_index in range(self.flow_grpo_config.group_size):
                self.maybe_stop()
                candidate_id = str(uuid.uuid4())
                seed = self._candidate_seed(task_row, order_index)
                state = self._generate_candidate_state(
                    task_id=task_id,
                    candidate_id=candidate_id,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    sampler=sampler,
                    scheduler=scheduler,
                )
                image_path = task_dir / f"{order_index:02d}_{candidate_id}.webp"
                state_path = task_dir / f"{order_index:02d}_{candidate_id}.pt"
                preview_configs.append(
                    self._build_preview_image_config(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        latents=state.latents[:, 0].to(self.device_torch, dtype=self.sd.torch_dtype),
                        image_path=image_path,
                    )
                )
                self._save_candidate_state(state, state_path)
                candidate_rows.append(
                    (
                        candidate_id,
                        self.job_id,
                        task_id,
                        order_index,
                        prompt,
                        negative_prompt,
                        seed,
                        guidance_scale,
                        num_inference_steps,
                        sampler,
                        scheduler,
                        str(image_path),
                        str(state_path),
                        "open",
                    )
                )

            self.sd.generate_images(preview_configs, sampler=sampler)

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

    def _compute_step_logprob(
        self,
        rollout_scheduler: Any,
        candidate_state: CandidateState,
        step_index: int,
        *,
        disable_lora: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = candidate_state.latents[:, step_index].to(self.device_torch, dtype=self.sd.torch_dtype)
        timestep = candidate_state.timesteps[step_index].reshape(1).repeat(latents.shape[0]).to(
            self.device_torch,
            dtype=torch.float32,
        )

        context = self._lora_disabled() if disable_lora else contextlib.nullcontext()
        with context:
            predict_kwargs: dict[str, Any] = {
                "latents": latents,
                "conditional_embeddings": candidate_state.conditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype),
                "unconditional_embeddings": (
                    candidate_state.unconditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype)
                    if candidate_state.unconditional_embeds is not None
                    else None
                ),
                "timestep": timestep,
                "guidance_scale": float(candidate_state.guidance_scale),
                "batch": _EmptyRolloutBatch(),
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
            step_output = rollout_scheduler.step(
                model_output=noise_pred,
                timestep=candidate_state.timesteps[step_index].to(self.device_torch, dtype=torch.float32),
                sample=latents,
                noise_level=self.flow_grpo_config.noise_level,
                sde_type=self.flow_grpo_config.sde_type,
                prev_sample=candidate_state.next_latents[:, step_index].to(self.device_torch, dtype=self.sd.torch_dtype),
                return_log_prob=True,
                return_transition=True,
            )
            log_prob = step_output.log_prob
            prev_mean = step_output.transition_mean
            transition_std = step_output.transition_std
            if log_prob is None or prev_mean is None or transition_std is None:
                raise RuntimeError("Flow-GRPO scheduler did not return required transition outputs.")
        return log_prob, prev_mean, transition_std

    def _backward_candidate_loss(
        self,
        candidate_state: CandidateState,
        advantage: float,
        loss_scale: float,
        progress_bar: Optional[tqdm] = None,
        progress_label: str = "",
    ) -> tuple[Optional[float], dict[str, float]]:
        total_steps = candidate_state.log_probs.shape[1]
        train_steps = max(
            1,
            int(total_steps * max(0.0, min(self.flow_grpo_config.timestep_fraction, 1.0))),
        )
        train_indices = list(range(train_steps))
        total_loss_value: Optional[float] = None
        policy_losses: list[float] = []
        kl_losses: list[float] = []
        approx_kls: list[float] = []
        clipfracs: list[float] = []
        rollout_seed_latents = candidate_state.latents[:, 0].to(self.device_torch, dtype=self.sd.torch_dtype)
        policy_scheduler, _, _, _ = self._prepare_scheduler_run(
            sampler=candidate_state.sampler,
            scheduler_name=candidate_state.scheduler,
            num_inference_steps=candidate_state.num_inference_steps,
            latents=rollout_seed_latents,
        )
        reference_scheduler = None
        if self.flow_grpo_config.beta > 0.0:
            reference_scheduler, _, _, _ = self._prepare_scheduler_run(
                sampler=candidate_state.sampler,
                scheduler_name=candidate_state.scheduler,
                num_inference_steps=candidate_state.num_inference_steps,
                latents=rollout_seed_latents,
            )

        for step_index in train_indices:
            log_prob, prev_mean, transition_std = self._compute_step_logprob(
                policy_scheduler,
                candidate_state,
                step_index,
            )
            old_log_prob = candidate_state.log_probs[:, step_index].to(self.device_torch, dtype=log_prob.dtype)

            advantages = torch.full_like(log_prob, float(advantage))
            advantages = torch.clamp(
                advantages,
                -self.flow_grpo_config.adv_clip_max,
                self.flow_grpo_config.adv_clip_max,
            )

            ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0))
            unclipped = -advantages * ratio
            clipped = -advantages * torch.clamp(
                ratio,
                1.0 - self.flow_grpo_config.clip_range,
                1.0 + self.flow_grpo_config.clip_range,
            )
            policy_loss = torch.mean(torch.maximum(unclipped, clipped))
            loss = policy_loss

            if self.flow_grpo_config.beta > 0.0:
                if reference_scheduler is None:
                    raise RuntimeError("Flow-GRPO reference scheduler was not initialized.")
                with torch.no_grad():
                    _, ref_prev_mean, _ = self._compute_step_logprob(
                        reference_scheduler,
                        candidate_state,
                        step_index,
                        disable_lora=True,
                    )
                std_safe = torch.clamp(transition_std, min=1e-6)
                kl_tensor = ((prev_mean - ref_prev_mean) ** 2) / (2.0 * (std_safe**2))
                kl_loss = kl_tensor.flatten(start_dim=1).sum(dim=1).mean()
                loss = loss + self.flow_grpo_config.beta * kl_loss
                kl_losses.append(float(kl_loss.detach().cpu().item()))

            scaled_loss = loss * (float(loss_scale) / float(len(train_indices)))
            step_loss_value = float(scaled_loss.detach().cpu().item())
            total_loss_value = (
                step_loss_value
                if total_loss_value is None
                else total_loss_value + step_loss_value
            )
            self.accelerator.backward(scaled_loss)

            policy_losses.append(float(policy_loss.detach().cpu().item()))
            approx_kls.append(float((0.5 * torch.mean((log_prob - old_log_prob) ** 2)).detach().cpu().item()))
            clipfracs.append(
                float(
                    torch.mean((torch.abs(ratio - 1.0) > self.flow_grpo_config.clip_range).float())
                    .detach()
                    .cpu()
                    .item()
                )
            )
            del (
                log_prob,
                prev_mean,
                transition_std,
                old_log_prob,
                advantages,
                ratio,
                unclipped,
                clipped,
                policy_loss,
                loss,
                scaled_loss,
            )
            if self.flow_grpo_config.beta > 0.0:
                del ref_prev_mean, std_safe, kl_tensor, kl_loss
            if progress_bar is not None:
                if progress_label:
                    progress_bar.set_postfix_str(f"{progress_label} step {step_index + 1}/{len(train_indices)}")
                progress_bar.update(1)

        metrics = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        }
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

        total_train_timesteps = sum(
            max(
                1,
                int(
                    int(candidate_row_by_id[candidate_id]["num_inference_steps"])
                    * max(0.0, min(self.flow_grpo_config.timestep_fraction, 1.0))
                ),
            )
            for candidate_id, _ in active_advantages
        )

        with tqdm(
            total=total_train_timesteps,
            desc="Flow-GRPO train timesteps",
            leave=True,
        ) as progress_bar:
            for candidate_index, (candidate_id, advantage) in enumerate(active_advantages, start=1):
                candidate_state = self._load_candidate_state(candidate_row_by_id[candidate_id]["state_path"])
                candidate_loss_value, metrics = self._backward_candidate_loss(
                    candidate_state,
                    advantage,
                    loss_scale=1.0 / float(max(1, len(active_advantages))),
                    progress_bar=progress_bar,
                    progress_label=f"candidate {candidate_index}/{len(active_advantages)}",
                )
                if candidate_loss_value is None:
                    continue
                total_loss_value = (
                    candidate_loss_value
                    if total_loss_value is None
                    else total_loss_value + candidate_loss_value
                )
                for key, value in metrics.items():
                    metric_accumulator[key].append(value)
                del candidate_state

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
