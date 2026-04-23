from __future__ import annotations

import contextlib
import inspect
import os
import sqlite3
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from PIL import Image

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.prompt_utils import PromptEmbeds


@dataclass
class FlowGRPOTrainerConfig:
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    beta: float = 0.0
    noise_level: float = 0.7
    sde_type: Literal["sde", "cps"] = "sde"
    timestep_fraction: float = 1.0
    candidates_per_task: int = 4
    max_pending_tasks: int = 1
    poll_interval_sec: float = 2.0

    def __init__(self, **kwargs):
        self.clip_range = float(kwargs.get("clip_range", self.clip_range))
        self.adv_clip_max = float(kwargs.get("adv_clip_max", self.adv_clip_max))
        self.beta = float(kwargs.get("beta", self.beta))
        self.noise_level = float(kwargs.get("noise_level", self.noise_level))
        self.sde_type = kwargs.get("sde_type", self.sde_type)
        self.timestep_fraction = float(kwargs.get("timestep_fraction", self.timestep_fraction))
        self.candidates_per_task = max(2, int(kwargs.get("candidates_per_task", self.candidates_per_task)))
        self.max_pending_tasks = max(1, int(kwargs.get("max_pending_tasks", self.max_pending_tasks)))
        self.poll_interval_sec = max(0.25, float(kwargs.get("poll_interval_sec", self.poll_interval_sec)))


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


def _clone_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone().cpu()


def _clone_prompt_embeds(prompt_embeds: Optional[PromptEmbeds]) -> Optional[PromptEmbeds]:
    if prompt_embeds is None:
        return None
    return prompt_embeds.detach().clone().to("cpu")


def _expand_like(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = value
    while out.ndim < ref.ndim:
        out = out.unsqueeze(-1)
    return out


def _flowmatch_step_with_logprob(
    *,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    noise_level: float,
    sde_type: str,
    prev_sample: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model_output = model_output.float()
    sample = sample.float()
    sigma = sigma.float().flatten()
    sigma_next = sigma_next.float().flatten()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sigma_e = _expand_like(sigma, sample)
    sigma_next_e = _expand_like(sigma_next, sample)
    dt = sigma_next_e - sigma_e

    eps = 1e-8
    sqrt_2pi = torch.sqrt(torch.as_tensor(2.0 * np.pi, device=sample.device, dtype=torch.float32))

    if sde_type == "sde":
        sigma_safe = torch.clamp(sigma_e, min=eps)
        one_minus_sigma = torch.clamp(1.0 - sigma_e, min=eps)
        std_dev_t = torch.sqrt(sigma_safe / one_minus_sigma) * float(noise_level)
        dt_safe = torch.clamp(-dt, min=eps)
        step_scale = torch.sqrt(dt_safe)

        prev_sample_mean = (
            sample * (1.0 + ((std_dev_t**2) / (2.0 * sigma_safe)) * dt)
            + model_output * (1.0 + ((std_dev_t**2) * (1.0 - sigma_e) / (2.0 * sigma_safe))) * dt
        )
        if prev_sample is None:
            prev_sample = prev_sample_mean + std_dev_t * step_scale * torch.randn_like(model_output)

        transition_std = torch.clamp(std_dev_t * step_scale, min=eps)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * (transition_std**2))
            - torch.log(transition_std)
            - torch.log(sqrt_2pi)
        )
    elif sde_type == "cps":
        std_dev_t = torch.clamp(
            sigma_next_e * np.sin(float(noise_level) * np.pi / 2.0),
            min=eps,
        )
        pred_original_sample = sample - sigma_e * model_output
        noise_estimate = sample + model_output * (1.0 - sigma_e)
        sqrt_term = torch.sqrt(torch.clamp((sigma_next_e**2) - (std_dev_t**2), min=eps))
        prev_sample_mean = pred_original_sample * (1.0 - sigma_next_e) + noise_estimate * sqrt_term
        if prev_sample is None:
            prev_sample = prev_sample_mean + std_dev_t * torch.randn_like(model_output)
        transition_std = std_dev_t
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * (transition_std**2))
            - torch.log(transition_std)
            - torch.log(sqrt_2pi)
        )
    else:
        raise ValueError(f"Unsupported Flow-GRPO sde_type '{sde_type}'.")

    reduce_dims = tuple(range(1, log_prob.ndim))
    log_prob = log_prob.mean(dim=reduce_dims)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


class FlowGRPOTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        if not self.is_ui_trainer:
            raise ValueError("flow_grpo_trainer requires the UI SQLite runtime.")
        self.train_config.disable_sampling = True
        self.flow_grpo_config = FlowGRPOTrainerConfig(**self.config.get("grpo", {}))
        self._prompt_index = 0
        self._task_counter = 0
        self._flow_grpo_root = Path(self.save_root) / "flow_grpo"
        self._candidate_root = self._flow_grpo_root / "candidates"
        self._candidate_root.mkdir(parents=True, exist_ok=True)

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
            return int(task_seed) + order_index + (self._task_counter * 1000)
        return int(self.sample_config.seed) + order_index + self.step_num + (self._task_counter * 1000)

    def _prepare_scheduler_run(
        self,
        *,
        num_inference_steps: int,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scheduler = self.sd.noise_scheduler
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
                kwargs["patch_size"] = 2 if self.model_config.is_flux else 1
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
            sigma_current[0] = (sigma_current[0] + sigma_next[0]) * 0.5
        return timesteps, sigma_current, sigma_next

    def _decode_preview_image(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.sd.decode_latents(latents, device=self.device_torch, dtype=self.sd.torch_dtype)
        if decoded.ndim == 5:
            decoded = decoded[:, :, 0, :, :]
        if decoded.shape[1] == 1:
            decoded = decoded.repeat(1, 3, 1, 1)
        if decoded.shape[1] > 3:
            decoded = decoded[:, :3, :, :]
        return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)

    def _save_preview_image(self, image_tensor: torch.Tensor, image_path: Path) -> None:
        image = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(image).save(image_path)

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
    ) -> tuple[CandidateState, torch.Tensor]:
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

        latents = self.sd.get_latent_noise(pixel_height=height, pixel_width=width, batch_size=1).to(
            self.device_torch, dtype=self.sd.torch_dtype
        )
        timesteps, sigma_current, sigma_next = self._prepare_scheduler_run(
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
                )

            next_latents, log_prob, _, _ = _flowmatch_step_with_logprob(
                model_output=noise_pred,
                sample=latents,
                sigma=sigma_current[idx].reshape(1).repeat(latents.shape[0]),
                sigma_next=sigma_next[idx].reshape(1).repeat(latents.shape[0]),
                noise_level=self.flow_grpo_config.noise_level,
                sde_type=self.flow_grpo_config.sde_type,
            )
            latents_before.append(_clone_tensor(latents))
            latents_after.append(_clone_tensor(next_latents))
            log_probs.append(_clone_tensor(log_prob))
            latents = next_latents

        preview = self._decode_preview_image(latents).detach().cpu()
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
        return state, preview

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
        requested_candidates = max(2, int(task_row["requested_candidates"] or self.flow_grpo_config.candidates_per_task))
        width = int(task_row["width"] or self.sample_config.width)
        height = int(task_row["height"] or self.sample_config.height)
        guidance_scale = float(task_row["guidance_scale"] or self.sample_config.guidance_scale)
        num_inference_steps = int(task_row["num_inference_steps"] or self.sample_config.sample_steps)
        sampler = task_row["sampler"] or self.sample_config.sampler or "flowmatch"
        scheduler = task_row["scheduler"] or self.train_config.noise_scheduler or "flowmatch"
        task_dir = self._task_dir(task_id)

        self._db_execute_write(
            "UPDATE FlowGRPOVoteTask SET status = 'generating' WHERE id = ? AND job_id = ?",
            (task_id, self.job_id),
        )

        candidate_rows: list[tuple] = []
        try:
            for order_index in range(requested_candidates):
                self.update_status(
                    "running",
                    f"Generating Flow-GRPO candidates {order_index + 1}/{requested_candidates}",
                )
                candidate_id = str(uuid.uuid4())
                seed = self._candidate_seed(task_row, order_index)
                state, preview = self._generate_candidate_state(
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
                self._save_preview_image(preview, image_path)
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

            self._db_execute_many(
                """
                INSERT INTO FlowGRPOCandidate (
                    id, job_id, vote_task_id, order_index, prompt, negative_prompt, seed,
                    guidance_scale, num_inference_steps, sampler, scheduler, image_path, state_path, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                candidate_rows,
            )
            self._db_execute_write(
                "UPDATE FlowGRPOVoteTask SET status = 'open' WHERE id = ? AND job_id = ?",
                (task_id, self.job_id),
            )
            self._task_counter += 1
        except Exception:
            self._db_execute_write(
                "UPDATE FlowGRPOVoteTask SET status = 'failed' WHERE id = ? AND job_id = ?",
                (task_id, self.job_id),
            )
            self._db_execute_write(
                "UPDATE FlowGRPOCandidate SET status = 'failed' WHERE vote_task_id = ?",
                (task_id,),
            )
            raise

    def _promote_requested_vote_tasks(self) -> None:
        active_count = self._count_active_vote_tasks()
        while active_count < self.flow_grpo_config.max_pending_tasks:
            task_row = self._next_requested_task()
            if task_row is None:
                return
            self._generate_requested_task(task_row)
            active_count += 1

    def _next_voted_task_id(self) -> Optional[str]:
        rows = self._db_execute(
            """
            SELECT id
            FROM FlowGRPOVoteTask
            WHERE job_id = ? AND status = 'voted'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (self.job_id,),
        )
        return rows[0]["id"] if rows else None

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
            noise_pred = self.sd.predict_noise(
                latents=latents,
                conditional_embeddings=candidate_state.conditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype),
                unconditional_embeddings=(
                    candidate_state.unconditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype)
                    if candidate_state.unconditional_embeds is not None
                    else None
                ),
                timestep=timestep,
                guidance_scale=float(candidate_state.guidance_scale),
            )
            _, log_prob, prev_mean, std_dev_t = _flowmatch_step_with_logprob(
                model_output=noise_pred,
                sample=latents,
                sigma=candidate_state.sigma_current[step_index].reshape(1).repeat(latents.shape[0]).to(self.device_torch),
                sigma_next=candidate_state.sigma_next[step_index].reshape(1).repeat(latents.shape[0]).to(self.device_torch),
                noise_level=self.flow_grpo_config.noise_level,
                sde_type=self.flow_grpo_config.sde_type,
                prev_sample=candidate_state.next_latents[:, step_index].to(self.device_torch, dtype=self.sd.torch_dtype),
            )
        return log_prob, prev_mean, std_dev_t

    def _compute_candidate_loss(
        self,
        candidate_state: CandidateState,
        advantage: float,
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        total_steps = candidate_state.log_probs.shape[1]
        train_steps = max(
            1,
            int(total_steps * max(0.0, min(self.flow_grpo_config.timestep_fraction, 1.0))),
        )
        train_indices = list(range(train_steps))
        total_loss: Optional[torch.Tensor] = None
        policy_losses: list[float] = []
        kl_losses: list[float] = []
        approx_kls: list[float] = []
        clipfracs: list[float] = []

        for step_index in train_indices:
            log_prob, prev_mean, std_dev_t = self._compute_step_logprob(candidate_state, step_index)
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
                with torch.no_grad():
                    _, ref_prev_mean, _ = self._compute_step_logprob(candidate_state, step_index, disable_lora=True)
                std_safe = torch.clamp(std_dev_t, min=1e-6)
                kl_loss = (((prev_mean - ref_prev_mean) ** 2) / (2.0 * (std_safe**2))).mean()
                loss = loss + self.flow_grpo_config.beta * kl_loss
                kl_losses.append(float(kl_loss.detach().cpu().item()))

            scaled_loss = loss / float(len(train_indices))
            total_loss = scaled_loss if total_loss is None else total_loss + scaled_loss

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

        metrics = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        }
        return total_loss, metrics

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
            normalized = ordered_rewards - reward_mean
        else:
            normalized = (ordered_rewards - reward_mean) / reward_std

        return {
            candidate_row["id"]: float(normalized[idx])
            for idx, candidate_row in enumerate(candidate_rows)
        }

    def _process_voted_task(self, task_id: str) -> Optional[dict[str, float]]:
        _, candidate_rows, vote_rows = self._load_task_rows(task_id)
        if not vote_rows:
            return None

        if any(vote_row["value"] == "skip" for vote_row in vote_rows):
            self._mark_task_processed(task_id, "skipped", "skipped")
            return None

        candidate_row_by_id = {row["id"]: row for row in candidate_rows}
        train_votes = [vote_row for vote_row in vote_rows if vote_row["candidate_id"] in candidate_row_by_id]
        if not train_votes:
            self._mark_task_processed(task_id, "processed", "processed")
            return None

        if self.network is None:
            raise ValueError("Flow-GRPO trainer requires a trainable LoRA target.")

        self.network.train()
        self.network.is_active = True
        self.optimizer.zero_grad(set_to_none=True)

        total_loss: Optional[torch.Tensor] = None
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

        for candidate_id, advantage in active_advantages:
            candidate_state = self._load_candidate_state(candidate_row_by_id[candidate_id]["state_path"])
            candidate_loss, metrics = self._compute_candidate_loss(candidate_state, advantage)
            if candidate_loss is None:
                continue
            scaled_loss = candidate_loss / float(max(1, len(active_advantages)))
            total_loss = scaled_loss if total_loss is None else total_loss + scaled_loss
            for key, value in metrics.items():
                metric_accumulator[key].append(value)

        if total_loss is None:
            self._mark_task_processed(task_id, "processed", "processed")
            return None

        total_loss.backward()
        loss_value = float(total_loss.detach().cpu().item())
        params_to_clip = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if isinstance(param, torch.Tensor) and param.requires_grad:
                    params_to_clip.append(param)
        if params_to_clip:
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

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
            task_id = self._next_voted_task_id()
            if task_id is None:
                self.update_status("running", "Waiting for Flow-GRPO votes")
                time.sleep(self.flow_grpo_config.poll_interval_sec)
                continue

            self.update_status("running", "Applying Flow-GRPO vote")
            metrics = self._process_voted_task(task_id)
            if metrics is None:
                continue
            self.update_status("running", "Waiting for Flow-GRPO votes")
            return metrics

    def get_training_info(self):
        info = super().get_training_info()
        info["flow_grpo_open_tasks"] = self._count_open_tasks()
        return info
