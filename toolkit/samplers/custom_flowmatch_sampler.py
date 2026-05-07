import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
import torch
import numpy as np
from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0
        self.timestep_type = "linear"

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000
            # Bell-Shaped Mean-Normalized Timestep Weighting
            # bsmntw? need a better name

            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

            # Shift minimum to 0
            y_shifted = y - y.min()

            # Scale to make mean 1
            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # only do half bell
            hbsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # flatten second half to max
            hbsmntw_weighing[num_timesteps //
                             2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            # Create linear timesteps from 1000 to 1
            timesteps = torch.linspace(1000, 1, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False, timestep_type="linear") -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item()
                        for t in timesteps]

        # Get the weights for the timesteps
        if timestep_type == "weighted":
            weights = torch.tensor(
                [default_weighing_scheme[i] for i in step_indices],
                device=timesteps.device,
                dtype=timesteps.dtype
            )
        elif v2:
            weights = self.linear_timesteps_weights2[step_indices].flatten()
        else:
            weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        # forward ODE
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
        # reverse ODE
        # noisy_model_input = (1 - t_01) * noise + t_01 * original_samples
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type='linear',
        latents=None,
        patch_size=1
    ):
        self.timestep_type = timestep_type
        if timestep_type == 'linear' or timestep_type == 'weighted':
            timesteps = torch.linspace(1000, 1, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        elif timestep_type == 'sigmoid':
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = ((1 - t) * 1000)

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)

            return timesteps
        elif timestep_type in ['flux_shift', 'lumina2_shift', 'shift']:
            # matches inference dynamic shifting
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(
                    self.sigma_min), num_timesteps
            )

            sigmas = timesteps / self.config.num_train_timesteps

            if self.config.use_dynamic_shifting:
                if latents is None:
                    raise ValueError('latents is None')

                # for flux we double up the patch size before sending her to simulate the latent reduction
                h = latents.shape[2]
                w = latents.shape[3]
                image_seq_len = h * w // (patch_size**2)

                mu = calculate_shift(
                    image_seq_len,
                    self.config.get("base_image_seq_len", 256),
                    self.config.get("max_image_seq_len", 4096),
                    self.config.get("base_shift", 0.5),
                    self.config.get("max_shift", 1.16),
                )
                sigmas = self.time_shift(mu, 1.0, sigmas)
            else:
                sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

            if self.config.shift_terminal:
                sigmas = self.stretch_shift_to_terminal(sigmas)

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)

            sigmas = torch.from_numpy(sigmas).to(
                dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps

            if self.config.invert_sigmas:
                sigmas = 1.0 - sigmas
                timesteps = sigmas * self.config.num_train_timesteps
                sigmas = torch.cat(
                    [sigmas, torch.ones(1, device=sigmas.device)])
            else:
                sigmas = torch.cat(
                    [sigmas, torch.zeros(1, device=sigmas.device)])

            self.timesteps = timesteps.to(device=device)
            self.sigmas = sigmas

            self.timesteps = timesteps.to(device=device)
            return timesteps

        elif timestep_type == 'lognorm_blend':
            # disgtribute timestepd to the center/early and blend in linear
            alpha = 0.75

            lognormal = LogNormal(loc=0, scale=0.333)

            # Sample from the distribution
            t1 = lognormal.sample((int(num_timesteps * alpha),)).to(device)

            # Scale and reverse the values to go from 1000 to 0
            t1 = ((1 - t1/t1.max()) * 1000)

            # add half of linear
            t2 = torch.linspace(1000, 1, int(
                num_timesteps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device=device)
            return timesteps
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")


@dataclass
class FlowMatchStepWithLogProbSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    log_prob: Optional[torch.FloatTensor] = None
    transition_mean: Optional[torch.FloatTensor] = None
    transition_std: Optional[torch.FloatTensor] = None


class FlowMatchStepWithLogProbScheduler(CustomFlowMatchEulerDiscreteScheduler):
    @staticmethod
    def _expand_like(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = value
        while out.ndim < ref.ndim:
            out = out.unsqueeze(-1)
        return out

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        noise_level: float = 0.7,
        sde_type: str = "sde",
        prev_sample: Optional[torch.FloatTensor] = None,
        return_log_prob: bool = False,
        return_transition: bool = False,
    ) -> Union[FlowMatchStepWithLogProbSchedulerOutput, Tuple]:
        if per_token_timesteps is not None:
            if return_log_prob or return_transition or prev_sample is not None:
                raise ValueError(
                    "FlowMatchStepWithLogProbScheduler does not support GRPO transition outputs with per_token_timesteps."
                )
            return super().step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                generator=generator,
                per_token_timesteps=per_token_timesteps,
                return_dict=return_dict,
            )

        # Default behavior remains the parent FlowMatch step.
        if not return_log_prob and not return_transition and prev_sample is None:
            return super().step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                generator=generator,
                per_token_timesteps=per_token_timesteps,
                return_dict=return_dict,
            )

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer timestep indices to FlowMatchStepWithLogProbScheduler.step() is not supported."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample_f = sample.to(torch.float32)
        model_output_f = model_output.to(torch.float32)
        if prev_sample is not None:
            prev_sample = prev_sample.to(torch.float32)

        sigma = self.sigmas[self.step_index].to(device=sample_f.device, dtype=sample_f.dtype).flatten()
        sigma_next = self.sigmas[self.step_index + 1].to(device=sample_f.device, dtype=sample_f.dtype).flatten()
        sigma_e = self._expand_like(sigma, sample_f)
        sigma_next_e = self._expand_like(sigma_next, sample_f)
        dt = sigma_next_e - sigma_e

        eps = 1e-8
        sqrt_2pi = torch.sqrt(torch.as_tensor(2.0 * np.pi, device=sample_f.device, dtype=torch.float32))

        if sde_type == "sde":
            sigma_safe = torch.clamp(sigma_e, min=eps)
            one_minus_sigma = torch.clamp(1.0 - sigma_e, min=eps)
            std_dev_t = torch.sqrt(sigma_safe / one_minus_sigma) * float(noise_level)
            dt_safe = torch.clamp(-dt, min=eps)
            step_scale = torch.sqrt(dt_safe)
            prev_sample_mean = (
                sample_f * (1.0 + ((std_dev_t**2) / (2.0 * sigma_safe)) * dt)
                + model_output_f * (1.0 + ((std_dev_t**2) * (1.0 - sigma_e) / (2.0 * sigma_safe))) * dt
            )
            if prev_sample is None:
                noise = torch.randn(
                    model_output_f.shape,
                    generator=generator,
                    device=model_output_f.device,
                    dtype=model_output_f.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * step_scale * noise

            transition_std = torch.clamp(std_dev_t * step_scale, min=eps)
        elif sde_type == "cps":
            std_dev_t = torch.clamp(
                sigma_next_e * np.sin(float(noise_level) * np.pi / 2.0),
                min=eps,
            )
            pred_original_sample = sample_f - sigma_e * model_output_f
            noise_estimate = sample_f + model_output_f * (1.0 - sigma_e)
            sqrt_term = torch.sqrt(torch.clamp((sigma_next_e**2) - (std_dev_t**2), min=eps))
            prev_sample_mean = pred_original_sample * (1.0 - sigma_next_e) + noise_estimate * sqrt_term
            if prev_sample is None:
                noise = torch.randn(
                    model_output_f.shape,
                    generator=generator,
                    device=model_output_f.device,
                    dtype=model_output_f.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * noise
            transition_std = std_dev_t
        else:
            raise ValueError(f"Unsupported Flow-GRPO sde_type '{sde_type}'.")

        log_prob = None
        if return_log_prob:
            squared_error = (prev_sample.detach() - prev_sample_mean) ** 2
            reduce_dims = tuple(range(1, squared_error.ndim))
            if sde_type == "cps":
                log_prob_tensor = -squared_error
            else:
                log_prob_tensor = (
                    -(squared_error) / (2.0 * (transition_std**2))
                    - torch.log(transition_std)
                    - torch.log(sqrt_2pi)
                )
            log_prob = log_prob_tensor.mean(dim=reduce_dims)

        self._step_index += 1
        prev_sample = prev_sample.to(model_output.dtype)

        transition_mean = prev_sample_mean if return_transition else None
        transition_std_out = transition_std if return_transition else None
        if not return_dict:
            output: list[torch.Tensor] = [prev_sample]
            if return_log_prob:
                output.append(log_prob)
            if return_transition:
                output.append(transition_mean)
                output.append(transition_std_out)
            return tuple(output)

        return FlowMatchStepWithLogProbSchedulerOutput(
            prev_sample=prev_sample,
            log_prob=log_prob,
            transition_mean=transition_mean,
            transition_std=transition_std_out,
        )
