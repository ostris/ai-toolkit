import math
from typing import Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
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
        print(10)
        with torch.no_grad():
            num_timesteps = 1000
            x = torch.arange(num_timesteps, dtype=torch.float32)
  
            # Prevent division by zero if num_timesteps is 0
            divisor = num_timesteps if num_timesteps > 0 else 1
            print(11, divisor)
            # Prevent division by zero in denominator
            safe_divisor = divisor if divisor != 0 else 1
            print(12, safe_divisor)
            y = 1#torch.exp(-2 * ((x - safe_divisor / 2) / safe_divisor) ** 2)
            print(14)
            y_shifted = y - y.min()
            # Scale to make mean 1, avoid division by zero
            sum_y_shifted = y_shifted.sum()
            scale_factor = safe_divisor / sum_y_shifted if sum_y_shifted != 0 else 1.0
            bsmntw_weighing = y_shifted * scale_factor
            print(15)
            hbsmntw_weighing = y_shifted * scale_factor
            hbsmntw_weighing[safe_divisor // 2:] = hbsmntw_weighing[safe_divisor // 2:].max()
            timesteps = torch.linspace(1000, 1, safe_divisor, device='cpu')
            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass
        print(20)

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
        if v2:
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
