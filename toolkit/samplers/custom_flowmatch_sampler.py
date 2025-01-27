import math
from typing import Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np


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
            hbsmntw_weighing[num_timesteps // 2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            # Create linear timesteps from 1000 to 0
            timesteps = torch.linspace(1000, 0, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False) -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]

        # Get the weights for the timesteps
        if v2:
            weights = self.linear_timesteps_weights2[step_indices].flatten()
        else:
            weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

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
        ## ref https://github.com/huggingface/diffusers/blob/fbe29c62984c33c6cf9cf7ad120a992fe6d20854/examples/dreambooth/train_dreambooth_sd3.py#L1578
        ## Add noise according to flow matching.
        ## zt = (1 - texp) * x + texp * z1

        # sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        # noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # timestep needs to be in [0, 1], we store them in [0, 1000]
        # noisy_sample = (1 - timestep) * latent + timestep * noise
        t_01 = (timesteps / 1000).to(original_samples.device)
        noisy_model_input = (1 - t_01) * original_samples + t_01 * noise

        # n_dim = original_samples.ndim
        # sigmas = self.get_sigmas(timesteps, n_dim, original_samples.dtype, original_samples.device)
        # noisy_model_input = (1.0 - sigmas) * original_samples + sigmas * noise
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(self, num_timesteps, device, timestep_type='linear', latents=None):
        self.timestep_type = timestep_type
        if timestep_type == 'linear':
            timesteps = torch.linspace(1000, 0, num_timesteps, device=device)
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
        elif timestep_type == 'flux_shift':
            # matches inference dynamic shifting
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_timesteps
            )

            sigmas = timesteps / self.config.num_train_timesteps
            
            if latents is None:
                raise ValueError('latents is None')
            
            h = latents.shape[2] // 2  # Divide by ph
            w = latents.shape[3] // 2  # Divide by pw
            image_seq_len = h * w

            # todo need to know the mu for the shift
            mu = calculate_shift(
                image_seq_len,
                self.config.get("base_image_seq_len", 256),
                self.config.get("max_image_seq_len", 4096),
                self.config.get("base_shift", 0.5),
                self.config.get("max_shift", 1.16),
            )
            sigmas = self.time_shift(mu, 1.0, sigmas)

            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps

            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

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
            t2 = torch.linspace(1000, 0, int(num_timesteps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device=device)
            return timesteps
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")
