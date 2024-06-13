from typing import Union

from diffusers import FlowMatchEulerDiscreteScheduler
import torch

class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):

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
        n_dim = original_samples.ndim
        sigmas = self.get_sigmas(timesteps, n_dim, original_samples.dtype, original_samples.device)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * original_samples
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample