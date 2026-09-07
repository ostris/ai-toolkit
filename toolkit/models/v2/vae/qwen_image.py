import torch
from diffusers import AutoencoderKLQwenImage

from .._mixin import OstrisModelMixin


class QwenImageVAE(AutoencoderKLQwenImage, OstrisModelMixin):
    """The Qwen-Image (wan-style video) VAE, shared by the qwen_image family,
    nucleus_image and krea2."""

    aitk_subfolder = "vae"


class QwenImageVAEHolderMixin:
    """BaseModel-side encode_images/decode_latents for models whose self.vae
    is the Qwen-Image VAE: it is a video VAE, so images ride in a single-frame
    dim and latents are normalized with the config's latents_mean/std."""

    # tile the decode when low_vram (decode only; encode stays untiled)
    vae_decode_tiled_on_low_vram = False

    def encode_images(self, image_list, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)
        images = images.unsqueeze(2)  # add the frame dim
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.squeeze(2)  # drop the frame dim
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)
        latents = latents.unsqueeze(2)  # add the frame dim

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        # full-resolution decode spikes VRAM; models opt in to tiling it
        tiled = self.vae_decode_tiled_on_low_vram and self.model_config.low_vram
        if tiled:
            self.vae.enable_tiling()
        try:
            images = self.vae.decode(latents).sample
        finally:
            if tiled:
                self.vae.disable_tiling()

        images = images.squeeze(2)  # drop the frame dim
        return images.to(device, dtype=dtype)
