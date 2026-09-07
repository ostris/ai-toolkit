from diffusers import AutoencoderKLWan

from .._mixin import OstrisModelMixin


class WanVAE(AutoencoderKLWan, OstrisModelMixin):
    """The wan 2.1/2.2 causal video VAE."""

    aitk_subfolder = "vae"
