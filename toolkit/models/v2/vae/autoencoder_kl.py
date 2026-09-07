from diffusers import AutoencoderKL

from .._mixin import OstrisModelMixin


class KLVAE(AutoencoderKL, OstrisModelMixin):
    """The diffusers AutoencoderKL (SD/SDXL/Flux1/Z-Image image VAEs), loaded
    from a checkpoint's vae/ subfolder through the universal loader."""

    aitk_subfolder = "vae"
