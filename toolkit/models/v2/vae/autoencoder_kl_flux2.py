from diffusers import AutoencoderKLFlux2

from .._mixin import OstrisModelMixin


class Flux2KLVAE(AutoencoderKLFlux2, OstrisModelMixin):
    """The diffusers Flux2 KL VAE (ernie_image; distinct from the hand-rolled
    BFL AutoEncoder in vae/flux2_kl.py that flux2/ideogram4 use)."""

    aitk_subfolder = "vae"
