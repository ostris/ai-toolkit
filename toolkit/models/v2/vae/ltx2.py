from diffusers.models.autoencoders import (
    AutoencoderKLLTX2Audio as DiffusersAutoencoderKLLTX2Audio,
)
from diffusers.models.autoencoders import (
    AutoencoderKLLTX2Video as DiffusersAutoencoderKLLTX2Video,
)

from .._mixin import OstrisModelMixin


class LTX2VideoVAE(DiffusersAutoencoderKLLTX2Video, OstrisModelMixin):
    aitk_subfolder = "vae"


class LTX2AudioVAE(DiffusersAutoencoderKLLTX2Audio, OstrisModelMixin):
    aitk_subfolder = "audio_vae"
