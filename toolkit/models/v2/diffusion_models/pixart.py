from diffusers import PixArtTransformer2DModel as DiffusersPixArtTransformer2DModel
from diffusers import Transformer2DModel as DiffusersTransformer2DModel

from .._mixin import OstrisModelMixin


class PixArtTransformer2DModel(
    DiffusersPixArtTransformer2DModel, OstrisModelMixin
):
    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]


class Transformer2DModel(DiffusersTransformer2DModel, OstrisModelMixin):
    """The generic diffusers DiT the pixart sigma path loads."""

    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]
