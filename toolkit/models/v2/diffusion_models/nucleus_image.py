from diffusers import (
    NucleusMoEImageTransformer2DModel as DiffusersNucleusMoEImageTransformer2DModel,
)

from .._mixin import OstrisModelMixin


class NucleusMoEImageTransformer2DModel(
    DiffusersNucleusMoEImageTransformer2DModel, OstrisModelMixin
):
    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]
