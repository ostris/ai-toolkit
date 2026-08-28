from diffusers import SD3Transformer2DModel as DiffusersSD3Transformer2DModel

from .._mixin import OstrisModelMixin


class SD3Transformer2DModel(DiffusersSD3Transformer2DModel, OstrisModelMixin):
    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]
