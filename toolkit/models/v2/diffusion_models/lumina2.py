from diffusers import Lumina2Transformer2DModel as DiffusersLumina2Transformer2DModel

from .._mixin import OstrisModelMixin


class Lumina2Transformer2DModel(
    DiffusersLumina2Transformer2DModel, OstrisModelMixin
):
    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]
