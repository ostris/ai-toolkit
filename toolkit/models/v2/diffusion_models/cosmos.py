from diffusers.models import (
    CosmosTransformer3DModel as DiffusersCosmosTransformer3DModel,
)

from .._mixin import OstrisModelMixin


class CosmosTransformer3DModel(DiffusersCosmosTransformer3DModel, OstrisModelMixin):
    """Cosmos video/image DiT (anima)."""

    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]
