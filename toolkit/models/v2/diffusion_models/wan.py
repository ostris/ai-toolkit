from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel

from .._mixin import OstrisModelMixin


class WanTransformer3DModel(DiffusersWanTransformer3DModel, OstrisModelMixin):
    """Wan 2.1/2.2 video DiT (wan22 loads two of these into its dual wrapper)."""

    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["blocks"]
