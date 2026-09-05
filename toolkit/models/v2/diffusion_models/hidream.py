from diffusers.models import (
    HiDreamImageTransformer2DModel as DiffusersHiDreamImageTransformer2DModel,
)

from .._mixin import OstrisModelMixin


class HiDreamImageTransformer2DModel(
    DiffusersHiDreamImageTransformer2DModel, OstrisModelMixin
):
    """The diffusers HiDream DiT (hidream_e1; the base hidream arch uses the
    vendored copy in the hidream extension)."""

    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["double_stream_blocks", "single_stream_blocks"]
