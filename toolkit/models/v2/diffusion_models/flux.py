from diffusers import FluxTransformer2DModel as DiffusersFluxTransformer2DModel

from .._mixin import OstrisModelMixin


class FluxTransformer2DModel(DiffusersFluxTransformer2DModel, OstrisModelMixin):
    """Flux1-family DiT (flux, flux_kontext, chroma-adjacent finetunes in
    diffusers layout)."""

    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks", "single_transformer_blocks"]
