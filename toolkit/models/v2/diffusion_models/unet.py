from diffusers import UNet2DConditionModel as DiffusersUNet2DConditionModel

from .._mixin import OstrisModelMixin


class UNet2DConditionModel(DiffusersUNet2DConditionModel, OstrisModelMixin):
    """The SD1/SD2/SDXL-family UNet."""

    aitk_subfolder = "unet"

    @classmethod
    def get_transformer_block_names(cls):
        return ["down_blocks", "up_blocks"]
