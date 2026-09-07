from diffusers import AuraFlowTransformer2DModel as DiffusersAuraFlowTransformer2DModel

from .._mixin import OstrisModelMixin


class AuraFlowTransformer2DModel(
    DiffusersAuraFlowTransformer2DModel, OstrisModelMixin
):
    aitk_subfolder = "transformer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["joint_transformer_blocks", "single_transformer_blocks"]
