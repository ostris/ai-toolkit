from transformers import CLIPVisionModel

from .._mixin import OstrisTransformersMixin


class CLIPVisionEncoder(CLIPVisionModel, OstrisTransformersMixin):
    """CLIP vision tower (wan21 i2v image conditioning)."""

    aitk_subfolder = "image_encoder"

    @classmethod
    def get_transformer_block_names(cls):
        return ["vision_model.encoder.layers"]
