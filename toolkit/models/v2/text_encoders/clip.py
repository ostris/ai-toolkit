from transformers import CLIPTextModel, CLIPTextModelWithProjection

from .._mixin import OstrisTransformersMixin


class CLIPTextEncoder(CLIPTextModel, OstrisTransformersMixin):
    """CLIP-L text encoder (flux-style checkpoints: text_encoder/ +
    tokenizer/)."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["text_model.encoder.layers"]


class CLIPTextEncoderWithProjection(CLIPTextModelWithProjection, OstrisTransformersMixin):
    """CLIP text encoder with the projection head (SDXL / SD3 / HiDream style
    checkpoints; the second encoder lives at text_encoder_2/ + tokenizer_2/)."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["text_model.encoder.layers"]
