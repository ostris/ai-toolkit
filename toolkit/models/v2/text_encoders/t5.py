from transformers import T5EncoderModel

from .._mixin import OstrisTransformersMixin


class T5TextEncoder(T5EncoderModel, OstrisTransformersMixin):
    """T5-XXL text encoder. Defaults to the flux-style checkpoint layout
    (text_encoder_2/ + tokenizer_2/); pass subfolder overrides for checkpoints
    that keep it at text_encoder/ + tokenizer/ (e.g. f-lite)."""

    aitk_subfolder = "text_encoder_2"
    aitk_tokenizer_subfolder = "tokenizer_2"

    @classmethod
    def get_transformer_block_names(cls):
        return ["encoder.block"]
