from transformers import Gemma2Model

from .._mixin import OstrisTransformersMixin


class Gemma2ModelEncoder(Gemma2Model, OstrisTransformersMixin):
    """Gemma2 base model (lumina2's text encoder — what AutoModel resolves)."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]
