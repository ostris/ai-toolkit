from transformers import Gemma3ForConditionalGeneration

from .._mixin import OstrisTransformersMixin


class Gemma3TextEncoder(Gemma3ForConditionalGeneration, OstrisTransformersMixin):
    """Gemma3 conditioning stack (ltx2 / ltx2.3)."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        # both layouts seen across transformers versions; missing paths skip
        return ["model.language_model.layers", "language_model.model.layers"]


try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    class Gemma4TextEncoder(Gemma4TextModel, OstrisTransformersMixin):
        """Gemma4 text decoder (ltx2.5's conditioning stack)."""

        aitk_subfolder = "text_encoder"
        aitk_tokenizer_subfolder = "tokenizer"

        @classmethod
        def get_transformer_block_names(cls):
            return ["layers"]

except ImportError:
    Gemma4TextEncoder = None
