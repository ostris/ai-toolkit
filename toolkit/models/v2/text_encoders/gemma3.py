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

    def get_offload_ignore_modules(self):
        return [self.model.language_model.base_model.embed_tokens]


try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    class Gemma4TextEncoder(Gemma4TextModel, OstrisTransformersMixin):
        """Gemma4 text decoder (ltx2.5's conditioning stack)."""

        aitk_subfolder = "text_encoder"
        aitk_tokenizer_subfolder = "tokenizer"

        @classmethod
        def get_transformer_block_names(cls):
            return ["layers"]

        def get_offload_ignore_modules(self):
            # layer_scalar is a bare tensor buffer on each decoder layer; the
            # manager never enumerates it, so it must ride along explicitly
            return [self.embed_tokens] + [
                layer.layer_scalar for layer in self.layers
            ]

except ImportError:
    Gemma4TextEncoder = None
