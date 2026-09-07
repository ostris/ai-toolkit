from transformers import Qwen3ForCausalLM, Qwen3Model

from .._mixin import OstrisTransformersMixin


class Qwen3TextEncoder(Qwen3ForCausalLM, OstrisTransformersMixin):
    """Qwen3 causal-LM text encoder (Z-Image family, Zeta-Chroma, ...). Loads
    from a checkpoint's text_encoder/ subfolder, a hub repo, or a single
    .safetensors file; the tokenizer rides in the checkpoint's tokenizer/
    subfolder."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["model.layers"]


class Qwen3ModelEncoder(Qwen3Model, OstrisTransformersMixin):
    """The inner Qwen3 base model (anima's text encoder)."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]
