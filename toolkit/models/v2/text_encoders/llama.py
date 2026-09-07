from transformers import LlamaForCausalLM

from .._mixin import OstrisTransformersMixin


class LlamaTextEncoder(LlamaForCausalLM, OstrisTransformersMixin):
    """Llama causal-LM text encoder (hidream's text_encoder_4)."""

    aitk_subfolder = "text_encoder_4"
    aitk_tokenizer_subfolder = "tokenizer_4"

    @classmethod
    def get_transformer_block_names(cls):
        return ["model.layers"]
