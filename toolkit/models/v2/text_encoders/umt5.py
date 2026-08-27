import torch
from transformers import T5Tokenizer, UMT5EncoderModel

from .._mixin import OstrisTransformersMixin


class PatchedT5Tokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        _spm_precompiled_charsmap=None,
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs,
    ):
        super().__init__(
            vocab=vocab,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            _spm_precompiled_charsmap=None,  # this is passing a empty byte string for some reason now.
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )


class UMT5TextEncoder(UMT5EncoderModel, OstrisTransformersMixin):
    """UMT5-XXL text encoder (wan family)."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["encoder.block"]

    @classmethod
    def load_tokenizer(cls, name_or_path=None, subfolder=None, **kwargs):
        # T5's tokenizer needs the _spm_precompiled_charsmap patch
        source = name_or_path if name_or_path is not None else cls.aitk_tokenizer_repo
        if subfolder is None:
            subfolder = cls.aitk_tokenizer_subfolder
        return PatchedT5Tokenizer.from_pretrained(
            source, subfolder=subfolder or "", **kwargs
        )
