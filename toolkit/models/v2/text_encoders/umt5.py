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
    aitk_config_repo = "ai-toolkit/umt5_xxl_encoder"
    # comfy fp8 umt5 files store non-quantized tensors in a dtype mix; T5
    # assumes a uniform dtype, so follow the load dtype
    aitk_cast_quantized_load = True

    aitk_comfy_repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    # comfy umt5 files already use the transformers key layout; the
    # fp8_e4m3fn_scaled variant is the legacy scaled-fp8 format (handled by
    # the importer's float8 backend)
    aitk_comfy_weight_names = {
        "ai-toolkit/umt5_xxl_encoder": [
            "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "split_files/text_encoders/umt5_xxl_fp16.safetensors",
        ],
    }

    @classmethod
    def get_transformer_block_names(cls):
        return ["encoder.block"]

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        # drop the embedded sentencepiece blob and materialize the tied
        # embed_tokens reference
        state_dict = dict(state_dict)
        state_dict.pop("spiece_model", None)
        if (
            "encoder.embed_tokens.weight" not in state_dict
            and "shared.weight" in state_dict
        ):
            state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
        return state_dict

    @classmethod
    def load_tokenizer(cls, name_or_path=None, subfolder=None, **kwargs):
        # T5's tokenizer needs the _spm_precompiled_charsmap patch
        source = name_or_path if name_or_path is not None else cls.aitk_tokenizer_repo
        if subfolder is None:
            subfolder = cls.aitk_tokenizer_subfolder
        return PatchedT5Tokenizer.from_pretrained(
            source, subfolder=subfolder or "", **kwargs
        )
