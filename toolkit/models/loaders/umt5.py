from typing import List
import torch
from transformers import T5Tokenizer, UMT5EncoderModel
from toolkit.models.loaders.comfy import get_comfy_path

class PatchedT5Tokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
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
            _spm_precompiled_charsmap=None, # this is passing a empty byte string for some reason now.
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

def get_umt5_encoder(
    model_path: str,
    tokenizer_subfolder: str = None,
    encoder_subfolder: str = None,
    torch_dtype: str = torch.bfloat16,
    comfy_files: List[str] = [
        "text_encoders/umt5_xxl_fp16.safetensors",
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ],
) -> UMT5EncoderModel:
    """
    Load the UMT5 encoder model from the specified path.
    """
    tokenizer = PatchedT5Tokenizer.from_pretrained(model_path, subfolder=tokenizer_subfolder)
    comfy_path = get_comfy_path(comfy_files)
    comfy_path = None
    if comfy_path is not None:
        text_encoder = UMT5EncoderModel.from_single_file(
            comfy_path, torch_dtype=torch_dtype
        )
    else:
        print(f"Using {model_path} for UMT5 encoder.")
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder=encoder_subfolder, torch_dtype=torch_dtype
        )
    return tokenizer, text_encoder
