from typing import List
import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from toolkit.models.loaders.comfy import get_comfy_path


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
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder=tokenizer_subfolder)
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
