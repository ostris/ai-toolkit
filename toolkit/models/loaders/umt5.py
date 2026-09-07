from typing import List

import torch

from toolkit.models.v2.text_encoders.umt5 import (
    PatchedT5Tokenizer,
    UMT5TextEncoder,
)


def get_umt5_encoder(
    model_path: str,
    tokenizer_subfolder: str = None,
    encoder_subfolder: str = None,
    torch_dtype: str = torch.bfloat16,
    # reserved for the comfy-weights flip (Phase 2); accepted for
    # signature compatibility, not consulted yet
    comfy_files: List[str] = None,
):
    """Load the UMT5 tokenizer + encoder. Thin compatibility wrapper around
    toolkit/models/v2/text_encoders/umt5.py."""
    tokenizer = UMT5TextEncoder.load_tokenizer(
        model_path, subfolder=tokenizer_subfolder or ""
    )
    print(f"Using {model_path} for UMT5 encoder.")
    text_encoder = UMT5TextEncoder.load_model(
        model_path, dtype=torch_dtype, subfolder=encoder_subfolder or ""
    )
    return tokenizer, text_encoder
