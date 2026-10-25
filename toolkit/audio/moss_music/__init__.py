import torch

from .configuration import MossMusicConfig, MossMusicEncoderConfig
from .modeling import MossMusicModel
from .processing import MelConfig, MossMusicProcessor

HF_REPO = "OpenMOSS-Team/MOSS-Music-8B-Instruct"


def load_moss_music(
    name_or_path: str = HF_REPO,
    dtype: torch.dtype = torch.bfloat16,
    device="cpu",
    enable_time_marker: bool = True,
):
    """Model + processor from a hub id or local checkpoint dir. Time markers
    (elapsed seconds interleaved into the audio tokens) are what the model was
    trained with and keep long transcriptions from looping; leave them on."""
    model = MossMusicModel.from_pretrained(name_or_path, dtype=dtype, device_map=device)
    model.eval()
    processor = MossMusicProcessor.from_pretrained(
        name_or_path, enable_time_marker=enable_time_marker
    )
    return model, processor


__all__ = [
    "HF_REPO",
    "MelConfig",
    "MossMusicConfig",
    "MossMusicEncoderConfig",
    "MossMusicModel",
    "MossMusicProcessor",
    "load_moss_music",
]
