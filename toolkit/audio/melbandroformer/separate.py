import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .model import MelBandRoformer

HF_REPO = "ai-toolkit/melbandroformer"
DEFAULT_WEIGHTS = "melbandroformer_vocals_kj.safetensors"
MODEL_SAMPLE_RATE = 44100


def get_weights_path(filename: str = DEFAULT_WEIGHTS) -> str:
    """MODELS_PATH/checkpoints/<filename>, pulled from the HF repo root if missing."""
    from toolkit.paths import MODELS_PATH  # lazy: read after the CLI has loaded .env
    ckpt_dir = os.path.join(MODELS_PATH, "checkpoints")
    path = os.path.join(ckpt_dir, filename)
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Downloading {HF_REPO}/{filename} to {ckpt_dir}")
    return hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=ckpt_dir)


def load_melbandroformer(filename: str = DEFAULT_WEIGHTS, device=None, compile: bool = False) -> MelBandRoformer:
    path = get_weights_path(filename)
    device = torch.device(device if device is not None else "cpu")
    with safe_open(path, framework="pt", device=str(device)) as f:
        metadata = f.metadata() or {}
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    if "config" not in metadata:
        raise ValueError(
            f"{path} has no model config in its metadata; convert it with scripts/convert_melbandroformer.py"
        )
    # meta init + assign: no cpu random init of 228M params, tensors go file -> device directly
    with torch.device("meta"):
        model = MelBandRoformer(**json.loads(metadata["config"]))
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.to(device)  # index buffers are built on cpu
    # chunk_size / num_overlap the weights were validated with
    model.inference_config = json.loads(metadata.get("inference", "{}"))
    model.eval()
    if compile:
        model.compile_core()
    return model


def _fade_window(chunk_size: int, fade_size: int) -> torch.Tensor:
    window = torch.ones(chunk_size)
    window[:fade_size] = torch.linspace(0, 1, fade_size)
    window[-fade_size:] = torch.linspace(1, 0, fade_size)
    return window


@torch.no_grad()
def separate_stems(
    model: MelBandRoformer,
    mix: torch.Tensor,
    chunk_size: int = None,
    num_overlap: int = None,
    batch_size: int = 4,
    dtype=torch.float16,
    progress: bool = False,
) -> torch.Tensor:
    """Overlap-add chunked inference (MSST `demix`). mix: [C, T] at MODEL_SAMPLE_RATE. Returns [num_stems, C, T] on cpu."""
    cfg = getattr(model, "inference_config", {})
    chunk_size = chunk_size or cfg.get("chunk_size", 352800)
    num_overlap = num_overlap or cfg.get("num_overlap", 2)

    device = next(model.parameters()).device
    mix = mix.to(device=device, dtype=torch.float32)
    length = mix.shape[-1]

    step = chunk_size // num_overlap
    fade_size = chunk_size // 10
    border = chunk_size - step

    # reflect-pad the edges so the first/last chunks are not fading in/out on real audio
    padded = length > 2 * border and border > 0
    if padded:
        mix = F.pad(mix, (border, border), mode="reflect")
    total = mix.shape[-1]

    window = _fade_window(chunk_size, fade_size).to(device)
    result = torch.zeros((model.num_stems, *mix.shape), device=device)
    counter = torch.zeros_like(result)

    use_autocast = dtype is not None and device.type == "cuda"
    positions = list(range(0, total, step))
    if progress:
        from tqdm import tqdm
        batches = tqdm(range(0, len(positions), batch_size), desc="separating", leave=False)
    else:
        batches = range(0, len(positions), batch_size)

    for b in batches:
        batch_pos = positions[b:b + batch_size]
        parts, lens = [], []
        for start in batch_pos:
            part = mix[:, start:start + chunk_size]
            chunk_len = part.shape[-1]
            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            parts.append(F.pad(part, (0, chunk_size - chunk_len), mode=pad_mode))
            lens.append(chunk_len)
        if len(parts) == 1:
            parts.append(parts[0])  # a batch of 1 would make dynamo specialize (0/1) and recompile

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
            out = model(torch.stack(parts))
        if out.ndim == 3:
            out = out[:, None]  # [b, n, c, t]
        out = out.float()

        for j, (start, seg_len) in enumerate(zip(batch_pos, lens)):
            w = window.clone()
            if start == 0:
                w[:fade_size] = 1
            if start + step >= total:
                w[-fade_size:] = 1
            result[..., start:start + seg_len] += out[j, ..., :seg_len] * w[:seg_len]
            counter[..., start:start + seg_len] += w[:seg_len]

    stems = torch.nan_to_num(result / counter)
    if padded:
        stems = stems[..., border:-border]
    return stems.cpu()


@torch.no_grad()
def separate(model: MelBandRoformer, wav: torch.Tensor, sample_rate: int, **kwargs):
    """wav: [C, T] or [T] at any sample rate. Returns (vocals, instrumental), same shape/sr/device as wav,
    with vocals + instrumental == wav exactly."""
    if wav.ndim == 1:
        wav = wav[None]
    wav = wav.float()
    channels, length = wav.shape
    if channels > 2:
        raise ValueError(f"expected mono or stereo audio, got {channels} channels")

    mix = wav.repeat(2, 1) if channels == 1 else wav
    if sample_rate != MODEL_SAMPLE_RATE:
        import torchaudio
        mix = torchaudio.functional.resample(mix, sample_rate, MODEL_SAMPLE_RATE)

    vocals = separate_stems(model, mix, **kwargs)[0]

    if sample_rate != MODEL_SAMPLE_RATE:
        vocals = torchaudio.functional.resample(vocals, MODEL_SAMPLE_RATE, sample_rate)
    vocals = F.pad(vocals[..., :length], (0, max(0, length - vocals.shape[-1])))
    if channels == 1:
        vocals = vocals.mean(0, keepdim=True)
    vocals = vocals.to(wav.device)

    instrumental = wav - vocals
    return vocals, instrumental
