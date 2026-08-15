"""Reference-video latents for ref2va, without dataloader machinery.

A control VIDEO gets the dataset's treatment — num_frames / auto_frame_count,
fps, resolution bucket with center crop — then a single VAE encode whose
result is cached next to the video in ``_latent_cache/`` (keyed like normal
latent caches: file signature + the config values that shape the latent).
Everything is deterministic (even frame spread, no random start) so the cache
is stable; the audio track is encoded alongside when possible (cached for
later use, unused in conditioning for now).
"""

import base64
import hashlib
import json
import os

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from toolkit.basic import get_quick_signature_string
from toolkit.buckets import get_bucket_for_image_size


def _cache_path(path: str, hash_dict: dict) -> str:
    latent_dir = os.path.join(os.path.dirname(path), "_latent_cache")
    name = os.path.splitext(os.path.basename(path))[0]
    hash_input = json.dumps(hash_dict, sort_keys=True).encode("utf-8")
    hash_str = (
        base64.urlsafe_b64encode(hashlib.md5(hash_input).digest())
        .decode("ascii")
        .replace("=", "")
    )
    return os.path.join(latent_dir, f"{name}_{hash_str}.safetensors")


@torch.no_grad()
def load_ref_video_latent(model, path: str, dataset_config) -> dict:
    """Returns {"latent": (C, T, h, w) cpu tensor, "num_frames": int},
    encoding + disk-caching on first use. ``model`` is the MinimaxH3 model
    (used for the VAE, audio encode and the frame-count snapper)."""
    mem_cache = getattr(model, "_ref_video_cache", None)
    if mem_cache is None:
        mem_cache = {}
        model._ref_video_cache = mem_cache
    if path in mem_cache:
        return mem_cache[path]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open reference video {path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or dataset_config.fps

    # dataset-identical frame count
    if dataset_config.auto_frame_count:
        num_frames = int(total / src_fps * dataset_config.fps)
        snapper = model.get_frame_count_snapper()
        if snapper is not None:
            num_frames = snapper(num_frames)
    else:
        num_frames = dataset_config.num_frames

    trim_tail = bool(
        dataset_config.auto_frame_count and dataset_config.trim_auto_frame_count_tail
    )
    hash_dict = {
        "signature": get_quick_signature_string(path),
        "resolution": dataset_config.resolution,
        "num_frames": num_frames,
        "fps": dataset_config.fps,
        "trim_tail": trim_tail,
        "latent_space_version": model.latent_space_version,
        "is_ref_video": True,
    }
    cache_file = _cache_path(path, hash_dict)
    if os.path.exists(cache_file):
        cap.release()
        sd = load_file(cache_file, device="cpu")
        entry = {
            "latent": sd["latent"],
            "num_frames": int(sd["num_frames"].item()),
            "audio_rows": sd.get("audio_latent"),
        }
        mem_cache[path] = entry
        return entry

    # dataset-identical bucket sizing (center crop, no random)
    bucket = get_bucket_for_image_size(
        src_w,
        src_h,
        resolution=dataset_config.resolution,
        divisibility=dataset_config.bucket_tolerance,
    )
    scale = max(bucket["width"] / src_w, bucket["height"] / src_h)
    scale_w, scale_h = int(np.ceil(src_w * scale)), int(np.ceil(src_h * scale))
    crop_x = (scale_w - bucket["width"]) // 2
    crop_y = (scale_h - bucket["height"]) // 2

    if trim_tail:
        # dataset trim mode: real-time pacing from the start, tail trimmed —
        # keeps motion speed honest and the soundtrack in sync
        fps_ratio = src_fps / dataset_config.fps if src_fps > 0 else 1.0
        indices = [min(round(i * fps_ratio), total - 1) for i in range(num_frames)]
    else:
        # deterministic even frame spread across the clip
        indices = [
            min(round(i * (total - 1) / max(num_frames - 1, 1)), total - 1)
            for i in range(num_frames)
        ]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {idx} of {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
        frame = frame[
            crop_y : crop_y + bucket["height"], crop_x : crop_x + bucket["width"]
        ]
        frames.append(frame)
    cap.release()

    pixels = torch.from_numpy(np.stack(frames)).float() / 255.0 * 2.0 - 1.0
    pixels = pixels.permute(0, 3, 1, 2)  # (T, C, H, W), [-1, 1]
    latent = model.encode_images([pixels])[0].to("cpu", torch.float16)

    state_dict = {
        "latent": latent,
        "num_frames": torch.tensor(num_frames, dtype=torch.int64),
    }
    # the soundtrack rides as clean condition rows; best effort (no track = None)
    audio_rows = None
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(path)
        if trim_tail:
            # frames cover [0, num_frames / fps) seconds; trim the soundtrack
            # to the same window so it stays in sync with the sampled frames
            keep = int(round(num_frames / dataset_config.fps * sample_rate))
            waveform = waveform[:, :keep]
        audio_latent = model.encode_audio(
            [{"waveform": waveform, "sample_rate": sample_rate}]
        )[0]
        audio_rows = audio_latent.to("cpu", torch.float16)
        state_dict["audio_latent"] = audio_rows
    except Exception:
        pass

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    save_file(state_dict, cache_file)
    entry = {"latent": latent, "num_frames": num_frames, "audio_rows": audio_rows}
    mem_cache[path] = entry
    return entry
