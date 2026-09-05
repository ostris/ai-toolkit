"""Reference-video latents for ref2va, without dataloader machinery.

A control VIDEO gets the dataset's temporal treatment — num_frames /
auto_frame_count, fps — and is area-matched to the target (own aspect kept, /32
grid), then a single VAE encode whose
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
from .packing import reference_video_pixel_size


def ref_frame_indices(total, src_fps, num_frames, dataset_fps, trim_tail):
    """Source frame indices a reference video is sampled at (dataset-identical)."""
    if trim_tail:
        # dataset trim mode: real-time pacing from the start, tail trimmed —
        # keeps motion speed honest and the soundtrack in sync
        fps_ratio = src_fps / dataset_fps if src_fps > 0 else 1.0
        return [min(round(i * fps_ratio), total - 1) for i in range(num_frames)]
    # deterministic even frame spread across the clip
    return [
        min(round(i * (total - 1) / max(num_frames - 1, 1)), total - 1)
        for i in range(num_frames)
    ]


def ref_video_num_frames(model, path, dataset_config):
    """Dataset-identical frame count for a reference video."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or dataset_config.fps
    cap.release()
    if dataset_config.auto_frame_count:
        num_frames = int(total / src_fps * dataset_config.fps)
        snapper = model.get_frame_count_snapper()
        if snapper is not None:
            num_frames = snapper(num_frames)
    else:
        num_frames = dataset_config.num_frames
    return num_frames, total, src_fps


def load_video_ref_for_te(model, path, dataset_config=None, max_frames=None):
    """Build the Qwen presentation from the SAME frames the latent rows use:
    2 fps over the frame-count-treated 24 fps clip (ComfyUI: frames[::12],
    timestamps i/2). Without a dataset config (sampling), the clip is
    treated as its own length capped at ``max_frames``, snapped to 17n+5."""
    from PIL import Image as _Image

    from .packing import align_num_frames_down
    from .text_encoder import VideoRef, video_has_audio

    if dataset_config is not None:
        num_frames, total, src_fps = ref_video_num_frames(model, path, dataset_config)
        ds_fps = dataset_config.fps
        trim = bool(
            dataset_config.auto_frame_count
            and dataset_config.trim_auto_frame_count_tail
        )
    else:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        ds_fps = 24
        n = int(total / src_fps * ds_fps)
        if max_frames:
            n = min(n, max_frames)
        num_frames = align_num_frames_down(max(n, 5))
        trim = True
    indices = ref_frame_indices(total, src_fps, num_frames, ds_fps, trim)
    # 2 fps over the treated 24fps clip: every 12th treated frame
    step = max(1, int(ds_fps // 2))
    picks = list(range(0, len(indices), step))
    cap = cv2.VideoCapture(path)
    raw_frames = read_frames_at(cap, [indices[j] for j in picks])
    cap.release()
    frames = [
        _Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in raw_frames
    ]
    times = [j / ds_fps for j in picks]
    return VideoRef(frames=frames, timestamps=times, has_audio=video_has_audio(path))


def static_image_video_ref(image, num_frames: int, fps: int = 24):
    """Present a still IMAGE as a silent static reference video: the same
    frame held for ``num_frames`` at ``fps``, sampled at 2 fps like
    :func:`load_video_ref_for_te` (frame picks every fps//2, timestamps
    j/fps). Used when image references are routed through the video-ref path
    (``image_refs_as_video``)."""
    from .text_encoder import VideoRef

    step = max(1, int(fps // 2))
    picks = list(range(0, int(num_frames), step))
    return VideoRef(
        frames=[image] * len(picks),
        timestamps=[j / fps for j in picks],
        has_audio=False,
    )


def read_frames_at(cap, indices):
    """Read the frames at sorted ``indices`` by decoding SEQUENTIALLY (seek-per-
    frame is both slow and unreliable on VBR/web clips). Container frame counts
    routinely overstate the decodable count by a few frames; when the stream
    ends early, missing tail frames repeat the last decoded frame instead of
    failing. Returns a list of BGR frames aligned with ``indices``."""
    wanted = list(indices)
    out = {}
    need = sorted(set(wanted))
    if not need:
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pos = 0
    max_idx = need[-1]
    ptr = 0
    last = None
    while ptr < len(need) and pos <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        while ptr < len(need) and need[ptr] == pos:
            out[pos] = frame
            ptr += 1
        last = frame
        pos += 1
    if not out and last is None:
        raise ValueError("Could not decode any frames")
    frames = []
    for idx in wanted:
        if idx in out:
            frames.append(out[idx])
        else:
            # ran past the decodable end: hold the last real frame
            frames.append(last)
    return frames


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
def load_ref_video_latent(
    model, path: str, dataset_config, target_height: int, target_width: int
) -> dict:
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
        "ref_sizing": "match_target_area",
        "target_area": int(target_height * target_width)
        if target_height and target_width
        else 0,
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

    # match the target's pixel area (the dataset bucket the target trains at)
    # with the ref's own aspect: same aspect -> identical size; aspect-
    # preserving resize, no crop
    out_h, out_w = reference_video_pixel_size(src_w, src_h, target_height, target_width)

    indices = ref_frame_indices(
        total, src_fps, num_frames, dataset_config.fps, trim_tail
    )
    try:
        raw_frames = read_frames_at(cap, indices)
    except ValueError as e:
        raise ValueError(f"{e}: {path}") from e
    cap.release()
    frames = []
    for frame in raw_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)

    pixels = torch.from_numpy(np.stack(frames)).float() / 255.0 * 2.0 - 1.0
    pixels = pixels.permute(0, 3, 1, 2)  # (T, C, H, W), [-1, 1]
    latent = model.encode_images([pixels])[0].to("cpu", torch.float16)

    state_dict = {
        "latent": latent,
        "num_frames": torch.tensor(num_frames, dtype=torch.int64),
    }
    # the soundtrack rides as clean condition rows iff the file has an audio
    # stream (the TE presentation's "<Audio j>" label uses the same test)
    audio_rows = None
    try:
        from .text_encoder import video_has_audio

        if not video_has_audio(path):
            raise RuntimeError("no audio stream")
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
