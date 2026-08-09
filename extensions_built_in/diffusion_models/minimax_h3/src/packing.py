"""Packed-sequence geometry for MiniMax-H3.

One transformer forward runs over a single packed 1-D sequence:

    [ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]

This module owns everything needed to place a row in that sequence and give it
its (t, h, w) rotary coordinate, plus the sigma-shift math that couples the
video (shift 12) and audio (shift 3) flow schedules.

Rotary coordinates are built in float64 and with numpy's ``linspace`` because
video and audio share one 40-units-per-second rotary clock (video advances
5/3 units per pixel frame at 24 fps, audio one unit per latent at 40/s) and
that shared clock is the released checkpoint's audio/video alignment.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Per-row modality tags — these index the transformer's AdaLN table, so the
# values are a checkpoint contract.
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
PAD_TAG = -1

FPS = 24
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32

# video VAE: 17 pixel frames per chunk -> 5 latent frames, 3 trailing latents
# dropped overall, so 17n+5 pixel frames <-> 5n+2 latent frames
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5

AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_RATE = 32000

# released flow shifts (exponential): video 12, audio 3
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0

# keyframe conditioning rows are noised to t = 0.999 and pinned there; the
# posterior sample of the keyframe VAE encode uses a fixed seed of 42
KEYFRAME_NOISE_AUG_T = 0.999
KEYFRAME_ENCODE_SEED = 42

# rotary-time constants: one latent frame spans 5/3 * frames_per_latent units,
# the (1, 4, 4, 4, 4) pattern mirroring the VAE's 17 -> 5 frame grouping
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


# ---------------------------------------------------------------------------
# Frame / canvas arithmetic
# ---------------------------------------------------------------------------


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count UP to the next 17n+5 the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def align_num_frames_down(num_frames: int) -> int:
    """Snap a frame count DOWN to the previous 17n+5 (minimum 5)."""
    num_frames = max(num_frames, LATENTS_PER_CHUNK)
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames -= 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    """17n+5 pixel frames -> 5n+2 latent frames."""
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be of the form 17n+5, got {num_frames}")
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """Audio latents covering `num_frames` video frames at 24 fps / 40 Hz."""
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> Tuple[int, int]:
    """Aspect ratio -> (height, width): short edge 768, area capped at 768*1344,
    both axes rounded to the nearest multiple of 32."""
    ratio = aspect_width / aspect_height
    if ratio >= 1.0:
        width, height = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        width, height = float(SHORT_EDGE), SHORT_EDGE / ratio
    area = width * height
    if area > MAX_PIXELS:
        scale = (MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def prepare_keyframe_image(
    image: Image.Image, height: int, width: int, stretch: bool = True
):
    """Put a keyframe onto the target canvas: the geometry anchor is stretched,
    a follower keyframe is cover-cropped."""
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (
        max(width, round(image.size[0] * scale)),
        max(height, round(image.size[1] * scale)),
    )
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


# ---------------------------------------------------------------------------
# Row packing
# ---------------------------------------------------------------------------


def patchify_video_latents(latents: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """(B, C, T, H, W) -> (B, N, C * prod(patch)) rows, frame-major then
    row-major, feature order [c, pt, ph, pw]."""
    pt, ph, pw = patch_size
    b, c, t, h, w = latents.shape
    latents = latents.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(b, -1, c * pt * ph * pw).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int = 24,
    patch_size=(1, 2, 2),
) -> torch.Tensor:
    """(B, N, C * prod(patch)) -> (B, C, T, H, W). Inverse of patchify."""
    pt, ph, pw = patch_size
    b = rows.shape[0]
    rows = rows.reshape(
        b,
        num_latent_frames // pt,
        latent_height // ph,
        latent_width // pw,
        channels,
        pt,
        ph,
        pw,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(
        b, channels, num_latent_frames, latent_height, latent_width
    ).contiguous()


def pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, 2, C, T) stereo audio latents -> (B, 2*T, C) channel-major rows
    (all T frames of channel 0, then channel 1)."""
    return (
        latents.permute(0, 1, 3, 2)
        .reshape(latents.shape[0], -1, latents.shape[2])
        .contiguous()
    )


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    """(B, 2*T, C) channel-major rows -> (B, 2, C, T)."""
    b, _, c = rows.shape
    rows = rows.reshape(b, AUDIO_CHANNELS, num_audio_latents, c)
    return rows.permute(0, 1, 3, 2).contiguous()


# ---------------------------------------------------------------------------
# Rotary grids (float64, numpy linspace — the released grid must reproduce)
# ---------------------------------------------------------------------------


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # numpy linspace(endpoint=False) is start + arange(n) * (stop-start)/n,
    # which is not bit-identical to torch.linspace
    grid = (
        np.linspace(left, left + ratio, dim // patch, endpoint=False)
        * _ROPE_SPATIAL_SCALE
    )
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE
            * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


def _temporal_position_span(num_latent_frames: int) -> float:
    # numpy pairwise sum on purpose: the reference computes the "last" keyframe
    # anchor this way and the summation orders differ in the last ulp
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for i in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[i :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[i]
    return float(spans.sum())


# ---------------------------------------------------------------------------
# Sequence layout
# ---------------------------------------------------------------------------


@dataclass
class PackedLayout:
    """Structural description of one packed sequence (one batch item)."""

    sequence_length: int
    position_ids: torch.Tensor  # (S, 3) float64
    token_tags: torch.Tensor  # (S,) long
    video_indices: torch.Tensor  # condition rows first, then target rows
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int


def build_packed_sequence(
    text_token_tags: torch.Tensor,  # (L,) long: 1 text, 0 for vision-block rows
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size=(1, 2, 2),
    keyframe_anchors: Tuple[str, ...] = (),
) -> PackedLayout:
    """Build the [text | keyframe conditions | target audio | target video]
    layout used by t2va and fl2va."""
    _, ph, pw = patch_size
    rows_per_frame = (latent_height // ph) * (latent_width // pw)
    num_text = int(text_token_tags.shape[0])
    num_cond = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    seq_len = num_text + num_cond + num_audio_rows + num_video_rows

    cond_start = num_text
    audio_start = cond_start + num_cond
    video_start = audio_start + num_audio_rows

    # text rows sit on the time axis at their row index; the media clock
    # continues from there, so prompt length shifts the whole media clock
    position_ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)

    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, ph, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, pw, sqrt_area)
    frame_grid = torch.stack(
        [g.reshape(-1) for g in torch.meshgrid(height_grid, width_grid, indexing="ij")],
        dim=-1,
    )

    for i, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text)
        elif anchor == "last":
            anchor_time = (
                float(num_text)
                + _temporal_position_span(num_latent_frames)
                - _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(
                f"keyframe anchor must be 'first' or 'last', got {anchor!r}"
            )
        rows = slice(
            cond_start + i * rows_per_frame, cond_start + (i + 1) * rows_per_frame
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # audio rows: channel-major, one rotary unit per latent (40/s = 24fps*5/3),
    # no height coordinate, width pinned to the grid extremes per channel
    audio_time = float(num_text) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full(
                (num_audio_latents,), float(width_grid[-1]), dtype=torch.float64
            ),
        ]
    )

    video_pos = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_pos[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text))[
        :, None
    ]
    video_pos[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_pos.reshape(-1, 3)

    video_indices = torch.cat(
        [torch.arange(cond_start, audio_start), torch.arange(video_start, seq_len)]
    )
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text)

    token_tags = torch.empty(seq_len, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    return PackedLayout(
        sequence_length=seq_len,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_cond,
    )


def build_row_timesteps(
    layout: PackedLayout,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: Optional[float] = None,
) -> torch.Tensor:
    """Per-row timestep values (S,) float32. Text rows inherit the video
    timestep; condition video rows stay pinned at their noise-aug level."""
    if condition_video_timestep is None:
        condition_video_timestep = max(video_timestep, KEYFRAME_NOISE_AUG_T)
    row_t = torch.full(
        (layout.sequence_length,), float(video_timestep), dtype=torch.float32
    )
    row_t[layout.video_indices[: layout.num_condition_video_rows]] = float(
        condition_video_timestep
    )
    row_t[layout.audio_indices] = float(audio_timestep)
    return row_t


def pad_layouts_to_batch(layouts: List[PackedLayout]):
    """Stack per-item layouts that share the same media geometry but may have
    different text lengths into batched transformer inputs.

    Items are right-padded in the TEXT segment to the batch's max text length
    (pad rows tagged -1, masked out of attention as keys, positions zero).
    Returns (position_ids (B, S, 3) f64, token_tags (B, S) long,
    video_indices, audio_indices, text_indices, pad_counts) where the index
    tensors describe the shared padded layout: [text_max | cond | audio | video].
    """
    max_text = max(int(l.text_indices.shape[0]) for l in layouts)
    ref = layouts[0]
    media_len = ref.sequence_length - int(ref.text_indices.shape[0])
    for l in layouts:
        if l.sequence_length - int(l.text_indices.shape[0]) != media_len:
            raise ValueError(
                "all layouts in a batch must share the same media geometry"
            )
    seq_len = max_text + media_len

    b = len(layouts)
    position_ids = torch.zeros(b, seq_len, 3, dtype=torch.float64)
    token_tags = torch.full((b, seq_len), PAD_TAG, dtype=torch.long)
    pad_counts = []
    for i, l in enumerate(layouts):
        lt = int(l.text_indices.shape[0])
        position_ids[i, :lt] = l.position_ids[:lt]
        position_ids[i, max_text:] = l.position_ids[lt:]
        token_tags[i, :lt] = l.token_tags[:lt]
        token_tags[i, max_text:] = l.token_tags[lt:]
        pad_counts.append(max_text - lt)

    offset = max_text - int(ref.text_indices.shape[0])
    video_indices = ref.video_indices + offset
    audio_indices = ref.audio_indices + offset
    text_indices = torch.arange(max_text)
    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        pad_counts,
    )


# ---------------------------------------------------------------------------
# Sigma-shift math (video shift 12, audio shift 3, exponential)
# ---------------------------------------------------------------------------


def shift_sigma(sigma, shift: float):
    """Exponential timeshift: shift * sigma / (1 + (shift - 1) * sigma)."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def remap_sigma(
    sigma, from_shift: float = VIDEO_SIGMA_SHIFT, to_shift: float = AUDIO_SIGMA_SHIFT
):
    """Map a sigma on the `from_shift` schedule onto the `to_shift` schedule
    at the same underlying schedule position (the video/audio coupling)."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return shift_sigma(base, to_shift)


def build_sigma_schedule(
    num_inference_steps: int, shift: float = VIDEO_SIGMA_SHIFT
) -> torch.Tensor:
    """The released sampling grid: linspace(1, 0, steps + 1) through the
    exponential shift, consecutive duplicates collapsed — `steps` yields
    `steps` model evaluations (the released repo counts the terminal 0 in
    `steps`; we don't, so sample_steps means model evals)."""
    base = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
    sigmas = shift_sigma(base, shift)
    return torch.unique_consecutive(sigmas)
