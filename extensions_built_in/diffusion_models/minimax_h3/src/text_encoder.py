"""Qwen3-VL conditioning for MiniMax-H3.

MiniMax-H3 conditions on the **unnormalized** ``hidden_states[50]`` of its
Qwen3-VL-32B conditioner (``hidden_states[0]`` is the embedding output, so
this is the output of decoder layer 49, before the final norm). The LM head
and layers 50..63 are never used, which lets the loader truncate the stack.

The presentation is raw tokens — no chat template, no special tokens:

  - t2va: the verbatim prompt.
  - fl2va: per keyframe, a ``"<Picture i>: "`` label plus a vision block
    (``<|vision_start|>`` + one ``<|image_pad|>`` per merged vision patch +
    ``<|vision_end|>``), then the verbatim prompt. Vision-block rows are
    tagged as *video* (0) rather than text (1) — the transformer's AdaLN
    modality selection keys off these tags.
"""

from typing import List, Optional

import torch

from dataclasses import dataclass, field

from .packing import TEXT_TAG, VIDEO_TAG

TEXT_ENCODER_LAYER = 50


@dataclass
class VideoRef:
    """A reference VIDEO for the Qwen3-VL presentation: frames sampled at
    2 fps with their timestamps (seconds). Presented ComfyUI-style as
    ``<Video k>: `` plus one ``<T.T seconds>``-stamped vision block per
    merged frame pair."""

    frames: list = field(default_factory=list)  # PIL images
    timestamps: list = field(default_factory=list)  # float seconds, per frame
    # a soundtrack that rides as reference audio rows: the presentation gets an
    # "<Audio j>: " label emitted BEFORE the "<Video k>: " block (audio itself
    # never enters Qwen)
    has_audio: bool = False


def video_has_audio(path) -> bool:
    try:
        import av

        with av.open(path) as c:
            return len(c.streams.audio) > 0
    except Exception:
        return False


def load_video_ref(path, max_frames: int = 0, has_audio=None) -> "VideoRef":
    """Sample a video at 2 fps (slot rounding on its native fps) into a
    VideoRef with per-frame timestamps in seconds."""
    import cv2
    from PIL import Image as _Image

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open control video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = []
    i = 0
    while True:
        idx = round(i * fps / 2.0)
        if idx >= total:
            break
        if not indices or idx != indices[-1]:
            indices.append(idx)
        i += 1
        if max_frames and len(indices) >= max_frames:
            break
    frames, times = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        times.append(idx / fps)
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from control video {path}")
    if has_audio is None:
        has_audio = video_has_audio(path)
    return VideoRef(frames=frames, timestamps=times, has_audio=has_audio)


def trim_caption_tokens(embeds: torch.Tensor, tags: torch.Tensor, max_length):
    """Cap the CAPTION rows of an already-encoded prompt at ``max_length``.

    The presentation is ``[labels + vision blocks ...] + caption``: everything
    up to and including the last vision row (tag 0) is structural conditioning
    and kept intact; only the trailing text-tagged caption tail is truncated.
    Lets embeds cached with a longer max_text_length serve a shorter one
    without re-encoding. Returns (embeds, tags) — same objects when nothing to
    trim.
    """
    if max_length is None or max_length <= 0:
        return embeds, tags
    is_vision = tags.to("cpu") == VIDEO_TAG
    if bool(is_vision.any()):
        head = int(is_vision.nonzero()[-1].item()) + 1
    else:
        head = 0
    caption_len = int(tags.shape[0]) - head
    if caption_len <= max_length:
        return embeds, tags
    keep = head + max_length
    return embeds[:keep], tags[:keep]


@torch.no_grad()
def encode_minimax_h3_prompt(
    text_encoder,  # transformers Qwen3VLForConditionalGeneration
    tokenizer,  # Qwen2TokenizerFast
    processor,  # Qwen3VLProcessor (needed only when keyframes are present)
    prompt: str,
    keyframes: Optional[List] = None,  # PIL images already on the target canvas
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_length: Optional[
        int
    ] = None,  # cap on PROMPT tokens (vision blocks are never cut)
):
    """Encode ONE prompt (with optional keyframes) into MiniMax-H3 conditioning.

    Returns (embeds (L, 5120), token_tags (L,) long). The embeds come from
    ``hidden_states[50]`` unnormalized. A stack truncated to exactly 50 layers
    also works ONLY if the final ``model.norm`` has been replaced with an
    Identity (transformers applies the final norm to the last entry of
    ``hidden_states``); the loader in minimax_h3.py does exactly that.
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers < TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 needs at least {TEXT_ENCODER_LAYER} Qwen3-VL decoder "
            f"layers to read hidden_states[{TEXT_ENCODER_LAYER}], got {num_layers}"
        )
    if device is None:
        device = text_encoder.device

    pixel_values, image_grid_thw = None, None
    pixel_values_videos, video_grid_thw = None, None
    token_ids: List[int] = []
    token_tags: List[int] = []
    if keyframes:
        images = [k for k in keyframes if not isinstance(k, VideoRef)]
        videos = [k for k in keyframes if isinstance(k, VideoRef)]
        merge = processor.image_processor.merge_size**2
        vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
        if videos:
            import numpy as np

            # frames are already sampled at 2 fps: do_sample_frames=False keeps
            # the video processor from resampling them (official diffusers
            # ref2va encoder passes the same flag)
            vids = processor.video_processor(
                videos=[np.stack([np.asarray(f.convert("RGB")) for f in v.frames]) for v in videos],
                do_sample_frames=False,
                return_tensors="pt",
            )
            pixel_values_videos = vids["pixel_values_videos"]
            video_grid_thw = vids["video_grid_thw"]

        pic_idx, vid_idx, aud_idx = 0, 0, 0
        for k in keyframes:
            if isinstance(k, VideoRef):
                grid = video_grid_thw[vid_idx]
                per_pair = int(grid[1] * grid[2]) // merge
                label_ids = []
                if k.has_audio:
                    aud_idx += 1
                    label_ids += tokenizer(
                        f"<Audio {aud_idx}>: ", add_special_tokens=False
                    )["input_ids"]
                label_ids += tokenizer(
                    f"<Video {vid_idx + 1}>: ", add_special_tokens=False
                )["input_ids"]
                token_ids += label_ids
                token_tags += [TEXT_TAG] * len(label_ids)
                # one timestamped vision block per merged frame PAIR: the
                # video processor merges temporal_patch_size=2 frames, repeat-
                # padding an odd count; the timestamp is the pair's mean time
                times = list(k.timestamps)
                if len(times) % 2 == 1:
                    times.append(times[-1])
                for t_pair in range(int(grid[0])):
                    mean_t = (times[2 * t_pair] + times[2 * t_pair + 1]) / 2.0
                    ts_ids = tokenizer(
                        f"<{round(mean_t, 1):.1f} seconds>", add_special_tokens=False
                    )["input_ids"]
                    vision_ids = [vision_start] + [video_pad] * per_pair + [vision_end]
                    token_ids += ts_ids + vision_ids
                    token_tags += [TEXT_TAG] * len(ts_ids) + [VIDEO_TAG] * len(
                        vision_ids
                    )
                vid_idx += 1
            else:
                num_image_tokens = int(image_grid_thw[pic_idx].prod()) // merge
                label_ids = tokenizer(
                    f"<Picture {pic_idx + 1}>: ", add_special_tokens=False
                )["input_ids"]
                vision_ids = (
                    [vision_start] + [image_pad] * num_image_tokens + [vision_end]
                )
                token_ids += label_ids + vision_ids
                token_tags += [TEXT_TAG] * len(label_ids) + [VIDEO_TAG] * len(
                    vision_ids
                )
                pic_idx += 1

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if max_length is not None and max_length > 0:
        # the cap applies to the caption only; a keyframe's vision block is
        # structural conditioning and cannot be truncated without corrupting it
        prompt_ids = prompt_ids[:max_length]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    if len(token_ids) == 0:
        # empty (unconditional) prompt: a single pad token keeps the sequence
        # non-degenerate; the model was not trained with CFG so this is only
        # ever a fallback
        token_ids = [tokenizer.pad_token_id or 0]
        token_tags = [TEXT_TAG]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
    )

    # call the inner .model directly: the LM head's vocab projection is dead
    # weight here and hidden_states[50] is all that is consumed
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None
        if pixel_values is None
        else pixel_values.to(device, text_encoder.dtype),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
        pixel_values_videos=None
        if pixel_values_videos is None
        else pixel_values_videos.to(device, text_encoder.dtype),
        video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(device),
        use_cache=False,
        output_hidden_states=True,
    )
    layer = min(TEXT_ENCODER_LAYER, len(outputs.hidden_states) - 1)
    embeds = outputs.hidden_states[layer][0]
    if dtype is not None:
        embeds = embeds.to(dtype)
    return embeds, torch.tensor(token_tags, dtype=torch.long)
