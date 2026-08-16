"""MiniMax-H3 sampling pipeline for ai-toolkit training previews.

Covers t2v (t2va) and first-frame i2v (fl2va), always denoising the joint
audio stream alongside the video (the packed sequence contains audio rows by
construction; decoding the audio track is optional).

MiniMax-H3 is guidance-distilled: there is no negative prompt, no CFG and
exactly one transformer forward per step. ``unconditional_embeds`` and
``guidance_scale`` are accepted for harness compatibility and ignored.

Scheduler (the released math, not diffusers'):
  - sigma grid: ``linspace(1, 0, steps + 1)`` through the exponential shift
    (video 12, audio 3), consecutive duplicates collapsed; ``steps`` yields
    ``steps`` model evaluations (steps = 1 is one full 1 -> 0 step)
  - the model consumes ``t = 1 - sigma`` (t = 1 means clean) and predicts the
    data-ward velocity ``clean - noise``: ``denoised = x + sigma * v``
  - Euler update ``x_next = r * x + (1 - r) * denoised`` with
    ``r = sigma_next / sigma``, evaluated in float32
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from . import packing
from .text_encoder import trim_caption_tokens
from .packing import (
    AUDIO_CHANNELS,
    AUDIO_SIGMA_SHIFT,
    FPS,
    KEYFRAME_NOISE_AUG_T,
    VIDEO_SIGMA_SHIFT,
    build_packed_sequence,
    build_row_timesteps,
    build_sigma_schedule,
    pack_audio_latents,
    patchify_video_latents,
    remap_sigma,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)


class MiniMaxH3Pipeline:
    """Lightweight sampler; receives the MinimaxH3Model (BaseModel subclass)
    and reuses its VAEs / transformer / device bookkeeping."""

    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,  # AdvancedPromptEmbeds: text_embeds [(L, 5120)], text_token_tags [(L,)]
        unconditional_embeds=None,  # ignored: MiniMax-H3 is guidance-distilled
        height: int = 768,
        width: int = 768,
        num_frames: int = 124,
        num_inference_steps: int = 28,
        guidance_scale: float = 1.0,  # ignored
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        ctrl_img: Optional[
            Image.Image
        ] = None,  # first-frame keyframe, already canvas-sized
        ref_images: Optional[
            list
        ] = None,  # ref2va references, already area-matched (own aspect, /32)
        with_audio: bool = True,
        **kwargs,
    ):
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer

        is_video = num_frames > 1
        if is_video:
            num_frames = packing.align_num_frames_down(num_frames)
            t_lat = packing.video_latent_num_frames(num_frames)
        else:
            # true single-frame generation (image mode, LTX-2.3 style): one
            # latent frame with keyframe-row geometry — the same layout image
            # datasets train with, so image LoRAs sample in-distribution
            num_frames = 1
            t_lat = 1
        h_lat = height // 16
        w_lat = width // 16
        a_lat = packing.audio_latent_num_frames(num_frames)

        text_embeds, token_tags = trim_caption_tokens(
            conditional_embeds.text_embeds[0],
            conditional_embeds.text_token_tags[0],
            getattr(model, "max_text_length", None),
        )
        text_embeds = text_embeds.to(device, dtype)
        token_tags = token_tags.to("cpu", torch.long)

        # --- packed layout -------------------------------------------------
        if ctrl_img is not None and ref_images:
            raise ValueError("ctrl_img (first frame) and ref_images are exclusive")
        anchors = ("first",) if ctrl_img is not None else ()
        # references keep their own aspect: latent dims come from each image
        # (PIL images are single-frame blocks; video refs arrive as latent
        # tensors (C, T, h, w) already encoded by the caller)
        ref_blocks = []
        for r in ref_images or []:
            if isinstance(r, dict):
                lat = r["latent"]
                a = r.get("audio_rows")
                a_lat = int(a.shape[0]) // 2 if a is not None else 0
                ref_blocks.append((lat.shape[1], lat.shape[2], lat.shape[3], a_lat))
            elif isinstance(r, torch.Tensor):
                ref_blocks.append((r.shape[1], r.shape[2], r.shape[3]))
            else:
                ref_blocks.append((1, r.size[1] // 16, r.size[0] // 16))
        ref_blocks = tuple(ref_blocks)
        layout = build_packed_sequence(
            text_token_tags=token_tags,
            num_latent_frames=t_lat,
            latent_height=h_lat,
            latent_width=w_lat,
            num_audio_latents=a_lat,
            keyframe_anchors=anchors,
            ref_blocks=ref_blocks,
        )
        num_cond = layout.num_condition_video_rows

        # --- conditioning rows (draw order: condition noise, video, audio) --
        def encode_condition_image(img: Image.Image) -> torch.Tensor:
            cond_noise = randn_tensor(
                (1, 24, 1, img.size[1] // 16, img.size[0] // 16),
                generator=generator,
                dtype=torch.float32,
            ).to(device)
            frame = torch.from_numpy(np.array(img)).float()
            frame = (frame / 255.0) * 2.0 - 1.0  # (H, W, 3) -> [-1, 1]
            frame = frame.permute(2, 0, 1)[None, :, None]  # (1, 3, 1, H, W)
            cond_latents = model.encode_keyframe_latents(frame)  # (1, 24, 1, h, w) fp32
            # released noise-aug recipe: x = t * clean + (1 - t) * noise at t = 0.999
            cond_latents = (
                KEYFRAME_NOISE_AUG_T * cond_latents.to(device)
                + (1.0 - KEYFRAME_NOISE_AUG_T) * cond_noise
            )
            return patchify_video_latents(cond_latents)  # (1, rows, 96)

        def noise_aug_rows(latents: torch.Tensor) -> torch.Tensor:
            cond_noise = randn_tensor(
                latents.shape, generator=generator, dtype=torch.float32
            ).to(device)
            mixed = (
                KEYFRAME_NOISE_AUG_T * latents.to(device, torch.float32)
                + (1.0 - KEYFRAME_NOISE_AUG_T) * cond_noise
            )
            return patchify_video_latents(mixed)

        cond_rows = None
        cond_audio_rows = None
        if ctrl_img is not None:
            cond_rows = encode_condition_image(ctrl_img)
        elif ref_images:
            parts = []
            audio_parts = []
            for r in ref_images:
                if isinstance(r, dict):
                    # pre-encoded video reference (+ optional clean soundtrack)
                    parts.append(noise_aug_rows(r["latent"][None]))
                    if r.get("audio_rows") is not None:
                        audio_parts.append(r["audio_rows"][None].to(device))
                elif isinstance(r, torch.Tensor):
                    parts.append(noise_aug_rows(r[None]))
                else:
                    parts.append(encode_condition_image(r))
            cond_rows = torch.cat(parts, dim=1)
            if audio_parts:
                cond_audio_rows = torch.cat(audio_parts, dim=1).float()

        # --- initial noise -------------------------------------------------
        if latents is None:
            latents = randn_tensor(
                (1, 24, t_lat, h_lat, w_lat), generator=generator, dtype=torch.float32
            )
        video_rows = patchify_video_latents(latents.to(device).float())  # (1, V, 96)
        audio_noise = randn_tensor(
            (1, AUDIO_CHANNELS, 32, a_lat), generator=generator, dtype=torch.float32
        ).to(device)
        audio_rows = pack_audio_latents(audio_noise)  # (1, 2*A, 32)

        # --- schedules -----------------------------------------------------
        sigmas_v = build_sigma_schedule(num_inference_steps, VIDEO_SIGMA_SHIFT).to(
            device
        )
        # the audio schedule follows the video grid through the closed-form
        # shift remap so both streams sit at the same underlying position
        sigmas_a = remap_sigma(sigmas_v, VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT)

        position_ids = layout.position_ids[None].to(device)
        tags = layout.token_tags[None].to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)

        # --- denoise loop --------------------------------------------------
        num_steps = sigmas_v.shape[0] - 1
        for i in range(num_steps):
            sv, sv_next = sigmas_v[i], sigmas_v[i + 1]
            sa, sa_next = sigmas_a[i], sigmas_a[i + 1]
            t_v = 1.0 - float(sv)
            t_a = 1.0 - float(sa)

            row_t = build_row_timesteps(layout, t_v, t_a)[None].to(device)

            video_in = video_rows
            if cond_rows is not None:
                video_in = torch.cat([cond_rows, video_rows], dim=1)
            audio_in = audio_rows
            if cond_audio_rows is not None:
                audio_in = torch.cat([cond_audio_rows, audio_rows], dim=1)

            video_pred, audio_pred = transformer(
                hidden_states=video_in.to(dtype),
                audio_hidden_states=audio_in.to(dtype),
                encoder_hidden_states=text_embeds[None],
                row_timesteps=row_t,
                token_tags=tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
            )
            v_video = video_pred[:, num_cond:].float()
            v_audio = audio_pred[:, layout.num_condition_audio_rows :].float()

            denoised_v = video_rows + sv * v_video
            ratio_v = sv_next / sv
            video_rows = ratio_v * video_rows + (1.0 - ratio_v) * denoised_v

            denoised_a = audio_rows + sa * v_audio
            ratio_a = sa_next / sa if float(sa) != 0.0 else 0.0
            audio_rows = ratio_a * audio_rows + (1.0 - ratio_a) * denoised_a

        # --- decode --------------------------------------------------------
        video_latents = unpatchify_video_tokens(video_rows, t_lat, h_lat, w_lat)
        video = model.decode_latents(video_latents)  # (1, 3, T, H, W) in [-1, 1]
        video = ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        video = video[0].permute(1, 2, 3, 0).cpu()  # (T, H, W, C)

        if not is_video:
            return [Image.fromarray(video[0].numpy())]

        audio_out = None
        if with_audio:
            audio_latents = unpack_audio_tokens(audio_rows, a_lat)[0]  # (2, 32, A)
            waveform = model.decode_audio_latents(
                audio_latents.float()
            )  # (2, 1, samples)
            audio_out = waveform[:, 0].cpu()  # (2, samples) stereo

        return {
            "video": video,
            "fps": FPS,
            "audio": audio_out,
            "audio_sample_rate": packing.AUDIO_SAMPLE_RATE,
            "output_path": None,
        }
