from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import (
    AudioLatentShape,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@dataclass
class LTX23CachedPromptEmbeddings:
    video_context_positive: torch.Tensor
    audio_context_positive: torch.Tensor
    video_context_negative: Optional[torch.Tensor] = None
    audio_context_negative: Optional[torch.Tensor] = None


@dataclass
class LTX23TiledDecodingConfig:
    enabled: bool = True
    tile_size_pixels: int = 192
    tile_overlap_pixels: int = 64
    tile_size_frames: int = 48
    tile_overlap_frames: int = 24


@dataclass
class LTX23GenerationConfig:
    height: int
    width: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    guidance_scale: float
    seed: int
    cached_embeddings: LTX23CachedPromptEmbeddings
    condition_image: Optional[torch.Tensor] = None
    generate_audio: bool = True
    tiled_decoding: Optional[LTX23TiledDecodingConfig] = None

    def __post_init__(self):
        if self.tiled_decoding is None:
            self.tiled_decoding = LTX23TiledDecodingConfig()


class LTX23OfficialSampler:
    def __init__(
        self,
        transformer: torch.nn.Module,
        vae_decoder: torch.nn.Module,
        vae_encoder: Optional[torch.nn.Module] = None,
        audio_decoder: Optional[torch.nn.Module] = None,
        vocoder: Optional[torch.nn.Module] = None,
    ) -> None:
        self._transformer = transformer
        self._vae_decoder = vae_decoder
        self._vae_encoder = vae_encoder
        self._audio_decoder = audio_decoder
        self._vocoder = vocoder
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

    def set_transformer(self, transformer: torch.nn.Module) -> None:
        self._transformer = transformer

    def set_vae_encoder(self, vae_encoder: torch.nn.Module) -> None:
        self._vae_encoder = vae_encoder

    @property
    def output_sampling_rate(self) -> int:
        return int(getattr(self._vocoder, "output_sampling_rate", 24000))

    @torch.no_grad()
    def generate(
        self,
        config: LTX23GenerationConfig,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._validate_config(config)

        cached = config.cached_embeddings
        v_ctx_pos = cached.video_context_positive.to(device=device, dtype=torch.bfloat16)
        a_ctx_pos = cached.audio_context_positive.to(device=device, dtype=torch.bfloat16)
        v_ctx_neg = (
            cached.video_context_negative.to(device=device, dtype=torch.bfloat16)
            if cached.video_context_negative is not None
            else None
        )
        a_ctx_neg = (
            cached.audio_context_negative.to(device=device, dtype=torch.bfloat16)
            if cached.audio_context_negative is not None
            else None
        )

        generator = torch.Generator(device=device).manual_seed(int(config.seed))
        video_tools = self._create_video_latent_tools(config)
        audio_tools = self._create_audio_latent_tools(config) if config.generate_audio else None

        video_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        audio_clean_state = (
            audio_tools.create_initial_state(device=device, dtype=torch.bfloat16)
            if audio_tools is not None
            else None
        )

        if config.condition_image is not None:
            video_clean_state = self._apply_image_conditioning(
                video_state=video_clean_state,
                image=config.condition_image,
                config=config,
                device=device,
            )

        noiser = GaussianNoiser(generator=generator)
        video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)
        audio_state = (
            noiser(latent_state=audio_clean_state, noise_scale=1.0)
            if audio_clean_state is not None
            else None
        )

        original_transformer_device = _module_device(self._transformer)
        video_state, audio_state = self._run_denoising(
            config=config,
            video_state=video_state,
            audio_state=audio_state,
            video_clean_state=video_clean_state,
            audio_clean_state=audio_clean_state,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            device=device,
        )
        if original_transformer_device == torch.device("cpu"):
            self._transformer.to("cpu")

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        video_output = self._decode_video(video_state.latent, config, device)

        audio_output = None
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            audio_output = self._decode_audio(audio_state.latent, device)

        return video_output, audio_output

    def _validate_config(self, config: LTX23GenerationConfig) -> None:
        if config.height % 32 != 0 or config.width % 32 != 0:
            raise ValueError(
                "LTX-2.3 sample height and width must be divisible by 32, "
                "got {}x{}".format(config.height, config.width)
            )
        if config.num_frames % 8 != 1:
            raise ValueError(
                "LTX-2.3 sample num_frames must satisfy num_frames % 8 == 1, got {}".format(
                    config.num_frames
                )
            )
        if config.generate_audio and (self._audio_decoder is None or self._vocoder is None):
            raise ValueError("LTX-2.3 audio generation requires the official audio decoder and vocoder")
        if config.condition_image is not None and self._vae_encoder is None:
            raise ValueError("LTX-2.3 image-to-video sampling requires the official video encoder")

    def _create_video_latent_tools(self, config: LTX23GenerationConfig) -> VideoLatentTools:
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.frame_rate,
        )
        return VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(shape=pixel_shape),
            fps=config.frame_rate,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        )

    def _create_audio_latent_tools(self, config: LTX23GenerationConfig) -> AudioLatentTools:
        duration = float(config.num_frames) / float(config.frame_rate)
        return AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=duration),
        )

    def _run_denoising(
        self,
        config: LTX23GenerationConfig,
        video_state,
        audio_state,
        video_clean_state,
        audio_clean_state,
        v_ctx_pos: torch.Tensor,
        a_ctx_pos: torch.Tensor,
        v_ctx_neg: Optional[torch.Tensor],
        a_ctx_neg: Optional[torch.Tensor],
        device: torch.device,
    ):
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=config.num_inference_steps).to(device).float()
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(config.guidance_scale)

        video = Modality(
            enabled=True,
            latent=video_state.latent,
            sigma=sigmas[0].repeat(video_state.latent.shape[0]),
            timesteps=sigmas[0] * video_state.denoise_mask.squeeze(-1),
            positions=video_state.positions,
            context=v_ctx_pos,
            context_mask=None,
        )

        audio = None
        if audio_state is not None:
            audio = Modality(
                enabled=True,
                latent=audio_state.latent,
                sigma=sigmas[0].repeat(audio_state.latent.shape[0]),
                timesteps=sigmas[0] * audio_state.denoise_mask.squeeze(-1),
                positions=audio_state.positions,
                context=a_ctx_pos,
                context_mask=None,
            )

        self._transformer.to(device)
        x0_model = X0Model(self._transformer)

        with _autocast_context(device):
            for step_index, sigma in enumerate(sigmas[:-1]):
                video = replace(
                    video,
                    timesteps=sigma * video_state.denoise_mask.squeeze(-1),
                    sigma=sigma.repeat(video_state.latent.shape[0]),
                    latent=video_state.latent,
                    positions=video_state.positions,
                )

                if audio is not None and audio_state is not None:
                    audio = replace(
                        audio,
                        timesteps=sigma * audio_state.denoise_mask.squeeze(-1),
                        sigma=sigma.repeat(audio_state.latent.shape[0]),
                        latent=audio_state.latent,
                        positions=audio_state.positions,
                    )

                pos_video, pos_audio = x0_model(video=video, audio=audio, perturbations=None)
                denoised_video, denoised_audio = pos_video, pos_audio

                if cfg_guider.enabled() and v_ctx_neg is not None:
                    video_neg = replace(video, context=v_ctx_neg)
                    audio_neg = (
                        replace(audio, context=a_ctx_neg if a_ctx_neg is not None else a_ctx_pos)
                        if audio is not None
                        else None
                    )
                    neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)
                    denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)
                    if audio is not None and denoised_audio is not None and neg_audio is not None:
                        denoised_audio = denoised_audio + cfg_guider.delta(pos_audio, neg_audio)

                denoised_video = denoised_video * video_state.denoise_mask + video_clean_state.latent.float() * (
                    1 - video_state.denoise_mask
                )
                if audio is not None and audio_state is not None and audio_clean_state is not None:
                    denoised_audio = denoised_audio * audio_state.denoise_mask + audio_clean_state.latent.float() * (
                        1 - audio_state.denoise_mask
                    )

                video_state = replace(
                    video_state,
                    latent=stepper.step(
                        sample=video.latent,
                        denoised_sample=denoised_video,
                        sigmas=sigmas,
                        step_index=step_index,
                    ),
                )
                if audio is not None and audio_state is not None:
                    audio_state = replace(
                        audio_state,
                        latent=stepper.step(
                            sample=audio.latent,
                            denoised_sample=denoised_audio,
                            sigmas=sigmas,
                            step_index=step_index,
                        ),
                    )

        return video_state, audio_state

    def _apply_image_conditioning(
        self,
        video_state,
        image: torch.Tensor,
        config: LTX23GenerationConfig,
        device: torch.device,
    ):
        encoded_image = self._encode_conditioning_image(image, config.height, config.width, device)
        patchified_image = self._video_patchifier.patchify(encoded_image)
        num_image_tokens = patchified_image.shape[1]

        latent = video_state.latent.clone()
        clean_latent = video_state.clean_latent.clone()
        denoise_mask = video_state.denoise_mask.clone()

        latent[:, :num_image_tokens] = patchified_image.to(latent.dtype)
        clean_latent[:, :num_image_tokens] = patchified_image.to(clean_latent.dtype)
        denoise_mask[:, :num_image_tokens] = 0.0

        return replace(
            video_state,
            latent=latent,
            clean_latent=clean_latent,
            denoise_mask=denoise_mask,
        )

    def _encode_conditioning_image(
        self,
        image: torch.Tensor,
        target_height: int,
        target_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        current_height, current_width = image.shape[1:]
        if current_height != target_height or current_width != target_width:
            aspect_ratio = float(current_width) / float(current_height)
            target_aspect_ratio = float(target_width) / float(target_height)

            if aspect_ratio > target_aspect_ratio:
                resize_height = target_height
                resize_width = int(round(target_height * aspect_ratio))
            else:
                resize_height = int(round(target_width / aspect_ratio))
                resize_width = target_width

            image = image.unsqueeze(0)
            image = F.interpolate(
                image,
                size=(resize_height, resize_width),
                mode="bilinear",
                align_corners=False,
            )
            h_start = (resize_height - target_height) // 2
            w_start = (resize_width - target_width) // 2
            image = image[:, :, h_start : h_start + target_height, w_start : w_start + target_width]
        else:
            image = image.unsqueeze(0)

        image = rearrange(image, "b c h w -> b c 1 h w")
        image = (image * 2.0 - 1.0).to(device=device, dtype=torch.float32)

        original_encoder_device = _module_device(self._vae_encoder)
        self._vae_encoder.to(device)
        with _autocast_context(device):
            encoded = self._vae_encoder(image)
        if original_encoder_device == torch.device("cpu"):
            self._vae_encoder.to("cpu")

        return encoded

    def _decode_video(
        self,
        latent: torch.Tensor,
        config: LTX23GenerationConfig,
        device: torch.device,
    ) -> torch.Tensor:
        original_decoder_device = _module_device(self._vae_decoder)
        self._vae_decoder.to(device)
        latent = latent.to(dtype=torch.bfloat16)

        tiled_config = config.tiled_decoding
        if tiled_config is not None and tiled_config.enabled:
            tiling_config = TilingConfig(
                spatial_config=SpatialTilingConfig(
                    tile_size_in_pixels=tiled_config.tile_size_pixels,
                    tile_overlap_in_pixels=tiled_config.tile_overlap_pixels,
                ),
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=tiled_config.tile_size_frames,
                    tile_overlap_in_frames=tiled_config.tile_overlap_frames,
                ),
            )
            chunks = []
            for chunk in self._vae_decoder.tiled_decode(latent, tiling_config=tiling_config):
                chunks.append(chunk)
            decoded_video = torch.cat(chunks, dim=2)
        else:
            with _autocast_context(device):
                decoded_video = self._vae_decoder(latent)

        if original_decoder_device == torch.device("cpu"):
            self._vae_decoder.to("cpu")

        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        return decoded_video[0].float().cpu()

    def _decode_audio(
        self,
        latent: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        original_audio_decoder_device = _module_device(self._audio_decoder)
        self._audio_decoder.to(device)
        latent = latent.to(dtype=torch.bfloat16)
        with _autocast_context(device):
            decoded_audio = self._audio_decoder(latent)
        if original_audio_decoder_device == torch.device("cpu"):
            self._audio_decoder.to("cpu")

        original_vocoder_device = _module_device(self._vocoder)
        self._vocoder.to(device)
        with _autocast_context(device):
            audio_waveform = self._vocoder(decoded_audio)
        if original_vocoder_device == torch.device("cpu"):
            self._vocoder.to("cpu")

        return audio_waveform.squeeze(0).float().cpu()
