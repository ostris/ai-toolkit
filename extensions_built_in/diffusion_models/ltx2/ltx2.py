from functools import partial
import os
from typing import List, Optional

import torch
import torchaudio
from transformers import Gemma3Config
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from accelerate import init_empty_weights
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file
from PIL import Image

try:
    from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline
    from diffusers.models.autoencoders import (
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
    )
    from diffusers.models.transformers import LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2.export_utils import encode_video
    from transformers import (
        Gemma3ForConditionalGeneration,
        GemmaTokenizerFast,
    )
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from .convert_ltx2_to_diffusers import (
        get_model_state_dict_from_combined_ckpt,
        convert_ltx2_transformer,
        convert_ltx2_video_vae,
        convert_ltx2_audio_vae,
        convert_ltx2_vocoder,
        convert_ltx2_connectors,
        dequantize_state_dict,
        convert_comfy_gemma3_to_transformers,
        convert_lora_original_to_diffusers,
        convert_lora_diffusers_to_original,
    )
except ImportError as e:
    print("Diffusers import error:", e)
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


scheduler_config = {
    "base_image_seq_len": 1024,
    "base_shift": 0.95,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 2.05,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.1,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

dit_prefix = "model.diffusion_model."
vae_prefix = "vae."
audio_vae_prefix = "audio_vae."
vocoder_prefix = "vocoder."


def new_save_image_function(
    self: GenerateImageConfig,
    image,  # will contain a dict that can be dumped ditectly into encode_video, just add output_path to it.
    count: int = 0,
    max_count: int = 0,
    **kwargs,
):
    # this replaces gen image config save image function so we can save the video with sound from ltx2
    image["output_path"] = self.get_image_path(count, max_count)
    # make sample directory if it does not exist
    os.makedirs(os.path.dirname(image["output_path"]), exist_ok=True)
    encode_video(**image)
    flush()


def blank_log_image_function(self, *args, **kwargs):
    # todo handle wandb logging of videos with audio
    return


class ComboVae(torch.nn.Module):
    """Combines video and audio VAEs for joint encoding and decoding."""

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.audio_vae = audio_vae

    @property
    def device(self):
        return self.vae.device

    @property
    def dtype(self):
        return self.vae.dtype

    def encode(
        self,
        *args,
        **kwargs,
    ):
        return self.vae.encode(*args, **kwargs)

    def decode(
        self,
        *args,
        **kwargs,
    ):
        return self.vae.decode(*args, **kwargs)


class AudioProcessor(torch.nn.Module):
    """Converts audio waveforms to log-mel spectrograms with optional resampling."""

    def __init__(
        self,
        sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_waveform(
        self,
        waveform: torch.Tensor,
        source_rate: int,
        target_rate: int,
    ) -> torch.Tensor:
        """Resample waveform to target sample rate if needed."""
        if source_rate == target_rate:
            return waveform
        resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
        return resampled.to(device=waveform.device, dtype=waveform.dtype)

    def waveform_to_mel(
        self,
        waveform: torch.Tensor,
        waveform_sample_rate: int,
    ) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram [batch, channels, time, n_mels]."""
        waveform = self.resample_waveform(
            waveform, waveform_sample_rate, self.sample_rate
        )

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()


class LTX2Model(BaseModel):
    arch = "ltx2"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["LTX2VideoTransformer3DModel"]
        # defines if the model supports model paths. Only some will
        self.supports_model_paths = True
        # use the new format on this new model by default
        self.use_old_lokr_format = False
        self.audio_processor = None
        
        # gemma needs left side padding
        self.te_padding_side = "left"

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 32

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading LTX2 model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        combined_state_dict = None

        self.print_and_status_update("Loading transformer")
        # if we have a safetensors file it is a mono checkpoint
        if os.path.exists(model_path) and model_path.endswith(".safetensors"):
            combined_state_dict = load_file(model_path)
            combined_state_dict = dequantize_state_dict(combined_state_dict)

        if combined_state_dict is not None:
            original_dit_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, dit_prefix
            )
            transformer = convert_ltx2_transformer(original_dit_ckpt)
            transformer = transformer.to(dtype)
        else:
            transformer_path = model_path
            transformer_subfolder = "transformer"
            if os.path.exists(transformer_path):
                transformer_subfolder = None
                transformer_path = os.path.join(transformer_path, "transformer")
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, "text_encoder")
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = LTX2VideoTransformer3DModel.from_pretrained(
                transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
            )

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            ignore_modules = []
            for block in transformer.transformer_blocks:
                ignore_modules.append(block.scale_shift_table)
                ignore_modules.append(block.audio_scale_shift_table)
                ignore_modules.append(block.video_a2v_cross_attn_scale_shift_table)
                ignore_modules.append(block.audio_a2v_cross_attn_scale_shift_table)
            ignore_modules.append(transformer.scale_shift_table)
            ignore_modules.append(transformer.audio_scale_shift_table)
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=ignore_modules,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Loading text encoder")
        if (
            self.model_config.te_name_or_path is not None
            and self.model_config.te_name_or_path.endswith(".safetensors")
        ):
            # load from comfyui gemma3 checkpoint
            tokenizer = GemmaTokenizerFast.from_pretrained(
                "Lightricks/LTX-2", subfolder="tokenizer"
            )

            with init_empty_weights():
                text_encoder = Gemma3ForConditionalGeneration(
                    Gemma3Config(
                        **{
                            "boi_token_index": 255999,
                            "bos_token_id": 2,
                            "eoi_token_index": 256000,
                            "eos_token_id": 106,
                            "image_token_index": 262144,
                            "initializer_range": 0.02,
                            "mm_tokens_per_image": 256,
                            "model_type": "gemma3",
                            "pad_token_id": 0,
                            "text_config": {
                                "attention_bias": False,
                                "attention_dropout": 0.0,
                                "attn_logit_softcapping": None,
                                "cache_implementation": "hybrid",
                                "final_logit_softcapping": None,
                                "head_dim": 256,
                                "hidden_activation": "gelu_pytorch_tanh",
                                "hidden_size": 3840,
                                "initializer_range": 0.02,
                                "intermediate_size": 15360,
                                "max_position_embeddings": 131072,
                                "model_type": "gemma3_text",
                                "num_attention_heads": 16,
                                "num_hidden_layers": 48,
                                "num_key_value_heads": 8,
                                "query_pre_attn_scalar": 256,
                                "rms_norm_eps": 1e-06,
                                "rope_local_base_freq": 10000,
                                "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
                                "rope_theta": 1000000,
                                "sliding_window": 1024,
                                "sliding_window_pattern": 6,
                                "torch_dtype": "bfloat16",
                                "use_cache": True,
                                "vocab_size": 262208,
                            },
                            "torch_dtype": "bfloat16",
                            "transformers_version": "4.51.3",
                            "unsloth_fixed": True,
                            "vision_config": {
                                "attention_dropout": 0.0,
                                "hidden_act": "gelu_pytorch_tanh",
                                "hidden_size": 1152,
                                "image_size": 896,
                                "intermediate_size": 4304,
                                "layer_norm_eps": 1e-06,
                                "model_type": "siglip_vision_model",
                                "num_attention_heads": 16,
                                "num_channels": 3,
                                "num_hidden_layers": 27,
                                "patch_size": 14,
                                "torch_dtype": "bfloat16",
                                "vision_use_head": False,
                            },
                        }
                    )
                )
            te_state_dict = load_file(self.model_config.te_name_or_path)
            te_state_dict = convert_comfy_gemma3_to_transformers(te_state_dict)
            for key in te_state_dict:
                te_state_dict[key] = te_state_dict[key].to(dtype)

            text_encoder.load_state_dict(te_state_dict, assign=True, strict=True)
            del te_state_dict
            flush()
        else:
            if self.model_config.te_name_or_path is not None:
                te_path = self.model_config.te_name_or_path
            else:
                te_path = base_model_path
            tokenizer = GemmaTokenizerFast.from_pretrained(
                te_path, subfolder="tokenizer"
            )
            text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                te_path, subfolder="text_encoder", dtype=dtype
            )

        # remove the vision tower
        text_encoder.model.vision_tower = None
        flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
                ignore_modules=[
                    text_encoder.model.language_model.base_model.embed_tokens
                ],
            )

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAEs and other components")
        if combined_state_dict is not None:
            original_vae_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, vae_prefix
            )
            vae = convert_ltx2_video_vae(original_vae_ckpt).to(dtype)
            del original_vae_ckpt
            original_audio_vae_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, audio_vae_prefix
            )
            audio_vae = convert_ltx2_audio_vae(original_audio_vae_ckpt).to(dtype)
            del original_audio_vae_ckpt
            original_connectors_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, dit_prefix
            )
            connectors = convert_ltx2_connectors(original_connectors_ckpt).to(dtype)
            del original_connectors_ckpt
            original_vocoder_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, vocoder_prefix
            )
            vocoder = convert_ltx2_vocoder(original_vocoder_ckpt).to(dtype)
            del original_vocoder_ckpt
            del combined_state_dict
            flush()
        else:
            vae = AutoencoderKLLTX2Video.from_pretrained(
                base_model_path, subfolder="vae", torch_dtype=dtype
            )
            audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
                base_model_path, subfolder="audio_vae", torch_dtype=dtype
            )

            connectors = LTX2TextConnectors.from_pretrained(
                base_model_path, subfolder="connectors", torch_dtype=dtype
            )

            vocoder = LTX2Vocoder.from_pretrained(
                base_model_path, subfolder="vocoder", torch_dtype=dtype
            )

        self.noise_scheduler = LTX2Model.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: LTX2Pipeline = LTX2Pipeline(
            scheduler=self.noise_scheduler,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=None,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=None,
            vocoder=vocoder,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = ComboVae(pipe.vae, pipe.audio_vae)
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe

        self.audio_processor = AudioProcessor(
            sample_rate=pipe.audio_sampling_rate,
            mel_bins=audio_vae.config.mel_bins,
            mel_hop_length=pipe.audio_hop_length,
            n_fft=1024,  # todo get this from vae if we can, I couldnt find it.
        ).to(self.device_torch, dtype=torch.float32)

        self.print_and_status_update("Model Loaded")

    @torch.no_grad()
    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.pipeline.vae.device == torch.device("cpu"):
            self.pipeline.vae.to(device)
        self.pipeline.vae.eval()
        self.pipeline.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]

        # Normalize shapes
        norm_images = []
        for image in image_list:
            if image.ndim == 3:
                # (C, H, W) -> (C, 1, H, W)
                norm_images.append(image.unsqueeze(1))
            elif image.ndim == 4:
                # (T, C, H, W) -> (C, T, H, W)
                norm_images.append(image.permute(1, 0, 2, 3))
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

        # Stack to (B, C, T, H, W)
        images = torch.stack(norm_images)

        latents = self.pipeline.vae.encode(images).latent_dist.mode()

        # Normalize latents across the channel dimension [B, C, F, H, W]
        scaling_factor = 1.0
        latents_mean = self.pipeline.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = self.pipeline.vae.latents_std.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * scaling_factor / latents_std

        return latents.to(device, dtype=dtype)

    def get_generation_pipeline(self):
        scheduler = LTX2Model.get_train_scheduler()

        pipeline: LTX2Pipeline = LTX2Pipeline(
            scheduler=scheduler,
            vae=unwrap_model(self.pipeline.vae),
            audio_vae=unwrap_model(self.pipeline.audio_vae),
            text_encoder=None,
            tokenizer=unwrap_model(self.pipeline.tokenizer),
            connectors=unwrap_model(self.pipeline.connectors),
            transformer=None,
            vocoder=unwrap_model(self.pipeline.vocoder),
        )
        pipeline.transformer = unwrap_model(self.model)
        pipeline.text_encoder = unwrap_model(self.text_encoder[0])

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: LTX2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # handle control image
        if gen_config.ctrl_img is not None:
            # switch to image to video pipeline
            pipeline = LTX2ImageToVideoPipeline(
                scheduler=pipeline.scheduler,
                vae=pipeline.vae,
                audio_vae=pipeline.audio_vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                connectors=pipeline.connectors,
                transformer=pipeline.transformer,
                vocoder=pipeline.vocoder,
            )

        is_video = gen_config.num_frames > 1
        # override the generate single image to handle video + audio generation
        if is_video:
            gen_config._orig_save_image_function = gen_config.save_image
            gen_config.save_image = partial(new_save_image_function, gen_config)
            gen_config.log_image = partial(blank_log_image_function, gen_config)
            # set output extension to mp4
            gen_config.output_ext = "mp4"

        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        pipeline = pipeline.to(self.device_torch)

        # make sure dimensions are valid
        bd = self.get_bucket_divisibility()
        gen_config.height = (gen_config.height // bd) * bd
        gen_config.width = (gen_config.width // bd) * bd

        # handle control image
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img).convert("RGB")
            # resize the control image
            control_img = control_img.resize(
                (gen_config.width, gen_config.height), Image.LANCZOS
            )
            # add the control image to the extra dict
            extra["image"] = control_img

        # frames must be divisible by 8 then + 1. so 1, 9, 17, 25, etc.
        if gen_config.num_frames != 1:
            if (gen_config.num_frames - 1) % 8 != 0:
                gen_config.num_frames = ((gen_config.num_frames - 1) // 8) * 8 + 1

        if self.low_vram:
            # set vae to tile decode
            pipeline.vae.enable_tiling(
                tile_sample_min_height=256,
                tile_sample_min_width=256,
                tile_sample_min_num_frames=8,
                tile_sample_stride_height=224,
                tile_sample_stride_width=224,
                tile_sample_stride_num_frames=4,
            )
        
        # We only encode and store the minimum prompt tokens, but need them padded to 1024 for LTX2
        conditional_embeds = self.pad_embeds(conditional_embeds)
        unconditional_embeds = self.pad_embeds(unconditional_embeds)

        video, audio = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            prompt_attention_mask=conditional_embeds.attention_mask.to(
                self.device_torch
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            negative_prompt_attention_mask=unconditional_embeds.attention_mask.to(
                self.device_torch
            ),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="np" if is_video else "pil",
            **extra,
        )
        if self.low_vram:
            # Restore no tiling
            pipeline.vae.use_tiling = False

        if is_video:
            # redurn as a dict, we will handle it with an override function
            video = (video * 255).round().astype("uint8")
            video = torch.from_numpy(video)
            return {
                "video": video[0],
                "fps": gen_config.fps,
                "audio": audio[0].float().cpu(),
                "audio_sample_rate": pipeline.vocoder.config.output_sampling_rate,  # should be 24000
                "output_path": None,
            }
        else:
            # shape = [1, frames, channels, height, width]
            # make sure this is right
            video = video[0]  # list of pil images
            audio = audio[0]  # tensor
            if gen_config.num_frames > 1:
                return video  # return the frames.
            else:
                # get just the first image
                img = video[0]
            return img

    def encode_audio(self, audio_data_list):
        # audio_date_list is a list of {"waveform": waveform[C, L], "sample_rate": int(sample_rate)}
        if self.pipeline.audio_vae.device == torch.device("cpu"):
            self.pipeline.audio_vae.to(self.device_torch)

        output_tensor = None
        audio_num_frames = None

        # do them seperatly for now
        for audio_data in audio_data_list:
            waveform = audio_data["waveform"].to(
                device=self.device_torch, dtype=torch.float32
            )
            sample_rate = audio_data["sample_rate"]

            # Add batch dimension if needed: [channels, samples] -> [batch, channels, samples]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            if waveform.shape[1] == 1:
                # make sure it is stereo
                waveform = waveform.repeat(1, 2, 1)

            # Convert waveform to mel spectrogram using AudioProcessor
            mel_spectrogram = self.audio_processor.waveform_to_mel(
                waveform, waveform_sample_rate=sample_rate
            )
            mel_spectrogram = mel_spectrogram.to(dtype=self.torch_dtype)

            # Encode mel spectrogram to latents
            latents = self.pipeline.audio_vae.encode(
                mel_spectrogram.to(self.device_torch, dtype=self.torch_dtype)
            ).latent_dist.mode()

            if audio_num_frames is None:
                audio_num_frames = latents.shape[2]  # (latents is [B, C, T, F])

            packed_latents = self.pipeline._pack_audio_latents(
                latents,
                # patch_size=self.pipeline.transformer.config.audio_patch_size,
                # patch_size_t=self.pipeline.transformer.config.audio_patch_size_t,
            )  # [B, L, C * M]
            if output_tensor is None:
                output_tensor = packed_latents
            else:
                output_tensor = torch.cat([output_tensor, packed_latents], dim=0)

        # normalize latents, opposite of (latents * latents_std) + latents_mean
        latents_mean = self.pipeline.audio_vae.latents_mean
        latents_std = self.pipeline.audio_vae.latents_std
        output_tensor = (output_tensor - latents_mean) / latents_std
        return output_tensor
    
    def pad_embeds(self, embeds: PromptEmbeds):
        # ltx-2 connector requires 1024 tokens for good results. Any smaller and it degrades.
        target_length = 1024
        current_length = embeds.text_embeds.shape[1]
        if current_length < target_length:
            pad_length = target_length - current_length
            pad_tensor = torch.zeros(
                (embeds.text_embeds.shape[0], pad_length, embeds.text_embeds.shape[2]),
                device=embeds.text_embeds.device,
                dtype=embeds.text_embeds.dtype,
            )
            embeds.text_embeds = torch.cat([pad_tensor, embeds.text_embeds], dim=1)
            if embeds.attention_mask is not None:
                pad_mask = torch.zeros(
                    (embeds.attention_mask.shape[0], pad_length),
                    device=embeds.attention_mask.device,
                    dtype=embeds.attention_mask.dtype,
                )
                embeds.attention_mask = torch.cat(
                    [pad_mask, embeds.attention_mask], dim=1
                )
        return embeds

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        with torch.no_grad():
            if self.model.device == torch.device("cpu"):
                self.model.to(self.device_torch)
                
            # We only encode and store the minimum prompt tokens, but need them padded to 1024 for LTX2
            text_embeddings = self.pad_embeds(text_embeddings)

            batch_size, C, latent_num_frames, latent_height, latent_width = (
                latent_model_input.shape
            )

            video_timestep = timestep.clone()

            # i2v from first frame
            if batch.dataset_config.do_i2v and batch.dataset_config.num_frames > 1:
                # check to see if we had it cached
                if batch.first_frame_latents is not None:
                    init_latents = batch.first_frame_latents.to(
                        self.device_torch, dtype=self.torch_dtype
                    )
                else:
                    # extract the first frame and encode it
                    # videos come in (bs, num_frames, channels, height, width)
                    # images come in (bs, channels, height, width)
                    frames = batch.tensor
                    if len(frames.shape) == 4:
                        first_frames = frames
                    elif len(frames.shape) == 5:
                        first_frames = frames[:, 0]
                    else:
                        raise ValueError(f"Unknown frame shape {frames.shape}")
                    # first frame doesnt have time dim, add it back
                    init_latents = self.encode_images(
                        first_frames, device=self.device_torch, dtype=self.torch_dtype
                    )

                # expand the latents to match video frames
                init_latents = init_latents.repeat(1, 1, latent_num_frames, 1, 1)
                mask_shape = (
                    batch_size,
                    1,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                )
                # First condition is image latents and those should be kept clean.
                conditioning_mask = torch.zeros(
                    mask_shape, device=self.device_torch, dtype=self.torch_dtype
                )
                conditioning_mask[:, :, 0] = 1.0

                # use conditioning mask to replace latents
                latent_model_input = (
                    latent_model_input * (1 - conditioning_mask)
                    + init_latents * conditioning_mask
                )

                # set video timestep
                video_timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)

            # todo get this somehow
            frame_rate = 24
            # check frame dimension
            # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
            packed_latents = self.pipeline._pack_latents(
                latent_model_input,
                patch_size=self.pipeline.transformer_spatial_patch_size,
                patch_size_t=self.pipeline.transformer_temporal_patch_size,
            )

            if batch.audio_latents is not None or batch.audio_tensor is not None:
                if batch.audio_latents is not None:
                    # we have audio latents cached
                    raw_audio_latents = batch.audio_latents.to(
                        self.device_torch, dtype=self.torch_dtype
                    )
                else:
                    # we have audio waveforms to encode
                    # use audio from the batch if available
                    raw_audio_latents = self.encode_audio(batch.audio_data)

                audio_num_frames = raw_audio_latents.shape[1]
                # add the audio targets to the batch for loss calculation later
                audio_noise = torch.randn_like(raw_audio_latents)
                batch.audio_target = (audio_noise - raw_audio_latents).detach()
                audio_latents = self.add_noise(
                    raw_audio_latents,
                    audio_noise,
                    timestep,
                ).to(self.device_torch, dtype=self.torch_dtype)
            else:
                # no audio
                num_mel_bins = self.pipeline.audio_vae.config.mel_bins
                # latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
                num_channels_latents_audio = (
                    self.pipeline.audio_vae.config.latent_channels
                )
                audio_latents, audio_num_frames = self.pipeline.prepare_audio_latents(
                    batch_size,
                    num_channels_latents=num_channels_latents_audio,
                    num_mel_bins=num_mel_bins,
                    num_frames=batch.dataset_config.num_frames,
                    frame_rate=frame_rate,
                    sampling_rate=self.pipeline.audio_sampling_rate,
                    hop_length=self.pipeline.audio_hop_length,
                    dtype=torch.float32,
                    device=self.transformer.device,
                    generator=None,
                    latents=None,
                )

            if self.pipeline.connectors.device != self.transformer.device:
                self.pipeline.connectors.to(self.transformer.device)

            # TODO this is how diffusers does this on inference, not sure I understand why, check this
            additive_attention_mask = (
                1 - text_embeddings.attention_mask.to(self.transformer.dtype)
            ) * -1000000.0
            (
                connector_prompt_embeds,
                connector_audio_prompt_embeds,
                connector_attention_mask,
            ) = self.pipeline.connectors(
                text_embeddings.text_embeds, additive_attention_mask, additive_mask=True
            )

            # compute video and audio positional ids
            video_coords = self.transformer.rope.prepare_video_coords(
                packed_latents.shape[0],
                latent_num_frames,
                latent_height,
                latent_width,
                packed_latents.device,
                fps=frame_rate,
            )
            audio_coords = self.transformer.audio_rope.prepare_audio_coords(
                audio_latents.shape[0], audio_num_frames, audio_latents.device
            )

        noise_pred_video, noise_pred_audio = self.transformer(
            hidden_states=packed_latents,
            audio_hidden_states=audio_latents.to(self.transformer.dtype),
            encoder_hidden_states=connector_prompt_embeds,
            audio_encoder_hidden_states=connector_audio_prompt_embeds,
            timestep=video_timestep,
            audio_timestep=timestep,
            encoder_attention_mask=connector_attention_mask,
            audio_encoder_attention_mask=connector_attention_mask,
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            fps=frame_rate,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            # rope_interpolation_scale=rope_interpolation_scale,
            attention_kwargs=None,
            return_dict=False,
        )

        # add audio latent to batch if we had audio
        if batch.audio_target is not None:
            batch.audio_pred = noise_pred_audio

        unpacked_output = self.pipeline._unpack_latents(
            latents=noise_pred_video,
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            patch_size=self.pipeline.transformer_spatial_patch_size,
            patch_size_t=self.pipeline.transformer_temporal_patch_size,
        )

        return unpacked_output

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        device = self.device_torch
        scale_factor = 8
        batch_size = len(prompt)
        # Gemma expects left padding for chat-style prompts
        self.tokenizer[0].padding_side = "left"
        if self.tokenizer[0].pad_token is None:
            self.tokenizer[0].pad_token = self.tokenizer[0].eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer[0](
            prompt,
            # padding="max_length",
            padding="longest",
            max_length=1024,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        
        text_input_ids = text_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        text_encoder_outputs = self.text_encoder[0](
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        sequence_lengths = prompt_attention_mask.sum(dim=-1)

        prompt_embeds = self.pipeline._pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=device,
            padding_side=self.tokenizer[0].padding_side,
            scale_factor=scale_factor,
        )
        prompt_embeds = prompt_embeds.to(dtype=self.torch_dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * 1, seq_len, -1
        )

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(1, 1)
        
        pe = PromptEmbeds([prompt_embeds, None])
        pe.attention_mask = prompt_attention_mask
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: LTX2VideoTransformer3DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "ltx2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        new_sd = convert_lora_diffusers_to_original(new_sd)
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        state_dict = convert_lora_original_to_diffusers(state_dict)
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
