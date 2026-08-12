from functools import partial
import json
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
from toolkit.paths import MODELS_PATH
from safetensors.torch import load_file
from PIL import Image
import huggingface_hub

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
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE
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
base_te_path = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"

HF_TOKEN = os.getenv("HF_TOKEN", None)


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

    @property
    def config(self):
        return self.vae.config

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
    ltx_version = "2.0"
    ltx_te_path = None

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

        # loss mask for i2v conditioning (1 = train, 0 = conditioned token), set per step in get_noise_prediction
        self._i2v_loss_mask = None

        # invalidate older caches
        self.latent_space_version = f"{self.arch}_v2"

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

        if not os.path.exists(model_path) and model_path.endswith(".safetensors"):
            # download the model from the Hugging Face Hub if it is not a local path
            splits = model_path.split("/")
            if len(splits) < 3:
                raise ValueError(
                    f"Invalid model path: {model_path}. Must be in the format 'repo_id/repo/filename.safetensors' or 'repo_id/repo/subfolder/filename.safetensors' to download from the Hugging Face Hub."
                )
            rel_path = "/".join(splits[2:])
            # use the file from the models folder if it is already there
            local_candidates = [
                os.path.join(MODELS_PATH, rel_path),
                os.path.join(MODELS_PATH, splits[-1]),
            ]
            for candidate in local_candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            else:
                # download the model from the hub into the models folder
                model_path = huggingface_hub.hf_hub_download(
                    repo_id="/".join(splits[:2]),
                    filename=rel_path,
                    token=HF_TOKEN,
                    local_dir=MODELS_PATH,
                )

        # if we have a safetensors file it is a mono checkpoint
        if os.path.exists(model_path) and model_path.endswith(".safetensors"):
            combined_state_dict = load_file(model_path)
            combined_state_dict = dequantize_state_dict(combined_state_dict)

        if combined_state_dict is not None:
            original_dit_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, dit_prefix
            )
            transformer = convert_ltx2_transformer(
                original_dit_ckpt, version=self.ltx_version
            )
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
            tokenizer = GemmaTokenizerFast.from_pretrained(base_te_path)

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
        elif self.model_config.te_name_or_path is not None:
            # a repo or folder
            tokenizer = GemmaTokenizerFast.from_pretrained(
                self.model_config.te_name_or_path
            )
            text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_config.te_name_or_path, dtype=dtype
            )
        elif self.ltx_te_path is not None:
            # pull from model specific te
            tokenizer = GemmaTokenizerFast.from_pretrained(self.ltx_te_path)
            text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                self.ltx_te_path, dtype=dtype
            )
        else:
            # using combo hf repo
            tokenizer = GemmaTokenizerFast.from_pretrained(
                self.model_config.name_or_path, subfolder="tokenizer"
            )
            text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_config.name_or_path, subfolder="text_encoder", dtype=dtype
            )

        # remove the vision tower
        text_encoder.model.vision_tower = None
        flush()
        
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
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

        self.print_and_status_update("Loading VAEs and other components")
        if combined_state_dict is not None:
            original_vae_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, vae_prefix
            )
            vae = convert_ltx2_video_vae(
                original_vae_ckpt, version=self.ltx_version
            ).to(dtype)
            del original_vae_ckpt
            original_audio_vae_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, audio_vae_prefix
            )
            audio_vae = convert_ltx2_audio_vae(
                original_audio_vae_ckpt, version=self.ltx_version
            ).to(dtype)
            del original_audio_vae_ckpt
            original_connectors_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, dit_prefix
            )
            connectors = convert_ltx2_connectors(
                original_connectors_ckpt, version=self.ltx_version
            ).to(dtype)
            del original_connectors_ckpt
            original_vocoder_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, vocoder_prefix
            )
            vocoder = convert_ltx2_vocoder(
                original_vocoder_ckpt, version=self.ltx_version
            ).to(dtype)
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

            vocoder_cls = LTX2Vocoder
            if self.ltx_version in ("2.3", "2.5"):
                vocoder_cls = LTX2VocoderWithBWE

            vocoder = vocoder_cls.from_pretrained(
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

        if self.model_config.low_vram:
            self.pipeline.vae.tile_sample_min_num_frames = 64
            self.pipeline.vae.tile_sample_stride_num_frames = 16
            # they check the wrong flat on encode currently so set both to future proof
            self.pipeline.vae.use_framewise_decoding = True
            self.pipeline.vae.use_framewise_encoding = True

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

        latents = self.pipeline.vae.encode(images).latent_dist.sample()

        # Normalize latents across the channel dimension [B, C, F, H, W]
        scaling_factor = 1.0
        latents_mean = self.pipeline.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = self.pipeline.vae.latents_std.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * scaling_factor / latents_std

        if self.model_config.low_vram:
            self.pipeline.vae.use_framewise_decoding = False
            self.pipeline.vae.use_framewise_encoding = False

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
            # pipeline.vae.enable_tiling(
            #     tile_sample_min_height=256,
            #     tile_sample_min_width=256,
            #     tile_sample_min_num_frames=8,
            #     tile_sample_stride_height=224,
            #     tile_sample_stride_width=224,
            #     tile_sample_stride_num_frames=4,
            # )
            self.pipeline.vae.tile_sample_min_num_frames = 16
            self.pipeline.vae.tile_sample_stride_num_frames = 8
            self.pipeline.vae.use_framewise_decoding = True

        # We only encode and store the minimum prompt tokens, but need them padded to 1024 for LTX2
        conditional_embeds = self.pad_embeds(conditional_embeds)
        unconditional_embeds = self.pad_embeds(unconditional_embeds)

        if self.ltx_version in ("2.3", "2.5"):
            extra["stg_scale"] = 1.0
            extra["modality_scale"] = 3.0
            extra["guidance_rescale"] = 0.7
            extra["audio_guidance_scale"] = 7.0
            extra["audio_stg_scale"] = 1.0
            extra["audio_modality_scale"] = 3.0
            extra["audio_guidance_rescale"] = 0.7
            extra["spatio_temporal_guidance_blocks"] = [28]
            extra["use_cross_timestep"] = (
                True  # they dont set this in some examples in diffusers, but I believe it should always be true for 2.3
            )

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
            # pipeline.vae.use_tiling = False
            self.pipeline.vae.use_framewise_decoding = False

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
            ).latent_dist.sample()

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
        # a grad-enabled prediction is the primary (loss carrying) one unless
        # the trainer declared a secondary slot on the batch (prior /
        # guidance-unconditional / preservation passes). Trainers that make
        # several grad predictions per step (e.g. turbo rollouts) get one
        # primary per prediction, last writer wins.
        is_primary_pred = (
            torch.is_grad_enabled()
            and batch is not None
            and batch.audio_pred_slot is None
        )
        with torch.no_grad():
            if self.model.device == torch.device("cpu"):
                self.model.to(self.device_torch)

            # We only encode and store the minimum prompt tokens, but need them padded to 1024 for LTX2
            text_embeddings = self.pad_embeds(text_embeddings)

            batch_size, C, latent_num_frames, latent_height, latent_width = (
                latent_model_input.shape
            )

            video_timestep = timestep.clone()
            self._i2v_loss_mask = None

            # i2v from first frame
            if batch.dataset_config.do_i2v and batch.num_frames > 1:
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
                    init_latents * conditioning_mask
                    + latent_model_input * (1 - conditioning_mask)
                )

                # conditioned tokens are clean with timestep 0 and their prediction is
                # discarded at inference, so they must not contribute to the loss
                self._i2v_loss_mask = 1.0 - conditioning_mask

                packed_conditioning_mask = self.pipeline._pack_latents(
                    conditioning_mask,
                    patch_size=self.pipeline.transformer_spatial_patch_size,
                    patch_size_t=self.pipeline.transformer_temporal_patch_size,
                ).squeeze(-1)

                # set video timestep
                video_timestep = timestep.unsqueeze(-1) * (1 - packed_conditioning_mask)

            frame_rate = batch.dataset_config.fps
            # check frame dimension
            # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
            packed_latents = self.pipeline._pack_latents(
                latent_model_input,
                patch_size=self.pipeline.transformer_spatial_patch_size,
                patch_size_t=self.pipeline.transformer_temporal_patch_size,
            )

            # audio only trains for video batches from datasets that asked for
            # it. Cached latents can carry audio after do_audio was turned off,
            # and image (single frame) batches must never pick up a soundtrack.
            do_audio = (
                batch.dataset_config is not None
                and batch.dataset_config.do_audio
                and getattr(batch, "num_frames", 1) > 1
            )
            if do_audio and (
                batch.audio_latents is not None or batch.audio_tensor is not None
            ):
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
                # the audio noise is drawn once per step and shared by every
                # pass (prior, primary, cfg/guidance, preservation) so they all
                # see the same soundtrack and the stored target keeps matching
                if (
                    batch.audio_noise is not None
                    and batch.audio_noise.shape == raw_audio_latents.shape
                ):
                    audio_noise = batch.audio_noise.to(
                        raw_audio_latents.device, dtype=raw_audio_latents.dtype
                    )
                else:
                    audio_noise = torch.randn_like(raw_audio_latents)
                    batch.audio_noise = audio_noise
                if batch.audio_target is None:
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
                duration_s = batch.num_frames / frame_rate
                audio_latents_per_second = (
                    self.pipeline.audio_sampling_rate
                    / self.pipeline.audio_hop_length
                    / float(self.pipeline.audio_vae_temporal_compression_ratio)
                )
                audio_num_frames = round(duration_s * audio_latents_per_second)
                audio_latents = self.pipeline.prepare_audio_latents(
                    batch_size,
                    num_channels_latents=num_channels_latents_audio,
                    audio_latent_length=audio_num_frames,
                    num_mel_bins=num_mel_bins,
                    noise_scale=0.0,
                    dtype=torch.float32,
                    device=self.transformer.device,
                    generator=None,
                    latents=None,
                )

            if self.pipeline.connectors.device != self.transformer.device:
                self.pipeline.connectors.to(self.transformer.device)

            # Padding side for default Gemma3-12B text encoder
            tokenizer_padding_side = "left"
            if getattr(self, "tokenizer", None) is not None:
                tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")
            (
                connector_prompt_embeds,
                connector_audio_prompt_embeds,
                connector_attention_mask,
            ) = self.pipeline.connectors(
                text_embeddings.text_embeds,
                text_embeddings.attention_mask.to(self.transformer.dtype),
                padding_side=tokenizer_padding_side,
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

        # use_cross_timestep - Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when
        # calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior;
        # `False` is the legacy LTX-2.0 behavior.
        use_cross_timestep = self.ltx_version in ("2.3", "2.5")

        noise_pred_video, noise_pred_audio = self.transformer(
            hidden_states=packed_latents,
            audio_hidden_states=audio_latents.to(self.transformer.dtype),
            encoder_hidden_states=connector_prompt_embeds,
            audio_encoder_hidden_states=connector_audio_prompt_embeds,
            timestep=video_timestep,
            sigma=timestep,  # Used by LTX-2.3
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
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep=use_cross_timestep,
            attention_kwargs=None,
            return_dict=False,
        )

        # add audio latent to batch if we had audio
        if batch.audio_target is not None:
            if is_primary_pred:
                batch.audio_pred = noise_pred_audio
            else:
                batch.set_secondary_audio_pred(noise_pred_audio)

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
        prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(
            dtype=self.torch_dtype
        )  # Pack to 3D

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size * 1, seq_len, -1)

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

    def scale_loss(self, loss):
        # zero out the loss on i2v conditioned tokens, renormalized so the loss
        # magnitude matches unconditioned batches (masked mean)
        if self._i2v_loss_mask is not None:
            loss_mask = self._i2v_loss_mask.to(loss.device, dtype=loss.dtype)
            loss = loss * loss_mask / loss_mask.mean().clamp(min=1e-8)
            self._i2v_loss_mask = None
        return loss

    def get_base_model_version(self):
        return "ltx2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        new_sd = convert_lora_diffusers_to_original(new_sd, version=self.ltx_version)
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        state_dict = convert_lora_original_to_diffusers(
            state_dict, version=self.ltx_version
        )
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd


class LTX23Model(LTX2Model):
    arch = "ltx2.3"
    ltx_version = "2.3"
    ltx_te_path = base_te_path


# LTX-2.5 ships as ComfyUI-style split files (no diffusers folders, no mono
# checkpoint). Files are used in place when present under MODELS_PATH and
# downloaded to exactly these locations only when missing, so the models
# folder stays shareable with a ComfyUI install. The int8 ConvRot files are
# the defaults; bf16 variants stay selectable via model_kwargs overrides.
COMFY_LTX25_REPO = "Lightricks/LTX-2.5"
COMFY_LTX25_FILES = {
    "dit": "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    # the "-conv-" file is the classic conv VAE; the default 2.5 vae file is a
    # new diffusion-decoder VAE that diffusers has no class for
    "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    # bundles the BWE vocoder alongside the audio VAE
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
}


class LTX25Model(LTX2Model):
    arch = "ltx2.5"
    ltx_version = "2.5"
    ltx_te_path = None

    # ------------------------------------------------------------------
    # ComfyUI-style file resolution (mirrors MinimaxH3Model)
    # ------------------------------------------------------------------
    @staticmethod
    def _find_file_recursive(root_dir: str, filename: str) -> Optional[str]:
        if not os.path.isdir(root_dir):
            return None
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def _resolve_comfy_file(self, component: str) -> str:
        """Find a weight file at its local location, or download it there
        when (and only when) it is missing.

        Search order: model_kwargs override, the repo-relative path under
        MODELS_PATH (diffusion_models/, text_encoders/, vae/), the bare
        filename at the root, any subfolder of the component's category
        folder (recursive), then the hub — downloaded to the repo-relative
        path under MODELS_PATH.
        """
        override = self.model_config.model_kwargs.get(f"{component}_path", None)
        if override is not None:
            if not os.path.exists(override):
                raise FileNotFoundError(
                    f"model_kwargs.{component}_path does not exist: {override}"
                )
            return override

        rel_path = COMFY_LTX25_FILES[component]
        filename = os.path.basename(rel_path)
        category = os.path.dirname(rel_path)
        for rel in (rel_path, filename):
            candidate = os.path.join(MODELS_PATH, rel)
            if os.path.exists(candidate):
                return candidate
        found = self._find_file_recursive(os.path.join(MODELS_PATH, category), filename)
        if found is not None:
            return found

        repo_id = COMFY_LTX25_REPO
        name_or_path = self.model_config.name_or_path
        if (
            name_or_path
            and not os.path.exists(name_or_path)
            and not name_or_path.endswith(".safetensors")
            and "/" in name_or_path
        ):
            repo_id = name_or_path
        self.print_and_status_update(
            f"Downloading {rel_path} from {repo_id} into {MODELS_PATH}"
        )
        return huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=rel_path, token=HF_TOKEN, local_dir=MODELS_PATH
        )

    def _resolve_named_file(self, path: str, component: str) -> str:
        """Resolve an explicit .safetensors path: local file, models-folder
        file, or an 'org/repo/path/file.safetensors' hub path (downloaded
        into the models folder)."""
        if os.path.exists(path):
            return path
        splits = path.split("/")
        if len(splits) < 3:
            raise ValueError(
                f"Invalid {component} path: {path}. Must be a local file or "
                "'repo_id/repo/filename.safetensors' to download from the Hugging Face Hub."
            )
        rel_path = "/".join(splits[2:])
        for candidate in (
            os.path.join(MODELS_PATH, rel_path),
            os.path.join(MODELS_PATH, splits[-1]),
        ):
            if os.path.exists(candidate):
                return candidate
        return huggingface_hub.hf_hub_download(
            repo_id="/".join(splits[:2]),
            filename=rel_path,
            token=HF_TOKEN,
            local_dir=MODELS_PATH,
        )

    def _resolve_dit_path(self) -> str:
        name_or_path = self.model_config.name_or_path
        if name_or_path and name_or_path.endswith(".safetensors"):
            return self._resolve_named_file(name_or_path, "transformer")
        return self._resolve_comfy_file("dit")

    def _resolve_te_path(self) -> str:
        te_name_or_path = self.model_config.te_name_or_path
        if te_name_or_path is not None:
            return self._resolve_named_file(te_name_or_path, "text encoder")
        return self._resolve_comfy_file("text_encoder")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_quantized_module(self, module, state_dict, name: str) -> int:
        """Attach pre-quantized (int8 ConvRot) linears onto the toolkit's
        quantization backends and load the rest of the (meta-built) module
        from the state dict. Works unchanged for bf16 checkpoints, where no
        quant markers exist and everything strict-loads."""
        from toolkit.util.comfy_quant_import import import_comfy_quantized_layers
        from toolkit.util.ostris_quant import OstrisLinear

        state_dict, num_quantized = import_comfy_quantized_layers(
            module, state_dict, orig_dtype=self.torch_dtype
        )
        if num_quantized:
            self.print_and_status_update(
                f" - attached {num_quantized} pre-quantized ConvRot layers to {name}"
            )
        result = module.load_state_dict(state_dict, assign=True, strict=False)
        # quantized linears hold their weight as backend buffers and had their
        # bias assigned by the importer, so both report as "missing" here
        quantized_param_keys = set()
        for mod_name, m in module.named_modules():
            if isinstance(m, OstrisLinear):
                quantized_param_keys.add(f"{mod_name}.weight")
                if m.bias is not None:
                    quantized_param_keys.add(f"{mod_name}.bias")
        bad_missing = [k for k in result.missing_keys if k not in quantized_param_keys]
        if bad_missing or result.unexpected_keys:
            raise ValueError(
                f"LTX-2.5 {name} load mismatch: missing {bad_missing[:8]}, "
                f"unexpected {result.unexpected_keys[:8]}"
            )
        # nothing may be left on the meta device (e.g. a bias the importer
        # should have filled)
        leftover_meta = [
            param_name
            for param_name, p in module.named_parameters()
            if p.is_meta
        ]
        if leftover_meta:
            raise ValueError(
                f"LTX-2.5 {name} load left meta parameters: {leftover_meta[:8]}"
            )
        return num_quantized

    def _load_gemma4_text_encoder(self, te_path: str, te_state_dict: dict):
        """Build the Gemma-4 12B text stack from the single comfy file. Only
        the text decoder is loaded — the unified checkpoint's vision/audio
        tower pieces and the connector projections are used elsewhere or
        dropped, matching how the Gemma-3 vision tower was discarded."""
        from safetensors import safe_open
        from transformers import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

        with safe_open(te_path, framework="pt") as f:
            metadata = f.metadata() or {}
        gemma_config = json.loads(metadata["gemma_config"])
        text_config = {
            k: v for k, v in gemma_config["text_config"].items() if k != "dtype"
        }

        with init_empty_weights():
            text_encoder = Gemma4TextModel(Gemma4TextConfig(**text_config))

        def strip_model_prefix(key: str) -> str:
            return key[len("model.") :] if key.startswith("model.") else key

        te_sd = {
            strip_model_prefix(k): v
            for k, v in te_state_dict.items()
            if k.startswith("model.")
        }
        num_quantized = self._load_quantized_module(text_encoder, te_sd, "text encoder")
        return text_encoder, num_quantized

    def _load_gemma4_tokenizer(self, te_path: str, te_state_dict: dict):
        """The comfy file embeds the tokenizer and its configs as uint8
        tensors; extract them next to the file once and load from there."""
        from transformers import AutoTokenizer

        assets_dir = os.path.splitext(te_path)[0] + "_hf_assets"
        assets = {
            "tokenizer.json": "tokenizer_json",
            "tokenizer_config.json": "hf_asset__tokenizer_config.json",
            "chat_template.jinja": "hf_asset__chat_template.jinja",
        }
        os.makedirs(assets_dir, exist_ok=True)
        for filename, tensor_key in assets.items():
            out_path = os.path.join(assets_dir, filename)
            blob = te_state_dict.get(tensor_key, None)
            if blob is None or os.path.exists(out_path):
                continue
            with open(out_path, "wb") as f:
                f.write(bytes(blob.cpu().numpy().tobytes()))
        # the embedded tokenizer.json has an empty post-processor (ComfyUI
        # prepends BOS in its own wrapper); restore the standard Gemma
        # behavior so blank prompts still yield a token
        return AutoTokenizer.from_pretrained(assets_dir, add_bos_token=True)

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading LTX-2.5 model")

        # ---- transformer + embedding connectors (one comfy file) ----
        dit_path = self._resolve_dit_path()
        te_path = self._resolve_te_path()
        self.print_and_status_update(
            f"Loading transformer from {os.path.basename(dit_path)}"
        )
        combined = load_file(dit_path)
        dit_sd = get_model_state_dict_from_combined_ckpt(combined, dit_prefix)
        del combined

        # the per-modality text projections ride in the text encoder file but
        # belong to the connectors module
        te_state_dict = load_file(te_path)
        for key in list(te_state_dict.keys()):
            if key.startswith("text_embedding_projection."):
                dit_sd[key] = te_state_dict.pop(key)

        transformer, transformer_sd = convert_ltx2_transformer(
            dit_sd, version=self.ltx_version, load=False
        )
        num_quantized_dit = self._load_quantized_module(
            transformer, transformer_sd, "transformer"
        )
        del transformer_sd
        if num_quantized_dit == 0:
            transformer = transformer.to(dtype)
        flush()

        if self.model_config.quantize:
            if num_quantized_dit:
                self.print_and_status_update(
                    "Transformer is pre-quantized (ConvRot); skipping quantize"
                )
            else:
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

        self.print_and_status_update("Loading connectors")
        connectors, connectors_sd = convert_ltx2_connectors(
            dit_sd, version=self.ltx_version, load=False
        )
        num_quantized_connectors = self._load_quantized_module(
            connectors, connectors_sd, "connectors"
        )
        del connectors_sd, dit_sd
        if num_quantized_connectors == 0:
            connectors = connectors.to(dtype)
        flush()

        # ---- text encoder (Gemma-4 12B, single comfy file) ----
        self.print_and_status_update("Loading text encoder")
        tokenizer = self._load_gemma4_tokenizer(te_path, te_state_dict)
        text_encoder, num_quantized_te = self._load_gemma4_text_encoder(
            te_path, te_state_dict
        )
        del te_state_dict
        flush()

        if self.model_config.quantize_te:
            if num_quantized_te:
                self.print_and_status_update(
                    "Text encoder is pre-quantized (ConvRot); skipping quantize"
                )
            else:
                self.print_and_status_update("Quantizing Text Encoder")
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                freeze(text_encoder)
                flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            # layer_scalar is a bare tensor buffer on each decoder layer; the
            # manager never enumerates it, so it must ride along explicitly
            ignore_modules = [text_encoder.embed_tokens]
            for layer in text_encoder.layers:
                ignore_modules.append(layer.layer_scalar)
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
                ignore_modules=ignore_modules,
            )

        text_encoder.to(self.device_torch)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        flush()

        # ---- VAEs + vocoder ----
        self.print_and_status_update("Loading VAEs and other components")
        video_vae_path = self._resolve_comfy_file("video_vae")
        vae = convert_ltx2_video_vae(
            load_file(video_vae_path), version=self.ltx_version
        ).to(dtype)
        flush()

        audio_vae_path = self._resolve_comfy_file("audio_vae")
        audio_combined = load_file(audio_vae_path)
        audio_sd = get_model_state_dict_from_combined_ckpt(
            audio_combined, audio_vae_prefix
        )
        audio_vae = convert_ltx2_audio_vae(audio_sd, version=self.ltx_version).to(dtype)
        vocoder_sd = get_model_state_dict_from_combined_ckpt(
            audio_combined, vocoder_prefix
        )
        vocoder = convert_ltx2_vocoder(vocoder_sd, version=self.ltx_version).to(dtype)
        del audio_combined, audio_sd, vocoder_sd
        flush()

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
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")
        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()

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
