from functools import partial
import os
from typing import List, Optional

import huggingface_hub
import torch
from transformers import Gemma3Config
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
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

try:
    from diffusers import LTX2Pipeline
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
ltx_model_version = "2.0"


def new_save_image_function(
    self: GenerateImageConfig,
    image,  # will contain a dict that can be dumped ditectly into encode_video, just add output_path to it.
    count: int = 0,
    max_count: int = 0,
    **kwargs,
):
    # this replaces gen image config save image function so we can save the video with sound from ltx2
    image["output_path"] = self.get_image_path(count, max_count)
    encode_video(**image)


def blank_log_image_function(self, *args, **kwargs):
    # todo handle wandb logging of videos with audio
    return


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

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # 16 for the VAE, 2 for patch size

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
            transformer = convert_ltx2_transformer(original_dit_ckpt, ltx_model_version)
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
        if self.model_config.te_name_or_path is not None:
            tokenizer = GemmaTokenizerFast.from_pretrained("unsloth/gemma-3-12b-it-qat")
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
            # the model uses https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
            # which is trained with quantized aware training so it works at quant q4_0 directly
            # a prequantized gguf can be found here: unsloth/gemma-3-12b-it-qat-GGUF
            tokenizer = GemmaTokenizerFast.from_pretrained(
                base_model_path, subfolder="tokenizer"
            )
            text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                base_model_path, subfolder="text_encoder", dtype=dtype
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
            vae = convert_ltx2_video_vae(
                original_vae_ckpt, version=ltx_model_version
            ).to(dtype)
            del original_vae_ckpt
            original_audio_vae_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, audio_vae_prefix
            )
            audio_vae = convert_ltx2_audio_vae(
                original_audio_vae_ckpt, version=ltx_model_version
            ).to(dtype)
            del original_audio_vae_ckpt
            original_connectors_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, dit_prefix
            )
            connectors = convert_ltx2_connectors(
                original_connectors_ckpt, version=ltx_model_version
            ).to(dtype)
            del original_connectors_ckpt
            original_vocoder_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, vocoder_prefix
            )
            vocoder = convert_ltx2_vocoder(
                original_vocoder_ckpt, version=ltx_model_version
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
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

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

        if self.low_vram:
            pipeline.enable_model_cpu_offload(device=self.device_torch)

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
        self.model.to(self.device_torch, dtype=self.torch_dtype)

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

        # frames must be divisible by 8 then + 1. so 1, 9, 17, 25, etc.
        if gen_config.num_frames != 1:
            if (gen_config.num_frames - 1) % 8 != 0:
                gen_config.num_frames = ((gen_config.num_frames - 1) // 8) * 8 + 1

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

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        timestep_model_input = (1000 - timestep) / 1000

        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            text_embeddings.text_embeds,
        )[0]

        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, prompt_attention_mask, _, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device_torch,
        )
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
        # TODO convert to original ltx2 keys
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        # TODO convert to diffusers ltx2 keys
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
