# WIP, coming soon ish
from functools import partial
import torch
import yaml
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import  WanPipeline, WanTransformer3DModel, AutoencoderKL
from .autoencoder_kl_wan import AutoencoderKLWan
import os
import sys

import weakref
import torch
import yaml

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds

import os
import copy
from toolkit.config_modules import ModelConfig, GenerateImageConfig, ModelArch
import torch
from optimum.quanto import freeze, qfloat8, QTensor, qint4
from toolkit.util.quantize import quantize, get_qtype
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from typing import TYPE_CHECKING, List
from toolkit.accelerator import unwrap_model
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import XLA_AVAILABLE
# from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from typing import Any, Callable, Dict, List, Optional, Union
from toolkit.models.wan21.wan_lora_convert import convert_to_diffusers, convert_to_original

# for generation only?
scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.33.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False
}

# for training. I think it is right
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False
}


class AggressiveWanUnloadPipeline(WanPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        self._exec_device = device
    @property
    def _execution_device(self):
        return self._exec_device
    
    def __call__(
        self: WanPipeline,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None],
                  PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # unload vae and transformer
        vae_device = self.vae.device
        transformer_device = self.transformer.device
        text_encoder_device = self.text_encoder.device
        device = self.transformer.device
        
        print("Unloading vae")
        self.vae.to("cpu")
        self.text_encoder.to(device)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # unload text encoder
        print("Unloading text encoder")
        self.text_encoder.to("cpu")

        self.transformer.to(device)

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(device, transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device, transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(device, transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * \
                        (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # unload transformer
        # load vae
        print("Loading Vae")
        self.vae.to(vae_device)

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


class Wan21(BaseModel):
    arch = 'wan21'
    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(device, model_config, dtype,
                         custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['WanTransformer3DModel']

        # cache for holding noise
        self.effective_noise = None
        
    def get_bucket_divisibility(self):
        return 16

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def load_model(self):
        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading Wan2.1 model")
        subfolder = 'transformer'
        transformer_path = model_path
        if os.path.exists(transformer_path):
            subfolder = None
            transformer_path = os.path.join(transformer_path, 'transformer')
        
        te_path = self.model_config.extras_name_or_path    
        if os.path.exists(os.path.join(model_path, 'text_encoder')):
            te_path = model_path
        
        vae_path = self.model_config.extras_name_or_path
        if os.path.exists(os.path.join(model_path, 'vae')):
            vae_path = model_path

        self.print_and_status_update("Loading transformer")
        transformer = WanTransformer3DModel.from_pretrained(
            transformer_path,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.1 models")

        if not self.model_config.low_vram:
            # quantize on the device
            transformer.to(self.quantize_device, dtype=dtype)
            flush()

        if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
            raise ValueError(
                "Assistant LoRA is not supported for Wan2.1 models currently")

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for Wan2.1 models currently")

        flush()

        if self.model_config.quantize:
            print("Quantizing Transformer")
            quantization_args = self.model_config.quantize_kwargs
            if 'exclude' not in quantization_args:
                quantization_args['exclude'] = []
            # patch the state dict method
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update("Quantizing transformer")
            if self.model_config.low_vram:
                print("Quantizing blocks")
                orig_exclude = copy.deepcopy(quantization_args['exclude'])
                # quantize each block
                idx = 0
                for block in tqdm(transformer.blocks):
                    block.to(self.device_torch)
                    quantize(block, weights=quantization_type,
                             **quantization_args)
                    freeze(block)
                    idx += 1
                    flush()

                print("Quantizing the rest")
                low_vram_exclude = copy.deepcopy(quantization_args['exclude'])
                low_vram_exclude.append('blocks.*')
                quantization_args['exclude'] = low_vram_exclude
                # quantize the rest
                transformer.to(self.device_torch)
                quantize(transformer, weights=quantization_type,
                         **quantization_args)

                quantization_args['exclude'] = orig_exclude
            else:
                # do it in one go
                quantize(transformer, weights=quantization_type,
                         **quantization_args)
            freeze(transformer)
            # move it to the cpu for now
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)

        flush()

        self.print_and_status_update("Loading UMT5EncoderModel")
        tokenizer = AutoTokenizer.from_pretrained(
            te_path, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder = UMT5EncoderModel.from_pretrained(
            te_path, subfolder="text_encoder", torch_dtype=dtype).to(dtype=dtype)

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing UMT5EncoderModel")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        if self.model_config.low_vram:
            print("Moving transformer back to GPU")
            # we can move it back to the gpu now
            transformer.to(self.device_torch)

        scheduler = Wan21.get_train_scheduler()
        self.print_and_status_update("Loading VAE")
        # todo, example does float 32? check if quality suffers
        vae = AutoencoderKLWan.from_pretrained(
            vae_path, subfolder="vae", torch_dtype=dtype).to(dtype=dtype)
        flush()

        self.print_and_status_update("Making pipe")
        pipe: WanPipeline = WanPipeline(
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        text_encoder.to(self.device_torch)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()
        self.pipeline = pipe
        self.model = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def get_generation_pipeline(self):
        scheduler = UniPCMultistepScheduler(**scheduler_configUniPC)
        if self.model_config.low_vram:
            pipeline = AggressiveWanUnloadPipeline(
                vae=self.vae,
                transformer=self.model,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=scheduler,
                device=self.device_torch
            )
        else:
            pipeline = WanPipeline(
                vae=self.vae,
                transformer=self.unet,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=scheduler,
            )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: WanPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        pipeline = pipeline.to(self.device_torch)
        # todo, figure out how to do video
        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        # vae_scale_factor_spatial = 8
        # vae_scale_factor_temporal = 4
        # num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # shape = (
        #     batch_size,
        #     num_channels_latents, # 16
        #     num_latent_frames,  # 81
        #     int(height) // self.vae_scale_factor_spatial,
        #     int(width) // self.vae_scale_factor_spatial,
        # )

        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=512,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )
        return PromptEmbeds(prompt_embeds)

    @torch.no_grad()
    def encode_images(
            self,
            image_list: List[torch.Tensor],
            device=None,
            dtype=None
    ):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == 'cpu':
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

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
        B, C, T, H, W = images.shape

        # Resize if needed (B * T, C, H, W)
        if H % 8 != 0 or W % 8 != 0:
            target_h = H // 8 * 8
            target_w = W // 8 * 8
            images = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            images = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
            images = images.view(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)

        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std

        return latents.to(device, dtype=dtype)

    def get_model_has_grad(self):
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        return self.text_encoder.encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad

    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: Wan21 = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w', encoding='utf-8') as f:
            yaml.dump(meta, f, allow_unicode=True)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        if batch is None:
            raise ValueError("Batch is not provided")
        if noise is None:
            raise ValueError("Noise is not provided")
        return (noise - batch.latents).detach()

    def convert_lora_weights_before_save(self, state_dict):
        return convert_to_original(state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        return convert_to_diffusers(state_dict)
    
    def get_base_model_version(self):
        return "wan_2.1"
