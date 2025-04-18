# WIP, coming soon ish
from functools import partial
import torch
import yaml
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.prompt_utils import PromptEmbeds
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
import os
import sys

import weakref
import torch
import yaml
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.prompt_utils import PromptEmbeds

import os
import copy
from toolkit.config_modules import ModelConfig, GenerateImageConfig
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch.nn.functional as F

from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import XLA_AVAILABLE
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from PIL import Image

from .wan21 import \
    scheduler_configUniPC, \
    scheduler_config, \
    Wan21

from .wan_utils import add_first_frame_conditioning


class AggressiveWanI2VUnloadPipeline(WanImageToVideoPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
        )
        self._exec_device = device
        
    @property
    def _execution_device(self):
        return self._exec_device
    
    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
            
        # unload vae and transformer
        device = self.transformer.device
        
        self.text_encoder.to(device)
        
        self.vae.to('cpu')
        self.image_encoder.to('cpu')
        flush()

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
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
        flush()

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        self.image_encoder.to(device)
        self.vae.to(device)
        image_embeds = self.encode_image(image)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        latents, condition = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.bfloat16,
            device,
            generator,
            latents,
        )
        self.image_encoder.to('cpu')
        self.vae.to('cpu')
        flush()

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,  # todo I think unconditional should be scaled down version
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        self.vae.to(device)

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
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    def encode_image(self, image: PipelineImageInput):
        image = self.image_processor(images=image, return_tensors="pt")
        image = {k: v.to(self.image_encoder.device, dtype=self.image_encoder.dtype) for k, v in image.items()}
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]
    




class Wan21I2V(Wan21):
    arch = 'wan21_i2v'
    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(
            device, model_config, dtype,
            custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['WanTransformer3DModel']
        self.image_encoder: CLIPVisionModel = None
        self.image_processor: CLIPImageProcessor = None

    def load_model(self):
        # call the super class to load most of the model
        super().load_model()
        if self.model_config.low_vram:
            # unload text encoder
            self.text_encoder.to("cpu")
        # all the base stuff is loaded. We now need to load the vision encoder stuff
        dtype = self.torch_dtype
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.model_config.extras_name_or_path  , 
                subfolder="image_processor"
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                self.model_config.extras_name_or_path,
                subfolder="image_encoder",
                torch_dtype=dtype,
            )
        except Exception as e:
            # load from name_or_path
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.model_config.name_or_path_original, 
                subfolder="image_processor"
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                self.model_config.name_or_path_original,
                subfolder="image_encoder",
                torch_dtype=dtype,
            )
        self.image_encoder.to(self.device_torch, dtype=dtype)
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)
        
        if self.model_config.low_vram:
            # unload image encoder
            self.image_encoder.to("cpu")
        
        # rebuild the pipeline
        self.pipeline = self.get_generation_pipeline()
        flush()
    
    def generate_images(
            self,
            image_configs,
            sampler=None,
            pipeline=None,
    ):
        # will oom on 24gb vram if we dont unload vision encoder first
        if self.model_config.low_vram:
            # unload image encoder
            self.image_encoder.to("cpu")
            self.vae.to("cpu")
            self.transformer.to("cpu")
        flush()
        super().generate_images(
            image_configs,
            sampler=sampler,
            pipeline=pipeline,
        )
    
    def set_device_state_preset(self, *args, **kwargs):
        # set the device state to cpu for the image encoder
        if self.model_config.low_vram:
            return
        super().set_device_state_preset(*args, **kwargs)
        

    def get_generation_pipeline(self):
        scheduler = UniPCMultistepScheduler(**scheduler_configUniPC)
        if self.model_config.low_vram:
            pipeline = AggressiveWanI2VUnloadPipeline(
                vae=self.vae,
                transformer=self.model,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=scheduler,
                image_encoder=self.image_encoder,
                image_processor=self.image_processor,
                device=self.device_torch
            )
        else:
            pipeline = WanImageToVideoPipeline(
                vae=self.vae,
                transformer=self.unet,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=scheduler,
                image_encoder=self.image_encoder,
                image_processor=self.image_processor,
            )

        # pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: WanImageToVideoPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        # pipeline = pipeline.to(self.device_torch)
        
        
        if gen_config.ctrl_img is None:
            raise ValueError("I2V samples must have a control image")
        
        control_img = Image.open(gen_config.ctrl_img).convert("RGB")
        
        height = gen_config.height
        width = gen_config.width
        
        # make sure they are divisible by 16
        height = height // 16 * 16
        width = width // 16 * 16
        
        # resize the control image
        control_img = control_img.resize((width, height), Image.LANCZOS)
                
        output = pipeline(
            image=control_img,
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=height,
            width=width,
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


    def preprocess_clip_image(self, image_n1p1):
        # tensor shape: (bs, ch, height, width) with values in range [-1, 1]
        # Convert from [-1, 1] to [0, 1] range
        tensor = (image_n1p1 + 1) / 2
        
        # Resize to 224x224 (using bilinear interpolation, which is resample=3 in PIL)
        if tensor.shape[2] != 224 or tensor.shape[3] != 224:
            tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize with mean and std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(tensor.device)
        tensor = (tensor - mean) / std
        
        return tensor

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs
    ):
        # videos come in (bs, num_frames, channels, height, width)
        # images come in (bs, channels, height, width)
        with torch.no_grad():
            frames = batch.tensor
            if len(frames.shape) == 4:
                first_frames = frames
            elif len(frames.shape) == 5:
                first_frames = frames[:, 0]
            else:
                raise ValueError(f"Unknown frame shape {frames.shape}")
            
            # first_frames shape is (bs, channels, height, width), -1 to 1
            preprocessed_frames = self.preprocess_clip_image(first_frames)
            preprocessed_frames = preprocessed_frames.to(self.device_torch, dtype=self.torch_dtype)
            # preprocessed_frame shape is (bs, 3, 224, 224)
            self.image_encoder.to(self.device_torch)
            image_embeds_full = self.image_encoder(preprocessed_frames, output_hidden_states=True)
            image_embeds = image_embeds_full.hidden_states[-2]
            image_embeds = image_embeds.to(self.device_torch, dtype=self.torch_dtype)
            
            # Add conditioning using the standalone function
            conditioned_latent = add_first_frame_conditioning(
                latent_model_input=latent_model_input,
                first_frame=first_frames,
                vae=self.vae
            )
        
        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            encoder_hidden_states_image=image_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred