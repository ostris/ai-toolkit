import gc
import json
import shutil
import typing
from typing import Union, List, Tuple, Iterator
import sys
import os
from collections import OrderedDict

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from safetensors.torch import save_file, load_file
from torch.nn import Parameter
from tqdm import tqdm
from torchvision.transforms import Resize

from library.model_util import convert_unet_state_dict_to_sd, convert_text_encoder_state_dict_to_sd_v2, \
    convert_vae_state_dict, load_vae
from toolkit import train_tools
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.paths import REPOS_ROOT, KEYMAPS_ROOT
from toolkit.prompt_utils import inject_trigger_into_prompt, PromptEmbeds
from toolkit.sampler import get_sampler
from toolkit.saving import save_ldm_model_from_diffusers
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import torch
from library import model_util
from library.sdxl_model_util import convert_text_encoder_2_state_dict_to_sdxl
from diffusers.schedulers import DDPMScheduler
from toolkit.pipelines import CustomStableDiffusionXLPipeline, CustomStableDiffusionPipeline, \
    StableDiffusionKDiffusionXLPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import diffusers

# tell it to shut up
diffusers.logging.set_verbosity(diffusers.logging.ERROR)

SD_PREFIX_VAE = "vae"
SD_PREFIX_UNET = "unet"
SD_PREFIX_TEXT_ENCODER = "te"

SD_PREFIX_TEXT_ENCODER1 = "te0"
SD_PREFIX_TEXT_ENCODER2 = "te1"

# prefixed diffusers keys
DO_NOT_TRAIN_WEIGHTS = [
    "unet_time_embedding.linear_1.bias",
    "unet_time_embedding.linear_1.weight",
    "unet_time_embedding.linear_2.bias",
    "unet_time_embedding.linear_2.weight",
]


class BlankNetwork:

    def __init__(self):
        self.multiplier = 1.0
        self.is_active = True
        self.is_normalizing = False

    def apply_stored_normalizer(self, target_normalize_scaler: float = 1.0):
        pass

    def __enter__(self):
        self.is_active = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_active = False


def flush():
    torch.cuda.empty_cache()
    gc.collect()


UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

# if is type checking
if typing.TYPE_CHECKING:
    from diffusers import \
        StableDiffusionPipeline, \
        AutoencoderKL, \
        UNet2DConditionModel
    from diffusers.schedulers import KarrasDiffusionSchedulers
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection


class StableDiffusion:

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='fp16',
            custom_pipeline=None,
            noise_scheduler=None,
    ):
        self.custom_pipeline = custom_pipeline
        self.device = device
        self.dtype = dtype
        self.torch_dtype = get_torch_dtype(dtype)
        self.device_torch = torch.device(self.device)
        self.model_config = model_config
        self.prediction_type = "v_prediction" if self.model_config.is_v_pred else "epsilon"

        self.pipeline: Union[None, 'StableDiffusionPipeline', 'CustomStableDiffusionXLPipeline']
        self.vae: Union[None, 'AutoencoderKL']
        self.unet: Union[None, 'UNet2DConditionModel']
        self.text_encoder: Union[None, 'CLIPTextModel', List[Union['CLIPTextModel', 'CLIPTextModelWithProjection']]]
        self.tokenizer: Union[None, 'CLIPTokenizer', List['CLIPTokenizer']]
        self.noise_scheduler: Union[None, 'KarrasDiffusionSchedulers', 'DDPMScheduler'] = noise_scheduler

        # sdxl stuff
        self.logit_scale = None
        self.ckppt_info = None
        self.is_loaded = False

        # to hold network if there is one
        self.network = None
        self.is_xl = model_config.is_xl
        self.is_v2 = model_config.is_v2

        self.use_text_encoder_1 = model_config.use_text_encoder_1
        self.use_text_encoder_2 = model_config.use_text_encoder_2

        self.config_file = None

    def load_model(self):
        if self.is_loaded:
            return
        dtype = get_torch_dtype(self.dtype)

        # TODO handle other schedulers
        # sch = KDPM2DiscreteScheduler
        if self.noise_scheduler is None:
            sch = DDPMScheduler
            # do our own scheduler
            prediction_type = "v_prediction" if self.model_config.is_v_pred else "epsilon"
            scheduler = sch(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
                clip_sample=False,
                prediction_type=prediction_type,
                steps_offset=0
            )
            self.noise_scheduler = scheduler

        # move the betas alphas and  alphas_cumprod to device. Sometimed they get stuck on cpu, not sure why
        self.noise_scheduler.betas = self.noise_scheduler.betas.to(self.device_torch)
        self.noise_scheduler.alphas = self.noise_scheduler.alphas.to(self.device_torch)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device_torch)

        model_path = self.model_config.name_or_path
        if 'civitai.com' in self.model_config.name_or_path:
            # load is a civit ai model, use the loader.
            from toolkit.civitai import get_model_path_from_url
            model_path = get_model_path_from_url(self.model_config.name_or_path)

        if self.model_config.is_xl:
            if self.custom_pipeline is not None:
                pipln = self.custom_pipeline
            else:
                pipln = StableDiffusionXLPipeline
                # pipln = StableDiffusionKDiffusionXLPipeline

            # see if path exists
            if not os.path.exists(model_path):
                # try to load with default diffusers
                pipe = pipln.from_pretrained(
                    model_path,
                    dtype=dtype,
                    scheduler_type='ddpm',
                    device=self.device_torch,
                ).to(self.device_torch)
            else:
                pipe = pipln.from_single_file(
                    model_path,
                    dtype=dtype,
                    scheduler_type='ddpm',
                    device=self.device_torch,
                ).to(self.device_torch)

            text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
            tokenizer = [pipe.tokenizer, pipe.tokenizer_2]
            for text_encoder in text_encoders:
                text_encoder.to(self.device_torch, dtype=dtype)
                text_encoder.requires_grad_(False)
                text_encoder.eval()
            text_encoder = text_encoders

            if self.model_config.experimental_xl:
                print("Experimental XL mode enabled")
                print("Loading and injecting alt weights")
                # load the mismatched weight and force it in
                raw_state_dict = load_file(model_path)
                replacement_weight = raw_state_dict['conditioner.embedders.1.model.text_projection'].clone()
                del raw_state_dict
                #  get state dict for  for 2nd text encoder
                te1_state_dict = text_encoders[1].state_dict()
                # replace weight with mismatched weight
                te1_state_dict['text_projection.weight'] = replacement_weight.to(self.device_torch, dtype=dtype)
                flush()
                print("Injecting alt weights")

        else:
            if self.custom_pipeline is not None:
                pipln = self.custom_pipeline
            else:
                pipln = StableDiffusionPipeline

            # see if path exists
            if not os.path.exists(model_path):
                # try to load with default diffusers
                pipe = pipln.from_pretrained(
                    model_path,
                    dtype=dtype,
                    scheduler_type='dpm',
                    device=self.device_torch,
                    load_safety_checker=False,
                    requires_safety_checker=False,
                    safety_checker=False
                ).to(self.device_torch)
            else:
                pipe = pipln.from_single_file(
                    model_path,
                    dtype=dtype,
                    scheduler_type='dpm',
                    device=self.device_torch,
                    load_safety_checker=False,
                    requires_safety_checker=False,
                    safety_checker=False
                ).to(self.device_torch)

            pipe.register_to_config(requires_safety_checker=False)
            text_encoder = pipe.text_encoder
            text_encoder.to(self.device_torch, dtype=dtype)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            tokenizer = pipe.tokenizer

        # scheduler doesn't get set sometimes, so we set it here
        pipe.scheduler = self.noise_scheduler

        if self.model_config.vae_path is not None:
            external_vae = load_vae(self.model_config.vae_path, dtype)
            pipe.vae = external_vae

        self.unet = pipe.unet
        self.vae = pipe.vae.to(self.device_torch, dtype=dtype)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.unet.to(self.device_torch, dtype=dtype)
        self.unet.requires_grad_(False)
        self.unet.eval()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.pipeline = pipe
        self.is_loaded = True

    def generate_images(self, image_configs: List[GenerateImageConfig], sampler=None):
        # sample_folder = os.path.join(self.save_root, 'samples')
        if self.network is not None:
            self.network.eval()
            network = self.network
        else:
            network = BlankNetwork()

        was_network_normalizing = network.is_normalizing
        # apply the normalizer if it is normalizing before inference and disable it
        if network.is_normalizing:
            network.apply_stored_normalizer()
            network.is_normalizing = False

        # save current seed state for training
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        original_device_dict = {
            'vae': self.vae.device,
            'unet': self.unet.device,
            # 'tokenizer': self.tokenizer.device,
        }

        # handle sdxl text encoder
        if isinstance(self.text_encoder, list):
            for encoder, i in zip(self.text_encoder, range(len(self.text_encoder))):
                original_device_dict[f'text_encoder_{i}'] = encoder.device
                encoder.to(self.device_torch)
        else:
            original_device_dict['text_encoder'] = self.text_encoder.device
            self.text_encoder.to(self.device_torch)

        self.vae.to(self.device_torch)
        self.unet.to(self.device_torch)

        noise_scheduler = self.noise_scheduler
        if sampler is not None:
            if sampler.startswith("sample_"):  # sample_dpmpp_2m
                # using ksampler
                noise_scheduler = get_sampler('lms')
            else:
                noise_scheduler = get_sampler(sampler)

        if sampler.startswith("sample_") and self.is_xl:
            # using kdiffusion
            Pipe = StableDiffusionKDiffusionXLPipeline
        else:
            Pipe = StableDiffusionXLPipeline


        # TODO add clip skip
        if self.is_xl:
            pipeline = Pipe(
                vae=self.vae,
                unet=self.unet,
                text_encoder=self.text_encoder[0],
                text_encoder_2=self.text_encoder[1],
                tokenizer=self.tokenizer[0],
                tokenizer_2=self.tokenizer[1],
                scheduler=noise_scheduler,
                add_watermarker=False,
            ).to(self.device_torch)
            # force turn that (ruin your images with obvious green and red dots) the #$@@ off!!!
            pipeline.watermark = None
        else:
            pipeline = StableDiffusionPipeline(
                vae=self.vae,
                unet=self.unet,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(self.device_torch)
        # disable progress bar
        pipeline.set_progress_bar_config(disable=True)

        if sampler.startswith("sample_"):
            pipeline.set_scheduler(sampler)

        start_multiplier = 1.0
        if self.network is not None:
            start_multiplier = self.network.multiplier

        pipeline.to(self.device_torch)
        with network:
            with torch.no_grad():
                if self.network is not None:
                    assert self.network.is_active

                for i in tqdm(range(len(image_configs)), desc=f"Generating Images", leave=False):
                    gen_config = image_configs[i]

                    if self.network is not None:
                        self.network.multiplier = gen_config.network_multiplier
                    torch.manual_seed(gen_config.seed)
                    torch.cuda.manual_seed(gen_config.seed)

                    # todo do we disable text encoder here as well if disabled for model, or only do that for training?
                    if self.is_xl:
                        # fix guidance rescale for sdxl
                        # was trained on 0.7 (I believe)

                        grs = gen_config.guidance_rescale
                        if grs is None or grs < 0.00001:
                            grs = 0.7
                        # grs = 0.0

                        extra = {}
                        if sampler.startswith("sample_"):
                            extra['use_karras_sigmas'] = True


                        img = pipeline(
                            prompt=gen_config.prompt,
                            prompt_2=gen_config.prompt_2,
                            negative_prompt=gen_config.negative_prompt,
                            negative_prompt_2=gen_config.negative_prompt_2,
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            guidance_rescale=grs,
                            **extra
                        ).images[0]
                    else:
                        img = pipeline(
                            prompt=gen_config.prompt,
                            negative_prompt=gen_config.negative_prompt,
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                        ).images[0]

                    gen_config.save_image(img)

        # clear pipeline and cache to reduce vram usage
        del pipeline
        torch.cuda.empty_cache()

        # restore training state
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        self.vae.to(original_device_dict['vae'])
        self.unet.to(original_device_dict['unet'])
        if isinstance(self.text_encoder, list):
            for encoder, i in zip(self.text_encoder, range(len(self.text_encoder))):
                encoder.to(original_device_dict[f'text_encoder_{i}'])
        else:
            self.text_encoder.to(original_device_dict['text_encoder'])
        if self.network is not None:
            self.network.train()
            self.network.multiplier = start_multiplier
            self.network.is_normalizing = was_network_normalizing
        # self.tokenizer.to(original_device_dict['tokenizer'])

    def get_latent_noise(
            self,
            height=None,
            width=None,
            pixel_height=None,
            pixel_width=None,
            batch_size=1,
            noise_offset=0.0,
    ):
        if height is None and pixel_height is None:
            raise ValueError("height or pixel_height must be specified")
        if width is None and pixel_width is None:
            raise ValueError("width or pixel_width must be specified")
        if height is None:
            height = pixel_height // VAE_SCALE_FACTOR
        if width is None:
            width = pixel_width // VAE_SCALE_FACTOR

        noise = torch.randn(
            (
                batch_size,
                self.unet.config['in_channels'],
                height,
                width,
            ),
            device=self.unet.device,
        )
        noise = apply_noise_offset(noise, noise_offset)
        return noise

    def get_time_ids_from_latents(self, latents: torch.Tensor):
        bs, ch, h, w = list(latents.shape)

        height = h * VAE_SCALE_FACTOR
        width = w * VAE_SCALE_FACTOR

        dtype = latents.dtype

        if self.is_xl:
            prompt_ids = train_tools.get_add_time_ids(
                height,
                width,
                dynamic_crops=False,  # look into this
                dtype=dtype,
            ).to(self.device_torch, dtype=dtype)
            return prompt_ids
        else:
            return None

    def predict_noise(
            self,
            latents: torch.Tensor,
            text_embeddings: Union[PromptEmbeds, None] = None,
            timestep: Union[int, torch.Tensor] = 1,
            guidance_scale=7.5,
            guidance_rescale=0,  # 0.7 sdxl
            add_time_ids=None,
            conditional_embeddings: Union[PromptEmbeds, None] = None,
            unconditional_embeddings: Union[PromptEmbeds, None] = None,
            **kwargs,
    ):
        # get the embeddings
        if text_embeddings is None and conditional_embeddings is None:
            raise ValueError("Either text_embeddings or conditional_embeddings must be specified")
        if text_embeddings is None and unconditional_embeddings is not None:
            text_embeddings = train_tools.concat_prompt_embeddings(
                unconditional_embeddings,  # negative embedding
                conditional_embeddings,  # positive embedding
                1,  # batch size
            )
        elif text_embeddings is None and conditional_embeddings is not None:
            # not doing cfg
            text_embeddings = conditional_embeddings

        # CFG is comparing neg and positive, if we have concatenated embeddings
        # then we are doing it, otherwise we are not and takes half the time.
        do_classifier_free_guidance = True

        # check if batch size of embeddings matches batch size of latents
        if latents.shape[0] == text_embeddings.text_embeds.shape[0]:
            do_classifier_free_guidance = False
        elif latents.shape[0] * 2 != text_embeddings.text_embeds.shape[0]:
            raise ValueError("Batch size of latents must be the same or half the batch size of text embeddings")

        if self.is_xl:
            if add_time_ids is None:
                add_time_ids = self.get_time_ids_from_latents(latents)

                if do_classifier_free_guidance:
                    # todo check this with larget batches
                    add_time_ids = train_tools.concat_embeddings(
                        add_time_ids, add_time_ids, int(latents.shape[0])
                    )
                else:
                    # concat to fit batch size
                    add_time_ids = torch.cat([add_time_ids] * latents.shape[0])

            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep)

            added_cond_kwargs = {
                # todo can we zero here the second text encoder? or match a blank string?
                "text_embeds": text_embeddings.pooled_embeds,
                "time_ids": add_time_ids,
            }

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings.text_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            if do_classifier_free_guidance:
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

                # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        else:
            if do_classifier_free_guidance:
                # if we are doing classifier free guidance, need to double up
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep)

            # check if we need to concat timesteps
            if isinstance(timestep, torch.Tensor) and len(timestep.shape) > 1:
                ts_bs = timestep.shape[0]
                if ts_bs != latent_model_input.shape[0]:
                    if ts_bs == 1:
                        timestep = torch.cat([timestep] * latent_model_input.shape[0])
                    elif ts_bs * 2 == latent_model_input.shape[0]:
                        timestep = torch.cat([timestep] * 2)
                    else:
                        raise ValueError(
                            f"Batch size of latents {latent_model_input.shape[0]} must be the same or half the batch size of timesteps {timestep.shape[0]}")

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings.text_embeds,
            ).sample

            if do_classifier_free_guidance:
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

                # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        return noise_pred

    # ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
    def diffuse_some_steps(
            self,
            latents: torch.FloatTensor,
            text_embeddings: PromptEmbeds,
            total_timesteps: int = 1000,
            start_timesteps=0,
            guidance_scale=1,
            add_time_ids=None,
            **kwargs,
    ):

        for timestep in tqdm(self.noise_scheduler.timesteps[start_timesteps:total_timesteps], leave=False):
            noise_pred = self.predict_noise(
                latents,
                text_embeddings,
                timestep,
                guidance_scale=guidance_scale,
                add_time_ids=add_time_ids,
                **kwargs,
            )
            latents = self.noise_scheduler.step(noise_pred, timestep, latents).prev_sample

        # return latents_steps
        return latents

    def encode_prompt(self, prompt, num_images_per_prompt=1) -> PromptEmbeds:
        prompt = prompt
        # if it is not a list, make it one
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.is_xl:
            return PromptEmbeds(
                train_tools.encode_prompts_xl(
                    self.tokenizer,
                    self.text_encoder,
                    prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    use_text_encoder_1=self.use_text_encoder_1,
                    use_text_encoder_2=self.use_text_encoder_2,
                )
            )
        else:
            return PromptEmbeds(
                train_tools.encode_prompts(
                    self.tokenizer, self.text_encoder, prompt
                )
            )

    @torch.no_grad()
    def encode_images(
            self,
            image_list: List[torch.Tensor],
            device=None,
            dtype=None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.torch_dtype

        latent_list = []
        # Move to vae to device if on cpu
        if self.vae.device == 'cpu':
            self.vae.to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        # move to device and dtype
        image_list = [image.to(self.device, dtype=self.torch_dtype) for image in image_list]

        # resize images if not divisible by 8
        for i in range(len(image_list)):
            image = image_list[i]
            if image.shape[1] % 8 != 0 or image.shape[2] % 8 != 0:
                image_list[i] = Resize((image.shape[1] // 8 * 8, image.shape[2] // 8 * 8))(image)

        images = torch.stack(image_list)
        flush()
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config['scaling_factor']
        latents = latents.to(device, dtype=dtype)

        return latents

    def decode_latents(
            self,
            latents: torch.Tensor,
            device=None,
            dtype=None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.torch_dtype

        # Move to vae to device if on cpu
        if self.vae.device == 'cpu':
            self.vae.to(self.device)
        latents = latents.to(device, dtype=dtype)
        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        images = images.to(device, dtype=dtype)

        return images

    def encode_image_prompt_pairs(
            self,
            prompt_list: List[str],
            image_list: List[torch.Tensor],
            device=None,
            dtype=None
    ):
        # todo check image types and expand and rescale as needed
        # device and dtype are for outputs
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.torch_dtype

        embedding_list = []
        latent_list = []
        # embed the prompts
        for prompt in prompt_list:
            embedding = self.encode_prompt(prompt).to(self.device_torch, dtype=dtype)
            embedding_list.append(embedding)

        return embedding_list, latent_list

    def get_weight_by_name(self, name):
        # weights begin with te{te_num}_ for text encoder
        # weights begin with unet_ for unet_
        if name.startswith('te'):
            key = name[4:]
            # text encoder
            te_num = int(name[2])
            if isinstance(self.text_encoder, list):
                return self.text_encoder[te_num].state_dict()[key]
            else:
                return self.text_encoder.state_dict()[key]
        elif name.startswith('unet'):
            key = name[5:]
            # unet
            return self.unet.state_dict()[key]

        raise ValueError(f"Unknown weight name: {name}")

    def inject_trigger_into_prompt(self, prompt, trigger=None, to_replace_list=None, add_if_not_present=False):
        return inject_trigger_into_prompt(
            prompt,
            trigger=trigger,
            to_replace_list=to_replace_list,
            add_if_not_present=add_if_not_present,
        )

    def state_dict(self, vae=True, text_encoder=True, unet=True):
        state_dict = OrderedDict()
        if vae:
            for k, v in self.vae.state_dict().items():
                new_key = k if k.startswith(f"{SD_PREFIX_VAE}") else f"{SD_PREFIX_VAE}_{k}"
                state_dict[new_key] = v
        if text_encoder:
            if isinstance(self.text_encoder, list):
                for i, encoder in enumerate(self.text_encoder):
                    for k, v in encoder.state_dict().items():
                        new_key = k if k.startswith(
                            f"{SD_PREFIX_TEXT_ENCODER}{i}_") else f"{SD_PREFIX_TEXT_ENCODER}{i}_{k}"
                        state_dict[new_key] = v
            else:
                for k, v in self.text_encoder.state_dict().items():
                    new_key = k if k.startswith(f"{SD_PREFIX_TEXT_ENCODER}_") else f"{SD_PREFIX_TEXT_ENCODER}_{k}"
                    state_dict[new_key] = v
        if unet:
            for k, v in self.unet.state_dict().items():
                new_key = k if k.startswith(f"{SD_PREFIX_UNET}_") else f"{SD_PREFIX_UNET}_{k}"
                state_dict[new_key] = v
        return state_dict

    def named_parameters(self, vae=True, text_encoder=True, unet=True, state_dict_keys=False) -> OrderedDict[str, Parameter]:
        named_params: OrderedDict[str, Parameter] = OrderedDict()
        if vae:
            for name, param in self.vae.named_parameters(recurse=True, prefix=f"{SD_PREFIX_VAE}"):
                named_params[name] = param
        if text_encoder:
            if isinstance(self.text_encoder, list):
                for i, encoder in enumerate(self.text_encoder):
                    for name, param in encoder.named_parameters(recurse=True, prefix=f"{SD_PREFIX_TEXT_ENCODER}{i}"):
                        named_params[name] = param
            else:
                for name, param in self.text_encoder.named_parameters(recurse=True, prefix=f"{SD_PREFIX_TEXT_ENCODER}"):
                    named_params[name] = param
        if unet:
            for name, param in self.unet.named_parameters(recurse=True, prefix=f"{SD_PREFIX_UNET}"):
                named_params[name] = param

        # convert to state dict keys, jsut replace . with _ on keys
        if state_dict_keys:
            new_named_params = OrderedDict()
            for k, v in named_params.items():
                # replace only the first . with an _
                new_key = k.replace('.', '_', 1)
                new_named_params[new_key] = v
            named_params = new_named_params

        return named_params


    def save(self, output_file: str, meta: OrderedDict, save_dtype=get_torch_dtype('fp16'), logit_scale=None):
        version_string = '1'
        if self.is_v2:
            version_string = '2'
        if self.is_xl:
            version_string = 'sdxl'
        save_ldm_model_from_diffusers(
            sd=self,
            output_file=output_file,
            meta=meta,
            save_dtype=save_dtype,
            sd_version=version_string,
        )
        if self.config_file is not None:
            output_path_no_ext = os.path.splitext(output_file)[0]
            output_config_path = f"{output_path_no_ext}.yaml"
            shutil.copyfile(self.config_file, output_config_path)

    def prepare_optimizer_params(
            self,
            unet=False,
            text_encoder=False,
            text_encoder_lr=None,
            unet_lr=None,
            default_lr=1e-6,
    ):
        # todo maybe only get locon ones?
        # not all items are saved, to make it match, we need to match out save mappings
        # and not train anything not mapped. Also add learning rate
        version = 'sd1'
        if self.is_xl:
            version = 'sdxl'
        if self.is_v2:
            version = 'sd2'
        mapping_filename = f"stable_diffusion_{version}.json"
        mapping_path = os.path.join(KEYMAPS_ROOT, mapping_filename)
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        ldm_diffusers_keymap = mapping['ldm_diffusers_keymap']

        trainable_parameters = []

        # we use state dict to find params

        if unet:
            named_params = self.named_parameters(vae=False, unet=unet, text_encoder=False, state_dict_keys=True)
            unet_lr = unet_lr if unet_lr is not None else default_lr
            params = []
            for key, diffusers_key in ldm_diffusers_keymap.items():
                if diffusers_key in named_params and diffusers_key not in DO_NOT_TRAIN_WEIGHTS:
                    if named_params[diffusers_key].requires_grad:
                        params.append(named_params[diffusers_key])
            param_data = {"params": params, "lr": unet_lr}
            trainable_parameters.append(param_data)
            print(f"Found {len(params)} trainable parameter in unet")

        if text_encoder:
            named_params = self.named_parameters(vae=False, unet=False, text_encoder=text_encoder, state_dict_keys=True)
            text_encoder_lr = text_encoder_lr if text_encoder_lr is not None else default_lr
            params = []
            for key, diffusers_key in ldm_diffusers_keymap.items():
                if diffusers_key in named_params and diffusers_key not in DO_NOT_TRAIN_WEIGHTS:
                    if named_params[diffusers_key].requires_grad:
                        params.append(named_params[diffusers_key])
            param_data = {"params": params, "lr": text_encoder_lr}
            trainable_parameters.append(param_data)

            print(f"Found {len(params)} trainable parameter in text encoder")

        return trainable_parameters
