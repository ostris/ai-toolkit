import gc
import typing
from typing import Union, List, Tuple
import sys
import os
from collections import OrderedDict

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from safetensors.torch import save_file
from tqdm import tqdm
from torchvision.transforms import Resize

from library.model_util import convert_unet_state_dict_to_sd, convert_text_encoder_state_dict_to_sd_v2, \
    convert_vae_state_dict, load_vae
from toolkit import train_tools
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.paths import REPOS_ROOT
from toolkit.saving import save_ldm_model_from_diffusers
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import torch
from library import model_util
from library.sdxl_model_util import convert_text_encoder_2_state_dict_to_sdxl
from diffusers.schedulers import DDPMScheduler
from toolkit.pipelines import CustomStableDiffusionXLPipeline, CustomStableDiffusionPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import diffusers

# tell it to shut up
diffusers.logging.set_verbosity(diffusers.logging.ERROR)

VAE_PREFIX_UNET = "vae"
SD_PREFIX_UNET = "unet"
SD_PREFIX_TEXT_ENCODER = "te"

SD_PREFIX_TEXT_ENCODER1 = "te1"
SD_PREFIX_TEXT_ENCODER2 = "te2"


class BlankNetwork:
    multiplier = 1.0
    is_active = True

    def __init__(self):
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


class PromptEmbeds:
    text_embeds: torch.Tensor
    pooled_embeds: Union[torch.Tensor, None]

    def __init__(self, args: Union[Tuple[torch.Tensor], List[torch.Tensor], torch.Tensor]) -> None:
        if isinstance(args, list) or isinstance(args, tuple):
            # xl
            self.text_embeds = args[0]
            self.pooled_embeds = args[1]
        else:
            # sdv1.x, sdv2.x
            self.text_embeds = args
            self.pooled_embeds = None

    def to(self, *args, **kwargs):
        self.text_embeds = self.text_embeds.to(*args, **kwargs)
        if self.pooled_embeds is not None:
            self.pooled_embeds = self.pooled_embeds.to(*args, **kwargs)
        return self


# if is type checking
if typing.TYPE_CHECKING:
    from diffusers import \
        StableDiffusionPipeline, \
        AutoencoderKL, \
        UNet2DConditionModel
    from diffusers.schedulers import KarrasDiffusionSchedulers
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection


class StableDiffusion:
    pipeline: Union[None, 'StableDiffusionPipeline', 'CustomStableDiffusionXLPipeline']
    vae: Union[None, 'AutoencoderKL']
    unet: Union[None, 'UNet2DConditionModel']
    text_encoder: Union[None, 'CLIPTextModel', List[Union['CLIPTextModel', 'CLIPTextModelWithProjection']]]
    tokenizer: Union[None, 'CLIPTokenizer', List['CLIPTokenizer']]
    noise_scheduler: Union[None, 'KarrasDiffusionSchedulers', 'DDPMScheduler']
    device: str
    dtype: str
    torch_dtype: torch.dtype
    device_torch: torch.device
    model_config: ModelConfig

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='fp16',
            custom_pipeline=None
    ):
        self.custom_pipeline = custom_pipeline
        self.device = device
        self.dtype = dtype
        self.torch_dtype = get_torch_dtype(dtype)
        self.device_torch = torch.device(self.device)
        self.model_config = model_config
        self.prediction_type = "v_prediction" if self.model_config.is_v_pred else "epsilon"

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

    def load_model(self):
        if self.is_loaded:
            return
        dtype = get_torch_dtype(self.dtype)

        # TODO handle other schedulers
        # sch = KDPM2DiscreteScheduler
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
            steps_offset=1
        )

        model_path = self.model_config.name_or_path
        if 'civitai.com' in self.model_config.name_or_path:
            # load is a civit ai model, use the loader.
            from toolkit.civitai import get_model_path_from_url
            model_path = get_model_path_from_url(self.model_config.name_or_path)

        if self.model_config.is_xl:
            if self.custom_pipeline is not None:
                pipln = self.custom_pipeline
            else:
                pipln = CustomStableDiffusionXLPipeline

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
        else:
            if self.custom_pipeline is not None:
                pipln = self.custom_pipeline
            else:
                pipln = CustomStableDiffusionPipeline

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
        pipe.scheduler = scheduler

        if self.model_config.vae_path is not None:
            external_vae = load_vae(self.model_config.vae_path, dtype)
            pipe.vae = external_vae

        self.unet = pipe.unet
        self.noise_scheduler = pipe.scheduler
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

    def generate_images(self, image_configs: List[GenerateImageConfig]):
        # sample_folder = os.path.join(self.save_root, 'samples')
        if self.network is not None:
            self.network.eval()
            network = self.network
        else:
            network = BlankNetwork()

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

        # TODO add clip skip
        if self.is_xl:
            pipeline = StableDiffusionXLPipeline(
                vae=self.vae,
                unet=self.unet,
                text_encoder=self.text_encoder[0],
                text_encoder_2=self.text_encoder[1],
                tokenizer=self.tokenizer[0],
                tokenizer_2=self.tokenizer[1],
                scheduler=self.noise_scheduler,
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
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(self.device_torch)
        # disable progress bar
        pipeline.set_progress_bar_config(disable=True)

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
                        img = pipeline(
                            prompt=gen_config.prompt,
                            prompt_2=gen_config.prompt_2,
                            negative_prompt=gen_config.negative_prompt,
                            negative_prompt_2=gen_config.negative_prompt_2,
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            guidance_rescale=gen_config.guidance_rescale,
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
                UNET_IN_CHANNELS,
                height,
                width,
            ),
            device="cpu",
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
                latents.shape[0],  # batch size
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
        # move to device and dtype
        image_list = [image.to(self.device, dtype=self.torch_dtype) for image in image_list]

        # resize images if not divisible by 8
        for i in range(len(image_list)):
            image = image_list[i]
            if image.shape[1] % 8 != 0 or image.shape[2] % 8 != 0:
                image_list[i] = Resize((image.shape[1] // 8 * 8, image.shape[2] // 8 * 8))(image)

        images = torch.stack(image_list)
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        latents = latents.to(device, dtype=dtype)

        return latents

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

    def state_dict(self, vae=True, text_encoder=True, unet=True):
        state_dict = OrderedDict()
        if vae:
            for k, v in self.vae.state_dict().items():
                new_key = k if k.startswith(f"{VAE_PREFIX_UNET}") else f"{VAE_PREFIX_UNET}_{k}"
                state_dict[new_key] = v
        if text_encoder:
            if isinstance(self.text_encoder, list):
                for i, encoder in enumerate(self.text_encoder):
                    for k, v in encoder.state_dict().items():
                        new_key = k if k.startswith(
                            f"{SD_PREFIX_TEXT_ENCODER}{i}") else f"{SD_PREFIX_TEXT_ENCODER}{i}_{k}"
                        state_dict[new_key] = v
            else:
                for k, v in self.text_encoder.state_dict().items():
                    new_key = k if k.startswith(f"{SD_PREFIX_TEXT_ENCODER}") else f"{SD_PREFIX_TEXT_ENCODER}_{k}"
                    state_dict[new_key] = v
        if unet:
            for k, v in self.unet.state_dict().items():
                new_key = k if k.startswith(f"{SD_PREFIX_UNET}") else f"{SD_PREFIX_UNET}_{k}"
                state_dict[new_key] = v
        return state_dict

    def save(self, output_file: str, meta: OrderedDict, save_dtype=get_torch_dtype('fp16'), logit_scale=None):
        state_dict = {}
        # prepare metadata
        meta = get_meta_for_safetensors(meta)

        def update_sd(prefix, sd):
            for k, v in sd.items():
                key = prefix + k
                v = v.detach().clone()
                state_dict[key] = v.to("cpu", dtype=get_torch_dtype(save_dtype))
                # make sure there are not nan values
                if torch.isnan(state_dict[key]).any():
                    raise ValueError(f"NaN value in state dict: {key}")

        # todo see what logit scale is
        if self.is_xl:
            save_ldm_model_from_diffusers(
                sd=self,
                output_file=output_file,
                meta=meta,
                save_dtype=save_dtype,
                sd_version='sdxl',
            )

        else:
            # Convert the UNet model
            unet_state_dict = convert_unet_state_dict_to_sd(self.is_v2, self.unet.state_dict())
            update_sd("model.diffusion_model.", unet_state_dict)

            # Convert the text encoder model
            if self.is_v2:
                make_dummy = True
                text_enc_dict = convert_text_encoder_state_dict_to_sd_v2(self.text_encoder.state_dict(), make_dummy)
                update_sd("cond_stage_model.model.", text_enc_dict)
            else:
                text_enc_dict = self.text_encoder.state_dict()
                update_sd("cond_stage_model.transformer.", text_enc_dict)

                # Convert the VAE
            if self.vae is not None:
                vae_dict = model_util.convert_vae_state_dict(self.vae.state_dict())
                update_sd("first_stage_model.", vae_dict)

            # make sure parent folder exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            save_file(state_dict, output_file, metadata=meta)
