import copy
import gc
import inspect
import json
import random
import shutil
import typing
from typing import Optional, Union, List, Literal
import os
from collections import OrderedDict
import copy
import yaml
from extensions_built_in.dataset_tools.tools.image_tools import load_image
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from torch.nn import Parameter
from tqdm import tqdm
from torchvision.transforms import Resize, transforms

from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.ip_adapter import IPAdapter
from toolkit.config_modules import ModelConfig, GenerateImageConfig, ModelArch
from toolkit.models.decorator import Decorator
from toolkit.paths import KEYMAPS_ROOT
from toolkit.prompt_utils import inject_trigger_into_prompt, PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.sd_device_states_presets import empty_preset
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import torch
from toolkit.pipelines import CustomStableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, T2IAdapter, DDPMScheduler, \
    LCMScheduler, Transformer2DModel, AutoencoderTiny, ControlNetModel
import diffusers
from diffusers import \
    AutoencoderKL, \
    UNet2DConditionModel
from diffusers import PixArtAlphaPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from toolkit.accelerator import get_accelerator, unwrap_model
from typing import TYPE_CHECKING
from toolkit.print import print_acc

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

# tell it to shut up
diffusers.logging.set_verbosity(diffusers.logging.ERROR)

SD_PREFIX_VAE = "vae"
SD_PREFIX_UNET = "unet"
SD_PREFIX_REFINER_UNET = "refiner_unet"
SD_PREFIX_TEXT_ENCODER = "te"

SD_PREFIX_TEXT_ENCODER1 = "te0"
SD_PREFIX_TEXT_ENCODER2 = "te1"

# prefixed diffusers keys
DO_NOT_TRAIN_WEIGHTS = [
    "unet_time_embedding.linear_1.bias",
    "unet_time_embedding.linear_1.weight",
    "unet_time_embedding.linear_2.bias",
    "unet_time_embedding.linear_2.weight",
    "refiner_unet_time_embedding.linear_1.bias",
    "refiner_unet_time_embedding.linear_1.weight",
    "refiner_unet_time_embedding.linear_2.bias",
    "refiner_unet_time_embedding.linear_2.weight",
]

DeviceStatePreset = Literal['cache_latents', 'generate']


class BlankNetwork:

    def __init__(self):
        self.multiplier = 1.0
        self.is_active = True
        self.is_merged_in = False
        self.can_merge_in = False

    def __enter__(self):
        self.is_active = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_active = False

    def train(self):
        pass


def flush():
    torch.cuda.empty_cache()
    gc.collect()


UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
# VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8


class BaseModel:
    # override these in child classes
    arch = None

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='fp16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        self.accelerator = get_accelerator()
        self.custom_pipeline = custom_pipeline
        self.device = device
        self.dtype = dtype
        self.torch_dtype = get_torch_dtype(dtype)
        self.device_torch = torch.device(device)

        self.vae_device_torch = torch.device(device)
        self.vae_torch_dtype = get_torch_dtype(model_config.vae_dtype)

        self.te_device_torch = torch.device(device)
        self.te_torch_dtype = get_torch_dtype(model_config.te_dtype)

        self.model_config = model_config
        self.prediction_type = "v_prediction" if self.model_config.is_v_pred else "epsilon"

        self.device_state = None

        self.pipeline: Union[None, 'StableDiffusionPipeline',
                             'CustomStableDiffusionXLPipeline', 'PixArtAlphaPipeline']
        self.vae: Union[None, 'AutoencoderKL']
        self.model: Union[None, 'Transformer2DModel', 'UNet2DConditionModel']
        self.text_encoder: Union[None, 'CLIPTextModel',
                                 List[Union['CLIPTextModel', 'CLIPTextModelWithProjection']]]
        self.tokenizer: Union[None, 'CLIPTokenizer', List['CLIPTokenizer']]
        self.noise_scheduler: Union[None, 'DDPMScheduler'] = noise_scheduler

        self.refiner_unet: Union[None, 'UNet2DConditionModel'] = None
        self.assistant_lora: Union[None, 'LoRASpecialNetwork'] = None

        # sdxl stuff
        self.logit_scale = None
        self.ckppt_info = None
        self.is_loaded = False

        # to hold network if there is one
        self.network = None
        self.adapter: Union['ControlNetModel', 'T2IAdapter',
                            'IPAdapter', 'ReferenceAdapter', None] = None
        self.decorator: Union[Decorator, None] = None
        self.arch: ModelArch = model_config.arch

        self.use_text_encoder_1 = model_config.use_text_encoder_1
        self.use_text_encoder_2 = model_config.use_text_encoder_2

        self.config_file = None

        self.is_flow_matching = False

        self.quantize_device = self.device_torch
        self.low_vram = self.model_config.low_vram

        # merge in and preview active with -1 weight
        self.invert_assistant_lora = False
        self._after_sample_img_hooks = []
        self._status_update_hooks = []
        self.is_transformer = False

    # properties for old arch for backwards compatibility
    @property
    def unet(self):
        return self.model
    
    # set unet to model
    @unet.setter
    def unet(self, value):
        self.model = value
        
    @property
    def transformer(self):
        return self.model
    
    @transformer.setter
    def transformer(self, value):
        self.model = value

    @property
    def unet_unwrapped(self):
        return unwrap_model(self.model)

    @property
    def model_unwrapped(self):
        return unwrap_model(self.model)

    @property
    def is_xl(self):
        return self.arch == 'sdxl'

    @property
    def is_v2(self):
        return self.arch == 'sd2'

    @property
    def is_ssd(self):
        return self.arch == 'ssd'

    @property
    def is_v3(self):
        return self.arch == 'sd3'

    @property
    def is_vega(self):
        return self.arch == 'vega'

    @property
    def is_pixart(self):
        return self.arch == 'pixart'

    @property
    def is_auraflow(self):
        return self.arch == 'auraflow'

    @property
    def is_flux(self):
        return self.arch == 'flux'

    @property
    def is_lumina2(self):
        return self.arch == 'lumina2'

    def get_bucket_divisibility(self):
        if self.vae is None:
            return 8
        try:
            divisibility = 2 ** (len(self.vae.config['block_out_channels']) - 1)
        except:
            # if we have a custom vae, it might not have this
            divisibility = 8
        
        # flux packs this again,
        if self.is_flux:
            divisibility = divisibility * 2
        return divisibility

    # these must be implemented in child classes
    def load_model(self):
        # override this in child classes
        raise NotImplementedError(
            "load_model must be implemented in child classes")

    def get_generation_pipeline(self):
        # override this in child classes
        raise NotImplementedError(
            "get_generation_pipeline must be implemented in child classes")

    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # override this in child classes
        raise NotImplementedError(
            "generate_single_image must be implemented in child classes")

    def get_noise_prediction(
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        raise NotImplementedError(
            "get_noise_prediction must be implemented in child classes")

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        raise NotImplementedError(
            "get_prompt_embeds must be implemented in child classes")
        
    def get_model_has_grad(self):
        raise NotImplementedError(
            "get_model_has_grad must be implemented in child classes")
    
    def get_te_has_grad(self):
        raise NotImplementedError(
            "get_te_has_grad must be implemented in child classes")
        
    def save_model(self, output_path, meta, save_dtype):
        # todo handle dtype without overloading anything (vram, cpu, etc)
        unwrap_model(self.pipeline).save_pretrained(
            save_directory=output_path,
            safe_serialization=True,
        )
        # save out meta config
        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w', encoding='utf-8') as f:
            yaml.dump(meta, f, allow_unicode=True)
    # end must be implemented in child classes

    def te_train(self):
        if isinstance(self.text_encoder, list):
            for te in self.text_encoder:
                te.train()
        elif self.text_encoder is not None:
            self.text_encoder.train()

    def te_eval(self):
        if isinstance(self.text_encoder, list):
            for te in self.text_encoder:
                te.eval()
        elif self.text_encoder is not None:
            self.text_encoder.eval()

    def _after_sample_image(self, img_num, total_imgs):
        # process all hooks
        for hook in self._after_sample_img_hooks:
            hook(img_num, total_imgs)

    def add_after_sample_image_hook(self, func):
        self._after_sample_img_hooks.append(func)

    def _status_update(self, status: str):
        for hook in self._status_update_hooks:
            hook(status)

    def print_and_status_update(self, status: str):
        print_acc(status)
        self._status_update(status)

    def add_status_update_hook(self, func):
        self._status_update_hooks.append(func)

    @torch.no_grad()
    def generate_images(
            self,
            image_configs: List[GenerateImageConfig],
            sampler=None,
            pipeline: Union[None, StableDiffusionPipeline,
                            StableDiffusionXLPipeline] = None,
    ):
        network = self.network
        merge_multiplier = 1.0
        flush()
        # if using assistant, unfuse it
        if self.model_config.assistant_lora_path is not None:
            print_acc("Unloading assistant lora")
            if self.invert_assistant_lora:
                self.assistant_lora.is_active = True
                # move weights on to the device
                self.assistant_lora.force_to(
                    self.device_torch, self.torch_dtype)
            else:
                self.assistant_lora.is_active = False

        if self.model_config.inference_lora_path is not None:
            print_acc("Loading inference lora")
            self.assistant_lora.is_active = True
            # move weights on to the device
            self.assistant_lora.force_to(self.device_torch, self.torch_dtype)

        if network is not None:
            network = unwrap_model(self.network)
            network.eval()
            # check if we have the same network weight for all samples. If we do, we can merge in th
            # the network to drastically speed up inference
            unique_network_weights = set(
                [x.network_multiplier for x in image_configs])
            if len(unique_network_weights) == 1 and network.can_merge_in:
                can_merge_in = True
                merge_multiplier = unique_network_weights.pop()
                network.merge_in(merge_weight=merge_multiplier)
        else:
            network = BlankNetwork()

        self.save_device_state()
        self.set_device_state_preset('generate')

        # save current seed state for training
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        if pipeline is None:
            pipeline = self.get_generation_pipeline()
            try:
                pipeline.set_progress_bar_config(disable=True)
            except:
                pass

        start_multiplier = 1.0
        if network is not None:
            start_multiplier = network.multiplier

        # pipeline.to(self.device_torch)

        with network:
            with torch.no_grad():
                if network is not None:
                    assert network.is_active

                for i in tqdm(range(len(image_configs)), desc=f"Generating Images", leave=False):
                    gen_config = image_configs[i]

                    extra = {}
                    validation_image = None
                    if self.adapter is not None and gen_config.adapter_image_path is not None:
                        validation_image = load_image(gen_config.adapter_image_path)
                        if ".inpaint." not in gen_config.adapter_image_path:
                            validation_image = validation_image.convert("RGB")
                        else:
                            # make sure it has an alpha
                            if validation_image.mode != "RGBA":
                                raise ValueError("Inpainting images must have an alpha channel")
                        if isinstance(self.adapter, T2IAdapter):
                            # not sure why this is double??
                            validation_image = validation_image.resize(
                                (gen_config.width * 2, gen_config.height * 2))
                            extra['image'] = validation_image
                            extra['adapter_conditioning_scale'] = gen_config.adapter_conditioning_scale
                        if isinstance(self.adapter, ControlNetModel):
                            validation_image = validation_image.resize(
                                (gen_config.width, gen_config.height))
                            extra['image'] = validation_image
                            extra['controlnet_conditioning_scale'] = gen_config.adapter_conditioning_scale
                        if isinstance(self.adapter, CustomAdapter) and self.adapter.control_lora is not None:
                            validation_image = validation_image.resize((gen_config.width, gen_config.height))
                            extra['control_image'] = validation_image
                            extra['control_image_idx'] = gen_config.ctrl_idx
                        if isinstance(self.adapter, IPAdapter) or isinstance(self.adapter, ClipVisionAdapter):
                            transform = transforms.Compose([
                                transforms.ToTensor(),
                            ])
                            validation_image = transform(validation_image)
                        if isinstance(self.adapter, CustomAdapter):
                            # todo allow loading multiple
                            transform = transforms.Compose([
                                transforms.ToTensor(),
                            ])
                            validation_image = transform(validation_image)
                            self.adapter.num_images = 1
                        if isinstance(self.adapter, ReferenceAdapter):
                            # need -1 to 1
                            validation_image = transforms.ToTensor()(validation_image)
                            validation_image = validation_image * 2.0 - 1.0
                            validation_image = validation_image.unsqueeze(0)
                            self.adapter.set_reference_images(validation_image)

                    if network is not None:
                        network.multiplier = gen_config.network_multiplier
                    torch.manual_seed(gen_config.seed)
                    torch.cuda.manual_seed(gen_config.seed)

                    generator = torch.manual_seed(gen_config.seed)

                    if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter) \
                            and gen_config.adapter_image_path is not None:
                        # run through the adapter to saturate the embeds
                        conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                            validation_image)
                        self.adapter(conditional_clip_embeds)

                    if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
                        # handle condition the prompts
                        gen_config.prompt = self.adapter.condition_prompt(
                            gen_config.prompt,
                            is_unconditional=False,
                        )
                        gen_config.prompt_2 = gen_config.prompt
                        gen_config.negative_prompt = self.adapter.condition_prompt(
                            gen_config.negative_prompt,
                            is_unconditional=True,
                        )
                        gen_config.negative_prompt_2 = gen_config.negative_prompt

                    if self.adapter is not None and isinstance(self.adapter, CustomAdapter) and validation_image is not None:
                        self.adapter.trigger_pre_te(
                            tensors_0_1=validation_image,
                            is_training=False,
                            has_been_preprocessed=False,
                            quad_count=4
                        )

                    # encode the prompt ourselves so we can do fun stuff with embeddings
                    if isinstance(self.adapter, CustomAdapter):
                        self.adapter.is_unconditional_run = False
                    conditional_embeds = self.encode_prompt(
                        gen_config.prompt, gen_config.prompt_2, force_all=True)

                    if isinstance(self.adapter, CustomAdapter):
                        self.adapter.is_unconditional_run = True
                    unconditional_embeds = self.encode_prompt(
                        gen_config.negative_prompt, gen_config.negative_prompt_2, force_all=True
                    )
                    if isinstance(self.adapter, CustomAdapter):
                        self.adapter.is_unconditional_run = False

                    # allow any manipulations to take place to embeddings
                    gen_config.post_process_embeddings(
                        conditional_embeds,
                        unconditional_embeds,
                    )

                    if self.decorator is not None:
                        # apply the decorator to the embeddings
                        conditional_embeds.text_embeds = self.decorator(
                            conditional_embeds.text_embeds)
                        unconditional_embeds.text_embeds = self.decorator(
                            unconditional_embeds.text_embeds, is_unconditional=True)

                    if self.adapter is not None and isinstance(self.adapter, IPAdapter) \
                            and gen_config.adapter_image_path is not None:
                        # apply the image projection
                        conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                            validation_image)
                        unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(validation_image,
                                                                                                    True)
                        conditional_embeds = self.adapter(
                            conditional_embeds, conditional_clip_embeds, is_unconditional=False)
                        unconditional_embeds = self.adapter(
                            unconditional_embeds, unconditional_clip_embeds, is_unconditional=True)

                    if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
                        conditional_embeds = self.adapter.condition_encoded_embeds(
                            tensors_0_1=validation_image,
                            prompt_embeds=conditional_embeds,
                            is_training=False,
                            has_been_preprocessed=False,
                            is_generating_samples=True,
                        )
                        unconditional_embeds = self.adapter.condition_encoded_embeds(
                            tensors_0_1=validation_image,
                            prompt_embeds=unconditional_embeds,
                            is_training=False,
                            has_been_preprocessed=False,
                            is_unconditional=True,
                            is_generating_samples=True,
                        )

                    if self.adapter is not None and isinstance(self.adapter, CustomAdapter) and len(
                            gen_config.extra_values) > 0:
                        extra_values = torch.tensor([gen_config.extra_values], device=self.device_torch,
                                                    dtype=self.torch_dtype)
                        # apply extra values to the embeddings
                        self.adapter.add_extra_values(
                            extra_values, is_unconditional=False)
                        self.adapter.add_extra_values(torch.zeros_like(
                            extra_values), is_unconditional=True)
                        pass  # todo remove, for debugging

                    if self.refiner_unet is not None and gen_config.refiner_start_at < 1.0:
                        # if we have a refiner loaded, set the denoising end at the refiner start
                        extra['denoising_end'] = gen_config.refiner_start_at
                        extra['output_type'] = 'latent'
                        if not self.is_xl:
                            raise ValueError(
                                "Refiner is only supported for XL models")

                    conditional_embeds = conditional_embeds.to(
                        self.device_torch, dtype=self.unet.dtype)
                    unconditional_embeds = unconditional_embeds.to(
                        self.device_torch, dtype=self.unet.dtype)

                    img = self.generate_single_image(
                        pipeline,
                        gen_config,
                        conditional_embeds,
                        unconditional_embeds,
                        generator,
                        extra,
                    )

                    gen_config.save_image(img, i)
                    gen_config.log_image(img, i)
                    self._after_sample_image(i, len(image_configs))
                    flush()

                if self.adapter is not None and isinstance(self.adapter, ReferenceAdapter):
                    self.adapter.clear_memory()

        # clear pipeline and cache to reduce vram usage
        del pipeline
        torch.cuda.empty_cache()

        # restore training state
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        self.restore_device_state()
        if network is not None:
            network.train()
            network.multiplier = start_multiplier

        self.unet.to(self.device_torch, dtype=self.torch_dtype)
        if network.is_merged_in:
            network.merge_out(merge_multiplier)
        # self.tokenizer.to(original_device_dict['tokenizer'])

        # refuse loras
        if self.model_config.assistant_lora_path is not None:
            print_acc("Loading assistant lora")
            if self.invert_assistant_lora:
                self.assistant_lora.is_active = False
                # move weights off the device
                self.assistant_lora.force_to('cpu', self.torch_dtype)
            else:
                self.assistant_lora.is_active = True

        if self.model_config.inference_lora_path is not None:
            print_acc("Unloading inference lora")
            self.assistant_lora.is_active = False
            # move weights off the device
            self.assistant_lora.force_to('cpu', self.torch_dtype)
        flush()

    def get_latent_noise(
            self,
            height=None,
            width=None,
            pixel_height=None,
            pixel_width=None,
            batch_size=1,
            noise_offset=0.0,
    ):
        VAE_SCALE_FACTOR = 2 ** (
            len(self.vae.config['block_out_channels']) - 1)
        if height is None and pixel_height is None:
            raise ValueError("height or pixel_height must be specified")
        if width is None and pixel_width is None:
            raise ValueError("width or pixel_width must be specified")
        if height is None:
            height = pixel_height // VAE_SCALE_FACTOR
        if width is None:
            width = pixel_width // VAE_SCALE_FACTOR

        num_channels = self.unet_unwrapped.config['in_channels']
        if self.is_flux:
            # has 64 channels in for some reason
            num_channels = 16
        noise = torch.randn(
            (
                batch_size,
                num_channels,
                height,
                width,
            ),
            device=self.unet.device,
        )
        noise = apply_noise_offset(noise, noise_offset)
        return noise
    
    def get_latent_noise_from_latents(
        self,
        latents: torch.Tensor,
        noise_offset=0.0
    ):
        noise = torch.randn_like(latents)
        noise = apply_noise_offset(noise, noise_offset)
        return noise

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
            **kwargs,
    ) -> torch.FloatTensor:
        original_samples_chunks = torch.chunk(
            original_samples, original_samples.shape[0], dim=0)
        noise_chunks = torch.chunk(noise, noise.shape[0], dim=0)
        timesteps_chunks = torch.chunk(timesteps, timesteps.shape[0], dim=0)

        if len(timesteps_chunks) == 1 and len(timesteps_chunks) != len(original_samples_chunks):
            timesteps_chunks = [timesteps_chunks[0]] * \
                len(original_samples_chunks)

        noisy_latents_chunks = []

        for idx in range(original_samples.shape[0]):
            noisy_latents = self.noise_scheduler.add_noise(original_samples_chunks[idx], noise_chunks[idx],
                                                           timesteps_chunks[idx])
            noisy_latents_chunks.append(noisy_latents)

        noisy_latents = torch.cat(noisy_latents_chunks, dim=0)
        return noisy_latents

    def predict_noise(
            self,
            latents: torch.Tensor,
            text_embeddings: Union[PromptEmbeds, None] = None,
            timestep: Union[int, torch.Tensor] = 1,
            guidance_scale=7.5,
            guidance_rescale=0,
            add_time_ids=None,
            conditional_embeddings: Union[PromptEmbeds, None] = None,
            unconditional_embeddings: Union[PromptEmbeds, None] = None,
            is_input_scaled=False,
            detach_unconditional=False,
            rescale_cfg=None,
            return_conditional_pred=False,
            guidance_embedding_scale=1.0,
            bypass_guidance_embedding=False,
            batch: Union[None, 'DataLoaderBatchDTO'] = None,
            **kwargs,
    ):
        conditional_pred = None
        # get the embeddings
        if text_embeddings is None and conditional_embeddings is None:
            raise ValueError(
                "Either text_embeddings or conditional_embeddings must be specified")
        if text_embeddings is None and unconditional_embeddings is not None:
            text_embeddings = concat_prompt_embeds([
                unconditional_embeddings,  # negative embedding
                conditional_embeddings,  # positive embedding
            ])
        elif text_embeddings is None and conditional_embeddings is not None:
            # not doing cfg
            text_embeddings = conditional_embeddings

        # CFG is comparing neg and positive, if we have concatenated embeddings
        # then we are doing it, otherwise we are not and takes half the time.
        do_classifier_free_guidance = True

        # check if batch size of embeddings matches batch size of latents
        if isinstance(text_embeddings.text_embeds, list):
            te_batch_size = text_embeddings.text_embeds[0].shape[0]
        else:
            te_batch_size = text_embeddings.text_embeds.shape[0]
        if latents.shape[0] == te_batch_size:
            do_classifier_free_guidance = False
        elif latents.shape[0] * 2 != te_batch_size:
            raise ValueError(
                "Batch size of latents must be the same or half the batch size of text embeddings")
        latents = latents.to(self.device_torch)
        text_embeddings = text_embeddings.to(self.device_torch)
        timestep = timestep.to(self.device_torch)

        # if timestep is zero dim, unsqueeze it
        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)

        # if we only have 1 timestep, we can just use the same timestep for all
        if timestep.shape[0] == 1 and latents.shape[0] > 1:
            # check if it is rank 1 or 2
            if len(timestep.shape) == 1:
                timestep = timestep.repeat(latents.shape[0])
            else:
                timestep = timestep.repeat(latents.shape[0], 0)

        # handle t2i adapters
        if 'down_intrablock_additional_residuals' in kwargs:
            # go through each item and concat if doing cfg and it doesnt have the same shape
            for idx, item in enumerate(kwargs['down_intrablock_additional_residuals']):
                if do_classifier_free_guidance and item.shape[0] != text_embeddings.text_embeds.shape[0]:
                    kwargs['down_intrablock_additional_residuals'][idx] = torch.cat([
                                                                                    item] * 2, dim=0)

        # handle controlnet
        if 'down_block_additional_residuals' in kwargs and 'mid_block_additional_residual' in kwargs:
            # go through each item and concat if doing cfg and it doesnt have the same shape
            for idx, item in enumerate(kwargs['down_block_additional_residuals']):
                if do_classifier_free_guidance and item.shape[0] != text_embeddings.text_embeds.shape[0]:
                    kwargs['down_block_additional_residuals'][idx] = torch.cat([
                                                                               item] * 2, dim=0)
            for idx, item in enumerate(kwargs['mid_block_additional_residual']):
                if do_classifier_free_guidance and item.shape[0] != text_embeddings.text_embeds.shape[0]:
                    kwargs['mid_block_additional_residual'][idx] = torch.cat(
                        [item] * 2, dim=0)

        def scale_model_input(model_input, timestep_tensor):
            if is_input_scaled:
                return model_input
            mi_chunks = torch.chunk(model_input, model_input.shape[0], dim=0)
            timestep_chunks = torch.chunk(
                timestep_tensor, timestep_tensor.shape[0], dim=0)
            out_chunks = []
            # unsqueeze if timestep is zero dim
            for idx in range(model_input.shape[0]):
                # if scheduler has step_index
                if hasattr(self.noise_scheduler, '_step_index'):
                    self.noise_scheduler._step_index = None
                out_chunks.append(
                    self.noise_scheduler.scale_model_input(
                        mi_chunks[idx], timestep_chunks[idx])
                )
            return torch.cat(out_chunks, dim=0)

        with torch.no_grad():
            if do_classifier_free_guidance:
                # if we are doing classifier free guidance, need to double up
                latent_model_input = torch.cat([latents] * 2, dim=0)
                timestep = torch.cat([timestep] * 2)
            else:
                latent_model_input = latents

            latent_model_input = scale_model_input(
                latent_model_input, timestep)

            # check if we need to concat timesteps
            if isinstance(timestep, torch.Tensor) and len(timestep.shape) > 1:
                ts_bs = timestep.shape[0]
                if ts_bs != latent_model_input.shape[0]:
                    if ts_bs == 1:
                        timestep = torch.cat(
                            [timestep] * latent_model_input.shape[0])
                    elif ts_bs * 2 == latent_model_input.shape[0]:
                        timestep = torch.cat([timestep] * 2, dim=0)
                    else:
                        raise ValueError(
                            f"Batch size of latents {latent_model_input.shape[0]} must be the same or half the batch size of timesteps {timestep.shape[0]}")

        # predict the noise residual
        if self.unet.device != self.device_torch:
            self.unet.to(self.device_torch)
        if self.unet.dtype != self.torch_dtype:
            self.unet = self.unet.to(dtype=self.torch_dtype)
            
        # check if get_noise prediction has guidance_embedding_scale
        # if it does not, we dont pass it
        signatures =  inspect.signature(self.get_noise_prediction).parameters
        
        if 'guidance_embedding_scale' in signatures:
            kwargs['guidance_embedding_scale'] = guidance_embedding_scale
        if 'bypass_guidance_embedding' in signatures:
            kwargs['bypass_guidance_embedding'] = bypass_guidance_embedding
        if 'batch' in signatures:
            kwargs['batch'] = batch

        noise_pred = self.get_noise_prediction(
            latent_model_input=latent_model_input,
            timestep=timestep,
            text_embeddings=text_embeddings,
            **kwargs
        )

        conditional_pred = noise_pred

        if do_classifier_free_guidance:
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            conditional_pred = noise_pred_text
            if detach_unconditional:
                noise_pred_uncond = noise_pred_uncond.detach()
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            if rescale_cfg is not None and rescale_cfg != guidance_scale:
                with torch.no_grad():
                    # do cfg at the target rescale so we can match it
                    target_pred_mean_std = noise_pred_uncond + rescale_cfg * (
                        noise_pred_text - noise_pred_uncond
                    )
                    target_mean = target_pred_mean_std.mean(
                        [1, 2, 3], keepdim=True).detach()
                    target_std = target_pred_mean_std.std(
                        [1, 2, 3], keepdim=True).detach()

                    pred_mean = noise_pred.mean(
                        [1, 2, 3], keepdim=True).detach()
                    pred_std = noise_pred.std([1, 2, 3], keepdim=True).detach()

                # match the mean and std
                noise_pred = (noise_pred - pred_mean) / pred_std
                noise_pred = (noise_pred * target_std) + target_mean

            # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
            if guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        if return_conditional_pred:
            return noise_pred, conditional_pred
        return noise_pred

    def step_scheduler(self, model_input, latent_input, timestep_tensor, noise_scheduler=None):
        if noise_scheduler is None:
            noise_scheduler = self.noise_scheduler
        # // sometimes they are on the wrong device, no idea why
        if isinstance(noise_scheduler, DDPMScheduler) or isinstance(noise_scheduler, LCMScheduler):
            try:
                noise_scheduler.betas = noise_scheduler.betas.to(
                    self.device_torch)
                noise_scheduler.alphas = noise_scheduler.alphas.to(
                    self.device_torch)
                noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                    self.device_torch)
            except Exception as e:
                pass

        mi_chunks = torch.chunk(model_input, model_input.shape[0], dim=0)
        latent_chunks = torch.chunk(latent_input, latent_input.shape[0], dim=0)
        timestep_chunks = torch.chunk(
            timestep_tensor, timestep_tensor.shape[0], dim=0)
        out_chunks = []
        if len(timestep_chunks) == 1 and len(mi_chunks) > 1:
            # expand timestep to match
            timestep_chunks = timestep_chunks * len(mi_chunks)

        for idx in range(model_input.shape[0]):
            # Reset it so it is unique for the
            if hasattr(noise_scheduler, '_step_index'):
                noise_scheduler._step_index = None
            if hasattr(noise_scheduler, 'is_scale_input_called'):
                noise_scheduler.is_scale_input_called = True
            out_chunks.append(
                noise_scheduler.step(mi_chunks[idx], timestep_chunks[idx], latent_chunks[idx], return_dict=False)[
                    0]
            )
        return torch.cat(out_chunks, dim=0)

    # ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
    def diffuse_some_steps(
            self,
            latents: torch.FloatTensor,
            text_embeddings: PromptEmbeds,
            total_timesteps: int = 1000,
            start_timesteps=0,
            guidance_scale=1,
            add_time_ids=None,
            bleed_ratio: float = 0.5,
            bleed_latents: torch.FloatTensor = None,
            is_input_scaled=False,
            return_first_prediction=False,
            **kwargs,
    ):
        timesteps_to_run = self.noise_scheduler.timesteps[start_timesteps:total_timesteps]

        first_prediction = None

        for timestep in tqdm(timesteps_to_run, leave=False):
            timestep = timestep.unsqueeze_(0)
            noise_pred, conditional_pred = self.predict_noise(
                latents,
                text_embeddings,
                timestep,
                guidance_scale=guidance_scale,
                add_time_ids=add_time_ids,
                is_input_scaled=is_input_scaled,
                return_conditional_pred=True,
                **kwargs,
            )
            # some schedulers need to run separately, so do that. (euler for example)

            if return_first_prediction and first_prediction is None:
                first_prediction = conditional_pred

            latents = self.step_scheduler(noise_pred, latents, timestep)

            # if not last step, and bleeding, bleed in some latents
            if bleed_latents is not None and timestep != self.noise_scheduler.timesteps[-1]:
                latents = (latents * (1 - bleed_ratio)) + \
                    (bleed_latents * bleed_ratio)

            # only skip first scaling
            is_input_scaled = False

        # return latents_steps
        if return_first_prediction:
            return latents, first_prediction
        return latents

    def encode_prompt(
            self,
            prompt,
            prompt2=None,
            num_images_per_prompt=1,
            force_all=False,
            long_prompts=False,
            max_length=None,
            dropout_prob=0.0,
    ) -> PromptEmbeds:
        # sd1.5 embeddings are (bs, 77, 768)
        prompt = prompt
        # if it is not a list, make it one
        if not isinstance(prompt, list):
            prompt = [prompt]

        if prompt2 is not None and not isinstance(prompt2, list):
            prompt2 = [prompt2]

        return self.get_prompt_embeds(prompt)

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

        latent_list = []
        # Move to vae to device if on cpu
        if self.vae.device == 'cpu':
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        # move to device and dtype
        image_list = [image.to(device, dtype=dtype) for image in image_list]

        VAE_SCALE_FACTOR = 2 ** (
            len(self.vae.config['block_out_channels']) - 1)

        # resize images if not divisible by 8
        for i in range(len(image_list)):
            image = image_list[i]
            if image.shape[1] % VAE_SCALE_FACTOR != 0 or image.shape[2] % VAE_SCALE_FACTOR != 0:
                image_list[i] = Resize((image.shape[1] // VAE_SCALE_FACTOR * VAE_SCALE_FACTOR,
                                        image.shape[2] // VAE_SCALE_FACTOR * VAE_SCALE_FACTOR))(image)

        images = torch.stack(image_list)
        if isinstance(self.vae, AutoencoderTiny):
            latents = self.vae.encode(images, return_dict=False)[0]
        else:
            latents = self.vae.encode(images).latent_dist.sample()
        shift = self.vae.config['shift_factor'] if self.vae.config['shift_factor'] is not None else 0

        # flux ref https://github.com/black-forest-labs/flux/blob/c23ae247225daba30fbd56058d247cc1b1fc20a3/src/flux/modules/autoencoder.py#L303
        # z = self.scale_factor * (z - self.shift_factor)
        latents = self.vae.config['scaling_factor'] * (latents - shift)
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
        latents = (
            latents / self.vae.config['scaling_factor']) + self.vae.config['shift_factor']
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
            embedding = self.encode_prompt(prompt).to(
                self.device_torch, dtype=dtype)
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
                new_key = k if k.startswith(
                    f"{SD_PREFIX_VAE}") else f"{SD_PREFIX_VAE}_{k}"
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
                    new_key = k if k.startswith(
                        f"{SD_PREFIX_TEXT_ENCODER}_") else f"{SD_PREFIX_TEXT_ENCODER}_{k}"
                    state_dict[new_key] = v
        if unet:
            for k, v in self.unet.state_dict().items():
                new_key = k if k.startswith(
                    f"{SD_PREFIX_UNET}_") else f"{SD_PREFIX_UNET}_{k}"
                state_dict[new_key] = v
        return state_dict

    def named_parameters(self, vae=True, text_encoder=True, unet=True, refiner=False, state_dict_keys=False) -> \
            OrderedDict[
                str, Parameter]:
        named_params: OrderedDict[str, Parameter] = OrderedDict()
        if vae:
            for name, param in self.vae.named_parameters(recurse=True, prefix=f"{SD_PREFIX_VAE}"):
                named_params[name] = param
        if text_encoder:
            if isinstance(self.text_encoder, list):
                for i, encoder in enumerate(self.text_encoder):
                    if self.is_xl and not self.model_config.use_text_encoder_1 and i == 0:
                        # dont add these params
                        continue
                    if self.is_xl and not self.model_config.use_text_encoder_2 and i == 1:
                        # dont add these params
                        continue

                    for name, param in encoder.named_parameters(recurse=True, prefix=f"{SD_PREFIX_TEXT_ENCODER}{i}"):
                        named_params[name] = param
            else:
                for name, param in self.text_encoder.named_parameters(recurse=True, prefix=f"{SD_PREFIX_TEXT_ENCODER}"):
                    named_params[name] = param
        if unet:
            if self.is_flux or self.is_lumina2 or self.is_transformer:
                for name, param in self.unet.named_parameters(recurse=True, prefix="transformer"):
                    named_params[name] = param
            else:
                for name, param in self.unet.named_parameters(recurse=True, prefix=f"{SD_PREFIX_UNET}"):
                    named_params[name] = param

            if self.model_config.ignore_if_contains is not None:
                # remove params that contain the ignore_if_contains from named params
                for key in list(named_params.keys()):
                    if any([s in f"transformer.{key}" for s in self.model_config.ignore_if_contains]):
                        del named_params[key]
            if self.model_config.only_if_contains is not None:
                # remove params that do not contain the only_if_contains from named params
                for key in list(named_params.keys()):
                    if not any([s in f"transformer.{key}" for s in self.model_config.only_if_contains]):
                        del named_params[key]

        if refiner:
            for name, param in self.refiner_unet.named_parameters(recurse=True, prefix=f"{SD_PREFIX_REFINER_UNET}"):
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
        self.save_model(
            output_path=output_file,
            meta=meta,
            save_dtype=save_dtype
        )

    def prepare_optimizer_params(
            self,
            unet=False,
            text_encoder=False,
            text_encoder_lr=None,
            unet_lr=None,
            refiner_lr=None,
            refiner=False,
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
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        ldm_diffusers_keymap = mapping['ldm_diffusers_keymap']

        trainable_parameters = []

        # we use state dict to find params

        if unet:
            named_params = self.named_parameters(
                vae=False, unet=unet, text_encoder=False, state_dict_keys=True)
            unet_lr = unet_lr if unet_lr is not None else default_lr
            params = []
            for param in named_params.values():
                if param.requires_grad:
                    params.append(param)
           
            param_data = {"params": params, "lr": unet_lr}
            trainable_parameters.append(param_data)
            print_acc(f"Found {len(params)} trainable parameter in unet")

        if text_encoder:
            named_params = self.named_parameters(
                vae=False, unet=False, text_encoder=text_encoder, state_dict_keys=True)
            text_encoder_lr = text_encoder_lr if text_encoder_lr is not None else default_lr
            params = []
            for key, diffusers_key in ldm_diffusers_keymap.items():
                if diffusers_key in named_params and diffusers_key not in DO_NOT_TRAIN_WEIGHTS:
                    if named_params[diffusers_key].requires_grad:
                        params.append(named_params[diffusers_key])
            param_data = {"params": params, "lr": text_encoder_lr}
            trainable_parameters.append(param_data)

            print_acc(
                f"Found {len(params)} trainable parameter in text encoder")

        if refiner:
            named_params = self.named_parameters(vae=False, unet=False, text_encoder=False, refiner=True,
                                                 state_dict_keys=True)
            refiner_lr = refiner_lr if refiner_lr is not None else default_lr
            params = []
            for key, diffusers_key in ldm_diffusers_keymap.items():
                diffusers_key = f"refiner_{diffusers_key}"
                if diffusers_key in named_params and diffusers_key not in DO_NOT_TRAIN_WEIGHTS:
                    if named_params[diffusers_key].requires_grad:
                        params.append(named_params[diffusers_key])
            param_data = {"params": params, "lr": refiner_lr}
            trainable_parameters.append(param_data)

            print_acc(f"Found {len(params)} trainable parameter in refiner")

        return trainable_parameters

    def save_device_state(self):
        # saves the current device state for all modules
        # this is useful for when we want to alter the state and restore it
        unet_has_grad = self.get_model_has_grad()

        self.device_state = {
            **empty_preset,
            'vae': {
                'training': self.vae.training,
                'device': self.vae.device,
            },
            'unet': {
                'training': self.unet.training,
                'device': self.unet.device,
                'requires_grad': unet_has_grad,
            },
        }
        if isinstance(self.text_encoder, list):
            self.device_state['text_encoder']: List[dict] = []
            for encoder in self.text_encoder:
                te_has_grad = self.get_te_has_grad()
                self.device_state['text_encoder'].append({
                    'training': encoder.training,
                    'device': encoder.device,
                    # todo there has to be a better way to do this
                    'requires_grad': te_has_grad
                })
        else:
            te_has_grad = self.get_te_has_grad()

            self.device_state['text_encoder'] = {
                'training': self.text_encoder.training,
                'device': self.text_encoder.device,
                'requires_grad': te_has_grad
            }
        if self.adapter is not None:
            if isinstance(self.adapter, IPAdapter):
                requires_grad = self.adapter.image_proj_model.training
                adapter_device = self.unet.device
            elif isinstance(self.adapter, T2IAdapter):
                requires_grad = self.adapter.adapter.conv_in.weight.requires_grad
                adapter_device = self.adapter.device
            elif isinstance(self.adapter, ControlNetModel):
                requires_grad = self.adapter.conv_in.training
                adapter_device = self.adapter.device
            elif isinstance(self.adapter, ClipVisionAdapter):
                requires_grad = self.adapter.embedder.training
                adapter_device = self.adapter.device
            elif isinstance(self.adapter, CustomAdapter):
                requires_grad = self.adapter.training
                adapter_device = self.adapter.device
            elif isinstance(self.adapter, ReferenceAdapter):
                # todo update this!!
                requires_grad = True
                adapter_device = self.adapter.device
            else:
                raise ValueError(f"Unknown adapter type: {type(self.adapter)}")
            self.device_state['adapter'] = {
                'training': self.adapter.training,
                'device': adapter_device,
                'requires_grad': requires_grad,
            }

        if self.refiner_unet is not None:
            self.device_state['refiner_unet'] = {
                'training': self.refiner_unet.training,
                'device': self.refiner_unet.device,
                'requires_grad': self.refiner_unet.conv_in.weight.requires_grad,
            }

    def restore_device_state(self):
        # restores the device state for all modules
        # this is useful for when we want to alter the state and restore it
        if self.device_state is None:
            return
        self.set_device_state(self.device_state)
        self.device_state = None

    def set_device_state(self, state):
        if state['vae']['training']:
            self.vae.train()
        else:
            self.vae.eval()
        self.vae.to(state['vae']['device'])
        if state['unet']['training']:
            self.unet.train()
        else:
            self.unet.eval()
        self.unet.to(state['unet']['device'])
        if state['unet']['requires_grad']:
            self.unet.requires_grad_(True)
        else:
            self.unet.requires_grad_(False)
        if isinstance(self.text_encoder, list):
            for i, encoder in enumerate(self.text_encoder):
                if isinstance(state['text_encoder'], list):
                    if state['text_encoder'][i]['training']:
                        encoder.train()
                    else:
                        encoder.eval()
                    encoder.to(state['text_encoder'][i]['device'])
                    encoder.requires_grad_(
                        state['text_encoder'][i]['requires_grad'])
                else:
                    if state['text_encoder']['training']:
                        encoder.train()
                    else:
                        encoder.eval()
                    encoder.to(state['text_encoder']['device'])
                    encoder.requires_grad_(
                        state['text_encoder']['requires_grad'])
        else:
            if state['text_encoder']['training']:
                self.text_encoder.train()
            else:
                self.text_encoder.eval()
            self.text_encoder.to(state['text_encoder']['device'])
            self.text_encoder.requires_grad_(
                state['text_encoder']['requires_grad'])

        if self.adapter is not None:
            self.adapter.to(state['adapter']['device'])
            self.adapter.requires_grad_(state['adapter']['requires_grad'])
            if state['adapter']['training']:
                self.adapter.train()
            else:
                self.adapter.eval()

        if self.refiner_unet is not None:
            self.refiner_unet.to(state['refiner_unet']['device'])
            self.refiner_unet.requires_grad_(
                state['refiner_unet']['requires_grad'])
            if state['refiner_unet']['training']:
                self.refiner_unet.train()
            else:
                self.refiner_unet.eval()
        flush()

    def set_device_state_preset(self, device_state_preset: DeviceStatePreset):
        # sets a preset for device state

        # save current state first
        self.save_device_state()

        active_modules = []
        training_modules = []
        if device_state_preset in ['cache_latents']:
            active_modules = ['vae']
        if device_state_preset in ['cache_clip']:
            active_modules = ['clip']
        if device_state_preset in ['generate']:
            active_modules = ['vae', 'unet',
                              'text_encoder', 'adapter', 'refiner_unet']

        state = copy.deepcopy(empty_preset)
        # vae
        state['vae'] = {
            'training': 'vae' in training_modules,
            'device': self.vae_device_torch if 'vae' in active_modules else 'cpu',
            'requires_grad': 'vae' in training_modules,
        }

        # unet
        state['unet'] = {
            'training': 'unet' in training_modules,
            'device': self.device_torch if 'unet' in active_modules else 'cpu',
            'requires_grad': 'unet' in training_modules,
        }

        if self.refiner_unet is not None:
            state['refiner_unet'] = {
                'training': 'refiner_unet' in training_modules,
                'device': self.device_torch if 'refiner_unet' in active_modules else 'cpu',
                'requires_grad': 'refiner_unet' in training_modules,
            }

        # text encoder
        if isinstance(self.text_encoder, list):
            state['text_encoder'] = []
            for i, encoder in enumerate(self.text_encoder):
                state['text_encoder'].append({
                    'training': 'text_encoder' in training_modules,
                    'device': self.te_device_torch if 'text_encoder' in active_modules else 'cpu',
                    'requires_grad': 'text_encoder' in training_modules,
                })
        else:
            state['text_encoder'] = {
                'training': 'text_encoder' in training_modules,
                'device': self.te_device_torch if 'text_encoder' in active_modules else 'cpu',
                'requires_grad': 'text_encoder' in training_modules,
            }

        if self.adapter is not None:
            state['adapter'] = {
                'training': 'adapter' in training_modules,
                'device': self.device_torch if 'adapter' in active_modules else 'cpu',
                'requires_grad': 'adapter' in training_modules,
            }

        self.set_device_state(state)

    def text_encoder_to(self, *args, **kwargs):
        if isinstance(self.text_encoder, list):
            for encoder in self.text_encoder:
                encoder.to(*args, **kwargs)
        else:
            self.text_encoder.to(*args, **kwargs)
    
    def convert_lora_weights_before_save(self, state_dict):
        # can be overridden in child classes to convert weights before saving
        return state_dict
    
    def convert_lora_weights_before_load(self, state_dict):
        # can be overridden in child classes to convert weights before loading
        return state_dict
    
    def condition_noisy_latents(self, latents: torch.Tensor, batch:'DataLoaderBatchDTO'):
        # can be overridden in child classes to condition latents before noise prediction
        return latents
    
    def get_transformer_block_names(self) -> Optional[List[str]]:
        # override in child classes to get transformer block names for lora targeting
        return None
    
    def get_base_model_version(self) -> str:
        # override in child classes to get the base model version
        return "unknown"
