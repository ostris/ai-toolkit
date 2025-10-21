import copy
import gc
import json
import random
import shutil
import typing
from typing import Optional, Union, List, Literal, Iterator
import sys
import os
from collections import OrderedDict
import copy
import yaml
from PIL import Image
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_1024_BIN, ASPECT_RATIO_512_BIN, \
    ASPECT_RATIO_2048_BIN, ASPECT_RATIO_256_BIN
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from safetensors.torch import save_file, load_file
from torch import autocast
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from torchvision.transforms import Resize, transforms

from toolkit.assistant_lora import load_assistant_lora_from_path
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.ip_adapter import IPAdapter
from toolkit.util.vae import load_vae
from toolkit import train_tools
from toolkit.config_modules import ModelConfig, GenerateImageConfig, ModelArch
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.decorator import Decorator
from toolkit.paths import KEYMAPS_ROOT
from toolkit.prompt_utils import inject_trigger_into_prompt, PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.sampler import get_sampler
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.saving import save_ldm_model_from_diffusers, get_ldm_state_dict_from_diffusers
from toolkit.sd_device_states_presets import empty_preset
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
from einops import rearrange, repeat
import torch
from toolkit.pipelines import CustomStableDiffusionXLPipeline, CustomStableDiffusionPipeline, \
    StableDiffusionKDiffusionXLPipeline, StableDiffusionXLRefinerPipeline, FluxWithCFGPipeline, \
    FluxAdvancedControlPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, T2IAdapter, DDPMScheduler, \
    StableDiffusionXLAdapterPipeline, StableDiffusionAdapterPipeline, DiffusionPipeline, PixArtTransformer2DModel, \
    StableDiffusionXLImg2ImgPipeline, LCMScheduler, Transformer2DModel, AutoencoderTiny, ControlNetModel, \
    StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, StableDiffusion3Pipeline, \
    StableDiffusion3Img2ImgPipeline, PixArtSigmaPipeline, AuraFlowPipeline, AuraFlowTransformer2DModel, FluxPipeline, \
    FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, Lumina2Pipeline, \
    FluxControlPipeline, Lumina2Transformer2DModel
import diffusers
from diffusers import \
    AutoencoderKL, \
    UNet2DConditionModel
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler, PixArtSigmaPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig, UMT5EncoderModel, T5TokenizerFast
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from toolkit.paths import ORIG_CONFIGS_ROOT, DIFFUSERS_CONFIGS_ROOT
from huggingface_hub import hf_hub_download
from toolkit.models.flux import add_model_gpu_splitter_to_flux, bypass_flux_guidance, restore_flux_guidance

from optimum.quanto import freeze, qfloat8, QTensor, qint4
from toolkit.util.quantize import quantize, get_qtype
from toolkit.accelerator import get_accelerator, unwrap_model
from typing import TYPE_CHECKING
from toolkit.print import print_acc
from diffusers import FluxFillPipeline
from transformers import AutoModel, AutoTokenizer, Gemma2Model, Qwen2Model, LlamaModel

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork

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



class StableDiffusion:

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='fp16',
            custom_pipeline=None,
            noise_scheduler=None,
            quantize_device=None,
    ):
        self.accelerator = get_accelerator()
        self.custom_pipeline = custom_pipeline
        self.device = str(device)
        if "cuda" in self.device and ":" not in self.device:
            self.device = f"{self.device}:0"
        self.device_torch = torch.device(device)
        self.dtype = dtype
        self.torch_dtype = get_torch_dtype(dtype)

        self.vae_device_torch = torch.device(device)
        self.vae_torch_dtype = get_torch_dtype(model_config.vae_dtype)

        self.te_device_torch = torch.device(device)
        self.te_torch_dtype = get_torch_dtype(model_config.te_dtype)

        self.model_config = model_config
        self.prediction_type = "v_prediction" if self.model_config.is_v_pred else "epsilon"
        self.arch = model_config.arch

        self.device_state = None

        self.pipeline: Union[None, 'StableDiffusionPipeline', 'CustomStableDiffusionXLPipeline', 'PixArtAlphaPipeline']
        self.vae: Union[None, 'AutoencoderKL']
        self.unet: Union[None, 'UNet2DConditionModel']
        self.text_encoder: Union[None, 'CLIPTextModel', List[Union['CLIPTextModel', 'CLIPTextModelWithProjection']]]
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
        self.adapter: Union['ControlNetModel', 'T2IAdapter', 'IPAdapter', 'ReferenceAdapter', None] = None
        self.decorator: Union[Decorator, None] = None
        self.arch: ModelArch = model_config.arch
        # self.is_xl = model_config.is_xl
        # self.is_v2 = model_config.is_v2
        # self.is_ssd = model_config.is_ssd
        # self.is_v3 = model_config.is_v3
        # self.is_vega = model_config.is_vega
        # self.is_pixart = model_config.is_pixart
        # self.is_auraflow = model_config.is_auraflow
        # self.is_flux = model_config.is_flux
        # self.is_lumina2 = model_config.is_lumina2

        self.use_text_encoder_1 = model_config.use_text_encoder_1
        self.use_text_encoder_2 = model_config.use_text_encoder_2

        self.config_file = None

        self.is_flow_matching = False
        if self.is_flux or self.is_v3 or self.is_auraflow or self.is_lumina2 or isinstance(self.noise_scheduler, CustomFlowMatchEulerDiscreteScheduler):
            self.is_flow_matching = True

        self.quantize_device = self.device_torch
        self.low_vram = self.model_config.low_vram

        # merge in and preview active with -1 weight
        self.invert_assistant_lora = False
        self._after_sample_img_hooks = []
        self._status_update_hooks = []
        # todo update this based on the model
        self.is_transformer = False
        
        self.sample_prompts_cache = None
        
        self.is_multistage = False
        # a list of multistage boundaries starting with train step 1000 to first idx
        self.multistage_boundaries: List[float] = [0.0]
        # a list of trainable multistage boundaries
        self.trainable_multistage_boundaries: List[int] = [0]
        
        # set true for models that encode control image into text embeddings
        self.encode_control_in_text_embeddings = False
        # control images will come in as a list for encoding some things if true
        self.has_multiple_control_images = False
        # do not resize control images
        self.use_raw_control_images = False
        
    # properties for old arch for backwards compatibility
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
    
    @property
    def unet_unwrapped(self):
        return unwrap_model(self.unet)
    
    def get_bucket_divisibility(self):
        if self.vae is None:
            return 16
        divisibility = 2 ** (len(self.vae.config['block_out_channels']) - 1)
        
        # flux packs this again,
        if self.is_flux or self.is_v3:
            divisibility = divisibility * 2
        return divisibility * 2 # todo remove this
        

    def load_model(self):
        if self.is_loaded:
            return
        dtype = get_torch_dtype(self.dtype)

        # move the betas alphas and  alphas_cumprod to device. Sometimd they get stuck on cpu, not sure why
        # self.noise_scheduler.betas = self.noise_scheduler.betas.to(self.device_torch)
        # self.noise_scheduler.alphas = self.noise_scheduler.alphas.to(self.device_torch)
        # self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device_torch)

        model_path = self.model_config.name_or_path
        if 'civitai.com' in self.model_config.name_or_path:
            # load is a civit ai model, use the loader.
            from toolkit.civitai import get_model_path_from_url
            model_path = get_model_path_from_url(self.model_config.name_or_path)

        load_args = {}
        if self.noise_scheduler:
            load_args['scheduler'] = self.noise_scheduler

        if self.model_config.vae_path is not None:
            load_args['vae'] = load_vae(self.model_config.vae_path, dtype)
        if self.model_config.is_xl or self.model_config.is_ssd or self.model_config.is_vega:
            if self.custom_pipeline is not None:
                pipln = self.custom_pipeline
            else:
                pipln = StableDiffusionXLPipeline
                # pipln = StableDiffusionKDiffusionXLPipeline

            # see if path exists
            if not os.path.exists(model_path) or os.path.isdir(model_path):
                # try to load with default diffusers
                pipe = pipln.from_pretrained(
                    model_path,
                    dtype=dtype,
                    device=self.device_torch,
                    # variant="fp16",
                    use_safetensors=True,
                    **load_args
                )
            else:
                pipe = pipln.from_single_file(
                    model_path,
                    device=self.device_torch,
                    torch_dtype=self.torch_dtype,
                )

            if 'vae' in load_args and load_args['vae'] is not None:
                pipe.vae = load_args['vae']
            flush()

            text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
            tokenizer = [pipe.tokenizer, pipe.tokenizer_2]
            for text_encoder in text_encoders:
                text_encoder.to(self.te_device_torch, dtype=self.te_torch_dtype)
                text_encoder.requires_grad_(False)
                text_encoder.eval()
            text_encoder = text_encoders

            pipe.vae = pipe.vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)

            if self.model_config.experimental_xl:
                print_acc("Experimental XL mode enabled")
                print_acc("Loading and injecting alt weights")
                # load the mismatched weight and force it in
                raw_state_dict = load_file(model_path)
                replacement_weight = raw_state_dict['conditioner.embedders.1.model.text_projection'].clone()
                del raw_state_dict
                #  get state dict for  for 2nd text encoder
                te1_state_dict = text_encoders[1].state_dict()
                # replace weight with mismatched weight
                te1_state_dict['text_projection.weight'] = replacement_weight.to(self.device_torch, dtype=dtype)
                flush()
                print_acc("Injecting alt weights")
        elif self.model_config.is_v3:
            if self.custom_pipeline is not None:
                pipln = self.custom_pipeline
            else:
                pipln = StableDiffusion3Pipeline
            
            print_acc("Loading SD3 model")
            # assume it is the large model
            base_model_path = "stabilityai/stable-diffusion-3.5-large"
            print_acc("Loading transformer")
            subfolder = 'transformer'
            transformer_path = model_path
            # check if HF_DATASETS_OFFLINE or TRANSFORMERS_OFFLINE is set
            if os.path.exists(transformer_path):
                subfolder = None
                transformer_path = os.path.join(transformer_path, 'transformer')
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, 'text_encoder')
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path
            else:
                # is remote use whatever path we were given
                base_model_path = model_path
            
            transformer = SD3Transformer2DModel.from_pretrained(
                transformer_path,
                subfolder=subfolder,
                torch_dtype=dtype,
            )
            if not self.low_vram:
                # for low v ram, we leave it on the cpu. Quantizes slower, but allows training on primary gpu
                transformer.to(self.quantize_device, dtype=dtype)
            flush()
            
            if self.model_config.lora_path is not None:
                raise ValueError("LoRA is not supported for SD3 models currently")
            
            if self.model_config.quantize:
                quantization_type = get_qtype(self.model_config.qtype)
                print_acc("Quantizing transformer")
                quantize(transformer, weights=quantization_type)
                freeze(transformer)
                transformer.to(self.device_torch)
            else:
                transformer.to(self.device_torch, dtype=dtype)
                
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
            print_acc("Loading vae")
            vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
            flush()
            
            print_acc("Loading t5")
            tokenizer_3 = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_3", torch_dtype=dtype)
            text_encoder_3 = T5EncoderModel.from_pretrained(
                base_model_path, 
                subfolder="text_encoder_3", 
                torch_dtype=dtype
            )
            
            text_encoder_3.to(self.device_torch, dtype=dtype)
            flush()

            if self.model_config.quantize:
                print_acc("Quantizing T5")
                quantize(text_encoder_3, weights=get_qtype(self.model_config.qtype))
                freeze(text_encoder_3)
                flush()
                

            # see if path exists
            if not os.path.exists(model_path) or os.path.isdir(model_path):
                try:
                    # try to load with default diffusers
                    pipe = pipln.from_pretrained(
                        base_model_path,
                        dtype=dtype,
                        device=self.device_torch,
                        tokenizer_3=tokenizer_3,
                        text_encoder_3=text_encoder_3,
                        transformer=transformer,
                        # variant="fp16",
                        use_safetensors=True,
                        repo_type="model",
                        ignore_patterns=["*.md", "*..gitattributes"],
                        **load_args
                    )
                except Exception as e:
                    print_acc(f"Error loading from pretrained: {e}")
                    raise e

            else:
                pipe = pipln.from_single_file(
                    model_path,
                    transformer=transformer,
                    device=self.device_torch,
                    torch_dtype=self.torch_dtype,
                    tokenizer_3=tokenizer_3,
                    text_encoder_3=text_encoder_3,
                    **load_args
                )

            flush()

            text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
            tokenizer = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
            # replace the to function with a no-op since it throws an error instead of a warning
            # text_encoders[2].to = lambda *args, **kwargs: None
            for text_encoder in text_encoders:
                text_encoder.to(self.device_torch, dtype=dtype)
                text_encoder.requires_grad_(False)
                text_encoder.eval()
            text_encoder = text_encoders


        elif self.model_config.is_pixart:
            te_kwargs = {}
            # handle quantization of TE
            te_is_quantized = False
            if self.model_config.text_encoder_bits == 8:
                te_kwargs['load_in_8bit'] = True
                te_kwargs['device_map'] = "auto"
                te_is_quantized = True
            elif self.model_config.text_encoder_bits == 4:
                te_kwargs['load_in_4bit'] = True
                te_kwargs['device_map'] = "auto"
                te_is_quantized = True

            main_model_path = "PixArt-alpha/PixArt-XL-2-1024-MS"
            if self.model_config.is_pixart_sigma:
                main_model_path = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"

            main_model_path = model_path

            # load the TE in 8bit mode
            text_encoder = T5EncoderModel.from_pretrained(
                main_model_path,
                subfolder="text_encoder",
                torch_dtype=self.torch_dtype,
                **te_kwargs
            )

            # load the transformer
            subfolder = "transformer"
            # check if it is just the unet
            if os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, subfolder)):
                subfolder = None

            if te_is_quantized:
                # replace the to function with a no-op since it throws an error instead of a warning
                text_encoder.to = lambda *args, **kwargs: None

            text_encoder.to(self.te_device_torch, dtype=self.te_torch_dtype)

            if self.model_config.is_pixart_sigma:
                # load the transformer only from the save
                transformer = Transformer2DModel.from_pretrained(
                    model_path if self.model_config.unet_path is None else self.model_config.unet_path,
                    torch_dtype=self.torch_dtype,
                    subfolder='transformer'
                )
                pipe: PixArtSigmaPipeline = PixArtSigmaPipeline.from_pretrained(
                    main_model_path,
                    transformer=transformer,
                    text_encoder=text_encoder,
                    dtype=dtype,
                    device=self.device_torch,
                    **load_args
                )

            else:

                # load the transformer only from the save
                transformer = Transformer2DModel.from_pretrained(model_path, torch_dtype=self.torch_dtype,
                                                                 subfolder=subfolder)
                pipe: PixArtAlphaPipeline = PixArtAlphaPipeline.from_pretrained(
                    main_model_path,
                    transformer=transformer,
                    text_encoder=text_encoder,
                    dtype=dtype,
                    device=self.device_torch,
                    **load_args
                ).to(self.device_torch)

            if self.model_config.unet_sample_size is not None:
                pipe.transformer.config.sample_size = self.model_config.unet_sample_size
            pipe.transformer = pipe.transformer.to(self.device_torch, dtype=dtype)

            flush()
            # text_encoder = pipe.text_encoder
            # text_encoder.to(self.device_torch, dtype=dtype)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            pipe.transformer = pipe.transformer.to(self.device_torch, dtype=dtype)
            tokenizer = pipe.tokenizer

            pipe.vae = pipe.vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)
            if self.noise_scheduler is None:
                self.noise_scheduler = pipe.scheduler


        elif self.model_config.is_auraflow:
            te_kwargs = {}
            # handle quantization of TE
            te_is_quantized = False
            if self.model_config.text_encoder_bits == 8:
                te_kwargs['load_in_8bit'] = True
                te_kwargs['device_map'] = "auto"
                te_is_quantized = True
            elif self.model_config.text_encoder_bits == 4:
                te_kwargs['load_in_4bit'] = True
                te_kwargs['device_map'] = "auto"
                te_is_quantized = True

            main_model_path = model_path

            # load the TE in 8bit mode
            text_encoder = UMT5EncoderModel.from_pretrained(
                main_model_path,
                subfolder="text_encoder",
                torch_dtype=self.torch_dtype,
                **te_kwargs
            )

            # load the transformer
            subfolder = "transformer"
            # check if it is just the unet
            if os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, subfolder)):
                subfolder = None

            if te_is_quantized:
                # replace the to function with a no-op since it throws an error instead of a warning
                text_encoder.to = lambda *args, **kwargs: None

            # load the transformer only from the save
            transformer = AuraFlowTransformer2DModel.from_pretrained(
                model_path if self.model_config.unet_path is None else self.model_config.unet_path,
                torch_dtype=self.torch_dtype,
                subfolder='transformer'
            )
            pipe: AuraFlowPipeline = AuraFlowPipeline.from_pretrained(
                main_model_path,
                transformer=transformer,
                text_encoder=text_encoder,
                dtype=dtype,
                device=self.device_torch,
                **load_args
            )

            pipe.transformer = pipe.transformer.to(self.device_torch, dtype=dtype)

            # patch auraflow so it can handle other aspect ratios
            # patch_auraflow_pos_embed(pipe.transformer.pos_embed)

            flush()
            # text_encoder = pipe.text_encoder
            # text_encoder.to(self.device_torch, dtype=dtype)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            pipe.transformer = pipe.transformer.to(self.device_torch, dtype=dtype)
            tokenizer = pipe.tokenizer

        elif self.model_config.is_flux:
            self.print_and_status_update("Loading Flux model")
            # base_model_path = "black-forest-labs/FLUX.1-schnell"
            base_model_path = self.model_config.name_or_path_original
            self.print_and_status_update("Loading transformer")
            subfolder = 'transformer'
            transformer_path = model_path
            local_files_only = False
            # check if HF_DATASETS_OFFLINE or TRANSFORMERS_OFFLINE is set
            if os.path.exists(transformer_path):
                subfolder = None
                transformer_path = os.path.join(transformer_path, 'transformer')
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, 'text_encoder')
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                subfolder=subfolder,
                torch_dtype=dtype,
                # low_cpu_mem_usage=False,
                # device_map=None
            )
            # hack in model gpu splitter
            if self.model_config.split_model_over_gpus:
                add_model_gpu_splitter_to_flux(
                    transformer, 
                    other_module_param_count_scale=self.model_config.split_model_other_module_param_count_scale
                )
            
            if not self.low_vram:
                # for low v ram, we leave it on the cpu. Quantizes slower, but allows training on primary gpu
                transformer.to(self.quantize_device, dtype=dtype)
            flush()

            if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
                if self.model_config.inference_lora_path is not None and self.model_config.assistant_lora_path is not None:
                    raise ValueError("Cannot load both assistant lora and inference lora at the same time")
                
                if self.model_config.lora_path:
                    raise ValueError("Cannot load both assistant lora and lora at the same time")

                if not self.is_flux:
                    raise ValueError("Assistant/ inference lora is only supported for flux models currently")
                
                load_lora_path = self.model_config.inference_lora_path
                if load_lora_path is None:
                    load_lora_path = self.model_config.assistant_lora_path

                if os.path.isdir(load_lora_path):
                    load_lora_path = os.path.join(
                        load_lora_path, "pytorch_lora_weights.safetensors"
                    )
                elif not os.path.exists(load_lora_path):
                    print_acc(f"Grabbing lora from the hub: {load_lora_path}")
                    new_lora_path = hf_hub_download(
                        load_lora_path,
                        filename="pytorch_lora_weights.safetensors"
                    )
                    # replace the path
                    load_lora_path = new_lora_path
                    
                    if self.model_config.inference_lora_path is not None:
                        self.model_config.inference_lora_path = new_lora_path
                    if self.model_config.assistant_lora_path is not None:
                        self.model_config.assistant_lora_path = new_lora_path

                if self.model_config.assistant_lora_path is not None:
                    # for flux, we assume it is flux schnell. We cannot merge in the assistant lora and unmerge it on
                    # quantized weights so it had to process unmerged (slow). Since schnell samples in just 4 steps
                    # it is better to merge it in now, and sample slowly later, otherwise training is slowed in half
                    # so we will merge in now and sample with -1 weight later
                    self.invert_assistant_lora = True
                    # trigger it to get merged in
                    self.model_config.lora_path = self.model_config.assistant_lora_path

            if self.model_config.lora_path is not None:
                print_acc("Fusing in LoRA")
                # need the pipe for peft
                pipe: FluxPipeline = FluxPipeline(
                    scheduler=None,
                    text_encoder=None,
                    tokenizer=None,
                    text_encoder_2=None,
                    tokenizer_2=None,
                    vae=None,
                    transformer=transformer,
                )
                if self.low_vram:
                    # we cannot fuse the loras all at once without ooming in lowvram mode, so we have to do it in parts
                    # we can do it on the cpu but it takes about 5-10 mins vs seconds on the gpu
                    # we are going to separate it into the two transformer blocks one at a time

                    lora_state_dict = load_file(self.model_config.lora_path)
                    single_transformer_lora = {}
                    single_block_key = "transformer.single_transformer_blocks."
                    double_transformer_lora = {}
                    double_block_key = "transformer.transformer_blocks."
                    for key, value in lora_state_dict.items():
                        if single_block_key in key:
                            single_transformer_lora[key] = value
                        elif double_block_key in key:
                            double_transformer_lora[key] = value
                        else:
                            raise ValueError(f"Unknown lora key: {key}. Cannot load this lora in low vram mode")

                    # double blocks
                    transformer.transformer_blocks = transformer.transformer_blocks.to(
                        self.quantize_device, dtype=dtype
                    )
                    pipe.load_lora_weights(double_transformer_lora, adapter_name=f"lora1_double")
                    pipe.fuse_lora()
                    pipe.unload_lora_weights()
                    transformer.transformer_blocks = transformer.transformer_blocks.to(
                        'cpu', dtype=dtype
                    )

                    # single blocks
                    transformer.single_transformer_blocks = transformer.single_transformer_blocks.to(
                        self.quantize_device, dtype=dtype
                    )
                    pipe.load_lora_weights(single_transformer_lora, adapter_name=f"lora1_single")
                    pipe.fuse_lora()
                    pipe.unload_lora_weights()
                    transformer.single_transformer_blocks = transformer.single_transformer_blocks.to(
                        'cpu', dtype=dtype
                    )

                    # cleanup
                    del single_transformer_lora
                    del double_transformer_lora
                    del lora_state_dict
                    flush()

                else:
                    # need the pipe to do this unfortunately for now
                    # we have to fuse in the weights before quantizing
                    pipe.load_lora_weights(self.model_config.lora_path, adapter_name="lora1")
                    pipe.fuse_lora()
                    # unfortunately, not an easier way with peft
                    pipe.unload_lora_weights()
            flush()
            
            if self.model_config.quantize:
                # patch the state dict method
                patch_dequantization_on_save(transformer)
                quantization_type = get_qtype(self.model_config.qtype)
                self.print_and_status_update("Quantizing transformer")
                quantize(transformer, weights=quantization_type, **self.model_config.quantize_kwargs)
                freeze(transformer)
                transformer.to(self.device_torch)
            else:
                transformer.to(self.device_torch, dtype=dtype)

            flush()

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
            self.print_and_status_update("Loading VAE")
            if self.model_config.vae_path is not None:
                vae = load_vae(self.model_config.vae_path, dtype)
            else:
                vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
            flush()
            
            self.print_and_status_update("Loading T5")
            tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_2", torch_dtype=dtype)
            text_encoder_2 = T5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder_2",
                                                            torch_dtype=dtype)

            text_encoder_2.to(self.device_torch, dtype=dtype)
            flush()

            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing T5")
                quantize(text_encoder_2, weights=get_qtype(self.model_config.qtype))
                freeze(text_encoder_2)
                flush()
                
            self.print_and_status_update("Loading CLIP")
            text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", torch_dtype=dtype)
            text_encoder.to(self.device_torch, dtype=dtype)

            self.print_and_status_update("Making pipe")
            Pipe = FluxPipeline
            
            pipe: Pipe = Pipe(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=None,
                tokenizer_2=tokenizer_2,
                vae=vae,
                transformer=None,
            )
            pipe.text_encoder_2 = text_encoder_2
            pipe.transformer = transformer

            self.print_and_status_update("Preparing Model")

            text_encoder = [pipe.text_encoder, pipe.text_encoder_2]
            tokenizer = [pipe.tokenizer, pipe.tokenizer_2]

            pipe.transformer = pipe.transformer.to(self.device_torch)

            flush()
            text_encoder[0].to(self.device_torch)
            text_encoder[0].requires_grad_(False)
            text_encoder[0].eval()
            text_encoder[1].to(self.device_torch)
            text_encoder[1].requires_grad_(False)
            text_encoder[1].eval()
            pipe.transformer = pipe.transformer.to(self.device_torch)
            flush()
        elif self.model_config.is_lumina2:
            self.print_and_status_update("Loading Lumina2 model")
            # base_model_path = "black-forest-labs/FLUX.1-schnell"
            base_model_path = self.model_config.name_or_path_original
            self.print_and_status_update("Loading transformer")
            subfolder = 'transformer'
            transformer_path = model_path
            if os.path.exists(transformer_path):
                subfolder = None
                transformer_path = os.path.join(transformer_path, 'transformer')
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, 'text_encoder')
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = Lumina2Transformer2DModel.from_pretrained(
                transformer_path,
                subfolder=subfolder,
                torch_dtype=dtype,
            )
            
            if self.model_config.split_model_over_gpus:
                raise ValueError("Splitting model over gpus is not supported for Lumina2 models")
            
            transformer.to(self.quantize_device, dtype=dtype)
            flush()

            if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
                raise ValueError("Assistant LoRA is not supported for Lumina2 models currently")

            if self.model_config.lora_path is not None:
                raise ValueError("Loading LoRA is not supported for Lumina2 models currently")
            
            flush()
            
            if self.model_config.quantize:
                # patch the state dict method
                patch_dequantization_on_save(transformer)
                quantization_type = get_qtype(self.model_config.qtype)
                self.print_and_status_update("Quantizing transformer")
                quantize(transformer, weights=quantization_type, **self.model_config.quantize_kwargs)
                freeze(transformer)
                transformer.to(self.device_torch)
            else:
                transformer.to(self.device_torch, dtype=dtype)

            flush()

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
            self.print_and_status_update("Loading vae")
            vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
            flush()
            
            if self.model_config.te_name_or_path is not None:
                self.print_and_status_update("Loading TE")
                tokenizer = AutoTokenizer.from_pretrained(self.model_config.te_name_or_path, torch_dtype=dtype)
                text_encoder = AutoModel.from_pretrained(self.model_config.te_name_or_path, torch_dtype=dtype)
            else:
                self.print_and_status_update("Loading Gemma2")
                tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", torch_dtype=dtype)
                text_encoder = AutoModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype)

            text_encoder.to(self.device_torch, dtype=dtype)
            flush()

            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing Gemma2")
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
                freeze(text_encoder)
                flush()

            self.print_and_status_update("Making pipe")
            pipe: Lumina2Pipeline = Lumina2Pipeline(
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
        else:
            if self.custom_pipeline is not None:
                pipln = self.custom_pipeline
            else:
                pipln = StableDiffusionPipeline

            if self.model_config.text_encoder_bits < 16:
                # this is only supported for T5 models for now
                te_kwargs = {}
                # handle quantization of TE
                te_is_quantized = False
                if self.model_config.text_encoder_bits == 8:
                    te_kwargs['load_in_8bit'] = True
                    te_kwargs['device_map'] = "auto"
                    te_is_quantized = True
                elif self.model_config.text_encoder_bits == 4:
                    te_kwargs['load_in_4bit'] = True
                    te_kwargs['device_map'] = "auto"
                    te_is_quantized = True

                text_encoder = T5EncoderModel.from_pretrained(
                    model_path,
                    subfolder="text_encoder",
                    torch_dtype=self.te_torch_dtype,
                    **te_kwargs
                )
                # replace the to function with a no-op since it throws an error instead of a warning
                text_encoder.to = lambda *args, **kwargs: None

                load_args['text_encoder'] = text_encoder

            # see if path exists
            if not os.path.exists(model_path) or os.path.isdir(model_path):
                # try to load with default diffusers
                pipe = pipln.from_pretrained(
                    model_path,
                    dtype=dtype,
                    device=self.device_torch,
                    load_safety_checker=False,
                    requires_safety_checker=False,
                    safety_checker=None,
                    # variant="fp16",
                    trust_remote_code=True,
                    **load_args
                )
            else:
                pipe = pipln.from_single_file(
                    model_path,
                    dtype=dtype,
                    device=self.device_torch,
                    load_safety_checker=False,
                    requires_safety_checker=False,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    trust_remote_code=True,
                    **load_args
                )
            flush()

            pipe.register_to_config(requires_safety_checker=False)
            text_encoder = pipe.text_encoder
            text_encoder.to(self.te_device_torch, dtype=self.te_torch_dtype)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            tokenizer = pipe.tokenizer

        # scheduler doesn't get set sometimes, so we set it here
        pipe.scheduler = self.noise_scheduler

        # add hacks to unet to help training
        # pipe.unet = prepare_unet_for_training(pipe.unet)

        if self.is_pixart or self.is_v3 or self.is_auraflow or self.is_flux or self.is_lumina2:
            # pixart and sd3 dont use a unet
            self.unet = pipe.transformer
        else:
            self.unet: 'UNet2DConditionModel' = pipe.unet
        self.vae: 'AutoencoderKL' = pipe.vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)
        self.vae.eval()
        self.vae.requires_grad_(False)
        VAE_SCALE_FACTOR = 2 ** (len(self.vae.config['block_out_channels']) - 1)
        self.vae_scale_factor = VAE_SCALE_FACTOR
        self.unet.to(self.device_torch, dtype=dtype)
        self.unet.requires_grad_(False)
        self.unet.eval()

        # load any loras we have
        if self.model_config.lora_path is not None and not self.is_flux and not self.is_lumina2:
            pipe.load_lora_weights(self.model_config.lora_path, adapter_name="lora1")
            pipe.fuse_lora()
            # unfortunately, not an easier way with peft
            pipe.unload_lora_weights()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.pipeline = pipe
        self.load_refiner()
        self.is_loaded = True

        if self.model_config.assistant_lora_path is not None:
            print_acc("Loading assistant lora")
            self.assistant_lora: 'LoRASpecialNetwork' = load_assistant_lora_from_path(
                self.model_config.assistant_lora_path, self)

            if self.invert_assistant_lora:
                # invert and disable during training
                self.assistant_lora.multiplier = -1.0
                self.assistant_lora.is_active = False
                
        if self.model_config.inference_lora_path is not None:
            print_acc("Loading inference lora")
            self.assistant_lora: 'LoRASpecialNetwork' = load_assistant_lora_from_path(
                self.model_config.inference_lora_path, self)
            # disable during training
            self.assistant_lora.is_active = False

        if self.is_pixart and self.vae_scale_factor == 16:
            # TODO make our own pipeline?
            # we generate an image 2x larger, so we need to copy the sizes from larger ones down
            # ASPECT_RATIO_1024_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_2048_BIN, ASPECT_RATIO_256_BIN
            for key in ASPECT_RATIO_256_BIN.keys():
                ASPECT_RATIO_256_BIN[key] = [ASPECT_RATIO_256_BIN[key][0] * 2, ASPECT_RATIO_256_BIN[key][1] * 2]
            for key in ASPECT_RATIO_512_BIN.keys():
                ASPECT_RATIO_512_BIN[key] = [ASPECT_RATIO_512_BIN[key][0] * 2, ASPECT_RATIO_512_BIN[key][1] * 2]
            for key in ASPECT_RATIO_1024_BIN.keys():
                ASPECT_RATIO_1024_BIN[key] = [ASPECT_RATIO_1024_BIN[key][0] * 2, ASPECT_RATIO_1024_BIN[key][1] * 2]
            for key in ASPECT_RATIO_2048_BIN.keys():
                ASPECT_RATIO_2048_BIN[key] = [ASPECT_RATIO_2048_BIN[key][0] * 2, ASPECT_RATIO_2048_BIN[key][1] * 2]

    def te_train(self):
        if isinstance(self.text_encoder, list):
            for te in self.text_encoder:
                te.train()
        else:
            self.text_encoder.train()

    def te_eval(self):
        if isinstance(self.text_encoder, list):
            for te in self.text_encoder:
                te.eval()
        else:
            self.text_encoder.eval()

    def load_refiner(self):
        # for now, we are just going to rely on the TE from the base model
        # which is TE2 for SDXL and TE for SD (no refiner currently)
        # and completely ignore a TE that may or may not be packaged with the refiner
        if self.model_config.refiner_name_or_path is not None:
            refiner_config_path = os.path.join(ORIG_CONFIGS_ROOT, 'sd_xl_refiner.yaml')
            # load the refiner model
            dtype = get_torch_dtype(self.dtype)
            model_path = self.model_config.refiner_name_or_path
            if not os.path.exists(model_path) or os.path.isdir(model_path):
                # TODO only load unet??
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_path,
                    dtype=dtype,
                    device=self.device_torch,
                    # variant="fp16",
                    use_safetensors=True,
                ).to(self.device_torch)
            else:
                refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_path,
                    dtype=dtype,
                    device=self.device_torch,
                    torch_dtype=self.torch_dtype,
                    original_config_file=refiner_config_path,
                ).to(self.device_torch)

            self.refiner_unet = refiner.unet
            del refiner
            flush()
            
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
            pipeline: Union[None, StableDiffusionPipeline, StableDiffusionXLPipeline] = None,
    ):
        print_acc("[DEBUG] generate_images: start")
        network = unwrap_model(self.network)
        merge_multiplier = 1.0
        flush()
        print_acc("[DEBUG] generate_images: after unwrap_model and flush")
        # if using assistant, unfuse it
        if self.model_config.assistant_lora_path is not None:
            print_acc("[DEBUG] generate_images: Unloading assistant lora")
            if self.invert_assistant_lora:
                self.assistant_lora.is_active = True
                self.assistant_lora.force_to(self.device_torch, self.torch_dtype)
            else:
                self.assistant_lora.is_active = False
        if self.model_config.inference_lora_path is not None:
            print_acc("[DEBUG] generate_images: Loading inference lora")
            self.assistant_lora.is_active = True
            self.assistant_lora.force_to(self.device_torch, self.torch_dtype)
        print_acc("[DEBUG] generate_images: after lora handling")
        if network is not None:
            network.eval()
            unique_network_weights = set([x.network_multiplier for x in image_configs])
            if len(unique_network_weights) == 1 and network.can_merge_in:
                self.unet.to(self.device_torch)
                can_merge_in = True
                merge_multiplier = unique_network_weights.pop()
                network.merge_in(merge_weight=merge_multiplier)
        else:
            network = BlankNetwork()
        print_acc("[DEBUG] generate_images: after network merge_in")
        self.save_device_state()
        self.set_device_state_preset('generate')
        print_acc("[DEBUG] generate_images: after device state preset")
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        print_acc("[DEBUG] generate_images: after RNG state save")
        # ...existing code...
        if pipeline is None:
            noise_scheduler = self.noise_scheduler
            if sampler is not None:
                if sampler.startswith("sample_"):
                    noise_scheduler = get_sampler(
                        'lms', {
                            "prediction_type": self.prediction_type,
                        })
                else:
                    arch = 'sd'
                    if self.is_pixart:
                        arch = 'pixart'
                    if self.is_flux:
                        arch = 'flux'
                    if self.is_lumina2:
                        arch = 'lumina2'
                    noise_scheduler = get_sampler(
                        sampler,
                        {
                            "prediction_type": self.prediction_type,
                        },
                        arch=arch
                    )
                try:
                    noise_scheduler = noise_scheduler.to(self.device_torch, self.torch_dtype)
                except:
                    pass
            print_acc("[DEBUG] generate_images: after sampler/noise_scheduler setup")
            # ...existing code...
            if sampler.startswith("sample_") and self.is_xl:
                Pipe = StableDiffusionKDiffusionXLPipeline
            elif self.is_xl:
                Pipe = StableDiffusionXLPipeline
            elif self.is_v3:
                Pipe = StableDiffusion3Pipeline
            else:
                Pipe = StableDiffusionPipeline
            extra_args = {}
            if self.adapter is not None:
                if isinstance(self.adapter, T2IAdapter):
                    if self.is_xl:
                        Pipe = StableDiffusionXLAdapterPipeline
                    else:
                        Pipe = StableDiffusionAdapterPipeline
                    extra_args['adapter'] = self.adapter
                elif isinstance(self.adapter, ControlNetModel):
                    if self.is_xl:
                        Pipe = StableDiffusionXLControlNetPipeline
                    else:
                        Pipe = StableDiffusionControlNetPipeline
                    extra_args['controlnet'] = self.adapter
                elif isinstance(self.adapter, ReferenceAdapter):
                    self.adapter.noise_scheduler = noise_scheduler
                else:
                    if self.is_xl:
                        extra_args['add_watermarker'] = False
            print_acc("[DEBUG] generate_images: after adapter extra_args setup")
            # ...existing code...
            if self.is_xl:
                pipeline = Pipe(
                    vae=self.vae,
                    unet=self.unet,
                    text_encoder=self.text_encoder[0],
                    text_encoder_2=self.text_encoder[1],
                    tokenizer=self.tokenizer[0],
                    tokenizer_2=self.tokenizer[1],
                    scheduler=noise_scheduler,
                    **extra_args
                ).to(self.device_torch)
                pipeline.watermark = None
            elif self.is_flux:
                if self.model_config.use_flux_cfg:
                    pipeline = FluxWithCFGPipeline(
                        vae=self.vae,
                        transformer=unwrap_model(self.unet),
                        text_encoder=unwrap_model(self.text_encoder[0]),
                        text_encoder_2=unwrap_model(self.text_encoder[1]),
                        tokenizer=self.tokenizer[0],
                        tokenizer_2=self.tokenizer[1],
                        scheduler=noise_scheduler,
                        **extra_args
                    )
                else:
                    Pipe = FluxPipeline
                    if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
                        if self.adapter.control_lora is not None:
                            Pipe = FluxAdvancedControlPipeline
                            extra_args['do_inpainting'] = self.adapter.config.has_inpainting_input
                            extra_args['num_controls'] = self.adapter.config.num_control_images
                    pipeline = Pipe(
                        vae=self.vae,
                        transformer=unwrap_model(self.unet),
                        text_encoder=unwrap_model(self.text_encoder[0]),
                        text_encoder_2=unwrap_model(self.text_encoder[1]),
                        tokenizer=self.tokenizer[0],
                        tokenizer_2=self.tokenizer[1],
                        scheduler=noise_scheduler,
                        **extra_args
                    )
                pipeline.watermark = None
            elif self.is_lumina2:
                pipeline = Lumina2Pipeline(
                    vae=self.vae,
                    transformer=self.unet,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    scheduler=noise_scheduler,
                    **extra_args
                )
            elif self.is_v3:
                pipeline = Pipe(
                    vae=self.vae,
                    transformer=self.unet,
                    text_encoder=self.text_encoder[0],
                    text_encoder_2=self.text_encoder[1],
                    text_encoder_3=self.text_encoder[2],
                    tokenizer=self.tokenizer[0],
                    tokenizer_2=self.tokenizer[1],
                    tokenizer_3=self.tokenizer[2],
                    scheduler=noise_scheduler,
                    **extra_args
                )
            elif self.is_pixart:
                pipeline = PixArtSigmaPipeline(
                    vae=self.vae,
                    transformer=self.unet,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    scheduler=noise_scheduler,
                    **extra_args
                )
            elif self.is_auraflow:
                pipeline = AuraFlowPipeline(
                    vae=self.vae,
                    transformer=self.unet,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    scheduler=noise_scheduler,
                    **extra_args
                )
            else:
                pipeline = Pipe(
                    vae=self.vae,
                    unet=self.unet,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    scheduler=noise_scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                    **extra_args
                )
            flush()
            print_acc("[DEBUG] generate_images: after pipeline creation")
            pipeline.set_progress_bar_config(disable=True)
            if sampler.startswith("sample_"):
                pipeline.set_scheduler(sampler)
        print_acc("[DEBUG] generate_images: after pipeline/scheduler setup")
        refiner_pipeline = None
        if self.refiner_unet:
            refiner_pipeline = StableDiffusionXLImg2ImgPipeline(
                vae=pipeline.vae,
                unet=self.refiner_unet,
                text_encoder=None,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=None,
                tokenizer_2=pipeline.tokenizer_2,
                scheduler=pipeline.scheduler,
                add_watermarker=False,
                requires_aesthetics_score=True,
            ).to(self.device_torch)
            refiner_pipeline.watermark = None
            refiner_pipeline.set_progress_bar_config(disable=True)
            flush()
        print_acc("[DEBUG] generate_images: after refiner_pipeline setup")
        start_multiplier = 1.0
        if network is not None:
            start_multiplier = network.multiplier
        print_acc("[DEBUG] generate_images: before with network")
        # pipeline.to(self.device_torch)
        with network:
            with torch.no_grad():
                if network is not None:
                    assert network.is_active
                print_acc("[DEBUG] generate_images: inside with network/no_grad")
                for i in tqdm(range(len(image_configs)), desc=f"Generating Images", leave=False):
                    if i % 10 == 0:
                        print_acc(f"[DEBUG] generate_images: processing image {i}/{len(image_configs)}")
                    gen_config = image_configs[i]
                    extra = {}
                    validation_image = None
                    if self.adapter is not None and gen_config.adapter_image_path is not None:
                        validation_image = Image.open(gen_config.adapter_image_path)
                        if ".inpaint." not in gen_config.adapter_image_path:
                            validation_image = validation_image.convert("RGB")
                        else:
                            if validation_image.mode != "RGBA":
                                raise ValueError("Inpainting images must have an alpha channel")
                        if isinstance(self.adapter, T2IAdapter):
                            validation_image = validation_image.resize((gen_config.width * 2, gen_config.height * 2))
                            extra['image'] = validation_image
                            extra['adapter_conditioning_scale'] = gen_config.adapter_conditioning_scale
                        if isinstance(self.adapter, ControlNetModel):
                            validation_image = validation_image.resize((gen_config.width, gen_config.height))
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
                            transform = transforms.Compose([
                                transforms.ToTensor(),
                            ])
                            validation_image = transform(validation_image)
                            self.adapter.num_images = 1
                        if isinstance(self.adapter, ReferenceAdapter):
                            validation_image = transforms.ToTensor()(validation_image)
                            validation_image = validation_image * 2.0 - 1.0
                            validation_image = validation_image.unsqueeze(0)
                            self.adapter.set_reference_images(validation_image)
                    if i % 10 == 0:
                        print_acc(f"[DEBUG] generate_images: after validation_image setup for image {i}")
                    if network is not None:
                        network.multiplier = gen_config.network_multiplier
                    torch.manual_seed(gen_config.seed)
                    torch.cuda.manual_seed(gen_config.seed)
                    generator = torch.manual_seed(gen_config.seed)
                    if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter) \
                            and gen_config.adapter_image_path is not None:
                        conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(validation_image)
                        self.adapter(conditional_clip_embeds)
                    if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
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
                    if self.sample_prompts_cache is not None:
                        conditional_embeds = self.sample_prompts_cache[i]['conditional'].to(self.device_torch, dtype=self.torch_dtype)
                        unconditional_embeds = self.sample_prompts_cache[i]['unconditional'].to(self.device_torch, dtype=self.torch_dtype)
                    else: 
                        if isinstance(self.adapter, CustomAdapter):
                            self.adapter.is_unconditional_run = False
                        conditional_embeds = self.encode_prompt(gen_config.prompt, gen_config.prompt_2, force_all=True)
                        if isinstance(self.adapter, CustomAdapter):
                            self.adapter.is_unconditional_run = True
                        unconditional_embeds = self.encode_prompt(
                            gen_config.negative_prompt, gen_config.negative_prompt_2, force_all=True
                        )
                        if isinstance(self.adapter, CustomAdapter):
                            self.adapter.is_unconditional_run = False
                    if i % 10 == 0:
                        print_acc(f"[DEBUG] generate_images: after prompt encoding for image {i}")
                    gen_config.post_process_embeddings(
                        conditional_embeds,
                        unconditional_embeds,
                    )
                    if self.decorator is not None:
                        conditional_embeds.text_embeds = self.decorator(conditional_embeds.text_embeds)
                        unconditional_embeds.text_embeds = self.decorator(unconditional_embeds.text_embeds, is_unconditional=True)
                    if self.adapter is not None and isinstance(self.adapter, IPAdapter) \
                            and gen_config.adapter_image_path is not None:
                        conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(validation_image)
                        unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(validation_image,
                                                                                                    True)
                        conditional_embeds = self.adapter(conditional_embeds, conditional_clip_embeds, is_unconditional=False)
                        unconditional_embeds = self.adapter(unconditional_embeds, unconditional_clip_embeds, is_unconditional=True)
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
                        self.adapter.add_extra_values(extra_values, is_unconditional=False)
                        self.adapter.add_extra_values(torch.zeros_like(extra_values), is_unconditional=True)
                        pass
                    if self.refiner_unet is not None and gen_config.refiner_start_at < 1.0:
                        extra['denoising_end'] = gen_config.refiner_start_at
                        extra['output_type'] = 'latent'
                        if not self.is_xl:
                            raise ValueError("Refiner is only supported for XL models")
                    conditional_embeds = conditional_embeds.to(self.device_torch, dtype=self.unet.dtype)
                    unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=self.unet.dtype)
                    if i % 10 == 0:
                        print_acc(f"[DEBUG] generate_images: after embedding post-processing for image {i}")
                    # ...existing code for image generation and saving...
                    # ...existing code...
        # ...existing code for cleanup and restoring state...
        print_acc("[DEBUG] generate_images: end")
