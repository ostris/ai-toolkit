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

        # move the betas alphas and  alphas_cumprod to device. Sometimed they get stuck on cpu, not sure why
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
        network = unwrap_model(self.network)
        merge_multiplier = 1.0
        flush()
        # if using assistant, unfuse it
        if self.model_config.assistant_lora_path is not None:
            print_acc("Unloading assistant lora")
            if self.invert_assistant_lora:
                self.assistant_lora.is_active = True
                # move weights on to the device
                self.assistant_lora.force_to(self.device_torch, self.torch_dtype)
            else:
                self.assistant_lora.is_active = False
                
        if self.model_config.inference_lora_path is not None:
            print_acc("Loading inference lora")
            self.assistant_lora.is_active = True
            # move weights on to the device
            self.assistant_lora.force_to(self.device_torch, self.torch_dtype)

        if network is not None:
            network.eval()
            # check if we have the same network weight for all samples. If we do, we can merge in th
            # the network to drastically speed up inference
            unique_network_weights = set([x.network_multiplier for x in image_configs])
            if len(unique_network_weights) == 1 and network.can_merge_in:
                # make sure it is on device before merging. 
                self.unet.to(self.device_torch)
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
            noise_scheduler = self.noise_scheduler
            if sampler is not None:
                if sampler.startswith("sample_"):  # sample_dpmpp_2m
                    # using ksampler
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

            if sampler.startswith("sample_") and self.is_xl:
                # using kdiffusion
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
                    # pass the noise scheduler to the adapter
                    self.adapter.noise_scheduler = noise_scheduler
                else:
                    if self.is_xl:
                        extra_args['add_watermarker'] = False

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
                        # see if it is a control lora
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
            # disable progress bar
            pipeline.set_progress_bar_config(disable=True)

            if sampler.startswith("sample_"):
                pipeline.set_scheduler(sampler)

        refiner_pipeline = None
        if self.refiner_unet:
            # build refiner pipeline
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
            # refiner_pipeline.register_to_config(requires_aesthetics_score=False)
            refiner_pipeline.watermark = None
            refiner_pipeline.set_progress_bar_config(disable=True)
            flush()

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
                        validation_image = Image.open(gen_config.adapter_image_path)
                        # if the name doesnt have .inpainting. in it, make sure it is rgb
                        if ".inpaint." not in gen_config.adapter_image_path:
                            validation_image = validation_image.convert("RGB")
                        else:
                            # make sure it has an alpha
                            if validation_image.mode != "RGBA":
                                raise ValueError("Inpainting images must have an alpha channel")
                        if isinstance(self.adapter, T2IAdapter):
                            # not sure why this is double??
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
                        conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(validation_image)
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

                    if self.sample_prompts_cache is not None:
                        conditional_embeds = self.sample_prompts_cache[i]['conditional'].to(self.device_torch, dtype=self.torch_dtype)
                        unconditional_embeds = self.sample_prompts_cache[i]['unconditional'].to(self.device_torch, dtype=self.torch_dtype)
                    else: 
                        # encode the prompt ourselves so we can do fun stuff with embeddings
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

                    # allow any manipulations to take place to embeddings
                    gen_config.post_process_embeddings(
                        conditional_embeds,
                        unconditional_embeds,
                    )
                    
                    if self.decorator is not None:
                        # apply the decorator to the embeddings
                        conditional_embeds.text_embeds = self.decorator(conditional_embeds.text_embeds)
                        unconditional_embeds.text_embeds = self.decorator(unconditional_embeds.text_embeds, is_unconditional=True)

                    if self.adapter is not None and isinstance(self.adapter, IPAdapter) \
                            and gen_config.adapter_image_path is not None:
                        # apply the image projection
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
                        # apply extra values to the embeddings
                        self.adapter.add_extra_values(extra_values, is_unconditional=False)
                        self.adapter.add_extra_values(torch.zeros_like(extra_values), is_unconditional=True)
                        pass  # todo remove, for debugging

                    if self.refiner_unet is not None and gen_config.refiner_start_at < 1.0:
                        # if we have a refiner loaded, set the denoising end at the refiner start
                        extra['denoising_end'] = gen_config.refiner_start_at
                        extra['output_type'] = 'latent'
                        if not self.is_xl:
                            raise ValueError("Refiner is only supported for XL models")

                    conditional_embeds = conditional_embeds.to(self.device_torch, dtype=self.unet.dtype)
                    unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=self.unet.dtype)

                    if self.is_xl:
                        # fix guidance rescale for sdxl
                        # was trained on 0.7 (I believe)

                        grs = gen_config.guidance_rescale
                        # if grs is None or grs < 0.00001:
                        #     grs = 0.7
                        # grs = 0.0

                        if sampler.startswith("sample_"):
                            extra['use_karras_sigmas'] = True
                            extra = {
                                **extra,
                                **gen_config.extra_kwargs,
                            }

                        img = pipeline(
                            # prompt=gen_config.prompt,
                            # prompt_2=gen_config.prompt_2,
                            prompt_embeds=conditional_embeds.text_embeds,
                            pooled_prompt_embeds=conditional_embeds.pooled_embeds,
                            negative_prompt_embeds=unconditional_embeds.text_embeds,
                            negative_pooled_prompt_embeds=unconditional_embeds.pooled_embeds,
                            # negative_prompt=gen_config.negative_prompt,
                            # negative_prompt_2=gen_config.negative_prompt_2,
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            guidance_rescale=grs,
                            latents=gen_config.latents,
                            generator=generator,
                            **extra
                        ).images[0]
                    elif self.is_v3:
                        img = pipeline(
                            prompt_embeds=conditional_embeds.text_embeds,
                            pooled_prompt_embeds=conditional_embeds.pooled_embeds,
                            negative_prompt_embeds=unconditional_embeds.text_embeds,
                            negative_pooled_prompt_embeds=unconditional_embeds.pooled_embeds,
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            latents=gen_config.latents,
                            generator=generator,
                            **extra
                        ).images[0]
                    elif self.is_flux:
                        if self.model_config.use_flux_cfg:
                            img = pipeline(
                                prompt_embeds=conditional_embeds.text_embeds,
                                pooled_prompt_embeds=conditional_embeds.pooled_embeds,
                                negative_prompt_embeds=unconditional_embeds.text_embeds,
                                negative_pooled_prompt_embeds=unconditional_embeds.pooled_embeds,
                                height=gen_config.height,
                                width=gen_config.width,
                                num_inference_steps=gen_config.num_inference_steps,
                                guidance_scale=gen_config.guidance_scale,
                                latents=gen_config.latents,
                                generator=generator,
                                **extra
                            ).images[0]
                        else:
                            # Fix a bug in diffusers/torch
                            def callback_on_step_end(pipe, i, t, callback_kwargs):
                                latents = callback_kwargs["latents"]
                                if latents.dtype != self.unet.dtype:
                                    latents = latents.to(self.unet.dtype)
                                return {"latents": latents}
                            img = pipeline(
                                prompt_embeds=conditional_embeds.text_embeds,
                                pooled_prompt_embeds=conditional_embeds.pooled_embeds,
                                # negative_prompt_embeds=unconditional_embeds.text_embeds,
                                # negative_pooled_prompt_embeds=unconditional_embeds.pooled_embeds,
                                height=gen_config.height,
                                width=gen_config.width,
                                num_inference_steps=gen_config.num_inference_steps,
                                guidance_scale=gen_config.guidance_scale,
                                latents=gen_config.latents,
                                generator=generator,
                                callback_on_step_end=callback_on_step_end,
                                **extra
                            ).images[0]
                    elif self.is_lumina2:
                        pipeline: Lumina2Pipeline = pipeline

                        img = pipeline(
                            prompt_embeds=conditional_embeds.text_embeds,
                            prompt_attention_mask=conditional_embeds.attention_mask.to(self.device_torch, dtype=torch.int64),
                            negative_prompt_embeds=unconditional_embeds.text_embeds,
                            negative_prompt_attention_mask=unconditional_embeds.attention_mask.to(self.device_torch, dtype=torch.int64),
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            latents=gen_config.latents,
                            generator=generator,
                            **extra
                        ).images[0]
                    elif self.is_pixart:
                        # needs attention masks for some reason
                        img = pipeline(
                            prompt=None,
                            prompt_embeds=conditional_embeds.text_embeds.to(self.device_torch, dtype=self.unet.dtype),
                            prompt_attention_mask=conditional_embeds.attention_mask.to(self.device_torch,
                                                                                       dtype=self.unet.dtype),
                            negative_prompt_embeds=unconditional_embeds.text_embeds.to(self.device_torch,
                                                                                       dtype=self.unet.dtype),
                            negative_prompt_attention_mask=unconditional_embeds.attention_mask.to(self.device_torch,
                                                                                                  dtype=self.unet.dtype),
                            negative_prompt=None,
                            # negative_prompt=gen_config.negative_prompt,
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            latents=gen_config.latents,
                            generator=generator,
                            **extra
                        ).images[0]
                    elif self.is_auraflow:
                        pipeline: AuraFlowPipeline = pipeline

                        img = pipeline(
                            prompt=None,
                            prompt_embeds=conditional_embeds.text_embeds.to(self.device_torch, dtype=self.unet.dtype),
                            prompt_attention_mask=conditional_embeds.attention_mask.to(self.device_torch,
                                                                                       dtype=self.unet.dtype),
                            negative_prompt_embeds=unconditional_embeds.text_embeds.to(self.device_torch,
                                                                                       dtype=self.unet.dtype),
                            negative_prompt_attention_mask=unconditional_embeds.attention_mask.to(self.device_torch,
                                                                                                  dtype=self.unet.dtype),
                            negative_prompt=None,
                            # negative_prompt=gen_config.negative_prompt,
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            latents=gen_config.latents,
                            generator=generator,
                            **extra
                        ).images[0]
                    else:
                        img = pipeline(
                            # prompt=gen_config.prompt,
                            prompt_embeds=conditional_embeds.text_embeds,
                            negative_prompt_embeds=unconditional_embeds.text_embeds,
                            # negative_prompt=gen_config.negative_prompt,
                            height=gen_config.height,
                            width=gen_config.width,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            latents=gen_config.latents,
                            generator=generator,
                            **extra
                        ).images[0]

                    if self.refiner_unet is not None and gen_config.refiner_start_at < 1.0:
                        # slide off just the last 1280 on the last dim as refiner does not use first text encoder
                        # todo, should we just use the Text encoder for the refiner? Fine tuned versions will differ
                        refiner_text_embeds = conditional_embeds.text_embeds[:, :, -1280:]
                        refiner_unconditional_text_embeds = unconditional_embeds.text_embeds[:, :, -1280:]
                        # run through refiner
                        img = refiner_pipeline(
                            # prompt=gen_config.prompt,
                            # prompt_2=gen_config.prompt_2,

                            # slice these as it does not use both text encoders
                            # height=gen_config.height,
                            # width=gen_config.width,
                            prompt_embeds=refiner_text_embeds,
                            pooled_prompt_embeds=conditional_embeds.pooled_embeds,
                            negative_prompt_embeds=refiner_unconditional_text_embeds,
                            negative_pooled_prompt_embeds=unconditional_embeds.pooled_embeds,
                            num_inference_steps=gen_config.num_inference_steps,
                            guidance_scale=gen_config.guidance_scale,
                            guidance_rescale=grs,
                            denoising_start=gen_config.refiner_start_at,
                            denoising_end=gen_config.num_inference_steps,
                            image=img.unsqueeze(0),
                            generator=generator,
                        ).images[0]

                    gen_config.save_image(img, i)
                    gen_config.log_image(img, i)
                    self._after_sample_image(i, len(image_configs))
                    flush()

                if self.adapter is not None and isinstance(self.adapter, ReferenceAdapter):
                    self.adapter.clear_memory()

        # clear pipeline and cache to reduce vram usage
        del pipeline
        if refiner_pipeline is not None:
            del refiner_pipeline
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
            num_channels=None,
    ):
        VAE_SCALE_FACTOR = 2 ** (len(self.vae.config['block_out_channels']) - 1)
        if height is None and pixel_height is None:
            raise ValueError("height or pixel_height must be specified")
        if width is None and pixel_width is None:
            raise ValueError("width or pixel_width must be specified")
        if height is None:
            height = pixel_height // VAE_SCALE_FACTOR
        if width is None:
            width = pixel_width // VAE_SCALE_FACTOR

        if num_channels is None:
            num_channels = self.unet_unwrapped.config['in_channels']
            if self.is_flux:
                # it gets packed, unpack it
                num_channels = num_channels // 4
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

    def get_time_ids_from_latents(self, latents: torch.Tensor, requires_aesthetic_score=False):
        VAE_SCALE_FACTOR = 2 ** (len(self.vae.config['block_out_channels']) - 1)
        if self.is_xl:
            bs, ch, h, w = list(latents.shape)

            height = h * VAE_SCALE_FACTOR
            width = w * VAE_SCALE_FACTOR

            dtype = latents.dtype
            # just do it without any cropping nonsense
            target_size = (height, width)
            original_size = (height, width)
            crops_coords_top_left = (0, 0)
            if requires_aesthetic_score:
                # refiner
                # https://huggingface.co/papers/2307.01952
                aesthetic_score = 6.0  # simulate one
                add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            else:
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(latents.device, dtype=dtype)

            batch_time_ids = torch.cat(
                [add_time_ids for _ in range(bs)]
            )
            return batch_time_ids
        else:
            return None

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
            **kwargs,
    ) -> torch.FloatTensor:
        original_samples_chunks = torch.chunk(original_samples, original_samples.shape[0], dim=0)
        noise_chunks = torch.chunk(noise, noise.shape[0], dim=0)
        timesteps_chunks = torch.chunk(timesteps, timesteps.shape[0], dim=0)

        if len(timesteps_chunks) == 1 and len(timesteps_chunks) != len(original_samples_chunks):
            timesteps_chunks = [timesteps_chunks[0]] * len(original_samples_chunks)

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
            raise ValueError("Either text_embeddings or conditional_embeddings must be specified")
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
        if latents.shape[0] == text_embeddings.text_embeds.shape[0]:
            do_classifier_free_guidance = False
        elif latents.shape[0] * 2 != text_embeddings.text_embeds.shape[0]:
            raise ValueError("Batch size of latents must be the same or half the batch size of text embeddings")
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
                    kwargs['down_intrablock_additional_residuals'][idx] = torch.cat([item] * 2, dim=0)

        # handle controlnet
        if 'down_block_additional_residuals' in kwargs and 'mid_block_additional_residual' in kwargs:
            # go through each item and concat if doing cfg and it doesnt have the same shape
            for idx, item in enumerate(kwargs['down_block_additional_residuals']):
                if do_classifier_free_guidance and item.shape[0] != text_embeddings.text_embeds.shape[0]:
                    kwargs['down_block_additional_residuals'][idx] = torch.cat([item] * 2, dim=0)
            for idx, item in enumerate(kwargs['mid_block_additional_residual']):
                if do_classifier_free_guidance and item.shape[0] != text_embeddings.text_embeds.shape[0]:
                    kwargs['mid_block_additional_residual'][idx] = torch.cat([item] * 2, dim=0)

        def scale_model_input(model_input, timestep_tensor):
            if is_input_scaled:
                return model_input
            mi_chunks = torch.chunk(model_input, model_input.shape[0], dim=0)
            timestep_chunks = torch.chunk(timestep_tensor, timestep_tensor.shape[0], dim=0)
            out_chunks = []
            # unsqueeze if timestep is zero dim
            for idx in range(model_input.shape[0]):
                # if scheduler has step_index
                if hasattr(self.noise_scheduler, '_step_index'):
                    self.noise_scheduler._step_index = None
                out_chunks.append(
                    self.noise_scheduler.scale_model_input(mi_chunks[idx], timestep_chunks[idx])
                )
            return torch.cat(out_chunks, dim=0)

        if self.is_xl:
            with torch.no_grad():
                # 16, 6 for bs of 4
                if add_time_ids is None:
                    add_time_ids = self.get_time_ids_from_latents(latents)

                    if do_classifier_free_guidance:
                        # todo check this with larget batches
                        add_time_ids = torch.cat([add_time_ids] * 2)

                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    timestep = torch.cat([timestep] * 2)
                else:
                    latent_model_input = latents

                latent_model_input = scale_model_input(latent_model_input, timestep)

                added_cond_kwargs = {
                    # todo can we zero here the second text encoder? or match a blank string?
                    "text_embeds": text_embeddings.pooled_embeds,
                    "time_ids": add_time_ids,
                }

            if self.model_config.refiner_name_or_path is not None:
                # we have the refiner on the second half of everything. Do Both
                if do_classifier_free_guidance:
                    raise ValueError("Refiner is not supported with classifier free guidance")

                if self.unet.training:
                    input_chunks = torch.chunk(latent_model_input, 2, dim=0)
                    timestep_chunks = torch.chunk(timestep, 2, dim=0)
                    added_cond_kwargs_chunked = {
                        "text_embeds": torch.chunk(text_embeddings.pooled_embeds, 2, dim=0),
                        "time_ids": torch.chunk(add_time_ids, 2, dim=0),
                    }
                    text_embeds_chunks = torch.chunk(text_embeddings.text_embeds, 2, dim=0)

                    # predict the noise residual
                    base_pred = self.unet(
                        input_chunks[0],
                        timestep_chunks[0],
                        encoder_hidden_states=text_embeds_chunks[0],
                        added_cond_kwargs={
                            "text_embeds": added_cond_kwargs_chunked['text_embeds'][0],
                            "time_ids": added_cond_kwargs_chunked['time_ids'][0],
                        },
                        **kwargs,
                    ).sample

                    refiner_pred = self.refiner_unet(
                        input_chunks[1],
                        timestep_chunks[1],
                        encoder_hidden_states=text_embeds_chunks[1][:, :, -1280:],
                        # just use the first second text encoder
                        added_cond_kwargs={
                            "text_embeds": added_cond_kwargs_chunked['text_embeds'][1],
                            # "time_ids": added_cond_kwargs_chunked['time_ids'][1],
                            "time_ids": self.get_time_ids_from_latents(input_chunks[1], requires_aesthetic_score=True),
                        },
                        **kwargs,
                    ).sample

                    noise_pred = torch.cat([base_pred, refiner_pred], dim=0)
                else:
                    noise_pred = self.refiner_unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embeddings.text_embeds[:, :, -1280:],
                        # just use the first second text encoder
                        added_cond_kwargs={
                            "text_embeds": text_embeddings.pooled_embeds,
                            "time_ids": self.get_time_ids_from_latents(latent_model_input,
                                                                       requires_aesthetic_score=True),
                        },
                        **kwargs,
                    ).sample

            else:

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input.to(self.device_torch, self.torch_dtype),
                    timestep,
                    encoder_hidden_states=text_embeddings.text_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    **kwargs,
                ).sample

            conditional_pred = noise_pred

            if do_classifier_free_guidance:
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                conditional_pred = noise_pred_text
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

                # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        else:
            with torch.no_grad():
                if do_classifier_free_guidance:
                    # if we are doing classifier free guidance, need to double up
                    latent_model_input = torch.cat([latents] * 2, dim=0)
                    timestep = torch.cat([timestep] * 2)
                else:
                    latent_model_input = latents

                latent_model_input = scale_model_input(latent_model_input, timestep)

                # check if we need to concat timesteps
                if isinstance(timestep, torch.Tensor) and len(timestep.shape) > 1:
                    ts_bs = timestep.shape[0]
                    if ts_bs != latent_model_input.shape[0]:
                        if ts_bs == 1:
                            timestep = torch.cat([timestep] * latent_model_input.shape[0])
                        elif ts_bs * 2 == latent_model_input.shape[0]:
                            timestep = torch.cat([timestep] * 2, dim=0)
                        else:
                            raise ValueError(
                                f"Batch size of latents {latent_model_input.shape[0]} must be the same or half the batch size of timesteps {timestep.shape[0]}")

            # predict the noise residual
            if self.is_pixart:
                VAE_SCALE_FACTOR = 2 ** (len(self.vae.config['block_out_channels']) - 1)
                batch_size, ch, h, w = list(latents.shape)

                height = h * VAE_SCALE_FACTOR
                width = w * VAE_SCALE_FACTOR

                if self.pipeline.transformer.config.sample_size == 256:
                    aspect_ratio_bin = ASPECT_RATIO_2048_BIN
                elif self.pipeline.transformer.config.sample_size == 128:
                    aspect_ratio_bin = ASPECT_RATIO_1024_BIN
                elif self.pipeline.transformer.config.sample_size == 64:
                    aspect_ratio_bin = ASPECT_RATIO_512_BIN
                elif self.pipeline.transformer.config.sample_size == 32:
                    aspect_ratio_bin = ASPECT_RATIO_256_BIN
                else:
                    raise ValueError(f"Invalid sample size: {self.pipeline.transformer.config.sample_size}")
                orig_height, orig_width = height, width
                height, width = self.pipeline.image_processor.classify_height_width_bin(height, width,
                                                                                        ratios=aspect_ratio_bin)

                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
                if self.unet_unwrapped.config.sample_size == 128 or (
                        self.vae_scale_factor == 16 and self.unet_unwrapped.config.sample_size == 64):
                    resolution = torch.tensor([height, width]).repeat(batch_size, 1)
                    aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
                    resolution = resolution.to(dtype=text_embeddings.text_embeds.dtype, device=self.device_torch)
                    aspect_ratio = aspect_ratio.to(dtype=text_embeddings.text_embeds.dtype, device=self.device_torch)

                    if do_classifier_free_guidance:
                        resolution = torch.cat([resolution, resolution], dim=0)
                        aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

                    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

                noise_pred = self.unet(
                    latent_model_input.to(self.device_torch, self.torch_dtype),
                    encoder_hidden_states=text_embeddings.text_embeds,
                    encoder_attention_mask=text_embeddings.attention_mask,
                    timestep=timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    **kwargs
                )[0]

                # learned sigma
                if self.unet_unwrapped.config.out_channels // 2 == self.unet_unwrapped.config.in_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred
            else:
                if self.unet.device != self.device_torch:
                    try:
                        self.unet.to(self.device_torch)
                    except Exception as e:
                        pass
                if self.unet.dtype != self.torch_dtype:
                    self.unet = self.unet.to(dtype=self.torch_dtype)
                if self.is_flux:
                    with torch.no_grad():

                        bs, c, h, w = latent_model_input.shape
                        latent_model_input_packed = rearrange(
                            latent_model_input,
                            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                            ph=2,
                            pw=2
                        )

                        img_ids = torch.zeros(h // 2, w // 2, 3)
                        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
                        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
                        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs).to(self.device_torch)

                        txt_ids = torch.zeros(bs, text_embeddings.text_embeds.shape[1], 3).to(self.device_torch)

                        # # handle guidance
                        if self.unet_unwrapped.config.guidance_embeds:
                            if isinstance(guidance_embedding_scale, list):
                                guidance = torch.tensor(guidance_embedding_scale, device=self.device_torch)
                            else:
                                guidance = torch.tensor([guidance_embedding_scale], device=self.device_torch)
                                guidance = guidance.expand(latents.shape[0])
                        else:
                            guidance = None
                    
                    if bypass_guidance_embedding:
                        bypass_flux_guidance(self.unet)

                    cast_dtype = self.unet.dtype
                    # changes from orig implementation
                    if txt_ids.ndim == 3:
                        txt_ids = txt_ids[0]
                    if img_ids.ndim == 3:
                        img_ids = img_ids[0]
                    # with torch.amp.autocast(device_type='cuda', dtype=cast_dtype):
                    noise_pred = self.unet(
                        hidden_states=latent_model_input_packed.to(self.device_torch, cast_dtype),  # [1, 4096, 64]
                        # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                        # todo make sure this doesnt change
                        timestep=timestep / 1000,  # timestep is 1000 scale
                        encoder_hidden_states=text_embeddings.text_embeds.to(self.device_torch, cast_dtype),
                        # [1, 512, 4096]
                        pooled_projections=text_embeddings.pooled_embeds.to(self.device_torch, cast_dtype),  # [1, 768]
                        txt_ids=txt_ids,  # [1, 512, 3]
                        img_ids=img_ids,  # [1, 4096, 3]
                        guidance=guidance,
                        return_dict=False,
                        **kwargs,
                    )[0]

                    if isinstance(noise_pred, QTensor):
                        noise_pred = noise_pred.dequantize()

                    noise_pred = rearrange(
                        noise_pred,
                        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                        h=latent_model_input.shape[2] // 2,
                        w=latent_model_input.shape[3] // 2,
                        ph=2,
                        pw=2,
                        # c=latent_model_input.shape[1],
                        c=self.vae.config.latent_channels
                    )
                    
                    if bypass_guidance_embedding:
                        restore_flux_guidance(self.unet)
                elif self.is_lumina2:
                    # reverse the timestep since Lumina uses t=0 as the noise and t=1 as the image
                    t = 1 - timestep / self.noise_scheduler.config.num_train_timesteps
                    with self.accelerator.autocast():
                        noise_pred = self.unet(
                            hidden_states=latent_model_input.to(self.device_torch, self.torch_dtype),
                            timestep=t,
                            encoder_attention_mask=text_embeddings.attention_mask.to(self.device_torch, dtype=torch.int64),
                            encoder_hidden_states=text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype),
                            **kwargs,
                        ).sample
                    
                    # lumina2 does this before stepping. Should we do it here?
                    noise_pred = -noise_pred
                elif self.is_v3:
                    noise_pred = self.unet(
                        hidden_states=latent_model_input.to(self.device_torch, self.torch_dtype),
                        timestep=timestep,
                        encoder_hidden_states=text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype),
                        pooled_projections=text_embeddings.pooled_embeds.to(self.device_torch, self.torch_dtype),
                        **kwargs,
                    ).sample
                    if isinstance(noise_pred, QTensor):
                        noise_pred = noise_pred.dequantize()
                elif self.is_auraflow:
                    # aura use timestep value between 0 and 1, with t=1 as noise and t=0 as the image
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    t = torch.tensor([timestep / 1000]).expand(latent_model_input.shape[0])
                    t = t.to(self.device_torch, self.torch_dtype)

                    noise_pred = self.unet(
                        latent_model_input,
                        encoder_hidden_states=text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype),
                        timestep=t,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = self.unet(
                        latent_model_input.to(self.device_torch, self.torch_dtype),
                        timestep=timestep,
                        encoder_hidden_states=text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype),
                        **kwargs,
                    ).sample

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
                        target_mean = target_pred_mean_std.mean([1, 2, 3], keepdim=True).detach()
                        target_std = target_pred_mean_std.std([1, 2, 3], keepdim=True).detach()

                        pred_mean = noise_pred.mean([1, 2, 3], keepdim=True).detach()
                        pred_std = noise_pred.std([1, 2, 3], keepdim=True).detach()

                    # match the mean and std
                    noise_pred = (noise_pred - pred_mean) / pred_std
                    noise_pred = (noise_pred * target_std) + target_mean

                # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        if return_conditional_pred:
            return noise_pred, conditional_pred
        return noise_pred

    def step_scheduler(self, model_input, latent_input, timestep_tensor, noise_scheduler=None):
        if noise_scheduler is None:
            noise_scheduler = self.noise_scheduler
        # // sometimes they are on the wrong device, no idea why
        if isinstance(noise_scheduler, DDPMScheduler) or isinstance(noise_scheduler, LCMScheduler):
            try:
                noise_scheduler.betas = noise_scheduler.betas.to(self.device_torch)
                noise_scheduler.alphas = noise_scheduler.alphas.to(self.device_torch)
                noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(self.device_torch)
            except Exception as e:
                pass

        mi_chunks = torch.chunk(model_input, model_input.shape[0], dim=0)
        latent_chunks = torch.chunk(latent_input, latent_input.shape[0], dim=0)
        timestep_chunks = torch.chunk(timestep_tensor, timestep_tensor.shape[0], dim=0)
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
            bypass_guidance_embedding=False,
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
                bypass_guidance_embedding=bypass_guidance_embedding,
                **kwargs,
            )
            # some schedulers need to run separately, so do that. (euler for example)

            if return_first_prediction and first_prediction is None:
                first_prediction = conditional_pred

            latents = self.step_scheduler(noise_pred, latents, timestep)

            # if not last step, and bleeding, bleed in some latents
            if bleed_latents is not None and timestep != self.noise_scheduler.timesteps[-1]:
                latents = (latents * (1 - bleed_ratio)) + (bleed_latents * bleed_ratio)

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
            control_images=None,
    ) -> PromptEmbeds:
        # sd1.5 embeddings are (bs, 77, 768)
        prompt = prompt
        # if it is not a list, make it one
        if not isinstance(prompt, list):
            prompt = [prompt]

        if prompt2 is not None and not isinstance(prompt2, list):
            prompt2 = [prompt2]
        if self.is_xl:
            # todo make this a config
            # 50% chance to use an encoder anyway even if it is disabled
            # allows the other TE to compensate for the disabled one
            # use_encoder_1 = self.use_text_encoder_1 or force_all or random.random() > 0.5
            # use_encoder_2 = self.use_text_encoder_2 or force_all or random.random() > 0.5
            use_encoder_1 = True
            use_encoder_2 = True

            return PromptEmbeds(
                train_tools.encode_prompts_xl(
                    self.tokenizer,
                    self.text_encoder,
                    prompt,
                    prompt2,
                    num_images_per_prompt=num_images_per_prompt,
                    use_text_encoder_1=use_encoder_1,
                    use_text_encoder_2=use_encoder_2,
                    truncate=not long_prompts,
                    max_length=max_length,
                    dropout_prob=dropout_prob,
                )
            )
        if self.is_v3:
            return PromptEmbeds(
                train_tools.encode_prompts_sd3(
                    self.tokenizer,
                    self.text_encoder,
                    prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    truncate=not long_prompts,
                    max_length=max_length,
                    dropout_prob=dropout_prob,
                    pipeline=self.pipeline,
                )
            )
        elif self.is_pixart:
            embeds, attention_mask = train_tools.encode_prompts_pixart(
                self.tokenizer,
                self.text_encoder,
                prompt,
                truncate=not long_prompts,
                max_length=300 if self.model_config.is_pixart_sigma else 120,
                dropout_prob=dropout_prob
            )
            return PromptEmbeds(
                embeds,
                attention_mask=attention_mask,
            )
        elif self.is_auraflow:
            embeds, attention_mask = train_tools.encode_prompts_auraflow(
                self.tokenizer,
                self.text_encoder,
                prompt,
                truncate=not long_prompts,
                max_length=256,
                dropout_prob=dropout_prob
            )
            return PromptEmbeds(
                embeds,
                attention_mask=attention_mask,  # not used
            )
        elif self.is_flux:
            prompt_embeds, pooled_prompt_embeds = train_tools.encode_prompts_flux(
                self.tokenizer,  # list
                self.text_encoder,  # list
                prompt,
                truncate=not long_prompts,
                max_length=512,
                dropout_prob=dropout_prob,
                attn_mask=self.model_config.attn_masking
            )
            pe = PromptEmbeds(
                prompt_embeds
            )
            pe.pooled_embeds = pooled_prompt_embeds
            return pe

        elif self.is_lumina2:
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.pipeline.encode_prompt(
                prompt,
                do_classifier_free_guidance=False,
                num_images_per_prompt=1,
                device=self.device_torch,
                max_sequence_length=256, # should it be 512?
            )
            return PromptEmbeds(
                prompt_embeds,
                attention_mask=prompt_attention_mask,
            )

        elif isinstance(self.text_encoder, T5EncoderModel):
            embeds, attention_mask = train_tools.encode_prompts_pixart(
                self.tokenizer,
                self.text_encoder,
                prompt,
                truncate=not long_prompts,
                max_length=256,
                dropout_prob=dropout_prob
            )
            
            # just mask the attention mask
            prompt_attention_mask = attention_mask.unsqueeze(-1).expand(embeds.shape)
            embeds = embeds * prompt_attention_mask.to(dtype=embeds.dtype, device=embeds.device)
            return PromptEmbeds(
                embeds,
                
                # do we want attn mask here?
                # attention_mask=attention_mask,
            )
        else:
            return PromptEmbeds(
                train_tools.encode_prompts(
                    self.tokenizer,
                    self.text_encoder,
                    prompt,
                    truncate=not long_prompts,
                    max_length=max_length,
                    dropout_prob=dropout_prob
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

        VAE_SCALE_FACTOR = 2 ** (len(self.vae.config['block_out_channels']) - 1)

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
            self.vae.to(self.device_torch)
        latents = latents.to(self.device_torch, dtype=self.torch_dtype)
        latents = (latents / self.vae.config['scaling_factor']) + self.vae.config['shift_factor']
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
            if self.is_flux or self.is_lumina2:
                for name, param in self.unet.named_parameters(recurse=True, prefix="transformer"):
                    named_params[name] = param
            else:
                for name, param in self.unet.named_parameters(recurse=True, prefix=f"{SD_PREFIX_UNET}"):
                    named_params[name] = param
            
            if self.model_config.ignore_if_contains is not None:
                # remove params that contain the ignore_if_contains from named params
                for key in list(named_params.keys()):
                    if any([s in key for s in self.model_config.ignore_if_contains]):
                        del named_params[key]
            if self.model_config.only_if_contains is not None:
                # remove params that do not contain the only_if_contains from named params
                for key in list(named_params.keys()):
                    if not any([s in key for s in self.model_config.only_if_contains]):
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

    def save_refiner(self, output_file: str, meta: OrderedDict, save_dtype=get_torch_dtype('fp16')):

        # load the full refiner since we only train unet
        if self.model_config.refiner_name_or_path is None:
            raise ValueError("Refiner must be specified to save it")
        refiner_config_path = os.path.join(ORIG_CONFIGS_ROOT, 'sd_xl_refiner.yaml')
        # load the refiner model
        dtype = get_torch_dtype(self.dtype)
        model_path = self.model_config._original_refiner_name_or_path
        if not os.path.exists(model_path) or os.path.isdir(model_path):
            # TODO only load unet??
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_path,
                dtype=dtype,
                device='cpu',
                # variant="fp16",
                use_safetensors=True,
            )
        else:
            refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
                model_path,
                dtype=dtype,
                device='cpu',
                torch_dtype=self.torch_dtype,
                original_config_file=refiner_config_path,
            )
        # replace original unet
        refiner.unet = self.refiner_unet
        flush()

        diffusers_state_dict = OrderedDict()
        for k, v in refiner.vae.state_dict().items():
            new_key = k if k.startswith(f"{SD_PREFIX_VAE}") else f"{SD_PREFIX_VAE}_{k}"
            diffusers_state_dict[new_key] = v
        for k, v in refiner.text_encoder_2.state_dict().items():
            new_key = k if k.startswith(f"{SD_PREFIX_TEXT_ENCODER2}_") else f"{SD_PREFIX_TEXT_ENCODER2}_{k}"
            diffusers_state_dict[new_key] = v
        for k, v in refiner.unet.state_dict().items():
            new_key = k if k.startswith(f"{SD_PREFIX_UNET}_") else f"{SD_PREFIX_UNET}_{k}"
            diffusers_state_dict[new_key] = v

        converted_state_dict = get_ldm_state_dict_from_diffusers(
            diffusers_state_dict,
            'sdxl_refiner',
            device='cpu',
            dtype=save_dtype
        )

        # make sure parent folder exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(converted_state_dict, output_file, metadata=meta)

        if self.config_file is not None:
            output_path_no_ext = os.path.splitext(output_file)[0]
            output_config_path = f"{output_path_no_ext}.yaml"
            shutil.copyfile(self.config_file, output_config_path)

    def save(self, output_file: str, meta: OrderedDict, save_dtype=get_torch_dtype('fp16'), logit_scale=None):
        version_string = '1'
        if self.is_v2:
            version_string = '2'
        if self.is_xl:
            version_string = 'sdxl'
        if self.is_ssd:
            # overwrite sdxl because both wil be true here
            version_string = 'ssd'
        if self.is_ssd and self.is_vega:
            version_string = 'vega'
        # if output file does not end in .safetensors, then it is a directory and we are
        # saving in diffusers format
        if not output_file.endswith('.safetensors'):
            # diffusers
            if self.is_flux:
                # only save the unet
                transformer: FluxTransformer2DModel = unwrap_model(self.unet)
                transformer.save_pretrained(
                    save_directory=os.path.join(output_file, 'transformer'),
                    safe_serialization=True,
                )
            elif self.is_lumina2:
                # only save the unet
                transformer: Lumina2Transformer2DModel = unwrap_model(self.unet)
                transformer.save_pretrained(
                    save_directory=os.path.join(output_file, 'transformer'),
                    safe_serialization=True,
                )
                
            else:

                self.pipeline.save_pretrained(
                    save_directory=output_file,
                    safe_serialization=True,
                )
            # save out meta config
            meta_path = os.path.join(output_file, 'aitk_meta.yaml')
            with open(meta_path, 'w') as f:
                yaml.dump(meta, f)

        else:
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
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        ldm_diffusers_keymap = mapping['ldm_diffusers_keymap']

        trainable_parameters = []

        # we use state dict to find params

        if unet:
            named_params = self.named_parameters(vae=False, unet=unet, text_encoder=False, state_dict_keys=True)
            unet_lr = unet_lr if unet_lr is not None else default_lr
            params = []
            if self.is_pixart or self.is_auraflow or self.is_flux or self.is_v3 or self.is_lumina2:
                for param in named_params.values():
                    if param.requires_grad:
                        params.append(param)
            else:
                for key, diffusers_key in ldm_diffusers_keymap.items():
                    if diffusers_key in named_params and diffusers_key not in DO_NOT_TRAIN_WEIGHTS:
                        if named_params[diffusers_key].requires_grad:
                            params.append(named_params[diffusers_key])
            param_data = {"params": params, "lr": unet_lr}
            trainable_parameters.append(param_data)
            print_acc(f"Found {len(params)} trainable parameter in unet")

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

            print_acc(f"Found {len(params)} trainable parameter in text encoder")

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
        unet_has_grad = False

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
                if isinstance(encoder, LlamaModel):
                    te_has_grad = encoder.layers[0].mlp.gate_proj.weight.requires_grad
                else:
                    try:
                        te_has_grad = encoder.text_model.final_layer_norm.weight.requires_grad
                    except:
                        te_has_grad = encoder.encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad
                self.device_state['text_encoder'].append({
                    'training': encoder.training,
                    'device': encoder.device,
                    # todo there has to be a better way to do this
                    'requires_grad': te_has_grad
                })
        else:
            if isinstance(self.text_encoder, T5EncoderModel) or isinstance(self.text_encoder, UMT5EncoderModel):
                te_has_grad = self.text_encoder.encoder.block[0].layer[0].SelfAttention.q.weight.requires_grad
            elif isinstance(self.text_encoder, Gemma2Model):
                te_has_grad = self.text_encoder.layers[0].mlp.gate_proj.weight.requires_grad
            elif isinstance(self.text_encoder, Qwen2Model):
                te_has_grad = self.text_encoder.layers[0].mlp.gate_proj.weight.requires_grad
            elif isinstance(self.text_encoder, LlamaModel):
                te_has_grad = self.text_encoder.layers[0].mlp.gate_proj.weight.requires_grad
            else:
                te_has_grad = self.text_encoder.text_model.final_layer_norm.weight.requires_grad

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
                    encoder.requires_grad_(state['text_encoder'][i]['requires_grad'])
                else:
                    if state['text_encoder']['training']:
                        encoder.train()
                    else:
                        encoder.eval()
                    encoder.to(state['text_encoder']['device'])
                    encoder.requires_grad_(state['text_encoder']['requires_grad'])
        else:
            if state['text_encoder']['training']:
                self.text_encoder.train()
            else:
                self.text_encoder.eval()
            self.text_encoder.to(state['text_encoder']['device'])
            self.text_encoder.requires_grad_(state['text_encoder']['requires_grad'])

        if self.adapter is not None:
            self.adapter.to(state['adapter']['device'])
            self.adapter.requires_grad_(state['adapter']['requires_grad'])
            if state['adapter']['training']:
                self.adapter.train()
            else:
                self.adapter.eval()

        if self.refiner_unet is not None:
            self.refiner_unet.to(state['refiner_unet']['device'])
            self.refiner_unet.requires_grad_(state['refiner_unet']['requires_grad'])
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
        if device_state_preset in ['cache_text_encoder']:
            active_modules = ['text_encoder']
        if device_state_preset in ['unload']:
            active_modules = []
        if device_state_preset in ['generate']:
            active_modules = ['vae', 'unet', 'text_encoder', 'adapter', 'refiner_unet']

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
        if self.is_pixart:
            return 'pixart'
        if self.is_v3:
            return 'sd_3'
        if self.is_auraflow:
            return 'auraflow'
        if self.is_flux:
            return 'flux.1'
        if self.is_lumina2:
            return 'lumina2'
        if self.is_ssd:
            return 'ssd'
        if self.is_vega:
            return 'vega'
        if self.is_xl:
            return 'sdxl_1.0'
        if self.is_v2:
            return 'sd_2.1'
        return 'sd_1.5'

    def get_model_to_train(self):
        return self.unet
