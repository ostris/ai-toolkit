import os
import time
from typing import List, Optional, Literal, Tuple, Union, TYPE_CHECKING, Dict
import random

import torch
import torchaudio

from toolkit.prompt_utils import PromptEmbeds

ImgExt = Literal['jpg', 'png', 'webp']

SaveFormat = Literal['safetensors', 'diffusers']

if TYPE_CHECKING:
    from toolkit.guidance import GuidanceType
    from toolkit.logging_aitk import EmptyLogger
else:
    EmptyLogger = None

class SaveConfig:
    def __init__(self, **kwargs):
        self.save_every: int = kwargs.get('save_every', 1000)
        self.dtype: str = kwargs.get('dtype', 'float16')
        self.max_step_saves_to_keep: int = kwargs.get('max_step_saves_to_keep', 5)
        self.save_format: SaveFormat = kwargs.get('save_format', 'safetensors')
        if self.save_format not in ['safetensors', 'diffusers']:
            raise ValueError(f"save_format must be safetensors or diffusers, got {self.save_format}")
        self.push_to_hub: bool = kwargs.get("push_to_hub", False)
        self.hf_repo_id: Optional[str] = kwargs.get("hf_repo_id", None)
        self.hf_private: Optional[str] = kwargs.get("hf_private", False)

class LoggingConfig:
    def __init__(self, **kwargs):
        self.log_every: int = kwargs.get('log_every', 100)
        self.verbose: bool = kwargs.get('verbose', False)
        self.use_wandb: bool = kwargs.get('use_wandb', False)
        self.project_name: str = kwargs.get('project_name', 'ai-toolkit')
        self.run_name: str = kwargs.get('run_name', None)

class SampleItem:
    def __init__(
        self,
        sample_config: 'SampleConfig',
        **kwargs
    ):
        # prompt should always be in the kwargs
        self.prompt = kwargs.get('prompt', None)
        self.width: int = kwargs.get('width', sample_config.width)
        self.height: int = kwargs.get('height', sample_config.height)
        self.neg: str = kwargs.get('neg', sample_config.neg)
        self.seed: Optional[int] = kwargs.get('seed', None) # if none, default to autogen seed
        self.guidance_scale: float = kwargs.get('guidance_scale', sample_config.guidance_scale)
        self.sample_steps: int = kwargs.get('sample_steps', sample_config.sample_steps)
        self.fps: int = kwargs.get('fps', sample_config.fps)
        self.num_frames: int = kwargs.get('num_frames', sample_config.num_frames)
        self.ctrl_img: Optional[str] = kwargs.get('ctrl_img', None)
        self.ctrl_idx: int = kwargs.get('ctrl_idx', 0)
        self.network_multiplier: float = kwargs.get('network_multiplier', sample_config.network_multiplier)
        

class SampleConfig:
    def __init__(self, **kwargs):
        self.sampler: str = kwargs.get('sampler', 'ddpm')
        self.sample_every: int = kwargs.get('sample_every', 100)
        self.width: int = kwargs.get('width', 512)
        self.height: int = kwargs.get('height', 512)
        self.neg = kwargs.get('neg', False)
        self.seed = kwargs.get('seed', 0)
        self.walk_seed = kwargs.get('walk_seed', False)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)
        self.network_multiplier = kwargs.get('network_multiplier', 1)
        self.guidance_rescale = kwargs.get('guidance_rescale', 0.0)
        self.ext: ImgExt = kwargs.get('format', 'jpg')
        self.adapter_conditioning_scale = kwargs.get('adapter_conditioning_scale', 1.0)
        self.refiner_start_at = kwargs.get('refiner_start_at',
                                           0.5)  # step to start using refiner on sample if it exists
        self.extra_values = kwargs.get('extra_values', [])
        self.num_frames = kwargs.get('num_frames', 1)
        self.fps: int = kwargs.get('fps', 16)
        if self.num_frames > 1 and self.ext not in ['webp']:
            print("Changing sample extention to animated webp")
            self.ext = 'webp'
        
        prompts: list[str] = kwargs.get('prompts', [])
        
        self.samples: Optional[List[SampleItem]] = None
        # use the legacy prompts if it is passed that way to get samples object
        default_samples_kwargs = [
            {"prompt": x} for x in prompts
        ]
        raw_samples = kwargs.get('samples', default_samples_kwargs)
        self.samples = [SampleItem(self, **item) for item in raw_samples]
        
    @property
    def prompts(self):
        # for backwards compatibility as this is checked for length frequently
        return [sample.prompt for sample in self.samples if sample.prompt is not None]
  
                


class LormModuleSettingsConfig:
    def __init__(self, **kwargs):
        self.contains: str = kwargs.get('contains', '4nt$3')
        self.extract_mode: str = kwargs.get('extract_mode', 'ratio')
        # min num parameters to attach to
        self.parameter_threshold: int = kwargs.get('parameter_threshold', 0)
        self.extract_mode_param: dict = kwargs.get('extract_mode_param', 0.25)


class LoRMConfig:
    def __init__(self, **kwargs):
        self.extract_mode: str = kwargs.get('extract_mode', 'ratio')
        self.do_conv: bool = kwargs.get('do_conv', False)
        self.extract_mode_param: dict = kwargs.get('extract_mode_param', 0.25)
        self.parameter_threshold: int = kwargs.get('parameter_threshold', 0)
        module_settings = kwargs.get('module_settings', [])
        default_module_settings = {
            'extract_mode': self.extract_mode,
            'extract_mode_param': self.extract_mode_param,
            'parameter_threshold': self.parameter_threshold,
        }
        module_settings = [{**default_module_settings, **module_setting, } for module_setting in module_settings]
        self.module_settings: List[LormModuleSettingsConfig] = [LormModuleSettingsConfig(**module_setting) for
                                                                module_setting in module_settings]

    def get_config_for_module(self, block_name):
        for setting in self.module_settings:
            contain_pieces = setting.contains.split('|')
            if all(contain_piece in block_name for contain_piece in contain_pieces):
                return setting
            # try replacing the . with _
            contain_pieces = setting.contains.replace('.', '_').split('|')
            if all(contain_piece in block_name for contain_piece in contain_pieces):
                return setting
            # do default
        return LormModuleSettingsConfig(**{
            'extract_mode': self.extract_mode,
            'extract_mode_param': self.extract_mode_param,
            'parameter_threshold': self.parameter_threshold,
        })


NetworkType = Literal['lora', 'locon', 'lorm', 'lokr']


class NetworkConfig:
    def __init__(self, **kwargs):
        self.type: NetworkType = kwargs.get('type', 'lora')
        rank = kwargs.get('rank', None)
        linear = kwargs.get('linear', None)
        if rank is not None:
            self.rank: int = rank  # rank for backward compatibility
            self.linear: int = rank
        elif linear is not None:
            self.rank: int = linear
            self.linear: int = linear
        self.conv: int = kwargs.get('conv', None)
        self.alpha: float = kwargs.get('alpha', 1.0)
        self.linear_alpha: float = kwargs.get('linear_alpha', self.alpha)
        self.conv_alpha: float = kwargs.get('conv_alpha', self.conv)
        self.dropout: Union[float, None] = kwargs.get('dropout', None)
        self.network_kwargs: dict = kwargs.get('network_kwargs', {})

        self.lorm_config: Union[LoRMConfig, None] = None
        lorm = kwargs.get('lorm', None)
        if lorm is not None:
            self.lorm_config: LoRMConfig = LoRMConfig(**lorm)

        if self.type == 'lorm':
            # set linear to arbitrary values so it makes them
            self.linear = 4
            self.rank = 4
            if self.lorm_config.do_conv:
                self.conv = 4

        self.transformer_only = kwargs.get('transformer_only', True)
        
        self.lokr_full_rank = kwargs.get('lokr_full_rank', False)
        if self.lokr_full_rank and self.type.lower() == 'lokr':
            self.linear = 9999999999
            self.linear_alpha = 9999999999
            self.conv = 9999999999
            self.conv_alpha = 9999999999
        # -1 automatically finds the largest factor
        self.lokr_factor = kwargs.get('lokr_factor', -1)
        
        # for multi stage models
        self.split_multistage_loras = kwargs.get('split_multistage_loras', True)


AdapterTypes = Literal['t2i', 'ip', 'ip+', 'clip', 'ilora', 'photo_maker', 'control_net', 'control_lora', 'i2v']

CLIPLayer = Literal['penultimate_hidden_states', 'image_embeds', 'last_hidden_state']


class AdapterConfig:
    def __init__(self, **kwargs):
        self.type: AdapterTypes = kwargs.get('type', 't2i')  # t2i, ip, clip, control_net, i2v
        self.in_channels: int = kwargs.get('in_channels', 3)
        self.channels: List[int] = kwargs.get('channels', [320, 640, 1280, 1280])
        self.num_res_blocks: int = kwargs.get('num_res_blocks', 2)
        self.downscale_factor: int = kwargs.get('downscale_factor', 8)
        self.adapter_type: str = kwargs.get('adapter_type', 'full_adapter')
        self.image_dir: str = kwargs.get('image_dir', None)
        self.test_img_path: List[str] = kwargs.get('test_img_path', None)
        if self.test_img_path is not None:
            if isinstance(self.test_img_path, str):
                self.test_img_path = self.test_img_path.split(',')
                self.test_img_path = [p.strip() for p in self.test_img_path]
                self.test_img_path = [p for p in self.test_img_path if p != '']
                
        self.train: str = kwargs.get('train', False)
        self.image_encoder_path: str = kwargs.get('image_encoder_path', None)
        self.name_or_path = kwargs.get('name_or_path', None)

        num_tokens = kwargs.get('num_tokens', None)
        if num_tokens is None and self.type.startswith('ip'):
            if self.type == 'ip+':
                num_tokens = 16
                num_tokens = 16
            elif self.type == 'ip':
                num_tokens = 4

        self.num_tokens: int = num_tokens
        self.train_image_encoder: bool = kwargs.get('train_image_encoder', False)
        self.train_only_image_encoder: bool = kwargs.get('train_only_image_encoder', False)
        if self.train_only_image_encoder:
            self.train_image_encoder = True
        self.train_only_image_encoder_positional_embedding: bool = kwargs.get(
            'train_only_image_encoder_positional_embedding', False)
        self.image_encoder_arch: str = kwargs.get('image_encoder_arch', 'clip')  # clip vit vit_hybrid, safe
        self.safe_reducer_channels: int = kwargs.get('safe_reducer_channels', 512)
        self.safe_channels: int = kwargs.get('safe_channels', 2048)
        self.safe_tokens: int = kwargs.get('safe_tokens', 8)
        self.quad_image: bool = kwargs.get('quad_image', False)

        # clip vision
        self.trigger = kwargs.get('trigger', 'tri993r')
        self.trigger_class_name = kwargs.get('trigger_class_name', None)

        self.class_names = kwargs.get('class_names', [])

        self.clip_layer: CLIPLayer = kwargs.get('clip_layer', None)
        if self.clip_layer is None:
            if self.type.startswith('ip+'):
                self.clip_layer = 'penultimate_hidden_states'
            else:
                self.clip_layer = 'last_hidden_state'

        # text encoder
        self.text_encoder_path: str = kwargs.get('text_encoder_path', None)
        self.text_encoder_arch: str = kwargs.get('text_encoder_arch', 'clip')  # clip t5

        self.train_scaler: bool = kwargs.get('train_scaler', False)
        self.scaler_lr: Optional[float] = kwargs.get('scaler_lr', None)

        # trains with a scaler to easy channel bias but merges it in on save
        self.merge_scaler: bool = kwargs.get('merge_scaler', False)

        # for ilora
        self.head_dim: int = kwargs.get('head_dim', 1024)
        self.num_heads: int = kwargs.get('num_heads', 1)
        self.ilora_down: bool = kwargs.get('ilora_down', True)
        self.ilora_mid: bool = kwargs.get('ilora_mid', True)
        self.ilora_up: bool = kwargs.get('ilora_up', True)
        
        self.pixtral_max_image_size: int = kwargs.get('pixtral_max_image_size', 512)
        self.pixtral_random_image_size: int = kwargs.get('pixtral_random_image_size', False)

        self.flux_only_double: bool = kwargs.get('flux_only_double', False)
        
        # train and use a conv layer to pool the embedding
        self.conv_pooling: bool = kwargs.get('conv_pooling', False)
        self.conv_pooling_stacks: int = kwargs.get('conv_pooling_stacks', 1)
        self.sparse_autoencoder_dim: Optional[int] = kwargs.get('sparse_autoencoder_dim', None)
        
        # for llm adapter
        self.num_cloned_blocks: int = kwargs.get('num_cloned_blocks', 0)
        self.quantize_llm: bool = kwargs.get('quantize_llm', False)
        
        # for control lora only
        lora_config: dict = kwargs.get('lora_config', None)
        if lora_config is not None:
            self.lora_config: NetworkConfig = NetworkConfig(**lora_config)
        else:
            self.lora_config = None
        self.num_control_images: int = kwargs.get('num_control_images', 1)
        # decimal for how often the control is dropped out and replaced with noise 1.0 is 100%
        self.control_image_dropout: float = kwargs.get('control_image_dropout', 0.0)
        self.has_inpainting_input: bool = kwargs.get('has_inpainting_input', False)
        self.invert_inpaint_mask_chance: float = kwargs.get('invert_inpaint_mask_chance', 0.0)
        
        # for subpixel adapter
        self.subpixel_downscale_factor: int = kwargs.get('subpixel_downscale_factor', 8)
        
        # for i2v adapter
        # append the masked start frame. During pretraining we will only do the vision encoder
        self.i2v_do_start_frame: bool = kwargs.get('i2v_do_start_frame', False)


class EmbeddingConfig:
    def __init__(self, **kwargs):
        self.trigger = kwargs.get('trigger', 'custom_embedding')
        self.tokens = kwargs.get('tokens', 4)
        self.init_words = kwargs.get('init_words', '*')
        self.save_format = kwargs.get('save_format', 'safetensors')
        self.trigger_class_name = kwargs.get('trigger_class_name', None)  # used for inverted masked prior


class DecoratorConfig:
    def __init__(self, **kwargs):
        self.num_tokens: str = kwargs.get('num_tokens', 4)


ContentOrStyleType = Literal['balanced', 'style', 'content']
LossTarget = Literal['noise', 'source', 'unaugmented', 'differential_noise']


class TrainConfig:
    def __init__(self, **kwargs):
        self.noise_scheduler = kwargs.get('noise_scheduler', 'ddpm')
        self.content_or_style: ContentOrStyleType = kwargs.get('content_or_style', 'balanced')
        self.content_or_style_reg: ContentOrStyleType = kwargs.get('content_or_style', 'balanced')
        self.steps: int = kwargs.get('steps', 1000)
        self.lr = kwargs.get('lr', 1e-6)
        self.unet_lr = kwargs.get('unet_lr', self.lr)
        self.text_encoder_lr = kwargs.get('text_encoder_lr', self.lr)
        self.refiner_lr = kwargs.get('refiner_lr', self.lr)
        self.embedding_lr = kwargs.get('embedding_lr', self.lr)
        self.adapter_lr = kwargs.get('adapter_lr', self.lr)
        self.optimizer = kwargs.get('optimizer', 'adamw')
        self.optimizer_params = kwargs.get('optimizer_params', {})
        self.lr_scheduler = kwargs.get('lr_scheduler', 'constant')
        self.lr_scheduler_params = kwargs.get('lr_scheduler_params', {})
        self.min_denoising_steps: int = kwargs.get('min_denoising_steps', 0)
        self.max_denoising_steps: int = kwargs.get('max_denoising_steps', 999)
        self.batch_size: int = kwargs.get('batch_size', 1)
        self.orig_batch_size: int = self.batch_size
        self.dtype: str = kwargs.get('dtype', 'fp32')
        self.xformers = kwargs.get('xformers', False)
        self.sdp = kwargs.get('sdp', False)
        self.train_unet = kwargs.get('train_unet', True)
        self.train_text_encoder = kwargs.get('train_text_encoder', False)
        self.train_refiner = kwargs.get('train_refiner', True)
        self.train_turbo = kwargs.get('train_turbo', False)
        self.show_turbo_outputs = kwargs.get('show_turbo_outputs', False)
        self.min_snr_gamma = kwargs.get('min_snr_gamma', None)
        self.snr_gamma = kwargs.get('snr_gamma', None)
        # trains a gamma, offset, and scale to adjust loss to adapt to timestep differentials
        # this should balance the learning rate across all timesteps over time
        self.learnable_snr_gos = kwargs.get('learnable_snr_gos', False)
        self.noise_offset = kwargs.get('noise_offset', 0.0)
        self.skip_first_sample = kwargs.get('skip_first_sample', False)
        self.force_first_sample = kwargs.get('force_first_sample', False)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
        self.weight_jitter = kwargs.get('weight_jitter', 0.0)
        self.merge_network_on_save = kwargs.get('merge_network_on_save', False)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.start_step = kwargs.get('start_step', None)
        self.free_u = kwargs.get('free_u', False)
        self.adapter_assist_name_or_path: Optional[str] = kwargs.get('adapter_assist_name_or_path', None)
        self.adapter_assist_type: Optional[str] = kwargs.get('adapter_assist_type', 't2i')  # t2i, control_net
        self.noise_multiplier = kwargs.get('noise_multiplier', 1.0)
        self.target_noise_multiplier = kwargs.get('target_noise_multiplier', 1.0)
        self.random_noise_multiplier = kwargs.get('random_noise_multiplier', 0.0)
        self.random_noise_shift = kwargs.get('random_noise_shift', 0.0)
        self.img_multiplier = kwargs.get('img_multiplier', 1.0)
        self.noisy_latent_multiplier = kwargs.get('noisy_latent_multiplier', 1.0)
        self.latent_multiplier = kwargs.get('latent_multiplier', 1.0)
        self.negative_prompt = kwargs.get('negative_prompt', None)
        self.max_negative_prompts = kwargs.get('max_negative_prompts', 1)
        # multiplier applied to loos on regularization images
        self.reg_weight = kwargs.get('reg_weight', 1.0)
        self.num_train_timesteps = kwargs.get('num_train_timesteps', 1000)
        # automatically adapte the vae scaling based on the image norm
        self.adaptive_scaling_factor = kwargs.get('adaptive_scaling_factor', False)

        # dropout that happens before encoding. It functions independently per text encoder
        self.prompt_dropout_prob = kwargs.get('prompt_dropout_prob', 0.0)

        # match the norm of the noise before computing loss. This will help the model maintain its
        # current understandin of the brightness of images.

        self.match_noise_norm = kwargs.get('match_noise_norm', False)

        # set to -1 to accumulate gradients for entire epoch
        # warning, only do this with a small dataset or you will run out of memory
        # This is legacy but left in for backwards compatibility
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)

        # this will do proper gradient accumulation where you will not see a step until the end of the accumulation
        # the method above will show a step every accumulation
        self.gradient_accumulation = kwargs.get('gradient_accumulation', 1)
        if self.gradient_accumulation > 1:
            if self.gradient_accumulation_steps != 1:
                raise ValueError("gradient_accumulation and gradient_accumulation_steps are mutually exclusive")

        # short long captions will double your batch size. This only works when a dataset is
        # prepared with a json caption file that has both short and long captions in it. It will
        # Double up every image and run it through with both short and long captions. The idea
        # is that the network will learn how to generate good images with both short and long captions
        self.short_and_long_captions = kwargs.get('short_and_long_captions', False)
        # if above is NOT true, this will make it so the long caption foes to te2 and the short caption goes to te1 for sdxl only
        self.short_and_long_captions_encoder_split = kwargs.get('short_and_long_captions_encoder_split', False)

        # basically gradient accumulation but we run just 1 item through the network
        # and accumulate gradients. This can be used as basic gradient accumulation but is very helpful
        # for training tricks that increase batch size but need a single gradient step
        self.single_item_batching = kwargs.get('single_item_batching', False)

        match_adapter_assist = kwargs.get('match_adapter_assist', False)
        self.match_adapter_chance = kwargs.get('match_adapter_chance', 0.0)
        self.loss_target: LossTarget = kwargs.get('loss_target',
                                                  'noise')  # noise, source, unaugmented, differential_noise

        # When a mask is passed in a dataset, and this is true,
        # we will predict noise without a the LoRa network and use the prediction as a target for
        # unmasked reign. It is unmasked regularization basically
        self.inverted_mask_prior = kwargs.get('inverted_mask_prior', False)
        self.inverted_mask_prior_multiplier = kwargs.get('inverted_mask_prior_multiplier', 0.5)
        
        # DOP will will run the same image and prompt through the network without the trigger word blank and use it as a target
        self.diff_output_preservation = kwargs.get('diff_output_preservation', False)
        self.diff_output_preservation_multiplier = kwargs.get('diff_output_preservation_multiplier', 1.0)
        # If the trigger word is in the prompt, we will use this class name to replace it eg. "sks woman" -> "woman"
        self.diff_output_preservation_class = kwargs.get('diff_output_preservation_class', '')

        # legacy
        if match_adapter_assist and self.match_adapter_chance == 0.0:
            self.match_adapter_chance = 1.0

        # standardize inputs to the meand std of the model knowledge
        self.standardize_images = kwargs.get('standardize_images', False)
        self.standardize_latents = kwargs.get('standardize_latents', False)

        # if self.train_turbo and not self.noise_scheduler.startswith("euler"):
        #     raise ValueError(f"train_turbo is only supported with euler and wuler_a noise schedulers")

        self.dynamic_noise_offset = kwargs.get('dynamic_noise_offset', False)
        self.do_cfg = kwargs.get('do_cfg', False)
        self.do_random_cfg = kwargs.get('do_random_cfg', False)
        self.cfg_scale = kwargs.get('cfg_scale', 1.0)
        self.max_cfg_scale = kwargs.get('max_cfg_scale', self.cfg_scale)
        self.cfg_rescale = kwargs.get('cfg_rescale', None)
        if self.cfg_rescale is None:
            self.cfg_rescale = self.cfg_scale

        # applies the inverse of the prediction mean and std to the target to correct
        # for norm drift
        self.correct_pred_norm = kwargs.get('correct_pred_norm', False)
        self.correct_pred_norm_multiplier = kwargs.get('correct_pred_norm_multiplier', 1.0)

        self.loss_type = kwargs.get('loss_type', 'mse') # mse, mae, wavelet, pixelspace, mean_flow

        # scale the prediction by this. Increase for more detail, decrease for less
        self.pred_scaler = kwargs.get('pred_scaler', 1.0)

        # repeats the prompt a few times to saturate the encoder
        self.prompt_saturation_chance = kwargs.get('prompt_saturation_chance', 0.0)

        # applies negative loss on the prior to encourage network to diverge from it
        self.do_prior_divergence = kwargs.get('do_prior_divergence', False)

        ema_config: Union[Dict, None] = kwargs.get('ema_config', None)
        # if it is set explicitly to false, leave it false. 
        if ema_config is not None and ema_config.get('use_ema', False):
            ema_config['use_ema'] = True
            print(f"Using EMA")
        else:
            ema_config = {'use_ema': False}

        self.ema_config: EMAConfig = EMAConfig(**ema_config)

        # adds an additional loss to the network to encourage it output a normalized standard deviation
        self.target_norm_std = kwargs.get('target_norm_std', None)
        self.target_norm_std_value = kwargs.get('target_norm_std_value', 1.0)
        self.timestep_type = kwargs.get('timestep_type', 'sigmoid')  # sigmoid, linear, lognorm_blend, next_sample, weighted, one_step
        self.next_sample_timesteps = kwargs.get('next_sample_timesteps', 8)
        self.linear_timesteps = kwargs.get('linear_timesteps', False)
        self.linear_timesteps2 = kwargs.get('linear_timesteps2', False)
        self.disable_sampling = kwargs.get('disable_sampling', False)

        # will cache a blank prompt or the trigger word, and unload the text encoder to cpu
        # will make training faster and use less vram
        self.unload_text_encoder = kwargs.get('unload_text_encoder', False)
        # will toggle all datasets to cache text embeddings
        self.cache_text_embeddings: bool = kwargs.get('cache_text_embeddings', False)
        # for swapping which parameters are trained during training
        self.do_paramiter_swapping = kwargs.get('do_paramiter_swapping', False)
        # 0.1 is 10% of the parameters active at a time lower is less vram, higher is more
        self.paramiter_swapping_factor = kwargs.get('paramiter_swapping_factor', 0.1)
        # bypass the guidance embedding for training. For open flux with guidance embedding
        self.bypass_guidance_embedding = kwargs.get('bypass_guidance_embedding', False)
        
        # diffusion feature extractor
        self.latent_feature_extractor_path = kwargs.get('latent_feature_extractor_path', None)
        self.latent_feature_loss_weight = kwargs.get('latent_feature_loss_weight', 1.0)
        
        # we use this in the code, but it really needs to be called latent_feature_extractor as that makes more sense with new architecture
        self.diffusion_feature_extractor_path = kwargs.get('diffusion_feature_extractor_path', self.latent_feature_extractor_path)
        self.diffusion_feature_extractor_weight = kwargs.get('diffusion_feature_extractor_weight', self.latent_feature_loss_weight)
        
        # optimal noise pairing
        self.optimal_noise_pairing_samples = kwargs.get('optimal_noise_pairing_samples', 1)
        
        # forces same noise for the same image at a given size.
        self.force_consistent_noise = kwargs.get('force_consistent_noise', False)
        self.blended_blur_noise = kwargs.get('blended_blur_noise', False)
        
        # contrastive loss
        self.do_guidance_loss = kwargs.get('do_guidance_loss', False)
        self.guidance_loss_target: Union[int, List[int, int]] = kwargs.get('guidance_loss_target', 3.0)
        self.unconditional_prompt: str = kwargs.get('unconditional_prompt', '')
        if isinstance(self.guidance_loss_target, tuple):
            self.guidance_loss_target = list(self.guidance_loss_target)
        
        # for multi stage models, how often to switch the boundary
        self.switch_boundary_every: int = kwargs.get('switch_boundary_every', 1)


ModelArch = Literal['sd1', 'sd2', 'sd3', 'sdxl', 'pixart', 'pixart_sigma', 'auraflow', 'flux', 'flex1', 'flex2', 'lumina2', 'vega', 'ssd', 'wan21']


class ModelConfig:
    def __init__(self, **kwargs):
        self.name_or_path: str = kwargs.get('name_or_path', None)
        # name or path is updated on fine tuning. Keep a copy of the original
        self.name_or_path_original: str = self.name_or_path
        self.is_v2: bool = kwargs.get('is_v2', False)
        self.is_xl: bool = kwargs.get('is_xl', False)
        self.is_pixart: bool = kwargs.get('is_pixart', False)
        self.is_pixart_sigma: bool = kwargs.get('is_pixart_sigma', False)
        self.is_auraflow: bool = kwargs.get('is_auraflow', False)
        self.is_v3: bool = kwargs.get('is_v3', False)
        self.is_flux: bool = kwargs.get('is_flux', False)
        self.is_lumina2: bool = kwargs.get('is_lumina2', False)
        if self.is_pixart_sigma:
            self.is_pixart = True
        self.use_flux_cfg = kwargs.get('use_flux_cfg', False)
        self.is_ssd: bool = kwargs.get('is_ssd', False)
        self.is_vega: bool = kwargs.get('is_vega', False)
        self.is_v_pred: bool = kwargs.get('is_v_pred', False)
        self.dtype: str = kwargs.get('dtype', 'float16')
        self.vae_path = kwargs.get('vae_path', None)
        self.refiner_name_or_path = kwargs.get('refiner_name_or_path', None)
        self._original_refiner_name_or_path = self.refiner_name_or_path
        self.refiner_start_at = kwargs.get('refiner_start_at', 0.5)
        self.lora_path = kwargs.get('lora_path', None)
        # mainly for decompression loras for distilled models
        self.assistant_lora_path = kwargs.get('assistant_lora_path', None)
        self.inference_lora_path = kwargs.get('inference_lora_path', None)
        self.latent_space_version = kwargs.get('latent_space_version', None)

        # only for SDXL models for now
        self.use_text_encoder_1: bool = kwargs.get('use_text_encoder_1', True)
        self.use_text_encoder_2: bool = kwargs.get('use_text_encoder_2', True)

        self.experimental_xl: bool = kwargs.get('experimental_xl', False)

        if self.name_or_path is None:
            raise ValueError('name_or_path must be specified')

        if self.is_ssd:
            # sed sdxl as true since it is mostly the same architecture
            self.is_xl = True

        if self.is_vega:
            self.is_xl = True

        # for text encoder quant. Only works with pixart currently
        self.text_encoder_bits = kwargs.get('text_encoder_bits', 16)  # 16, 8, 4
        self.unet_path = kwargs.get("unet_path", None)
        self.unet_sample_size = kwargs.get("unet_sample_size", None)
        self.vae_device = kwargs.get("vae_device", None)
        self.vae_dtype = kwargs.get("vae_dtype", self.dtype)
        self.te_device = kwargs.get("te_device", None)
        self.te_dtype = kwargs.get("te_dtype", self.dtype)

        # only for flux for now
        self.quantize = kwargs.get("quantize", False)
        self.quantize_te = kwargs.get("quantize_te", self.quantize)
        self.qtype = kwargs.get("qtype", "qfloat8")
        self.qtype_te = kwargs.get("qtype_te", "qfloat8")
        self.low_vram = kwargs.get("low_vram", False)
        self.attn_masking = kwargs.get("attn_masking", False)
        if self.attn_masking and not self.is_flux:
            raise ValueError("attn_masking is only supported with flux models currently")
        # for targeting a specific layers
        self.ignore_if_contains: Optional[List[str]] = kwargs.get("ignore_if_contains", None)
        self.only_if_contains: Optional[List[str]] = kwargs.get("only_if_contains", None)
        self.quantize_kwargs = kwargs.get("quantize_kwargs", {})
        
        # splits the model over the available gpus WIP
        self.split_model_over_gpus = kwargs.get("split_model_over_gpus", False)
        if self.split_model_over_gpus and not self.is_flux:
            raise ValueError("split_model_over_gpus is only supported with flux models currently")
        self.split_model_other_module_param_count_scale = kwargs.get("split_model_other_module_param_count_scale", 0.3)
        
        self.te_name_or_path = kwargs.get("te_name_or_path", None)
        
        self.arch: ModelArch = kwargs.get("arch", None)
        
        # can be used to load the extras like text encoder or vae from here
        # only setup for some models but will prevent having to download the te for
        # 20 different model variants
        self.extras_name_or_path = kwargs.get("extras_name_or_path", self.name_or_path)
        
        # path to an accuracy recovery adapter, either local or remote
        self.accuracy_recovery_adapter = kwargs.get("accuracy_recovery_adapter", None)
        
        # parse ARA from qtype
        if self.qtype is not None and "|" in self.qtype:
            self.qtype, self.accuracy_recovery_adapter = self.qtype.split('|')

        # compile the model with torch compile
        self.compile = kwargs.get("compile", False)
        
        # kwargs to pass to the model
        self.model_kwargs = kwargs.get("model_kwargs", {})
        
        # allow frontend to pass arch with a color like arch:tag
        # but remove the tag
        if self.arch is not None:
            if ':' in self.arch:
                self.arch = self.arch.split(':')[0]
        
        if self.arch == "flex1":
            self.arch = "flux"
        
        # handle migrating to new model arch
        if self.arch is not None:
            # reverse the arch to the old style
            if self.arch == 'sd2':
                self.is_v2 = True
            elif self.arch == 'sd3':
                self.is_v3 = True
            elif self.arch == 'sdxl':
                self.is_xl = True
            elif self.arch == 'pixart':
                self.is_pixart = True
            elif self.arch == 'pixart_sigma':
                self.is_pixart_sigma = True
            elif self.arch == 'auraflow':
                self.is_auraflow = True
            elif self.arch == 'flux':
                self.is_flux = True
            elif self.arch == 'lumina2':
                self.is_lumina2 = True
            elif self.arch == 'vega':
                self.is_vega = True
            elif self.arch == 'ssd':
                self.is_ssd = True
            else:
                pass
        if self.arch is None:
            if kwargs.get('is_v2', False):
                self.arch = 'sd2'
            elif kwargs.get('is_v3', False):
                self.arch = 'sd3'
            elif kwargs.get('is_xl', False):
                self.arch = 'sdxl'
            elif kwargs.get('is_pixart', False):
                self.arch = 'pixart'
            elif kwargs.get('is_pixart_sigma', False):
                self.arch = 'pixart_sigma'
            elif kwargs.get('is_auraflow', False):
                self.arch = 'auraflow'
            elif kwargs.get('is_flux', False):
                self.arch = 'flux'
            elif kwargs.get('is_lumina2', False):
                self.arch = 'lumina2'
            elif kwargs.get('is_vega', False):
                self.arch = 'vega'
            elif kwargs.get('is_ssd', False):
                self.arch = 'ssd'
            else:
                self.arch = 'sd1'
        


class EMAConfig:
    def __init__(self, **kwargs):
        self.use_ema: bool = kwargs.get('use_ema', False)
        self.ema_decay: float = kwargs.get('ema_decay', 0.999)
        # feeds back the decay difference into the parameter
        self.use_feedback: bool = kwargs.get('use_feedback', False)
        
        # every update, the params are multiplied by this amount
        # only use for things without a bias like lora
        # similar to a decay in an optimizer but the opposite
        self.param_multiplier: float = kwargs.get('param_multiplier', 1.0)


class ReferenceDatasetConfig:
    def __init__(self, **kwargs):
        # can pass with a side by side pait or a folder with pos and neg folder
        self.pair_folder: str = kwargs.get('pair_folder', None)
        self.pos_folder: str = kwargs.get('pos_folder', None)
        self.neg_folder: str = kwargs.get('neg_folder', None)

        self.network_weight: float = float(kwargs.get('network_weight', 1.0))
        self.pos_weight: float = float(kwargs.get('pos_weight', self.network_weight))
        self.neg_weight: float = float(kwargs.get('neg_weight', self.network_weight))
        # make sure they are all absolute values no negatives
        self.pos_weight = abs(self.pos_weight)
        self.neg_weight = abs(self.neg_weight)

        self.target_class: str = kwargs.get('target_class', '')
        self.size: int = kwargs.get('size', 512)


class SliderTargetConfig:
    def __init__(self, **kwargs):
        self.target_class: str = kwargs.get('target_class', '')
        self.positive: str = kwargs.get('positive', '')
        self.negative: str = kwargs.get('negative', '')
        self.multiplier: float = kwargs.get('multiplier', 1.0)
        self.weight: float = kwargs.get('weight', 1.0)
        self.shuffle: bool = kwargs.get('shuffle', False)


class GuidanceConfig:
    def __init__(self, **kwargs):
        self.target_class: str = kwargs.get('target_class', '')
        self.guidance_scale: float = kwargs.get('guidance_scale', 1.0)
        self.positive_prompt: str = kwargs.get('positive_prompt', '')
        self.negative_prompt: str = kwargs.get('negative_prompt', '')


class SliderConfigAnchors:
    def __init__(self, **kwargs):
        self.prompt = kwargs.get('prompt', '')
        self.neg_prompt = kwargs.get('neg_prompt', '')
        self.multiplier = kwargs.get('multiplier', 1.0)


class SliderConfig:
    def __init__(self, **kwargs):
        targets = kwargs.get('targets', [])
        anchors = kwargs.get('anchors', [])
        anchors = [SliderConfigAnchors(**anchor) for anchor in anchors]
        self.anchors: List[SliderConfigAnchors] = anchors
        self.resolutions: List[List[int]] = kwargs.get('resolutions', [[512, 512]])
        self.prompt_file: str = kwargs.get('prompt_file', None)
        self.prompt_tensors: str = kwargs.get('prompt_tensors', None)
        self.batch_full_slide: bool = kwargs.get('batch_full_slide', True)
        self.use_adapter: bool = kwargs.get('use_adapter', None)  # depth
        self.adapter_img_dir = kwargs.get('adapter_img_dir', None)
        self.low_ram = kwargs.get('low_ram', False)

        # expand targets if shuffling
        from toolkit.prompt_utils import get_slider_target_permutations
        self.targets: List[SliderTargetConfig] = []
        targets = [SliderTargetConfig(**target) for target in targets]
        # do permutations if shuffle is true
        print(f"Building slider targets")
        for target in targets:
            if target.shuffle:
                target_permutations = get_slider_target_permutations(target, max_permutations=8)
                self.targets = self.targets + target_permutations
            else:
                self.targets.append(target)
        print(f"Built {len(self.targets)} slider targets (with permutations)")

ControlTypes = Literal['depth', 'line', 'pose', 'inpaint', 'mask']

class DatasetConfig:
    """
    Dataset config for sd-datasets

    """

    def __init__(self, **kwargs):
        self.type = kwargs.get('type', 'image')  # sd, slider, reference
        # will be legacy
        self.folder_path: str = kwargs.get('folder_path', None)
        # can be json or folder path
        self.dataset_path: str = kwargs.get('dataset_path', None)

        self.default_caption: str = kwargs.get('default_caption', None)
        # trigger word for just this dataset
        self.trigger_word: str = kwargs.get('trigger_word', None)
        random_triggers = kwargs.get('random_triggers', [])
        # if they are a string, load them from a file
        if isinstance(random_triggers, str) and os.path.exists(random_triggers):
            with open(random_triggers, 'r') as f:
                random_triggers = f.read().splitlines()
                # remove empty lines
                random_triggers = [line for line in random_triggers if line.strip() != '']
        self.random_triggers: List[str] = random_triggers
        self.random_triggers_max: int = kwargs.get('random_triggers_max', 1)
        self.caption_ext: str = kwargs.get('caption_ext', '.txt')
        # if caption_ext doesnt start with a dot, add it
        if self.caption_ext and not self.caption_ext.startswith('.'):
            self.caption_ext = '.' + self.caption_ext
        self.random_scale: bool = kwargs.get('random_scale', False)
        self.random_crop: bool = kwargs.get('random_crop', False)
        self.resolution: int = kwargs.get('resolution', 512)
        self.scale: float = kwargs.get('scale', 1.0)
        self.buckets: bool = kwargs.get('buckets', True)
        self.bucket_tolerance: int = kwargs.get('bucket_tolerance', 64)
        self.is_reg: bool = kwargs.get('is_reg', False)
        self.network_weight: float = float(kwargs.get('network_weight', 1.0))
        self.token_dropout_rate: float = float(kwargs.get('token_dropout_rate', 0.0))
        self.shuffle_tokens: bool = kwargs.get('shuffle_tokens', False)
        self.caption_dropout_rate: float = float(kwargs.get('caption_dropout_rate', 0.0))
        self.keep_tokens: int = kwargs.get('keep_tokens', 0)  # #of first tokens to always keep unless caption dropped
        self.flip_x: bool = kwargs.get('flip_x', False)
        self.flip_y: bool = kwargs.get('flip_y', False)
        self.augments: List[str] = kwargs.get('augments', [])
        self.control_path: Union[str,List[str]] = kwargs.get('control_path', None)  # depth maps, etc
        if self.control_path == '':
            self.control_path = None
        
        # color for transparent reigon of control images with transparency
        self.control_transparent_color: List[int] = kwargs.get('control_transparent_color', [0, 0, 0])
        # inpaint images should be webp/png images with alpha channel. The alpha 0 (invisible) section will
        # be the part conditioned to be inpainted. The alpha 1 (visible) section will be the part that is ignored
        self.inpaint_path: Union[str,List[str]] = kwargs.get('inpaint_path', None)
        # instead of cropping ot match image, it will serve the full size control image (clip images ie for ip adapters)
        self.full_size_control_images: bool = kwargs.get('full_size_control_images', True)
        self.alpha_mask: bool = kwargs.get('alpha_mask', False)  # if true, will use alpha channel as mask
        self.mask_path: str = kwargs.get('mask_path',
                                         None)  # focus mask (black and white. White has higher loss than black)
        self.unconditional_path: str = kwargs.get('unconditional_path',
                                                  None)  # path where matching unconditional images are located
        self.invert_mask: bool = kwargs.get('invert_mask', False)  # invert mask
        self.mask_min_value: float = kwargs.get('mask_min_value', 0.0)  # min value for . 0 - 1
        self.poi: Union[str, None] = kwargs.get('poi',
                                                None)  # if one is set and in json data, will be used as auto crop scale point of interes
        self.use_short_captions: bool = kwargs.get('use_short_captions', False)  # if true, will use 'caption_short' from json
        self.num_repeats: int = kwargs.get('num_repeats', 1)  # number of times to repeat dataset
        # cache latents will store them in memory
        self.cache_latents: bool = kwargs.get('cache_latents', False)
        # cache latents to disk will store them on disk. If both are true, it will save to disk, but keep in memory
        self.cache_latents_to_disk: bool = kwargs.get('cache_latents_to_disk', False)
        self.cache_clip_vision_to_disk: bool = kwargs.get('cache_clip_vision_to_disk', False)
        self.cache_text_embeddings: bool = kwargs.get('cache_text_embeddings', False)

        self.standardize_images: bool = kwargs.get('standardize_images', False)

        # https://albumentations.ai/docs/api_reference/augmentations/transforms
        # augmentations are returned as a separate image and cannot currently be cached
        self.augmentations: List[dict] = kwargs.get('augmentations', None)
        self.shuffle_augmentations: bool = kwargs.get('shuffle_augmentations', False)

        has_augmentations = self.augmentations is not None and len(self.augmentations) > 0

        if (len(self.augments) > 0 or has_augmentations) and (self.cache_latents or self.cache_latents_to_disk):
            print(f"WARNING: Augments are not supported with caching latents. Setting cache_latents to False")
            self.cache_latents = False
            self.cache_latents_to_disk = False

        # legacy compatability
        legacy_caption_type = kwargs.get('caption_type', None)
        if legacy_caption_type:
            self.caption_ext = legacy_caption_type
        self.caption_type = self.caption_ext
        self.guidance_type: GuidanceType = kwargs.get('guidance_type', 'targeted')

        # ip adapter / reference dataset
        self.clip_image_path: str = kwargs.get('clip_image_path', None)  # depth maps, etc
        # get the clip image randomly from the same folder as the image. Useful for folder grouped pairs.
        self.clip_image_from_same_folder: bool = kwargs.get('clip_image_from_same_folder', False)
        self.clip_image_augmentations: List[dict] = kwargs.get('clip_image_augmentations', None)
        self.clip_image_shuffle_augmentations: bool = kwargs.get('clip_image_shuffle_augmentations', False)
        self.replacements: List[str] = kwargs.get('replacements', [])
        self.loss_multiplier: float = kwargs.get('loss_multiplier', 1.0)

        self.num_workers: int = kwargs.get('num_workers', 2)
        self.prefetch_factor: int = kwargs.get('prefetch_factor', 2)
        self.extra_values: List[float] = kwargs.get('extra_values', [])
        self.square_crop: bool = kwargs.get('square_crop', False)
        # apply same augmentations to control images. Usually want this true unless special case
        self.replay_transforms: bool = kwargs.get('replay_transforms', True)
        
        # for video
        # if num_frames is greater than 1, the dataloader will look for video files.
        # num_frames will be the number of frames in the training batch. If num_frames is 1, it will look for images
        self.num_frames: int = kwargs.get('num_frames', 1)
        # if true, will shrink video to our frames. For instance, if we have a video with 100 frames and num_frames is 10,
        # we would pull frame 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 so they are evenly spaced
        self.shrink_video_to_frames: bool = kwargs.get('shrink_video_to_frames', True)
        # fps is only used if shrink_video_to_frames is false. This will attempt to pull the num_frames at the given fps
        # it will select a random start frame and pull the frames at the given fps
        # this could have various issues with shorter videos and videos with variable fps
        # I recommend trimming your videos to the desired length and using shrink_video_to_frames(default)
        self.fps: int = kwargs.get('fps', 16)
        
        # debug the frame count and frame selection. You dont need this. It is for debugging.
        self.debug: bool = kwargs.get('debug', False)
        
        # automatic controls
        self.controls: List[ControlTypes] = kwargs.get('controls', [])
        if isinstance(self.controls, str):
            self.controls = [self.controls]
        # remove empty strings
        self.controls = [control for control in self.controls if control.strip() != '']
        
        # if true, will use a fask method to get image sizes. This can result in errors. Do not use unless you know what you are doing
        self.fast_image_size: bool = kwargs.get('fast_image_size', False)
        
        self.do_i2v: bool = kwargs.get('do_i2v', True)  # do image to video on models that are both t2i and i2v capable


def preprocess_dataset_raw_config(raw_config: List[dict]) -> List[dict]:
    """
    This just splits up the datasets by resolutions so you dont have to do it manually
    :param raw_config:
    :return:
    """
    # split up datasets by resolutions
    new_config = []
    for dataset in raw_config:
        resolution = dataset.get('resolution', 512)
        if isinstance(resolution, list):
            resolution_list = resolution
        else:
            resolution_list = [resolution]
        for res in resolution_list:
            dataset_copy = dataset.copy()
            dataset_copy['resolution'] = res
            new_config.append(dataset_copy)
    return new_config


class GenerateImageConfig:
    def __init__(
            self,
            prompt: str = '',
            prompt_2: Optional[str] = None,
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: str = '',
            negative_prompt_2: Optional[str] = None,
            seed: int = -1,
            network_multiplier: float = 1.0,
            guidance_rescale: float = 0.0,
            # the tag [time] will be replaced with milliseconds since epoch
            output_path: str = None,  # full image path
            output_folder: str = None,  # folder to save image in if output_path is not specified
            output_ext: str = ImgExt,  # extension to save image as if output_path is not specified
            output_tail: str = '',  # tail to add to output filename
            add_prompt_file: bool = False,  # add a prompt file with generated image
            adapter_image_path: str = None,  # path to adapter image
            adapter_conditioning_scale: float = 1.0,  # scale for adapter conditioning
            latents: Union[torch.Tensor | None] = None,  # input latent to start with,
            extra_kwargs: dict = None,  # extra data to save with prompt file
            refiner_start_at: float = 0.5,  # start at this percentage of a step. 0.0 to 1.0 . 1.0 is the end
            extra_values: List[float] = None,  # extra values to save with prompt file
            logger: Optional[EmptyLogger] = None,
            ctrl_img: Optional[str] = None,  # control image for controlnet
            num_frames: int = 1,
            fps: int = 15,
            ctrl_idx: int = 0
    ):
        self.width: int = width
        self.height: int = height
        self.num_inference_steps: int = num_inference_steps
        self.guidance_scale: float = guidance_scale
        self.guidance_rescale: float = guidance_rescale
        self.prompt: str = prompt
        self.prompt_2: str = prompt_2
        self.negative_prompt: str = negative_prompt
        self.negative_prompt_2: str = negative_prompt_2
        self.latents: Union[torch.Tensor | None] = latents

        self.output_path: str = output_path
        self.seed: int = seed
        if self.seed == -1:
            # generate random one
            self.seed = random.randint(0, 2 ** 32 - 1)
        self.network_multiplier: float = network_multiplier
        self.output_folder: str = output_folder
        self.output_ext: str = output_ext
        self.add_prompt_file: bool = add_prompt_file
        self.output_tail: str = output_tail
        self.gen_time: int = int(time.time() * 1000)
        self.adapter_image_path: str = adapter_image_path
        self.adapter_conditioning_scale: float = adapter_conditioning_scale
        self.extra_kwargs = extra_kwargs if extra_kwargs is not None else {}
        self.refiner_start_at = refiner_start_at
        self.extra_values = extra_values if extra_values is not None else []
        self.num_frames = num_frames
        self.fps = fps
        self.ctrl_img = ctrl_img
        self.ctrl_idx = ctrl_idx
        

        # prompt string will override any settings above
        self._process_prompt_string()

        # handle dual text encoder prompts if nothing passed
        if negative_prompt_2 is None:
            self.negative_prompt_2 = negative_prompt

        if prompt_2 is None:
            self.prompt_2 = self.prompt

        # parse prompt paths
        if self.output_path is None and self.output_folder is None:
            raise ValueError('output_path or output_folder must be specified')
        elif self.output_path is not None:
            self.output_folder = os.path.dirname(self.output_path)
            self.output_ext = os.path.splitext(self.output_path)[1][1:]
            self.output_filename_no_ext = os.path.splitext(os.path.basename(self.output_path))[0]

        else:
            self.output_filename_no_ext = '[time]_[count]'
            if len(self.output_tail) > 0:
                self.output_filename_no_ext += '_' + self.output_tail
            self.output_path = os.path.join(self.output_folder, self.output_filename_no_ext + '.' + self.output_ext)

        # adjust height
        self.height = max(64, self.height - self.height % 8)  # round to divisible by 8
        self.width = max(64, self.width - self.width % 8)  # round to divisible by 8

        self.logger = logger

    def set_gen_time(self, gen_time: int = None):
        if gen_time is not None:
            self.gen_time = gen_time
        else:
            self.gen_time = int(time.time() * 1000)

    def _get_path_no_ext(self, count: int = 0, max_count=0):
        # zero pad count
        count_str = str(count).zfill(len(str(max_count)))
        # replace [time] with gen time
        filename = self.output_filename_no_ext.replace('[time]', str(self.gen_time))
        # replace [count] with count
        filename = filename.replace('[count]', count_str)
        return filename

    def get_image_path(self, count: int = 0, max_count=0):
        filename = self._get_path_no_ext(count, max_count)
        ext = self.output_ext
        # if it does not start with a dot add one
        if ext[0] != '.':
            ext = '.' + ext
        filename += ext
        # join with folder
        return os.path.join(self.output_folder, filename)

    def get_prompt_path(self, count: int = 0, max_count=0):
        filename = self._get_path_no_ext(count, max_count)
        filename += '.txt'
        # join with folder
        return os.path.join(self.output_folder, filename)

    def save_image(self, image, count: int = 0, max_count=0):
        # make parent dirs
        os.makedirs(self.output_folder, exist_ok=True)
        self.set_gen_time()
        if isinstance(image, list):
            # video
            if self.num_frames == 1:
                raise ValueError(f"Expected 1 img but got a list {len(image)}")
            if self.num_frames > 1 and self.output_ext not in ['webp']:
                self.output_ext = 'webp'
            if self.output_ext == 'webp':
                # save as animated webp
                duration = 1000 // self.fps  # Convert fps to milliseconds per frame
                image[0].save(
                    self.get_image_path(count, max_count),
                    format='WEBP',
                    append_images=image[1:],
                    save_all=True,
                    duration=duration,  # Duration per frame in milliseconds
                    loop=0,  # 0 means loop forever
                    quality=80  # Quality setting (0-100)
                )
            else:
                raise ValueError(f"Unsupported video format {self.output_ext}")
        elif self.output_ext in ['wav', 'mp3']:
            # save audio file
            torchaudio.save(
                self.get_image_path(count, max_count), 
                image[0].to('cpu'),
                sample_rate=48000, 
                format=None, 
                backend=None
            )
        else:
            # TODO save image gen header info for A1111 and us, our seeds probably wont match
            image.save(self.get_image_path(count, max_count))
            # do prompt file
            if self.add_prompt_file:
                self.save_prompt_file(count, max_count)

    def save_prompt_file(self, count: int = 0, max_count=0):
        # save prompt file
        with open(self.get_prompt_path(count, max_count), 'w') as f:
            prompt = self.prompt
            if self.prompt_2 is not None:
                prompt += ' --p2 ' + self.prompt_2
            if self.negative_prompt is not None:
                prompt += ' --n ' + self.negative_prompt
            if self.negative_prompt_2 is not None:
                prompt += ' --n2 ' + self.negative_prompt_2
            prompt += ' --w ' + str(self.width)
            prompt += ' --h ' + str(self.height)
            prompt += ' --seed ' + str(self.seed)
            prompt += ' --cfg ' + str(self.guidance_scale)
            prompt += ' --steps ' + str(self.num_inference_steps)
            prompt += ' --m ' + str(self.network_multiplier)
            prompt += ' --gr ' + str(self.guidance_rescale)

            # get gen info
            try:
                f.write(self.prompt)
            except Exception as e:
                print(f"Error writing prompt file. Prompt contains non-unicode characters. {e}")

    def _process_prompt_string(self):
        # we will try to support all sd-scripts where we can

        # FROM SD-SCRIPTS
        # --n Treat everything until the next option as a negative prompt.
        # --w Specify the width of the generated image.
        # --h Specify the height of the generated image.
        # --d Specify the seed for the generated image.
        # --l Specify the CFG scale for the generated image.
        # --s Specify the number of steps during generation.

        # OURS and some QOL additions
        # --m Specify the network multiplier for the generated image.
        # --p2 Prompt for the second text encoder (SDXL only)
        # --n2 Negative prompt for the second text encoder (SDXL only)
        # --gr Specify the guidance rescale for the generated image (SDXL only)

        # --seed Specify the seed for the generated image same as --d
        # --cfg Specify the CFG scale for the generated image same as --l
        # --steps Specify the number of steps during generation same as --s
        # --network_multiplier Specify the network multiplier for the generated image same as --m

        # process prompt string and update values if it has some
        if self.prompt is not None and len(self.prompt) > 0:
            # process prompt string
            prompt = self.prompt
            prompt = prompt.strip()
            p_split = prompt.split('--')
            self.prompt = p_split[0].strip()

            if len(p_split) > 1:
                for split in p_split[1:]:
                    # allows multi char flags
                    flag = split.split(' ')[0].strip()
                    content = split[len(flag):].strip()
                    if flag == 'p2':
                        self.prompt_2 = content
                    elif flag == 'n':
                        self.negative_prompt = content
                    elif flag == 'n2':
                        self.negative_prompt_2 = content
                    elif flag == 'w':
                        self.width = int(content)
                    elif flag == 'h':
                        self.height = int(content)
                    elif flag == 'd':
                        self.seed = int(content)
                    elif flag == 'seed':
                        self.seed = int(content)
                    elif flag == 'l':
                        self.guidance_scale = float(content)
                    elif flag == 'cfg':
                        self.guidance_scale = float(content)
                    elif flag == 's':
                        self.num_inference_steps = int(content)
                    elif flag == 'steps':
                        self.num_inference_steps = int(content)
                    elif flag == 'm':
                        self.network_multiplier = float(content)
                    elif flag == 'network_multiplier':
                        self.network_multiplier = float(content)
                    elif flag == 'gr':
                        self.guidance_rescale = float(content)
                    elif flag == 'a':
                        self.adapter_conditioning_scale = float(content)
                    elif flag == 'ref':
                        self.refiner_start_at = float(content)
                    elif flag == 'ev':
                        # split by comma
                        self.extra_values = [float(val) for val in content.split(',')]
                    elif flag == 'extra_values':
                        # split by comma
                        self.extra_values = [float(val) for val in content.split(',')]
                    elif flag == 'frames':
                        self.num_frames = int(content)
                    elif flag == 'num_frames':
                        self.num_frames = int(content)
                    elif flag == 'fps':
                        self.fps = int(content)
                    elif flag == 'ctrl_img':
                        self.ctrl_img = content
                    elif flag == 'ctrl_idx':
                        self.ctrl_idx = int(content)

    def post_process_embeddings(
            self,
            conditional_prompt_embeds: PromptEmbeds,
            unconditional_prompt_embeds: Optional[PromptEmbeds] = None,
    ):
        # this is called after prompt embeds are encoded. We can override them in the future here
        pass
    
    def log_image(self, image, count: int = 0, max_count=0):
        if self.logger is None:
            return

        self.logger.log_image(image, count, self.prompt)
        
        
def validate_configs(
    train_config: TrainConfig,
    model_config: ModelConfig,
    save_config: SaveConfig,
    dataset_configs: List[DatasetConfig]
):
    if model_config.is_flux:
        if save_config.save_format != 'diffusers':
            # make it diffusers
            save_config.save_format = 'diffusers'
        if model_config.use_flux_cfg:
            # bypass the embedding
            train_config.bypass_guidance_embedding = True
    if train_config.bypass_guidance_embedding and train_config.do_guidance_loss:
        raise ValueError("Cannot bypass guidance embedding and do guidance loss at the same time. "
                         "Please set bypass_guidance_embedding to False or do_guidance_loss to False.")

    # see if any datasets are caching text embeddings
    is_caching_text_embeddings = any(dataset.cache_text_embeddings for dataset in dataset_configs)
    if is_caching_text_embeddings:
        
        # check if they are doing differential output preservation
        if train_config.diff_output_preservation:
            raise ValueError("Cannot use differential output preservation with caching text embeddings. Please set diff_output_preservation to False.")
    
        # make sure they are all cached
        for dataset in dataset_configs:
            if not dataset.cache_text_embeddings:
                raise ValueError("All datasets must have cache_text_embeddings set to True when caching text embeddings is enabled.")
    
    # qwen image edit cannot cache text embeddings
    if model_config.arch == 'qwen_image_edit':
        if train_config.unload_text_encoder:
            raise ValueError("Cannot cache unload text encoder with qwen_image_edit model. Control images are encoded with text embeddings. You can cache the text embeddings though")

    
