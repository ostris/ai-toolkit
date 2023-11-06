import os
import time
from typing import List, Optional, Literal, Union
import random

import torch

from toolkit.prompt_utils import PromptEmbeds

ImgExt = Literal['jpg', 'png', 'webp']

SaveFormat = Literal['safetensors', 'diffusers']


class SaveConfig:
    def __init__(self, **kwargs):
        self.save_every: int = kwargs.get('save_every', 1000)
        self.dtype: str = kwargs.get('save_dtype', 'float16')
        self.max_step_saves_to_keep: int = kwargs.get('max_step_saves_to_keep', 5)
        self.save_format: SaveFormat = kwargs.get('save_format', 'safetensors')
        if self.save_format not in ['safetensors', 'diffusers']:
            raise ValueError(f"save_format must be safetensors or diffusers, got {self.save_format}")


class LogingConfig:
    def __init__(self, **kwargs):
        self.log_every: int = kwargs.get('log_every', 100)
        self.verbose: bool = kwargs.get('verbose', False)
        self.use_wandb: bool = kwargs.get('use_wandb', False)


class SampleConfig:
    def __init__(self, **kwargs):
        self.sampler: str = kwargs.get('sampler', 'ddpm')
        self.sample_every: int = kwargs.get('sample_every', 100)
        self.width: int = kwargs.get('width', 512)
        self.height: int = kwargs.get('height', 512)
        self.prompts: list[str] = kwargs.get('prompts', [])
        self.neg = kwargs.get('neg', False)
        self.seed = kwargs.get('seed', 0)
        self.walk_seed = kwargs.get('walk_seed', False)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)
        self.network_multiplier = kwargs.get('network_multiplier', 1)
        self.guidance_rescale = kwargs.get('guidance_rescale', 0.0)
        self.ext: ImgExt = kwargs.get('format', 'jpg')
        self.adapter_conditioning_scale = kwargs.get('adapter_conditioning_scale', 1.0)


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


NetworkType = Literal['lora', 'locon', 'lorm']


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


AdapterTypes = Literal['t2i', 'ip', 'ip+']


class AdapterConfig:
    def __init__(self, **kwargs):
        self.type: AdapterTypes = kwargs.get('type', 't2i')  # t2i, ip
        self.in_channels: int = kwargs.get('in_channels', 3)
        self.channels: List[int] = kwargs.get('channels', [320, 640, 1280, 1280])
        self.num_res_blocks: int = kwargs.get('num_res_blocks', 2)
        self.downscale_factor: int = kwargs.get('downscale_factor', 8)
        self.adapter_type: str = kwargs.get('adapter_type', 'full_adapter')
        self.image_dir: str = kwargs.get('image_dir', None)
        self.test_img_path: str = kwargs.get('test_img_path', None)
        self.train: str = kwargs.get('train', False)
        self.image_encoder_path: str = kwargs.get('image_encoder_path', None)
        self.name_or_path = kwargs.get('name_or_path', None)


class EmbeddingConfig:
    def __init__(self, **kwargs):
        self.trigger = kwargs.get('trigger', 'custom_embedding')
        self.tokens = kwargs.get('tokens', 4)
        self.init_words = kwargs.get('init_words', '*')
        self.save_format = kwargs.get('save_format', 'safetensors')


ContentOrStyleType = Literal['balanced', 'style', 'content']
LossTarget = Literal['noise', 'source', 'unaugmented', 'differential_noise']


class TrainConfig:
    def __init__(self, **kwargs):
        self.noise_scheduler = kwargs.get('noise_scheduler', 'ddpm')
        self.content_or_style: ContentOrStyleType = kwargs.get('content_or_style', 'balanced')
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
        self.max_denoising_steps: int = kwargs.get('max_denoising_steps', 1000)
        self.batch_size: int = kwargs.get('batch_size', 1)
        self.dtype: str = kwargs.get('dtype', 'fp32')
        self.xformers = kwargs.get('xformers', False)
        self.sdp = kwargs.get('sdp', False)
        self.train_unet = kwargs.get('train_unet', True)
        self.train_text_encoder = kwargs.get('train_text_encoder', True)
        self.train_refiner = kwargs.get('train_refiner', True)
        self.min_snr_gamma = kwargs.get('min_snr_gamma', None)
        self.snr_gamma = kwargs.get('snr_gamma', None)
        # trains a gamma, offset, and scale to adjust loss to adapt to timestep differentials
        # this should balance the learning rate across all timesteps over time
        self.learnable_snr_gos = kwargs.get('learnable_snr_gos', False)
        self.noise_offset = kwargs.get('noise_offset', 0.0)
        self.skip_first_sample = kwargs.get('skip_first_sample', False)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
        self.weight_jitter = kwargs.get('weight_jitter', 0.0)
        self.merge_network_on_save = kwargs.get('merge_network_on_save', False)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.start_step = kwargs.get('start_step', None)
        self.free_u = kwargs.get('free_u', False)
        self.adapter_assist_name_or_path: Optional[str] = kwargs.get('adapter_assist_name_or_path', None)
        self.noise_multiplier = kwargs.get('noise_multiplier', 1.0)
        self.img_multiplier = kwargs.get('img_multiplier', 1.0)

        # match the norm of the noise before computing loss. This will help the model maintain its
        # current understandin of the brightness of images.

        self.match_noise_norm = kwargs.get('match_noise_norm', False)

        # set to -1 to accumulate gradients for entire epoch
        # warning, only do this with a small dataset or you will run out of memory
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)

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

        # legacy
        if match_adapter_assist and self.match_adapter_chance == 0.0:
            self.match_adapter_chance = 1.0


class ModelConfig:
    def __init__(self, **kwargs):
        self.name_or_path: str = kwargs.get('name_or_path', None)
        self.is_v2: bool = kwargs.get('is_v2', False)
        self.is_xl: bool = kwargs.get('is_xl', False)
        self.is_ssd: bool = kwargs.get('is_ssd', False)
        self.is_v_pred: bool = kwargs.get('is_v_pred', False)
        self.dtype: str = kwargs.get('dtype', 'float16')
        self.vae_path = kwargs.get('vae_path', None)
        self.refiner_name_or_path = kwargs.get('refiner_name_or_path', None)
        self.refiner_start_at = kwargs.get('refiner_start_at', 0.5)

        # only for SDXL models for now
        self.use_text_encoder_1: bool = kwargs.get('use_text_encoder_1', True)
        self.use_text_encoder_2: bool = kwargs.get('use_text_encoder_2', True)

        self.experimental_xl: bool = kwargs.get('experimental_xl', False)

        if self.name_or_path is None:
            raise ValueError('name_or_path must be specified')

        if self.is_ssd:
            # sed sdxl as true since it is mostly the same architecture
            self.is_xl = True


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
                target_permutations = get_slider_target_permutations(target, max_permutations=100)
                self.targets = self.targets + target_permutations
            else:
                self.targets.append(target)
        print(f"Built {len(self.targets)} slider targets (with permutations)")


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
        self.random_triggers: List[str] = kwargs.get('random_triggers', [])
        self.caption_ext: str = kwargs.get('caption_ext', None)
        self.random_scale: bool = kwargs.get('random_scale', False)
        self.random_crop: bool = kwargs.get('random_crop', False)
        self.resolution: int = kwargs.get('resolution', 512)
        self.scale: float = kwargs.get('scale', 1.0)
        self.buckets: bool = kwargs.get('buckets', False)
        self.bucket_tolerance: int = kwargs.get('bucket_tolerance', 64)
        self.is_reg: bool = kwargs.get('is_reg', False)
        self.network_weight: float = float(kwargs.get('network_weight', 1.0))
        self.token_dropout_rate: float = float(kwargs.get('token_dropout_rate', 0.0))
        self.shuffle_tokens: bool = kwargs.get('shuffle_tokens', False)
        self.caption_dropout_rate: float = float(kwargs.get('caption_dropout_rate', 0.0))
        self.flip_x: bool = kwargs.get('flip_x', False)
        self.flip_y: bool = kwargs.get('flip_y', False)
        self.augments: List[str] = kwargs.get('augments', [])
        self.control_path: str = kwargs.get('control_path', None)  # depth maps, etc
        self.alpha_mask: bool = kwargs.get('alpha_mask', False)  # if true, will use alpha channel as mask
        self.mask_path: str = kwargs.get('mask_path',
                                         None)  # focus mask (black and white. White has higher loss than black)
        self.unconditional_path: str = kwargs.get('unconditional_path', None)  # path where matching unconditional images are located
        self.invert_mask: bool = kwargs.get('invert_mask', False)  # invert mask
        self.mask_min_value: float = kwargs.get('mask_min_value', 0.01)  # min value for . 0 - 1
        self.poi: Union[str, None] = kwargs.get('poi',
                                                None)  # if one is set and in json data, will be used as auto crop scale point of interes
        self.num_repeats: int = kwargs.get('num_repeats', 1)  # number of times to repeat dataset
        # cache latents will store them in memory
        self.cache_latents: bool = kwargs.get('cache_latents', False)
        # cache latents to disk will store them on disk. If both are true, it will save to disk, but keep in memory
        self.cache_latents_to_disk: bool = kwargs.get('cache_latents_to_disk', False)

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

        # prompt string will override any settings above
        self._process_prompt_string()

        # handle dual text encoder prompts if nothing passed
        if negative_prompt_2 is None:
            self.negative_prompt_2 = negative_prompt

        if prompt_2 is None:
            self.prompt_2 = prompt

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
            f.write(self.prompt)

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

    def post_process_embeddings(
            self,
            conditional_prompt_embeds: PromptEmbeds,
            unconditional_prompt_embeds: Optional[PromptEmbeds] = None,
    ):
        # this is called after prompt embeds are encoded. We can override them in the future here
        pass
