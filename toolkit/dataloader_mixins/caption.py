"""
Dataloader mixins - refactored for better organization.
"""

import base64
import glob
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Dict, Union
import traceback

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor

from toolkit.basic import flush, value_map
from toolkit.buckets import get_bucket_for_image_size, get_resolution
from toolkit.config_modules import ControlTypes
from toolkit.control_generator import ControlGenerator
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.pixtral_vision import PixtralVisionImagePreprocessorCompatible
from toolkit.prompt_utils import inject_trigger_into_prompt
from torchvision import transforms
from toolkit.dataloader_mixins.core import clean_caption
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageOps import exif_transpose
import albumentations as A
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator
from toolkit.prompt_utils import PromptEmbeds
from torchvision.transforms import functional as TF
from toolkit.train_tools import get_torch_dtype

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.stable_diffusion_model import StableDiffusion

accelerator = get_accelerator()


class CaptionMixin:
    def get_caption_item(self: 'AiToolkitDataset', index):
        if not hasattr(self, 'caption_type'):
            raise Exception('caption_type not found on class instance')
        if not hasattr(self, 'file_list'):
            raise Exception('file_list not found on class instance')
        img_path_or_tuple = self.file_list[index]
        ext = self.dataset_config.caption_ext
        if isinstance(img_path_or_tuple, tuple):
            img_path = img_path_or_tuple[0] if isinstance(img_path_or_tuple[0], str) else img_path_or_tuple[0].path
            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = None
            prompt_path = path_no_ext + ext
        else:
            img_path = img_path_or_tuple if isinstance(img_path_or_tuple, str) else img_path_or_tuple.path
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = path_no_ext + ext
                
        # allow folders to have a default prompt
        default_prompt_path = os.path.join(os.path.dirname(img_path), 'default.txt')
        default_prompt_path_with_ext = os.path.join(os.path.dirname(img_path), 'default' + ext)

        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                # check if is json
                if prompt_path.endswith('.json'):
                    prompt = json.loads(prompt)
                    if 'caption' in prompt:
                        prompt = prompt['caption']

                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path_with_ext):
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path):
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        else:
            prompt = ''
            # get default_prompt if it exists on the class instance
            if hasattr(self, 'default_prompt'):
                prompt = self.default_prompt
            if hasattr(self, 'default_caption'):
                prompt = self.default_caption

        # handle replacements
        replacement_list = self.dataset_config.replacements if isinstance(self.dataset_config.replacements, list) else []
        for replacement in replacement_list:
            from_string, to_string = replacement.split('|')
            prompt = prompt.replace(from_string, to_string)

        return prompt



class CaptionProcessingDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
            self.raw_caption: str = None
            self.raw_caption_short: str = None
            self.caption: str = None
            self.caption_short: str = None

            dataset_config: DatasetConfig = kwargs.get('dataset_config', None)
            self.extra_values: List[float] = dataset_config.extra_values
            self.trigger_word = dataset_config.trigger_word

    # todo allow for loading from sd-scripts style dict
    def load_caption(self: 'FileItemDTO', caption_dict: Union[dict, None]=None):
        if self.raw_caption is not None:
            # we already loaded it
            pass
        elif caption_dict is not None and self.path in caption_dict and "caption" in caption_dict[self.path]:
            self.raw_caption = caption_dict[self.path]["caption"]
            if 'caption_short' in caption_dict[self.path]:
                self.raw_caption_short = caption_dict[self.path]["caption_short"]
                if self.dataset_config.use_short_captions:
                    self.raw_caption = caption_dict[self.path]["caption_short"]
        else:
            # see if prompt file exists
            path_no_ext = os.path.splitext(self.path)[0]
            prompt_ext = self.dataset_config.caption_ext
            prompt_path = path_no_ext + prompt_ext
            short_caption = None

            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                    short_caption = None
                    if prompt_path.endswith('.json'):
                        # replace any line endings with commas for \n \r \r\n
                        prompt = prompt.replace('\r\n', ' ')
                        prompt = prompt.replace('\n', ' ')
                        prompt = prompt.replace('\r', ' ')

                        prompt_json = json.loads(prompt)
                        if 'caption' in prompt_json:
                            prompt = prompt_json['caption']
                        if 'caption_short' in prompt_json:
                            short_caption = prompt_json['caption_short']
                            if self.dataset_config.use_short_captions:
                                prompt = short_caption
                        if 'extra_values' in prompt_json:
                            self.extra_values = prompt_json['extra_values']

                    prompt = clean_caption(prompt)
                    if short_caption is not None:
                        short_caption = clean_caption(short_caption)
                    
                    if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                        prompt = self.dataset_config.default_caption
            else:
                prompt = ''
                if self.dataset_config.default_caption is not None:
                    prompt = self.dataset_config.default_caption

            if short_caption is None:
                short_caption = self.dataset_config.default_caption
            self.raw_caption = prompt
            self.raw_caption_short = short_caption

        self.caption = self.get_caption()
        if self.raw_caption_short is not None:
            self.caption_short = self.get_caption(short_caption=True)

    def get_caption(
            self: 'FileItemDTO',
            trigger=None,
            to_replace_list=None,
            add_if_not_present=False,
            short_caption=False
    ):
        if trigger is None and self.trigger_word is not None:
            trigger = self.trigger_word
        
        if trigger is not None and not self.is_reg:
            # add if not present if not regularization
            add_if_not_present = True
            
        if short_caption:
            raw_caption = self.raw_caption_short
        else:
            raw_caption = self.raw_caption
        if raw_caption is None:
            raw_caption = ''
        # handle dropout
        if self.dataset_config.caption_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
            # get a random float form 0 to 1
            rand = random.random()
            if rand < self.dataset_config.caption_dropout_rate:
                # drop the caption
                return ''

        # get tokens
        token_list = raw_caption.split(',')
        # trim whitespace
        token_list = [x.strip() for x in token_list]
        # remove empty strings
        token_list = [x for x in token_list if x]

        # handle token dropout
        if self.dataset_config.token_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
            new_token_list = []
            keep_tokens: int = self.dataset_config.keep_tokens
            for idx, token in enumerate(token_list):
                if idx < keep_tokens:
                    new_token_list.append(token)
                elif self.dataset_config.token_dropout_rate >= 1.0:
                    # drop the token
                    pass
                else:
                    # get a random float form 0 to 1
                    rand = random.random()
                    if rand > self.dataset_config.token_dropout_rate:
                        # keep the token
                        new_token_list.append(token)
            token_list = new_token_list

        if self.dataset_config.shuffle_tokens:
            random.shuffle(token_list)

        # join back together
        caption = ', '.join(token_list)
        caption = inject_trigger_into_prompt(caption, trigger, to_replace_list, add_if_not_present)

        if self.dataset_config.random_triggers:
            num_triggers = self.dataset_config.random_triggers_max
            if num_triggers > 1:
                num_triggers = random.randint(0, num_triggers)

            if num_triggers > 0:
                triggers = random.sample(self.dataset_config.random_triggers, num_triggers)
                caption = caption + ', ' + ', '.join(triggers)
                # add random triggers
                # for i in range(num_triggers):
                #     # fastest method
                #     trigger = self.dataset_config.random_triggers[int(random.random() * (len(self.dataset_config.random_triggers)))]
                #     caption = caption + ', ' + trigger

        if self.dataset_config.shuffle_tokens:
            # shuffle again
            token_list = caption.split(',')
            # trim whitespace
            token_list = [x.strip() for x in token_list]
            # remove empty strings
            token_list = [x for x in token_list if x]
            random.shuffle(token_list)
            caption = ', '.join(token_list)
        if caption == '':
            pass
        return caption


