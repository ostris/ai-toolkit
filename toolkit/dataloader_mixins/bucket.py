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


class Bucket:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.file_list_idx: List[int] = []



class BucketsMixin:
    def __init__(self):
        self.buckets: Dict[str, Bucket] = {}
        self.batch_indices: List[List[int]] = []

    def build_batch_indices(self: 'AiToolkitDataset'):
        self.batch_indices = []
        for key, bucket in self.buckets.items():
            for start_idx in range(0, len(bucket.file_list_idx), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(bucket.file_list_idx))
                batch = bucket.file_list_idx[start_idx:end_idx]
                self.batch_indices.append(batch)

    def shuffle_buckets(self: 'AiToolkitDataset'):
        for key, bucket in self.buckets.items():
            random.shuffle(bucket.file_list_idx)

    def setup_buckets(self: 'AiToolkitDataset', quiet=False):
        if not hasattr(self, 'file_list'):
            raise Exception(f'file_list not found on class instance {self.__class__.__name__}')
        if not hasattr(self, 'dataset_config'):
            raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')

        if self.epoch_num > 0 and self.dataset_config.poi is None:
            # no need to rebuild buckets for now
            # todo handle random cropping for buckets
            return
        self.buckets = {}  # clear it

        config: 'DatasetConfig' = self.dataset_config
        resolution = config.resolution
        bucket_tolerance = config.bucket_tolerance
        file_list: List['FileItemDTO'] = self.file_list

        # for file_item in enumerate(file_list):
        for idx, file_item in enumerate(file_list):
            file_item: 'FileItemDTO' = file_item
            width = int(file_item.width * file_item.dataset_config.scale)
            height = int(file_item.height * file_item.dataset_config.scale)

            did_process_poi = False
            if file_item.has_point_of_interest:
                # Attempt to process the poi if we can. It wont process if the image is smaller than the resolution
                did_process_poi = file_item.setup_poi_bucket()
            if self.dataset_config.square_crop:
                # we scale first so smallest size matches resolution
                scale_factor_x = resolution / width
                scale_factor_y = resolution / height
                scale_factor = max(scale_factor_x, scale_factor_y)
                file_item.scale_to_width = math.ceil(width * scale_factor)
                file_item.scale_to_height = math.ceil(height * scale_factor)
                file_item.crop_width = resolution
                file_item.crop_height = resolution
                if width > height:
                    file_item.crop_x = int(file_item.scale_to_width / 2 - resolution / 2)
                    file_item.crop_y = 0
                else:
                    file_item.crop_x = 0
                    file_item.crop_y = int(file_item.scale_to_height / 2 - resolution / 2)
            elif not did_process_poi:
                bucket_resolution = get_bucket_for_image_size(
                    width, height,
                    resolution=resolution,
                    divisibility=bucket_tolerance
                )

                # Calculate scale factors for width and height
                width_scale_factor = bucket_resolution["width"] / width
                height_scale_factor = bucket_resolution["height"] / height

                # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
                max_scale_factor = max(width_scale_factor, height_scale_factor)

                # round up
                file_item.scale_to_width = int(math.ceil(width * max_scale_factor))
                file_item.scale_to_height = int(math.ceil(height * max_scale_factor))

                file_item.crop_height = bucket_resolution["height"]
                file_item.crop_width = bucket_resolution["width"]

                new_width = bucket_resolution["width"]
                new_height = bucket_resolution["height"]

                if self.dataset_config.random_crop:
                    # random crop
                    crop_x = random.randint(0, file_item.scale_to_width - new_width)
                    crop_y = random.randint(0, file_item.scale_to_height - new_height)
                    file_item.crop_x = crop_x
                    file_item.crop_y = crop_y
                else:
                    # do central crop
                    file_item.crop_x = int((file_item.scale_to_width - new_width) / 2)
                    file_item.crop_y = int((file_item.scale_to_height - new_height) / 2)

                if file_item.crop_y < 0 or file_item.crop_x < 0:
                    print_acc('debug')

            # check if bucket exists, if not, create it
            bucket_key = f'{file_item.crop_width}x{file_item.crop_height}'
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(file_item.crop_width, file_item.crop_height)
            self.buckets[bucket_key].file_list_idx.append(idx)

        # print the buckets
        self.shuffle_buckets()
        self.build_batch_indices()
        if not quiet:
            print_acc(f'Bucket sizes for {self.dataset_path}:')
            for key, bucket in self.buckets.items():
                print_acc(f'{key}: {len(bucket.file_list_idx)} files')
            print_acc(f'{len(self.buckets)} buckets made')


