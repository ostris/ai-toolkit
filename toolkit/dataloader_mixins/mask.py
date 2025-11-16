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


class MaskFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_mask_image = False
        self.mask_path: Union[str, None] = None
        self.mask_tensor: Union[torch.Tensor, None] = None
        self.use_alpha_as_mask: bool = False
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.mask_min_value = dataset_config.mask_min_value
        if dataset_config.alpha_mask:
            self.use_alpha_as_mask = True
            self.mask_path = kwargs.get('path', None)
            self.has_mask_image = True
        elif dataset_config.mask_path is not None:
            # find the control image path
            mask_path = dataset_config.mask_path if dataset_config.mask_path is not None else dataset_config.alpha_mask
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(mask_path, file_name_no_ext + ext)):
                    self.mask_path = os.path.join(mask_path, file_name_no_ext + ext)
                    self.has_mask_image = True
                    break

    def load_mask_image(self: 'FileItemDTO'):
        try:
            img = Image.open(self.mask_path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        if self.use_alpha_as_mask:
            # pipeline expectws an rgb image so we need to put alpha in all channels
            np_img = np.array(img)
            np_img[:, :, :3] = np_img[:, :, 3:]

            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        if self.dataset_config.invert_mask:
            img = ImageOps.invert(img)
        w, h = img.size
        fix_size = False
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True

        if fix_size:
            # swap all the sizes
            self.scale_to_width, self.scale_to_height = self.scale_to_height, self.scale_to_width
            self.crop_width, self.crop_height = self.crop_height, self.crop_width
            self.crop_x, self.crop_y = self.crop_y, self.crop_x




        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # randomly apply a blur up to 0.5% of the size of the min (width, height)
        min_size = min(img.width, img.height)
        blur_radius = int(min_size * random.random() * 0.005)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # make grayscale
        img = img.convert('L')

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
            # crop
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            raise Exception("Mask images not supported for non-bucket datasets")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.aug_replay_spatial_transforms:
            self.mask_tensor = self.augment_spatial_control(img, transform=transform)
        else:
            self.mask_tensor = transform(img)
        self.mask_tensor = value_map(self.mask_tensor, 0, 1.0, self.mask_min_value, 1.0)
        # convert to grayscale

    def cleanup_mask(self: 'FileItemDTO'):
        self.mask_tensor = None



class PoiFileItemDTOMixin:
    # Point of interest bounding box. Allows for dynamic cropping without cropping out the main subject
    # items in the poi will always be inside the image when random cropping
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        # poi is a name of the box point of interest in the caption json file
        dataset_config = kwargs.get('dataset_config', None)
        path = kwargs.get('path', None)
        self.poi: Union[str, None] = dataset_config.poi
        self.has_point_of_interest = self.poi is not None
        self.poi_x: Union[int, None] = None
        self.poi_y: Union[int, None] = None
        self.poi_width: Union[int, None] = None
        self.poi_height: Union[int, None] = None

        if self.poi is not None:
            # make sure latent caching is off
            if dataset_config.cache_latents or dataset_config.cache_latents_to_disk:
                raise Exception(
                    f"Error: poi is not supported when caching latents. Please set cache_latents and cache_latents_to_disk to False in the dataset config"
                )
                # make sure we are loading through json
            if dataset_config.caption_ext != 'json':
                raise Exception(
                    f"Error: poi is only supported when using json captions. Please set caption_ext to json in the dataset config"
                )
            self.poi = self.poi.strip()
            # get the caption path
            file_path_no_ext = os.path.splitext(path)[0]
            caption_path = file_path_no_ext + '.json'
            if not os.path.exists(caption_path):
                raise Exception(f"Error: caption file not found for poi: {caption_path}")
            with open(caption_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            if 'poi' not in json_data:
                print_acc(f"Warning: poi not found in caption file: {caption_path}")
            if self.poi not in json_data['poi']:
                print_acc(f"Warning: poi not found in caption file: {caption_path}")
            # poi has, x, y, width, height
            # do full image if no poi
            self.poi_x = 0
            self.poi_y = 0
            self.poi_width = self.width
            self.poi_height = self.height
            try:
                if self.poi in json_data['poi']:
                    poi = json_data['poi'][self.poi]
                    self.poi_x = int(poi['x'])
                    self.poi_y = int(poi['y'])
                    self.poi_width = int(poi['width'])
                    self.poi_height = int(poi['height'])
            except Exception as e:
                pass

            # handle flipping
            if kwargs.get('flip_x', False):
                # flip the poi
                self.poi_x = self.width - self.poi_x - self.poi_width
            if kwargs.get('flip_y', False):
                # flip the poi
                self.poi_y = self.height - self.poi_y - self.poi_height

    def setup_poi_bucket(self: 'FileItemDTO'):
        initial_width = int(self.width * self.dataset_config.scale)
        initial_height = int(self.height * self.dataset_config.scale)
        # we are using poi, so we need to calculate the bucket based on the poi

        # if img resolution is less than dataset resolution, just return and let the normal bucketing happen
        img_resolution = get_resolution(initial_width, initial_height)
        if img_resolution <= self.dataset_config.resolution:
            return False  # will trigger normal bucketing

        bucket_tolerance = self.dataset_config.bucket_tolerance
        poi_x = int(self.poi_x * self.dataset_config.scale)
        poi_y = int(self.poi_y * self.dataset_config.scale)
        poi_width = int(self.poi_width * self.dataset_config.scale)
        poi_height = int(self.poi_height * self.dataset_config.scale)

        # loop to keep expanding until we are at the proper resolution. This is not ideal, we can probably handle it better
        num_loops = 0
        while True:
            # crop left
            if poi_x > 0:
                poi_x = random.randint(0, poi_x)
            else:
                poi_x = 0

            # crop right
            cr_min = poi_x + poi_width
            if cr_min < initial_width:
                crop_right = random.randint(poi_x + poi_width, initial_width)
            else:
                crop_right = initial_width

            poi_width = crop_right - poi_x

            if poi_y > 0:
                poi_y = random.randint(0, poi_y)
            else:
                poi_y = 0

            if poi_y + poi_height < initial_height:
                crop_bottom = random.randint(poi_y + poi_height, initial_height)
            else:
                crop_bottom = initial_height

            poi_height = crop_bottom - poi_y
            try:
                # now we have our random crop, but it may be smaller than resolution. Check and expand if needed
                current_resolution = get_resolution(poi_width, poi_height)
            except Exception as e:
                print_acc(f"Error: {e}")
                print_acc(f"Error getting resolution: {self.path}")
                raise e
                return False
            if current_resolution >= self.dataset_config.resolution:
                # We can break now
                break
            else:
                num_loops += 1
                if num_loops > 100:
                    print_acc(
                        f"Warning: poi bucketing looped too many times. This should not happen. Please report this issue.")
                    return False

        new_width = poi_width
        new_height = poi_height

        bucket_resolution = get_bucket_for_image_size(
            new_width, new_height,
            resolution=self.dataset_config.resolution,
            divisibility=bucket_tolerance
        )

        width_scale_factor = bucket_resolution["width"] / new_width
        height_scale_factor = bucket_resolution["height"] / new_height
        # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
        max_scale_factor = max(width_scale_factor, height_scale_factor)

        self.scale_to_width = math.ceil(initial_width * max_scale_factor)
        self.scale_to_height = math.ceil(initial_height * max_scale_factor)
        self.crop_width = bucket_resolution['width']
        self.crop_height = bucket_resolution['height']
        self.crop_x = int(poi_x * max_scale_factor)
        self.crop_y = int(poi_y * max_scale_factor)

        if self.scale_to_width < self.crop_x + self.crop_width or self.scale_to_height < self.crop_y + self.crop_height:
            # todo look into this. This still happens sometimes
            print_acc('size mismatch')

        return True


