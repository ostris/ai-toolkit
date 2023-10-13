import base64
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Dict, Union

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from toolkit.basic import flush, value_map
from toolkit.buckets import get_bucket_for_image_size
from toolkit.metadata import get_meta_for_safetensors
from toolkit.prompt_utils import inject_trigger_into_prompt
from torchvision import transforms
from PIL import Image, ImageFilter
from PIL.ImageOps import exif_transpose

from toolkit.train_tools import get_torch_dtype

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO

# def get_associated_caption_from_img_path(img_path):


transforms_dict = {
    'ColorJitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    'RandomEqualize': transforms.RandomEqualize(p=0.2),
}

caption_ext_list = ['txt', 'json', 'caption']


class CaptionMixin:
    def get_caption_item(self: 'AiToolkitDataset', index):
        if not hasattr(self, 'caption_type'):
            raise Exception('caption_type not found on class instance')
        if not hasattr(self, 'file_list'):
            raise Exception('file_list not found on class instance')
        img_path_or_tuple = self.file_list[index]
        if isinstance(img_path_or_tuple, tuple):
            img_path = img_path_or_tuple[0] if isinstance(img_path_or_tuple[0], str) else img_path_or_tuple[0].path
            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = None
            for ext in caption_ext_list:
                prompt_path = path_no_ext + '.' + ext
                if os.path.exists(prompt_path):
                    break
        else:
            img_path = img_path_or_tuple if isinstance(img_path_or_tuple, str) else img_path_or_tuple.path
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = None
            for ext in caption_ext_list:
                prompt_path = path_no_ext + '.' + ext
                if os.path.exists(prompt_path):
                    break

        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                # check if is json
                if prompt_path.endswith('.json'):
                    prompt = json.loads(prompt)
                    if 'caption' in prompt:
                        prompt = prompt['caption']

                # remove any newlines
                prompt = prompt.replace('\n', ', ')
                # remove new lines for all operating systems
                prompt = prompt.replace('\r', ', ')
                prompt_split = prompt.split(',')
                # remove empty strings
                prompt_split = [p.strip() for p in prompt_split if p.strip()]
                # join back together
                prompt = ', '.join(prompt_split)
        else:
            prompt = ''
            # get default_prompt if it exists on the class instance
            if hasattr(self, 'default_prompt'):
                prompt = self.default_prompt
            if hasattr(self, 'default_caption'):
                prompt = self.default_caption
        return prompt


if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO


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

            if file_item.has_point_of_interest:
                # let the poi module handle the bucketing
                file_item.setup_poi_bucket()
            else:
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
                    print('debug')

            # check if bucket exists, if not, create it
            bucket_key = f'{file_item.crop_width}x{file_item.crop_height}'
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(file_item.crop_width, file_item.crop_height)
            self.buckets[bucket_key].file_list_idx.append(idx)

        # print the buckets
        self.build_batch_indices()
        if not quiet:
            print(f'Bucket sizes for {self.dataset_path}:')
            for key, bucket in self.buckets.items():
                print(f'{key}: {len(bucket.file_list_idx)} files')
            print(f'{len(self.buckets)} buckets made')



class CaptionProcessingDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)

    # todo allow for loading from sd-scripts style dict
    def load_caption(self: 'FileItemDTO', caption_dict: Union[dict, None]):
        if self.raw_caption is not None:
            # we already loaded it
            pass
        elif caption_dict is not None and self.path in caption_dict and "caption" in caption_dict[self.path]:
            self.raw_caption = caption_dict[self.path]["caption"]
        else:
            # see if prompt file exists
            path_no_ext = os.path.splitext(self.path)[0]
            prompt_ext = self.dataset_config.caption_ext
            prompt_path = f"{path_no_ext}.{prompt_ext}"

            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                    if prompt_path.endswith('.json'):
                        prompt = json.loads(prompt)
                        if 'caption' in prompt:
                            prompt = prompt['caption']
                    # remove any newlines
                    prompt = prompt.replace('\n', ', ')
                    # remove new lines for all operating systems
                    prompt = prompt.replace('\r', ', ')
                    prompt_split = prompt.split(',')
                    # remove empty strings
                    prompt_split = [p.strip() for p in prompt_split if p.strip()]
                    # join back together
                    prompt = ', '.join(prompt_split)
            else:
                prompt = ''
                if self.dataset_config.default_caption is not None:
                    prompt = self.dataset_config.default_caption
            self.raw_caption = prompt

    def get_caption(
            self: 'FileItemDTO',
            trigger=None,
            to_replace_list=None,
            add_if_not_present=False
    ):
        raw_caption = self.raw_caption
        if raw_caption is None:
            raw_caption = ''
        # handle dropout
        if self.dataset_config.caption_dropout_rate > 0:
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

        if self.dataset_config.shuffle_tokens:
            random.shuffle(token_list)

        # handle token dropout
        if self.dataset_config.token_dropout_rate > 0:
            new_token_list = []
            for token in token_list:
                # get a random float form 0 to 1
                rand = random.random()
                if rand > self.dataset_config.token_dropout_rate:
                    # keep the token
                    new_token_list.append(token)
            token_list = new_token_list

        # join back together
        caption = ', '.join(token_list)
        caption = inject_trigger_into_prompt(caption, trigger, to_replace_list, add_if_not_present)
        return caption


class ImageProcessingDTOMixin:
    def load_and_process_image(
            self: 'FileItemDTO',
            transform: Union[None, transforms.Compose],
            only_load_latents=False
    ):
        # if we are caching latents, just do that
        if self.is_latent_cached:
            self.get_latent()
            if self.has_control_image:
                self.load_control_image()
            if self.has_mask_image:
                self.load_mask_image()
            return
        try:
            img = Image.open(self.path)
            img = exif_transpose(img)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error loading image: {self.path}")

        if self.use_alpha_as_mask:
            # we do this to make sure it does not replace the alpha with another color
            # we want the image just without the alpha channel
            np_img = np.array(img)
            # strip off alpha
            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            # crop to x_crop, y_crop, x_crop + crop_width, y_crop + crop_height
            if img.width < self.crop_x + self.crop_width or img.height < self.crop_y + self.crop_height:
                # todo look into this. This still happens sometimes
                print('size mismatch')
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))

            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
        else:
            # Downscale the source image first
            # TODO this is nto right
            img = img.resize(
                (int(img.size[0] * self.dataset_config.scale), int(img.size[1] * self.dataset_config.scale)),
                Image.BICUBIC)
            min_img_size = min(img.size)
            if self.dataset_config.random_crop:
                if self.dataset_config.random_scale and min_img_size > self.dataset_config.resolution:
                    if min_img_size < self.dataset_config.resolution:
                        print(
                            f"Unexpected values: min_img_size={min_img_size}, self.resolution={self.dataset_config.resolution}, image file={self.path}")
                        scale_size = self.dataset_config.resolution
                    else:
                        scale_size = random.randint(self.dataset_config.resolution, int(min_img_size))
                    scaler = scale_size / min_img_size
                    scale_width = int((img.width + 5) * scaler)
                    scale_height = int((img.height + 5) * scaler)
                    img = img.resize((scale_width, scale_height), Image.BICUBIC)
                img = transforms.RandomCrop(self.dataset_config.resolution)(img)
            else:
                img = transforms.CenterCrop(min_img_size)(img)
                img = img.resize((self.dataset_config.resolution, self.dataset_config.resolution), Image.BICUBIC)

        if self.augments is not None and len(self.augments) > 0:
            # do augmentations
            for augment in self.augments:
                if augment in transforms_dict:
                    img = transforms_dict[augment](img)

        if transform:
            img = transform(img)

        self.tensor = img
        if not only_load_latents:
            if self.has_control_image:
                self.load_control_image()
            if self.has_mask_image:
                self.load_mask_image()


class ControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_control_image = False
        self.control_path: Union[str, None] = None
        self.control_tensor: Union[torch.Tensor, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.control_path is not None:
            # find the control image path
            control_path = dataset_config.control_path
            # we are using control images
            img_path = kwargs.get('path', None)
            img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(control_path, file_name_no_ext + ext)):
                    self.control_path = os.path.join(control_path, file_name_no_ext + ext)
                    self.has_control_image = True
                    break

    def load_control_image(self: 'FileItemDTO'):
        try:
            img = Image.open(self.control_path).convert('RGB')
            img = exif_transpose(img)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error loading image: {self.control_path}")
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img.transpose(Image.FLIP_TOP_BOTTOM)

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
            raise Exception("Control images not supported for non-bucket datasets")

        self.control_tensor = transforms.ToTensor()(img)

    def cleanup_control(self: 'FileItemDTO'):
        self.control_tensor = None


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
            img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']
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
            print(f"Error: {e}")
            print(f"Error loading image: {self.mask_path}")

        if self.use_alpha_as_mask:
            # pipeline expectws an rgb image so we need to put alpha in all channels
            np_img = np.array(img)
            np_img[:, :, :3] = np_img[:, :, 3:]

            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img.transpose(Image.FLIP_TOP_BOTTOM)

        # randomly apply a blur up to 2% of the size of the min (width, height)
        min_size = min(img.width, img.height)
        blur_radius = int(min_size * random.random() * 0.02)
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

        self.mask_tensor = transforms.ToTensor()(img)
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
                raise Exception(f"Error: poi not found in caption file: {caption_path}")
            if self.poi not in json_data['poi']:
                raise Exception(f"Error: poi not found in caption file: {caption_path}")
            # poi has, x, y, width, height
            poi = json_data['poi'][self.poi]
            self.poi_x = int(poi['x'])
            self.poi_y = int(poi['y'])
            self.poi_width = int(poi['width'])
            self.poi_height = int(poi['height'])

            # handle flipping
            if kwargs.get('flip_x', False):
                # flip the poi
                self.poi_x = self.width - self.poi_x - self.poi_width
            if kwargs.get('flip_y', False):
                # flip the poi
                self.poi_y = self.height - self.poi_y - self.poi_height

    def setup_poi_bucket(self: 'FileItemDTO'):
        # we are using poi, so we need to calculate the bucket based on the poi

        resolution = self.dataset_config.resolution
        bucket_tolerance = self.dataset_config.bucket_tolerance
        initial_width = int(self.width * self.dataset_config.scale)
        initial_height = int(self.height * self.dataset_config.scale)
        poi_x = int(self.poi_x * self.dataset_config.scale)
        poi_y = int(self.poi_y * self.dataset_config.scale)
        poi_width = int(self.poi_width * self.dataset_config.scale)
        poi_height = int(self.poi_height * self.dataset_config.scale)

        # todo handle a poi that is smaller than resolution
        # determine new cropping
        crop_left = random.randint(0, poi_x)
        crop_right = random.randint(poi_x + poi_width, initial_width)
        crop_top = random.randint(0, poi_y)
        crop_bottom = random.randint(poi_y + poi_height, initial_height)

        new_width = crop_right - crop_left
        new_height = crop_bottom - crop_top

        bucket_resolution = get_bucket_for_image_size(
            new_width, new_height,
            resolution=resolution,
            divisibility=bucket_tolerance
        )

        width_scale_factor = bucket_resolution["width"] / new_width
        height_scale_factor = bucket_resolution["height"] / new_height
        # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
        max_scale_factor = max(width_scale_factor, height_scale_factor)

        self.scale_to_width = int(initial_width * max_scale_factor)
        self.scale_to_height = int(initial_height * max_scale_factor)
        self.crop_width = bucket_resolution['width']
        self.crop_height = bucket_resolution['height']
        self.crop_x = int(crop_left * max_scale_factor)
        self.crop_y = int(crop_top * max_scale_factor)


class ArgBreakMixin:
    # just stops super calls form hitting object
    def __init__(self, *args, **kwargs):
        pass


class LatentCachingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self._encoded_latent: Union[torch.Tensor, None] = None
        self._latent_path: Union[str, None] = None
        self.is_latent_cached = False
        self.is_caching_to_disk = False
        self.is_caching_to_memory = False
        self.latent_load_device = 'cpu'
        # sd1 or sdxl or others
        self.latent_space_version = 'sd1'
        # todo, increment this if we change the latent format to invalidate cache
        self.latent_version = 1

    def get_latent_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("filename", os.path.basename(self.path)),
            ("scale_to_width", self.scale_to_width),
            ("scale_to_height", self.scale_to_height),
            ("crop_x", self.crop_x),
            ("crop_y", self.crop_y),
            ("crop_width", self.crop_width),
            ("crop_height", self.crop_height),
            ("latent_space_version", self.latent_space_version),
            ("latent_version", self.latent_version),
        ])
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        return item

    def get_latent_path(self: 'FileItemDTO', recalculate=False):
        if self._latent_path is not None and not recalculate:
            return self._latent_path
        else:
            # we store latents in a folder in same path as image called _latent_cache
            img_dir = os.path.dirname(self.path)
            latent_dir = os.path.join(img_dir, '_latent_cache')
            hash_dict = self.get_latent_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._latent_path = os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._latent_path

    def cleanup_latent(self):
        if self._encoded_latent is not None:
            if not self.is_caching_to_memory:
                # we are caching on disk, don't save in memory
                self._encoded_latent = None
            else:
                # move it back to cpu
                self._encoded_latent = self._encoded_latent.to('cpu')

    def get_latent(self, device=None):
        if not self.is_latent_cached:
            return None
        if self._encoded_latent is None:
            # load it from disk
            state_dict = load_file(
                self.get_latent_path(),
                # device=device if device is not None else self.latent_load_device
                device='cpu'
            )
            self._encoded_latent = state_dict['latent']
        return self._encoded_latent


class LatentCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.latent_cache = {}

    def cache_latents_all_latents(self: 'AiToolkitDataset'):
        print(f"Caching latents for {self.dataset_path}")
        # cache all latents to disk
        to_disk = self.is_caching_latents_to_disk
        to_memory = self.is_caching_latents_to_memory

        if to_disk:
            print(" - Saving latents to disk")
        if to_memory:
            print(" - Keeping latents in memory")
        # move sd items to cpu except for vae
        self.sd.set_device_state_preset('cache_latents')

        # use tqdm to show progress
        for i, file_item in tqdm(enumerate(self.file_list), desc=f'Caching latents{" to disk" if to_disk else ""}'):
            # set latent space version
            if self.sd.is_xl:
                file_item.latent_space_version = 'sdxl'
            else:
                file_item.latent_space_version = 'sd1'
            file_item.is_caching_to_disk = to_disk
            file_item.is_caching_to_memory = to_memory
            file_item.latent_load_device = self.sd.device

            latent_path = file_item.get_latent_path(recalculate=True)
            # check if it is saved to disk already
            if os.path.exists(latent_path):
                if to_memory:
                    # load it into memory
                    state_dict = load_file(latent_path, device='cpu')
                    file_item._encoded_latent = state_dict['latent'].to('cpu', dtype=self.sd.torch_dtype)
            else:
                # not saved to disk, calculate
                # load the image first
                file_item.load_and_process_image(self.transform, only_load_latents=True)
                dtype = self.sd.torch_dtype
                device = self.sd.device_torch
                # add batch dimension
                imgs = file_item.tensor.unsqueeze(0).to(device, dtype=dtype)
                latent = self.sd.encode_images(imgs).squeeze(0)
                # save_latent
                if to_disk:
                    state_dict = OrderedDict([
                        ('latent', latent.clone().detach().cpu()),
                    ])
                    # metadata
                    meta = get_meta_for_safetensors(file_item.get_latent_info_dict())
                    os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                    save_file(state_dict, latent_path, metadata=meta)

                if to_memory:
                    # keep it in memory
                    file_item._encoded_latent = latent.to('cpu', dtype=self.sd.torch_dtype)

                del imgs
                del latent
                del file_item.tensor


                flush(garbage_collect=False)
            file_item.is_latent_cached = True
            # flush every 100
            # if i % 100 == 0:
            #     flush()

        # restore device state
        self.sd.restore_device_state()
