import base64
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Dict, Union

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from toolkit.basic import flush
from toolkit.buckets import get_bucket_for_image_size
from toolkit.metadata import get_meta_for_safetensors
from toolkit.prompt_utils import inject_trigger_into_prompt
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose

from toolkit.train_tools import get_torch_dtype

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO

# def get_associated_caption_from_img_path(img_path):


transforms_dict = {
    'ColorJitter': transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
    'RandomEqualize': transforms.RandomEqualize(p=0.2),
}


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
            prompt_path = path_no_ext + '.txt'
            if not os.path.exists(prompt_path):
                img_path = img_path_or_tuple[1] if isinstance(img_path_or_tuple[1], str) else img_path_or_tuple[1].path
                path_no_ext = os.path.splitext(img_path)[0]
                prompt_path = path_no_ext + '.txt'
        else:
            img_path = img_path_or_tuple if isinstance(img_path_or_tuple, str) else img_path_or_tuple.path
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = path_no_ext + '.txt'

        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
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
        for key, bucket in self.buckets.items():
            for start_idx in range(0, len(bucket.file_list_idx), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(bucket.file_list_idx))
                batch = bucket.file_list_idx[start_idx:end_idx]
                self.batch_indices.append(batch)

    def setup_buckets(self: 'AiToolkitDataset'):
        if not hasattr(self, 'file_list'):
            raise Exception(f'file_list not found on class instance {self.__class__.__name__}')
        if not hasattr(self, 'dataset_config'):
            raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')

        config: 'DatasetConfig' = self.dataset_config
        resolution = config.resolution
        bucket_tolerance = config.bucket_tolerance
        file_list: List['FileItemDTO'] = self.file_list

        total_pixels = resolution * resolution

        # for file_item in enumerate(file_list):
        for idx, file_item in enumerate(file_list):
            file_item: 'FileItemDTO' = file_item
            width = file_item.crop_width
            height = file_item.crop_height

            bucket_resolution = get_bucket_for_image_size(width, height, resolution=resolution, divisibility=bucket_tolerance)

            # set the scaling height and with to match smallest size, and keep aspect ratio
            if width > height:
                file_item.scale_to_height = bucket_resolution["height"]
                file_item.scale_to_width = int(width * (bucket_resolution["height"] / height))
            else:
                file_item.scale_to_width = bucket_resolution["width"]
                file_item.scale_to_height = int(height * (bucket_resolution["width"] / width))

            file_item.crop_height = bucket_resolution["height"]
            file_item.crop_width = bucket_resolution["width"]

            new_width = bucket_resolution["width"]
            new_height = bucket_resolution["height"]

            # check if bucket exists, if not, create it
            bucket_key = f'{new_width}x{new_height}'
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(new_width, new_height)
            self.buckets[bucket_key].file_list_idx.append(idx)

        # print the buckets
        self.build_batch_indices()
        name = f"{os.path.basename(self.dataset_path)} ({self.resolution})"
        print(f'Bucket sizes for {self.dataset_path}:')
        for key, bucket in self.buckets.items():
            print(f'{key}: {len(bucket.file_list_idx)} files')
        print(f'{len(self.buckets)} buckets made')

        # file buckets made


class CaptionProcessingDTOMixin:

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
            transform: Union[None, transforms.Compose]
    ):
        # if we are caching latents, just do that
        if self.is_latent_cached:
            self.get_latent()
            return
        try:
            img = Image.open(self.path).convert('RGB')
            img = exif_transpose(img)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error loading image: {self.path}")
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
            # todo allow scaling and cropping, will be hard to add
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
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


class LatentCachingFileItemDTOMixin:
    def __init__(self):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__()
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
                device=device if device is not None else self.latent_load_device
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
                file_item.load_and_process_image(self.transform)
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
