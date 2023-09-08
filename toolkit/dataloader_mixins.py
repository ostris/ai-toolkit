import math
import os
import random
from typing import TYPE_CHECKING, List, Dict, Union

from toolkit.buckets import get_bucket_for_image_size
from toolkit.prompt_utils import inject_trigger_into_prompt
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO


# def get_associated_caption_from_img_path(img_path):


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

            bucket_resolution = get_bucket_for_image_size(width, height, resolution=resolution)

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
        # todo make sure this matches
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

        if self.dataset_config.buckets:
            # todo allow scaling and cropping, will be hard to add
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
        else:
            # Downscale the source image first
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
                    img = img.resize((scale_size, scale_size), Image.BICUBIC)
                img = transforms.RandomCrop(self.dataset_config.resolution)(img)
            else:
                img = transforms.CenterCrop(min_img_size)(img)
                img = img.resize((self.dataset_config.resolution, self.dataset_config.resolution), Image.BICUBIC)

        if transform:
            img = transform(img)

        self.tensor = img
