import os
from typing import TYPE_CHECKING, List, Dict


class CaptionMixin:
    def get_caption_item(self, index):
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
    from toolkit.data_loader import FileItem


class Bucket:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.file_list_idx: List[int] = []


class BucketsMixin:
    def __init__(self):
        self.buckets: Dict[str, Bucket] = {}
        self.batch_indices: List[List[int]] = []

    def build_batch_indices(self):
        for key, bucket in self.buckets.items():
            for start_idx in range(0, len(bucket.file_list_idx), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(bucket.file_list_idx))
                batch = bucket.file_list_idx[start_idx:end_idx]
                self.batch_indices.append(batch)

    def setup_buckets(self):
        if not hasattr(self, 'file_list'):
            raise Exception(f'file_list not found on class instance {self.__class__.__name__}')
        if not hasattr(self, 'dataset_config'):
            raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')

        config: 'DatasetConfig' = self.dataset_config
        resolution = config.resolution
        bucket_tolerance = config.bucket_tolerance
        file_list: List['FileItem'] = self.file_list

        # make sure out resolution is divisible by bucket_tolerance
        if resolution % bucket_tolerance != 0:
            # reduce it to the nearest divisible number
            resolution = resolution - (resolution % bucket_tolerance)

        # for file_item in enumerate(file_list):
        for idx, file_item in enumerate(file_list):
            width = file_item.crop_width
            height = file_item.crop_height

            # determine new size, smallest dimension should be equal to resolution
            # the other dimension should be the same ratio it is now (bigger)
            new_width = resolution
            new_height = resolution
            if width > height:
                # scale width to match new resolution,
                new_width = int(width * (resolution / height))
                file_item.crop_width = new_width
                file_item.scale_to_width = new_width
                file_item.crop_height = resolution
                file_item.scale_to_height = resolution
                # make sure new_width is divisible by bucket_tolerance
                if new_width % bucket_tolerance != 0:
                    # reduce it to the nearest divisible number
                    reduction = new_width % bucket_tolerance
                    file_item.crop_width = new_width - reduction
                    new_width = file_item.crop_width
                    # adjust the new x position so we evenly crop
                    file_item.crop_x = int(file_item.crop_x + (reduction / 2))
            elif height > width:
                # scale height to match new resolution
                new_height = int(height * (resolution / width))
                file_item.crop_height = new_height
                file_item.scale_to_height = new_height
                file_item.scale_to_width = resolution
                file_item.crop_width = resolution
                # make sure new_height is divisible by bucket_tolerance
                if new_height % bucket_tolerance != 0:
                    # reduce it to the nearest divisible number
                    reduction = new_height % bucket_tolerance
                    file_item.crop_height = new_height - reduction
                    new_height = file_item.crop_height
                    # adjust the new x position so we evenly crop
                    file_item.crop_y = int(file_item.crop_y + (reduction / 2))
            else:
                # square image
                file_item.crop_height = resolution
                file_item.scale_to_height = resolution
                file_item.scale_to_width = resolution
                file_item.crop_width = resolution

            # check if bucket exists, if not, create it
            bucket_key = f'{new_width}x{new_height}'
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(new_width, new_height)
            self.buckets[bucket_key].file_list_idx.append(idx)

        # print the buckets
        self.build_batch_indices()
        print(f'Bucket sizes for {self.__class__.__name__}:')
        for key, bucket in self.buckets.items():
            print(f'{key}: {len(bucket.file_list_idx)} files')
        print(f'{len(self.buckets)} buckets made')

        # file buckets made
