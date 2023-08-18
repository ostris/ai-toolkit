import os
import random

import cv2
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A


class ImageDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.name = self.get_config('name', 'dataset')
        self.path = self.get_config('path', required=True)
        self.scale = self.get_config('scale', 1)
        self.random_scale = self.get_config('random_scale', False)
        # we always random crop if random scale is enabled
        self.random_crop = self.random_scale if self.random_scale else self.get_config('random_crop', False)

        self.resolution = self.get_config('resolution', 256)
        self.file_list = [os.path.join(self.path, file) for file in os.listdir(self.path) if
                          file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        # this might take a while
        print(f"  -  Preprocessing image dimensions")
        new_file_list = []
        bad_count = 0
        for file in tqdm(self.file_list):
            img = Image.open(file)
            if int(min(img.size) * self.scale) >= self.resolution:
                new_file_list.append(file)
            else:
                bad_count += 1

        print(f"  -  Found {len(self.file_list)} images")
        print(f"  -  Found {bad_count} images that are too small")
        assert len(self.file_list) > 0, f"no images found in {self.path}"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # normalize to [-1, 1]
        ])

    def get_config(self, key, default=None, required=False):
        if key in self.config:
            value = self.config[key]
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.dataset.{key}" key')
        else:
            return default

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = exif_transpose(Image.open(img_path)).convert('RGB')

        # Downscale the source image first
        img = img.resize((int(img.size[0] * self.scale), int(img.size[1] * self.scale)), Image.BICUBIC)
        min_img_size = min(img.size)

        if self.random_crop:
            if self.random_scale and min_img_size > self.resolution:
                if min_img_size < self.resolution:
                    print(
                        f"Unexpected values: min_img_size={min_img_size}, self.resolution={self.resolution}, image file={img_path}")
                    scale_size = self.resolution
                else:
                    scale_size = random.randint(self.resolution, int(min_img_size))
                img = img.resize((scale_size, scale_size), Image.BICUBIC)
            img = transforms.RandomCrop(self.resolution)(img)
        else:
            img = transforms.CenterCrop(min_img_size)(img)
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)

        img = self.transform(img)

        return img


class Augments:
    def __init__(self, **kwargs):
        self.method_name = kwargs.get('method', None)
        self.params = kwargs.get('params', {})

        # convert kwargs enums for cv2
        for key, value in self.params.items():
            if isinstance(value, str):
                # split the string
                split_string = value.split('.')
                if len(split_string) == 2 and split_string[0] == 'cv2':
                    if hasattr(cv2, split_string[1]):
                        self.params[key] = getattr(cv2, split_string[1].upper())
                    else:
                        raise ValueError(f"invalid cv2 enum: {split_string[1]}")


class AugmentedImageDataset(ImageDataset):
    def __init__(self, config):
        super().__init__(config)
        self.augmentations = self.get_config('augmentations', [])
        self.augmentations = [Augments(**aug) for aug in self.augmentations]

        augmentation_list = []
        for aug in self.augmentations:
            # make sure method name is valid
            assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
            # get the method
            method = getattr(A, aug.method_name)
            # add the method to the list
            augmentation_list.append(method(**aug.params))

        self.aug_transform = A.Compose(augmentation_list)
        self.original_transform = self.transform
        # replace transform so we get raw pil image
        self.transform = transforms.Compose([])

    def __getitem__(self, index):
        # get the original image
        # image is a PIL image, convert to bgr
        pil_image = super().__getitem__(index)
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # apply augmentations
        augmented = self.aug_transform(image=open_cv_image)["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        # return both # return image as 0 - 1 tensor
        return transforms.ToTensor()(pil_image), transforms.ToTensor()(augmented)


class PairedImageDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.size = self.get_config('size', 512)
        self.path = self.get_config('path', None)
        self.pos_folder = self.get_config('pos_folder', None)
        self.neg_folder = self.get_config('neg_folder', None)

        self.default_prompt = self.get_config('default_prompt', '')
        self.network_weight = self.get_config('network_weight', 1.0)
        self.pos_weight = self.get_config('pos_weight', self.network_weight)
        self.neg_weight = self.get_config('neg_weight', self.network_weight)

        supported_exts = ('.jpg', '.jpeg', '.png', '.webp', '.JPEG', '.JPG', '.PNG', '.WEBP')

        if self.pos_folder is not None and self.neg_folder is not None:
            # find matching files
            self.pos_file_list = [os.path.join(self.pos_folder, file) for file in os.listdir(self.pos_folder) if
                                  file.lower().endswith(supported_exts)]
            self.neg_file_list = [os.path.join(self.neg_folder, file) for file in os.listdir(self.neg_folder) if
                                    file.lower().endswith(supported_exts)]

            matched_files = []
            for pos_file in self.pos_file_list:
                pos_file_no_ext = os.path.splitext(pos_file)[0]
                for neg_file in self.neg_file_list:
                    neg_file_no_ext = os.path.splitext(neg_file)[0]
                    if os.path.basename(pos_file_no_ext) == os.path.basename(neg_file_no_ext):
                        matched_files.append((neg_file, pos_file))
                        break

            # remove duplicates
            matched_files = [t for t in (set(tuple(i) for i in matched_files))]


            self.file_list = matched_files
            print(f"  -  Found {len(self.file_list)} matching pairs")
        else:
            self.file_list = [os.path.join(self.path, file) for file in os.listdir(self.path) if
                              file.lower().endswith(supported_exts)]
            print(f"  -  Found {len(self.file_list)} images")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.file_list)

    def get_config(self, key, default=None, required=False):
        if key in self.config:
            value = self.config[key]
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.dataset.{key}" key')
        else:
            return default

    def __getitem__(self, index):
        img_path_or_tuple = self.file_list[index]
        if isinstance(img_path_or_tuple, tuple):
            # load both images
            img_path = img_path_or_tuple[0]
            img1 = exif_transpose(Image.open(img_path)).convert('RGB')
            img_path = img_path_or_tuple[1]
            img2 = exif_transpose(Image.open(img_path)).convert('RGB')
            # combine them side by side
            img = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
            img.paste(img1, (0, 0))
            img.paste(img2, (img1.width, 0))

            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path_or_tuple[0])[0]
            prompt_path = path_no_ext + '.txt'
            if not os.path.exists(prompt_path):
                path_no_ext = os.path.splitext(img_path_or_tuple[1])[0]
                prompt_path = path_no_ext + '.txt'
        else:
            img_path = img_path_or_tuple
            img = exif_transpose(Image.open(img_path)).convert('RGB')
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
            prompt = self.default_prompt

        height = self.size
        # determine width to keep aspect ratio
        width = int(img.size[0] * height / img.size[1])

        # Downscale the source image first
        img = img.resize((width, height), Image.BICUBIC)
        img = self.transform(img)

        return img, prompt, (self.neg_weight, self.pos_weight)

