import os
import random
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torch.utils.data import Dataset


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
        self.file_list = [file for file in self.file_list if
                          int(min(Image.open(file).size) * self.scale) >= self.resolution]

        print(f"  -  Found {len(self.file_list)} images")
        assert len(self.file_list) > 0, f"no images found in {self.path}"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
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

        if self.random_crop:
            if self.random_scale:
                scale_size = random.randint(int(img.size[0] * self.scale), self.resolution)
                img = img.resize((scale_size, scale_size), Image.BICUBIC)
            img = transforms.RandomCrop(self.resolution)(img)
        else:
            min_dimension = min(img.size)
            img = transforms.CenterCrop(min_dimension)(img)
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)

        img = self.transform(img)

        return img
