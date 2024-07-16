import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
import cv2
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolkit.paths import SD_SCRIPTS_ROOT
import torchvision.transforms.functional
from toolkit.image_utils import show_img

sys.path.append(SD_SCRIPTS_ROOT)

from library.model_util import load_vae
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.data_loader import AiToolkitDataset, get_dataloader_from_datasets, \
    trigger_dataloader_setup_epoch
from toolkit.config_modules import DatasetConfig
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('dataset_folder', type=str, default='input')
parser.add_argument('--epochs', type=int, default=1)



args = parser.parse_args()

dataset_folder = args.dataset_folder
resolution = 512
bucket_tolerance = 64
batch_size = 1


##

dataset_config = DatasetConfig(
    dataset_path=dataset_folder,
    control_path=dataset_folder,
    resolution=resolution,
    # caption_ext='json',
    default_caption='default',
    # clip_image_path='/mnt/Datasets2/regs/yetibear_xl_v14/random_aspect/',
    buckets=True,
    bucket_tolerance=bucket_tolerance,
    # poi='person',
    # shuffle_augmentations=True,
    # augmentations=[
    #     {
    #         'method': 'Posterize',
    #         'num_bits': [(0, 4), (0, 4), (0, 4)],
    #         'p': 1.0
    #     },
    #
    # ]
)

dataloader: DataLoader = get_dataloader_from_datasets([dataset_config], batch_size=batch_size)

def random_blur(img, min_kernel_size=3, max_kernel_size=23, p=0.5):
    if random.random() < p:
        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        # make sure it is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        img = torchvision.transforms.functional.gaussian_blur(img, kernel_size=kernel_size)
    return img

def quantize(image, palette):
    """
    Similar to PIL.Image.quantize() in PyTorch. Built to maintain gradient.
    Only works for one image i.e. CHW. Does NOT work for batches.
    ref https://discuss.pytorch.org/t/color-quantization/104528/4
    """

    orig_dtype = image.dtype

    C, H, W = image.shape
    n_colors = palette.shape[0]

    # Easier to work with list of colors
    flat_img = image.view(C, -1).T  # [C, H, W] -> [H*W, C]

    # Repeat image so that there are n_color number of columns of the same image
    flat_img_per_color = flat_img.unsqueeze(1).expand(-1, n_colors, -1)  # [H*W, C] -> [H*W, n_colors, C]

    # Get euclidean distance between each pixel in each column and the column's respective color
    # i.e. column 1 lists distance of each pixel to color #1 in palette, column 2 to color #2 etc.
    squared_distance = (flat_img_per_color - palette.unsqueeze(0)) ** 2
    euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=-1) + 1e-8)  # [H*W, n_colors, C] -> [H*W, n_colors]

    # Get the shortest distance (one value per row (H*W) is selected)
    min_distances, min_indices = torch.min(euclidean_distance, dim=-1)  # [H*W, n_colors] -> [H*W]

    # Create a mask for the closest colors
    one_hot_mask = torch.nn.functional.one_hot(min_indices, num_classes=n_colors).float()  # [H*W, n_colors]

    # Multiply the mask with the palette colors to get the quantized image
    quantized = torch.matmul(one_hot_mask, palette)  # [H*W, n_colors] @ [n_colors, C] -> [H*W, C]

    # Reshape it back to the original input format.
    quantized_img = quantized.T.view(C, H, W)  # [H*W, C] -> [C, H, W]

    return quantized_img.to(orig_dtype)



def color_block_imgs(img, neg1_1=False):
    # expects values 0 - 1
    orig_dtype = img.dtype
    if neg1_1:
        img = img * 0.5 + 0.5

    img = img * 255
    img = img.clamp(0, 255)
    img = img.to(torch.uint8)

    img_chunks = torch.chunk(img, img.shape[0], dim=0)

    posterized_chunks = []

    for chunk in img_chunks:
        img_size = (chunk.shape[2] + chunk.shape[3]) // 2
        # min kernel size of 1% of image, max 10%
        min_kernel_size = int(img_size * 0.01)
        max_kernel_size = int(img_size * 0.1)

        # blur first
        chunk = random_blur(chunk, min_kernel_size=min_kernel_size, max_kernel_size=max_kernel_size, p=0.8)
        num_colors = random.randint(1, 16)

        resize_to = 16
        # chunk = torchvision.transforms.functional.posterize(chunk, num_bits_to_use)

        # mean_color = [int(x.item()) for x in torch.mean(chunk.float(), dim=(0, 2, 3))]

        # shrink the image down to num_colors x num_colors
        shrunk = torchvision.transforms.functional.resize(chunk, [resize_to, resize_to])

        mean_color = [int(x.item()) for x in torch.mean(shrunk.float(), dim=(0, 2, 3))]

        colors = shrunk.view(3, -1).T
        # remove duplicates
        colors = torch.unique(colors, dim=0)
        colors = colors.numpy()
        colors = colors.tolist()

        use_colors = [random.choice(colors) for _ in range(num_colors)]

        pallette = torch.tensor([
            [0, 0, 0],
            mean_color,
            [255, 255, 255],
        ] + use_colors, dtype=torch.float32)
        chunk = quantize(chunk.squeeze(0), pallette).unsqueeze(0)

        # chunk = torchvision.transforms.functional.equalize(chunk)
        # color jitter
        if random.random() < 0.5:
            chunk = torchvision.transforms.functional.adjust_contrast(chunk, random.uniform(1.0, 1.5))
        if random.random() < 0.5:
            chunk = torchvision.transforms.functional.adjust_saturation(chunk, random.uniform(1.0, 2.0))
        # if random.random() < 0.5:
        #     chunk = torchvision.transforms.functional.adjust_brightness(chunk, random.uniform(0.5, 1.5))
        chunk = random_blur(chunk, p=0.6)
        posterized_chunks.append(chunk)

    img = torch.cat(posterized_chunks, dim=0)
    img = img.to(orig_dtype)
    img = img / 255

    if neg1_1:
        img = img * 2 - 1
    return img


# run through an epoch ang check sizes
dataloader_iterator = iter(dataloader)
for epoch in range(args.epochs):
    for batch in tqdm(dataloader):
        batch: 'DataLoaderBatchDTO'
        img_batch = batch.tensor

        # img_batch = color_block_imgs(img_batch, neg1_1=True)

        chunks = torch.chunk(img_batch, batch_size, dim=0)
        # put them so they are size by side
        big_img = torch.cat(chunks, dim=3)
        big_img = big_img.squeeze(0)

        control_chunks = torch.chunk(batch.control_tensor, batch_size, dim=0)
        big_control_img = torch.cat(control_chunks, dim=3)
        big_control_img = big_control_img.squeeze(0) * 2 - 1

        big_img = torch.cat([big_img, big_control_img], dim=2)

        min_val = big_img.min()
        max_val = big_img.max()

        big_img = (big_img / 2 + 0.5).clamp(0, 1)

        # convert to image
        img = transforms.ToPILImage()(big_img)

        # show_img(img)

        # time.sleep(1.0)
    # if not last epoch
    if epoch < args.epochs - 1:
        trigger_dataloader_setup_epoch(dataloader)

cv2.destroyAllWindows()

print('done')
