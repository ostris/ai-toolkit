import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolkit.paths import SD_SCRIPTS_ROOT

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
resolution = 1024
bucket_tolerance = 64
batch_size = 1


##

dataset_config = DatasetConfig(
    dataset_path=dataset_folder,
    resolution=resolution,
    # caption_ext='json',
    default_caption='default',
    # clip_image_path='/mnt/Datasets2/regs/yetibear_xl_v14/random_aspect/',
    buckets=True,
    bucket_tolerance=bucket_tolerance,
    # poi='person',
    # augmentations=[
    #     {
    #         'method': 'RandomBrightnessContrast',
    #         'brightness_limit': (-0.3, 0.3),
    #         'contrast_limit': (-0.3, 0.3),
    #         'brightness_by_max': False,
    #         'p': 1.0
    #     },
    #     {
    #         'method': 'HueSaturationValue',
    #         'hue_shift_limit': (-0, 0),
    #         'sat_shift_limit': (-40, 40),
    #         'val_shift_limit': (-40, 40),
    #         'p': 1.0
    #     },
        # {
        #     'method': 'RGBShift',
        #     'r_shift_limit': (-20, 20),
        #     'g_shift_limit': (-20, 20),
        #     'b_shift_limit': (-20, 20),
        #     'p': 1.0
        # },
    # ]


)

dataloader: DataLoader = get_dataloader_from_datasets([dataset_config], batch_size=batch_size)


# run through an epoch ang check sizes
dataloader_iterator = iter(dataloader)
for epoch in range(args.epochs):
    for batch in tqdm(dataloader):
        batch: 'DataLoaderBatchDTO'
        img_batch = batch.tensor

        chunks = torch.chunk(img_batch, batch_size, dim=0)
        # put them so they are size by side
        big_img = torch.cat(chunks, dim=3)
        big_img = big_img.squeeze(0)

        min_val = big_img.min()
        max_val = big_img.max()

        big_img = (big_img / 2 + 0.5).clamp(0, 1)

        # convert to image
        img = transforms.ToPILImage()(big_img)

        show_img(img)

        # time.sleep(0.1)
    # if not last epoch
    if epoch < args.epochs - 1:
        trigger_dataloader_setup_epoch(dataloader)

cv2.destroyAllWindows()

print('done')
