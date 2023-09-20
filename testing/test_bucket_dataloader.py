import time

import numpy as np
import torch
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
from toolkit.data_loader import AiToolkitDataset, get_dataloader_from_datasets
from toolkit.config_modules import DatasetConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_folder', type=str, default='input')


args = parser.parse_args()

dataset_folder = args.dataset_folder
resolution = 512
bucket_tolerance = 64
batch_size = 4

dataset_config = DatasetConfig(
    dataset_path=dataset_folder,
    resolution=resolution,
    caption_ext='txt',
    default_caption='default',
    buckets=True,
    bucket_tolerance=bucket_tolerance,
    augments=['ColorJitter', 'RandomEqualize'],

)

dataloader = get_dataloader_from_datasets([dataset_config], batch_size=batch_size)


# run through an epoch ang check sizes
for batch in dataloader:
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

    time.sleep(1.0)

cv2.destroyAllWindows()

print('done')
