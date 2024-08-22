import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
import cv2
import random
from transformers import CLIPImageProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolkit.paths import SD_SCRIPTS_ROOT
import torchvision.transforms.functional
from toolkit.image_utils import show_img, show_tensors

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

clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

class FakeAdapter:
    def __init__(self):
        self.clip_image_processor = clip_processor


## make fake sd
class FakeSD:
    def __init__(self):
        self.adapter = FakeAdapter()




dataset_config = DatasetConfig(
    dataset_path=dataset_folder,
    # clip_image_path=dataset_folder,
    # square_crop=True,
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

dataloader: DataLoader = get_dataloader_from_datasets([dataset_config], batch_size=batch_size, sd=FakeSD())


# run through an epoch ang check sizes
dataloader_iterator = iter(dataloader)
for epoch in range(args.epochs):
    for batch in tqdm(dataloader):
        batch: 'DataLoaderBatchDTO'
        img_batch = batch.tensor
        batch_size, channels, height, width = img_batch.shape

        # img_batch = color_block_imgs(img_batch, neg1_1=True)

        # chunks = torch.chunk(img_batch, batch_size, dim=0)
        # # put them so they are size by side
        # big_img = torch.cat(chunks, dim=3)
        # big_img = big_img.squeeze(0)
        #
        # control_chunks = torch.chunk(batch.clip_image_tensor, batch_size, dim=0)
        # big_control_img = torch.cat(control_chunks, dim=3)
        # big_control_img = big_control_img.squeeze(0) * 2 - 1
        #
        #
        # # resize control image
        # big_control_img = torchvision.transforms.Resize((width, height))(big_control_img)
        #
        # big_img = torch.cat([big_img, big_control_img], dim=2)
        #
        # min_val = big_img.min()
        # max_val = big_img.max()
        #
        # big_img = (big_img / 2 + 0.5).clamp(0, 1)

        big_img = img_batch
        # big_img = big_img.clamp(-1, 1)

        show_tensors(big_img)

        # convert to image
        # img = transforms.ToPILImage()(big_img)
        #
        # show_img(img)

        time.sleep(0.2)
    # if not last epoch
    if epoch < args.epochs - 1:
        trigger_dataloader_setup_epoch(dataloader)

cv2.destroyAllWindows()

print('done')
