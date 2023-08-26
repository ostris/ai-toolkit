from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
# make sure we can import from the toolkit
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    folder_path=dataset_folder,
    resolution=resolution,
    caption_type='txt',
    default_caption='default',
    buckets=True,
    bucket_tolerance=bucket_tolerance,
)

dataloader = get_dataloader_from_datasets([dataset_config], batch_size=batch_size)

# run through an epoch ang check sizes
for batch in dataloader:
    print(list(batch[0].shape))

print('done')
