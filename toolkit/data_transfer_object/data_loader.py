from typing import TYPE_CHECKING
import torch
import random

from toolkit.dataloader_mixins import CaptionProcessingDTOMixin

if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig


class FileItemDTO(CaptionProcessingDTOMixin):
    def __init__(self, **kwargs):
        self.path = kwargs.get('path', None)
        self.caption_path: str = kwargs.get('caption_path', None)
        self.raw_caption: str = kwargs.get('raw_caption', None)
        self.width: int = kwargs.get('width', None)
        self.height: int = kwargs.get('height', None)
        # we scale first, then crop
        self.scale_to_width: int = kwargs.get('scale_to_width', self.width)
        self.scale_to_height: int = kwargs.get('scale_to_height', self.height)
        # crop values are from scaled size
        self.crop_x: int = kwargs.get('crop_x', 0)
        self.crop_y: int = kwargs.get('crop_y', 0)
        self.crop_width: int = kwargs.get('crop_width', self.scale_to_width)
        self.crop_height: int = kwargs.get('crop_height', self.scale_to_height)

        # process config
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)

        self.network_network_weight: float = self.dataset_config.network_weight


class DataLoaderBatchDTO:
    def __init__(self, **kwargs):
        self.file_item: 'FileItemDTO' = kwargs.get('file_item', None)
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
