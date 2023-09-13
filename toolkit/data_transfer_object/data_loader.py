from typing import TYPE_CHECKING, List, Union
import torch
import random

from PIL import Image
from PIL.ImageOps import exif_transpose

from toolkit import image_utils
from toolkit.dataloader_mixins import CaptionProcessingDTOMixin, ImageProcessingDTOMixin, LatentCachingFileItemDTOMixin

if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig

printed_messages = []


def print_once(msg):
    global printed_messages
    if msg not in printed_messages:
        print(msg)
        printed_messages.append(msg)


class FileItemDTO(LatentCachingFileItemDTOMixin, CaptionProcessingDTOMixin, ImageProcessingDTOMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.path = kwargs.get('path', None)
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        # process width and height
        try:
            w, h = image_utils.get_image_size(self.path)
        except image_utils.UnknownImageFormat:
            print_once(f'Warning: Some images in the dataset cannot be fast read. ' + \
                       f'This process is faster for png, jpeg')
            img = exif_transpose(Image.open(self.path))
            h, w = img.size
        self.width: int = w
        self.height: int = h

        # self.caption_path: str = kwargs.get('caption_path', None)
        self.raw_caption: str = kwargs.get('raw_caption', None)
        # we scale first, then crop
        self.scale_to_width: int = kwargs.get('scale_to_width', int(self.width * self.dataset_config.scale))
        self.scale_to_height: int = kwargs.get('scale_to_height', int(self.height * self.dataset_config.scale))
        # crop values are from scaled size
        self.crop_x: int = kwargs.get('crop_x', 0)
        self.crop_y: int = kwargs.get('crop_y', 0)
        self.crop_width: int = kwargs.get('crop_width', self.scale_to_width)
        self.crop_height: int = kwargs.get('crop_height', self.scale_to_height)

        self.network_weight: float = self.dataset_config.network_weight
        self.is_reg = self.dataset_config.is_reg
        self.tensor: Union[torch.Tensor, None] = None

    def cleanup(self):
        self.tensor = None
        self.cleanup_latent()


class DataLoaderBatchDTO:
    def __init__(self, **kwargs):
        self.file_items: List['FileItemDTO'] = kwargs.get('file_items', None)
        is_latents_cached = self.file_items[0].is_latent_cached
        self.tensor: Union[torch.Tensor, None] = None
        self.latents: Union[torch.Tensor, None] = None
        if not is_latents_cached:
            # only return a tensor if latents are not cached
            self.tensor: torch.Tensor = torch.cat([x.tensor.unsqueeze(0) for x in self.file_items])
        # if we have encoded latents, we concatenate them
        self.latents: Union[torch.Tensor, None] = None
        if is_latents_cached:
            self.latents = torch.cat([x.get_latent().unsqueeze(0) for x in self.file_items])

    def get_is_reg_list(self):
        return [x.is_reg for x in self.file_items]

    def get_network_weight_list(self):
        return [x.network_weight for x in self.file_items]

    def get_caption_list(
            self,
            trigger=None,
            to_replace_list=None,
            add_if_not_present=True
    ):
        return [x.get_caption(
            trigger=trigger,
            to_replace_list=to_replace_list,
            add_if_not_present=add_if_not_present
        ) for x in self.file_items]

    def cleanup(self):
        del self.latents
        del self.tensor
        for file_item in self.file_items:
            file_item.cleanup()
