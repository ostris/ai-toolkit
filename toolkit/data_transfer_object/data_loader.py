import os
import weakref
from _weakref import ReferenceType
from typing import TYPE_CHECKING, List, Union
import cv2
import torch
import random

from extensions_built_in.dataset_tools.tools.image_tools import load_image

from toolkit import image_utils
from toolkit.basic import get_quick_signature_string
from toolkit.dataloader_mixins import CaptionProcessingDTOMixin, ImageProcessingDTOMixin, LatentCachingFileItemDTOMixin, \
    ControlFileItemDTOMixin, ArgBreakMixin, PoiFileItemDTOMixin, MaskFileItemDTOMixin, AugmentationFileItemDTOMixin, \
    UnconditionalFileItemDTOMixin, ClipImageFileItemDTOMixin, InpaintControlFileItemDTOMixin


if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig
    from toolkit.stable_diffusion_model import StableDiffusion

printed_messages = []


def print_once(msg):
    global printed_messages
    if msg not in printed_messages:
        print(msg)
        printed_messages.append(msg)


class FileItemDTO(
    LatentCachingFileItemDTOMixin,
    CaptionProcessingDTOMixin,
    ImageProcessingDTOMixin,
    ControlFileItemDTOMixin,
    InpaintControlFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    MaskFileItemDTOMixin,
    AugmentationFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
    PoiFileItemDTOMixin,
    ArgBreakMixin,
):
    def __init__(self, *args, **kwargs):
        self.path = kwargs.get('path', '')
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.is_video = self.dataset_config.num_frames > 1
        size_database = kwargs.get('size_database', {})
        dataset_root =  kwargs.get('dataset_root', None)
        if dataset_root is not None:
            # remove dataset root from path
            file_key = self.path.replace(dataset_root, '')
        else:
            file_key = os.path.basename(self.path)
        
        file_signature = get_quick_signature_string(self.path)
        if file_signature is None:
            raise Exception("Error: Could not get file signature for {self.path}")
        
        use_db_entry = False
        if file_key in size_database:
            db_entry = size_database[file_key]
            if db_entry is not None and len(db_entry) >= 3 and db_entry[2] == file_signature:
                use_db_entry = True
        
        if use_db_entry:
            w, h, _ = size_database[file_key]
        elif self.is_video:
            # Open the video file
            video = cv2.VideoCapture(self.path)
            
            # Check if video opened successfully
            if not video.isOpened():
                raise Exception(f"Error: Could not open video file {self.path}")
            
            # Get width and height
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w, h = width, height
            
            # Release the video capture object immediately
            video.release()
            size_database[file_key] = (width, height, file_signature)
        else:
            # original method is significantly faster, but some images are read sideways. Not sure why. Do slow method for now.
            # process width and height
            # try:
            #     w, h = image_utils.get_image_size(self.path)
            # except image_utils.UnknownImageFormat:
            #     print_once(f'Warning: Some images in the dataset cannot be fast read. ' + \
            #                f'This process is faster for png, jpeg')
            img = load_image(self.path)
            w, h = img.size
            size_database[file_key] = (w, h, file_signature)
        self.width: int = w
        self.height: int = h
        self.dataloader_transforms = kwargs.get('dataloader_transforms', None)
        super().__init__(*args, **kwargs)

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
        self.flip_x: bool = kwargs.get('flip_x', False)
        self.flip_y: bool = kwargs.get('flip_x', False)
        self.augments: List[str] = self.dataset_config.augments
        self.loss_multiplier: float = self.dataset_config.loss_multiplier

        self.network_weight: float = self.dataset_config.network_weight
        self.is_reg = self.dataset_config.is_reg
        self.tensor: Union[torch.Tensor, None] = None

    def cleanup(self):
        self.tensor = None
        self.cleanup_latent()
        self.cleanup_control()
        self.cleanup_inpaint()
        self.cleanup_clip_image()
        self.cleanup_mask()
        self.cleanup_unconditional()


class DataLoaderBatchDTO:
    def __init__(self, **kwargs):
        try:
            self.file_items: List['FileItemDTO'] = kwargs.get('file_items', None)
            is_latents_cached = self.file_items[0].is_latent_cached
            self.tensor: Union[torch.Tensor, None] = None
            self.latents: Union[torch.Tensor, None] = None
            self.control_tensor: Union[torch.Tensor, None] = None
            self.clip_image_tensor: Union[torch.Tensor, None] = None
            self.mask_tensor: Union[torch.Tensor, None] = None
            self.unaugmented_tensor: Union[torch.Tensor, None] = None
            self.unconditional_tensor: Union[torch.Tensor, None] = None
            self.unconditional_latents: Union[torch.Tensor, None] = None
            self.clip_image_embeds: Union[List[dict], None] = None
            self.clip_image_embeds_unconditional: Union[List[dict], None] = None
            self.sigmas: Union[torch.Tensor, None] = None  # can be added elseware and passed along training code
            self.extra_values: Union[torch.Tensor, None] = torch.tensor([x.extra_values for x in self.file_items]) if len(self.file_items[0].extra_values) > 0 else None
            if not is_latents_cached:
                # only return a tensor if latents are not cached
                self.tensor: torch.Tensor = torch.cat([x.tensor.unsqueeze(0) for x in self.file_items])
            # if we have encoded latents, we concatenate them
            self.latents: Union[torch.Tensor, None] = None
            if is_latents_cached:
                self.latents = torch.cat([x.get_latent().unsqueeze(0) for x in self.file_items])
            self.control_tensor: Union[torch.Tensor, None] = None
            # if self.file_items[0].control_tensor is not None:
            # if any have a control tensor, we concatenate them
            if any([x.control_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_control_tensor = None
                for x in self.file_items:
                    if x.control_tensor is not None:
                        base_control_tensor = x.control_tensor
                        break
                control_tensors = []
                for x in self.file_items:
                    if x.control_tensor is None:
                        control_tensors.append(torch.zeros_like(base_control_tensor))
                    else:
                        control_tensors.append(x.control_tensor)
                self.control_tensor = torch.cat([x.unsqueeze(0) for x in control_tensors])
                
            self.inpaint_tensor: Union[torch.Tensor, None] = None
            if any([x.inpaint_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_inpaint_tensor = None
                for x in self.file_items:
                    if x.inpaint_tensor is not None:
                        base_inpaint_tensor = x.inpaint_tensor
                        break
                inpaint_tensors = []
                for x in self.file_items:
                    if x.inpaint_tensor is None:
                        inpaint_tensors.append(torch.zeros_like(base_inpaint_tensor))
                    else:
                        inpaint_tensors.append(x.inpaint_tensor)
                self.inpaint_tensor = torch.cat([x.unsqueeze(0) for x in inpaint_tensors])

            self.loss_multiplier_list: List[float] = [x.loss_multiplier for x in self.file_items]

            if any([x.clip_image_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_clip_image_tensor = None
                for x in self.file_items:
                    if x.clip_image_tensor is not None:
                        base_clip_image_tensor = x.clip_image_tensor
                        break
                clip_image_tensors = []
                for x in self.file_items:
                    if x.clip_image_tensor is None:
                        clip_image_tensors.append(torch.zeros_like(base_clip_image_tensor))
                    else:
                        clip_image_tensors.append(x.clip_image_tensor)
                self.clip_image_tensor = torch.cat([x.unsqueeze(0) for x in clip_image_tensors])

            if any([x.mask_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_mask_tensor = None
                for x in self.file_items:
                    if x.mask_tensor is not None:
                        base_mask_tensor = x.mask_tensor
                        break
                mask_tensors = []
                for x in self.file_items:
                    if x.mask_tensor is None:
                        mask_tensors.append(torch.zeros_like(base_mask_tensor))
                    else:
                        mask_tensors.append(x.mask_tensor)
                self.mask_tensor = torch.cat([x.unsqueeze(0) for x in mask_tensors])

            # add unaugmented tensors for ones with augments
            if any([x.unaugmented_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_unaugmented_tensor = None
                for x in self.file_items:
                    if x.unaugmented_tensor is not None:
                        base_unaugmented_tensor = x.unaugmented_tensor
                        break
                unaugmented_tensor = []
                for x in self.file_items:
                    if x.unaugmented_tensor is None:
                        unaugmented_tensor.append(torch.zeros_like(base_unaugmented_tensor))
                    else:
                        unaugmented_tensor.append(x.unaugmented_tensor)
                self.unaugmented_tensor = torch.cat([x.unsqueeze(0) for x in unaugmented_tensor])

            # add unconditional tensors
            if any([x.unconditional_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_unconditional_tensor = None
                for x in self.file_items:
                    if x.unaugmented_tensor is not None:
                        base_unconditional_tensor = x.unconditional_tensor
                        break
                unconditional_tensor = []
                for x in self.file_items:
                    if x.unconditional_tensor is None:
                        unconditional_tensor.append(torch.zeros_like(base_unconditional_tensor))
                    else:
                        unconditional_tensor.append(x.unconditional_tensor)
                self.unconditional_tensor = torch.cat([x.unsqueeze(0) for x in unconditional_tensor])

            if any([x.clip_image_embeds is not None for x in self.file_items]):
                self.clip_image_embeds = []
                for x in self.file_items:
                    if x.clip_image_embeds is not None:
                        self.clip_image_embeds.append(x.clip_image_embeds)
                    else:
                        raise Exception("clip_image_embeds is None for some file items")

            if any([x.clip_image_embeds_unconditional is not None for x in self.file_items]):
                self.clip_image_embeds_unconditional = []
                for x in self.file_items:
                    if x.clip_image_embeds_unconditional is not None:
                        self.clip_image_embeds_unconditional.append(x.clip_image_embeds_unconditional)
                    else:
                        raise Exception("clip_image_embeds_unconditional is None for some file items")

        except Exception as e:
            print(e)
            raise e

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
        return [x.caption for x in self.file_items]

    def get_caption_short_list(
            self,
            trigger=None,
            to_replace_list=None,
            add_if_not_present=True
    ):
        return [x.caption_short for x in self.file_items]

    def cleanup(self):
        del self.latents
        del self.tensor
        del self.control_tensor
        for file_item in self.file_items:
            file_item.cleanup()
