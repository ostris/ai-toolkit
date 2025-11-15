"""
Dataloader mixins - refactored for better organization.
"""

import base64
import glob
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Dict, Union
import traceback

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor

from toolkit.basic import flush, value_map
from toolkit.buckets import get_bucket_for_image_size, get_resolution
from toolkit.config_modules import ControlTypes
from toolkit.control_generator import ControlGenerator
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.pixtral_vision import PixtralVisionImagePreprocessorCompatible
from toolkit.prompt_utils import inject_trigger_into_prompt
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageOps import exif_transpose
import albumentations as A
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator
from toolkit.prompt_utils import PromptEmbeds
from torchvision.transforms import functional as TF
from toolkit.train_tools import get_torch_dtype

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.stable_diffusion_model import StableDiffusion

accelerator = get_accelerator()


class InpaintControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_inpaint_image = False
        self.inpaint_path: Union[str, None] = None
        self.inpaint_tensor: Union[torch.Tensor, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.inpaint_path is not None:
            # find the control image path
            inpaint_path = dataset_config.inpaint_path
            # we are using control images
            img_path = kwargs.get('path', None)
            img_inpaint_ext_list = ['.png', '.webp']
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]

            for ext in img_inpaint_ext_list:
                p = os.path.join(inpaint_path, file_name_no_ext + ext)
                if os.path.exists(p):
                    self.inpaint_path = p
                    self.has_inpaint_image = True
                    break
                
    def load_inpaint_image(self: 'FileItemDTO'):
        try:
            # image must have alpha channel for inpaint
            img = Image.open(self.inpaint_path)
            # make sure has aplha
            if img.mode != 'RGBA':
                return
            img = exif_transpose(img)
        
            w, h = img.size
            if w > h and self.scale_to_width < self.scale_to_height:
                # throw error, they should match
                raise ValueError(
                    f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            elif h > w and self.scale_to_height < self.scale_to_width:
                # throw error, they should match
                raise ValueError(
                    f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

            if self.flip_x:
                # do a flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.flip_y:
                # do a flip
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            if self.dataset_config.buckets:
                # scale and crop based on file item
                img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
                # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
                # crop
                img = img.crop((
                    self.crop_x,
                    self.crop_y,
                    self.crop_x + self.crop_width,
                    self.crop_y + self.crop_height
                ))
            else:
                raise Exception("Inpaint images not supported for non-bucket datasets")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if self.aug_replay_spatial_transforms:
                tensor = self.augment_spatial_control(img, transform=transform)
            else:
                tensor = transform(img)
            
            # is 0 to 1 with alpha
            self.inpaint_tensor = tensor
        
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.inpaint_path}")

    
    def cleanup_inpaint(self: 'FileItemDTO'):
        self.inpaint_tensor = None
                


class ControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_control_image = False
        self.control_path: Union[str, List[str], None] = None
        self.control_tensor: Union[torch.Tensor, None] = None
        self.control_tensor_list: Union[List[torch.Tensor], None] = None
        sd = kwargs.get('sd', None)
        self.use_raw_control_images = sd is not None and sd.use_raw_control_images
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.full_size_control_images = False
        if dataset_config.control_path is not None:
            # find the control image path
            control_path_list = dataset_config.control_path
            if not isinstance(control_path_list, list):
                control_path_list = [control_path_list]
            self.full_size_control_images = dataset_config.full_size_control_images
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            
            found_control_images = []
            for control_path in control_path_list:
                for ext in img_ext_list:
                    if os.path.exists(os.path.join(control_path, file_name_no_ext + ext)):
                        found_control_images.append(os.path.join(control_path, file_name_no_ext + ext))
                        self.has_control_image = True
                        break
            self.control_path = found_control_images
            if len(self.control_path) == 0:
                self.control_path = None
            elif len(self.control_path) == 1:
                # only do one
                self.control_path = self.control_path[0]

    def load_control_image(self: 'FileItemDTO'):
        control_tensors = []
        control_path_list = self.control_path
        if not isinstance(self.control_path, list):
            control_path_list = [self.control_path]
        
        for control_path in control_path_list:
            try:
                img = Image.open(control_path)
                img = exif_transpose(img)

                if img.mode in ("RGBA", "LA"):
                    # Create a background with the specified transparent color
                    transparent_color = tuple(self.dataset_config.control_transparent_color)
                    background = Image.new("RGB", img.size, transparent_color)
                    # Paste the image on top using its alpha channel as mask
                    background.paste(img, mask=img.getchannel("A"))
                    img = background
                else:
                    # Already no alpha channel
                    img = img.convert("RGB")
            except Exception as e:
                print_acc(f"Error: {e}")
                print_acc(f"Error loading image: {control_path}")
            
            if not self.full_size_control_images:
                # we just scale them to 512x512:
                w, h = img.size
                img = img.resize((512, 512), Image.BICUBIC)

            elif not self.use_raw_control_images:
                w, h = img.size
                if self.flip_x:
                    # do a flip
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.flip_y:
                    # do a flip
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)

                if self.dataset_config.buckets:
                    # scale and crop based on file item
                    img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
                    # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
                    # crop
                    img = img.crop((
                        self.crop_x,
                        self.crop_y,
                        self.crop_x + self.crop_width,
                        self.crop_y + self.crop_height
                    ))
                else:
                    raise Exception("Control images not supported for non-bucket datasets")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if self.aug_replay_spatial_transforms:
                tensor = self.augment_spatial_control(img, transform=transform)
            else:
                tensor = transform(img)
            control_tensors.append(tensor)
            
        if len(control_tensors) == 0:
            self.control_tensor = None
        elif len(control_tensors) == 1:
            self.control_tensor = control_tensors[0]
        elif self.use_raw_control_images:
            # just send the list of tensors as their shapes wont match
            self.control_tensor_list = control_tensors
        else:
            self.control_tensor = torch.stack(control_tensors, dim=0)

    def cleanup_control(self: 'FileItemDTO'):
        self.control_tensor = None
        self.control_tensor_list = None



class ClipImageFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_clip_image = False
        self.clip_image_path: Union[str, None] = None
        self.clip_image_tensor: Union[torch.Tensor, None] = None
        self.clip_image_embeds: Union[dict, None] = None
        self.clip_image_embeds_unconditional: Union[dict, None] = None
        self.has_clip_augmentations = False
        self.clip_image_aug_transform: Union[None, A.Compose] = None
        self.clip_image_processor: Union[None, CLIPImageProcessor] = None
        self.clip_image_encoder_path: Union[str, None] = None
        self.is_caching_clip_vision_to_disk = False
        self.is_vision_clip_cached = False
        self.clip_vision_is_quad = False
        self.clip_vision_load_device = 'cpu'
        self.clip_vision_unconditional_paths: Union[List[str], None] = None
        self._clip_vision_embeddings_path: Union[str, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.clip_image_path is not None or dataset_config.clip_image_from_same_folder:
            # copy the clip image processor so the dataloader can do it
            sd = kwargs.get('sd', None)
            if hasattr(sd.adapter, 'clip_image_processor'):
                self.clip_image_processor = sd.adapter.clip_image_processor
        if dataset_config.clip_image_path is not None:
            # find the control image path
            clip_image_path = dataset_config.clip_image_path
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(clip_image_path, file_name_no_ext + ext)):
                    self.clip_image_path = os.path.join(clip_image_path, file_name_no_ext + ext)
                    self.has_clip_image = True
                    break
            self.build_clip_imag_augmentation_transform()
            
        if dataset_config.clip_image_from_same_folder:
            # assume we have one. We will pull it on load.
            self.has_clip_image = True
            self.build_clip_imag_augmentation_transform()

    def build_clip_imag_augmentation_transform(self: 'FileItemDTO'):
        if self.dataset_config.clip_image_augmentations is not None and len(self.dataset_config.clip_image_augmentations) > 0:
            self.has_clip_augmentations = True
            augmentations = [Augments(**aug) for aug in self.dataset_config.clip_image_augmentations]

            if self.dataset_config.clip_image_shuffle_augmentations:
                random.shuffle(augmentations)

            augmentation_list = []
            for aug in augmentations:
                # make sure method name is valid
                assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
                # get the method
                method = getattr(A, aug.method_name)
                # add the method to the list
                augmentation_list.append(method(**aug.params))

            self.clip_image_aug_transform = A.Compose(augmentation_list)

    def augment_clip_image(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose], ):
        if self.dataset_config.clip_image_shuffle_augmentations:
            self.build_clip_imag_augmentation_transform()

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        if self.clip_vision_is_quad:
            # image is in a 2x2 gris. split, run augs, and recombine
            # split
            img1, img2 = np.hsplit(open_cv_image, 2)
            img1_1, img1_2 = np.vsplit(img1, 2)
            img2_1, img2_2 = np.vsplit(img2, 2)
            # apply augmentations
            img1_1 = self.clip_image_aug_transform(image=img1_1)["image"]
            img1_2 = self.clip_image_aug_transform(image=img1_2)["image"]
            img2_1 = self.clip_image_aug_transform(image=img2_1)["image"]
            img2_2 = self.clip_image_aug_transform(image=img2_2)["image"]
            # recombine
            augmented = np.vstack((np.hstack((img1_1, img1_2)), np.hstack((img2_1, img2_2))))

        else:
            # apply augmentations
            augmented = self.clip_image_aug_transform(image=open_cv_image)["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)

        return augmented_tensor

    def get_clip_vision_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("image_encoder_path", self.clip_image_encoder_path),
            ("filename", os.path.basename(self.clip_image_path)),
            ("is_quad", self.clip_vision_is_quad)
        ])
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        return item
    def get_clip_vision_embeddings_path(self: 'FileItemDTO', recalculate=False):
        if self._clip_vision_embeddings_path is not None and not recalculate:
            return self._clip_vision_embeddings_path
        else:
            # we store latents in a folder in same path as image called _latent_cache
            img_dir = os.path.dirname(self.clip_image_path)
            latent_dir = os.path.join(img_dir, '_clip_vision_cache')
            hash_dict = self.get_clip_vision_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.clip_image_path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._clip_vision_embeddings_path = os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._clip_vision_embeddings_path
    
    def get_new_clip_image_path(self: 'FileItemDTO'):
        if self.dataset_config.clip_image_from_same_folder:
            # randomly grab an image path from the same folder
            pool_folder = os.path.dirname(self.path)
            # find all images in the folder
            img_files = []
            for ext in img_ext_list:
                img_files += glob.glob(os.path.join(pool_folder, f'*{ext}'))
            # remove the current image if len is greater than 1
            if len(img_files) > 1:
                img_files.remove(self.path)
            # randomly grab one
            return random.choice(img_files)
        else:
            return self.clip_image_path

    def load_clip_image(self: 'FileItemDTO'):
        is_dynamic_size_and_aspect = isinstance(self.clip_image_processor, PixtralVisionImagePreprocessorCompatible) or \
                                    isinstance(self.clip_image_processor, SiglipImageProcessor)
        if self.clip_image_processor is None:
            is_dynamic_size_and_aspect = True # serving it raw
        if self.is_vision_clip_cached:
            # Use memory-mapped loading for CLIP embeddings to reduce RAM usage
            # CLIP embeddings can be large (multiple tensors with high dimensions)
            clip_path = self.get_clip_vision_embeddings_path()
            with safe_open(clip_path, framework="pt", device="cpu") as f:
                # Load all tensors from the file
                self.clip_image_embeds = {key: f.get_tensor(key).clone() for key in f.keys()}

            # get a random unconditional image
            if self.clip_vision_unconditional_paths is not None:
                unconditional_path = random.choice(self.clip_vision_unconditional_paths)
                with safe_open(unconditional_path, framework="pt", device="cpu") as f:
                    self.clip_image_embeds_unconditional = {key: f.get_tensor(key).clone() for key in f.keys()}

            return
        clip_image_path = self.get_new_clip_image_path()
        try:
            img = Image.open(clip_image_path).convert('RGB')
            img = exif_transpose(img)
        except Exception as e:
            # make a random noise image
            img = Image.new('RGB', (self.dataset_config.resolution, self.dataset_config.resolution))
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {clip_image_path}")

        img = img.convert('RGB')

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
        if is_dynamic_size_and_aspect:
            pass  # let the image processor handle it
        elif img.width != img.height:
            min_size = min(img.width, img.height)
            if self.dataset_config.square_crop:
                # center crop to a square
                img = transforms.CenterCrop(min_size)(img)
            else:
                # image must be square. If it is not, we will resize/squish it so it is, that way we don't crop out data
                # resize to the smallest dimension
                img = img.resize((min_size, min_size), Image.BICUBIC)

        if self.has_clip_augmentations:
            self.clip_image_tensor = self.augment_clip_image(img, transform=None)
        else:
            self.clip_image_tensor = transforms.ToTensor()(img)

        # random crop
        # if self.dataset_config.clip_image_random_crop:
        #     # crop up to 20% on all sides. Keep is square
        #     crop_percent = random.randint(0, 20) / 100
        #     crop_width = int(self.clip_image_tensor.shape[2] * crop_percent)
        #     crop_height = int(self.clip_image_tensor.shape[1] * crop_percent)
        #     crop_left = random.randint(0, crop_width)
        #     crop_top = random.randint(0, crop_height)
        #     crop_right = self.clip_image_tensor.shape[2] - crop_width - crop_left
        #     crop_bottom = self.clip_image_tensor.shape[1] - crop_height - crop_top
        #     if len(self.clip_image_tensor.shape) == 3:
        #         self.clip_image_tensor = self.clip_image_tensor[:, crop_top:-crop_bottom, crop_left:-crop_right]
        #     elif len(self.clip_image_tensor.shape) == 4:
        #         self.clip_image_tensor = self.clip_image_tensor[:, :, crop_top:-crop_bottom, crop_left:-crop_right]

        if self.clip_image_processor is not None:
            # run it
            tensors_0_1 = self.clip_image_tensor.to(dtype=torch.float16)
            clip_out = self.clip_image_processor(
                images=tensors_0_1,
                return_tensors="pt",
                do_resize=True,
                do_rescale=False,
            ).pixel_values
            self.clip_image_tensor = clip_out.squeeze(0).clone().detach()

    def cleanup_clip_image(self: 'FileItemDTO'):
        self.clip_image_tensor = None
        self.clip_image_embeds = None





class UnconditionalFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_unconditional = False
        self.unconditional_path: Union[str, None] = None
        self.unconditional_tensor: Union[torch.Tensor, None] = None
        self.unconditional_latent: Union[torch.Tensor, None] = None
        self.unconditional_transforms = self.dataloader_transforms
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)

        if dataset_config.unconditional_path is not None:
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(dataset_config.unconditional_path, file_name_no_ext + ext)):
                    self.unconditional_path = os.path.join(dataset_config.unconditional_path, file_name_no_ext + ext)
                    self.has_unconditional = True
                    break

    def load_unconditional_image(self: 'FileItemDTO'):
        try:
            img = Image.open(self.unconditional_path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
            # crop
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            raise Exception("Unconditional images are not supported for non-bucket datasets")

        if self.aug_replay_spatial_transforms:
            self.unconditional_tensor = self.augment_spatial_control(img, transform=self.unconditional_transforms)
        else:
            self.unconditional_tensor = self.unconditional_transforms(img)

    def cleanup_unconditional(self: 'FileItemDTO'):
        self.unconditional_tensor = None
        self.unconditional_latent = None


