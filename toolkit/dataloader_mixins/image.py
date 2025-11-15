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


class ImageProcessingDTOMixin:
    def load_and_process_video(
        self: 'FileItemDTO',
        transform: Union[None, transforms.Compose],
        only_load_latents=False
    ):
        if self.is_latent_cached:
            raise Exception('Latent caching not supported for videos')
        
        if self.augments is not None and len(self.augments) > 0:
            raise Exception('Augments not supported for videos')
            
        if self.has_augmentations:
            raise Exception('Augmentations not supported for videos')
        
        if not self.dataset_config.buckets:
            raise Exception('Buckets required for video processing')
        
        try:
            # Use OpenCV to capture video frames
            cap = cv2.VideoCapture(self.path)
            
            if not cap.isOpened():
                raise Exception(f"Failed to open video file: {self.path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate the max valid frame index (accounting for zero-indexing)
            max_frame_index = total_frames - 1
            
            # Only log video properties if in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"Video properties: {self.path}")
                print_acc(f"  Total frames: {total_frames}")
                print_acc(f"  Max valid frame index: {max_frame_index}")
                print_acc(f"  FPS: {video_fps}")
            
            frames_to_extract = []
            
            # Always stretch/shrink to the requested number of frames if needed
            if self.dataset_config.shrink_video_to_frames or total_frames < self.dataset_config.num_frames:
                # Distribute frames evenly across the entire video
                interval = max_frame_index / (self.dataset_config.num_frames - 1) if self.dataset_config.num_frames > 1 else 0
                frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(self.dataset_config.num_frames)]
            else:
                # Calculate frame interval based on FPS ratio
                fps_ratio = video_fps / self.dataset_config.fps
                frame_interval = max(1, int(round(fps_ratio)))
                
                # Calculate max consecutive frames we can extract at desired FPS
                max_consecutive_frames = (total_frames // frame_interval)
                
                if max_consecutive_frames < self.dataset_config.num_frames:
                    # Not enough frames at desired FPS, so stretch instead
                    interval = max_frame_index / (self.dataset_config.num_frames - 1) if self.dataset_config.num_frames > 1 else 0
                    frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(self.dataset_config.num_frames)]
                else:
                    # Calculate max start frame to ensure we can get all num_frames
                    max_start_frame = max_frame_index - ((self.dataset_config.num_frames - 1) * frame_interval)
                    start_frame = random.randint(0, max(0, max_start_frame))
                    
                    # Generate list of frames to extract
                    frames_to_extract = [start_frame + (i * frame_interval) for i in range(self.dataset_config.num_frames)]
                    
            # Final safety check - ensure no frame exceeds max valid index
            frames_to_extract = [min(frame_idx, max_frame_index) for frame_idx in frames_to_extract]
            
            # Only log frames to extract if in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"  Frames to extract: {frames_to_extract}")
            
            # Extract frames
            frames = []
            for frame_idx in frames_to_extract:
                # Safety check - ensure frame_idx is within bounds (silently fix)
                if frame_idx > max_frame_index:
                    frame_idx = max_frame_index
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Silently verify position was set correctly (no warnings unless debug mode)
                if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if actual_pos != frame_idx:
                        print_acc(f"Warning: Failed to set exact frame position. Requested: {frame_idx}, Actual: {actual_pos}")
                
                ret, frame = cap.read()
                if not ret:
                    # Try to provide more detailed error information
                    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_pos_info = f"Requested frame: {frame_idx}, Actual frame position: {actual_frame}"
                    
                    # Try to read the next available frame as a fallback
                    fallback_success = False
                    for fallback_offset in [1, -1, 5, -5, 10, -10]:
                        fallback_pos = max(0, min(frame_idx + fallback_offset, max_frame_index))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_pos)
                        fallback_ret, fallback_frame = cap.read()
                        if fallback_ret:
                            # Only log in debug mode
                            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                                print_acc(f"Falling back to nearby frame {fallback_pos} instead of {frame_idx}")
                            frame = fallback_frame
                            fallback_success = True
                            break
                    else:
                        # No fallback worked, raise a more detailed exception
                        video_info = f"Video: {self.path}, Total frames: {total_frames}, FPS: {video_fps}"
                        raise Exception(f"Failed to read frame {frame_idx} from video. {frame_pos_info}. {video_info}")
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                img = Image.fromarray(frame)
                
                # Apply the same processing as for single images
                img = img.convert('RGB')
                
                if self.flip_x:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.flip_y:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                # Apply bucketing
                img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
                img = img.crop((
                    self.crop_x,
                    self.crop_y,
                    self.crop_x + self.crop_width,
                    self.crop_y + self.crop_height
                ))
                
                # Apply transform if provided
                if transform:
                    img = transform(img)
                
                frames.append(img)
            
            # Release the video capture
            cap.release()
            
            # Stack frames into tensor [frames, channels, height, width]
            self.tensor = torch.stack(frames)
            
            # Only log success in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"Successfully loaded video with {len(frames)} frames: {self.path}")
        
        except Exception as e:
            # Print full traceback
            traceback.print_exc()
            
            # Provide more context about the error
            error_msg = str(e)
            try:
                if 'Failed to read frame' in error_msg and cap is not None:
                    # Try to get more info about the video that failed
                    cap_status = "Opened" if cap.isOpened() else "Closed"
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if cap.isOpened() else "Unknown"
                    reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "Unknown"
                    
                    print_acc(f"Video details when error occurred:")
                    print_acc(f"  Cap status: {cap_status}")
                    print_acc(f"  Current position: {current_pos}")
                    print_acc(f"  Reported total frames: {reported_total}")
                    
                    # Try to verify if the video is corrupted
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go to start
                        start_ret, _ = cap.read()
                        
                        # Try to read the last frame to check if it's accessible
                        if reported_total > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, reported_total - 1)
                            end_ret, _ = cap.read()
                            print_acc(f"  Can read first frame: {start_ret}, Can read last frame: {end_ret}")
                    
                    # Close the cap if it's still open
                    cap.release()
            except Exception as debug_err:
                print_acc(f"Error during error diagnosis: {debug_err}")
            
            print_acc(f"Error: {error_msg}")
            print_acc(f"Error loading video: {self.path}")
            
            # Re-raise with more detailed information
            raise Exception(f"Video loading error ({self.path}): {error_msg}") from e
        
    def load_and_process_image(
            self: 'FileItemDTO',
            transform: Union[None, transforms.Compose],
            only_load_latents=False
    ):
        if self.dataset_config.num_frames > 1:
            self.load_and_process_video(transform, only_load_latents)
            return
        # handle get_prompt_embedding
        if self.is_text_embedding_cached:
            self.load_prompt_embedding()
        # if we are caching latents, just do that
        if self.is_latent_cached:
            self.get_latent()
            if self.has_control_image:
                self.load_control_image()
            if self.has_inpaint_image:
                self.load_inpaint_image()
            if self.has_clip_image:
                self.load_clip_image()
            if self.has_mask_image:
                self.load_mask_image()
            if self.has_unconditional:
                self.load_unconditional_image()
            return
        try:
            img = Image.open(self.path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.path}")

        if self.use_alpha_as_mask:
            # we do this to make sure it does not replace the alpha with another color
            # we want the image just without the alpha channel
            np_img = np.array(img)
            # strip off alpha
            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(
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
            # crop to x_crop, y_crop, x_crop + crop_width, y_crop + crop_height
            if img.width < self.crop_x + self.crop_width or img.height < self.crop_y + self.crop_height:
                # todo look into this. This still happens sometimes
                print_acc('size mismatch')
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))

            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
        else:
            # Downscale the source image first
            # TODO this is nto right
            img = img.resize(
                (int(img.size[0] * self.dataset_config.scale), int(img.size[1] * self.dataset_config.scale)),
                Image.BICUBIC)
            min_img_size = min(img.size)
            if self.dataset_config.random_crop:
                if self.dataset_config.random_scale and min_img_size > self.dataset_config.resolution:
                    if min_img_size < self.dataset_config.resolution:
                        print_acc(
                            f"Unexpected values: min_img_size={min_img_size}, self.resolution={self.dataset_config.resolution}, image file={self.path}")
                        scale_size = self.dataset_config.resolution
                    else:
                        scale_size = random.randint(self.dataset_config.resolution, int(min_img_size))
                    scaler = scale_size / min_img_size
                    scale_width = int((img.width + 5) * scaler)
                    scale_height = int((img.height + 5) * scaler)
                    img = img.resize((scale_width, scale_height), Image.BICUBIC)
                img = transforms.RandomCrop(self.dataset_config.resolution)(img)
            else:
                img = transforms.CenterCrop(min_img_size)(img)
                img = img.resize((self.dataset_config.resolution, self.dataset_config.resolution), Image.BICUBIC)

        if self.augments is not None and len(self.augments) > 0:
            # do augmentations
            for augment in self.augments:
                if augment in transforms_dict:
                    img = transforms_dict[augment](img)

        if self.has_augmentations:
            # augmentations handles transforms
            img = self.augment_image(img, transform=transform)
        elif transform:
            img = transform(img)

        self.tensor = img
        if not only_load_latents:
            if self.has_control_image:
                self.load_control_image()
            if self.has_inpaint_image:
                self.load_inpaint_image()
            if self.has_clip_image:
                self.load_clip_image()
            if self.has_mask_image:
                self.load_mask_image()
            if self.has_unconditional:
                self.load_unconditional_image()



class AugmentationFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_augmentations = False
        self.unaugmented_tensor: Union[torch.Tensor, None] = None
        # self.augmentations: Union[None, List[Augments]] = None
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.aug_transform: Union[None, A.Compose] = None
        self.aug_replay_spatial_transforms = None
        self.build_augmentation_transform()

    def build_augmentation_transform(self: 'FileItemDTO'):
        if self.dataset_config.augmentations is not None and len(self.dataset_config.augmentations) > 0:
            self.has_augmentations = True
            augmentations = [Augments(**aug) for aug in self.dataset_config.augmentations]

            if self.dataset_config.shuffle_augmentations:
                random.shuffle(augmentations)

            augmentation_list = []
            for aug in augmentations:
                # make sure method name is valid
                assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
                # get the method
                method = getattr(A, aug.method_name)
                # add the method to the list
                augmentation_list.append(method(**aug.params))

            # add additional targets so we can augment the control image
            self.aug_transform = A.ReplayCompose(augmentation_list, additional_targets={'image2': 'image'})

    def augment_image(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose], ):

        # rebuild each time if shuffle
        if self.dataset_config.shuffle_augmentations:
            self.build_augmentation_transform()

        # save the original tensor
        self.unaugmented_tensor = transforms.ToTensor()(img) if transform is None else transform(img)

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # apply augmentations
        transformed = self.aug_transform(image=open_cv_image)
        augmented = transformed["image"]

        # save just the spatial transforms for controls and masks
        augmented_params = transformed["replay"]
        spatial_transforms = ['Rotate', 'Flip', 'HorizontalFlip', 'VerticalFlip', 'Resize', 'Crop', 'RandomCrop',
                              'ElasticTransform', 'GridDistortion', 'OpticalDistortion']
        # only store the spatial transforms
        augmented_params['transforms'] = [t for t in augmented_params['transforms'] if t['__class_fullname__'].split('.')[-1] in spatial_transforms]

        if self.dataset_config.replay_transforms:
            self.aug_replay_spatial_transforms = augmented_params

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)

        return augmented_tensor

    # augment control images spatially consistent with transforms done to the main image
    def augment_spatial_control(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose] ):
        if self.aug_replay_spatial_transforms is None:
            # no transforms
            return transform(img)

        # save colorspace to convert back to
        colorspace = img.mode

        # convert to rgb
        img = img.convert('RGB')

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Replay transforms
        transformed = A.ReplayCompose.replay(self.aug_replay_spatial_transforms, image=open_cv_image)
        augmented = transformed["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        # convert back to original colorspace
        augmented = augmented.convert(colorspace)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)
        return augmented_tensor

    def cleanup_control(self: 'FileItemDTO'):
        self.unaugmented_tensor = None


