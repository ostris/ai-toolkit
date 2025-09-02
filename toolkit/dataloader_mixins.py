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
from safetensors.torch import load_file, save_file
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

# def get_associated_caption_from_img_path(img_path):
# https://demo.albumentations.ai/
class Augments:
    def __init__(self, **kwargs):
        self.method_name = kwargs.get('method', None)
        self.params = kwargs.get('params', {})

        # convert kwargs enums for cv2
        for key, value in self.params.items():
            if isinstance(value, str):
                # split the string
                split_string = value.split('.')
                if len(split_string) == 2 and split_string[0] == 'cv2':
                    if hasattr(cv2, split_string[1]):
                        self.params[key] = getattr(cv2, split_string[1].upper())
                    else:
                        raise ValueError(f"invalid cv2 enum: {split_string[1]}")


transforms_dict = {
    'ColorJitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    'RandomEqualize': transforms.RandomEqualize(p=0.2),
}

img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']


def standardize_images(images):
    """
    Standardize the given batch of images using the specified mean and std.
    Expects values of 0 - 1

    Args:
    images (torch.Tensor): A batch of images in the shape of (N, C, H, W),
                           where N is the number of images, C is the number of channels,
                           H is the height, and W is the width.

    Returns:
    torch.Tensor: Standardized images.
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    # Define the normalization transform
    normalize = transforms.Normalize(mean=mean, std=std)

    # Apply normalization to each image in the batch
    standardized_images = torch.stack([normalize(img) for img in images])

    return standardized_images

def clean_caption(caption):
    # this doesnt make any sense anymore in a world that is not based on comma seperated tokens
    # # remove any newlines
    # caption = caption.replace('\n', ', ')
    # # remove new lines for all operating systems
    # caption = caption.replace('\r', ', ')
    # caption_split = caption.split(',')
    # # remove empty strings
    # caption_split = [p.strip() for p in caption_split if p.strip()]
    # # join back together
    # caption = ', '.join(caption_split)
    return caption


class CaptionMixin:
    def get_caption_item(self: 'AiToolkitDataset', index):
        if not hasattr(self, 'caption_type'):
            raise Exception('caption_type not found on class instance')
        if not hasattr(self, 'file_list'):
            raise Exception('file_list not found on class instance')
        img_path_or_tuple = self.file_list[index]
        ext = self.dataset_config.caption_ext
        if isinstance(img_path_or_tuple, tuple):
            img_path = img_path_or_tuple[0] if isinstance(img_path_or_tuple[0], str) else img_path_or_tuple[0].path
            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = None
            prompt_path = path_no_ext + ext
        else:
            img_path = img_path_or_tuple if isinstance(img_path_or_tuple, str) else img_path_or_tuple.path
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = path_no_ext + ext
                
        # allow folders to have a default prompt
        default_prompt_path = os.path.join(os.path.dirname(img_path), 'default.txt')
        default_prompt_path_with_ext = os.path.join(os.path.dirname(img_path), 'default' + ext)

        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                # check if is json
                if prompt_path.endswith('.json'):
                    prompt = json.loads(prompt)
                    if 'caption' in prompt:
                        prompt = prompt['caption']

                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path_with_ext):
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path):
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        else:
            prompt = ''
            # get default_prompt if it exists on the class instance
            if hasattr(self, 'default_prompt'):
                prompt = self.default_prompt
            if hasattr(self, 'default_caption'):
                prompt = self.default_caption

        # handle replacements
        replacement_list = self.dataset_config.replacements if isinstance(self.dataset_config.replacements, list) else []
        for replacement in replacement_list:
            from_string, to_string = replacement.split('|')
            prompt = prompt.replace(from_string, to_string)

        return prompt


if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO


class Bucket:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.file_list_idx: List[int] = []


class BucketsMixin:
    def __init__(self):
        self.buckets: Dict[str, Bucket] = {}
        self.batch_indices: List[List[int]] = []

    def build_batch_indices(self: 'AiToolkitDataset'):
        self.batch_indices = []
        for key, bucket in self.buckets.items():
            for start_idx in range(0, len(bucket.file_list_idx), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(bucket.file_list_idx))
                batch = bucket.file_list_idx[start_idx:end_idx]
                self.batch_indices.append(batch)

    def shuffle_buckets(self: 'AiToolkitDataset'):
        for key, bucket in self.buckets.items():
            random.shuffle(bucket.file_list_idx)

    def setup_buckets(self: 'AiToolkitDataset', quiet=False):
        if not hasattr(self, 'file_list'):
            raise Exception(f'file_list not found on class instance {self.__class__.__name__}')
        if not hasattr(self, 'dataset_config'):
            raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')

        if self.epoch_num > 0 and self.dataset_config.poi is None:
            # no need to rebuild buckets for now
            # todo handle random cropping for buckets
            return
        self.buckets = {}  # clear it

        config: 'DatasetConfig' = self.dataset_config
        resolution = config.resolution
        bucket_tolerance = config.bucket_tolerance
        file_list: List['FileItemDTO'] = self.file_list

        # for file_item in enumerate(file_list):
        for idx, file_item in enumerate(file_list):
            file_item: 'FileItemDTO' = file_item
            width = int(file_item.width * file_item.dataset_config.scale)
            height = int(file_item.height * file_item.dataset_config.scale)

            did_process_poi = False
            if file_item.has_point_of_interest:
                # Attempt to process the poi if we can. It wont process if the image is smaller than the resolution
                did_process_poi = file_item.setup_poi_bucket()
            if self.dataset_config.square_crop:
                # we scale first so smallest size matches resolution
                scale_factor_x = resolution / width
                scale_factor_y = resolution / height
                scale_factor = max(scale_factor_x, scale_factor_y)
                file_item.scale_to_width = math.ceil(width * scale_factor)
                file_item.scale_to_height = math.ceil(height * scale_factor)
                file_item.crop_width = resolution
                file_item.crop_height = resolution
                if width > height:
                    file_item.crop_x = int(file_item.scale_to_width / 2 - resolution / 2)
                    file_item.crop_y = 0
                else:
                    file_item.crop_x = 0
                    file_item.crop_y = int(file_item.scale_to_height / 2 - resolution / 2)
            elif not did_process_poi:
                bucket_resolution = get_bucket_for_image_size(
                    width, height,
                    resolution=resolution,
                    divisibility=bucket_tolerance
                )

                # Calculate scale factors for width and height
                width_scale_factor = bucket_resolution["width"] / width
                height_scale_factor = bucket_resolution["height"] / height

                # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
                max_scale_factor = max(width_scale_factor, height_scale_factor)

                # round up
                file_item.scale_to_width = int(math.ceil(width * max_scale_factor))
                file_item.scale_to_height = int(math.ceil(height * max_scale_factor))

                file_item.crop_height = bucket_resolution["height"]
                file_item.crop_width = bucket_resolution["width"]

                new_width = bucket_resolution["width"]
                new_height = bucket_resolution["height"]

                if self.dataset_config.random_crop:
                    # random crop
                    crop_x = random.randint(0, file_item.scale_to_width - new_width)
                    crop_y = random.randint(0, file_item.scale_to_height - new_height)
                    file_item.crop_x = crop_x
                    file_item.crop_y = crop_y
                else:
                    # do central crop
                    file_item.crop_x = int((file_item.scale_to_width - new_width) / 2)
                    file_item.crop_y = int((file_item.scale_to_height - new_height) / 2)

                if file_item.crop_y < 0 or file_item.crop_x < 0:
                    print_acc('debug')

            # check if bucket exists, if not, create it
            bucket_key = f'{file_item.crop_width}x{file_item.crop_height}'
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(file_item.crop_width, file_item.crop_height)
            self.buckets[bucket_key].file_list_idx.append(idx)

        # print the buckets
        self.shuffle_buckets()
        self.build_batch_indices()
        if not quiet:
            print_acc(f'Bucket sizes for {self.dataset_path}:')
            for key, bucket in self.buckets.items():
                print_acc(f'{key}: {len(bucket.file_list_idx)} files')
            print_acc(f'{len(self.buckets)} buckets made')


class CaptionProcessingDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
            self.raw_caption: str = None
            self.raw_caption_short: str = None
            self.caption: str = None
            self.caption_short: str = None

            dataset_config: DatasetConfig = kwargs.get('dataset_config', None)
            self.extra_values: List[float] = dataset_config.extra_values

    # todo allow for loading from sd-scripts style dict
    def load_caption(self: 'FileItemDTO', caption_dict: Union[dict, None]=None):
        if self.raw_caption is not None:
            # we already loaded it
            pass
        elif caption_dict is not None and self.path in caption_dict and "caption" in caption_dict[self.path]:
            self.raw_caption = caption_dict[self.path]["caption"]
            if 'caption_short' in caption_dict[self.path]:
                self.raw_caption_short = caption_dict[self.path]["caption_short"]
                if self.dataset_config.use_short_captions:
                    self.raw_caption = caption_dict[self.path]["caption_short"]
        else:
            # see if prompt file exists
            path_no_ext = os.path.splitext(self.path)[0]
            prompt_ext = self.dataset_config.caption_ext
            prompt_path = path_no_ext + prompt_ext
            short_caption = None

            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                    short_caption = None
                    if prompt_path.endswith('.json'):
                        # replace any line endings with commas for \n \r \r\n
                        prompt = prompt.replace('\r\n', ' ')
                        prompt = prompt.replace('\n', ' ')
                        prompt = prompt.replace('\r', ' ')

                        prompt_json = json.loads(prompt)
                        if 'caption' in prompt_json:
                            prompt = prompt_json['caption']
                        if 'caption_short' in prompt_json:
                            short_caption = prompt_json['caption_short']
                            if self.dataset_config.use_short_captions:
                                prompt = short_caption
                        if 'extra_values' in prompt_json:
                            self.extra_values = prompt_json['extra_values']

                    prompt = clean_caption(prompt)
                    if short_caption is not None:
                        short_caption = clean_caption(short_caption)
            else:
                prompt = ''
                if self.dataset_config.default_caption is not None:
                    prompt = self.dataset_config.default_caption

            if short_caption is None:
                short_caption = self.dataset_config.default_caption
            self.raw_caption = prompt
            self.raw_caption_short = short_caption

        self.caption = self.get_caption()
        if self.raw_caption_short is not None:
            self.caption_short = self.get_caption(short_caption=True)

    def get_caption(
            self: 'FileItemDTO',
            trigger=None,
            to_replace_list=None,
            add_if_not_present=False,
            short_caption=False
    ):
        if short_caption:
            raw_caption = self.raw_caption_short
        else:
            raw_caption = self.raw_caption
        if raw_caption is None:
            raw_caption = ''
        # handle dropout
        if self.dataset_config.caption_dropout_rate > 0 and not short_caption:
            # get a random float form 0 to 1
            rand = random.random()
            if rand < self.dataset_config.caption_dropout_rate:
                # drop the caption
                return ''

        # get tokens
        token_list = raw_caption.split(',')
        # trim whitespace
        token_list = [x.strip() for x in token_list]
        # remove empty strings
        token_list = [x for x in token_list if x]

        # handle token dropout
        if self.dataset_config.token_dropout_rate > 0 and not short_caption:
            new_token_list = []
            keep_tokens: int = self.dataset_config.keep_tokens
            for idx, token in enumerate(token_list):
                if idx < keep_tokens:
                    new_token_list.append(token)
                elif self.dataset_config.token_dropout_rate >= 1.0:
                    # drop the token
                    pass
                else:
                    # get a random float form 0 to 1
                    rand = random.random()
                    if rand > self.dataset_config.token_dropout_rate:
                        # keep the token
                        new_token_list.append(token)
            token_list = new_token_list

        if self.dataset_config.shuffle_tokens:
            random.shuffle(token_list)

        # join back together
        caption = ', '.join(token_list)
        # caption = inject_trigger_into_prompt(caption, trigger, to_replace_list, add_if_not_present)

        if self.dataset_config.random_triggers:
            num_triggers = self.dataset_config.random_triggers_max
            if num_triggers > 1:
                num_triggers = random.randint(0, num_triggers)

            if num_triggers > 0:
                triggers = random.sample(self.dataset_config.random_triggers, num_triggers)
                caption = caption + ', ' + ', '.join(triggers)
                # add random triggers
                # for i in range(num_triggers):
                #     # fastest method
                #     trigger = self.dataset_config.random_triggers[int(random.random() * (len(self.dataset_config.random_triggers)))]
                #     caption = caption + ', ' + trigger

        if self.dataset_config.shuffle_tokens:
            # shuffle again
            token_list = caption.split(',')
            # trim whitespace
            token_list = [x.strip() for x in token_list]
            # remove empty strings
            token_list = [x for x in token_list if x]
            random.shuffle(token_list)
            caption = ', '.join(token_list)

        return caption


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

            else:
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
        else:
            self.control_tensor = torch.stack(control_tensors, dim=0)

    def cleanup_control(self: 'FileItemDTO'):
        self.control_tensor = None


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
            self.clip_image_embeds = load_file(self.get_clip_vision_embeddings_path())

            # get a random unconditional image
            if self.clip_vision_unconditional_paths is not None:
                unconditional_path = random.choice(self.clip_vision_unconditional_paths)
                self.clip_image_embeds_unconditional = load_file(unconditional_path)

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


class MaskFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_mask_image = False
        self.mask_path: Union[str, None] = None
        self.mask_tensor: Union[torch.Tensor, None] = None
        self.use_alpha_as_mask: bool = False
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.mask_min_value = dataset_config.mask_min_value
        if dataset_config.alpha_mask:
            self.use_alpha_as_mask = True
            self.mask_path = kwargs.get('path', None)
            self.has_mask_image = True
        elif dataset_config.mask_path is not None:
            # find the control image path
            mask_path = dataset_config.mask_path if dataset_config.mask_path is not None else dataset_config.alpha_mask
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(mask_path, file_name_no_ext + ext)):
                    self.mask_path = os.path.join(mask_path, file_name_no_ext + ext)
                    self.has_mask_image = True
                    break

    def load_mask_image(self: 'FileItemDTO'):
        try:
            img = Image.open(self.mask_path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        if self.use_alpha_as_mask:
            # pipeline expectws an rgb image so we need to put alpha in all channels
            np_img = np.array(img)
            np_img[:, :, :3] = np_img[:, :, 3:]

            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        if self.dataset_config.invert_mask:
            img = ImageOps.invert(img)
        w, h = img.size
        fix_size = False
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True

        if fix_size:
            # swap all the sizes
            self.scale_to_width, self.scale_to_height = self.scale_to_height, self.scale_to_width
            self.crop_width, self.crop_height = self.crop_height, self.crop_width
            self.crop_x, self.crop_y = self.crop_y, self.crop_x




        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # randomly apply a blur up to 0.5% of the size of the min (width, height)
        min_size = min(img.width, img.height)
        blur_radius = int(min_size * random.random() * 0.005)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # make grayscale
        img = img.convert('L')

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
            raise Exception("Mask images not supported for non-bucket datasets")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.aug_replay_spatial_transforms:
            self.mask_tensor = self.augment_spatial_control(img, transform=transform)
        else:
            self.mask_tensor = transform(img)
        self.mask_tensor = value_map(self.mask_tensor, 0, 1.0, self.mask_min_value, 1.0)
        # convert to grayscale

    def cleanup_mask(self: 'FileItemDTO'):
        self.mask_tensor = None


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


class PoiFileItemDTOMixin:
    # Point of interest bounding box. Allows for dynamic cropping without cropping out the main subject
    # items in the poi will always be inside the image when random cropping
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        # poi is a name of the box point of interest in the caption json file
        dataset_config = kwargs.get('dataset_config', None)
        path = kwargs.get('path', None)
        self.poi: Union[str, None] = dataset_config.poi
        self.has_point_of_interest = self.poi is not None
        self.poi_x: Union[int, None] = None
        self.poi_y: Union[int, None] = None
        self.poi_width: Union[int, None] = None
        self.poi_height: Union[int, None] = None

        if self.poi is not None:
            # make sure latent caching is off
            if dataset_config.cache_latents or dataset_config.cache_latents_to_disk:
                raise Exception(
                    f"Error: poi is not supported when caching latents. Please set cache_latents and cache_latents_to_disk to False in the dataset config"
                )
                # make sure we are loading through json
            if dataset_config.caption_ext != 'json':
                raise Exception(
                    f"Error: poi is only supported when using json captions. Please set caption_ext to json in the dataset config"
                )
            self.poi = self.poi.strip()
            # get the caption path
            file_path_no_ext = os.path.splitext(path)[0]
            caption_path = file_path_no_ext + '.json'
            if not os.path.exists(caption_path):
                raise Exception(f"Error: caption file not found for poi: {caption_path}")
            with open(caption_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            if 'poi' not in json_data:
                print_acc(f"Warning: poi not found in caption file: {caption_path}")
            if self.poi not in json_data['poi']:
                print_acc(f"Warning: poi not found in caption file: {caption_path}")
            # poi has, x, y, width, height
            # do full image if no poi
            self.poi_x = 0
            self.poi_y = 0
            self.poi_width = self.width
            self.poi_height = self.height
            try:
                if self.poi in json_data['poi']:
                    poi = json_data['poi'][self.poi]
                    self.poi_x = int(poi['x'])
                    self.poi_y = int(poi['y'])
                    self.poi_width = int(poi['width'])
                    self.poi_height = int(poi['height'])
            except Exception as e:
                pass

            # handle flipping
            if kwargs.get('flip_x', False):
                # flip the poi
                self.poi_x = self.width - self.poi_x - self.poi_width
            if kwargs.get('flip_y', False):
                # flip the poi
                self.poi_y = self.height - self.poi_y - self.poi_height

    def setup_poi_bucket(self: 'FileItemDTO'):
        initial_width = int(self.width * self.dataset_config.scale)
        initial_height = int(self.height * self.dataset_config.scale)
        # we are using poi, so we need to calculate the bucket based on the poi

        # if img resolution is less than dataset resolution, just return and let the normal bucketing happen
        img_resolution = get_resolution(initial_width, initial_height)
        if img_resolution <= self.dataset_config.resolution:
            return False  # will trigger normal bucketing

        bucket_tolerance = self.dataset_config.bucket_tolerance
        poi_x = int(self.poi_x * self.dataset_config.scale)
        poi_y = int(self.poi_y * self.dataset_config.scale)
        poi_width = int(self.poi_width * self.dataset_config.scale)
        poi_height = int(self.poi_height * self.dataset_config.scale)

        # loop to keep expanding until we are at the proper resolution. This is not ideal, we can probably handle it better
        num_loops = 0
        while True:
            # crop left
            if poi_x > 0:
                poi_x = random.randint(0, poi_x)
            else:
                poi_x = 0

            # crop right
            cr_min = poi_x + poi_width
            if cr_min < initial_width:
                crop_right = random.randint(poi_x + poi_width, initial_width)
            else:
                crop_right = initial_width

            poi_width = crop_right - poi_x

            if poi_y > 0:
                poi_y = random.randint(0, poi_y)
            else:
                poi_y = 0

            if poi_y + poi_height < initial_height:
                crop_bottom = random.randint(poi_y + poi_height, initial_height)
            else:
                crop_bottom = initial_height

            poi_height = crop_bottom - poi_y
            try:
                # now we have our random crop, but it may be smaller than resolution. Check and expand if needed
                current_resolution = get_resolution(poi_width, poi_height)
            except Exception as e:
                print_acc(f"Error: {e}")
                print_acc(f"Error getting resolution: {self.path}")
                raise e
                return False
            if current_resolution >= self.dataset_config.resolution:
                # We can break now
                break
            else:
                num_loops += 1
                if num_loops > 100:
                    print_acc(
                        f"Warning: poi bucketing looped too many times. This should not happen. Please report this issue.")
                    return False

        new_width = poi_width
        new_height = poi_height

        bucket_resolution = get_bucket_for_image_size(
            new_width, new_height,
            resolution=self.dataset_config.resolution,
            divisibility=bucket_tolerance
        )

        width_scale_factor = bucket_resolution["width"] / new_width
        height_scale_factor = bucket_resolution["height"] / new_height
        # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
        max_scale_factor = max(width_scale_factor, height_scale_factor)

        self.scale_to_width = math.ceil(initial_width * max_scale_factor)
        self.scale_to_height = math.ceil(initial_height * max_scale_factor)
        self.crop_width = bucket_resolution['width']
        self.crop_height = bucket_resolution['height']
        self.crop_x = int(poi_x * max_scale_factor)
        self.crop_y = int(poi_y * max_scale_factor)

        if self.scale_to_width < self.crop_x + self.crop_width or self.scale_to_height < self.crop_y + self.crop_height:
            # todo look into this. This still happens sometimes
            print_acc('size mismatch')

        return True


class ArgBreakMixin:
    # just stops super calls form hitting object
    def __init__(self, *args, **kwargs):
        pass


class LatentCachingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self._encoded_latent: Union[torch.Tensor, None] = None
        self._latent_path: Union[str, None] = None
        self.is_latent_cached = False
        self.is_caching_to_disk = False
        self.is_caching_to_memory = False
        self.latent_load_device = 'cpu'
        # sd1 or sdxl or others
        self.latent_space_version = 'sd1'
        # todo, increment this if we change the latent format to invalidate cache
        self.latent_version = 1

    def get_latent_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("filename", os.path.basename(self.path)),
            ("scale_to_width", self.scale_to_width),
            ("scale_to_height", self.scale_to_height),
            ("crop_x", self.crop_x),
            ("crop_y", self.crop_y),
            ("crop_width", self.crop_width),
            ("crop_height", self.crop_height),
            ("latent_space_version", self.latent_space_version),
            ("latent_version", self.latent_version),
        ])
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        return item

    def get_latent_path(self: 'FileItemDTO', recalculate=False):
        if self._latent_path is not None and not recalculate:
            return self._latent_path
        else:
            # we store latents in a folder in same path as image called _latent_cache
            img_dir = os.path.dirname(self.path)
            latent_dir = os.path.join(img_dir, '_latent_cache')
            hash_dict = self.get_latent_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._latent_path = os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._latent_path

    def cleanup_latent(self):
        if self._encoded_latent is not None:
            if not self.is_caching_to_memory:
                # we are caching on disk, don't save in memory
                self._encoded_latent = None
            else:
                # move it back to cpu
                self._encoded_latent = self._encoded_latent.to('cpu')

    def get_latent(self, device=None):
        if not self.is_latent_cached:
            return None
        if self._encoded_latent is None:
            # load it from disk
            state_dict = load_file(
                self.get_latent_path(),
                # device=device if device is not None else self.latent_load_device
                device='cpu'
            )
            self._encoded_latent = state_dict['latent']
        return self._encoded_latent


class LatentCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.latent_cache = {}

    def cache_latents_all_latents(self: 'AiToolkitDataset'):
        if self.dataset_config.num_frames > 1:
            raise Exception("Error: caching latents is not supported for multi-frame datasets")
        with accelerator.main_process_first():
            print_acc(f"Caching latents for {self.dataset_path}")
            # cache all latents to disk
            to_disk = self.is_caching_latents_to_disk
            to_memory = self.is_caching_latents_to_memory

            if to_disk:
                print_acc(" - Saving latents to disk")
            if to_memory:
                print_acc(" - Keeping latents in memory")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_latents')

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc=f'Caching latents{" to disk" if to_disk else ""}'):
                # set latent space version
                if self.sd.model_config.latent_space_version is not None:
                    file_item.latent_space_version = self.sd.model_config.latent_space_version
                elif self.sd.is_xl:
                    file_item.latent_space_version = 'sdxl'
                elif self.sd.is_v3:
                    file_item.latent_space_version = 'sd3'
                elif self.sd.is_auraflow:
                    file_item.latent_space_version = 'sdxl'
                elif self.sd.is_flux:
                    file_item.latent_space_version = 'flux1'
                elif self.sd.model_config.is_pixart_sigma:
                    file_item.latent_space_version = 'sdxl'
                else:
                    file_item.latent_space_version = self.sd.model_config.arch
                file_item.is_caching_to_disk = to_disk
                file_item.is_caching_to_memory = to_memory
                file_item.latent_load_device = self.sd.device

                latent_path = file_item.get_latent_path(recalculate=True)
                # check if it is saved to disk already
                if os.path.exists(latent_path):
                    if to_memory:
                        # load it into memory
                        state_dict = load_file(latent_path, device='cpu')
                        file_item._encoded_latent = state_dict['latent'].to('cpu', dtype=self.sd.torch_dtype)
                else:
                    # not saved to disk, calculate
                    # load the image first
                    file_item.load_and_process_image(self.transform, only_load_latents=True)
                    dtype = self.sd.torch_dtype
                    device = self.sd.device_torch
                    # add batch dimension
                    try:
                        imgs = file_item.tensor.unsqueeze(0).to(device, dtype=dtype)
                        latent = self.sd.encode_images(imgs).squeeze(0)
                    except Exception as e:
                        print_acc(f"Error processing image: {file_item.path}")
                        print_acc(f"Error: {str(e)}")
                        raise e
                    # save_latent
                    if to_disk:
                        state_dict = OrderedDict([
                            ('latent', latent.clone().detach().cpu()),
                        ])
                        # metadata
                        meta = get_meta_for_safetensors(file_item.get_latent_info_dict())
                        os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                        save_file(state_dict, latent_path, metadata=meta)

                    if to_memory:
                        # keep it in memory
                        file_item._encoded_latent = latent.to('cpu', dtype=self.sd.torch_dtype)

                    del imgs
                    del latent
                    del file_item.tensor

                    # flush(garbage_collect=False)
                file_item.is_latent_cached = True
                i += 1
                # flush every 100
                # if i % 100 == 0:
                #     flush()

            # restore device state
            self.sd.restore_device_state()


class TextEmbeddingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.prompt_embeds: Union[PromptEmbeds, None] = None
        self._text_embedding_path: Union[str, None] = None
        self.is_text_embedding_cached = False
        self.text_embedding_load_device = 'cpu'
        self.text_embedding_space_version = 'sd1'
        self.text_embedding_version = 1

    def get_text_embedding_info_dict(self: 'FileItemDTO'):
        # make sure the caption is loaded here
        # TODO: we need a way to cache all the other features like trigger words, DOP, etc. For now, we need to throw an error if not compatible.
        if self.caption is None:
            self.load_caption()
            # throw error is [trigger] in caption as we cannot inject it while caching
            if '[trigger]' in self.caption:
                raise Exception("Error: [trigger] in caption is not supported when caching text embeddings. Please remove it from the caption.")
        item = OrderedDict([
            ("caption", self.caption),
            ("text_embedding_space_version", self.text_embedding_space_version),
            ("text_embedding_version", self.text_embedding_version),
        ])
        # if we have a control image, cache the path
        if self.encode_control_in_text_embeddings and self.control_path is not None:
            item["control_path"] = self.control_path
        return item

    def get_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._text_embedding_path is not None and not recalculate:
            return self._text_embedding_path
        else:
            # we store text embeddings in a folder in same path as image called _text_embedding_cache
            img_dir = os.path.dirname(self.path)
            te_dir = os.path.join(img_dir, '_t_e_cache')
            hash_dict = self.get_text_embedding_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._text_embedding_path = os.path.join(te_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._text_embedding_path

    def cleanup_text_embedding(self):
        if self.prompt_embeds is not None:
            # we are caching on disk, don't save in memory
            self.prompt_embeds = None

    def load_prompt_embedding(self, device=None):
        if not self.is_text_embedding_cached:
            return
        if self.prompt_embeds is None:
            # load it from disk
            self.prompt_embeds = PromptEmbeds.load(self.get_text_embedding_path())

class TextEmbeddingCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.is_caching_text_embeddings = self.dataset_config.cache_text_embeddings

    def cache_text_embeddings(self: 'AiToolkitDataset'):
        with accelerator.main_process_first():
            print_acc(f"Caching text_embeddings for {self.dataset_path}")
            print_acc(" - Saving text embeddings to disk")
            
            did_move = False

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc='Caching text embeddings to disk'):
                file_item.text_embedding_space_version = self.sd.model_config.arch
                file_item.latent_load_device = self.sd.device

                text_embedding_path = file_item.get_text_embedding_path(recalculate=True)
                # only process if not saved to disk
                if not os.path.exists(text_embedding_path):
                    # load if not loaded
                    if not did_move:
                        self.sd.set_device_state_preset('cache_text_encoder')
                        did_move = True
                        
                    if file_item.encode_control_in_text_embeddings:
                        if file_item.control_path is None:
                            raise Exception(f"Could not find a control image for {file_item.path} which is needed for this model")
                        # load the control image and feed it into the text encoder
                        ctrl_img = Image.open(file_item.control_path).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img = (
                            TF.to_tensor(ctrl_img)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        prompt_embeds: PromptEmbeds = self.sd.encode_prompt(file_item.caption, control_images=ctrl_img)
                    else:
                        prompt_embeds: PromptEmbeds = self.sd.encode_prompt(file_item.caption)
                    # save it
                    prompt_embeds.save(text_embedding_path)
                    del prompt_embeds
                file_item.is_text_embedding_cached = True
                i += 1
            # restore device state
            # if did_move:
            #     self.sd.restore_device_state()


class CLIPCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.clip_vision_num_unconditional_cache = 20
        self.clip_vision_unconditional_cache = []

    def cache_clip_vision_to_disk(self: 'AiToolkitDataset'):
        if not self.is_caching_clip_vision_to_disk:
            return
        with torch.no_grad():
            print_acc(f"Caching clip vision for {self.dataset_path}")

            print_acc(" - Saving clip to disk")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_clip')

            # make sure the adapter has attributes
            if self.sd.adapter is None:
                raise Exception("Error: must have an adapter to cache clip vision to disk")

            clip_image_processor: CLIPImageProcessor = None
            if hasattr(self.sd.adapter, 'clip_image_processor'):
                clip_image_processor = self.sd.adapter.clip_image_processor

            if clip_image_processor is None:
                raise Exception("Error: must have a clip image processor to cache clip vision to disk")

            vision_encoder: CLIPVisionModelWithProjection = None
            if hasattr(self.sd.adapter, 'image_encoder'):
                vision_encoder = self.sd.adapter.image_encoder
            if hasattr(self.sd.adapter, 'vision_encoder'):
                vision_encoder = self.sd.adapter.vision_encoder

            if vision_encoder is None:
                raise Exception("Error: must have a vision encoder to cache clip vision to disk")

            # move vision encoder to device
            vision_encoder.to(self.sd.device)

            is_quad = self.sd.adapter.config.quad_image
            image_encoder_path = self.sd.adapter.config.image_encoder_path

            dtype = self.sd.torch_dtype
            device = self.sd.device_torch
            if hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero:
                # just to do this, we did :)
                # need more samples as it is random noise
                self.clip_vision_num_unconditional_cache = self.clip_vision_num_unconditional_cache
            else:
                # only need one since it doesnt change
                self.clip_vision_num_unconditional_cache = 1

            # cache unconditionals
            print_acc(f" - Caching {self.clip_vision_num_unconditional_cache} unconditional clip vision to disk")
            clip_vision_cache_path = os.path.join(self.dataset_config.clip_image_path, '_clip_vision_cache')

            unconditional_paths = []

            is_noise_zero = hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero

            for i in range(self.clip_vision_num_unconditional_cache):
                hash_dict = OrderedDict([
                    ("image_encoder_path", image_encoder_path),
                    ("is_quad", is_quad),
                    ("is_noise_zero", is_noise_zero),
                ])
                # get base64 hash of md5 checksum of hash_dict
                hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
                hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
                hash_str = hash_str.replace('=', '')

                uncond_path = os.path.join(clip_vision_cache_path, f'uncond_{hash_str}_{i}.safetensors')
                if os.path.exists(uncond_path):
                    # skip it
                    unconditional_paths.append(uncond_path)
                    continue

                # generate a random image
                img_shape = (1, 3, self.sd.adapter.input_size, self.sd.adapter.input_size)
                if is_noise_zero:
                    tensors_0_1 = torch.rand(img_shape).to(device, dtype=torch.float32)
                else:
                    tensors_0_1 = torch.zeros(img_shape).to(device, dtype=torch.float32)
                clip_image = clip_image_processor(
                    images=tensors_0_1,
                    return_tensors="pt",
                    do_resize=True,
                    do_rescale=False,
                ).pixel_values

                if is_quad:
                    # split the 4x4 grid and stack on batch
                    ci1, ci2 = clip_image.chunk(2, dim=2)
                    ci1, ci3 = ci1.chunk(2, dim=3)
                    ci2, ci4 = ci2.chunk(2, dim=3)
                    clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                clip_output = vision_encoder(
                    clip_image.to(device, dtype=dtype),
                    output_hidden_states=True
                )
                # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                state_dict = OrderedDict([
                    ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                    ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                    ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                ])

                os.makedirs(os.path.dirname(uncond_path), exist_ok=True)
                save_file(state_dict, uncond_path)
                unconditional_paths.append(uncond_path)

            self.clip_vision_unconditional_cache = unconditional_paths

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc=f'Caching clip vision to disk'):
                file_item.is_caching_clip_vision_to_disk = True
                file_item.clip_vision_load_device = self.sd.device
                file_item.clip_vision_is_quad = is_quad
                file_item.clip_image_encoder_path = image_encoder_path
                file_item.clip_vision_unconditional_paths = unconditional_paths
                if file_item.has_clip_augmentations:
                    raise Exception("Error: clip vision caching is not supported with clip augmentations")

                embedding_path = file_item.get_clip_vision_embeddings_path(recalculate=True)
                # check if it is saved to disk already
                if not os.path.exists(embedding_path):
                    # load the image first
                    file_item.load_clip_image()
                    # add batch dimension
                    clip_image = file_item.clip_image_tensor.unsqueeze(0).to(device, dtype=dtype)

                    if is_quad:
                        # split the 4x4 grid and stack on batch
                        ci1, ci2 = clip_image.chunk(2, dim=2)
                        ci1, ci3 = ci1.chunk(2, dim=3)
                        ci2, ci4 = ci2.chunk(2, dim=3)
                        clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                    clip_output = vision_encoder(
                        clip_image.to(device, dtype=dtype),
                        output_hidden_states=True
                    )

                    # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                    state_dict = OrderedDict([
                        ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                        ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                        ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                    ])
                    # metadata
                    meta = get_meta_for_safetensors(file_item.get_clip_vision_info_dict())
                    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
                    save_file(state_dict, embedding_path, metadata=meta)

                    del clip_image
                    del clip_output
                    del file_item.clip_image_tensor

                    # flush(garbage_collect=False)
                file_item.is_vision_clip_cached = True
                i += 1
            # flush every 100
            # if i % 100 == 0:
            #     flush()

        # restore device state
        self.sd.restore_device_state()



class ControlCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
            self.control_generator: ControlGenerator = None
    
    def add_control_path_to_file_item(self: 'AiToolkitDataset', file_item: 'FileItemDTO', control_path: str, control_type: ControlTypes):
        if control_type == 'inpaint':
            file_item.inpaint_path = control_path
            file_item.has_inpaint_image = True
        elif control_type == 'mask':
            file_item.mask_path = control_path
            file_item.has_mask_image = True
        else:
            if file_item.control_path is None:
                file_item.control_path = [control_path]
            elif isinstance(file_item.control_path, str):
                file_item.control_path = [file_item.control_path, control_path]
            elif isinstance(file_item.control_path, list):
                file_item.control_path.append(control_path)
            else:
                raise Exception(f"Error: control_path is not a string or list: {file_item.control_path}")
            file_item.has_control_image = True

    def setup_controls(self: 'AiToolkitDataset'):
        if not self.is_generating_controls:
            return
        with torch.no_grad():
            print_acc(f"Generating controls for {self.dataset_path}")
            device = self.sd.device
            
            self.control_generator = ControlGenerator(
                device=device,
                sd=self.sd,
            )

            # use tqdm to show progress
            for file_item in tqdm(self.file_list, desc=f'Generating Controls'):
                for control_type in self.dataset_config.controls:
                    # generates the control if it is not already there
                    control_path = self.control_generator.get_control_path(file_item.path, control_type)
                    if control_path is not None:
                        self.add_control_path_to_file_item(file_item, control_path, control_type)
                
            # remove models
            self.control_generator.cleanup()
            self.control_generator = None
            
            flush()
