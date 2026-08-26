import base64
import glob
import hashlib
import itertools
import json
import math
import os
import random
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Dict, Union
import traceback

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor

from toolkit.audio.preserve_pitch import time_stretch_preserve_pitch
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
video_ext_list = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv']


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

def waveform_to_stereo(waveform):
    c = waveform.shape[0]
    if c == 2:
        return waveform
    if c == 1:
        return waveform.expand(2, -1)
    if c == 6:  # 5.1: FL, FR, FC, LFE, BL, BR
        fl, fr, fc, _, bl, br = waveform
        k = 0.7071
        return torch.stack([fl + k * fc + k * bl, fr + k * fc + k * br])
    if c == 8:  # 7.1: FL, FR, FC, LFE, BL, BR, SL, SR
        fl, fr, fc, _, bl, br, sl, sr = waveform
        k = 0.7071
        return torch.stack([fl + k * fc + k * (bl + sl), fr + k * fc + k * (br + sr)])
    return waveform.mean(0, keepdim=True).expand(2, -1)


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
                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path_with_ext):
            with open(default_prompt_path_with_ext, 'r', encoding='utf-8') as f:
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
                # if the bucket has fewer items left than the requested batch size,
                # duplicate items from this batch to pad it up to batch_size
                if len(batch) < self.batch_size and len(batch) > 0:
                    pad = [batch[i % len(batch)] for i in range(self.batch_size - len(batch))]
                    batch = batch + pad
                self.batch_indices.append(batch)

    def shuffle_buckets(self: 'AiToolkitDataset'):
        for key, bucket in self.buckets.items():
            random.shuffle(bucket.file_list_idx)

    def setup_buckets(self: 'AiToolkitDataset', quiet=False):
        if not hasattr(self, 'file_list'):
            raise Exception(f'file_list not found on class instance {self.__class__.__name__}')
        if not hasattr(self, 'dataset_config'):
            raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')

        if self.epoch_num > 0:
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
            if self.is_audio_model:
                bucket_key = f"{file_item.width}ms"
                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = Bucket(file_item.width, 1)
                self.buckets[bucket_key].file_list_idx.append(idx)
                continue
            width = int(file_item.width * file_item.dataset_config.scale)
            height = int(file_item.height * file_item.dataset_config.scale)

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
            else:
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
            if self.is_video:
                # images (1 frame) and videos must not mix in a batch
                bucket_key += f'x{file_item.num_frames}f'
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
            # caption with the trigger word replaced by the diff output preservation class
            self.caption_dop: str = None
            # D-OPSD teacher caption (trigger word replaced by <Picture 1>/<Video 1>)
            self.caption_dopsd: str = None

            dataset_config: DatasetConfig = kwargs.get('dataset_config', None)
            self.extra_values: List[float] = dataset_config.extra_values
            self.trigger_word = dataset_config.trigger_word

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
                    prompt = clean_caption(prompt)
                    if short_caption is not None:
                        short_caption = clean_caption(short_caption)
                    
                    if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                        prompt = self.dataset_config.default_caption
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
        if self.dataset_config.diff_output_preservation:
            # replace this dataset's trigger word with the preservation class.
            # do it on the final caption so token order matches the normal caption
            self.caption_dop = self.caption
            if self.trigger_word is not None:
                self.caption_dop = self.caption.replace(
                    self.trigger_word, self.dataset_config.diff_output_preservation_class
                )
        if getattr(self, 'dopsd_self_ref', False):
            # trigger word -> the self-reference token, or the token prepended
            # when there is no trigger word
            if self.trigger_word is not None:
                self.caption_dopsd = self.caption.replace(
                    self.trigger_word, self.get_dopsd_ref_token()
                )
            else:
                self.caption_dopsd = f"{self.get_dopsd_ref_token()} {self.caption}".strip()

    def get_dopsd_ref_token(self: 'FileItemDTO') -> str:
        # the item is always the only reference in D-OPSD mode
        return "<Video 1>" if self.is_video else "<Picture 1>"

    def get_caption(
            self: 'FileItemDTO',
            trigger=None,
            to_replace_list=None,
            add_if_not_present=False,
            short_caption=False
    ):
        if trigger is None and self.trigger_word is not None:
            trigger = self.trigger_word
        
        if trigger is not None and not self.is_reg:
            # add if not present if not regularization
            add_if_not_present = True
            
        if short_caption:
            raw_caption = self.raw_caption_short
        else:
            raw_caption = self.raw_caption
        if raw_caption is None:
            raw_caption = ''
        # handle dropout
        if self.dataset_config.caption_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
            # get a random float form 0 to 1
            rand = random.random()
            if rand < self.dataset_config.caption_dropout_rate:
                # drop the caption
                return ''

        # get tokens
        token_list = raw_caption.split(',')

        # handle token dropout
        if self.dataset_config.token_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
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
        caption = inject_trigger_into_prompt(caption, trigger, to_replace_list, add_if_not_present)

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
            random.shuffle(token_list)
            caption = ', '.join(token_list)
        if caption == '':
            pass
        return caption

class AudioProcessingDTOMixin:
    def load_and_process_audio(self: 'FileItemDTO'):
        # Default to "no audio" unless we successfully extract it
        self.audio_data = None
        self.audio_tensor = None
        self.tensor = None
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(self.path)  # [channels, samples]
            waveform = waveform_to_stereo(waveform)  # Convert to stereo if not already
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            self.tensor = waveform
            self.audio_tensor = waveform
            self.audio_data = {"waveform": waveform, "sample_rate": int(self.sample_rate)}

        except Exception as e:
            # if issue with libtorchcodec "Could not load libtorchcodec"
            raise Exception(f"** WARNING ** - Error Processing audio for {self.path}. Error: {e}")
        

class ImageProcessingDTOMixin:
    def get_auto_frame_count(self: 'FileItemDTO', total_frames: int, video_fps: float) -> int:
        # frame count this video will train at with auto_frame_count. Also called at
        # FileItemDTO init so bucket keys carry the real frame count, so it must give
        # the same answer at bucketing time and at load time.
        # allow for any length video here but make sure it is temporally compressable.
        vid_length_seconds = total_frames / video_fps

        desired_num_frames = int(vid_length_seconds * self.dataset_config.fps)

        if getattr(self, 'frame_count_snapper', None) is not None:
            # model-specific valid-frame-count grid (e.g. minimax_h3's 17n+5)
            desired_num_frames = self.frame_count_snapper(desired_num_frames)
        else:
            # make sure it is divisible by temporal_compression
            if self.dataset_config.trim_auto_frame_count_tail:
                # snap to the largest valid count that fits inside the video (after the
                # key frame +1 below) so trim mode never overshoots the source, which
                # would freeze the last frame and pad the audio tail with silence
                desired_num_frames = max(0, desired_num_frames - 1) // self.temporal_compression * self.temporal_compression
            else:
                desired_num_frames = desired_num_frames // self.temporal_compression * self.temporal_compression

            # TODO, all models currently add a key frame, but future models may not, update here if this changes.
            desired_num_frames += 1  # add one for the key frame that is always added

        return desired_num_frames

    def load_and_process_video(
        self: 'FileItemDTO',
        transform: Union[None, transforms.Compose],
        only_load_latents=False
    ):
        
        if self.augments is not None and len(self.augments) > 0:
            raise Exception('Augments not supported for videos')
            
        if self.has_augmentations:
            raise Exception('Augmentations not supported for videos')
        
        if not self.dataset_config.buckets:
            raise Exception('Buckets required for video processing')
        
        do_audio = self.dataset_config.do_audio
        
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
            
            if self.dataset_config.auto_frame_count:
                self.num_frames = self.get_auto_frame_count(total_frames, video_fps)


            # Always stretch/shrink to the requested number of frames if needed
            if self.dataset_config.auto_frame_count and self.dataset_config.trim_auto_frame_count_tail:
                # preserve real time: pull frames at the dataset fps from the start of the
                # video and trim the tail that didn't fit the snapped frame count, instead of
                # shrinking the whole video to fit (which speeds up motion / chipmunks audio).
                # Critical for audio models (e.g. minimax_h3) where audio must stay in sync.
                fps_ratio = video_fps / self.dataset_config.fps if video_fps and video_fps > 0 else 1.0
                frames_to_extract = [min(round(i * fps_ratio), max_frame_index) for i in range(self.num_frames)]
            elif self.dataset_config.shrink_video_to_frames or total_frames < self.num_frames:
                # Distribute frames evenly across the entire video
                interval = max_frame_index / (self.num_frames - 1) if self.num_frames > 1 else 0
                frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(self.num_frames)]
            else:
                # Calculate frame interval based on FPS ratio
                fps_ratio = video_fps / self.dataset_config.fps
                frame_interval = max(1, int(round(fps_ratio)))
                
                # Calculate max consecutive frames we can extract at desired FPS
                max_consecutive_frames = (total_frames // frame_interval)
                
                if max_consecutive_frames < self.num_frames:
                    # Not enough frames at desired FPS, so stretch instead
                    interval = max_frame_index / (self.num_frames - 1) if self.num_frames > 1 else 0
                    frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(self.num_frames)]
                else:
                    # Calculate max start frame to ensure we can get all num_frames
                    max_start_frame = max_frame_index - ((self.num_frames - 1) * frame_interval)
                    start_frame = random.randint(0, max(0, max_start_frame))
                    
                    # Generate list of frames to extract
                    frames_to_extract = [start_frame + (i * frame_interval) for i in range(self.num_frames)]
                    
            # Final safety check - ensure no frame exceeds max valid index
            frames_to_extract = [min(frame_idx, max_frame_index) for frame_idx in frames_to_extract]
            
            # Only log frames to extract if in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"  Frames to extract: {frames_to_extract}")
            
            # Extract frames -- decode sequentially in a single pass. A cap.set() seek per
            # frame forces a keyframe seek + GOP re-decode for every extracted frame
            # (~20x slower); frames_to_extract is always ascending, so seek once to the
            # first frame then grab() through the gaps.
            frames = []
            unique_frame_idxs = sorted(set(frames_to_extract))
            processed_frames = {}  # frame_idx -> processed frame (duplicates reuse it)

            def process_frame(rgb_frame):
                # Convert to PIL Image
                img = Image.fromarray(rgb_frame)

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

                return img

            decode_with_pyav = False

            # Set frame position
            pos = unique_frame_idxs[0]
            if pos > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

                # Silently verify position was set correctly (no warnings unless debug mode)
                if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if actual_pos != pos:
                        print_acc(f"Warning: Failed to set exact frame position. Requested: {pos}, Actual: {actual_pos}")

            for frame_idx in unique_frame_idxs:
                # skip past frames between targets without decoding them to images
                while pos < frame_idx and cap.grab():
                    pos += 1

                ret, frame = cap.read()
                if ret:
                    pos += 1
                else:
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
                            # resync sequential position after the fallback seek
                            pos = fallback_pos + 1
                            break
                    else:
                        # No fallback worked. cv2's bundled ffmpeg cannot decode some codecs
                        # at all (e.g. AV1 has no software decoder there), so retry the
                        # remaining frames with PyAV below.
                        decode_with_pyav = True
                        break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                processed_frames[frame_idx] = process_frame(frame)

            if decode_with_pyav:
                # cv2 could not decode this video (e.g. AV1: OpenCV's bundled ffmpeg has no
                # software AV1 decoder). Decode the still-missing frames in one sequential
                # PyAV pass; PyAV ships libdav1d so it handles codecs cv2 cannot.
                import av

                needed = {i for i in unique_frame_idxs if i not in processed_frames}
                last_av_frame = None
                with av.open(self.path) as container:
                    decoded_idx = -1
                    for av_frame in container.decode(video=0):
                        decoded_idx += 1
                        last_av_frame = av_frame
                        if decoded_idx in needed:
                            processed_frames[decoded_idx] = process_frame(av_frame.to_ndarray(format='rgb24'))
                            needed.discard(decoded_idx)
                            if not needed:
                                break

                if needed:
                    if last_av_frame is None:
                        video_info = f"Video: {self.path}, Total frames: {total_frames}, FPS: {video_fps}"
                        raise Exception(f"Failed to read frames {sorted(needed)} from video with both cv2 and PyAV. {video_info}")
                    # metadata frame count overshot the real stream; reuse the last decoded frame
                    tail_frame = process_frame(last_av_frame.to_ndarray(format='rgb24'))
                    for frame_idx in needed:
                        processed_frames[frame_idx] = tail_frame

            # assemble in extraction order; stretched clips repeat decoded frames
            frames = [processed_frames[frame_idx] for frame_idx in frames_to_extract]

            # Release the video capture
            cap.release()
            
            # Stack frames into tensor [frames, channels, height, width]
            self.tensor = torch.stack(frames)

            # ------------------------------
            # Audio extraction + stretching
            # ------------------------------
            if do_audio:
                # Default to "no audio" unless we successfully extract it
                self.audio_data = None
                self.audio_tensor = None

                try:
                    import torchaudio
                    import torch.nn.functional as F

                    # Compute the time range of the selected frames in the *source* video
                    # Include the last frame by extending to the next frame boundary.
                    if video_fps and video_fps > 0 and len(frames_to_extract) > 0:
                        clip_start_frame = int(frames_to_extract[0])
                        clip_end_frame = int(frames_to_extract[-1])
                        clip_start_time = clip_start_frame / float(video_fps)
                        clip_end_time = (clip_end_frame + 1) / float(video_fps)
                        source_duration = max(0.0, clip_end_time - clip_start_time)
                    else:
                        clip_start_time = 0.0
                        clip_end_time = 0.0
                        source_duration = 0.0

                    # Target duration is how this sampled/stretched clip is interpreted for training
                    # (i.e. num_frames at the configured dataset FPS).
                    if hasattr(self.dataset_config, "fps") and self.dataset_config.fps and self.dataset_config.fps > 0:
                        target_duration = float(self.num_frames) / float(self.dataset_config.fps)
                    else:
                        target_duration = source_duration

                    # torchcodec's AudioDecoder raises when a video has no audio
                    # track, so probe for a stream before decoding.
                    import av
                    with av.open(self.path) as container:
                        has_audio_stream = len(container.streams.audio) > 0

                    waveform = None
                    if has_audio_stream:
                        waveform, sample_rate = torchaudio.load(self.path)  # [channels, samples]

                        waveform = waveform_to_stereo(waveform)  # Convert to stereo if not already

                        if self.dataset_config.audio_normalize:
                            peak = waveform.abs().amax()  # global peak across channels
                            eps = 1e-9
                            target_peak = 0.999  # ~ -0.01 dBFS
                            gain = target_peak / (peak + eps)
                            waveform = waveform * gain

                        trim_tail_audio = (
                            self.dataset_config.auto_frame_count
                            and self.dataset_config.trim_auto_frame_count_tail
                        )

                        # Slice to the selected clip region (when we have a meaningful time range)
                        if source_duration > 0.0:
                            start_sample = int(round(clip_start_time * sample_rate))
                            if trim_tail_audio and target_duration > 0.0:
                                # time must stay 1:1 with the video — cut exactly the
                                # training duration so no stretch is needed below
                                end_sample = start_sample + round(target_duration * sample_rate)
                            else:
                                end_sample = round(clip_end_time * sample_rate)
                            start_sample = max(0, min(start_sample, waveform.shape[-1]))
                            end_sample = max(0, min(end_sample, waveform.shape[-1]))
                            if end_sample > start_sample:
                                waveform = waveform[..., start_sample:end_sample]
                            else:
                                # No valid audio segment
                                waveform = None
                        else:
                            # If we can't compute a meaningful time range, treat as no-audio
                            waveform = None

                    if waveform is not None and waveform.numel() > 0:
                        target_samples = round(target_duration * sample_rate)
                        if target_samples > 0 and waveform.shape[-1] != target_samples:
                            # Time-stretch/shrink to match the video clip duration implied by dataset FPS.
                            if trim_tail_audio:
                                # never stretch/contract in trim mode. The waveform can only be
                                # short here (audio/video ended a hair before the target) —
                                # pad the tail with silence, or cut any rounding overshoot
                                pad = target_samples - waveform.shape[-1]
                                if pad > 0:
                                    waveform = F.pad(waveform, (0, pad))
                                else:
                                    waveform = waveform[..., :target_samples]
                            elif self.dataset_config.audio_preserve_pitch:
                                waveform = time_stretch_preserve_pitch(waveform, sample_rate, target_samples)  # waveform is [C, L]
                            else:
                                # Use linear interpolation over the time axis.
                                wf = waveform.unsqueeze(0)  # [1, C, L]
                                wf = F.interpolate(wf, size=target_samples, mode="linear", align_corners=False)
                                waveform = wf.squeeze(0)  # [C, L]

                        self.audio_tensor = waveform
                        self.audio_data = {"waveform": waveform, "sample_rate": int(sample_rate)}

                except Exception as e:
                    # if issue with libtorchcodec "Could not load libtorchcodec"
                    raise Exception(f"** WARNING ** - Error Processing audio for {self.path}. Error: {e}")
            
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
        # handle get_prompt_embedding
        if self.is_text_embedding_cached:
            self.load_prompt_embedding()
        # if we are caching latents, just do that
        if self.is_latent_cached:
            self.get_latent()
            # if load_image_when_caching_latents is set, we still need the raw image
            # tensor in addition to the cached latent, so fall through to load it below
            if not self.dataset_config.load_image_when_caching_latents:
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
        if self.is_audio_model:
            self.load_and_process_audio()
            return
        if self.is_video:
            self.load_and_process_video(transform, only_load_latents)
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
            found_control_videos = []
            allow_video_controls = sd is not None and getattr(
                sd, 'supports_video_control_images', False)
            for control_path in control_path_list:
                for ext in img_ext_list:
                    if os.path.exists(os.path.join(control_path, file_name_no_ext + ext)):
                        found_control_images.append(os.path.join(control_path, file_name_no_ext + ext))
                        self.has_control_image = True
                        break
                else:
                    if allow_video_controls:
                        for ext in video_ext_list:
                            if os.path.exists(os.path.join(control_path, file_name_no_ext + ext)):
                                found_control_videos.append(os.path.join(control_path, file_name_no_ext + ext))
                                self.has_control_image = True
                                break
            # control VIDEO paths ride on the item; the model encodes and
            # disk-caches them on first use (see minimax_h3 ref2va)
            self.control_video_paths = found_control_videos or None
            self.control_path = found_control_images
            if len(self.control_path) == 0:
                self.control_path = None
            elif len(self.control_path) == 1:
                # only do one
                self.control_path = self.control_path[0]

        if dataset_config.control_from_same_folder:
            # assume we have them. We will pull them on load.
            self.full_size_control_images = dataset_config.full_size_control_images
            self.has_control_image = True

    def get_new_control_paths(self: 'FileItemDTO'):
        if self.dataset_config.control_from_same_folder:
            # randomly grab image paths from the same folder as if they came from control_path
            pool_folder = os.path.dirname(self.path)
            # find all images in the folder
            img_files = []
            for ext in img_ext_list:
                img_files += glob.glob(os.path.join(pool_folder, f'*{ext}'))
            # remove the current image if len is greater than 1
            if len(img_files) > 1:
                img_files.remove(self.path)
            num_controls = min(self.dataset_config.num_controls_from_same_folder, len(img_files))
            # randomly grab them
            return random.sample(img_files, num_controls)
        else:
            return self.control_path

    def load_control_image(self: 'FileItemDTO'):
        control_tensors = []
        control_path_list = self.get_new_control_paths()
        if not isinstance(control_path_list, list):
            control_path_list = [control_path_list]
        # video-only controls leave control_path as None (their latents come
        # from the ref-video cache, not this image loader)
        control_path_list = [p for p in control_path_list if p is not None]
        
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

class ArgBreakMixin:
    # just stops super calls form hitting object
    def __init__(self, *args, **kwargs):
        pass


def _latent_to_uint8(latent: torch.Tensor) -> torch.Tensor:
    # pixel-space latents in [-1, 1] -> uint8 0..255 for compact caching
    return ((latent.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def _latent_from_uint8(latent: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # uint8 0..255 -> pixel-space latents in [-1, 1]
    return (latent.to(torch.float32) / 127.5 - 1.0).to(dtype)


def _waveform_to_int16(waveform: torch.Tensor) -> torch.Tensor:
    # audio waveform in [-1, 1] -> int16 for compact caching. 8 bits is too coarse for audio.
    return (waveform.float().clamp(-1, 1) * 32767.0).round().to(torch.int16)


def _waveform_from_int16(waveform: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # int16 -> audio waveform in [-1, 1]
    return (waveform.to(torch.float32) / 32767.0).to(dtype)


class LatentCachingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self._encoded_latent: Union[torch.Tensor, None] = None
        self._cached_first_frame_latent: Union[torch.Tensor, None] = None
        self._cached_audio_latent: Union[torch.Tensor, None] = None
        self._cached_tensor_uint8: Union[torch.Tensor, None] = None
        self._cached_waveform_int16: Union[torch.Tensor, None] = None
        self._cached_waveform_sample_rate: Union[int, None] = None
        self._latent_path: Union[str, None] = None
        self.is_latent_cached = False
        self.is_caching_to_disk = False
        self.is_caching_to_memory = False
        self.latent_load_device = 'cpu'
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
        is_video = False
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        if self.is_video and self.dataset_config.auto_frame_count:
            # don't store num frames here as it is calculated dynamically
            item["auto_frame_count"] = True
            if self.dataset_config.trim_auto_frame_count_tail:
                # changes frame selection; only added when on so caches made before
                # this option existed stay valid when it is off
                item["trim_auto_frame_count_tail"] = True
            is_video = True
        elif self.is_video and self.dataset_config.num_frames > 1:
            item["num_frames"] = self.dataset_config.num_frames
            is_video = True
        if is_video and self.dataset_config.fps != 24:
            # only add fps if it deviates from the default
            item["fps"] = self.dataset_config.fps
        if is_video and self.dataset_config.do_i2v:
                item["do_i2v"] = True
        if is_video and self.dataset_config.do_audio:
            item["do_audio"] = True
            if self.dataset_config.audio_normalize:
                item["audio_normalize"] = True
            if self.dataset_config.audio_preserve_pitch:
                item["audio_preserve_pitch"] = True
        if self.is_audio_model:
            item["is_audio_model"] = True
            item["sample_rate"] = self.sample_rate
        if self.dataset_config.cache_tensors_to_disk:
            # tensor is stored in the cache file, invalidate caches made without it
            item["cache_tensors_to_disk"] = True
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
                self._cached_first_frame_latent = None
                self._cached_audio_latent = None
                self._cached_tensor_uint8 = None
                self._cached_waveform_int16 = None
                self._cached_waveform_sample_rate = None
            else:
                # move it back to cpu
                self._encoded_latent = self._encoded_latent.to('cpu')
                if self._cached_first_frame_latent is not None:
                    self._cached_first_frame_latent = self._cached_first_frame_latent.to('cpu')
                if self._cached_audio_latent is not None:
                    self._cached_audio_latent = self._cached_audio_latent.to('cpu')

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
            if self._encoded_latent.dtype == torch.uint8:
                # pixel-space latents cached as uint8
                self._encoded_latent = _latent_from_uint8(self._encoded_latent)
            if 'first_frame_latent' in state_dict:
                self._cached_first_frame_latent = state_dict['first_frame_latent']
                if self._cached_first_frame_latent.dtype == torch.uint8:
                    self._cached_first_frame_latent = _latent_from_uint8(self._cached_first_frame_latent)
            if 'audio_latent' in state_dict:
                self._cached_audio_latent = state_dict['audio_latent']
            if 'num_frames' in state_dict:
                self.num_frames = int(state_dict['num_frames'].item())
            if 'tensor' in state_dict:
                self._cached_tensor_uint8 = state_dict['tensor']
            if 'waveform' in state_dict:
                self._cached_waveform_int16 = state_dict['waveform']
                self._cached_waveform_sample_rate = int(state_dict['waveform_sample_rate'].item())
        if self._cached_tensor_uint8 is not None and getattr(self, 'tensor', None) is None:
            # rebuild the pixel tensor as it would be if loaded without caching
            self.tensor = _latent_from_uint8(self._cached_tensor_uint8)
        if self._cached_waveform_int16 is not None and self.audio_data is None:
            # rebuild the audio waveform as it would be if loaded without caching
            waveform = _waveform_from_int16(self._cached_waveform_int16)
            self.audio_tensor = waveform
            self.audio_data = {"waveform": waveform, "sample_rate": self._cached_waveform_sample_rate}
            if self.is_audio_model:
                # audio-only models use the waveform as the main tensor
                self.tensor = waveform
        return self._encoded_latent


class LatentCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.latent_cache = {}

    def cache_latents_all_latents(self: 'AiToolkitDataset'):
        with accelerator.main_process_first():
            print_acc(f"Caching latents for {self.dataset_path}")
            # cache all latents to disk
            to_disk = self.is_caching_latents_to_disk
            to_memory = self.is_caching_latents_to_memory

            if to_disk:
                print_acc(" - Saving latents to disk")
            if to_memory:
                print_acc(" - Keeping latents in memory")
            # move sd items to cpu except for vae. Only done on the first item that
            # actually needs encoding so fully cached datasets don't shuffle models around
            did_move = False

            # prep (video decode, frame extraction, audio load, disk reads) is done by a
            # thread pool so the next items are ready while the current one is encoding.
            # the in-flight window is bounded so decoded videos don't pile up in RAM.
            num_workers = max(1, self.dataset_config.cache_latents_num_workers)

            def _prep(prep_item: 'FileItemDTO'):
                prep_item.is_caching_to_disk = to_disk
                prep_item.is_caching_to_memory = to_memory
                prep_item.latent_load_device = self.sd.device

                prep_latent_path = prep_item.get_latent_path(recalculate=True)
                try:
                    if os.path.exists(prep_latent_path):
                        cached_state_dict = load_file(prep_latent_path, device='cpu') if to_memory else None
                        return prep_item, prep_latent_path, cached_state_dict, False
                    # not saved to disk, load the image/video/audio
                    prep_item.load_and_process_image(self.transform, only_load_latents=True)
                except Exception as e:
                    print_acc(f"Error processing image: {prep_item.path}")
                    print_acc(f"Error: {str(e)}")
                    raise e
                return prep_item, prep_latent_path, None, True

            # use tqdm to show progress
            i = 0
            pbar = tqdm(total=len(self.file_list), desc=f'Caching latents{" to disk" if to_disk else ""}')
            executor = ThreadPoolExecutor(max_workers=num_workers)
            try:
                pending = deque()
                file_iter = iter(self.file_list)
                for queued_item in itertools.islice(file_iter, num_workers + 2):
                    pending.append(executor.submit(_prep, queued_item))
                while pending:
                    file_item, latent_path, cached_state_dict, needs_encode = pending.popleft().result()
                    # keep the window full
                    next_item = next(file_iter, None)
                    if next_item is not None:
                        pending.append(executor.submit(_prep, next_item))
                    if needs_encode and not did_move:
                        self.sd.set_device_state_preset('cache_latents')
                        did_move = True
                    self._cache_one_latent(file_item, latent_path, cached_state_dict, needs_encode, to_disk, to_memory)
                    file_item.is_latent_cached = True
                    i += 1
                    pbar.update(1)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
                pbar.close()

            # restore device state
            if did_move:
                self.sd.restore_device_state()

    def _cache_one_latent(
            self: 'AiToolkitDataset',
            file_item: 'FileItemDTO',
            latent_path: str,
            cached_state_dict,
            needs_encode: bool,
            to_disk: bool,
            to_memory: bool,
    ):
        # check if it is saved to disk already
        if not needs_encode:
            if to_memory:
                # load it into memory
                state_dict = cached_state_dict
                cached_latent = state_dict['latent']
                if cached_latent.dtype == torch.uint8:
                    # pixel-space latents cached as uint8
                    cached_latent = _latent_from_uint8(cached_latent)
                file_item._encoded_latent = cached_latent.to('cpu', dtype=self.sd.torch_dtype)
                if 'first_frame_latent' in state_dict:
                    cached_first_frame = state_dict['first_frame_latent']
                    if cached_first_frame.dtype == torch.uint8:
                        cached_first_frame = _latent_from_uint8(cached_first_frame)
                    file_item._cached_first_frame_latent = cached_first_frame.to('cpu', dtype=self.sd.torch_dtype)
                if 'audio_latent' in state_dict:
                    file_item._cached_audio_latent = state_dict['audio_latent'].to('cpu', dtype=self.sd.torch_dtype)
                if 'tensor' in state_dict:
                    file_item._cached_tensor_uint8 = state_dict['tensor']
                if 'waveform' in state_dict:
                    file_item._cached_waveform_int16 = state_dict['waveform']
                    file_item._cached_waveform_sample_rate = int(state_dict['waveform_sample_rate'].item())
        else:
            # not saved to disk, calculate
            # the image/video/audio was already loaded by the prep thread
            dtype = self.sd.torch_dtype
            device = self.sd.device_torch
            state_dict = OrderedDict()
            first_frame_latent = None
            audio_latent = None
            frames = None
            # add batch dimension
            cache_uint8 = getattr(self.sd, 'cache_latents_as_uint8', False)
            if self.dataset_config.cache_tensors_to_disk:
                if not self.is_audio_model:
                    tensor_uint8 = _latent_to_uint8(file_item.tensor).cpu()
                    if to_disk:
                        state_dict['tensor'] = tensor_uint8
                    if to_memory:
                        file_item._cached_tensor_uint8 = tensor_uint8
                if file_item.audio_data is not None:
                    # audio-only models: tensor IS the waveform, stored here as int16 instead of uint8
                    waveform_int16 = _waveform_to_int16(file_item.audio_data['waveform']).cpu()
                    sample_rate = int(file_item.audio_data['sample_rate'])
                    if to_disk:
                        state_dict['waveform'] = waveform_int16
                        state_dict['waveform_sample_rate'] = torch.tensor(sample_rate, dtype=torch.int32)
                    if to_memory:
                        file_item._cached_waveform_int16 = waveform_int16
                        file_item._cached_waveform_sample_rate = sample_rate
            try:
                imgs = file_item.tensor.unsqueeze(0).to(device, dtype=dtype)
                latent = self.sd.encode_images(imgs).squeeze(0)
                if to_disk:
                    if cache_uint8:
                        state_dict['latent'] = _latent_to_uint8(latent).cpu()
                    else:
                        state_dict['latent'] = latent.clone().detach().cpu()
            except Exception as e:
                print_acc(f"Error processing image: {file_item.path}")
                print_acc(f"Error: {str(e)}")
                raise e
            # do first frame
            is_video = file_item.is_video
            if is_video and self.dataset_config.do_i2v:
                frames = file_item.tensor.unsqueeze(0).to(device, dtype=dtype)
                if len(frames.shape) == 4:
                    first_frames = frames
                elif len(frames.shape) == 5:
                    first_frames = frames[:, 0]
                else:
                    raise ValueError(f"Unknown frame shape {frames.shape}")
                first_frame_latent = self.sd.encode_images(first_frames).squeeze(0)
                if to_disk:
                    if cache_uint8:
                        state_dict['first_frame_latent'] = _latent_to_uint8(first_frame_latent).cpu()
                    else:
                        state_dict['first_frame_latent'] = first_frame_latent.clone().detach().cpu()

            # audio (video+audio models only — audio-only models already encoded above via encode_images)
            if not self.is_audio_model and file_item.audio_data is not None:
                audio_latent = self.sd.encode_audio([file_item.audio_data]).squeeze(0)
                if to_disk:
                    state_dict['audio_latent'] = audio_latent.clone().detach().cpu()

            if is_video:
                state_dict['num_frames'] = torch.tensor(file_item.num_frames, dtype=torch.int32)

            # save_latent
            if to_disk:
                # metadata
                meta = get_meta_for_safetensors(file_item.get_latent_info_dict())
                os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                save_file(state_dict, latent_path, metadata=meta)

            if to_memory:
                # keep it in memory
                file_item._encoded_latent = latent.to('cpu', dtype=self.sd.torch_dtype)
                if first_frame_latent is not None:
                    file_item._cached_first_frame_latent = first_frame_latent.to('cpu', dtype=self.sd.torch_dtype)
                if audio_latent is not None:
                    file_item._cached_audio_latent = audio_latent.to('cpu', dtype=self.sd.torch_dtype)

            del imgs
            del latent
            del frames
            del file_item.tensor
            del state_dict
            del first_frame_latent
            del audio_latent
            file_item.cleanup()


class TextEmbeddingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.prompt_embeds: Union[PromptEmbeds, None] = None
        self._text_embedding_path: Union[str, None] = None
        # diff output preservation embeds (caption with trigger word replaced by class)
        self.dop_prompt_embeds: Union[PromptEmbeds, None] = None
        self._dop_text_embedding_path: Union[str, None] = None
        # blank caption embeds used for caption dropout when caching text embeddings
        self._blank_text_embedding_path: Union[str, None] = None
        # DOP embeds for dropout steps (dropout caption with trigger replaced by class)
        self._dop_blank_text_embedding_path: Union[str, None] = None
        # D-OPSD teacher embeds (caption with trigger replaced by the self-reference
        # token, encoded WITH the item's own image/video as the vision reference)
        self.dopsd_prompt_embeds: Union[PromptEmbeds, None] = None
        self._dopsd_text_embedding_path: Union[str, None] = None
        self._dopsd_blank_text_embedding_path: Union[str, None] = None
        self._loaded_text_embedding_path: Union[str, None] = None
        self._caption_was_dropped = False
        self.is_text_embedding_cached = False
        self.text_embedding_load_device = 'cpu'
        self.text_embedding_version = 1

    def get_text_embedding_info_dict(self: 'FileItemDTO', caption_override=None, text_only=False, dopsd_self_ref=False):
        # make sure the caption is loaded here
        # TODO: we need a way to cache all the other features like trigger words, DOP, etc. For now, we need to throw an error if not compatible.
        if self.caption is None:
            self.load_caption()
        item = OrderedDict([
            ("caption", self.caption if caption_override is None else caption_override),
            ("text_embedding_space_version", self.text_embedding_space_version),
            ("text_embedding_version", self.text_embedding_version),
        ])
        if dopsd_self_ref:
            # teacher embeds carry the item's own media as the vision reference
            item["dopsd_self_ref"] = True
            return item
        # dropout embeds are encoded as plain text, keep control conditioning
        # out of their cache key
        if text_only:
            return item
        # if we have a control image, cache the path
        if self.encode_control_in_text_embeddings and self.control_path is not None:
            item["control_path"] = self.control_path
        if self.encode_control_in_text_embeddings and getattr(self, 'control_video_paths', None):
            item["control_videos"] = sorted(self.control_video_paths)
            # v2: reference-video vision blocks are no longer resampled by the
            # processor (do_sample_frames=False); older video-ref embeds are
            # misaligned with their presentation. Only items WITH control
            # videos carry this key, so no other cache is touched.
            item["control_videos_version"] = 2
        # first-frame vision conditioning changes the embedding content -> new cache key
        elif (
            getattr(self, "encode_first_frame_in_text_embeddings", False)
            and self.dataset_config.do_i2v
            and self.is_video
        ):
            item["first_frame_in_te"] = True
        return item

    def _build_text_embedding_path(self: 'FileItemDTO', caption_override=None, text_only=False, dopsd_self_ref=False):
        # we store text embeddings in a folder in same path as image called _text_embedding_cache
        img_dir = os.path.dirname(self.path)
        te_dir = os.path.join(img_dir, '_t_e_cache')
        hash_dict = self.get_text_embedding_info_dict(caption_override=caption_override, text_only=text_only, dopsd_self_ref=dopsd_self_ref)
        filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
        # get base64 hash of md5 checksum of hash_dict
        hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
        hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
        hash_str = hash_str.replace('=', '')
        return os.path.join(te_dir, f'{filename_no_ext}_{hash_str}.safetensors')

    def get_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._text_embedding_path is not None and not recalculate:
            return self._text_embedding_path
        else:
            self._text_embedding_path = self._build_text_embedding_path()

        return self._text_embedding_path

    def get_dop_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._dop_text_embedding_path is not None and not recalculate:
            return self._dop_text_embedding_path
        else:
            # make sure the caption is loaded so caption_dop is built
            if self.caption is None:
                self.load_caption()
            # if the trigger word is not in the caption, this hashes to the same
            # path as the normal embedding and the cache file is shared
            self._dop_text_embedding_path = self._build_text_embedding_path(
                caption_override=self.caption_dop
            )

        return self._dop_text_embedding_path

    def get_dropout_caption(self: 'FileItemDTO'):
        # when encoding live, dropped captions still get the trigger word injected
        # downstream (add_if_not_present when not a reg image), so match that here
        if self.trigger_word is not None and not self.is_reg:
            return inject_trigger_into_prompt('', trigger=self.trigger_word, add_if_not_present=True)
        return ''

    def get_dop_dropout_caption(self: 'FileItemDTO'):
        # live encoding replaces the trigger word with the preservation class on the
        # dropped caption (class only), so the cached DOP dropout caption must match
        dropout_caption = self.get_dropout_caption()
        if self.trigger_word is not None:
            return dropout_caption.replace(
                self.trigger_word, self.dataset_config.diff_output_preservation_class
            )
        return dropout_caption

    def get_dop_blank_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._dop_blank_text_embedding_path is not None and not recalculate:
            return self._dop_blank_text_embedding_path
        else:
            # if the DOP dropout caption matches the dropout caption, this hashes to
            # the same path as the blank embedding and the cache file is shared.
            # text_only: dropout embeds carry no control conditioning
            self._dop_blank_text_embedding_path = self._build_text_embedding_path(
                caption_override=self.get_dop_dropout_caption(), text_only=True
            )

        return self._dop_blank_text_embedding_path

    def get_dopsd_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._dopsd_text_embedding_path is not None and not recalculate:
            return self._dopsd_text_embedding_path
        # make sure the caption is loaded so caption_dopsd is built
        if self.caption is None:
            self.load_caption()
        self._dopsd_text_embedding_path = self._build_text_embedding_path(
            caption_override=self.caption_dopsd, dopsd_self_ref=True
        )
        return self._dopsd_text_embedding_path

    def get_dopsd_dropout_caption(self: 'FileItemDTO'):
        # dropout caption with the trigger word swapped for the self-reference token
        dropout_caption = self.get_dropout_caption()
        if self.trigger_word is not None:
            return dropout_caption.replace(
                self.trigger_word, self.get_dopsd_ref_token()
            )
        return f"{self.get_dopsd_ref_token()} {dropout_caption}".strip()

    def get_dopsd_blank_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._dopsd_blank_text_embedding_path is not None and not recalculate:
            return self._dopsd_blank_text_embedding_path
        # unlike DOP, dropout embeds keep the vision reference
        self._dopsd_blank_text_embedding_path = self._build_text_embedding_path(
            caption_override=self.get_dopsd_dropout_caption(), dopsd_self_ref=True
        )
        return self._dopsd_blank_text_embedding_path

    def get_blank_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._blank_text_embedding_path is not None and not recalculate:
            return self._blank_text_embedding_path
        else:
            # if the dropout caption matches the normal caption (and the item has no
            # control conditioning), this hashes to the same path as the normal
            # embedding and the cache file is shared.
            # text_only: dropout embeds carry no control conditioning
            self._blank_text_embedding_path = self._build_text_embedding_path(
                caption_override=self.get_dropout_caption(), text_only=True
            )

        return self._blank_text_embedding_path

    def cleanup_text_embedding(self):
        if self.prompt_embeds is not None:
            # we are caching on disk, don't save in memory
            self.prompt_embeds = None
        if self.dop_prompt_embeds is not None:
            self.dop_prompt_embeds = None
        if self.dopsd_prompt_embeds is not None:
            self.dopsd_prompt_embeds = None

    def load_prompt_embedding(self, device=None):
        if not self.is_text_embedding_cached:
            return
        if self.prompt_embeds is None:
            te_path = self.get_text_embedding_path()
            self._caption_was_dropped = False
            if self.dataset_config.caption_dropout_rate > 0:
                # get a random float form 0 to 1
                rand = random.random()
                if rand < self.dataset_config.caption_dropout_rate:
                    # drop the caption by using the cached blank embedding
                    te_path = self.get_blank_text_embedding_path()
                    self._caption_was_dropped = True
            # load it from disk
            self.prompt_embeds = PromptEmbeds.load(te_path)
            self._loaded_text_embedding_path = te_path
        if self.dataset_config.diff_output_preservation and self.dop_prompt_embeds is None:
            if self._caption_was_dropped:
                # match live encoding, which builds the DOP caption from the
                # dropped caption (trigger word replaced with the class)
                dop_path = self.get_dop_blank_text_embedding_path()
            else:
                dop_path = self.get_dop_text_embedding_path()
            if dop_path == self._loaded_text_embedding_path:
                # no trigger word in caption, same embedding
                self.dop_prompt_embeds = self.prompt_embeds
            else:
                self.dop_prompt_embeds = PromptEmbeds.load(dop_path)
        if getattr(self, 'dopsd_self_ref', False) and self.dopsd_prompt_embeds is None:
            if self._caption_was_dropped:
                dopsd_path = self.get_dopsd_blank_text_embedding_path()
            else:
                dopsd_path = self.get_dopsd_text_embedding_path()
            self.dopsd_prompt_embeds = PromptEmbeds.load(dopsd_path)

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
                file_item.latent_load_device = self.sd.device

                text_embedding_path = file_item.get_text_embedding_path(recalculate=True)
                # (path, caption) pairs to encode for this item
                encode_targets = [(text_embedding_path, file_item.caption)]
                if self.dataset_config.diff_output_preservation:
                    dop_path = file_item.get_dop_text_embedding_path(recalculate=True)
                    if dop_path != text_embedding_path:
                        # trigger word was in the caption, cache the DOP version too
                        encode_targets.append((dop_path, file_item.caption_dop))
                # dropout embeds are encoded as plain text (no control images)
                dropout_target_paths = set()
                if self.dataset_config.caption_dropout_rate > 0:
                    blank_path = file_item.get_blank_text_embedding_path(recalculate=True)
                    if blank_path != text_embedding_path:
                        # cache the dropout caption embedding (blank, or trigger word only)
                        encode_targets.append((blank_path, file_item.get_dropout_caption()))
                        dropout_target_paths.add(blank_path)
                    if self.dataset_config.diff_output_preservation:
                        # cache the DOP version of the dropout caption (class only)
                        dop_blank_path = file_item.get_dop_blank_text_embedding_path(recalculate=True)
                        if dop_blank_path not in [t[0] for t in encode_targets] + [text_embedding_path]:
                            encode_targets.append((dop_blank_path, file_item.get_dop_dropout_caption()))
                            dropout_target_paths.add(dop_blank_path)
                # only process if not saved to disk
                encode_targets = [t for t in encode_targets if not os.path.exists(t[0])]
                if len(encode_targets) > 0:
                    # load if not loaded
                    if not did_move:
                        self.sd.set_device_state_preset('cache_text_encoder')
                        did_move = True

                    control_video_paths = getattr(file_item, 'control_video_paths', None) or []
                    if file_item.encode_control_in_text_embeddings and (
                        file_item.control_path is not None or len(control_video_paths) > 0
                    ):
                        ctrl_img_list = []
                        control_path_list = file_item.control_path
                        if control_path_list is None:
                            control_path_list = []
                        elif not isinstance(control_path_list, list):
                            control_path_list = [control_path_list]
                        for i in range(len(control_path_list)):
                            try:
                                img = Image.open(control_path_list[i]).convert("RGB")
                                img = exif_transpose(img)
                                # convert to 0 to 1 tensor
                                img = (
                                    TF.to_tensor(img)
                                    .unsqueeze(0)
                                    .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                                )
                                ctrl_img_list.append(img)
                            except Exception as e:
                                print_acc(f"Error: {e}")
                                print_acc(f"Error loading control image: {control_path_list[i]}")
                        # control VIDEOS ride into the presentation by path (models
                        # with supports_video_control_images turn them into
                        # timestamped vision blocks); images first, then videos.
                        # The model needs the dataset config to treat the clip
                        # exactly like its latent rows (frame count / trim)
                        ctrl_img_list.extend(control_video_paths)
                        if len(control_video_paths) > 0:
                            self.sd._ref_video_dataset_config = self.dataset_config
                        
                        if len(ctrl_img_list) == 0:
                            ctrl_img = None
                        elif not self.sd.has_multiple_control_images:
                            ctrl_img = ctrl_img_list[0]
                        else:
                            ctrl_img = ctrl_img_list
                        for path, caption in encode_targets:
                            if path in dropout_target_paths:
                                # dropout embeds are plain text. Only fall back to the
                                # control images if the model cannot encode without them
                                try:
                                    prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption)
                                except Exception:
                                    prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption, control_images=ctrl_img)
                            else:
                                prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption, control_images=ctrl_img)
                            prompt_embeds.save(path)
                            del prompt_embeds
                    elif (
                        getattr(self.sd, 'encode_first_frame_in_text_embeddings', False)
                        and self.dataset_config.do_i2v
                        and file_item.is_video
                    ):
                        # video item: encode the clip's FIRST FRAME into the text embeddings
                        # as a vision reference, matching sampling (where the ctrl image goes
                        # into the embeds and is held as the clean first frames)
                        file_item.load_and_process_image(self.transform, only_load_latents=True)
                        frames = file_item.tensor  # (T, C, H, W) or (C, H, W), in [-1, 1]
                        first = frames[0] if frames.dim() == 4 else frames
                        ctrl_img = (
                            ((first + 1.0) / 2.0)
                            .clamp(0, 1)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        if self.sd.has_multiple_control_images:
                            ctrl_img = [ctrl_img]
                        for path, caption in encode_targets:
                            if path in dropout_target_paths:
                                # dropout embeds are plain text. Only fall back to the
                                # control images if the model cannot encode without them
                                try:
                                    prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption)
                                except Exception:
                                    prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption, control_images=ctrl_img)
                            else:
                                prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption, control_images=ctrl_img)
                            prompt_embeds.save(path)
                            del prompt_embeds
                        file_item.tensor = None
                    else:
                        for path, caption in encode_targets:
                            prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption)
                            prompt_embeds.save(path)
                            del prompt_embeds
                if getattr(file_item, 'dopsd_self_ref', False):
                    # D-OPSD teacher embeds encode with the item's own media as the reference
                    control_video_paths = getattr(file_item, 'control_video_paths', None) or []
                    if file_item.control_path is not None or len(control_video_paths) > 0:
                        raise ValueError(
                            "D-OPSD self-reference training cannot be combined with "
                            "control images/videos: the item itself must be the only "
                            f"reference. Offending item: {file_item.path}"
                        )
                    dopsd_targets = [(
                        file_item.get_dopsd_text_embedding_path(recalculate=True),
                        file_item.caption_dopsd,
                    )]
                    if self.dataset_config.caption_dropout_rate > 0:
                        dopsd_blank_path = file_item.get_dopsd_blank_text_embedding_path(recalculate=True)
                        if dopsd_blank_path != dopsd_targets[0][0]:
                            dopsd_targets.append((dopsd_blank_path, file_item.get_dopsd_dropout_caption()))
                    dopsd_targets = [t for t in dopsd_targets if not os.path.exists(t[0])]
                    if len(dopsd_targets) > 0:
                        if not did_move:
                            self.sd.set_device_state_preset('cache_text_encoder')
                            did_move = True
                        if file_item.is_video:
                            # own path rides through the video-ref presentation
                            ctrl_img = [file_item.path]
                            self.sd._ref_video_dataset_config = self.dataset_config
                        else:
                            # own bucketed pixels as the reference image
                            file_item.load_and_process_image(self.transform, only_load_latents=True)
                            img = file_item.tensor  # (C, H, W) in [-1, 1]
                            ctrl_img = [
                                ((img + 1.0) / 2.0)
                                .clamp(0, 1)
                                .unsqueeze(0)
                                .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                            ]
                            # release the cached latent/pixels the load pulled in
                            file_item.cleanup()
                        if not self.sd.has_multiple_control_images:
                            ctrl_img = ctrl_img[0]
                        for path, caption in dopsd_targets:
                            prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption, control_images=ctrl_img)
                            prompt_embeds.save(path)
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
        elif control_type == 'mask' or control_type == 'sapiens2_mask':
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
