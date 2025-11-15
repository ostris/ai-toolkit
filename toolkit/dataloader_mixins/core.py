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



class ArgBreakMixin:
    # just stops super calls form hitting object
    def __init__(self, *args, **kwargs):
        pass


