from typing import Literal, Type, TYPE_CHECKING, Union

import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_avif
import imageio.v3 as iio

Step: Type = Literal['caption', 'caption_short', 'create_mask', 'contrast_stretch']

img_manipulation_steps = ['contrast_stretch']

img_ext = ['.jpg', '.jpeg', '.png', '.webp', '.avif']

if TYPE_CHECKING:
    from .llava_utils import LLaVAImageProcessor
    from .fuyu_utils import FuyuImageProcessor

ImageProcessor = Union['LLaVAImageProcessor', 'FuyuImageProcessor']


def pil_to_cv2(image):
    """Convert a PIL image to a cv2 image."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image):
    """Convert a cv2 image to a PIL image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def load_image(img_path: str, force_rgb: bool = False):
    img_array = iio.imread(img_path)
    if img_array.ndim == 2:
        image = Image.fromarray(img_array, mode='L')
    elif img_array.ndim == 3 or img_array.ndim == 4:
        if img_array.ndim == 4:
            # When the image has a frame dimension, only the first frame is taken.
            img_array = img_array[0]
        height, width, channels = img_array.shape
        if channels == 3:  # RGB
            image = Image.fromarray(img_array, mode='RGB')
        elif channels == 4:  # RGBA
            image = Image.fromarray(img_array, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")

    if force_rgb and image.mode != 'RGB':
        image = image.convert('RGB')
    
    try:
        # transpose with exif data
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        print(f"Error rotating {img_path}: {e}")
    return image


def resize_to_max(image, max_width=1024, max_height=1024):
    width, height = image.size
    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    width = int(width * scale)
    height = int(height * scale)

    return image.resize((width, height), Image.LANCZOS)
