from typing import Literal, Type, TYPE_CHECKING, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

Step: Type = Literal['caption', 'caption_short', 'create_mask', 'contrast_stretch']

img_manipulation_steps = ['contrast_stretch']

img_ext = ['.jpg', '.jpeg', '.png', '.webp']

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


def load_image(img_path: str):
    image = Image.open(img_path).convert('RGB')
    try:
        # transpose with exif data
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        pass
    return image


def resize_to_max(image, max_width=1024, max_height=1024):
    width, height = image.size
    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    width = int(width * scale)
    height = int(height * scale)

    return image.resize((width, height), Image.LANCZOS)
