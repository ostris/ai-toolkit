from typing import Literal, Type

import cv2
import numpy as np
from PIL import Image, ImageOps

Step: Type = Literal['caption', 'caption_short', 'create_mask', 'contrast_stretch']

img_manipulation_steps = ['contrast_stretch']


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

