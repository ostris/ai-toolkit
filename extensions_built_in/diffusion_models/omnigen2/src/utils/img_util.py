from typing import List

from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image

def resize_image(image, max_pixels, img_scale_num):
    width, height = image.size
    cur_pixels = height * width
    ratio = (max_pixels / cur_pixels) ** 0.5
    ratio = min(ratio, 1.0) # do not upscale input image

    new_height, new_width = int(height * ratio) // img_scale_num * img_scale_num, int(width * ratio) // img_scale_num * img_scale_num

    image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    return image

def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)