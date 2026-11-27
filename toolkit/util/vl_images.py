import torch
from PIL import Image


def tensor_to_vl_pil(img: torch.Tensor, height: int, width: int) -> Image.Image:
    """Convert a 0-1 float image tensor (ch, h, w) or (1, ch, h, w) into a uint8 PIL image resized to (height, width).

    Qwen2-VL style image processors rescale their input by 1/255. They expect uint8 (0-255) images, so a 0-1 float
    tensor gets divided by 255 a second time and the vision encoder sees an almost black image. Handing the
    processor a PIL image, as the diffusers pipelines do, avoids the double rescale. Lanczos matches the resize
    the diffusers edit pipelines apply to their condition images.
    """
    if img.dim() == 4:
        img = img[0]
    arr = (img.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr).resize((int(width), int(height)), Image.LANCZOS)
