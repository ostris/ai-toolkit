"""The VL condition image helper must give the Qwen2-VL processor the same pixels as a plain PIL image.

Runs on CPU without any model weights: python testing/test_vl_images.py
"""
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from toolkit.util.vl_images import tensor_to_vl_pil


def main():
    proc = Qwen2VLImageProcessor(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, (448, 640, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    t = torch.from_numpy(arr).permute(2, 0, 1).float()[None] / 255  # 0-1, as ai-toolkit loads control images

    reference = proc(images=[pil.resize((640, 448), Image.LANCZOS)], return_tensors="pt")["pixel_values"]
    fixed = proc(images=[tensor_to_vl_pil(t, 448, 640)], return_tensors="pt")["pixel_values"]
    raw_tensor = proc(images=[t], return_tensors="pt")["pixel_values"]  # what the old code path did

    assert torch.allclose(fixed, reference), "helper output differs from the plain PIL path"
    assert (raw_tensor - reference).abs().max() > 1.0, "expected the raw 0-1 tensor path to be wrong"
    print(f"ok: fixed vs reference max|d| {(fixed - reference).abs().max():.1e}, "
          f"raw tensor mean {raw_tensor.mean():.2f} vs reference {reference.mean():.2f}")


if __name__ == "__main__":
    main()
