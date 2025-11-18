import math
from typing import Callable, Union

import torch
from einops import rearrange
from PIL import Image
from torch import Tensor

from .model import Flux2
import torchvision


def compress_time(t_ids: Tensor) -> Tensor:
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(
        len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype
    )
    t_ids_compressed = t_remap[t_ids]
    return t_ids_compressed


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    t_coords = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
        t_coords.append(torch.unique(t_ids, sorted=True))
    return x_list


def encode_image_refs(
    ae,
    img_ctx: Union[list[Image.Image], list[torch.Tensor]],
    scale=10,
    limit_pixels=1024**2,
):
    if not img_ctx:
        return None, None

    img_ctx_prep = default_prep(img=img_ctx, limit_pixels=limit_pixels)
    if not isinstance(img_ctx_prep, list):
        img_ctx_prep = [img_ctx_prep]

    # Encode each reference image
    encoded_refs = []
    for img in img_ctx_prep:
        if img.ndim == 3:
            img = img.unsqueeze(0)
        encoded = ae.encode(img.to(ae.device, ae.dtype))[0]
        encoded_refs.append(encoded)

    # Create time offsets for each reference
    t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
    t_off = [t.view(-1) for t in t_off]

    # Process with position IDs
    ref_tokens, ref_ids = listed_prc_img(encoded_refs, t_coord=t_off)

    # Concatenate all references along sequence dimension
    ref_tokens = torch.cat(ref_tokens, dim=0)  # (total_ref_tokens, C)
    ref_ids = torch.cat(ref_ids, dim=0)  # (total_ref_tokens, 4)

    # Add batch dimension
    ref_tokens = ref_tokens.unsqueeze(0)  # (1, total_ref_tokens, C)
    ref_ids = ref_ids.unsqueeze(0)  # (1, total_ref_tokens, 4)

    return ref_tokens.to(torch.bfloat16), ref_ids


def prc_txt(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txts"

    _l, _ = x.shape  # noqa: F841

    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),  # dummy dimension
        "w": torch.arange(1),  # dummy dimension
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids.to(x.device)


def batched_wrapper(fn):
    def batched_prc(
        x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                    l_coord[i] if l_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return torch.stack(x), torch.stack(x_ids)

    return batched_prc


def listed_wrapper(fn):
    def listed_prc(
        x: list[Tensor],
        t_coord: list[Tensor] | None = None,
        l_coord: list[Tensor] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                    l_coord[i] if l_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return list(x), list(x_ids)

    return listed_prc


def prc_img(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    c, h, w = x.shape  # noqa: F841
    x_coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1) if l_coord is None else l_coord,
    }
    x_ids = torch.cartesian_prod(
        x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"]
    )
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids.to(x.device)


listed_prc_img = listed_wrapper(prc_img)
batched_prc_img = batched_wrapper(prc_img)
batched_prc_txt = batched_wrapper(prc_txt)


def center_crop_to_multiple_of_x(
    img: Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor], x: int
) -> Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor]:
    if isinstance(img, list):
        return [center_crop_to_multiple_of_x(_img, x) for _img in img]  # type: ignore

    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2], img.shape[-1]
    else:
        w, h = img.size
    new_w = (w // x) * x
    new_h = (h // x) * x

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    if isinstance(img, torch.Tensor):
        return img[..., top:bottom, left:right]
    resized = img.crop((left, top, right, bottom))
    return resized


def cap_pixels(
    img: Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor], k
):
    if isinstance(img, list):
        return [cap_pixels(_img, k) for _img in img]
    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2], img.shape[-1]
    else:
        w, h = img.size
    pixel_count = w * h

    if pixel_count <= k:
        return img

    # Scaling factor to reduce total pixels below K
    scale = math.sqrt(k / pixel_count)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if isinstance(img, torch.Tensor):
        did_expand = False
        if img.ndim == 3:
            img = img.unsqueeze(0)
            did_expand = True
        img = torch.nn.functional.interpolate(
            img,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        )
        if did_expand:
            img = img.squeeze(0)
        return img
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def cap_min_pixels(
    img: Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor],
    max_ar=8,
    min_sidelength=64,
):
    if isinstance(img, list):
        return [
            cap_min_pixels(_img, max_ar=max_ar, min_sidelength=min_sidelength)
            for _img in img
        ]
    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2], img.shape[-1]
    else:
        w, h = img.size
    if w < min_sidelength or h < min_sidelength:
        raise ValueError(
            f"Skipping due to minimal sidelength underschritten h {h} w {w}"
        )
    if w / h > max_ar or h / w > max_ar:
        raise ValueError(f"Skipping due to maximal ar overschritten h {h} w {w}")
    return img


def to_rgb(
    img: Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor],
) -> Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor]:
    if isinstance(img, list):
        return [
            to_rgb(
                _img,
            )
            for _img in img
        ]
    if isinstance(img, torch.Tensor):
        return img  # assume already in tensor format
    return img.convert("RGB")


def default_images_prep(
    x: Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor],
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(x, list):
        return [default_images_prep(e) for e in x]  # type: ignore
    if isinstance(x, torch.Tensor):
        return x  # assume already in tensor format
    x_tensor = torchvision.transforms.ToTensor()(x)
    return 2 * x_tensor - 1


def default_prep(
    img: Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor],
    limit_pixels: int,
    ensure_multiple: int = 16,
) -> torch.Tensor | list[torch.Tensor]:
    # if passing a tensor, assume it is -1 to 1 already
    img_rgb = to_rgb(img)
    img_min = cap_min_pixels(img_rgb)  # type: ignore
    img_cap = cap_pixels(img_min, limit_pixels)  # type: ignore
    img_crop = center_crop_to_multiple_of_x(img_cap, ensure_multiple)  # type: ignore
    img_tensor = default_images_prep(img_crop)
    return img_tensor


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux2,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None, (
                "You need to provide either both or neither of the sequence conditioning"
            )
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img
