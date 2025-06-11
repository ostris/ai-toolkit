import math
from typing import List, Optional, Tuple, Any, Union, TYPE_CHECKING
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import json

if TYPE_CHECKING:
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, **kwargs):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # type: ignore
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            head_dim: int,
            n_kv_heads: int,
            **kwargs,
    ):
        super().__init__()

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.n_kv_heads: int = n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            cache: Optional[Any] = None,
            mask: Optional['BlockDiagonalMask'] = None,
    ) -> torch.Tensor:
        from xformers.ops.fmha import memory_efficient_attention
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.max_seq_len,
                           self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.max_seq_len,
                           self.n_kv_heads, self.head_dim)

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, mask if cache is None else cache.mask)
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)  # type: ignore


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            n_heads: int,
            n_kv_heads: int,
            head_dim: int,
            norm_eps: float,
            **kwargs,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.feed_forward: nn.Module
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            cache: Optional[Any] = None,
            mask: Optional['BlockDiagonalMask'] = None,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


@dataclass
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float = 1e4  # for rope-2D
    image_token_id: int = 10


def precompute_freqs_cis_2d(
        dim: int,
        height: int,
        width: int,
        theta: float,
) -> torch.Tensor:
    """
    freqs_cis: 2D complex tensor of shape (height, width, dim // 2) to be indexed by
        (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)


def position_meshgrid(
        patch_embeds_list: list[torch.Tensor],
) -> torch.Tensor:
    positions = torch.cat(
        [
            torch.stack(
                torch.meshgrid(
                    torch.arange(p.shape[-2]),
                    torch.arange(p.shape[-1]),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            for p in patch_embeds_list
        ]
    )
    return positions


class PixtralVisionEncoder(nn.Module):
    def __init__(
            self,
            hidden_size: int = 1024,
            num_channels: int = 3,
            image_size: int = 1024,
            patch_size: int = 16,
            intermediate_size: int = 4096,
            num_hidden_layers: int = 24,
            num_attention_heads: int = 16,
            rope_theta: float = 1e4,  # for rope-2D
            image_token_id: int = 10,
            **kwargs,
    ):
        super().__init__()
        self.args = VisionEncoderArgs(
            hidden_size=hidden_size,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            rope_theta=rope_theta,
            image_token_id=image_token_id,
        )
        args = self.args
        self.patch_conv = nn.Conv2d(
            in_channels=args.num_channels,
            out_channels=args.hidden_size,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(args.hidden_size, eps=1e-5)
        self.transformer = VisionTransformerBlocks(args)

        head_dim = self.args.hidden_size // self.args.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> 'PixtralVisionEncoder':
        if os.path.isdir(pretrained_model_name_or_path):
            model_folder = pretrained_model_name_or_path
        else:
            model_folder = snapshot_download(pretrained_model_name_or_path)

        # make sure there is a config
        if not os.path.exists(os.path.join(model_folder, "config.json")):
            raise ValueError(f"Could not find config.json in {model_folder}")

        # load config
        with open(os.path.join(model_folder, "config.json"), "r", encoding='utf-8') as f:
            config = json.load(f)

        model = cls(**config)

        # see if there is a state_dict
        if os.path.exists(os.path.join(model_folder, "model.safetensors")):
            state_dict = load_file(os.path.join(
                model_folder, "model.safetensors"))
            model.load_state_dict(state_dict)

        return model

    @property
    def max_patches_per_side(self) -> int:
        return self.args.image_size // self.args.patch_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(
            self,
            images: List[torch.Tensor],
    ) -> torch.Tensor:
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        """
        Args:
            images: list of N_img images of variable sizes, each of shape (C, H, W)

        Returns:
            image_features: tensor of token features for all tokens of all images of
                shape (N_toks, D)
        """
        assert isinstance(
            images, list), f"Expected list of images, got {type(images)}"
        assert all(len(img.shape) == 3 for img in
                   images), f"Expected images with shape (C, H, W), got {[img.shape for img in images]}"
        # pass images through initial convolution independently
        patch_embeds_list = [self.patch_conv(
            img.unsqueeze(0)).squeeze(0) for img in images]

        # flatten to a single sequence
        patch_embeds = torch.cat([p.flatten(1).permute(1, 0)
                                 for p in patch_embeds_list], dim=0)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        positions = position_meshgrid(patch_embeds_list).to(self.device)
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

        # pass through Transformer with a block diagonal mask delimiting images
        mask = BlockDiagonalMask.from_seqlens(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
        )
        out = self.transformer(patch_embeds, mask=mask, freqs_cis=freqs_cis)

        # remove batch dimension of the single sequence
        return out  # type: ignore[no-any-return]


class VisionLanguageAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_in = nn.Linear(
            in_dim,
            out_dim,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # type: ignore[no-any-return]
        return self.w_out(self.gelu(self.w_in(x)))


class VisionTransformerBlocks(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(
                TransformerBlock(
                    dim=args.hidden_size,
                    hidden_dim=args.intermediate_size,
                    n_heads=args.num_attention_heads,
                    n_kv_heads=args.num_attention_heads,
                    head_dim=args.hidden_size // args.num_attention_heads,
                    norm_eps=1e-5,
                )
            )

    def forward(
            self,
            x: torch.Tensor,
            mask: 'BlockDiagonalMask',
            freqs_cis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=freqs_cis)
        return x


DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]  # RGB
DATASET_STD = [0.26862954, 0.26130258, 0.27577711]  # RGB


def normalize(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor image with mean and standard deviation.

    Args:
    image (torch.Tensor): Image to be normalized, shape (C, H, W), values in [0, 1].
    mean (torch.Tensor): Mean for each channel.
    std (torch.Tensor): Standard deviation for each channel.

    Returns:
    torch.Tensor: Normalized image with shape (C, H, W).
    """
    assert image.shape[0] == len(mean) == len(
        std), f"{image.shape=}, {mean.shape=}, {std.shape=}"

    # Reshape mean and std to (C, 1, 1) for broadcasting
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    return (image - mean) / std


def transform_image(image: torch.Tensor, new_size: tuple[int, int]) -> torch.Tensor:
    """
    Resize and normalize the input image.

    Args:
    image (torch.Tensor): Input image tensor of shape (C, H, W), values in [0, 1].
    new_size (tuple[int, int]): Target size (height, width) for resizing.

    Returns:
    torch.Tensor: Resized and normalized image tensor of shape (C, new_H, new_W).
    """
    # Resize the image
    resized_image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=new_size,
        mode='bicubic',
        align_corners=False
    ).squeeze(0)

    # Normalize the image
    normalized_image = normalize(
        resized_image,
        torch.tensor(DATASET_MEAN, device=image.device, dtype=image.dtype),
        torch.tensor(DATASET_STD, device=image.device, dtype=image.dtype)
    )

    return normalized_image


class PixtralVisionImagePreprocessor:
    def __init__(self, image_patch_size=16, max_image_size=1024) -> None:
        self.image_patch_size = image_patch_size
        self.max_image_size = max_image_size
        self.image_token = 10

    def _image_to_num_tokens(self, img: torch.Tensor, max_image_size = None) -> Tuple[int, int]:
        w: Union[int, float]
        h: Union[int, float]
        
        if max_image_size is None:
            max_image_size = self.max_image_size

        w, h = img.shape[-1], img.shape[-2]

        # originally, pixtral used the largest of the 2 dimensions, but we
        # will use the base size of the image based on number of pixels.
        # ratio = max(h / self.max_image_size, w / self.max_image_size)  # original
        
        base_size = int(math.sqrt(w * h))
        ratio = base_size / max_image_size
        if ratio > 1:
            w = round(w / ratio)
            h = round(h / ratio)

        width_tokens = (w - 1) // self.image_patch_size + 1
        height_tokens = (h - 1) // self.image_patch_size + 1

        return width_tokens, height_tokens

    def __call__(self, image: torch.Tensor, max_image_size=None) -> torch.Tensor:
        """
        Converts ImageChunks to numpy image arrays and image token ids

        Args:
        image torch tensor with values 0-1 and shape of (C, H, W)

        Returns:
        processed_image: tensor of token features for all tokens of all images of
        """
        # should not have batch
        if len(image.shape) == 4:
            raise ValueError(
                f"Expected image with shape (C, H, W), got {image.shape}")

        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(
                f"image tensor values must be between 0 and 1. Got min: {image.min()}, max: {image.max()}")
        
        if max_image_size is None:
            max_image_size = self.max_image_size

        w, h = self._image_to_num_tokens(image, max_image_size=max_image_size)
        assert w > 0
        assert h > 0

        new_image_size = (
            w * self.image_patch_size,
            h * self.image_patch_size,
        )

        processed_image = transform_image(image, new_image_size)

        return processed_image


class PixtralVisionImagePreprocessorCompatibleReturn:
    def __init__(self, pixel_values) -> None:
        self.pixel_values = pixel_values


# Compatable version with ai toolkit flow
class PixtralVisionImagePreprocessorCompatible(PixtralVisionImagePreprocessor):
    def __init__(self, image_patch_size=16, max_image_size=1024) -> None:
        super().__init__(
            image_patch_size=image_patch_size,
            max_image_size=max_image_size
        )
        self.size = {
            'height': max_image_size,
            'width': max_image_size
        }
        self.max_image_size = max_image_size
        self.image_mean = DATASET_MEAN
        self.image_std = DATASET_STD

    def __call__(
            self,
        images,
        return_tensors="pt",
        do_resize=True,
        do_rescale=False,
        max_image_size=None,
    ) -> torch.Tensor:
        if max_image_size is None:
            max_image_size = self.max_image_size
        out_stack = []
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        for i in range(images.shape[0]):
            image = images[i]
            processed_image = super().__call__(image, max_image_size=max_image_size)
            out_stack.append(processed_image)

        output = torch.stack(out_stack, dim=0)
        return PixtralVisionImagePreprocessorCompatibleReturn(output)


class PixtralVisionEncoderCompatibleReturn:
    def __init__(self, hidden_states) -> None:
        self.hidden_states = hidden_states


class PixtralVisionEncoderCompatibleConfig:
    def __init__(self):
        self.image_size = 1024
        self.hidden_size = 1024
        self.patch_size = 16


class PixtralVisionEncoderCompatible(PixtralVisionEncoder):
    def __init__(
            self,
            hidden_size: int = 1024,
            num_channels: int = 3,
            image_size: int = 1024,
            patch_size: int = 16,
            intermediate_size: int = 4096,
            num_hidden_layers: int = 24,
            num_attention_heads: int = 16,
            rope_theta: float = 1e4,  # for rope-2D
            image_token_id: int = 10,
            **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            rope_theta=rope_theta,
            image_token_id=image_token_id,
        )
        self.config = PixtralVisionEncoderCompatibleConfig()

    def forward(
            self,
            images,
            output_hidden_states=True,
    ) -> torch.Tensor:
        out_stack = []
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        for i in range(images.shape[0]):
            image = images[i]
            # must be in an array
            image_output = super().forward([image])
            out_stack.append(image_output)

        output = torch.stack(out_stack, dim=0)
        return PixtralVisionEncoderCompatibleReturn([output])
