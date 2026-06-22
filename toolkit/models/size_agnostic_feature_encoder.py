import os
from typing import Union, Optional

import torch
import torch.nn as nn
from transformers.image_processing_utils import BaseImageProcessor


class SAFEReducerBlock(nn.Module):
    """
    This is the block that reduces the size of an vactor w and h be half. It is designed to be iterative
    So it is run multiple times to reduce an image to a desired dimension while carrying a shrinking residual
    along for the ride. This is done to preserve information.
    """
    def __init__(self, channels=512):
        super(SAFEReducerBlock, self).__init__()
        self.channels = channels

        activation = nn.GELU

        self.reducer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            activation(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            activation(),
            nn.BatchNorm2d(channels),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.residual_shrink = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        res = self.residual_shrink(x)
        reduced = self.reducer(x)
        return reduced + res


class SizeAgnosticFeatureEncoder(nn.Module):
    def __init__(
            self,
            in_channels=3,
            num_tokens=8,
            num_vectors=768,
            reducer_channels=512,
            channels=2048,
            downscale_factor: int = 8,
    ):
        super(SizeAgnosticFeatureEncoder, self).__init__()
        self.num_tokens = num_tokens
        self.num_vectors = num_vectors
        self.channels = channels
        self.reducer_channels = reducer_channels
        self.gradient_checkpointing = False

        # input is minimum of (bs, 3, 256, 256)

        subpixel_channels = in_channels * downscale_factor ** 2

        # PixelUnshuffle(8 = # (bs, 3, 32, 32) -> (bs, 192, 32, 32)
        # PixelUnshuffle(16 = # (bs, 3, 16, 16) -> (bs, 48, 16, 16)

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)  # (bs, 3, 256, 256) -> (bs, 192, 32, 32)

        self.conv_in = nn.Conv2d(subpixel_channels, reducer_channels, kernel_size=3, padding=1)  # (bs, 192, 32, 32) -> (bs, 512, 32, 32)

        # run as many times as needed to get to min feature of 8 on the smallest dimension
        self.reducer = SAFEReducerBlock(reducer_channels)  # (bs, 512, 32, 32) -> (bs, 512, 8, 8)

        self.reduced_out = nn.Conv2d(
            reducer_channels, self.channels, kernel_size=3, padding=1
        )  # (bs, 512, 8, 8) -> (bs, 2048, 8, 8)

        # (bs, 2048, 8, 8)
        self.block1 = SAFEReducerBlock(self.channels)  # (bs, 2048, 8, 8) -> (bs, 2048, 4, 4)
        self.block2 = SAFEReducerBlock(self.channels)  # (bs, 2048, 8, 8) -> (bs, 2048, 2, 2)

        # reduce mean of dims 2 and 3
        self.adaptive_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # (bs, 2048)
        # linear layer to (bs, self.num_vectors * self.num_tokens)
        self.fc1 = nn.Linear(self.channels, self.num_vectors * self.num_tokens)

        # (bs, self.num_vectors * self.num_tokens) = (bs, 8 * 768) = (bs, 6144)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv_in(x)

        while True:
            # reduce until we get as close to 8x8 as possible without going under
            x = self.reducer(x)
            if x.shape[2] // 2 < 8 or x.shape[3] // 2 < 8:
                break

        x = self.reduced_out(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        x = self.fc1(x)

        # reshape
        x = x.view(-1, self.num_tokens, self.num_vectors)

        return x


class SAFEIPReturn:
    def __init__(self, pixel_values):
        self.pixel_values = pixel_values


class SAFEImageProcessor(BaseImageProcessor):
    def __init__(
            self,
            max_size=1024,
            min_size=256,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_size = max_size
        self.min_size = min_size

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            **kwargs,
    ):
        # not needed
        return cls(**kwargs)

    def __call__(
            self,
            images,
            **kwargs
    ):
        # TODO allow for random resizing
        # comes in 0 - 1 range
        # if any size is smaller than 256, resize to 256
        # if any size is larger than max_size, resize to max_size
        if images.min() < -0.3 or images.max() > 1.3:
            raise ValueError(
                "images fed into SAFEImageProcessor values must be between 0 and 1. Got min: {}, max: {}".format(
                    images.min(), images.max()
                ))

        # make sure we have (bs, 3, h, w)
        while len(images.shape) < 4:
            images = images.unsqueeze(0)

        # expand to 3 channels if we only have 1 channel
        if images.shape[1] == 1:
            images = torch.cat([images, images, images], dim=1)

        width = images.shape[3]
        height = images.shape[2]

        if width < self.min_size or height < self.min_size:
            # scale up so that the smallest size is 256
            if width < height:
                new_width = self.min_size
                new_height = int(height * (self.min_size / width))
            else:
                new_height = self.min_size
                new_width = int(width * (self.min_size / height))
            images = nn.functional.interpolate(images, size=(new_height, new_width), mode='bilinear',
                                               align_corners=False)

        elif width > self.max_size or height > self.max_size:
            # scale down so that the largest size is max_size but do not shrink the other size below 256
            if width > height:
                new_width = self.max_size
                new_height = int(height * (self.max_size / width))
            else:
                new_height = self.max_size
                new_width = int(width * (self.max_size / height))

            if new_width < self.min_size:
                new_width = self.min_size
                new_height = int(height * (self.min_size / width))

            if new_height < self.min_size:
                new_height = self.min_size
                new_width = int(width * (self.min_size / height))

            images = nn.functional.interpolate(images, size=(new_height, new_width), mode='bilinear',
                                               align_corners=False)

        # if wither side is not divisible by 16, mirror pad to make it so
        if images.shape[2] % 16 != 0:
            pad = 16 - (images.shape[2] % 16)
            pad1 = pad // 2
            pad2 = pad - pad1
            images = nn.functional.pad(images, (0, 0, pad1, pad2), mode='reflect')
        if images.shape[3] % 16 != 0:
            pad = 16 - (images.shape[3] % 16)
            pad1 = pad // 2
            pad2 = pad - pad1
            images = nn.functional.pad(images, (pad1, pad2, 0, 0), mode='reflect')

        return SAFEIPReturn(images)


class SAFEVMConfig:
    def __init__(
            self,
            in_channels=3,
            num_tokens=8,
            num_vectors=768,
            reducer_channels=512,
            channels=2048,
            downscale_factor: int = 8,
            **kwargs
    ):
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_vectors = num_vectors
        self.reducer_channels = reducer_channels
        self.channels = channels
        self.downscale_factor = downscale_factor
        self.image_size = 224

        self.hidden_size = num_vectors
        self.projection_dim = num_vectors


class SAFEVMReturn:
    def __init__(self, output):
        self.output = output
        # todo actually do hidden states. This is just for code compatability for now
        self.hidden_states = [output for _ in range(13)]


class SAFEVisionModel(SizeAgnosticFeatureEncoder):
    def __init__(self, **kwargs):
        self.config = SAFEVMConfig(**kwargs)
        self.image_size = None
        # super().__init__(**kwargs)
        super(SAFEVisionModel, self).__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # not needed
        return SAFEVisionModel(**kwargs)

    def forward(self, x, **kwargs):
        return SAFEVMReturn(super().forward(x))
