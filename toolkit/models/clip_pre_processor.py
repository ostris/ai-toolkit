import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GELU()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_up(x)
        x = self.conv_out(x)
        return x


class CLIPImagePreProcessor(nn.Module):
    def __init__(
            self,
            input_size=672,
            clip_input_size=224,
            downscale_factor: int = 6,
            channels=None,  # 108
    ):
        super().__init__()
        # make sure they are evenly divisible
        assert input_size % clip_input_size == 0
        in_channels = 3

        self.input_size = input_size
        self.clip_input_size = clip_input_size
        self.downscale_factor = downscale_factor

        subpixel_channels = in_channels * downscale_factor ** 2  # 3 * 6 ** 2 = 108

        if channels is None:
            channels = subpixel_channels

        upscale_factor = downscale_factor / int((input_size / clip_input_size))  # 6 / (672 / 224) = 2

        num_upsample_blocks = int(upscale_factor // 2)  # 2 // 2 = 1

        # do a pooling layer to downscale the input to 1/3 of the size
        # (bs, 3, 672, 672) -> (bs, 3, 224, 224)
        kernel_size = input_size // clip_input_size
        self.res_down = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=kernel_size
        )  # (bs, 3, 672, 672) -> (bs, 3, 224, 224)

        # make a blending for output residual with near 0 weight
        self.res_blend = nn.Parameter(torch.tensor(0.001))  # (bs, 3, 224, 224) -> (bs, 3, 224, 224)

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)  # (bs, 3, 672, 672) -> (bs, 108, 112, 112)

        self.conv_in = nn.Sequential(
            nn.Conv2d(
                subpixel_channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.GELU()
        )  # (bs, 108, 112, 112) -> (bs, 108, 112, 112)

        self.upsample_blocks = nn.ModuleList()
        current_channels = channels
        for _ in range(num_upsample_blocks):
            out_channels = current_channels // 2
            self.upsample_blocks.append(UpsampleBlock(current_channels, out_channels))
            current_channels = out_channels

            # (bs, 108, 112, 112) -> (bs, 54, 224, 224)

        self.conv_out = nn.Conv2d(
            current_channels,
            out_channels=3,
            kernel_size=3,
            padding=1
        )  # (bs, 54, 224, 224) -> (bs, 3, 224, 224)


    def forward(self, x):
        # resize to input_size x input_size
        x = nn.functional.interpolate(x, size=(self.input_size, self.input_size), mode='bicubic')

        res = self.res_down(x)

        x = self.unshuffle(x)
        x = self.conv_in(x)
        for up in self.upsample_blocks:
            x = up(x)
        x = self.conv_out(x)
        # blend residual
        x = x * self.res_blend + res
        return x
