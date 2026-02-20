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
            input_size=896,
            clip_input_size=224,
            downscale_factor: int = 16,
    ):
        super().__init__()
        # make sure they are evenly divisible
        assert input_size % clip_input_size == 0
        in_channels = 3

        self.input_size = input_size
        self.clip_input_size = clip_input_size
        self.downscale_factor = downscale_factor

        subpixel_channels = in_channels * downscale_factor ** 2  # 3 * 16 ** 2 = 768
        channels = subpixel_channels

        upscale_factor = downscale_factor / int((input_size / clip_input_size))  # 16 / (896 / 224) = 4

        num_upsample_blocks = int(upscale_factor // 2)  # 4 // 2 = 2

        # make the residual down up blocks
        self.upsample_blocks = nn.ModuleList()
        self.subpixel_blocks = nn.ModuleList()
        current_channels = channels
        current_downscale = downscale_factor
        for _ in range(num_upsample_blocks):
            # determine the reshuffled channel count for this dimension
            output_downscale = current_downscale // 2
            out_channels = in_channels * output_downscale ** 2
            # out_channels = current_channels // 2
            self.upsample_blocks.append(UpsampleBlock(current_channels, out_channels))
            current_channels = out_channels
            current_downscale = output_downscale
            self.subpixel_blocks.append(nn.PixelUnshuffle(current_downscale))

            # (bs, 768, 56, 56) -> (bs, 192, 112, 112)
            # (bs, 192, 112, 112) -> (bs, 48, 224, 224)

        self.conv_out = nn.Conv2d(
            current_channels,
            out_channels=3,
            kernel_size=3,
            padding=1
        )  # (bs, 48, 224, 224) -> (bs, 3, 224, 224)

        # do a pooling layer to downscale the input to 1/3 of the size
        # (bs, 3, 896, 896) -> (bs, 3, 224, 224)
        kernel_size = input_size // clip_input_size
        self.res_down = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=kernel_size
        )  # (bs, 3, 896, 896) -> (bs, 3, 224, 224)

        # make a blending for output residual with near 0 weight
        self.res_blend = nn.Parameter(torch.tensor(0.001))  # (bs, 3, 224, 224) -> (bs, 3, 224, 224)

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)  # (bs, 3, 896, 896) -> (bs, 768, 56, 56)

        self.conv_in = nn.Sequential(
            nn.Conv2d(
                subpixel_channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.GELU()
        )  # (bs, 768, 56, 56) -> (bs, 768, 56, 56)

        # make 2 deep blocks

    def forward(self, x):
        inputs = x
        # resize to input_size x input_size
        x = nn.functional.interpolate(x, size=(self.input_size, self.input_size), mode='bicubic')

        res = self.res_down(inputs)

        x = self.unshuffle(x)
        x = self.conv_in(x)
        for up, subpixel in zip(self.upsample_blocks, self.subpixel_blocks):
            x = up(x)
            block_res = subpixel(inputs)
            x = x + block_res
        x = self.conv_out(x)
        # blend residual
        x = x * self.res_blend + res
        return x
