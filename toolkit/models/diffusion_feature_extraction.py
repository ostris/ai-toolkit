import torch
import os
from torch import nn
from safetensors.torch import load_file
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels,
                              1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x + identity)
        return x


class DiffusionFeatureExtractor2(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.version = 2

        # Path 1: Upsample to 512x512 (1, 64, 512, 512)
        self.up_path = nn.ModuleList([
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(64, 64),
            nn.Conv2d(64, 64, 3, padding=1),
        ])

        # Path 2: Upsample to 256x256 (1, 128, 256, 256)
        self.path2 = nn.ModuleList([
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(128, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(128, 128),
            nn.Conv2d(128, 128, 3, padding=1),
        ])

        # Path 3: Upsample to 128x128 (1, 256, 128, 128)
        self.path3 = nn.ModuleList([
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(256, 256),
            nn.Conv2d(256, 256, 3, padding=1)
        ])

        # Path 4: Original size (1, 512, 64, 64)
        self.path4 = nn.ModuleList([
            nn.Conv2d(in_channels, 512, 3, padding=1),
            ResBlock(512, 512),
            ResBlock(512, 512),
            nn.Conv2d(512, 512, 3, padding=1)
        ])

        # Path 5: Downsample to 32x32 (1, 512, 32, 32)
        self.path5 = nn.ModuleList([
            nn.Conv2d(in_channels, 512, 3, padding=1),
            ResBlock(512, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 512),
            nn.Conv2d(512, 512, 3, padding=1)
        ])

    def forward(self, x):
        outputs = []

        # Path 1: 512x512
        x1 = x
        for layer in self.up_path:
            x1 = layer(x1)
        outputs.append(x1)  # [1, 64, 512, 512]

        # Path 2: 256x256
        x2 = x
        for layer in self.path2:
            x2 = layer(x2)
        outputs.append(x2)  # [1, 128, 256, 256]

        # Path 3: 128x128
        x3 = x
        for layer in self.path3:
            x3 = layer(x3)
        outputs.append(x3)  # [1, 256, 128, 128]

        # Path 4: 64x64
        x4 = x
        for layer in self.path4:
            x4 = layer(x4)
        outputs.append(x4)  # [1, 512, 64, 64]

        # Path 5: 32x32
        x5 = x
        for layer in self.path5:
            x5 = layer(x5)
        outputs.append(x5)  # [1, 512, 32, 32]

        return outputs


class DFEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x + x_in
        return x


class DiffusionFeatureExtractor(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.version = 1
        num_blocks = 6
        self.conv_in = nn.Conv2d(in_channels, 512, 1)
        self.blocks = nn.ModuleList([DFEBlock(512) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(512, 512, 1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)
        return x


def load_dfe(model_path) -> DiffusionFeatureExtractor:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # if it ende with safetensors
    if model_path.endswith('.safetensors'):
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

    if 'conv_in.weight' in state_dict:
        dfe = DiffusionFeatureExtractor()
    else:
        dfe = DiffusionFeatureExtractor2()

    dfe.load_state_dict(state_dict)
    dfe.eval()
    return dfe
