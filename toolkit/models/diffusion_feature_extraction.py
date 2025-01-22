import torch
import os
from torch import nn
from safetensors.torch import load_file


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
    dfe = DiffusionFeatureExtractor()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # if it ende with safetensors
    if model_path.endswith('.safetensors'):
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

    dfe.load_state_dict(state_dict)
    dfe.eval()
    return dfe
