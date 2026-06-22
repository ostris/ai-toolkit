from diffusers import AutoencoderKL
from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput


class PixelMixer(nn.Module):
    def __init__(self, in_channels, downscale_factor):
        super(PixelMixer, self).__init__()
        self.downscale_factor = downscale_factor
        self.in_channels = in_channels

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        return out

    def encode(self, x):
        return torch.nn.PixelUnshuffle(self.downscale_factor)(x)

    def decode(self, x):
        return torch.nn.PixelShuffle(self.downscale_factor)(x)


# for reference

# none of this matters with llvae, but we need to match the interface (latent_channels might matter)

class Config:
    in_channels = 3
    out_channels = 3
    down_block_types = ('1', '1',
                        '1', '1')
    up_block_types = ('1', '1',
                      '1', '1')
    block_out_channels = (1, 1, 1, 1)
    latent_channels = 192  # usually 4
    norm_num_groups = 32
    sample_size = 512
    # scaling_factor = 1
    # shift_factor = 0
    scaling_factor = 1.8
    shift_factor = -0.123
    # VAE
    # - Mean: -0.12306906282901764
    # - Std:  0.556016206741333
    # Normalization parameters:
    # - Shift factor: -0.12306906282901764
    # - Scaling factor: 1.7985087266803625

    def __getitem__(cls, x):
        return getattr(cls, x)


class AutoencoderPixelMixer(nn.Module):

    def __init__(self, in_channels=3, downscale_factor=8):
        super().__init__()
        self.mixer = PixelMixer(in_channels, downscale_factor)
        self._dtype = torch.float32
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()
        
        if downscale_factor == 8:
            # we go by len of block out channels in code, so simulate it
            self.config.block_out_channels = (1, 1, 1, 1)
            self.config.latent_channels = 192
        
        elif downscale_factor == 16:
            # we go by len of block out channels in code, so simulate it
            self.config.block_out_channels = (1, 1, 1, 1, 1)
            self.config.latent_channels = 768
        else:
            raise ValueError(
                f"downscale_factor {downscale_factor} not supported")

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    # mimic to from torch
    def to(self, *args, **kwargs):
        # pull out dtype and device if they exist
        if 'dtype' in kwargs:
            self._dtype = kwargs['dtype']
        if 'device' in kwargs:
            self._device = kwargs['device']
        return super().to(*args, **kwargs)

    def enable_xformers_memory_efficient_attention(self):
        pass

    # @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:

        h = self.mixer.encode(x)

        # moments = self.quant_conv(h)
        # posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (h,)

        class FakeDist:
            def __init__(self, x):
                self._sample = x

            def sample(self):
                return self._sample

        return AutoencoderKLOutput(latent_dist=FakeDist(h))

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        dec = self.mixer.decode(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    # @apply_forward_hook
    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _set_gradient_checkpointing(self, module, value=False):
        pass

    def enable_tiling(self, use_tiling: bool = True):
        pass

    def disable_tiling(self):
        pass

    def enable_slicing(self):
        pass

    def disable_slicing(self):
        pass

    def set_use_memory_efficient_attention_xformers(self, value: bool = True):
        pass

    def forward(
            self,
            sample: torch.FloatTensor,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


# test it
if __name__ == '__main__':
    import os
    from PIL import Image
    import torchvision.transforms as transforms
    user_path = os.path.expanduser('~')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    input_path = os.path.join(user_path, "Pictures/test/test.jpg")
    output_path = os.path.join(user_path, "Pictures/test/test.jpg")
    img = Image.open(input_path)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=dtype)
    print("input_shape: ", list(img_tensor.shape))
    vae = PixelMixer(in_channels=3, downscale_factor=8)
    latent = vae.encode(img_tensor)
    print("latent_shape: ", list(latent.shape))
    out_tensor = vae.decode(latent)
    print("out_shape: ", list(out_tensor.shape))

    mse_loss = nn.MSELoss()
    mse = mse_loss(img_tensor, out_tensor)
    print("roundtrip_loss: ", mse.item())
    out_img = transforms.ToPILImage()(out_tensor.squeeze(0))
    out_img.save(output_path)
