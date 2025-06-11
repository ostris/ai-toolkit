import torch
import torch.nn as nn
import numpy as np
import itertools


class LosslessLatentDecoder(nn.Module):
    def __init__(self, in_channels, latent_depth, dtype=torch.float32, trainable=False):
        super(LosslessLatentDecoder, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_depth = latent_depth
        self.in_channels = in_channels
        self.out_channels = int(in_channels // (latent_depth * latent_depth))
        numpy_kernel = self.build_kernel(in_channels, latent_depth)
        numpy_kernel = torch.from_numpy(numpy_kernel).to(device=device, dtype=dtype)
        if trainable:
            self.kernel = nn.Parameter(numpy_kernel)
        else:
            self.kernel = numpy_kernel

    def build_kernel(self, in_channels, latent_depth):
        # my old code from tensorflow.
        # tensorflow kernel is  (height, width, out_channels, in_channels)
        # pytorch kernel is     (in_channels, out_channels, height, width)
        out_channels = self.out_channels

        # kernel_shape = [kernel_filter_size, kernel_filter_size, out_channels, in_channels] # tensorflow
        kernel_shape = [in_channels, out_channels, latent_depth, latent_depth]  # pytorch
        kernel = np.zeros(kernel_shape, np.float32)

        # Build the kernel so that a 4 pixel cluster has each pixel come from a separate channel.
        for c in range(0, out_channels):
            i = 0
            for x, y in itertools.product(range(latent_depth), repeat=2):
                # kernel[y, x, c, c * latent_depth * latent_depth + i] = 1  # tensorflow
                kernel[c * latent_depth * latent_depth + i, c, y, x] = 1.0  # pytorch
                i += 1

        return kernel

    def forward(self, x):
        dtype = x.dtype
        if self.kernel.dtype != dtype:
            self.kernel = self.kernel.to(dtype=dtype)

        # Deconvolve input tensor with the kernel
        return nn.functional.conv_transpose2d(x, self.kernel, stride=self.latent_depth, padding=0, groups=1)


class LosslessLatentEncoder(nn.Module):
    def __init__(self, in_channels, latent_depth, dtype=torch.float32, trainable=False):
        super(LosslessLatentEncoder, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_depth = latent_depth
        self.in_channels = in_channels
        self.out_channels = int(in_channels * (latent_depth * latent_depth))
        numpy_kernel = self.build_kernel(in_channels, latent_depth)
        numpy_kernel = torch.from_numpy(numpy_kernel).to(device=device, dtype=dtype)
        if trainable:
            self.kernel = nn.Parameter(numpy_kernel)
        else:
            self.kernel = numpy_kernel


    def build_kernel(self, in_channels, latent_depth):
        # my old code from tensorflow.
        # tensorflow kernel is  (height, width, in_channels, out_channels)
        # pytorch kernel is     (out_channels, in_channels, height, width)
        out_channels = self.out_channels

        # kernel_shape = [latent_depth, latent_depth, in_channels, out_channels] # tensorflow
        kernel_shape = [out_channels, in_channels, latent_depth, latent_depth]  # pytorch
        kernel = np.zeros(kernel_shape, np.float32)

        # Build the kernel so that a 4 pixel cluster has each pixel come from a separate channel.
        for c in range(0, in_channels):
            i = 0
            for x, y in itertools.product(range(latent_depth), repeat=2):
                # kernel[y, x, c, c * latent_depth * latent_depth + i] = 1  # tensorflow
                kernel[c * latent_depth * latent_depth + i, c, y, x] = 1.0  # pytorch
                i += 1
        return kernel

    def forward(self, x):
        dtype = x.dtype
        if self.kernel.dtype != dtype:
            self.kernel = self.kernel.to(dtype=dtype)
        # Convolve input tensor with the kernel
        return nn.functional.conv2d(x, self.kernel, stride=self.latent_depth, padding=0, groups=1)


class LosslessLatentVAE(nn.Module):
    def __init__(self, in_channels, latent_depth, dtype=torch.float32, trainable=False):
        super(LosslessLatentVAE, self).__init__()
        self.latent_depth = latent_depth
        self.in_channels = in_channels
        self.encoder = LosslessLatentEncoder(in_channels, latent_depth, dtype=dtype, trainable=trainable)
        encoder_out_channels = self.encoder.out_channels
        self.decoder = LosslessLatentDecoder(encoder_out_channels, latent_depth, dtype=dtype, trainable=trainable)

    def forward(self, x):
        latent = self.latent_encoder(x)
        out = self.latent_decoder(latent)
        return out

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


# test it
if __name__ == '__main__':
    import os
    from extensions_built_in.dataset_tools.tools.image_tools import load_image
    import torchvision.transforms as transforms
    user_path = os.path.expanduser('~')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    input_path = os.path.join(user_path, "Pictures/sample_2_512.png")
    output_path = os.path.join(user_path, "Pictures/sample_2_512_llvae.png")
    img = load_image(input_path)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=dtype)
    print("input_shape: ", list(img_tensor.shape))
    vae = LosslessLatentVAE(in_channels=3, latent_depth=8, dtype=dtype).to(device=device, dtype=dtype)
    latent = vae.encode(img_tensor)
    print("latent_shape: ", list(latent.shape))
    out_tensor = vae.decode(latent)
    print("out_shape: ", list(out_tensor.shape))

    mse_loss = nn.MSELoss()
    mse = mse_loss(img_tensor, out_tensor)
    print("roundtrip_loss: ", mse.item())
    out_img = transforms.ToPILImage()(out_tensor.squeeze(0))
    out_img.save(output_path)
