import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint


class ReductionKernel(nn.Module):
    # Tensorflow
    def __init__(self, in_channels, kernel_size=2, dtype=torch.float32, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(ReductionKernel, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        numpy_kernel = self.build_kernel()
        self.kernel = torch.from_numpy(numpy_kernel).to(device=device, dtype=dtype)

    def build_kernel(self):
        # tensorflow kernel is  (height, width, in_channels, out_channels)
        # pytorch kernel is     (out_channels, in_channels, height, width)
        kernel_size = self.kernel_size
        channels = self.in_channels
        kernel_shape = [channels, channels, kernel_size, kernel_size]
        kernel = np.zeros(kernel_shape, np.float32)

        kernel_value = 1.0 / (kernel_size * kernel_size)
        for i in range(0, channels):
            kernel[i, i, :, :] = kernel_value
        return kernel

    def forward(self, x):
        return nn.functional.conv2d(x, self.kernel, stride=self.kernel_size, padding=0, groups=1)


class CheckpointGradients(nn.Module):
    def __init__(self, is_gradient_checkpointing=True):
        super(CheckpointGradients, self).__init__()
        self.is_gradient_checkpointing = is_gradient_checkpointing

    def forward(self, module, *args, num_chunks=1):
        if self.is_gradient_checkpointing:
            return checkpoint(module, *args, num_chunks=self.num_chunks)
        else:
            return module(*args)
