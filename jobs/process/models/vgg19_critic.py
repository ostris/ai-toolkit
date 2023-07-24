import torch
import torch.nn as nn


class MeanReduce(nn.Module):
    def __init__(self):
        super(MeanReduce, self).__init__()

    def forward(self, inputs):
        return torch.mean(inputs, dim=(1, 2, 3), keepdim=True)


class Vgg19Critic(nn.Module):
    def __init__(self):
        # vgg19 input (bs, 3, 512, 512)
        # pool1 (bs, 64, 256, 256)
        # pool2 (bs, 128, 128, 128)
        # pool3 (bs, 256, 64, 64)
        # pool4 (bs, 512, 32, 32) <- take this input

        super(Vgg19Critic, self).__init__()
        self.main = nn.Sequential(
            # input (bs, 512, 32, 32)
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),  # (bs, 512, 16, 16)
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),  # (bs, 512, 8, 8)
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            # (bs, 1, 4, 4)
            MeanReduce(),  # (bs, 1, 1, 1)
            nn.Flatten(),  # (bs, 1)

            # nn.Flatten(),  # (128*8*8) = 8192
            # nn.Linear(128 * 8 * 8, 1)
        )

    def forward(self, inputs):
        return self.main(inputs)
