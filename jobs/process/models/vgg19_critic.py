import glob
import os

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from toolkit.losses import get_gradient_penalty
from toolkit.metadata import get_meta_for_safetensors
from toolkit.optimizer import get_optimizer
from toolkit.train_tools import get_torch_dtype

from typing import TYPE_CHECKING, Union


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


if TYPE_CHECKING:
    from jobs.process.TrainVAEProcess import TrainVAEProcess
    from jobs.process.TrainESRGANProcess import TrainESRGANProcess


class Critic:
    process: Union['TrainVAEProcess', 'TrainESRGANProcess']

    def __init__(
            self,
            learning_rate=1e-5,
            device='cpu',
            optimizer='adam',
            num_critic_per_gen=1,
            dtype='float32',
            lambda_gp=10,
            start_step=0,
            warmup_steps=1000,
            process=None,
            optimizer_params=None,
    ):
        self.learning_rate = learning_rate
        self.device = device
        self.optimizer_type = optimizer
        self.num_critic_per_gen = num_critic_per_gen
        self.dtype = dtype
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.process = process
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.warmup_steps = warmup_steps
        self.start_step = start_step
        self.lambda_gp = lambda_gp

        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer_params = optimizer_params
        self.print = self.process.print
        print(f" Critic config: {self.__dict__}")

    def setup(self):
        self.model = Vgg19Critic().to(self.device, dtype=self.torch_dtype)
        self.load_weights()
        self.model.train()
        self.model.requires_grad_(True)
        params = self.model.parameters()
        self.optimizer = get_optimizer(params, self.optimizer_type, self.learning_rate,
                                       optimizer_params=self.optimizer_params)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(
            self.optimizer,
            total_iters=self.process.max_steps * self.num_critic_per_gen,
            factor=1,
            verbose=False
        )

    def load_weights(self):
        path_to_load = None
        self.print(f"Critic: Looking for latest checkpoint in {self.process.save_root}")
        files = glob.glob(os.path.join(self.process.save_root, f"CRITIC_{self.process.job.name}*.safetensors"))
        if files and len(files) > 0:
            latest_file = max(files, key=os.path.getmtime)
            print(f" - Latest checkpoint is: {latest_file}")
            path_to_load = latest_file
        else:
            self.print(f" - No checkpoint found, starting from scratch")
        if path_to_load:
            self.model.load_state_dict(load_file(path_to_load))

    def save(self, step=None):
        self.process.update_training_metadata()
        save_meta = get_meta_for_safetensors(self.process.meta, self.process.job.name)
        step_num = ''
        if step is not None:
            # zeropad 9 digits
            step_num = f"_{str(step).zfill(9)}"
        save_path = os.path.join(self.process.save_root, f"CRITIC_{self.process.job.name}{step_num}.safetensors")
        save_file(self.model.state_dict(), save_path, save_meta)
        self.print(f"Saved critic to {save_path}")

    def get_critic_loss(self, vgg_output):
        if self.start_step > self.process.step_num:
            return torch.tensor(0.0, dtype=self.torch_dtype, device=self.device)

        warmup_scaler = 1.0
        # we need a warmup when we come on of 1000 steps
        # we want to scale the loss by 0.0 at self.start_step steps and 1.0 at self.start_step + warmup_steps
        if self.process.step_num < self.start_step + self.warmup_steps:
            warmup_scaler = (self.process.step_num - self.start_step) / self.warmup_steps
        # set model to not train for generator loss
        self.model.eval()
        self.model.requires_grad_(False)
        vgg_pred, vgg_target = torch.chunk(vgg_output, 2, dim=0)

        # run model
        stacked_output = self.model(vgg_pred)

        return (-torch.mean(stacked_output)) * warmup_scaler

    def step(self, vgg_output):

        # train critic here
        self.model.train()
        self.model.requires_grad_(True)
        self.optimizer.zero_grad()

        critic_losses = []
        inputs = vgg_output.detach()
        inputs = inputs.to(self.device, dtype=self.torch_dtype)
        self.optimizer.zero_grad()

        vgg_pred, vgg_target = torch.chunk(inputs, 2, dim=0)

        stacked_output = self.model(inputs).float()
        out_pred, out_target = torch.chunk(stacked_output, 2, dim=0)

        # Compute gradient penalty
        gradient_penalty = get_gradient_penalty(self.model, vgg_target, vgg_pred, self.device)

        # Compute WGAN-GP critic loss
        critic_loss = -(torch.mean(out_target) - torch.mean(out_pred)) + self.lambda_gp * gradient_penalty
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        critic_losses.append(critic_loss.item())

        # avg loss
        loss = np.mean(critic_losses)
        return loss

    def get_lr(self):
        if self.optimizer_type.startswith('dadaptation'):
            learning_rate = (
                    self.optimizer.param_groups[0]["d"] *
                    self.optimizer.param_groups[0]["lr"]
            )
        else:
            learning_rate = self.optimizer.param_groups[0]['lr']

        return learning_rate

