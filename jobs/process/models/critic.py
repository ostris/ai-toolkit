import glob
import os
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from toolkit.losses import get_gradient_penalty
from toolkit.metadata import get_meta_for_safetensors
from toolkit.optimizer import get_optimizer
from toolkit.train_tools import get_torch_dtype


class MeanReduce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # global mean over spatial dims (keeps channel/batch)
        return torch.mean(inputs, dim=(2, 3), keepdim=True)


class SelfAttention2d(nn.Module):
    """
    Lightweight self-attention layer (SAGAN-style) that keeps spatial
    resolution unchanged. Adds minimal params / compute but improves
    long-range modelling – helpful for variable-sized inputs.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels,      1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.view(B, C, H * W)                    # (B,C,N)
        q = self.query(flat).permute(0, 2, 1)         # (B,N,C//8)
        k = self.key(flat)                            # (B,C//8,N)
        attn = torch.bmm(q, k)                        # (B,N,N)
        attn = attn.softmax(dim=-1)                   # softmax along last dim
        v = self.value(flat)                          # (B,C,N)
        out = torch.bmm(v, attn.permute(0, 2, 1))     # (B,C,N)
        out = out.view(B, C, H, W)                    # restore spatial dims
        return self.gamma * out + x                   # residual


class CriticModel(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()

        def sn_conv(in_c, out_c, k, s, p):
            return nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            )

        layers = [
            # initial down-sample
            sn_conv(3, base_channels, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_c = base_channels
        # progressive downsamples ×3 (64→128→256→512)
        for _ in range(3):
            out_c = min(in_c * 2, 1024)
            layers += [
                sn_conv(in_c, out_c, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            # single attention block after reaching 256 channels
            if out_c == 256:
                layers += [SelfAttention2d(out_c)]
            in_c = out_c

        # extra depth (keeps spatial size)
        layers += [
            sn_conv(in_c, 1024, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # final 1-channel prediction map
            sn_conv(1024, 1, 3, 1, 1),
            MeanReduce(),        # → (B,1,1,1)
            nn.Flatten(),        # → (B,1)
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, inputs):
        # force full-precision inside AMP ctx for stability
        with torch.cuda.amp.autocast(False):
            return self.main(inputs.float())


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
        self.model = CriticModel().to(self.device)
        self.load_weights()
        self.model.train()
        self.model.requires_grad_(True)
        params = self.model.parameters()
        self.optimizer = get_optimizer(
            params,
            self.optimizer_type,
            self.learning_rate,
            optimizer_params=self.optimizer_params,
        )
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(
            self.optimizer,
            total_iters=self.process.max_steps * self.num_critic_per_gen,
            factor=1,
            verbose=False,
        )

    def load_weights(self):
        path_to_load = None
        self.print(f"Critic: Looking for latest checkpoint in {self.process.save_root}")
        files = glob.glob(os.path.join(self.process.save_root, f"CRITIC_{self.process.job.name}*.safetensors"))
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(f" - Latest checkpoint is: {latest_file}")
            path_to_load = latest_file
        else:
            self.print(" - No checkpoint found, starting from scratch")
        if path_to_load:
            self.model.load_state_dict(load_file(path_to_load))

    def save(self, step=None):
        self.process.update_training_metadata()
        save_meta = get_meta_for_safetensors(self.process.meta, self.process.job.name)
        step_num = f"_{str(step).zfill(9)}" if step is not None else ''
        save_path = os.path.join(
            self.process.save_root, f"CRITIC_{self.process.job.name}{step_num}.safetensors"
        )
        save_file(self.model.state_dict(), save_path, save_meta)
        self.print(f"Saved critic to {save_path}")

    def get_critic_loss(self, vgg_output):
        # (caller still passes combined [pred|target] images)
        if self.start_step > self.process.step_num:
            return torch.tensor(0.0, dtype=self.torch_dtype, device=self.device)

        warmup_scaler = 1.0
        if self.process.step_num < self.start_step + self.warmup_steps:
            warmup_scaler = (self.process.step_num - self.start_step) / self.warmup_steps

        self.model.eval()
        self.model.requires_grad_(False)

        vgg_pred, _ = torch.chunk(vgg_output.float(), 2, dim=0)
        stacked_output = self.model(vgg_pred)
        return (-torch.mean(stacked_output)) * warmup_scaler

    def step(self, vgg_output):
        self.model.train()
        self.model.requires_grad_(True)
        self.optimizer.zero_grad()

        critic_losses = []
        inputs = vgg_output.detach().to(self.device, dtype=torch.float32)

        vgg_pred, vgg_target = torch.chunk(inputs, 2, dim=0)
        stacked_output = self.model(inputs).float()
        out_pred, out_target = torch.chunk(stacked_output, 2, dim=0)

        # hinge loss + gradient penalty
        loss_real = torch.relu(1.0 - out_target).mean()
        loss_fake = torch.relu(1.0 + out_pred).mean()
        gradient_penalty = get_gradient_penalty(self.model, vgg_target, vgg_pred, self.device)
        critic_loss = loss_real + loss_fake + self.lambda_gp * gradient_penalty

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        critic_losses.append(critic_loss.item())

        return float(np.mean(critic_losses))

    def get_lr(self):    
        if hasattr(self.optimizer, 'get_avg_learning_rate'):
            learning_rate = self.optimizer.get_avg_learning_rate()
        elif self.optimizer_type.startswith('dadaptation') or \
                self.optimizer_type.lower().startswith('prodigy'):
            learning_rate = (
                self.optimizer.param_groups[0]["d"] *
                self.optimizer.param_groups[0]["lr"]
            )
        else:
            learning_rate = self.optimizer.param_groups[0]['lr']
        return learning_rate
