import copy
import random
from collections import OrderedDict
import os
from typing import Optional, Union, List
from torch.utils.data import ConcatDataset, DataLoader
from toolkit.data_loader import PairedImageDataset
from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
import gc
from toolkit import train_tools
import torch
from jobs.process import BaseSDTrainProcess


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class ReferenceSliderConfig:
    def __init__(self, **kwargs):
        self.slider_pair_folder: str = kwargs.get('slider_pair_folder', None)
        self.resolutions: List[int] = kwargs.get('resolutions', [512])
        self.batch_full_slide: bool = kwargs.get('batch_full_slide', True)
        self.target_class: int = kwargs.get('target_class', '')


class ImageReferenceSliderTrainerProcess(BaseSDTrainProcess):
    sd: StableDiffusion
    data_loader: DataLoader = None

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.prompt_txt_list = None
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.slider_config = ReferenceSliderConfig(**self.get_conf('slider', {}))

    def load_datasets(self):
        if self.data_loader is None:
            print(f"Loading datasets")
            datasets = []
            for res in self.slider_config.resolutions:
                print(f" - Dataset: {self.slider_config.slider_pair_folder}")
                config = {
                    'path': self.slider_config.slider_pair_folder,
                    'size': res,
                    'default_prompt': self.slider_config.target_class
                }
                image_dataset = PairedImageDataset(config)
                datasets.append(image_dataset)

            concatenated_dataset = ConcatDataset(datasets)
            self.data_loader = DataLoader(
                concatenated_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                num_workers=2
            )

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        self.sd.vae.eval()
        self.sd.vae.to(self.device_torch)
        self.load_datasets()

        pass

    def hook_train_loop(self, batch):
        with torch.no_grad():
            imgs, prompts = batch
            dtype = get_torch_dtype(self.train_config.dtype)
            imgs: torch.Tensor = imgs.to(self.device_torch, dtype=dtype)

            # split batched images in half so left is negative and right is positive
            negative_images, positive_images = torch.chunk(imgs, 2, dim=3)

            height = positive_images.shape[2]
            width = positive_images.shape[3]
            batch_size = positive_images.shape[0]

            # encode the images
            positive_latents = self.sd.vae.encode(positive_images).latent_dist.sample()
            positive_latents = positive_latents * 0.18215
            negative_latents = self.sd.vae.encode(negative_images).latent_dist.sample()
            negative_latents = negative_latents * 0.18215

            embedding_list = []
            negative_embedding_list = []
            # embed the prompts
            for prompt in prompts:
                embedding = self.sd.encode_prompt(prompt).to(self.device_torch, dtype=dtype)
                embedding_list.append(embedding)
                # just empty for now
                # todo cache this?
                negative_embed = self.sd.encode_prompt('').to(self.device_torch, dtype=dtype)
                negative_embedding_list.append(negative_embed)

            conditional_embeds = concat_prompt_embeds(embedding_list)
            unconditional_embeds = concat_prompt_embeds(negative_embedding_list)

            if self.train_config.gradient_checkpointing:
                # may get disabled elsewhere
                self.sd.unet.enable_gradient_checkpointing()

            noise_scheduler = self.sd.noise_scheduler
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler
            loss_function = torch.nn.MSELoss()

        def get_noise_pred(neg, pos, gs, cts, dn):
            return self.sd.predict_noise(
                latents=dn,
                text_embeddings=train_tools.concat_prompt_embeddings(
                    neg,  # negative prompt
                    pos,  # positive prompt
                    self.train_config.batch_size,
                ),
                timestep=cts,
                guidance_scale=gs,
            )

        with torch.no_grad():
            self.sd.noise_scheduler.set_timesteps(
                self.train_config.max_denoising_steps, device=self.device_torch
            )

            timesteps = torch.randint(0, self.train_config.max_denoising_steps, (batch_size,), device=self.device_torch)
            timesteps = timesteps.long()

            # get noise
            noise = self.sd.get_latent_noise(
                pixel_height=height,
                pixel_width=width,
                batch_size=batch_size,
                noise_offset=self.train_config.noise_offset,
            ).to(self.device_torch, dtype=dtype)

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_positive_latents = noise_scheduler.add_noise(positive_latents, noise, timesteps)
            noisy_negative_latents = noise_scheduler.add_noise(negative_latents, noise, timesteps)

        flush()

        self.optimizer.zero_grad()
        with self.network:
            assert self.network.is_active
            loss_list = []
            for noisy_latents, network_multiplier in zip(
                    [noisy_positive_latents, noisy_negative_latents],
                    [1.0, -1.0],
            ):
                # do positive first
                self.network.multiplier = network_multiplier

                noise_pred = get_noise_pred(
                    unconditional_embeds,
                    conditional_embeds,
                    1,
                    timesteps,
                    noisy_latents
                )

                if self.sd.is_v2:  # check is vpred, don't want to track it down right now
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(noisy_latents, noise, timesteps)
                else:
                    target = noise

                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])

                # todo add snr gamma here

                loss = loss.mean()
                # back propagate loss to free ram
                loss.backward()
                loss_list.append(loss.item())

                flush()

        # apply gradients
        optimizer.step()
        lr_scheduler.step()

        loss_float = sum(loss_list) / len(loss_list)

        # reset network
        self.network.multiplier = 1.0

        loss_dict = OrderedDict(
            {'loss': loss_float},
        )
        return loss_dict
        # end hook_train_loop
