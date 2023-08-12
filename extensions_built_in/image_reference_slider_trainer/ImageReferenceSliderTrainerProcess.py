import copy
import random
from collections import OrderedDict
import os
from contextlib import nullcontext
from typing import Optional, Union, List
from torch.utils.data import ConcatDataset, DataLoader
from toolkit.data_loader import PairedImageDataset
from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion, PromptEmbeds
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
        self.additional_losses: List[str] = kwargs.get('additional_losses', [])


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
        do_mirror_loss = 'mirror' in self.slider_config.additional_losses

        with torch.no_grad():
            imgs, prompts = batch
            dtype = get_torch_dtype(self.train_config.dtype)
            imgs: torch.Tensor = imgs.to(self.device_torch, dtype=dtype)
            # split batched images in half so left is negative and right is positive
            negative_images, positive_images = torch.chunk(imgs, 2, dim=3)

            positive_latents = self.sd.encode_images(positive_images)
            negative_latents = self.sd.encode_images(negative_images)

            height = positive_images.shape[2]
            width = positive_images.shape[3]
            batch_size = positive_images.shape[0]

            if self.train_config.gradient_checkpointing:
                # may get disabled elsewhere
                self.sd.unet.enable_gradient_checkpointing()

            noise_scheduler = self.sd.noise_scheduler
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler

            self.sd.noise_scheduler.set_timesteps(
                self.train_config.max_denoising_steps, device=self.device_torch
            )

            timesteps = torch.randint(0, self.train_config.max_denoising_steps, (1,), device=self.device_torch)
            timesteps = timesteps.long()

            # get noise
            noise_positive = self.sd.get_latent_noise(
                pixel_height=height,
                pixel_width=width,
                batch_size=batch_size,
                noise_offset=self.train_config.noise_offset,
            ).to(self.device_torch, dtype=dtype)

            if do_mirror_loss:
                # mirror the noise
                # torch shape is [batch, channels, height, width]
                noise_negative = torch.flip(noise_positive.clone(), dims=[3])
            else:
                noise_negative = noise_positive.clone()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_positive_latents = noise_scheduler.add_noise(positive_latents, noise_positive, timesteps)
            noisy_negative_latents = noise_scheduler.add_noise(negative_latents, noise_negative, timesteps)

            noisy_latents = torch.cat([noisy_positive_latents, noisy_negative_latents], dim=0)
            noise = torch.cat([noise_positive, noise_negative], dim=0)
            timesteps = torch.cat([timesteps, timesteps], dim=0)
            network_multiplier = [1.0, -1.0]

        flush()

        loss_float = None
        loss_slide_float = None
        loss_mirror_float = None

        self.optimizer.zero_grad()
        noisy_latents.requires_grad = False

        # if training text encoder enable grads, else do context of no grad
        with torch.set_grad_enabled(self.train_config.train_text_encoder):
            # text encoding
            embedding_list = []
            # embed the prompts
            for prompt in prompts:
                embedding = self.sd.encode_prompt(prompt).to(self.device_torch, dtype=dtype)
                embedding_list.append(embedding)
            conditional_embeds = concat_prompt_embeds(embedding_list)
            conditional_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])

        with self.network:
            assert self.network.is_active

            self.network.multiplier = network_multiplier

            noise_pred = self.sd.predict_noise(
                latents=noisy_latents,
                conditional_embeddings=conditional_embeds,
                timestep=timesteps,
            )

            if self.sd.prediction_type == 'v_prediction':
                # v-parameterization training
                target = noise_scheduler.get_velocity(noisy_latents, noise, timesteps)
            else:
                target = noise

            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3])

            # todo add snr gamma here

            loss = loss.mean()
            loss_slide_float = loss.item()

            if do_mirror_loss:
                noise_pred_pos, noise_pred_neg = torch.chunk(noise_pred, 2, dim=0)
                # mirror the negative
                noise_pred_neg = torch.flip(noise_pred_neg.clone(), dims=[3])
                loss_mirror = torch.nn.functional.mse_loss(noise_pred_pos.float(), noise_pred_neg.float(), reduction="none")
                loss_mirror = loss_mirror.mean([1, 2, 3])
                loss_mirror = loss_mirror.mean()
                loss_mirror_float = loss_mirror.item()
                loss += loss_mirror

            loss_float = loss.item()

            # back propagate loss to free ram
            loss.backward()

            flush()

        # apply gradients
        optimizer.step()
        lr_scheduler.step()

        # reset network
        self.network.multiplier = 1.0

        loss_dict = OrderedDict(
            {'loss': loss_float},
        )

        if do_mirror_loss:
            loss_dict['l/s'] = loss_slide_float
            loss_dict['l/m'] = loss_mirror_float
        return loss_dict
        # end hook_train_loop
