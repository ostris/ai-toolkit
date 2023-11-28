import copy
import random
from collections import OrderedDict
import os
from contextlib import nullcontext
from typing import Optional, Union, List
from torch.utils.data import ConcatDataset, DataLoader

from toolkit.config_modules import ReferenceDatasetConfig
from toolkit.data_loader import PairedImageDataset
from toolkit.prompt_utils import concat_prompt_embeds, split_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion, PromptEmbeds
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
import gc
from toolkit import train_tools
import torch
from jobs.process import BaseSDTrainProcess
import random
from toolkit.basic import value_map


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class ReferenceSliderConfig:
    def __init__(self, **kwargs):
        self.additional_losses: List[str] = kwargs.get('additional_losses', [])
        self.weight_jitter: float = kwargs.get('weight_jitter', 0.0)
        self.datasets: List[ReferenceDatasetConfig] = [ReferenceDatasetConfig(**d) for d in kwargs.get('datasets', [])]


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
            for dataset in self.slider_config.datasets:
                print(f" - Dataset: {dataset.pair_folder}")
                config = {
                    'path': dataset.pair_folder,
                    'size': dataset.size,
                    'default_prompt': dataset.target_class,
                    'network_weight': dataset.network_weight,
                    'pos_weight': dataset.pos_weight,
                    'neg_weight': dataset.neg_weight,
                    'pos_folder': dataset.pos_folder,
                    'neg_folder': dataset.neg_folder,
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
            imgs, prompts, network_weights = batch
            network_pos_weight, network_neg_weight = network_weights

            if isinstance(network_pos_weight, torch.Tensor):
                network_pos_weight = network_pos_weight.item()
            if isinstance(network_neg_weight, torch.Tensor):
                network_neg_weight = network_neg_weight.item()

            # get an array of random floats between -weight_jitter and weight_jitter
            loss_jitter_multiplier = 1.0
            weight_jitter = self.slider_config.weight_jitter
            if weight_jitter > 0.0:
                jitter_list = random.uniform(-weight_jitter, weight_jitter)
                orig_network_pos_weight = network_pos_weight
                network_pos_weight += jitter_list
                network_neg_weight += (jitter_list * -1.0)
                # penalize the loss for its distance from network_pos_weight
                # a jitter_list of abs(3.0) on a weight of 5.0 is a 60% jitter
                # so the loss_jitter_multiplier needs to be 0.4
                loss_jitter_multiplier = value_map(abs(jitter_list), 0.0, weight_jitter, 1.0, 0.0)


            # if items in network_weight list are tensors, convert them to floats

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

            noise_negative = noise_positive.clone()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_positive_latents = noise_scheduler.add_noise(positive_latents, noise_positive, timesteps)
            noisy_negative_latents = noise_scheduler.add_noise(negative_latents, noise_negative, timesteps)

            noisy_latents = torch.cat([noisy_positive_latents, noisy_negative_latents], dim=0)
            noise = torch.cat([noise_positive, noise_negative], dim=0)
            timesteps = torch.cat([timesteps, timesteps], dim=0)
            network_multiplier = [network_pos_weight * 1.0, network_neg_weight * -1.0]

        self.optimizer.zero_grad()
        noisy_latents.requires_grad = False

        # if training text encoder enable grads, else do context of no grad
        with torch.set_grad_enabled(self.train_config.train_text_encoder):
            # fix issue with them being tuples sometimes
            prompt_list = []
            for prompt in prompts:
                if isinstance(prompt, tuple):
                    prompt = prompt[0]
                prompt_list.append(prompt)
            conditional_embeds = self.sd.encode_prompt(prompt_list).to(self.device_torch, dtype=dtype)
            conditional_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])

        # if self.model_config.is_xl:
        #     # todo also allow for setting this for low ram in general, but sdxl spikes a ton on back prop
        #     network_multiplier_list = network_multiplier
        #     noisy_latent_list = torch.chunk(noisy_latents, 2, dim=0)
        #     noise_list = torch.chunk(noise, 2, dim=0)
        #     timesteps_list = torch.chunk(timesteps, 2, dim=0)
        #     conditional_embeds_list = split_prompt_embeds(conditional_embeds)
        # else:
        network_multiplier_list = [network_multiplier]
        noisy_latent_list = [noisy_latents]
        noise_list = [noise]
        timesteps_list = [timesteps]
        conditional_embeds_list = [conditional_embeds]

        losses = []
        # allow to chunk it out to save vram
        for network_multiplier, noisy_latents, noise, timesteps, conditional_embeds in zip(
                network_multiplier_list, noisy_latent_list, noise_list, timesteps_list, conditional_embeds_list
        ):
            with self.network:
                assert self.network.is_active

                self.network.multiplier = network_multiplier

                noise_pred = self.sd.predict_noise(
                    latents=noisy_latents.to(self.device_torch, dtype=dtype),
                    conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                    timestep=timesteps,
                )
                noise = noise.to(self.device_torch, dtype=dtype)

                if self.sd.prediction_type == 'v_prediction':
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(noisy_latents, noise, timesteps)
                else:
                    target = noise

                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])

                if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
                    # add min_snr_gamma
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, self.train_config.min_snr_gamma)

                loss = loss.mean() * loss_jitter_multiplier

                loss_float = loss.item()
                losses.append(loss_float)

                # back propagate loss to free ram
                loss.backward()

        # apply gradients
        optimizer.step()
        lr_scheduler.step()

        # reset network
        self.network.multiplier = 1.0

        loss_dict = OrderedDict(
            {'loss': sum(losses) / len(losses) if len(losses) > 0 else 0.0}
        )

        return loss_dict
        # end hook_train_loop
