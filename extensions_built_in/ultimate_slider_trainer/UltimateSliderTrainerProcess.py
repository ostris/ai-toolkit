import copy
import random
from collections import OrderedDict
import os
from contextlib import nullcontext
from typing import Optional, Union, List
from torch.utils.data import ConcatDataset, DataLoader

from toolkit.config_modules import ReferenceDatasetConfig
from toolkit.data_loader import PairedImageDataset
from toolkit.prompt_utils import concat_prompt_embeds, split_prompt_embeds, build_latent_image_batch_for_prompt_pair
from toolkit.stable_diffusion_model import StableDiffusion, PromptEmbeds
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
import gc
from toolkit import train_tools
import torch
from jobs.process import BaseSDTrainProcess
import random

import random
from collections import OrderedDict
from tqdm import tqdm

from toolkit.config_modules import SliderConfig
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
import gc
from toolkit import train_tools
from toolkit.prompt_utils import \
    EncodedPromptPair, ACTION_TYPES_SLIDER, \
    EncodedAnchor, concat_prompt_pairs, \
    concat_anchors, PromptEmbedsCache, encode_prompts_to_cache, build_prompt_pair_batch_from_cache, split_anchors, \
    split_prompt_pairs

import torch


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class UltimateSliderConfig(SliderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.additional_losses: List[str] = kwargs.get('additional_losses', [])
        self.weight_jitter: float = kwargs.get('weight_jitter', 0.0)
        self.img_loss_weight: float = kwargs.get('img_loss_weight', 1.0)
        self.cfg_loss_weight: float = kwargs.get('cfg_loss_weight', 1.0)
        self.datasets: List[ReferenceDatasetConfig] = [ReferenceDatasetConfig(**d) for d in kwargs.get('datasets', [])]


class UltimateSliderTrainerProcess(BaseSDTrainProcess):
    sd: StableDiffusion
    data_loader: DataLoader = None

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.prompt_txt_list = None
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.slider_config = UltimateSliderConfig(**self.get_conf('slider', {}))

        self.prompt_cache = PromptEmbedsCache()
        self.prompt_pairs: list[EncodedPromptPair] = []
        self.anchor_pairs: list[EncodedAnchor] = []
        # keep track of prompt chunk size
        self.prompt_chunk_size = 1

        # store a list of all the prompts from the dataset so we can cache it
        self.dataset_prompts = []
        self.train_with_dataset = self.slider_config.datasets is not None and len(self.slider_config.datasets) > 0

    def load_datasets(self):
        if self.data_loader is None and \
                self.slider_config.datasets is not None and len(self.slider_config.datasets) > 0:
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

                # capture all the prompts from it so we can cache the embeds
                self.dataset_prompts += image_dataset.get_all_prompts()

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
        # load any datasets if they were passed
        self.load_datasets()

        # read line by line from file
        if self.slider_config.prompt_file:
            self.print(f"Loading prompt file from {self.slider_config.prompt_file}")
            with open(self.slider_config.prompt_file, 'r', encoding='utf-8') as f:
                self.prompt_txt_list = f.readlines()
                # clean empty lines
                self.prompt_txt_list = [line.strip() for line in self.prompt_txt_list if len(line.strip()) > 0]

            self.print(f"Found {len(self.prompt_txt_list)} prompts.")

            if not self.slider_config.prompt_tensors:
                print(f"Prompt tensors not found. Building prompt tensors for {self.train_config.steps} steps.")
                # shuffle
                random.shuffle(self.prompt_txt_list)
                # trim to max steps
                self.prompt_txt_list = self.prompt_txt_list[:self.train_config.steps]
                # trim list to our max steps

        cache = PromptEmbedsCache()

        # get encoded latents for our prompts
        with torch.no_grad():
            # list of neutrals. Can come from file or be empty
            neutral_list = self.prompt_txt_list if self.prompt_txt_list is not None else [""]

            # build the prompts to cache
            prompts_to_cache = []
            for neutral in neutral_list:
                for target in self.slider_config.targets:
                    prompt_list = [
                        f"{target.target_class}",  # target_class
                        f"{target.target_class} {neutral}",  # target_class with neutral
                        f"{target.positive}",  # positive_target
                        f"{target.positive} {neutral}",  # positive_target with neutral
                        f"{target.negative}",  # negative_target
                        f"{target.negative} {neutral}",  # negative_target with neutral
                        f"{neutral}",  # neutral
                        f"{target.positive} {target.negative}",  # both targets
                        f"{target.negative} {target.positive}",  # both targets reverse
                    ]
                    prompts_to_cache += prompt_list

            # remove duplicates
            prompts_to_cache = list(dict.fromkeys(prompts_to_cache))

            # trim to max steps if max steps is lower than prompt count
            prompts_to_cache = prompts_to_cache[:self.train_config.steps]

            if len(self.dataset_prompts) > 0:
                # add the prompts from the dataset
                prompts_to_cache += self.dataset_prompts

            # encode them
            cache = encode_prompts_to_cache(
                prompt_list=prompts_to_cache,
                sd=self.sd,
                cache=cache,
                prompt_tensor_file=self.slider_config.prompt_tensors
            )

            prompt_pairs = []
            prompt_batches = []
            for neutral in tqdm(neutral_list, desc="Building Prompt Pairs", leave=False):
                for target in self.slider_config.targets:
                    prompt_pair_batch = build_prompt_pair_batch_from_cache(
                        cache=cache,
                        target=target,
                        neutral=neutral,

                    )
                    if self.slider_config.batch_full_slide:
                        # concat the prompt pairs
                        # this allows us to run the entire 4 part process in one shot (for slider)
                        self.prompt_chunk_size = 4
                        concat_prompt_pair_batch = concat_prompt_pairs(prompt_pair_batch).to('cpu')
                        prompt_pairs += [concat_prompt_pair_batch]
                    else:
                        self.prompt_chunk_size = 1
                        # do them one at a time (probably not necessary after new optimizations)
                        prompt_pairs += [x.to('cpu') for x in prompt_pair_batch]

        # move to cpu to save vram
        # We don't need text encoder anymore, but keep it on cpu for sampling
        # if text encoder is list
        if isinstance(self.sd.text_encoder, list):
            for encoder in self.sd.text_encoder:
                encoder.to("cpu")
        else:
            self.sd.text_encoder.to("cpu")
        self.prompt_cache = cache
        self.prompt_pairs = prompt_pairs
        # end hook_before_train_loop

        # move vae to device so we can encode on the fly
        # todo cache latents
        self.sd.vae.to(self.device_torch)
        self.sd.vae.eval()
        self.sd.vae.requires_grad_(False)

        if self.train_config.gradient_checkpointing:
            # may get disabled elsewhere
            self.sd.unet.enable_gradient_checkpointing()

        flush()
        # end hook_before_train_loop

    def hook_train_loop(self, batch):
        dtype = get_torch_dtype(self.train_config.dtype)

        with torch.no_grad():
            ### LOOP SETUP ###
            noise_scheduler = self.sd.noise_scheduler
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler

            ### TARGET_PROMPTS ###
            # get a random pair
            prompt_pair: EncodedPromptPair = self.prompt_pairs[
                torch.randint(0, len(self.prompt_pairs), (1,)).item()
            ]
            # move to device and dtype
            prompt_pair.to(self.device_torch, dtype=dtype)

            ### PREP REFERENCE IMAGES ###

            imgs, prompts, network_weights = batch
            network_pos_weight, network_neg_weight = network_weights

            if isinstance(network_pos_weight, torch.Tensor):
                network_pos_weight = network_pos_weight.item()
            if isinstance(network_neg_weight, torch.Tensor):
                network_neg_weight = network_neg_weight.item()

            # get an array of random floats between -weight_jitter and weight_jitter
            weight_jitter = self.slider_config.weight_jitter
            if weight_jitter > 0.0:
                jitter_list = random.uniform(-weight_jitter, weight_jitter)
                network_pos_weight += jitter_list
                network_neg_weight += (jitter_list * -1.0)

            # if items in network_weight list are tensors, convert them to floats
            imgs: torch.Tensor = imgs.to(self.device_torch, dtype=dtype)
            # split batched images in half so left is negative and right is positive
            negative_images, positive_images = torch.chunk(imgs, 2, dim=3)

            height = positive_images.shape[2]
            width = positive_images.shape[3]
            batch_size = positive_images.shape[0]

            positive_latents = self.sd.encode_images(positive_images)
            negative_latents = self.sd.encode_images(negative_images)

            self.sd.noise_scheduler.set_timesteps(
                self.train_config.max_denoising_steps, device=self.device_torch
            )

            timesteps = torch.randint(0, self.train_config.max_denoising_steps, (1,), device=self.device_torch)
            current_timestep_index = timesteps.item()
            current_timestep = noise_scheduler.timesteps[current_timestep_index]
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

            ### CFG SLIDER TRAINING PREP ###

            # get CFG txt latents
            noisy_cfg_latents = build_latent_image_batch_for_prompt_pair(
                pos_latent=noisy_positive_latents,
                neg_latent=noisy_negative_latents,
                prompt_pair=prompt_pair,
                prompt_chunk_size=self.prompt_chunk_size,
            )
            noisy_cfg_latents.requires_grad = False

            assert not self.network.is_active

            # 4.20 GB RAM for 512x512
            positive_latents = self.sd.predict_noise(
                latents=noisy_cfg_latents,
                text_embeddings=train_tools.concat_prompt_embeddings(
                    prompt_pair.positive_target,  # negative prompt
                    prompt_pair.negative_target,  # positive prompt
                    self.train_config.batch_size,
                ),
                timestep=current_timestep,
                guidance_scale=1.0
            )
            positive_latents.requires_grad = False

            neutral_latents = self.sd.predict_noise(
                latents=noisy_cfg_latents,
                text_embeddings=train_tools.concat_prompt_embeddings(
                    prompt_pair.positive_target,  # negative prompt
                    prompt_pair.empty_prompt,  # positive prompt (normally neutral
                    self.train_config.batch_size,
                ),
                timestep=current_timestep,
                guidance_scale=1.0
            )
            neutral_latents.requires_grad = False

            unconditional_latents = self.sd.predict_noise(
                latents=noisy_cfg_latents,
                text_embeddings=train_tools.concat_prompt_embeddings(
                    prompt_pair.positive_target,  # negative prompt
                    prompt_pair.positive_target,  # positive prompt
                    self.train_config.batch_size,
                ),
                timestep=current_timestep,
                guidance_scale=1.0
            )
            unconditional_latents.requires_grad = False

            positive_latents_chunks = torch.chunk(positive_latents, self.prompt_chunk_size, dim=0)
            neutral_latents_chunks = torch.chunk(neutral_latents, self.prompt_chunk_size, dim=0)
            unconditional_latents_chunks = torch.chunk(unconditional_latents, self.prompt_chunk_size, dim=0)
            prompt_pair_chunks = split_prompt_pairs(prompt_pair, self.prompt_chunk_size)
            noisy_cfg_latents_chunks = torch.chunk(noisy_cfg_latents, self.prompt_chunk_size, dim=0)
            assert len(prompt_pair_chunks) == len(noisy_cfg_latents_chunks)

            noisy_latents = torch.cat([noisy_positive_latents, noisy_negative_latents], dim=0)
            noise = torch.cat([noise_positive, noise_negative], dim=0)
            timesteps = torch.cat([timesteps, timesteps], dim=0)
            network_multiplier = [network_pos_weight * 1.0, network_neg_weight * -1.0]

        flush()

        loss_float = None
        loss_mirror_float = None

        self.optimizer.zero_grad()
        noisy_latents.requires_grad = False

        # TODO allow both processed to train text encoder, for now, we just to unet and cache all text encodes
        # if training text encoder enable grads, else do context of no grad
        # with torch.set_grad_enabled(self.train_config.train_text_encoder):
        #     # text encoding
        #     embedding_list = []
        #     # embed the prompts
        #     for prompt in prompts:
        #         embedding = self.sd.encode_prompt(prompt).to(self.device_torch, dtype=dtype)
        #         embedding_list.append(embedding)
        #     conditional_embeds = concat_prompt_embeds(embedding_list)
        #     conditional_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])

        if self.train_with_dataset:
            embedding_list = []
            with torch.set_grad_enabled(self.train_config.train_text_encoder):
                for prompt in prompts:
                    # get embedding form cache
                    embedding = self.prompt_cache[prompt]
                    embedding = embedding.to(self.device_torch, dtype=dtype)
                    embedding_list.append(embedding)
                conditional_embeds = concat_prompt_embeds(embedding_list)
                # double up so we can do both sides of the slider
                conditional_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])
        else:
            # throw error. Not supported yet
            raise Exception("Datasets and targets required for ultimate slider")

        if self.model_config.is_xl:
            # todo also allow for setting this for low ram in general, but sdxl spikes a ton on back prop
            network_multiplier_list = network_multiplier
            noisy_latent_list = torch.chunk(noisy_latents, 2, dim=0)
            noise_list = torch.chunk(noise, 2, dim=0)
            timesteps_list = torch.chunk(timesteps, 2, dim=0)
            conditional_embeds_list = split_prompt_embeds(conditional_embeds)
        else:
            network_multiplier_list = [network_multiplier]
            noisy_latent_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditional_embeds_list = [conditional_embeds]

        ## DO REFERENCE IMAGE TRAINING ##

        reference_image_losses = []
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

                # todo add snr gamma here
                if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
                    # add min_snr_gamma
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, self.train_config.min_snr_gamma)

                loss = loss.mean()
                loss = loss * self.slider_config.img_loss_weight
                loss_slide_float = loss.item()

                loss_float = loss.item()
                reference_image_losses.append(loss_float)

                # back propagate loss to free ram
                loss.backward()
                flush()

        ## DO CFG SLIDER TRAINING ##

        cfg_loss_list = []

        with self.network:
            assert self.network.is_active
            for prompt_pair_chunk, \
                    noisy_cfg_latent_chunk, \
                    positive_latents_chunk, \
                    neutral_latents_chunk, \
                    unconditional_latents_chunk \
                    in zip(
                prompt_pair_chunks,
                noisy_cfg_latents_chunks,
                positive_latents_chunks,
                neutral_latents_chunks,
                unconditional_latents_chunks,
            ):
                self.network.multiplier = prompt_pair_chunk.multiplier_list

                target_latents = self.sd.predict_noise(
                    latents=noisy_cfg_latent_chunk,
                    text_embeddings=train_tools.concat_prompt_embeddings(
                        prompt_pair_chunk.positive_target,  # negative prompt
                        prompt_pair_chunk.target_class,  # positive prompt
                        self.train_config.batch_size,
                    ),
                    timestep=current_timestep,
                    guidance_scale=1.0
                )

                guidance_scale = 1.0

                offset = guidance_scale * (positive_latents_chunk - unconditional_latents_chunk)

                # make offset multiplier based on actions
                offset_multiplier_list = []
                for action in prompt_pair_chunk.action_list:
                    if action == ACTION_TYPES_SLIDER.ERASE_NEGATIVE:
                        offset_multiplier_list += [-1.0]
                    elif action == ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE:
                        offset_multiplier_list += [1.0]

                offset_multiplier = torch.tensor(offset_multiplier_list).to(offset.device, dtype=offset.dtype)
                # make offset multiplier match rank of offset
                offset_multiplier = offset_multiplier.view(offset.shape[0], 1, 1, 1)
                offset *= offset_multiplier

                offset_neutral = neutral_latents_chunk
                # offsets are already adjusted on a per-batch basis
                offset_neutral += offset

                # 16.15 GB RAM for 512x512 -> 4.20GB RAM for 512x512 with new grad_checkpointing
                loss = torch.nn.functional.mse_loss(target_latents.float(), offset_neutral.float(), reduction="none")
                loss = loss.mean([1, 2, 3])

                if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
                    # match batch size
                    timesteps_index_list = [current_timestep_index for _ in range(target_latents.shape[0])]
                    # add min_snr_gamma
                    loss = apply_snr_weight(loss, timesteps_index_list, noise_scheduler,
                                            self.train_config.min_snr_gamma)

                loss = loss.mean() * prompt_pair_chunk.weight * self.slider_config.cfg_loss_weight

                loss.backward()
                cfg_loss_list.append(loss.item())
                del target_latents
                del offset_neutral
                del loss
                flush()

        # apply gradients
        optimizer.step()
        lr_scheduler.step()

        # reset network
        self.network.multiplier = 1.0

        reference_image_loss = sum(reference_image_losses) / len(reference_image_losses) if len(
            reference_image_losses) > 0 else 0.0
        cfg_loss = sum(cfg_loss_list) / len(cfg_loss_list) if len(cfg_loss_list) > 0 else 0.0

        loss_dict = OrderedDict({
            'loss/img': reference_image_loss,
            'loss/cfg': cfg_loss,
        })

        return loss_dict
        # end hook_train_loop
