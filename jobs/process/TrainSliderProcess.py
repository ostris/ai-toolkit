# ref:
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
import random
from collections import OrderedDict
import os
from typing import Optional, Union

from safetensors.torch import save_file, load_file
import torch.utils.checkpoint as cp
from tqdm import tqdm

from toolkit.config_modules import SliderConfig
from toolkit.layers import CheckpointGradients
from toolkit.paths import REPOS_ROOT
import sys

from toolkit.stable_diffusion_model import PromptEmbeds
from toolkit.train_tools import get_torch_dtype
import gc
from toolkit import train_tools
from toolkit.prompt_utils import \
    EncodedPromptPair, ACTION_TYPES_SLIDER, \
    EncodedAnchor, concat_prompt_pairs, \
    concat_anchors, PromptEmbedsCache, encode_prompts_to_cache, build_prompt_pair_batch_from_cache, split_anchors, \
    split_prompt_pairs

import torch
from .BaseSDTrainProcess import BaseSDTrainProcess


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class TrainSliderProcess(BaseSDTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.prompt_txt_list = None
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.slider_config = SliderConfig(**self.get_conf('slider', {}))
        self.prompt_cache = PromptEmbedsCache()
        self.prompt_pairs: list[EncodedPromptPair] = []
        self.anchor_pairs: list[EncodedAnchor] = []
        # keep track of prompt chunk size
        self.prompt_chunk_size = 1

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):

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

            # setup anchors
            anchor_pairs = []
            for anchor in self.slider_config.anchors:
                # build the cache
                for prompt in [
                    anchor.prompt,
                    anchor.neg_prompt  # empty neutral
                ]:
                    if cache[prompt] == None:
                        cache[prompt] = self.sd.encode_prompt(prompt)

                anchor_batch = []
                # we get the prompt pair multiplier from first prompt pair
                # since they are all the same. We need to match their network polarity
                prompt_pair_multipliers = prompt_pairs[0].multiplier_list
                for prompt_multiplier in prompt_pair_multipliers:
                    # match the network multiplier polarity
                    anchor_scalar = 1.0 if prompt_multiplier > 0 else -1.0
                    anchor_batch += [
                        EncodedAnchor(
                            prompt=cache[anchor.prompt],
                            neg_prompt=cache[anchor.neg_prompt],
                            multiplier=anchor.multiplier * anchor_scalar
                        )
                    ]

                anchor_pairs += [
                    concat_anchors(anchor_batch).to('cpu')
                ]
            if len(anchor_pairs) > 0:
                self.anchor_pairs = anchor_pairs

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
        # self.anchor_pairs = anchor_pairs
        flush()
        # end hook_before_train_loop

    def hook_train_loop(self, batch):
        dtype = get_torch_dtype(self.train_config.dtype)

        # get a random pair
        prompt_pair: EncodedPromptPair = self.prompt_pairs[
            torch.randint(0, len(self.prompt_pairs), (1,)).item()
        ]
        # move to device and dtype
        prompt_pair.to(self.device_torch, dtype=dtype)

        # get a random resolution
        height, width = self.slider_config.resolutions[
            torch.randint(0, len(self.slider_config.resolutions), (1,)).item()
        ]
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

            self.optimizer.zero_grad()

            # ger a random number of steps
            timesteps_to = torch.randint(
                1, self.train_config.max_denoising_steps, (1,)
            ).item()

            # for a complete slider, the batch size is 4 to begin with now
            true_batch_size = prompt_pair.target_class.text_embeds.shape[0] * self.train_config.batch_size

            # get noise
            noise = self.sd.get_latent_noise(
                pixel_height=height,
                pixel_width=width,
                batch_size=true_batch_size,
                noise_offset=self.train_config.noise_offset,
            ).to(self.device_torch, dtype=dtype)

            # get latents
            latents = noise * self.sd.noise_scheduler.init_noise_sigma
            latents = latents.to(self.device_torch, dtype=dtype)

            with self.network:
                assert self.network.is_active
                # pass the multiplier list to the network
                self.network.multiplier = prompt_pair.multiplier_list
                denoised_latents = self.sd.diffuse_some_steps(
                    latents,  # pass simple noise latents
                    train_tools.concat_prompt_embeddings(
                        prompt_pair.positive_target,  # unconditional
                        prompt_pair.target_class,  # target
                        self.train_config.batch_size,
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            # split the latents into out prompt pair chunks
            denoised_latent_chunks = torch.chunk(denoised_latents, self.prompt_chunk_size, dim=0)

            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / self.train_config.max_denoising_steps)
            ]

            # flush()  # 4.2GB to 3GB on 512x512

            # 4.20 GB RAM for 512x512
            positive_latents = get_noise_pred(
                prompt_pair.positive_target,  # negative prompt
                prompt_pair.negative_target,  # positive prompt
                1,
                current_timestep,
                denoised_latents
            )
            positive_latents.requires_grad = False
            positive_latents_chunks = torch.chunk(positive_latents, self.prompt_chunk_size, dim=0)

            neutral_latents = get_noise_pred(
                prompt_pair.positive_target,  # negative prompt
                prompt_pair.empty_prompt,  # positive prompt (normally neutral
                1,
                current_timestep,
                denoised_latents
            )
            neutral_latents.requires_grad = False
            neutral_latents_chunks = torch.chunk(neutral_latents, self.prompt_chunk_size, dim=0)

            unconditional_latents = get_noise_pred(
                prompt_pair.positive_target,  # negative prompt
                prompt_pair.positive_target,  # positive prompt
                1,
                current_timestep,
                denoised_latents
            )
            unconditional_latents.requires_grad = False
            unconditional_latents_chunks = torch.chunk(unconditional_latents, self.prompt_chunk_size, dim=0)

        flush()  # 4.2GB to 3GB on 512x512

        # 4.20 GB RAM for 512x512
        anchor_loss_float = None
        if len(self.anchor_pairs) > 0:
            with torch.no_grad():
                # get a random anchor pair
                anchor: EncodedAnchor = self.anchor_pairs[
                    torch.randint(0, len(self.anchor_pairs), (1,)).item()
                ]
                anchor.to(self.device_torch, dtype=dtype)

                # first we get the target prediction without network active
                anchor_target_noise = get_noise_pred(
                    anchor.neg_prompt, anchor.prompt, 1, current_timestep, denoised_latents
                    # ).to("cpu", dtype=torch.float32)
                ).requires_grad_(False)

                # to save vram, we will run these through separately while tracking grads
                # otherwise it consumes a ton of vram and this isn't our speed bottleneck
                anchor_chunks = split_anchors(anchor, self.prompt_chunk_size)
                anchor_target_noise_chunks = torch.chunk(anchor_target_noise, self.prompt_chunk_size, dim=0)
                assert len(anchor_chunks) == len(denoised_latent_chunks)

            # 4.32 GB RAM for 512x512
            with self.network:
                assert self.network.is_active
                anchor_float_losses = []
                for anchor_chunk, denoised_latent_chunk, anchor_target_noise_chunk in zip(
                        anchor_chunks, denoised_latent_chunks, anchor_target_noise_chunks
                ):
                    self.network.multiplier = anchor_chunk.multiplier_list

                    anchor_pred_noise = get_noise_pred(
                        anchor_chunk.neg_prompt, anchor_chunk.prompt, 1, current_timestep, denoised_latent_chunk
                    )
                    # 9.42 GB RAM for 512x512 -> 4.20 GB RAM for 512x512 with new grad_checkpointing
                    anchor_loss = loss_function(
                        anchor_target_noise_chunk,
                        anchor_pred_noise,
                    )
                    anchor_float_losses.append(anchor_loss.item())
                    # compute anchor loss gradients
                    # we will accumulate them later
                    # this saves a ton of memory doing them separately
                    anchor_loss.backward()
                    del anchor_pred_noise
                    del anchor_target_noise_chunk
                    del anchor_loss
                    flush()

            anchor_loss_float = sum(anchor_float_losses) / len(anchor_float_losses)
            del anchor_chunks
            del anchor_target_noise_chunks
            del anchor_target_noise
            # move anchor back to cpu
            anchor.to("cpu")
            flush()

        prompt_pair_chunks = split_prompt_pairs(prompt_pair, self.prompt_chunk_size)
        assert len(prompt_pair_chunks) == len(denoised_latent_chunks)
        # 3.28 GB RAM for 512x512
        with self.network:
            assert self.network.is_active
            loss_list = []
            for prompt_pair_chunk, \
                    denoised_latent_chunk, \
                    positive_latents_chunk, \
                    neutral_latents_chunk, \
                    unconditional_latents_chunk \
                    in zip(
                prompt_pair_chunks,
                denoised_latent_chunks,
                positive_latents_chunks,
                neutral_latents_chunks,
                unconditional_latents_chunks,
            ):
                self.network.multiplier = prompt_pair_chunk.multiplier_list
                target_latents = get_noise_pred(
                    prompt_pair_chunk.positive_target,
                    prompt_pair_chunk.target_class,
                    1,
                    current_timestep,
                    denoised_latent_chunk
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
                loss = loss_function(
                    target_latents,
                    offset_neutral,
                ) * prompt_pair_chunk.weight

                loss.backward()
                loss_list.append(loss.item())
                del target_latents
                del offset_neutral
                del loss
                flush()

        optimizer.step()
        lr_scheduler.step()

        loss_float = sum(loss_list) / len(loss_list)
        if anchor_loss_float is not None:
            loss_float += anchor_loss_float

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            latents
        )
        # move back to cpu
        prompt_pair.to("cpu")
        flush()

        # reset network
        self.network.multiplier = 1.0

        loss_dict = OrderedDict(
            {'loss': loss_float},
        )
        if anchor_loss_float is not None:
            loss_dict['sl_l'] = loss_float
            loss_dict['an_l'] = anchor_loss_float

        return loss_dict
        # end hook_train_loop
