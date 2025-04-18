# ref:
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
import time
from collections import OrderedDict
import os
from typing import Optional

from toolkit.config_modules import SliderConfig
import sys

from toolkit.stable_diffusion_model import PromptEmbeds

from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import gc
from toolkit import train_tools

import torch
from .BaseSDTrainProcess import BaseSDTrainProcess, StableDiffusion


class ACTION_TYPES_SLIDER:
    ERASE_NEGATIVE = 0
    ENHANCE_NEGATIVE = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class EncodedPromptPair:
    def __init__(
            self,
            target_class,
            positive,
            negative,
            neutral,
            width=512,
            height=512,
            action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
            multiplier=1.0,
            weight=1.0
    ):
        self.target_class = target_class
        self.positive = positive
        self.negative = negative
        self.neutral = neutral
        self.width = width
        self.height = height
        self.action: int = action
        self.multiplier = multiplier
        self.weight = weight


class PromptEmbedsCache:  # 使いまわしたいので
    prompts: dict[str, PromptEmbeds] = {}

    def __setitem__(self, __name: str, __value: PromptEmbeds) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str) -> Optional[PromptEmbeds]:
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


class EncodedAnchor:
    def __init__(
            self,
            prompt,
            neg_prompt,
            multiplier=1.0
    ):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.multiplier = multiplier


class TrainSliderProcessOld(BaseSDTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.slider_config = SliderConfig(**self.get_conf('slider', {}))
        self.prompt_cache = PromptEmbedsCache()
        self.prompt_pairs: list[EncodedPromptPair] = []
        self.anchor_pairs: list[EncodedAnchor] = []

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        cache = PromptEmbedsCache()
        prompt_pairs: list[EncodedPromptPair] = []

        # get encoded latents for our prompts
        with torch.no_grad():
            neutral = ""
            for target in self.slider_config.targets:
                # build the cache
                for prompt in [
                    target.target_class,
                    target.positive,
                    target.negative,
                    neutral  # empty neutral
                ]:
                    if cache[prompt] is None:
                        cache[prompt] = self.sd.encode_prompt(prompt)
                for resolution in self.slider_config.resolutions:
                    width, height = resolution
                    only_erase = len(target.positive.strip()) == 0
                    only_enhance = len(target.negative.strip()) == 0

                    both = not only_erase and not only_enhance

                    if only_erase and only_enhance:
                        raise ValueError("target must have at least one of positive or negative or both")
                    # for slider we need to have an enhancer, an eraser, and then
                    # an inverse with negative weights to balance the network
                    # if we don't do this, we will get different contrast and focus.
                    # we only perform actions of enhancing and erasing on the negative
                    # todo work on way to do all of this in one shot

                    if both or only_erase:
                        prompt_pairs += [
                            # erase standard
                            EncodedPromptPair(
                                target_class=cache[target.target_class],
                                positive=cache[target.positive],
                                negative=cache[target.negative],
                                neutral=cache[neutral],
                                width=width,
                                height=height,
                                action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
                                multiplier=target.multiplier,
                                weight=target.weight
                            ),
                        ]
                    if both or only_enhance:
                        prompt_pairs += [
                            # enhance standard, swap pos neg
                            EncodedPromptPair(
                                target_class=cache[target.target_class],
                                positive=cache[target.negative],
                                negative=cache[target.positive],
                                neutral=cache[neutral],
                                width=width,
                                height=height,
                                action=ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE,
                                multiplier=target.multiplier,
                                weight=target.weight
                            ),
                        ]
                    if both:
                        prompt_pairs += [
                            # erase inverted
                            EncodedPromptPair(
                                target_class=cache[target.target_class],
                                positive=cache[target.negative],
                                negative=cache[target.positive],
                                neutral=cache[neutral],
                                width=width,
                                height=height,
                                action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
                                multiplier=target.multiplier * -1.0,
                                weight=target.weight
                            ),
                        ]
                        prompt_pairs += [
                            # enhance inverted
                            EncodedPromptPair(
                                target_class=cache[target.target_class],
                                positive=cache[target.positive],
                                negative=cache[target.negative],
                                neutral=cache[neutral],
                                width=width,
                                height=height,
                                action=ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE,
                                multiplier=target.multiplier * -1.0,
                                weight=target.weight
                            ),
                        ]

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

                anchor_pairs += [
                    EncodedAnchor(
                        prompt=cache[anchor.prompt],
                        neg_prompt=cache[anchor.neg_prompt],
                        multiplier=anchor.multiplier
                    )
                ]

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
        self.anchor_pairs = anchor_pairs
        flush()
        # end hook_before_train_loop

    def hook_train_loop(self, batch):
        dtype = get_torch_dtype(self.train_config.dtype)

        # get a random pair
        prompt_pair: EncodedPromptPair = self.prompt_pairs[
            torch.randint(0, len(self.prompt_pairs), (1,)).item()
        ]

        height = prompt_pair.height
        width = prompt_pair.width
        target_class = prompt_pair.target_class
        neutral = prompt_pair.neutral
        negative = prompt_pair.negative
        positive = prompt_pair.positive
        weight = prompt_pair.weight
        multiplier = prompt_pair.multiplier

        unet = self.sd.unet
        noise_scheduler = self.sd.noise_scheduler
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        loss_function = torch.nn.MSELoss()

        def get_noise_pred(p, n, gs, cts, dn):
            return self.sd.predict_noise(
                latents=dn,
                text_embeddings=train_tools.concat_prompt_embeddings(
                    p,  # unconditional
                    n,  # positive
                    self.train_config.batch_size,
                ),
                timestep=cts,
                guidance_scale=gs,
            )

        # set network multiplier
        self.network.multiplier = multiplier

        with torch.no_grad():
            self.sd.noise_scheduler.set_timesteps(
                self.train_config.max_denoising_steps, device=self.device_torch
            )

            self.optimizer.zero_grad()

            # ger a random number of steps
            timesteps_to = torch.randint(
                1, self.train_config.max_denoising_steps, (1,)
            ).item()

            # get noise
            noise = self.sd.get_latent_noise(
                pixel_height=height,
                pixel_width=width,
                batch_size=self.train_config.batch_size,
                noise_offset=self.train_config.noise_offset,
            ).to(self.device_torch, dtype=dtype)

            # get latents
            latents = noise * self.sd.noise_scheduler.init_noise_sigma
            latents = latents.to(self.device_torch, dtype=dtype)

            with self.network:
                assert self.network.is_active
                self.network.multiplier = multiplier
                denoised_latents = self.sd.diffuse_some_steps(
                    latents,  # pass simple noise latents
                    train_tools.concat_prompt_embeddings(
                        positive,  # unconditional
                        target_class,  # target
                        self.train_config.batch_size,
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / self.train_config.max_denoising_steps)
            ]

            positive_latents = get_noise_pred(
                positive, negative, 1, current_timestep, denoised_latents
            ).to("cpu", dtype=torch.float32)

            neutral_latents = get_noise_pred(
                positive, neutral, 1, current_timestep, denoised_latents
            ).to("cpu", dtype=torch.float32)

            unconditional_latents = get_noise_pred(
                positive, positive, 1, current_timestep, denoised_latents
            ).to("cpu", dtype=torch.float32)

        anchor_loss = None
        if len(self.anchor_pairs) > 0:
            # get a random anchor pair
            anchor: EncodedAnchor = self.anchor_pairs[
                torch.randint(0, len(self.anchor_pairs), (1,)).item()
            ]
            with torch.no_grad():
                anchor_target_noise = get_noise_pred(
                    anchor.prompt, anchor.neg_prompt, 1, current_timestep, denoised_latents
                ).to("cpu", dtype=torch.float32)
            with self.network:
                # anchor whatever weight  prompt pair is using
                pos_nem_mult = 1.0 if prompt_pair.multiplier > 0 else -1.0
                self.network.multiplier = anchor.multiplier * pos_nem_mult

                anchor_pred_noise = get_noise_pred(
                    anchor.prompt, anchor.neg_prompt, 1, current_timestep, denoised_latents
                ).to("cpu", dtype=torch.float32)

                self.network.multiplier = prompt_pair.multiplier

        with self.network:
            self.network.multiplier = prompt_pair.multiplier
            target_latents = get_noise_pred(
                positive, target_class, 1, current_timestep, denoised_latents
            ).to("cpu", dtype=torch.float32)

            # if self.logging_config.verbose:
            #     self.print("target_latents:", target_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        unconditional_latents.requires_grad = False
        if len(self.anchor_pairs) > 0:
            anchor_target_noise.requires_grad = False
            anchor_loss = loss_function(
                anchor_target_noise,
                anchor_pred_noise,
            )
        erase = prompt_pair.action == ACTION_TYPES_SLIDER.ERASE_NEGATIVE
        guidance_scale = 1.0

        offset = guidance_scale * (positive_latents - unconditional_latents)

        offset_neutral = neutral_latents
        if erase:
            offset_neutral -= offset
        else:
            # enhance
            offset_neutral += offset

        loss = loss_function(
            target_latents,
            offset_neutral,
        ) * weight

        loss_slide = loss.item()

        if anchor_loss is not None:
            loss += anchor_loss

        loss_float = loss.item()

        loss = loss.to(self.device_torch)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            target_latents,
            latents,
        )
        flush()

        # reset network
        self.network.multiplier = 1.0

        loss_dict = OrderedDict(
            {'loss': loss_float},
        )
        if anchor_loss is not None:
            loss_dict['sl_l'] = loss_slide
            loss_dict['an_l'] = anchor_loss.item()

        return loss_dict
        # end hook_train_loop
