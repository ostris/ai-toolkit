# ref:
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
import time
from collections import OrderedDict
import os

from toolkit.config_modules import SliderConfig
from toolkit.paths import REPOS_ROOT
import sys

sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import gc

import torch
from leco import train_util, model_util
from leco.prompt_util import PromptEmbedsCache
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


class TrainSliderProcess(BaseSDTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.slider_config = SliderConfig(**self.get_conf('slider', {}))

        self.prompt_cache = PromptEmbedsCache()
        self.prompt_pairs: list[EncodedPromptPair] = []

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        cache = PromptEmbedsCache()
        prompt_pairs: list[EncodedPromptPair] = []

        # get encoded latents for our prompts
        with torch.no_grad():
            neutral = ""
            for target in self.slider_config.targets:
                for resolution in self.slider_config.resolutions:
                    width, height = resolution
                    # build the cache
                    for prompt in [
                        target.target_class,
                        target.positive,
                        target.negative,
                        neutral  # empty neutral
                    ]:
                        if cache[prompt] == None:
                            cache[prompt] = train_util.encode_prompts(
                                self.sd.tokenizer, self.sd.text_encoder, [prompt]
                            )

                    # for slider we need to have an enhancer, an eraser, and then
                    # an inverse with negative weights to balance the network
                    # if we don't do this, we will get different contrast and focus.
                    # we only perform actions of enhancing and erasing on the negative
                    # todo work on way to do all of this in one shot
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

        # move to cpu to save vram
        # We don't need text encoder anymore, but keep it on cpu for sampling
        self.sd.text_encoder.to("cpu")
        self.prompt_cache = cache
        self.prompt_pairs = prompt_pairs
        flush()
        # end hook_before_train_loop

    def hook_train_loop(self):
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

        unet = self.sd.unet
        noise_scheduler = self.sd.noise_scheduler
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        loss_function = torch.nn.MSELoss()

        # set network multiplier
        self.network.multiplier = prompt_pair.multiplier

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
            noise = self.get_latent_noise(
                pixel_height=height,
                pixel_width=width,
            ).to(self.device_torch, dtype=dtype)

            # get latents
            latents = noise * self.sd.noise_scheduler.init_noise_sigma
            latents = latents.to(self.device_torch, dtype=dtype)

            with self.network:
                assert self.network.is_active
                # A little denoised one is returned
                denoised_latents = train_util.diffusion(
                    unet,
                    noise_scheduler,
                    latents,  # pass simple noise latents
                    train_util.concat_embeddings(
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

            # with network: 0 weight LoRA is enabled outside "with network:"
            positive_latents = train_util.predict_noise(  # positive_latents
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    positive,  # unconditional
                    negative,  # positive
                    self.train_config.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            neutral_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    positive,  # unconditional
                    neutral,  # neutral
                    self.train_config.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            unconditional_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    positive,  # unconditional
                    positive,  # unconditional
                    self.train_config.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

        with self.network:
            target_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    positive,  # unconditional
                    target_class,  # target
                    self.train_config.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            # if self.logging_config.verbose:
            #     self.print("target_latents:", target_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        unconditional_latents.requires_grad = False

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

        loss_float = loss.item()
        if self.train_config.optimizer.startswith('dadaptation'):
            learning_rate = (
                    optimizer.param_groups[0]["d"] *
                    optimizer.param_groups[0]["lr"]
            )
        else:
            learning_rate = optimizer.param_groups[0]['lr']

        self.progress_bar.set_postfix_str(f"lr: {learning_rate:.1e} loss: {loss.item():.3e}")

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

        return loss_float
        # end hook_train_loop
