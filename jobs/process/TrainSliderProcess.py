# ref:
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
import random
import time
from collections import OrderedDict
import os
from typing import Optional

from safetensors.torch import save_file, load_file
from tqdm import tqdm

from toolkit.config_modules import SliderConfig
from toolkit.paths import REPOS_ROOT
import sys

from toolkit.stable_diffusion_model import PromptEmbeds

sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import gc
from toolkit import train_tools

import torch
from leco import train_util, model_util
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
            positive_target,
            positive_target_with_neutral,
            negative_target,
            negative_target_with_neutral,
            neutral,
            both_targets,
            empty_prompt,
            action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
            multiplier=1.0,
            weight=1.0
    ):
        self.target_class = target_class
        self.positive_target = positive_target
        self.positive_target_with_neutral = positive_target_with_neutral
        self.negative_target = negative_target
        self.negative_target_with_neutral = negative_target_with_neutral
        self.neutral = neutral
        self.empty_prompt = empty_prompt
        self.both_targets = both_targets
        self.multiplier = multiplier
        self.action: int = action
        self.weight = weight

    # simulate torch to for tensors
    def to(self, *args, **kwargs):
        self.target_class = self.target_class.to(*args, **kwargs)
        self.positive_target = self.positive_target.to(*args, **kwargs)
        self.positive_target_with_neutral = self.positive_target_with_neutral.to(*args, **kwargs)
        self.negative_target = self.negative_target.to(*args, **kwargs)
        self.negative_target_with_neutral = self.negative_target_with_neutral.to(*args, **kwargs)
        self.neutral = self.neutral.to(*args, **kwargs)
        self.empty_prompt = self.empty_prompt.to(*args, **kwargs)
        self.both_targets = self.both_targets.to(*args, **kwargs)
        return self


class PromptEmbedsCache:
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

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        self.print(f"Loading prompt file from {self.slider_config.prompt_file}")

        # read line by line from file
        with open(self.slider_config.prompt_file, 'r') as f:
            self.prompt_txt_list = f.readlines()
            # clean empty lines
            self.prompt_txt_list = [line.strip() for line in self.prompt_txt_list if len(line.strip()) > 0]

        self.print(f"Loaded {len(self.prompt_txt_list)} prompts. Encoding them..")

        cache = PromptEmbedsCache()

        if not self.slider_config.prompt_tensors:
            # shuffle
            random.shuffle(self.prompt_txt_list)
            # trim to max steps
            self.prompt_txt_list = self.prompt_txt_list[:self.train_config.steps]
            # trim list to our max steps


        # get encoded latents for our prompts
        with torch.no_grad():
            if self.slider_config.prompt_tensors is not None:
                # check to see if it exists
                if os.path.exists(self.slider_config.prompt_tensors):
                    # load it.
                    self.print(f"Loading prompt tensors from {self.slider_config.prompt_tensors}")
                    prompt_tensors = load_file(self.slider_config.prompt_tensors, device='cpu')
                    # add them to the cache
                    for prompt_txt, prompt_tensor in tqdm(prompt_tensors.items(), desc="Loading prompts", leave=False):
                        if prompt_txt.startswith("te:"):
                            prompt = prompt_txt[3:]
                            # text_embeds
                            text_embeds = prompt_tensor
                            pooled_embeds = None
                            # find pool embeds
                            if f"pe:{prompt}" in prompt_tensors:
                                pooled_embeds = prompt_tensors[f"pe:{prompt}"]

                            # make it
                            prompt_embeds = PromptEmbeds([text_embeds, pooled_embeds])
                            cache[prompt] = prompt_embeds.to(device='cpu', dtype=torch.float32)

            if len(cache.prompts) == 0:
                print("Prompt tensors not found. Encoding prompts..")
                empty_prompt = ""
                # encode empty_prompt
                cache[empty_prompt] = self.sd.encode_prompt(empty_prompt)

                for neutral in tqdm(self.prompt_txt_list, desc="Encoding prompts", leave=False):
                    for target in self.slider_config.targets:
                        prompt_list = [
                            f"{target.target_class}", # target_class
                            f"{target.positive}",  # positive_target
                            f"{target.positive} {neutral}",  # positive_target with neutral
                            f"{target.negative}",  # negative_target
                            f"{target.negative} {neutral}",  # negative_target with neutral
                            f"{neutral}",  # neutral
                            f"{target.positive} {target.negative}",  # both targets
                            f"{target.negative} {target.positive}",  # both targets
                        ]
                        for p in prompt_list:
                            # build the cache
                            if cache[p] is None:
                                cache[p] = self.sd.encode_prompt(p).to(device="cpu", dtype=torch.float32)

                    erase_negative = len(target.positive.strip()) == 0
                    enhance_positive = len(target.negative.strip()) == 0

                    both = not erase_negative and not enhance_positive

                    if erase_negative and enhance_positive:
                        raise ValueError("target must have at least one of positive or negative or both")
                    # for slider we need to have an enhancer, an eraser, and then
                    # an inverse with negative weights to balance the network
                    # if we don't do this, we will get different contrast and focus.
                    # we only perform actions of enhancing and erasing on the negative
                    # todo work on way to do all of this in one shot
                if self.slider_config.prompt_tensors:
                    print(f"Saving prompt tensors to {self.slider_config.prompt_tensors}")
                    state_dict = {}
                    for prompt_txt, prompt_embeds in cache.prompts.items():
                        state_dict[f"te:{prompt_txt}"] = prompt_embeds.text_embeds.to("cpu",
                                                                                      dtype=get_torch_dtype('fp16'))
                        if prompt_embeds.pooled_embeds is not None:
                            state_dict[f"pe:{prompt_txt}"] = prompt_embeds.pooled_embeds.to("cpu",
                                                                                            dtype=get_torch_dtype(
                                                                                                'fp16'))
                    save_file(state_dict, self.slider_config.prompt_tensors)

            prompt_pairs = []
            for neutral in tqdm(self.prompt_txt_list, desc="Encoding prompts", leave=False):
                for target in self.slider_config.targets:

                    if both or erase_negative:
                        prompt_pairs += [
                            # erase standard
                            EncodedPromptPair(
                                target_class=cache[target.target_class],
                                positive_target=cache[f"{target.positive}"],
                                positive_target_with_neutral=cache[f"{target.positive} {neutral}"],
                                negative_target=cache[f"{target.negative}"],
                                negative_target_with_neutral=cache[f"{target.negative} {neutral}"],
                                neutral=cache[neutral],
                                action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
                                multiplier=target.multiplier,
                                both_targets=cache[f"{target.positive} {target.negative}"],
                                empty_prompt=cache[""],
                                weight=target.weight
                            ),
                        ]
                    if both or enhance_positive:
                        prompt_pairs += [
                            # enhance standard, swap pos neg
                            EncodedPromptPair(
                                target_class=cache[target.target_class],
                                positive_target=cache[f"{target.negative}"],
                                positive_target_with_neutral=cache[f"{target.negative} {neutral}"],
                                negative_target=cache[f"{target.positive}"],
                                negative_target_with_neutral=cache[f"{target.positive} {neutral}"],
                                neutral=cache[neutral],
                                action=ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE,
                                multiplier=target.multiplier,
                                both_targets=cache[f"{target.positive} {target.negative}"],
                                empty_prompt=cache[""],
                                weight=target.weight
                            ),
                        ]
                    if both or enhance_positive:
                        prompt_pairs += [
                            # erase inverted
                            EncodedPromptPair(
                                target_class=cache[target.target_class],
                                positive_target=cache[f"{target.negative}"],
                                positive_target_with_neutral=cache[f"{target.negative} {neutral}"],
                                negative_target=cache[f"{target.positive}"],
                                negative_target_with_neutral=cache[f"{target.positive} {neutral}"],
                                neutral=cache[neutral],
                                action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
                                both_targets=cache[f"{target.positive} {target.negative}"],
                                empty_prompt=cache[""],
                                multiplier=target.multiplier * -1.0,
                                weight=target.weight
                            ),
                        ]
                    if both or erase_negative:
                        prompt_pairs += [
                            # enhance inverted
                            EncodedPromptPair(
                                target_class=cache[target.target_class],
                                positive_target=cache[f"{target.positive}"],
                                positive_target_with_neutral=cache[f"{target.positive} {neutral}"],
                                negative_target=cache[f"{target.negative}"],
                                negative_target_with_neutral=cache[f"{target.negative} {neutral}"],
                                both_targets=cache[f"{target.positive} {target.negative}"],
                                neutral=cache[neutral],
                                action=ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE,
                                empty_prompt=cache[""],
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
            # self.print("Encoding complete. Building prompt pairs..")
            # for neutral in self.prompt_txt_list:
            #     for target in self.slider_config.targets:
            #         both_prompts_list = [
            #             f"{target.positive} {target.negative}",
            #             f"{target.negative} {target.positive}",
            #         ]
            #         # randomly pick one of the both prompts to prevent bias
            #         both_prompts = both_prompts_list[torch.randint(0, 2, (1,)).item()]
            #
            #         prompt_pair = EncodedPromptPair(
            #             positive_target=cache[f"{target.positive}"],
            #             positive_target_with_neutral=cache[f"{target.positive} {neutral}"],
            #             negative_target=cache[f"{target.negative}"],
            #             negative_target_with_neutral=cache[f"{target.negative} {neutral}"],
            #             neutral=cache[neutral],
            #             both_targets=cache[both_prompts],
            #             empty_prompt=cache[""],
            #             target_class=cache[f"{target.target_class}"],
            #             weight=target.weight,
            #         ).to(device="cpu", dtype=torch.float32)
            #         self.prompt_pairs.append(prompt_pair)

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

    def hook_train_loop(self):
        dtype = get_torch_dtype(self.train_config.dtype)

        # get random multiplier between 1 and 3
        rand_weight = torch.rand((1,)).item() * 2 + 1

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

        target_class = prompt_pair.target_class
        neutral = prompt_pair.neutral
        negative = prompt_pair.negative_target
        positive = prompt_pair.positive_target
        weight = prompt_pair.weight
        multiplier = prompt_pair.multiplier

        unet = self.sd.unet
        noise_scheduler = self.sd.noise_scheduler
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        loss_function = torch.nn.MSELoss()

        def get_noise_pred(p, n, gs, cts, dn):
            return self.predict_noise(
                latents=dn,
                text_embeddings=train_tools.concat_prompt_embeddings(
                    p,  # negative prompt
                    n,  # positive prompt
                    self.train_config.batch_size,
                ),
                timestep=cts,
                guidance_scale=gs,
            )

        # set network multiplier
        self.network.multiplier = multiplier * rand_weight

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
                self.network.multiplier = multiplier * rand_weight
                denoised_latents = self.diffuse_some_steps(
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
                self.network.multiplier = anchor.multiplier * pos_nem_mult * rand_weight

                anchor_pred_noise = get_noise_pred(
                    anchor.prompt, anchor.neg_prompt, 1, current_timestep, denoised_latents
                ).to("cpu", dtype=torch.float32)

                self.network.multiplier = prompt_pair.multiplier * rand_weight

        with self.network:
            self.network.multiplier = prompt_pair.multiplier * rand_weight
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
        # move back to cpu
        prompt_pair.to("cpu")
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
