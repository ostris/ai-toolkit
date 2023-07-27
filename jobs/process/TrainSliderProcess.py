# ref:
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
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
            positive_target,
            positive_target_with_neutral,
            negative_target,
            negative_target_with_neutral,
            neutral,
            both_targets,
            empty_prompt
    ):
        self.positive_target = positive_target
        self.positive_target_with_neutral = positive_target_with_neutral
        self.negative_target = negative_target
        self.negative_target_with_neutral = negative_target_with_neutral
        self.neutral = neutral
        self.empty_prompt = empty_prompt
        self.both_targets = both_targets

    # simulate torch to for tensors
    def to(self, *args, **kwargs):
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

        # get encoded latents for our prompts
        with torch.no_grad():
            if self.slider_config.prompt_tensors is not None:
                # check to see if it exists
                if os.path.exists(self.slider_config.prompt_tensors):
                    # load it.
                    self.print(f"Loading prompt tensors from {self.slider_config.prompt_tensors}")
                    prompt_tensors = load_file(self.slider_config.prompt_tensors, device='cpu')
                    # add them to the cache
                    for prompt_txt, prompt_tensor in prompt_tensors.items():
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

            self.print("Encoding complete. Building prompt pairs..")
            for neutral in self.prompt_txt_list:
                for target in self.slider_config.targets:
                    both_prompts_list = [
                        f"{target.positive} {target.negative}",
                        f"{target.negative} {target.positive}",
                    ]
                    # randomly pick one of the both prompts to prevent bias
                    both_prompts = both_prompts_list[torch.randint(0, 2, (1,)).item()]

                    prompt_pair = EncodedPromptPair(
                        positive_target=cache[f"{target.positive}"],
                        positive_target_with_neutral=cache[f"{target.positive} {neutral}"],
                        negative_target=cache[f"{target.negative}"],
                        negative_target_with_neutral=cache[f"{target.negative} {neutral}"],
                        neutral=cache[neutral],
                        both_targets=cache[both_prompts],
                        empty_prompt=cache[""],
                    ).to(device="cpu", dtype=torch.float32)
                    self.prompt_pairs.append(prompt_pair)

        # move to cpu to save vram
        # We don't need text encoder anymore, but keep it on cpu for sampling
        # if text encoder is list
        if isinstance(self.sd.text_encoder, list):
            for encoder in self.sd.text_encoder:
                encoder.to("cpu")
        else:
            self.sd.text_encoder.to("cpu")
        self.prompt_cache = cache

        flush()
        # end hook_before_train_loop

    def hook_train_loop(self):
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

        unet = self.sd.unet
        noise_scheduler = self.sd.noise_scheduler
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        loss_function = torch.nn.MSELoss()

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

            denoised_fraction = timesteps_to / (self.train_config.max_denoising_steps + 1)
            self.sd.pipeline.to(self.device_torch)
            torch.set_default_device(self.device_torch)
            self.sd.pipeline.set_progress_bar_config(disable=True)

            # get generate semi denoised latents without network
            # only neutrap in positive and both targets in negative
            assert not self.network.is_active
            # denoised_latents = self.sd.pipeline(
            #     num_inference_steps=self.train_config.max_denoising_steps,
            #     denoising_end=denoised_fraction,
            #     latents=latents,
            #     prompt_embeds=prompt_pair.neutral.text_embeds,
            #     negative_prompt_embeds=prompt_pair.both_targets.text_embeds,
            #     pooled_prompt_embeds=prompt_pair.neutral.pooled_embeds,
            #     negative_pooled_prompt_embeds=prompt_pair.both_targets.pooled_embeds,
            #     output_type="latent",
            #     num_images_per_prompt=self.train_config.batch_size,
            #     guidance_scale=3,
            # ).images.to(self.device_torch, dtype=dtype)

            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / self.train_config.max_denoising_steps)
            ]
            denoised_latents = noise

            # neutral prediction
            neutral_noise_prediction = self.sd.pipeline.predict_noise(
                latents=denoised_latents,
                prompt_embeds=prompt_pair.neutral.text_embeds,
                negative_prompt_embeds=prompt_pair.empty_prompt.text_embeds,
                pooled_prompt_embeds=prompt_pair.neutral.pooled_embeds,
                negative_pooled_prompt_embeds=prompt_pair.both_targets.pooled_embeds,
                timestep=current_timestep,
                guidance_scale=1,
                num_images_per_prompt=self.train_config.batch_size,
                num_inference_steps=1000,
            )

            # with self.network:
            #     assert self.network.is_active
            #     self.network.multiplier = 1.0
            #
            #     positive_pos_noise_prediction = self.sd.pipeline.predict_noise(
            #         latents=denoised_latents,
            #         prompt_embeds=prompt_pair.positive_target_with_neutral.text_embeds,
            #         negative_prompt_embeds=prompt_pair.negative_target.text_embeds,
            #         pooled_prompt_embeds=prompt_pair.positive_target_with_neutral.pooled_embeds,
            #         negative_pooled_prompt_embeds=prompt_pair.negative_target.pooled_embeds,
            #         timestep=current_timestep,
            #         guidance_scale=1,
            #         num_images_per_prompt=self.train_config.batch_size,
            #         num_inference_steps=1000
            #     )
            #
            #     self.network.multiplier = -1.0
            #
            #     negative_neg_noise_prediction = self.sd.pipeline.predict_noise(
            #         latents=denoised_latents,
            #         prompt_embeds=prompt_pair.negative_target_with_neutral.text_embeds,
            #         negative_prompt_embeds=prompt_pair.positive_target.text_embeds,
            #         pooled_prompt_embeds=prompt_pair.negative_target_with_neutral.pooled_embeds,
            #         negative_pooled_prompt_embeds=prompt_pair.positive_target.pooled_embeds,
            #         timestep=current_timestep,
            #         guidance_scale=1,
            #         num_images_per_prompt=self.train_config.batch_size,
            #         num_inference_steps=1000
            #     )

        # start grads
        self.optimizer.zero_grad()

        multiplier = 5.0

        # predict postiitive
        with self.network:
            assert self.network.is_active
            self.network.multiplier = multiplier * 1.0

            # positive_pos_noise_prediction = self.sd.pipeline.predict_noise(
            #     latents=denoised_latents,
            #     prompt_embeds=prompt_pair.positive_target_with_neutral.text_embeds,
            #     negative_prompt_embeds=prompt_pair.negative_target.text_embeds,
            #     pooled_prompt_embeds=prompt_pair.positive_target_with_neutral.pooled_embeds,
            #     negative_pooled_prompt_embeds=prompt_pair.negative_target.pooled_embeds,
            #     timestep=current_timestep,
            #     guidance_scale=1,
            #     num_images_per_prompt=self.train_config.batch_size,
            #     num_inference_steps=self.train_config.max_denoising_steps,
            # )

            negative_pos_noise_prediction = self.sd.pipeline.predict_noise(
                latents=denoised_latents,
                prompt_embeds=prompt_pair.negative_target_with_neutral.text_embeds,
                negative_prompt_embeds=prompt_pair.positive_target.text_embeds,
                pooled_prompt_embeds=prompt_pair.negative_target_with_neutral.pooled_embeds,
                negative_pooled_prompt_embeds=prompt_pair.positive_target.pooled_embeds,
                timestep=current_timestep,
                guidance_scale=1,
                num_images_per_prompt=self.train_config.batch_size,
                num_inference_steps=1000,
            )

            self.network.multiplier = multiplier * -1.0

            positive_neg_noise_prediction = self.sd.pipeline.predict_noise(
                latents=denoised_latents,
                prompt_embeds=prompt_pair.positive_target_with_neutral.text_embeds,
                negative_prompt_embeds=prompt_pair.negative_target.text_embeds,
                pooled_prompt_embeds=prompt_pair.positive_target_with_neutral.pooled_embeds,
                negative_pooled_prompt_embeds=prompt_pair.negative_target.pooled_embeds,
                timestep=current_timestep,
                guidance_scale=1,
                num_images_per_prompt=self.train_config.batch_size,
                num_inference_steps=1000,
            )

            # negative_neg_noise_prediction = self.sd.pipeline.predict_noise(
            #     latents=denoised_latents,
            #     prompt_embeds=prompt_pair.negative_target_with_neutral.text_embeds,
            #     negative_prompt_embeds=prompt_pair.positive_target.text_embeds,
            #     pooled_prompt_embeds=prompt_pair.negative_target_with_neutral.pooled_embeds,
            #     negative_pooled_prompt_embeds=prompt_pair.positive_target.pooled_embeds,
            #     timestep=current_timestep,
            #     guidance_scale=1,
            #     num_images_per_prompt=self.train_config.batch_size,
            #     num_inference_steps=self.train_config.max_denoising_steps,
            # )

            self.network.multiplier = 1.0

        neutral_noise_prediction.requires_grad = False
        # positive_pos_noise_prediction.requires_grad = False
        # negative_neg_noise_prediction.requires_grad = False

        # calculate loss
        loss_shrink_pos_neg = loss_function(
            negative_pos_noise_prediction,
            neutral_noise_prediction,
        )

        loss_shrink_neg_pos = loss_function(
            positive_neg_noise_prediction,
            negative_pos_noise_prediction,
        )

        loss = loss_shrink_pos_neg + loss_shrink_neg_pos

        loss_float = loss.item()

        loss = loss.to(self.device_torch)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            denoised_latents,
            positive_neg_noise_prediction,
            negative_pos_noise_prediction,
            neutral_noise_prediction,
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

        return loss_dict
        # end hook_train_loop
