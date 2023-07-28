# ref:
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
import time
from collections import OrderedDict
import os
from typing import Optional

import numpy as np
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from toolkit.config_modules import SliderConfig
from toolkit.layers import ReductionKernel
from toolkit.paths import REPOS_ROOT
import sys

from toolkit.stable_diffusion_model import PromptEmbeds
from toolkit.train_pipelines import TransferStableDiffusionXLPipeline

sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import gc
from toolkit import train_tools

import torch
from leco import train_util, model_util
from .BaseSDTrainProcess import BaseSDTrainProcess, StableDiffusion


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class RescaleConfig:
    def __init__(
            self,
            **kwargs
    ):
        self.from_resolution = kwargs.get('from_resolution', 512)
        self.scale = kwargs.get('scale', 0.5)
        self.prompt_file = kwargs.get('prompt_file', None)
        self.prompt_tensors = kwargs.get('prompt_tensors', None)
        self.to_resolution = kwargs.get('to_resolution', int(self.from_resolution * self.scale))

        if self.prompt_file is None:
            raise ValueError("prompt_file is required")


class PromptEmbedsCache:
    prompts: dict[str, PromptEmbeds] = {}

    def __setitem__(self, __name: str, __value: PromptEmbeds) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str) -> Optional[PromptEmbeds]:
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


class TrainSDRescaleProcess(BaseSDTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        # pass our custom pipeline to super so it sets it up
        super().__init__(process_id, job, config, custom_pipeline=TransferStableDiffusionXLPipeline)
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.prompt_cache = PromptEmbedsCache()
        self.rescale_config = RescaleConfig(**self.get_conf('rescale', required=True))
        self.reduce_size_fn = ReductionKernel(
            in_channels=4,
            kernel_size=int(self.rescale_config.from_resolution // self.rescale_config.to_resolution),
            dtype=get_torch_dtype(self.train_config.dtype),
            device=self.device_torch,
        )
        self.prompt_txt_list = []

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        self.print(f"Loading prompt file from {self.rescale_config.prompt_file}")

        # read line by line from file
        with open(self.rescale_config.prompt_file, 'r') as f:
            self.prompt_txt_list = f.readlines()
            # clean empty lines
            self.prompt_txt_list = [line.strip() for line in self.prompt_txt_list if len(line.strip()) > 0]

        self.print(f"Loaded {len(self.prompt_txt_list)} prompts. Encoding them..")

        cache = PromptEmbedsCache()

        # get encoded latents for our prompts
        with torch.no_grad():
            if self.rescale_config.prompt_tensors is not None:
                # check to see if it exists
                if os.path.exists(self.rescale_config.prompt_tensors):
                    # load it.
                    self.print(f"Loading prompt tensors from {self.rescale_config.prompt_tensors}")
                    prompt_tensors = load_file(self.rescale_config.prompt_tensors, device='cpu')
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
                neutral = ""
                # encode neutral
                cache[neutral] = self.sd.encode_prompt(neutral)
                for prompt in tqdm(self.prompt_txt_list, desc="Encoding prompts", leave=False):
                    # build the cache
                    if cache[prompt] is None:
                        cache[prompt] = self.sd.encode_prompt(prompt).to(device="cpu", dtype=torch.float32)

                if self.rescale_config.prompt_tensors:
                    print(f"Saving prompt tensors to {self.rescale_config.prompt_tensors}")
                    state_dict = {}
                    for prompt_txt, prompt_embeds in cache.prompts.items():
                        state_dict[f"te:{prompt_txt}"] = prompt_embeds.text_embeds.to("cpu",
                                                                                      dtype=get_torch_dtype('fp16'))
                        if prompt_embeds.pooled_embeds is not None:
                            state_dict[f"pe:{prompt_txt}"] = prompt_embeds.pooled_embeds.to("cpu",
                                                                                            dtype=get_torch_dtype(
                                                                                                'fp16'))
                    save_file(state_dict, self.rescale_config.prompt_tensors)

            self.print("Encoding complete.")

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

        # get random encoded prompt from cache
        prompt_txt = self.prompt_txt_list[
            torch.randint(0, len(self.prompt_txt_list), (1,)).item()
        ]
        prompt = self.prompt_cache[prompt_txt].to(device=self.device_torch, dtype=dtype)
        prompt.text_embeds.to(device=self.device_torch, dtype=dtype)
        neutral = self.prompt_cache[""].to(device=self.device_torch, dtype=dtype)
        neutral.text_embeds.to(device=self.device_torch, dtype=dtype)
        if hasattr(prompt, 'pooled_embeds') \
                and hasattr(neutral, 'pooled_embeds') \
                and prompt.pooled_embeds is not None \
                and neutral.pooled_embeds is not None:
            prompt.pooled_embeds.to(device=self.device_torch, dtype=dtype)
            neutral.pooled_embeds.to(device=self.device_torch, dtype=dtype)

        if prompt is None:
            raise ValueError(f"Prompt {prompt_txt} is not in cache")

        loss_function = torch.nn.MSELoss()

        with torch.no_grad():
            # self.sd.noise_scheduler.set_timesteps(
            #     self.train_config.max_denoising_steps, device=self.device_torch
            # )

            self.optimizer.zero_grad()

            # # ger a random number of steps
            timesteps_to = torch.randint(
                1, self.train_config.max_denoising_steps, (1,)
            ).item()

            # get noise
            latents = self.get_latent_noise(
                pixel_height=self.rescale_config.from_resolution,
                pixel_width=self.rescale_config.from_resolution,
            ).to(self.device_torch, dtype=dtype)

            self.sd.pipeline.to(self.device_torch)
            torch.set_default_device(self.device_torch)

            # turn off progress bar
            self.sd.pipeline.set_progress_bar_config(disable=True)

            # get random guidance scale from 1.0 to 10.0
            guidance_scale = torch.rand(1).item() * 9.0 + 1.0

            loss_arr = []


            max_len_timestep_str = len(str(self.train_config.max_denoising_steps))
            # pad with spaces
            timestep_str = str(timesteps_to).rjust(max_len_timestep_str, " ")
            new_description = f"{self.job.name} ts: {timestep_str}"
            self.progress_bar.set_description(new_description)

            def pre_condition_callback(target_pred, input_latents):
                # handle any manipulations before feeding to our network
                reduced_pred = self.reduce_size_fn(target_pred)
                reduced_latents = self.reduce_size_fn(input_latents)
                self.optimizer.zero_grad()
                return reduced_pred, reduced_latents

            def each_step_callback(noise_target, noise_train_pred):
                noise_target.requires_grad = False
                loss = loss_function(noise_target, noise_train_pred)
                loss_arr.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            # run the pipeline
            self.sd.pipeline.transfer_diffuse(
                num_inference_steps=timesteps_to,
                latents=latents,
                prompt_embeds=prompt.text_embeds,
                negative_prompt_embeds=neutral.text_embeds,
                pooled_prompt_embeds=prompt.pooled_embeds,
                negative_pooled_prompt_embeds=neutral.pooled_embeds,
                output_type="latent",
                num_images_per_prompt=self.train_config.batch_size,
                guidance_scale=guidance_scale,
                network=self.network,
                target_unet=self.sd.unet,
                pre_condition_callback=pre_condition_callback,
                each_step_callback=each_step_callback,
            )

        flush()

        # reset network
        self.network.multiplier = 1.0

        # average losses
        s = 0
        for num in loss_arr:
            s += num

        avg_loss = s / len(loss_arr)

        loss_dict = OrderedDict(
            {'loss': avg_loss},
        )

        return loss_dict
        # end hook_train_loop
