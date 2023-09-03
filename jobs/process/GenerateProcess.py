import gc
import os
from collections import OrderedDict
from typing import ForwardRef, List

import torch
from safetensors.torch import save_file, load_file

from jobs.process.BaseProcess import BaseProcess
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors, add_model_hash_to_meta, \
    add_base_model_info_to_meta
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
import random


class GenerateConfig:

    def __init__(self, **kwargs):
        self.prompts: List[str]
        self.sampler = kwargs.get('sampler', 'ddpm')
        self.width = kwargs.get('width', 512)
        self.height = kwargs.get('height', 512)
        self.neg = kwargs.get('neg', '')
        self.seed = kwargs.get('seed', -1)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)
        self.prompt_2 = kwargs.get('prompt_2', None)
        self.neg_2 = kwargs.get('neg_2', None)
        self.prompts = kwargs.get('prompts', None)
        self.guidance_rescale = kwargs.get('guidance_rescale', 0.0)
        self.ext = kwargs.get('ext', 'png')
        self.prompt_file = kwargs.get('prompt_file', False)
        if self.prompts is None:
            raise ValueError("Prompts must be set")
        if isinstance(self.prompts, str):
            if os.path.exists(self.prompts):
                with open(self.prompts, 'r', encoding='utf-8') as f:
                    self.prompts = f.read().splitlines()
                    self.prompts = [p.strip() for p in self.prompts if len(p.strip()) > 0]
            else:
                raise ValueError("Prompts file does not exist, put in list if you want to use a list of prompts")

        if kwargs.get('shuffle', False):
            # shuffle the prompts
            random.shuffle(self.prompts)


class GenerateProcess(BaseProcess):
    process_id: int
    config: OrderedDict
    progress_bar: ForwardRef('tqdm') = None
    sd: StableDiffusion

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.output_folder = self.get_conf('output_folder', required=True)
        self.model_config = ModelConfig(**self.get_conf('model', required=True))
        self.device = self.get_conf('device', self.job.device)
        self.generate_config = GenerateConfig(**self.get_conf('generate', required=True))

        self.progress_bar = None
        self.sd = StableDiffusion(
            device=self.device,
            model_config=self.model_config,
            dtype=self.model_config.dtype,
        )
        print(f"Using device {self.device}")

    def run(self):
        super().run()
        print("Loading model...")
        self.sd.load_model()

        print(f"Generating {len(self.generate_config.prompts)} images")
        # build prompt image configs
        prompt_image_configs = []
        for prompt in self.generate_config.prompts:
            prompt_image_configs.append(GenerateImageConfig(
                prompt=prompt,
                prompt_2=self.generate_config.prompt_2,
                width=self.generate_config.width,
                height=self.generate_config.height,
                num_inference_steps=self.generate_config.sample_steps,
                guidance_scale=self.generate_config.guidance_scale,
                negative_prompt=self.generate_config.neg,
                negative_prompt_2=self.generate_config.neg_2,
                seed=self.generate_config.seed,
                guidance_rescale=self.generate_config.guidance_rescale,
                output_ext=self.generate_config.ext,
                output_folder=self.output_folder,
                add_prompt_file=self.generate_config.prompt_file
            ))
        # generate images
        self.sd.generate_images(prompt_image_configs, sampler=self.generate_config.sampler)

        print("Done generating images")
        # cleanup
        del self.sd
        gc.collect()
        torch.cuda.empty_cache()
