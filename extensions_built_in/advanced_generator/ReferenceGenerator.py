import os
import random
from collections import OrderedDict
from typing import List

import numpy as np
from PIL import Image
from diffusers import T2IAdapter
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLAdapterPipeline, StableDiffusionAdapterPipeline
from tqdm import tqdm

from toolkit.config_modules import ModelConfig, GenerateImageConfig, preprocess_dataset_raw_config, DatasetConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from toolkit.sampler import get_sampler
from toolkit.stable_diffusion_model import StableDiffusion
import gc
import torch
from jobs.process import BaseExtensionProcess
from toolkit.data_loader import get_dataloader_from_datasets
from toolkit.train_tools import get_torch_dtype
from controlnet_aux.midas import MidasDetector
from diffusers.utils import load_image


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class GenerateConfig:

    def __init__(self, **kwargs):
        self.prompts: List[str]
        self.sampler = kwargs.get('sampler', 'ddpm')
        self.neg = kwargs.get('neg', '')
        self.seed = kwargs.get('seed', -1)
        self.walk_seed = kwargs.get('walk_seed', False)
        self.t2i_adapter_path = kwargs.get('t2i_adapter_path', None)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)
        self.prompt_2 = kwargs.get('prompt_2', None)
        self.neg_2 = kwargs.get('neg_2', None)
        self.prompts = kwargs.get('prompts', None)
        self.guidance_rescale = kwargs.get('guidance_rescale', 0.0)
        self.ext = kwargs.get('ext', 'png')
        self.adapter_conditioning_scale = kwargs.get('adapter_conditioning_scale', 1.0)
        if kwargs.get('shuffle', False):
            # shuffle the prompts
            random.shuffle(self.prompts)


class ReferenceGenerator(BaseExtensionProcess):

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.output_folder = self.get_conf('output_folder', required=True)
        self.device = self.get_conf('device', 'cuda')
        self.model_config = ModelConfig(**self.get_conf('model', required=True))
        self.generate_config = GenerateConfig(**self.get_conf('generate', required=True))
        self.is_latents_cached = True
        raw_datasets = self.get_conf('datasets', None)
        if raw_datasets is not None and len(raw_datasets) > 0:
            raw_datasets = preprocess_dataset_raw_config(raw_datasets)
        self.datasets = None
        self.datasets_reg = None
        self.dtype = self.get_conf('dtype', 'float16')
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.params = []
        if raw_datasets is not None and len(raw_datasets) > 0:
            for raw_dataset in raw_datasets:
                dataset = DatasetConfig(**raw_dataset)
                is_caching = dataset.cache_latents or dataset.cache_latents_to_disk
                if not is_caching:
                    self.is_latents_cached = False
                if dataset.is_reg:
                    if self.datasets_reg is None:
                        self.datasets_reg = []
                    self.datasets_reg.append(dataset)
                else:
                    if self.datasets is None:
                        self.datasets = []
                    self.datasets.append(dataset)

        self.progress_bar = None
        self.sd = StableDiffusion(
            device=self.device,
            model_config=self.model_config,
            dtype=self.dtype,
        )
        print(f"Using device {self.device}")
        self.data_loader: DataLoader = None
        self.adapter: T2IAdapter = None

    def run(self):
        super().run()
        print("Loading model...")
        self.sd.load_model()
        device = torch.device(self.device)

        if self.generate_config.t2i_adapter_path is not None:
            self.adapter = T2IAdapter.from_pretrained(
                self.generate_config.t2i_adapter_path,
                torch_dtype=self.torch_dtype,
                varient="fp16"
            ).to(device)

        midas_depth = MidasDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
        ).to(device)

        if self.model_config.is_xl:
            pipe = StableDiffusionXLAdapterPipeline(
                vae=self.sd.vae,
                unet=self.sd.unet,
                text_encoder=self.sd.text_encoder[0],
                text_encoder_2=self.sd.text_encoder[1],
                tokenizer=self.sd.tokenizer[0],
                tokenizer_2=self.sd.tokenizer[1],
                scheduler=get_sampler(self.generate_config.sampler),
                adapter=self.adapter,
            ).to(device, dtype=self.torch_dtype)
        else:
            pipe = StableDiffusionAdapterPipeline(
                vae=self.sd.vae,
                unet=self.sd.unet,
                text_encoder=self.sd.text_encoder,
                tokenizer=self.sd.tokenizer,
                scheduler=get_sampler(self.generate_config.sampler),
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                adapter=self.adapter,
            ).to(device, dtype=self.torch_dtype)
        pipe.set_progress_bar_config(disable=True)

        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        # midas_depth = torch.compile(midas_depth, mode="reduce-overhead", fullgraph=True)

        self.data_loader = get_dataloader_from_datasets(self.datasets, 1, self.sd)

        num_batches = len(self.data_loader)
        pbar = tqdm(total=num_batches, desc="Generating images")
        seed = self.generate_config.seed
        # load images from datasets, use tqdm
        for i, batch in enumerate(self.data_loader):
            batch: DataLoaderBatchDTO = batch

            file_item: FileItemDTO = batch.file_items[0]
            img_path = file_item.path
            img_filename = os.path.basename(img_path)
            img_filename_no_ext = os.path.splitext(img_filename)[0]
            output_path = os.path.join(self.output_folder, img_filename)
            output_caption_path = os.path.join(self.output_folder, img_filename_no_ext + '.txt')
            output_depth_path = os.path.join(self.output_folder, img_filename_no_ext + '.depth.png')

            caption = batch.get_caption_list()[0]

            img: torch.Tensor = batch.tensor.clone()
            # image comes in -1 to 1. convert to a PIL RGB image
            img = (img + 1) / 2
            img = img.clamp(0, 1)
            img = img[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            image = Image.fromarray(img)

            width, height = image.size
            min_res = min(width, height)

            if self.generate_config.walk_seed:
                seed = seed + 1

            if self.generate_config.seed == -1:
                # random
                seed = random.randint(0, 1000000)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # generate depth map
            image = midas_depth(
                image,
                detect_resolution=min_res,  # do 512 ?
                image_resolution=min_res
            )

            # image.save(output_depth_path)

            gen_images = pipe(
                prompt=caption,
                negative_prompt=self.generate_config.neg,
                image=image,
                num_inference_steps=self.generate_config.sample_steps,
                adapter_conditioning_scale=self.generate_config.adapter_conditioning_scale,
                guidance_scale=self.generate_config.guidance_scale,
            ).images[0]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            gen_images.save(output_path)

            # save caption
            with open(output_caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)

            pbar.update(1)
            batch.cleanup()

        pbar.close()
        print("Done generating images")
        # cleanup
        del self.sd
        gc.collect()
        torch.cuda.empty_cache()
