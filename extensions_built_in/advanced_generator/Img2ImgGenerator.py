import math
import os
import random
from collections import OrderedDict
from typing import List

import numpy as np
from PIL import Image
from diffusers import T2IAdapter
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLImg2ImgPipeline, PixArtSigmaPipeline
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
from torchvision.transforms import ToTensor


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
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)
        self.guidance_rescale = kwargs.get('guidance_rescale', 0.0)
        self.ext = kwargs.get('ext', 'png')
        self.denoise_strength = kwargs.get('denoise_strength', 0.5)
        self.trigger_word = kwargs.get('trigger_word', None)


class Img2ImgGenerator(BaseExtensionProcess):

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.output_folder = self.get_conf('output_folder', required=True)
        self.copy_inputs_to = self.get_conf('copy_inputs_to', None)
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

    def to_pil(self, img):
        # image comes in -1 to 1. convert to a PIL RGB image
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        image = Image.fromarray(img)
        return image

    def run(self):
        with torch.no_grad():
            super().run()
            print("Loading model...")
            self.sd.load_model()
            device = torch.device(self.device)

            if self.model_config.is_xl:
                pipe = StableDiffusionXLImg2ImgPipeline(
                    vae=self.sd.vae,
                    unet=self.sd.unet,
                    text_encoder=self.sd.text_encoder[0],
                    text_encoder_2=self.sd.text_encoder[1],
                    tokenizer=self.sd.tokenizer[0],
                    tokenizer_2=self.sd.tokenizer[1],
                    scheduler=get_sampler(self.generate_config.sampler),
                ).to(device, dtype=self.torch_dtype)
            elif self.model_config.is_pixart:
                pipe = self.sd.pipeline.to(device, dtype=self.torch_dtype)
            else:
                raise NotImplementedError("Only XL models are supported")
            pipe.set_progress_bar_config(disable=True)

            # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            # midas_depth = torch.compile(midas_depth, mode="reduce-overhead", fullgraph=True)

            self.data_loader = get_dataloader_from_datasets(self.datasets, 1, self.sd)

            num_batches = len(self.data_loader)
            pbar = tqdm(total=num_batches, desc="Generating images")
            seed = self.generate_config.seed
            # load images from datasets, use tqdm
            for i, batch in enumerate(self.data_loader):
                batch: DataLoaderBatchDTO = batch

                gen_seed = seed if seed > 0 else random.randint(0, 2 ** 32 - 1)
                generator = torch.manual_seed(gen_seed)

                file_item: FileItemDTO = batch.file_items[0]
                img_path = file_item.path
                img_filename = os.path.basename(img_path)
                img_filename_no_ext = os.path.splitext(img_filename)[0]
                img_filename = img_filename_no_ext + '.' + self.generate_config.ext
                output_path = os.path.join(self.output_folder, img_filename)
                output_caption_path = os.path.join(self.output_folder, img_filename_no_ext + '.txt')

                if self.copy_inputs_to is not None:
                    output_inputs_path = os.path.join(self.copy_inputs_to, img_filename)
                    output_inputs_caption_path = os.path.join(self.copy_inputs_to, img_filename_no_ext + '.txt')
                else:
                    output_inputs_path = None
                    output_inputs_caption_path = None

                caption = batch.get_caption_list()[0]
                if self.generate_config.trigger_word is not None:
                    caption = caption.replace('[trigger]', self.generate_config.trigger_word)

                img: torch.Tensor = batch.tensor.clone()
                image = self.to_pil(img)

                # image.save(output_depth_path)
                if self.model_config.is_pixart:
                    pipe: PixArtSigmaPipeline = pipe

                    # Encode the full image once
                    encoded_image = pipe.vae.encode(
                        pipe.image_processor.preprocess(image).to(device=pipe.device, dtype=pipe.dtype))
                    if hasattr(encoded_image, "latent_dist"):
                        latents = encoded_image.latent_dist.sample(generator)
                    elif hasattr(encoded_image, "latents"):
                        latents = encoded_image.latents
                    else:
                        raise AttributeError("Could not access latents of provided encoder_output")
                    latents = pipe.vae.config.scaling_factor * latents

                    # latents = self.sd.encode_images(img)

                    # self.sd.noise_scheduler.set_timesteps(self.generate_config.sample_steps)
                    # start_step = math.floor(self.generate_config.sample_steps * self.generate_config.denoise_strength)
                    # timestep = self.sd.noise_scheduler.timesteps[start_step].unsqueeze(0)
                    # timestep = timestep.to(device, dtype=torch.int32)
                    # latent = latent.to(device, dtype=self.torch_dtype)
                    # noise = torch.randn_like(latent, device=device, dtype=self.torch_dtype)
                    # latent = self.sd.add_noise(latent, noise, timestep)
                    # timesteps_to_use = self.sd.noise_scheduler.timesteps[start_step + 1:]
                    batch_size = 1
                    num_images_per_prompt = 1

                    shape = (batch_size, pipe.transformer.config.in_channels, image.height // pipe.vae_scale_factor,
                             image.width // pipe.vae_scale_factor)
                    noise = randn_tensor(shape, generator=generator, device=pipe.device, dtype=pipe.dtype)

                    # noise = torch.randn_like(latents, device=device, dtype=self.torch_dtype)
                    num_inference_steps = self.generate_config.sample_steps
                    strength = self.generate_config.denoise_strength
                    # Get timesteps
                    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
                    t_start = max(num_inference_steps - init_timestep, 0)
                    pipe.scheduler.set_timesteps(num_inference_steps, device="cpu")
                    timesteps = pipe.scheduler.timesteps[t_start:]
                    timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
                    latents = pipe.scheduler.add_noise(latents, noise, timestep)

                    gen_images = pipe.__call__(
                        prompt=caption,
                        negative_prompt=self.generate_config.neg,
                        latents=latents,
                        timesteps=timesteps,
                        width=image.width,
                        height=image.height,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=num_images_per_prompt,
                        guidance_scale=self.generate_config.guidance_scale,
                        # strength=self.generate_config.denoise_strength,
                        use_resolution_binning=False,
                        output_type="np"
                    ).images[0]
                    gen_images = (gen_images * 255).clip(0, 255).astype(np.uint8)
                    gen_images = Image.fromarray(gen_images)
                else:
                    pipe: StableDiffusionXLImg2ImgPipeline = pipe

                    gen_images = pipe.__call__(
                        prompt=caption,
                        negative_prompt=self.generate_config.neg,
                        image=image,
                        num_inference_steps=self.generate_config.sample_steps,
                        guidance_scale=self.generate_config.guidance_scale,
                        strength=self.generate_config.denoise_strength,
                    ).images[0]
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                gen_images.save(output_path)

                # save caption
                with open(output_caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                if output_inputs_path is not None:
                    os.makedirs(os.path.dirname(output_inputs_path), exist_ok=True)
                    image.save(output_inputs_path)
                    with open(output_inputs_caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)

                pbar.update(1)
                batch.cleanup()

            pbar.close()
            print("Done generating images")
            # cleanup
            del self.sd
            gc.collect()
            torch.cuda.empty_cache()
