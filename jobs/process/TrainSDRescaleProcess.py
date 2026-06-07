import glob
import os
from collections import OrderedDict
import random
from typing import Optional, List

from safetensors.torch import save_file, load_file
from tqdm import tqdm

from toolkit.layers import ReductionKernel
from toolkit.stable_diffusion_model import PromptEmbeds
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import gc
from toolkit import train_tools

import torch
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
        self.latent_tensor_dir = kwargs.get('latent_tensor_dir', None)
        self.num_latent_tensors = kwargs.get('num_latent_tensors', 1000)
        self.to_resolution = kwargs.get('to_resolution', int(self.from_resolution * self.scale))
        self.prompt_dropout = kwargs.get('prompt_dropout', 0.1)


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
        super().__init__(process_id, job, config)
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.rescale_config = RescaleConfig(**self.get_conf('rescale', required=True))
        self.reduce_size_fn = ReductionKernel(
            in_channels=4,
            kernel_size=int(self.rescale_config.from_resolution // self.rescale_config.to_resolution),
            dtype=get_torch_dtype(self.train_config.dtype),
            device=self.device_torch,
        )

        self.latent_paths: List[str] = []
        self.empty_embedding: PromptEmbeds = None

    def before_model_load(self):
        pass

    def get_latent_tensors(self):
        dtype = get_torch_dtype(self.train_config.dtype)

        num_to_generate = 0
        # check if dir exists
        if not os.path.exists(self.rescale_config.latent_tensor_dir):
            os.makedirs(self.rescale_config.latent_tensor_dir)
            num_to_generate = self.rescale_config.num_latent_tensors
        else:
            # find existing
            current_tensor_list = glob.glob(os.path.join(self.rescale_config.latent_tensor_dir, "*.safetensors"))
            num_to_generate = self.rescale_config.num_latent_tensors - len(current_tensor_list)
            self.latent_paths = current_tensor_list

        if num_to_generate > 0:
            print(f"Generating {num_to_generate}/{self.rescale_config.num_latent_tensors} latent tensors")

            # unload other model
            self.sd.unet.to('cpu')

            # load aux network
            self.sd_parent = StableDiffusion(
                self.device_torch,
                model_config=self.model_config,
                dtype=self.train_config.dtype,
            )
            self.sd_parent.load_model()
            self.sd_parent.unet.to(self.device_torch, dtype=dtype)
            # we dont need text encoder for this

            del self.sd_parent.text_encoder
            del self.sd_parent.tokenizer

            self.sd_parent.unet.eval()
            self.sd_parent.unet.requires_grad_(False)

            # save current seed state for training
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

            text_embeddings = train_tools.concat_prompt_embeddings(
                self.empty_embedding,  # unconditional (negative prompt)
                self.empty_embedding,  # conditional (positive prompt)
                self.train_config.batch_size,
            )
            torch.set_default_device(self.device_torch)

            for i in tqdm(range(num_to_generate)):
                dtype = get_torch_dtype(self.train_config.dtype)
                # get a random seed
                seed = torch.randint(0, 2 ** 32, (1,)).item()
                # zero pad seed string to max length
                seed_string = str(seed).zfill(10)
                # set seed
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # # ger a random number of steps
                timesteps_to = self.train_config.max_denoising_steps

                # set the scheduler to the number of steps
                self.sd.noise_scheduler.set_timesteps(
                    timesteps_to, device=self.device_torch
                )

                noise = self.sd.get_latent_noise(
                    pixel_height=self.rescale_config.from_resolution,
                    pixel_width=self.rescale_config.from_resolution,
                    batch_size=self.train_config.batch_size,
                    noise_offset=self.train_config.noise_offset,
                ).to(self.device_torch, dtype=dtype)

                # get latents
                latents = noise * self.sd.noise_scheduler.init_noise_sigma
                latents = latents.to(self.device_torch, dtype=dtype)

                # get random guidance scale from 1.0 to 10.0 (CFG)
                guidance_scale = torch.rand(1).item() * 9.0 + 1.0

                # do a timestep of 1
                timestep = 1

                noise_pred_target = self.sd_parent.predict_noise(
                    latents,
                    text_embeddings=text_embeddings,
                    timestep=timestep,
                    guidance_scale=guidance_scale
                )

                # build state dict
                state_dict = OrderedDict()
                state_dict['noise_pred_target'] = noise_pred_target.to('cpu', dtype=torch.float16)
                state_dict['latents'] = latents.to('cpu', dtype=torch.float16)
                state_dict['guidance_scale'] = torch.tensor(guidance_scale).to('cpu', dtype=torch.float16)
                state_dict['timestep'] = torch.tensor(timestep).to('cpu', dtype=torch.float16)
                state_dict['timesteps_to'] = torch.tensor(timesteps_to).to('cpu', dtype=torch.float16)
                state_dict['seed'] = torch.tensor(seed).to('cpu', dtype=torch.float32) # must be float 32 to prevent overflow

                file_name = f"{seed_string}_{i}.safetensors"
                file_path = os.path.join(self.rescale_config.latent_tensor_dir, file_name)
                save_file(state_dict, file_path)
                self.latent_paths.append(file_path)

            print("Removing parent model")
            # delete parent
            del self.sd_parent
            flush()

            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            self.sd.unet.to(self.device_torch, dtype=dtype)

    def hook_before_train_loop(self):
        # encode our empty prompt
        self.empty_embedding = self.sd.encode_prompt("")
        self.empty_embedding = self.empty_embedding.to(self.device_torch,
                                                       dtype=get_torch_dtype(self.train_config.dtype))

        # Move train model encoder to cpu
        if isinstance(self.sd.text_encoder, list):
            for encoder in self.sd.text_encoder:
                encoder.to('cpu')
                encoder.eval()
                encoder.requires_grad_(False)
        else:
            self.sd.text_encoder.to('cpu')
            self.sd.text_encoder.eval()
            self.sd.text_encoder.requires_grad_(False)

        # self.sd.unet.to('cpu')
        flush()

        self.get_latent_tensors()

        flush()
        # end hook_before_train_loop

    def hook_train_loop(self, batch):
        dtype = get_torch_dtype(self.train_config.dtype)

        loss_function = torch.nn.MSELoss()

        # train it
        # Begin gradient accumulation
        self.sd.unet.train()
        self.sd.unet.requires_grad_(True)
        self.sd.unet.to(self.device_torch, dtype=dtype)

        with torch.no_grad():
            self.optimizer.zero_grad()

            # pick random latent tensor
            latent_path = random.choice(self.latent_paths)
            latent_tensor = load_file(latent_path)

            noise_pred_target = (latent_tensor['noise_pred_target']).to(self.device_torch, dtype=dtype)
            latents = (latent_tensor['latents']).to(self.device_torch, dtype=dtype)
            guidance_scale = (latent_tensor['guidance_scale']).item()
            timestep = int((latent_tensor['timestep']).item())
            timesteps_to = int((latent_tensor['timesteps_to']).item())
            # seed = int((latent_tensor['seed']).item())

            text_embeddings = train_tools.concat_prompt_embeddings(
                self.empty_embedding,  # unconditional (negative prompt)
                self.empty_embedding,  # conditional (positive prompt)
                self.train_config.batch_size,
            )
            self.sd.noise_scheduler.set_timesteps(
                timesteps_to, device=self.device_torch
            )

            denoised_target = self.sd.noise_scheduler.step(noise_pred_target, timestep, latents).prev_sample

            # get the reduced latents
            # reduced_pred = self.reduce_size_fn(noise_pred_target.detach())
            denoised_target = self.reduce_size_fn(denoised_target.detach())
            reduced_latents = self.reduce_size_fn(latents.detach())

        denoised_target.requires_grad = False
        self.optimizer.zero_grad()
        noise_pred_train = self.sd.predict_noise(
            reduced_latents,
            text_embeddings=text_embeddings,
            timestep=timestep,
            guidance_scale=guidance_scale
        )
        denoised_pred = self.sd.noise_scheduler.step(noise_pred_train, timestep, reduced_latents).prev_sample
        loss = loss_function(denoised_pred, denoised_target)
        loss_float = loss.item()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        flush()

        loss_dict = OrderedDict(
            {'loss': loss_float},
        )

        return loss_dict
        # end hook_train_loop
