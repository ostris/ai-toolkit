import time
from collections import OrderedDict
import os

from toolkit.kohya_model_util import load_vae
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.optimizer import get_optimizer
from toolkit.paths import REPOS_ROOT
import sys

sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))

from diffusers import StableDiffusionPipeline

from jobs.process import BaseTrainProcess
from toolkit.metadata import get_meta_for_safetensors
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import gc

import torch
from tqdm import tqdm

from leco import train_util, model_util
from toolkit.config_modules import SaveConfig, LogingConfig, SampleConfig, NetworkConfig, TrainConfig, ModelConfig


def flush():
    torch.cuda.empty_cache()
    gc.collect()


UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8


class StableDiffusion:
    def __init__(self, vae, tokenizer, text_encoder, unet, noise_scheduler):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.noise_scheduler = noise_scheduler


class BaseSDTrainProcess(BaseTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.network_config = NetworkConfig(**self.get_conf('network', None))
        self.training_folder = self.get_conf('training_folder', self.job.training_folder)
        self.train_config = TrainConfig(**self.get_conf('train', {}))
        self.model_config = ModelConfig(**self.get_conf('model', {}))
        self.save_config = SaveConfig(**self.get_conf('save', {}))
        self.sample_config = SampleConfig(**self.get_conf('sample', {}))
        self.logging_config = LogingConfig(**self.get_conf('logging', {}))
        self.optimizer = None
        self.lr_scheduler = None
        self.sd = None

        # added later
        self.network = None
        self.scheduler = None

    def sample(self, step=None):
        sample_folder = os.path.join(self.save_root, 'samples')
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder, exist_ok=True)

        if self.network is not None:
            self.network.eval()

        # save current seed state for training
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        original_device_dict = {
            'vae': self.sd.vae.device,
            'unet': self.sd.unet.device,
            'text_encoder': self.sd.text_encoder.device,
            # 'tokenizer': self.sd.tokenizer.device,
        }

        self.sd.vae.to(self.device_torch)
        self.sd.unet.to(self.device_torch)
        self.sd.text_encoder.to(self.device_torch)
        # self.sd.tokenizer.to(self.device_torch)
        # TODO add clip skip

        pipeline = StableDiffusionPipeline(
            vae=self.sd.vae,
            unet=self.sd.unet,
            text_encoder=self.sd.text_encoder,
            tokenizer=self.sd.tokenizer,
            scheduler=self.sd.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        # disable progress bar
        pipeline.set_progress_bar_config(disable=True)

        start_seed = self.sample_config.seed
        start_multiplier = self.network.multiplier
        current_seed = start_seed

        pipeline.to(self.device_torch)
        with self.network:
            with torch.no_grad():
                if self.network is not None:
                    assert self.network.is_active
                    if self.logging_config.verbose:
                        print("network_state", {
                            'is_active': self.network.is_active,
                            'multiplier': self.network.multiplier,
                        })

                for i in tqdm(range(len(self.sample_config.prompts)), desc=f"Generating Samples - step: {step}"):
                    raw_prompt = self.sample_config.prompts[i]

                    neg = self.sample_config.neg
                    multiplier = self.sample_config.network_multiplier
                    p_split = raw_prompt.split('--')
                    prompt = p_split[0].strip()

                    if len(p_split) > 1:
                        for split in p_split:
                            flag = split[:1]
                            content = split[1:].strip()
                            if flag == 'n':
                                neg = content
                            elif flag == 'm':
                                # multiplier
                                multiplier = float(content)

                    height = self.sample_config.height
                    width = self.sample_config.width
                    height = max(64, height - height % 8)  # round to divisible by 8
                    width = max(64, width - width % 8)  # round to divisible by 8

                    if self.sample_config.walk_seed:
                        current_seed += i

                    if self.network is not None:
                        self.network.multiplier = multiplier
                    torch.manual_seed(current_seed)
                    torch.cuda.manual_seed(current_seed)

                    img = pipeline(
                        prompt,
                        height=height,
                        width=width,
                        num_inference_steps=self.sample_config.sample_steps,
                        guidance_scale=self.sample_config.guidance_scale,
                        negative_prompt=neg,
                    ).images[0]

                    step_num = ''
                    if step is not None:
                        # zero-pad 9 digits
                        step_num = f"_{str(step).zfill(9)}"
                    seconds_since_epoch = int(time.time())
                    # zero-pad 2 digits
                    i_str = str(i).zfill(2)
                    filename = f"{seconds_since_epoch}{step_num}_{i_str}.png"
                    output_path = os.path.join(sample_folder, filename)
                    img.save(output_path)

        # clear pipeline and cache to reduce vram usage
        del pipeline
        torch.cuda.empty_cache()

        # restore training state
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        self.sd.vae.to(original_device_dict['vae'])
        self.sd.unet.to(original_device_dict['unet'])
        self.sd.text_encoder.to(original_device_dict['text_encoder'])
        if self.network is not None:
            self.network.train()
            self.network.multiplier = start_multiplier
        # self.sd.tokenizer.to(original_device_dict['tokenizer'])

    def update_training_metadata(self):
        self.add_meta(OrderedDict({"training_info": self.get_training_info()}))

    def get_training_info(self):
        info = OrderedDict({
            'step': self.step_num + 1
        })
        return info

    def save(self, step=None):
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root, exist_ok=True)

        step_num = ''
        if step is not None:
            # zeropad 9 digits
            step_num = f"_{str(step).zfill(9)}"

        self.update_training_metadata()
        filename = f'{self.job.name}{step_num}.safetensors'
        file_path = os.path.join(self.save_root, filename)
        # prepare meta
        save_meta = get_meta_for_safetensors(self.meta, self.job.name)
        if self.network is not None:
            # TODO handle dreambooth, fine tuning, etc
            self.network.save_weights(
                file_path,
                dtype=get_torch_dtype(self.save_config.dtype),
                metadata=save_meta
            )
        else:
            # TODO handle dreambooth, fine tuning, etc
            # will probably have to convert dict back to LDM
            ValueError("Non network training is not currently supported")

        self.print(f"Saved to {file_path}")

    # Called before the model is loaded
    def hook_before_model_load(self):
        # override in subclass
        pass

    def hook_add_extra_train_params(self, params):
        # override in subclass
        return params

    def hook_before_train_loop(self):
        pass

    def get_latent_noise(
            self,
            height=None,
            width=None,
            pixel_height=None,
            pixel_width=None,
    ):
        if height is None and pixel_height is None:
            raise ValueError("height or pixel_height must be specified")
        if width is None and pixel_width is None:
            raise ValueError("width or pixel_width must be specified")
        if height is None:
            height = pixel_height // VAE_SCALE_FACTOR
        if width is None:
            width = pixel_width // VAE_SCALE_FACTOR

        noise = torch.randn(
            (
                self.train_config.batch_size,
                UNET_IN_CHANNELS,
                height,
                width,
            ),
            device="cpu",
        )
        noise = apply_noise_offset(noise, self.train_config.noise_offset)
        return noise

    def hook_train_loop(self):
        # return loss
        return 0.0

    def run(self):
        super().run()

        ### HOOK ###
        self.hook_before_model_load()

        dtype = get_torch_dtype(self.train_config.dtype)

        tokenizer, text_encoder, unet, noise_scheduler = model_util.load_models(
            self.model_config.name_or_path,
            scheduler_name=self.train_config.noise_scheduler,
            v2=self.model_config.is_v2,
            v_pred=self.model_config.is_v_pred,
        )
        # just for now or of we want to load a custom one
        # put on cpu for now, we only need it when sampling
        vae = load_vae(self.model_config.name_or_path, dtype=dtype).to('cpu', dtype=dtype)
        vae.eval()
        self.sd = StableDiffusion(vae, tokenizer, text_encoder, unet, noise_scheduler)

        text_encoder.to(self.device_torch, dtype=dtype)
        text_encoder.eval()

        unet.to(self.device_torch, dtype=dtype)
        if self.train_config.xformers:
            unet.enable_xformers_memory_efficient_attention()
        unet.requires_grad_(False)
        unet.eval()

        if self.network_config is not None:
            self.network = LoRASpecialNetwork(
                text_encoder=text_encoder,
                unet=unet,
                lora_dim=self.network_config.rank,
                multiplier=1.0,
                alpha=self.network_config.alpha
            )

            self.network.force_to(self.device_torch, dtype=dtype)

            self.network.apply_to(
                text_encoder,
                unet,
                self.train_config.train_text_encoder,
                self.train_config.train_unet
            )

            self.network.prepare_grad_etc(text_encoder, unet)

            params = self.network.prepare_optimizer_params(
                text_encoder_lr=self.train_config.lr,
                unet_lr=self.train_config.lr,
                default_lr=self.train_config.lr
            )

        else:
            params = []
            # assume dreambooth/finetune
            if self.train_config.train_text_encoder:
                text_encoder.requires_grad_(True)
                text_encoder.train()
                params += text_encoder.parameters()
            if self.train_config.train_unet:
                unet.requires_grad_(True)
                unet.train()
                params += unet.parameters()

        ### HOOK ###
        params = self.hook_add_extra_train_params(params)

        optimizer_type = self.train_config.optimizer.lower()
        optimizer = get_optimizer(params, optimizer_type, learning_rate=self.train_config.lr,
                                  optimizer_params=self.train_config.optimizer_params)
        self.optimizer = optimizer

        lr_scheduler = train_util.get_lr_scheduler(
            self.train_config.lr_scheduler,
            optimizer,
            max_iterations=self.train_config.steps,
            lr_min=self.train_config.lr / 100,  # not sure why leco did this, but ill do it to
        )

        self.lr_scheduler = lr_scheduler

        ### HOOK ###
        self.hook_before_train_loop()

        # sample first
        self.print("Generating baseline samples before training")
        self.sample(0)

        self.progress_bar = tqdm(
            total=self.train_config.steps,
            desc=self.job.name,
            leave=True
        )
        self.step_num = 0
        for step in range(self.train_config.steps):
            # todo handle dataloader here maybe, not sure

            ### HOOK ###
            loss = self.hook_train_loop()

            # don't do on first step
            if self.step_num != self.start_step:
                # pause progress bar
                self.progress_bar.unpause()  # makes it so doesn't track time
                if self.sample_config.sample_every and self.step_num % self.sample_config.sample_every == 0:
                    # print above the progress bar
                    self.sample(self.step_num)

                if self.save_config.save_every and self.step_num % self.save_config.save_every == 0:
                    # print above the progress bar
                    self.print(f"Saving at step {self.step_num}")
                    self.save(self.step_num)

                if self.logging_config.log_every and self.step_num % self.logging_config.log_every == 0:
                    # log to tensorboard
                    if self.writer is not None:
                        # get avg loss
                        self.writer.add_scalar(f"loss", loss, self.step_num)
                        if self.train_config.optimizer.startswith('dadaptation'):
                            learning_rate = (
                                    optimizer.param_groups[0]["d"] *
                                    optimizer.param_groups[0]["lr"]
                            )
                        else:
                            learning_rate = optimizer.param_groups[0]['lr']
                        self.writer.add_scalar(f"lr", learning_rate, self.step_num)
                self.progress_bar.refresh()

            # sets progress bar to match out step
            self.progress_bar.update(step - self.progress_bar.n)
            # end of step
            self.step_num = step

        self.sample(self.step_num + 1)
        print("")
        self.save()

        del (
            self.sd,
            unet,
            noise_scheduler,
            optimizer,
            self.network,
            tokenizer,
            text_encoder,
        )

        flush()
