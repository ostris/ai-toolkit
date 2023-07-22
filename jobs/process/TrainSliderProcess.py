# ref:
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
import time
from collections import OrderedDict
import os
from typing import List

from toolkit.kohya_model_util import load_vae
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.paths import REPOS_ROOT
import sys

sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))

from diffusers import StableDiffusionPipeline

from jobs.process import BaseTrainProcess
from toolkit.metadata import get_meta_for_safetensors
from toolkit.train_tools import get_torch_dtype
import gc

import torch
from tqdm import tqdm

from toolkit.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV, TRAINING_METHODS
from leco import train_util, model_util
from leco.prompt_util import PromptEmbedsCache, PromptEmbedsPair, ACTION_TYPES
from leco import debug_util


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class StableDiffusion:
    def __init__(self, vae, tokenizer, text_encoder, unet, noise_scheduler):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.noise_scheduler = noise_scheduler


class SaveConfig:
    def __init__(self, **kwargs):
        self.save_every: int = kwargs.get('save_every', 1000)
        self.dtype: str = kwargs.get('save_dtype', 'float16')


class LogingConfig:
    def __init__(self, **kwargs):
        self.log_every: int = kwargs.get('log_every', 100)
        self.verbose: bool = kwargs.get('verbose', False)
        self.use_wandb: bool = kwargs.get('use_wandb', False)


class SampleConfig:
    def __init__(self, **kwargs):
        self.sample_every: int = kwargs.get('sample_every', 100)
        self.width: int = kwargs.get('width', 512)
        self.height: int = kwargs.get('height', 512)
        self.prompts: list[str] = kwargs.get('prompts', [])
        self.neg = kwargs.get('neg', False)
        self.seed = kwargs.get('seed', 0)
        self.walk_seed = kwargs.get('walk_seed', False)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)


class NetworkConfig:
    def __init__(self, **kwargs):
        self.type: str = kwargs.get('type', 'lierla')
        self.rank: int = kwargs.get('rank', 4)
        self.alpha: float = kwargs.get('alpha', 1.0)


class TrainConfig:
    def __init__(self, **kwargs):
        self.noise_scheduler: 'model_util.AVAILABLE_SCHEDULERS' = kwargs.get('noise_scheduler', 'ddpm')
        self.steps: int = kwargs.get('steps', 1000)
        self.lr = kwargs.get('lr', 1e-6)
        self.optimizer = kwargs.get('optimizer', 'adamw')
        self.lr_scheduler = kwargs.get('lr_scheduler', 'constant')
        self.max_denoising_steps: int = kwargs.get('max_denoising_steps', 50)
        self.batch_size: int = kwargs.get('batch_size', 1)
        self.dtype: str = kwargs.get('dtype', 'fp32')
        self.xformers = kwargs.get('xformers', False)
        self.train_unet = kwargs.get('train_unet', True)
        self.train_text_encoder = kwargs.get('train_text_encoder', True)


class ModelConfig:
    def __init__(self, **kwargs):
        self.name_or_path: str = kwargs.get('name_or_path', None)
        self.is_v2: bool = kwargs.get('is_v2', False)
        self.is_v_pred: bool = kwargs.get('is_v_pred', False)

        if self.name_or_path is None:
            raise ValueError('name_or_path must be specified')


class SliderTargetConfig:
    def __init__(self, **kwargs):
        self.target_class: str = kwargs.get('target_class', None)
        self.positive: str = kwargs.get('positive', None)
        self.negative: str = kwargs.get('negative', None)


class SliderConfig:
    def __init__(self, **kwargs):
        targets = kwargs.get('targets', [])
        targets = [SliderTargetConfig(**target) for target in targets]
        self.targets: List[SliderTargetConfig] = targets
        self.resolutions: List[List[int]] = kwargs.get('resolutions', [[512, 512]])


class PromptSettingsOld:
    def __init__(self, **kwargs):
        self.target: str = kwargs.get('target', None)
        self.positive = kwargs.get('positive', None)  # if None, target will be used
        self.unconditional = kwargs.get('unconditional', "")  # default is ""
        self.neutral = kwargs.get('neutral', None)  # if None, unconditional will be used
        self.action: ACTION_TYPES = kwargs.get('action', "erase")  # default is "erase"
        self.guidance_scale: float = kwargs.get('guidance_scale', 1.0)  # default is 1.0
        self.resolution: int = kwargs.get('resolution', 512)  # default is 512
        self.dynamic_resolution: bool = kwargs.get('dynamic_resolution', False)  # default is False
        self.batch_size: int = kwargs.get('batch_size', 1)  # default is 1
        self.dynamic_crops: bool = kwargs.get('dynamic_crops', False)  # default is False. only used when model is XL


class EncodedPromptPair:
    def __init__(
            self,
            target_class,
            positive,
            negative,
            neutral,
            width=512,
            height=512
    ):
        self.target_class = target_class
        self.positive = positive
        self.negative = negative
        self.neutral = neutral
        self.width = width
        self.height = height


class TrainSliderProcess(BaseTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        self.network_config = NetworkConfig(**self.get_conf('network', {}))
        self.training_folder = self.get_conf('training_folder', self.job.training_folder)
        self.train_config = TrainConfig(**self.get_conf('train', {}))
        self.model_config = ModelConfig(**self.get_conf('model', {}))
        self.save_config = SaveConfig(**self.get_conf('save', {}))
        self.sample_config = SampleConfig(**self.get_conf('sample', {}))
        self.logging_config = LogingConfig(**self.get_conf('logging', {}))
        self.slider_config = SliderConfig(**self.get_conf('slider', {}))
        self.sd = None

        # added later
        self.network = None
        self.scheduler = None
        self.is_flipped = False

    def flip_network(self):
        for param in self.network.parameters():
            # apply opposite weight to the network
            param.data = -param.data
        self.is_flipped = not self.is_flipped

    def sample(self, step=None):
        sample_folder = os.path.join(self.save_root, 'samples')
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder, exist_ok=True)

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
        current_seed = start_seed

        pipeline.to(self.device_torch)
        with self.network:
            with torch.no_grad():
                assert self.network.is_active
                if self.logging_config.verbose:
                    print("network_state", {
                        'is_active': self.network.is_active,
                        'multiplier': self.network.multiplier,
                    })

                for i in tqdm(range(len(self.sample_config.prompts)), desc=f"Generating Samples - step: {step}"):
                    raw_prompt = self.sample_config.prompts[i]
                    prompt = raw_prompt
                    neg = self.sample_config.neg
                    p_split = raw_prompt.split('--n')
                    if len(p_split) > 1:
                        prompt = p_split[0].strip()
                        neg = p_split[1].strip()
                    height = self.sample_config.height
                    width = self.sample_config.width
                    height = max(64, height - height % 8)  # round to divisible by 8
                    width = max(64, width - width % 8)  # round to divisible by 8

                    if self.sample_config.walk_seed:
                        current_seed += i

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
        self.network.train()
        # self.sd.tokenizer.to(original_device_dict['tokenizer'])

    def update_training_metadata(self):
        self.add_meta(OrderedDict({"training_info": self.get_training_info()}))

    def get_training_info(self):
        info = OrderedDict({
            'step': self.step_num
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
        self.network.save_weights(
            file_path,
            dtype=get_torch_dtype(self.save_config.dtype),
            metadata=save_meta
        )

        self.print(f"Saved to {file_path}")

    def run(self):
        super().run()

        dtype = get_torch_dtype(self.train_config.dtype)

        modules = DEFAULT_TARGET_REPLACE
        loss = None
        if self.network_config.type == "c3lier":
            modules += UNET_TARGET_REPLACE_MODULE_CONV

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

        optimizer_type = self.train_config.optimizer.lower()
        # we call it something different than leco
        if optimizer_type == "dadaptation":
            optimizer_type = "dadaptadam"
        optimizer_module = train_util.get_optimizer(optimizer_type)
        optimizer = optimizer_module(
            self.network.prepare_optimizer_params(
                self.train_config.lr, self.train_config.lr, self.train_config.lr
            ),
            lr=self.train_config.lr
        )
        lr_scheduler = train_util.get_lr_scheduler(
            self.train_config.lr_scheduler,
            optimizer,
            max_iterations=self.train_config.steps,
            lr_min=self.train_config.lr / 100,  # not sure why leco did this, but ill do it to
        )
        loss_function = torch.nn.MSELoss()

        cache = PromptEmbedsCache()
        prompt_pairs: list[LatentPair] = []

        # get encoded latents for our prompts
        with torch.no_grad():
            neutral = ""
            for target in self.slider_config.targets:
                for resolution in self.slider_config.resolutions:
                    width, height = resolution
                    for prompt in [
                        target.target_class,
                        target.positive,
                        target.negative,
                        neutral  # empty neutral
                    ]:
                        if cache[prompt] == None:
                            cache[prompt] = train_util.encode_prompts(
                                tokenizer, text_encoder, [prompt]
                            )

                    prompt_pairs.append(
                        EncodedPromptPair(
                            target_class=cache[target.target_class],
                            positive=cache[target.positive],
                            negative=cache[target.negative],
                            neutral=cache[neutral],
                            width=width,
                            height=height,
                        )
                    )

        # move to cpu to save vram
        # tokenizer.to("cpu")
        text_encoder.to("cpu")
        flush()

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

            # get a random pair
            prompt_pair: EncodedPromptPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            height = prompt_pair.height
            width = prompt_pair.width
            positive = prompt_pair.positive
            target_class = prompt_pair.target_class
            neutral = prompt_pair.neutral
            negative = prompt_pair.negative

            # swap every other step and invert lora to spread slider
            do_swap = step % 2 == 0

            if do_swap:
                negative = prompt_pair.positive
                positive = prompt_pair.negative
                # set the network in a negative weight
                self.network.multiplier = -1.0


            with torch.no_grad():
                noise_scheduler.set_timesteps(
                    self.train_config.max_denoising_steps, device=self.device_torch
                )

                optimizer.zero_grad()

                # ger a random number of steps
                timesteps_to = torch.randint(
                    1, self.train_config.max_denoising_steps, (1,)
                ).item()

                latents = train_util.get_initial_latents(
                    noise_scheduler,
                    self.train_config.batch_size,
                    height,
                    width,
                    1
                ).to(self.device_torch, dtype=dtype)

                with self.network:
                    assert self.network.is_active
                    # A little denoised one is returned
                    denoised_latents = train_util.diffusion(
                        unet,
                        noise_scheduler,
                        latents,  # pass simple noise latents
                        train_util.concat_embeddings(
                            positive, # unconditional
                            target_class, # target
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
                        positive, # unconditional
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
                        target_class, # target
                        self.train_config.batch_size,
                    ),
                    guidance_scale=1,
                ).to("cpu", dtype=torch.float32)

                # if self.logging_config.verbose:
                #     self.print("target_latents:", target_latents[0, 0, :5, :5])

            positive_latents.requires_grad = False
            neutral_latents.requires_grad = False
            unconditional_latents.requires_grad = False

            erase = True
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
            )

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
                        self.writer.add_scalar(f"loss", loss_float, self.step_num)
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

        print("")

        self.save()


        del (
            unet,
            noise_scheduler,
            loss,
            optimizer,
            self.network,
            tokenizer,
            text_encoder,
        )

        flush()
