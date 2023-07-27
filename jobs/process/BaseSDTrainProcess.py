import glob
import time
from collections import OrderedDict
import os

import diffusers
from safetensors import safe_open

from library import sdxl_train_util, sdxl_model_util
from toolkit.kohya_model_util import load_vae
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.optimizer import get_optimizer
from toolkit.paths import REPOS_ROOT
import sys

sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMScheduler

from jobs.process import BaseTrainProcess
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import gc

import torch
from tqdm import tqdm

from leco import train_util, model_util
from toolkit.config_modules import SaveConfig, LogingConfig, SampleConfig, NetworkConfig, TrainConfig, ModelConfig
from toolkit.stable_diffusion_model import StableDiffusion, PromptEmbeds


def flush():
    torch.cuda.empty_cache()
    gc.collect()


UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8


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
        self.first_sample_config = SampleConfig(
            **self.get_conf('first_sample', {})) if 'first_sample' in self.config else self.sample_config
        self.logging_config = LogingConfig(**self.get_conf('logging', {}))
        self.optimizer = None
        self.lr_scheduler = None
        self.sd: 'StableDiffusion' = None

        # sdxl stuff
        self.logit_scale = None
        self.ckppt_info = None

        # added later
        self.network = None

    def sample(self, step=None, is_first=False):
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
            # 'tokenizer': self.sd.tokenizer.device,
        }

        # handle sdxl text encoder
        if isinstance(self.sd.text_encoder, list):
            for encoder, i in zip(self.sd.text_encoder, range(len(self.sd.text_encoder))):
                original_device_dict[f'text_encoder_{i}'] = encoder.device
                encoder.to(self.device_torch)
        else:
            original_device_dict['text_encoder'] = self.sd.text_encoder.device
            self.sd.text_encoder.to(self.device_torch)

        self.sd.vae.to(self.device_torch)
        self.sd.unet.to(self.device_torch)
        # self.sd.text_encoder.to(self.device_torch)
        # self.sd.tokenizer.to(self.device_torch)
        # TODO add clip skip
        if self.sd.is_xl:
            pipeline = StableDiffusionXLPipeline(
                vae=self.sd.vae,
                unet=self.sd.unet,
                text_encoder=self.sd.text_encoder[0],
                text_encoder_2=self.sd.text_encoder[1],
                tokenizer=self.sd.tokenizer[0],
                tokenizer_2=self.sd.tokenizer[1],
                scheduler=self.sd.noise_scheduler,
            )
        else:
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

        sample_config = self.first_sample_config if is_first else self.sample_config

        start_seed = sample_config.seed
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

                for i in tqdm(range(len(sample_config.prompts)), desc=f"Generating Samples - step: {step}",
                              leave=False):
                    raw_prompt = sample_config.prompts[i]

                    neg = sample_config.neg
                    multiplier = sample_config.network_multiplier
                    p_split = raw_prompt.split('--')
                    prompt = p_split[0].strip()
                    height = sample_config.height
                    width = sample_config.width

                    if len(p_split) > 1:
                        for split in p_split:
                            flag = split[:1]
                            content = split[1:].strip()
                            if flag == 'n':
                                neg = content
                            elif flag == 'm':
                                # multiplier
                                multiplier = float(content)
                            elif flag == 'w':
                                # multiplier
                                width = int(content)
                            elif flag == 'h':
                                # multiplier
                                height = int(content)

                    height = max(64, height - height % 8)  # round to divisible by 8
                    width = max(64, width - width % 8)  # round to divisible by 8

                    if sample_config.walk_seed:
                        current_seed += i

                    if self.network is not None:
                        self.network.multiplier = multiplier
                    torch.manual_seed(current_seed)
                    torch.cuda.manual_seed(current_seed)

                    if self.sd.is_xl:
                        img = pipeline(
                            prompt,
                            height=height,
                            width=width,
                            num_inference_steps=sample_config.sample_steps,
                            guidance_scale=sample_config.guidance_scale,
                            negative_prompt=neg,
                        ).images[0]
                    else:
                        img = pipeline(
                            prompt,
                            height=height,
                            width=width,
                            num_inference_steps=sample_config.sample_steps,
                            guidance_scale=sample_config.guidance_scale,
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
        if isinstance(self.sd.text_encoder, list):
            for encoder, i in zip(self.sd.text_encoder, range(len(self.sd.text_encoder))):
                encoder.to(original_device_dict[f'text_encoder_{i}'])
        else:
            self.sd.text_encoder.to(original_device_dict['text_encoder'])
        if self.network is not None:
            self.network.train()
            self.network.multiplier = start_multiplier
        # self.sd.tokenizer.to(original_device_dict['tokenizer'])

    def update_training_metadata(self):
        dict = OrderedDict({
            "training_info": self.get_training_info()
        })
        if self.model_config.is_v2:
            dict['ss_v2'] = True

        if self.model_config.is_xl:
            dict['ss_base_model_version'] = 'sdxl_1.0'

        dict['ss_output_name'] = self.job.name

        self.add_meta(dict)

    def get_training_info(self):
        info = OrderedDict({
            'step': self.step_num + 1
        })
        return info

    def clean_up_saves(self):
        # remove old saves
        # get latest saved step
        if os.path.exists(self.save_root):
            latest_file = None
            # pattern is {job_name}_{zero_filles_step}.safetensors but NOT {job_name}.safetensors
            pattern = f"{self.job.name}_*.safetensors"
            files = glob.glob(os.path.join(self.save_root, pattern))
            if len(files) > self.save_config.max_step_saves_to_keep:
                # remove all but the latest max_step_saves_to_keep
                files.sort(key=os.path.getctime)
                for file in files[:-self.save_config.max_step_saves_to_keep]:
                    self.print(f"Removing old save: {file}")
                    os.remove(file)
            return latest_file
        else:
            return None

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
            self.sd.save(
                file_path,
                save_meta,
                get_torch_dtype(self.save_config.dtype)
            )

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

    def get_time_ids_from_latents(self, latents):
        bs, ch, h, w = list(latents.shape)

        height = h * VAE_SCALE_FACTOR
        width = w * VAE_SCALE_FACTOR

        dtype = get_torch_dtype(self.train_config.dtype)

        if self.sd.is_xl:
            prompt_ids = train_util.get_add_time_ids(
                height,
                width,
                dynamic_crops=False,  # look into this
                dtype=dtype,
            ).to(self.device_torch, dtype=dtype)
            return train_util.concat_embeddings(
                prompt_ids, prompt_ids, bs
            )
        else:
            return None

    def predict_noise(
            self,
            latents: torch.FloatTensor,
            text_embeddings: PromptEmbeds,
            timestep: int,
            guidance_scale=7.5,
            guidance_rescale=0.7,
            add_time_ids=None,
            **kwargs,
    ):
        if self.sd.is_xl:
            if add_time_ids is None:
                add_time_ids = self.get_time_ids_from_latents(latents)
            # todo LECOs code looks like it is omitting noise_pred
            # noise_pred = train_util.predict_noise_xl(
            #     self.sd.unet,
            #     self.sd.noise_scheduler,
            #     timestep,
            #     latents,
            #     text_embeddings.text_embeds,
            #     text_embeddings.pooled_embeds,
            #     add_time_ids,
            #     guidance_scale=guidance_scale,
            #     guidance_rescale=guidance_rescale
            # )
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.sd.noise_scheduler.scale_model_input(latent_model_input, timestep)

            added_cond_kwargs = {
                "text_embeds": text_embeddings.pooled_embeds,
                "time_ids": add_time_ids,
            }

            # predict the noise residual
            noise_pred = self.sd.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings.text_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_target = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

            # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
            # noise_pred = rescale_noise_cfg(
            #     noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
            # )

            noise_pred = guided_target

        else:
            noise_pred = train_util.predict_noise(
                self.sd.unet,
                self.sd.noise_scheduler,
                timestep,
                latents,
                text_embeddings.text_embeds if hasattr(text_embeddings, 'text_embeds') else text_embeddings,
                guidance_scale=guidance_scale
            )

        return noise_pred

    # ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
    def diffuse_some_steps(
            self,
            latents: torch.FloatTensor,
            text_embeddings: PromptEmbeds,
            total_timesteps: int = 1000,
            start_timesteps=0,
            guidance_scale=1,
            add_time_ids=None,
            **kwargs,
    ):

        for timestep in tqdm(self.sd.noise_scheduler.timesteps[start_timesteps:total_timesteps], leave=False):
            noise_pred = self.predict_noise(
                latents,
                text_embeddings,
                timestep,
                guidance_scale=guidance_scale,
                add_time_ids=add_time_ids,
                **kwargs,
            )
            latents = self.sd.noise_scheduler.step(noise_pred, timestep, latents).prev_sample

        # return latents_steps
        return latents

    def get_latest_save_path(self):
        # get latest saved step
        if os.path.exists(self.save_root):
            latest_file = None
            # pattern is {job_name}_{zero_filles_step}.safetensors or {job_name}.safetensors
            pattern = f"{self.job.name}*.safetensors"
            files = glob.glob(os.path.join(self.save_root, pattern))
            if len(files) > 0:
                latest_file = max(files, key=os.path.getctime)
            return latest_file
        else:
            return None

    def load_weights(self, path):
        if self.network is not None:
            self.network.load_weights(path)
            meta = load_metadata_from_safetensors(path)
            # if 'training_info' in Orderdict keys
            if 'training_info' in meta and 'step' in meta['training_info']:
                self.step_num = meta['training_info']['step']
                self.start_step = self.step_num
                print(f"Found step {self.step_num} in metadata, starting from there")

        else:
            print("load_weights not implemented for non-network models")

    def run(self):
        super().run()

        ### HOOK ###
        self.hook_before_model_load()

        dtype = get_torch_dtype(self.train_config.dtype)

        if self.model_config.is_xl:

            pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_config.name_or_path,
                dtype=dtype,
                scheduler_type='pndm',
                device=self.device_torch
            )
            text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
            tokenizer = [pipe.tokenizer, pipe.tokenizer_2]
            unet = pipe.unet
            noise_scheduler = pipe.scheduler
            vae = pipe.vae.to('cpu', dtype=dtype)
            vae.eval()
            vae.set_use_memory_efficient_attention_xformers(True)

            for text_encoder in text_encoders:
                text_encoder.to(self.device_torch, dtype=dtype)
                text_encoder.requires_grad_(False)
                text_encoder.eval()

            text_encoder = text_encoders
            tokenizer = tokenizer
            del pipe
            flush()


        else:
            tokenizer, text_encoder, unet, noise_scheduler = model_util.load_models(
                self.model_config.name_or_path,
                scheduler_name=self.train_config.noise_scheduler,
                v2=self.model_config.is_v2,
                v_pred=self.model_config.is_v_pred,
            )

            text_encoder.to(self.device_torch, dtype=dtype)
            text_encoder.eval()
            vae = load_vae(self.model_config.name_or_path, dtype=dtype).to('cpu', dtype=dtype)
            vae.eval()
        flush()


        # just for now or of we want to load a custom one
        # put on cpu for now, we only need it when sampling
        # vae = load_vae(self.model_config.name_or_path, dtype=dtype).to('cpu', dtype=dtype)
        # vae.eval()
        self.sd = StableDiffusion(vae, tokenizer, text_encoder, unet, noise_scheduler, is_xl=self.model_config.is_xl)

        unet.to(self.device_torch, dtype=dtype)
        if self.train_config.xformers:
            unet.enable_xformers_memory_efficient_attention()
        if self.train_config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        unet.requires_grad_(False)
        unet.eval()

        if self.network_config is not None:
            conv = self.network_config.conv if self.network_config.conv is not None and self.network_config.conv > 0 else None
            self.network = LoRASpecialNetwork(
                text_encoder=text_encoder,
                unet=unet,
                lora_dim=self.network_config.linear,
                multiplier=1.0,
                alpha=self.network_config.alpha,
                train_unet=self.train_config.train_unet,
                train_text_encoder=self.train_config.train_text_encoder,
                conv_lora_dim=conv,
                conv_alpha=self.network_config.alpha if conv is not None else None,
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

            latest_save_path = self.get_latest_save_path()
            if latest_save_path is not None:
                self.print(f"#### IMPORTANT RESUMING FROM {latest_save_path} ####")
                self.print(f"Loading from {latest_save_path}")
                self.load_weights(latest_save_path)
                self.network.multiplier = 1.0



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
        if self.train_config.skip_first_sample:
            self.print("Skipping first sample due to config setting")
        else:
            self.print("Generating baseline samples before training")
            self.sample(0, is_first=True)

        self.progress_bar = tqdm(
            total=self.train_config.steps,
            desc=self.job.name,
            leave=True
        )
        # set it to our current step in case it was updated from a load
        self.progress_bar.update(self.step_num)
        # self.step_num = 0
        for step in range(self.step_num, self.train_config.steps):
            # todo handle dataloader here maybe, not sure

            ### HOOK ###
            loss_dict = self.hook_train_loop()

            if self.train_config.optimizer.startswith('dadaptation'):
                learning_rate = (
                        optimizer.param_groups[0]["d"] *
                        optimizer.param_groups[0]["lr"]
                )
            else:
                learning_rate = optimizer.param_groups[0]['lr']

            prog_bar_string = f"lr: {learning_rate:.1e}"
            for key, value in loss_dict.items():
                prog_bar_string += f" {key}: {value:.3e}"

            self.progress_bar.set_postfix_str(prog_bar_string)

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
                        for key, value in loss_dict.items():
                            self.writer.add_scalar(f"{key}", value, self.step_num)
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
