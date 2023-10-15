import os.path
from collections import OrderedDict
from typing import Union

from PIL import Image
from diffusers import T2IAdapter
from torch.utils.data import DataLoader

from toolkit.basic import value_map
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.ip_adapter import IPAdapter
from toolkit.prompt_utils import concat_prompt_embeds, split_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms


def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    # transforms.PILToTensor(),
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', None]

    def before_model_load(self):
        pass

    def before_dataset_load(self):
        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            # dont name this adapter since we are not training it
            self.assistant_adapter = T2IAdapter.from_pretrained(
                adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype), varient="fp16"
            ).to(self.device_torch)
            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()

    def hook_before_train_loop(self):
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()

    def hook_train_loop(self, batch):

        self.timer.start('preprocess_batch')
        dtype = get_torch_dtype(self.train_config.dtype)
        noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
        network_weight_list = batch.get_network_weight_list()

        has_adapter_img = batch.control_tensor is not None

        match_adapter_assist = False

        # check if we are matching the adapter assistant
        if self.assistant_adapter:
            if self.train_config.match_adapter_chance == 1.0:
                match_adapter_assist = True
            elif self.train_config.match_adapter_chance > 0.0:
                match_adapter_assist = torch.rand(
                    (1,), device=self.device_torch, dtype=dtype
                ) < self.train_config.match_adapter_chance

        self.timer.stop('preprocess_batch')

        with torch.no_grad():
            adapter_images = None
            sigmas = None
            if has_adapter_img and (self.adapter or self.assistant_adapter):
                with self.timer('get_adapter_images'):
                    # todo move this to data loader
                    if batch.control_tensor is not None:
                        adapter_images = batch.control_tensor.to(self.device_torch, dtype=dtype).detach()
                        # match in channels
                        if self.assistant_adapter is not None:
                            in_channels = self.assistant_adapter.config.in_channels
                            if adapter_images.shape[1] != in_channels:
                                # we need to match the channels
                                adapter_images = adapter_images[:, :in_channels, :, :]
                    else:
                        raise NotImplementedError("Adapter images now must be loaded with dataloader")
                    # not 100% sure what this does. But they do it here
                    # https://github.com/huggingface/diffusers/blob/38a664a3d61e27ab18cd698231422b3c38d6eebf/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1170
                    # sigmas = self.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    # noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

            mask_multiplier = 1.0
            if batch.mask_tensor is not None:
                with self.timer('get_mask_multiplier'):
                    # upsampling no supported for bfloat16
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=torch.float16).detach()
                    # scale down to the size of the latents, mask multiplier shape(bs, 1, width, height), noisy_latents shape(bs, channels, width, height)
                    mask_multiplier = torch.nn.functional.interpolate(
                        mask_multiplier, size=(noisy_latents.shape[2], noisy_latents.shape[3])
                    )
                    # expand to match latents
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()

        def get_adapter_multiplier():
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                # training a t2i adapter, not using as assistant.
                return 1.0
            elif match_adapter_assist:
                # training a texture. We want it high
                adapter_strength_min = 0.9
                adapter_strength_max = 1.0
            else:
                # training with assistance, we want it low
                # adapter_strength_min = 0.5
                # adapter_strength_max = 0.8
                adapter_strength_min = 0.9
                adapter_strength_max = 1.1

            adapter_conditioning_scale = torch.rand(
                (1,), device=self.device_torch, dtype=dtype
            )

            adapter_conditioning_scale = value_map(
                adapter_conditioning_scale,
                0.0,
                1.0,
                adapter_strength_min,
                adapter_strength_max
            )
            return adapter_conditioning_scale

        # flush()
        with self.timer('grad_setup'):

            # text encoding
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True

            if self.embedding:
                grad_on_text_encoder = True

            # have a blank network so we can wrap it in a context and set multipliers without checking every time
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()

            # set the weights
            network.multiplier = network_weight_list
            self.optimizer.zero_grad(set_to_none=True)

        # activate network if it exits
        with network:
            with self.timer('encode_prompt'):
                if grad_on_text_encoder:
                    with torch.set_grad_enabled(True):
                        conditional_embeds = self.sd.encode_prompt(conditioned_prompts).to(
                            self.device_torch,
                            dtype=dtype)
                else:
                    with torch.set_grad_enabled(False):
                        # make sure it is in eval mode
                        if isinstance(self.sd.text_encoder, list):
                            for te in self.sd.text_encoder:
                                te.eval()
                        else:
                            self.sd.text_encoder.eval()
                        conditional_embeds = self.sd.encode_prompt(conditioned_prompts).to(
                            self.device_torch,
                            dtype=dtype)

                    # detach the embeddings
                    conditional_embeds = conditional_embeds.detach()

            # flush()
            pred_kwargs = {}
            if has_adapter_img and ((self.adapter and isinstance(self.adapter, T2IAdapter)) or self.assistant_adapter):
                with torch.set_grad_enabled(self.adapter is not None):
                    adapter = self.adapter if self.adapter else self.assistant_adapter
                    adapter_multiplier = get_adapter_multiplier()
                    with self.timer('encode_adapter'):
                        down_block_additional_residuals = adapter(adapter_images)
                        if self.assistant_adapter:
                            # not training. detach
                            down_block_additional_residuals = [
                                sample.to(dtype=dtype).detach() * adapter_multiplier for sample in
                                down_block_additional_residuals
                            ]
                        else:
                            down_block_additional_residuals = [
                                sample.to(dtype=dtype) * adapter_multiplier for sample in
                                down_block_additional_residuals
                            ]

                        pred_kwargs['down_block_additional_residuals'] = down_block_additional_residuals

            control_pred = None
            if has_adapter_img and self.assistant_adapter and match_adapter_assist:
                with self.timer('predict_with_adapter'):
                    # do a prediction here so we can match its output with network multiplier set to 0.0
                    with torch.no_grad():
                        # dont use network on this
                        network.multiplier = 0.0
                        self.sd.unet.eval()
                        control_pred = self.sd.predict_noise(
                            latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype).detach(),
                            timestep=timesteps,
                            guidance_scale=1.0,
                            **pred_kwargs  # adapter residuals in here
                        )
                        self.sd.unet.train()
                        control_pred = control_pred.detach()
                        # remove the residuals as we wont use them on prediction when matching control
                        del pred_kwargs['down_block_additional_residuals']
                        # restore network
                        network.multiplier = network_weight_list

            if has_adapter_img and self.adapter and isinstance(self.adapter, IPAdapter):
                with self.timer('encode_adapter'):
                    with torch.no_grad():
                        conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(adapter_images)
                    conditional_embeds = self.adapter(conditional_embeds, conditional_clip_embeds)

            with self.timer('predict_unet'):
                noise_pred = self.sd.predict_noise(
                    latents=noisy_latents.to(self.device_torch, dtype=dtype),
                    conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                    timestep=timesteps,
                    guidance_scale=1.0,
                    **pred_kwargs
                )

            with self.timer('calculate_loss'):
                noise = noise.to(self.device_torch, dtype=dtype).detach()

                if control_pred is not None:
                    # matching adapter prediction
                    target = control_pred
                elif self.sd.prediction_type == 'v_prediction':
                    # v-parameterization training
                    target = self.sd.noise_scheduler.get_velocity(noisy_latents, noise, timesteps)
                else:
                    target = noise
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                # multiply by our mask
                loss = loss * mask_multiplier

                loss = loss.mean([1, 2, 3])

                if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
                    # add min_snr_gamma
                    loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

                loss = loss.mean()
            # check if nan
            if torch.isnan(loss):
                raise ValueError("loss is nan")

            with self.timer('backward'):
                # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                # it will destroy the gradients. This is because the network is a context manager
                # and will change the multipliers back to 0.0 when exiting. They will be
                # 0.0 for the backward pass and the gradients will be 0.0
                # I spent weeks on fighting this. DON'T DO IT
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
                # flush()

        with self.timer('optimizer_step'):
            # apply gradients
            self.optimizer.step()
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )

        return loss_dict
