from collections import OrderedDict
from typing import Union, Literal, List
from diffusers import T2IAdapter

import torch.functional as F
from toolkit import train_tools
from toolkit.basic import value_map, adain, get_mean_std
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GuidanceConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss
from toolkit.image_utils import show_tensors, show_latents
from toolkit.ip_adapter import IPAdapter
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
    apply_learnable_snr_gos, LearnableSNRGamma
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms


def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False

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
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)
        if self.adapter is not None:
            self.adapter.to(self.device_torch)

    # you can expand these in a child class to make customization easier
    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        loss_target = self.train_config.loss_target

        prior_mask_multiplier = None
        target_mask_multiplier = None

        has_mask = batch.mask_tensor is not None

        if self.train_config.match_noise_norm:
            # match the norm of the noise
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)

        if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
            # we need to make the noise prediction be a masked blending of noise and prior_pred
            stretched_mask_multiplier = value_map(
                mask_multiplier,
                batch.file_items[0].dataset_config.mask_min_value,
                1.0,
                0.0,
                1.0
            )

            prior_mask_multiplier = 1.0 - stretched_mask_multiplier

            # target_mask_multiplier = mask_multiplier
            # mask_multiplier = 1.0
            target = noise
            # target = (noise * mask_multiplier) + (prior_pred * prior_mask_multiplier)
            # set masked multiplier to 1.0 so we dont double apply it
            # mask_multiplier = 1.0
        elif prior_pred is not None:
            # matching adapter prediction
            target = prior_pred
        elif self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = self.sd.noise_scheduler.get_velocity(noisy_latents, noise, timesteps)
        else:
            target = noise

        pred = noise_pred

        ignore_snr = False

        if loss_target == 'source' or loss_target == 'unaugmented':
            # ignore_snr = True
            if batch.sigmas is None:
                raise ValueError("Batch sigmas is None. This should not happen")

            # src https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1190
            denoised_latents = noise_pred * (-batch.sigmas) + noisy_latents
            weighing = batch.sigmas ** -2.0
            if loss_target == 'source':
                # denoise the latent and compare to the latent in the batch
                target = batch.latents
            elif loss_target == 'unaugmented':
                # we have to encode images into latents for now
                # we also denoise as the unaugmented tensor is not a noisy diffirental
                with torch.no_grad():
                    unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor)
                    unaugmented_latents = unaugmented_latents * self.train_config.latent_multiplier
                    target = unaugmented_latents.detach()

                # Get the target for loss depending on the prediction type
                if self.sd.noise_scheduler.config.prediction_type == "epsilon":
                    target = target  # we are computing loss against denoise latents
                elif self.sd.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.sd.noise_scheduler.get_velocity(target, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.sd.noise_scheduler.config.prediction_type}")

            # mse loss without reduction
            loss_per_element = (weighing.float() * (denoised_latents.float() - target.float()) ** 2)
            loss = loss_per_element
        else:
            loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")

        # multiply by our mask
        loss = loss * mask_multiplier

        prior_loss = None
        if self.train_config.inverted_mask_prior and prior_pred is not None:
            # to a loss to unmasked areas of the prior for unmasked regularization
            prior_loss = torch.nn.functional.mse_loss(
                prior_pred.float(),
                pred.float(),
                reduction="none"
            )
            prior_loss = prior_loss * prior_mask_multiplier * self.train_config.inverted_mask_prior_multiplier
            if torch.isnan(prior_loss).any():
                raise ValueError("Prior loss is nan")

            # prior_loss = prior_loss.mean([1, 2, 3])
            loss = loss + prior_loss
        loss = loss.mean([1, 2, 3])

        if self.train_config.learnable_snr_gos:
            # add snr_gamma
            loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
        elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001 and not ignore_snr:
            # add snr_gamma
            loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma, fixed=True)
        elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001 and not ignore_snr:
            # add min_snr_gamma
            loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        loss = loss.mean()
        return loss

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            **kwargs
    ):
        loss = get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            **kwargs
        )

        return loss

    def get_guided_loss_targeted_polarity(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            # Perform targeted guidance (working title)
            dtype = get_torch_dtype(self.train_config.dtype)

            conditional_latents = batch.latents.to(self.device_torch, dtype=dtype).detach()
            unconditional_latents = batch.unconditional_latents.to(self.device_torch, dtype=dtype).detach()

            mean_latents = (conditional_latents + unconditional_latents) / 2.0

            unconditional_diff = (unconditional_latents - mean_latents)
            conditional_diff = (conditional_latents - mean_latents)

            # we need to determine the amount of signal and noise that would be present at the current timestep
            # conditional_signal = self.sd.add_noise(conditional_diff, torch.zeros_like(noise), timesteps)
            # unconditional_signal = self.sd.add_noise(torch.zeros_like(noise), unconditional_diff, timesteps)
            # unconditional_signal = self.sd.add_noise(unconditional_diff, torch.zeros_like(noise), timesteps)
            # conditional_blend = self.sd.add_noise(conditional_latents, unconditional_latents, timesteps)
            # unconditional_blend = self.sd.add_noise(unconditional_latents, conditional_latents, timesteps)

            # target_noise = noise + unconditional_signal

            conditional_noisy_latents = self.sd.add_noise(
                mean_latents,
                noise,
                timesteps
            ).detach()

            unconditional_noisy_latents = self.sd.add_noise(
                mean_latents,
                noise,
                timesteps
            ).detach()

            # Disable the LoRA network so we can predict parent network knowledge without it
            self.network.is_active = False
            self.sd.unet.eval()

            # Predict noise to get a baseline of what the parent network wants to do with the latents + noise.
            # This acts as our control to preserve the unaltered parts of the image.
            baseline_prediction = self.sd.predict_noise(
                latents=unconditional_noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                **pred_kwargs  # adapter residuals in here
            ).detach()

            # double up everything to run it through all at once
            cat_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])
            cat_latents = torch.cat([conditional_noisy_latents, conditional_noisy_latents], dim=0)
            cat_timesteps = torch.cat([timesteps, timesteps], dim=0)

            # since we are dividing the polarity from the middle out, we need to double our network
            # weights on training since the convergent point will be at half network strength

            negative_network_weights = [weight * -2.0 for weight in network_weight_list]
            positive_network_weights = [weight * 2.0 for weight in network_weight_list]
            cat_network_weight_list = positive_network_weights + negative_network_weights

            # turn the LoRA network back on.
            self.sd.unet.train()
            self.network.is_active = True

            self.network.multiplier = cat_network_weight_list

        # do our prediction with LoRA active on the scaled guidance latents
        prediction = self.sd.predict_noise(
            latents=cat_latents.to(self.device_torch, dtype=dtype).detach(),
            conditional_embeddings=cat_embeds.to(self.device_torch, dtype=dtype).detach(),
            timestep=cat_timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        )

        pred_pos, pred_neg = torch.chunk(prediction, 2, dim=0)

        pred_pos = pred_pos - baseline_prediction
        pred_neg = pred_neg - baseline_prediction

        pred_loss = torch.nn.functional.mse_loss(
            pred_pos.float(),
            unconditional_diff.float(),
            reduction="none"
        )
        pred_loss = pred_loss.mean([1, 2, 3])

        pred_neg_loss = torch.nn.functional.mse_loss(
            pred_neg.float(),
            conditional_diff.float(),
            reduction="none"
        )
        pred_neg_loss = pred_neg_loss.mean([1, 2, 3])

        loss = (pred_loss + pred_neg_loss) / 2.0

        # loss = self.apply_snr(loss, timesteps)
        loss = loss.mean()
        loss.backward()

        # detach it so parent class can run backward on no grads without throwing error
        loss = loss.detach()
        loss.requires_grad_(True)

        return loss

    def get_guided_loss_masked_polarity(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            # Perform targeted guidance (working title)
            dtype = get_torch_dtype(self.train_config.dtype)

            conditional_latents = batch.latents.to(self.device_torch, dtype=dtype).detach()
            unconditional_latents = batch.unconditional_latents.to(self.device_torch, dtype=dtype).detach()
            inverse_latents = unconditional_latents - (conditional_latents - unconditional_latents)

            mean_latents = (conditional_latents + unconditional_latents) / 2.0

            # unconditional_diff = (unconditional_latents - mean_latents)
            # conditional_diff = (conditional_latents - mean_latents)

            # we need to determine the amount of signal and noise that would be present at the current timestep
            # conditional_signal = self.sd.add_noise(conditional_diff, torch.zeros_like(noise), timesteps)
            # unconditional_signal = self.sd.add_noise(torch.zeros_like(noise), unconditional_diff, timesteps)
            # unconditional_signal = self.sd.add_noise(unconditional_diff, torch.zeros_like(noise), timesteps)
            # conditional_blend = self.sd.add_noise(conditional_latents, unconditional_latents, timesteps)
            # unconditional_blend = self.sd.add_noise(unconditional_latents, conditional_latents, timesteps)

            # make a differential mask
            differential_mask = torch.abs(conditional_latents - unconditional_latents)
            max_differential = \
                differential_mask.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            differential_scaler = 1.0 / max_differential
            differential_mask = differential_mask * differential_scaler
            spread_point = 0.1
            # adjust mask to amplify the differential at 0.1
            differential_mask = ((differential_mask - spread_point) * 10.0) + spread_point
            # clip it
            differential_mask = torch.clamp(differential_mask, 0.0, 1.0)

            # target_noise = noise + unconditional_signal

            conditional_noisy_latents = self.sd.add_noise(
                conditional_latents,
                noise,
                timesteps
            ).detach()

            unconditional_noisy_latents = self.sd.add_noise(
                unconditional_latents,
                noise,
                timesteps
            ).detach()

            inverse_noisy_latents = self.sd.add_noise(
                inverse_latents,
                noise,
                timesteps
            ).detach()

            # Disable the LoRA network so we can predict parent network knowledge without it
            self.network.is_active = False
            self.sd.unet.eval()

            # Predict noise to get a baseline of what the parent network wants to do with the latents + noise.
            # This acts as our control to preserve the unaltered parts of the image.
            # baseline_prediction = self.sd.predict_noise(
            #     latents=unconditional_noisy_latents.to(self.device_torch, dtype=dtype).detach(),
            #     conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype).detach(),
            #     timestep=timesteps,
            #     guidance_scale=1.0,
            #     **pred_kwargs  # adapter residuals in here
            # ).detach()

            # double up everything to run it through all at once
            cat_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])
            cat_latents = torch.cat([conditional_noisy_latents, unconditional_noisy_latents], dim=0)
            cat_timesteps = torch.cat([timesteps, timesteps], dim=0)

            # since we are dividing the polarity from the middle out, we need to double our network
            # weights on training since the convergent point will be at half network strength

            negative_network_weights = [weight * -1.0 for weight in network_weight_list]
            positive_network_weights = [weight * 1.0 for weight in network_weight_list]
            cat_network_weight_list = positive_network_weights + negative_network_weights

            # turn the LoRA network back on.
            self.sd.unet.train()
            self.network.is_active = True

            self.network.multiplier = cat_network_weight_list

        # do our prediction with LoRA active on the scaled guidance latents
        prediction = self.sd.predict_noise(
            latents=cat_latents.to(self.device_torch, dtype=dtype).detach(),
            conditional_embeddings=cat_embeds.to(self.device_torch, dtype=dtype).detach(),
            timestep=cat_timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        )

        pred_pos, pred_neg = torch.chunk(prediction, 2, dim=0)

        # create a loss to balance the mean to 0 between the two predictions
        differential_mean_pred_loss = torch.abs(pred_pos - pred_neg).mean([1, 2, 3]) ** 2.0

        # pred_pos = pred_pos - baseline_prediction
        # pred_neg = pred_neg - baseline_prediction

        pred_loss = torch.nn.functional.mse_loss(
            pred_pos.float(),
            noise.float(),
            reduction="none"
        )
        # apply mask
        pred_loss = pred_loss * (1.0 + differential_mask)
        pred_loss = pred_loss.mean([1, 2, 3])

        pred_neg_loss = torch.nn.functional.mse_loss(
            pred_neg.float(),
            noise.float(),
            reduction="none"
        )
        # apply inverse mask
        pred_neg_loss = pred_neg_loss * (1.0 - differential_mask)
        pred_neg_loss = pred_neg_loss.mean([1, 2, 3])

        # make a loss to balance to losses of the pos and neg so they are equal
        # differential_mean_loss_loss = torch.abs(pred_loss - pred_neg_loss)
        #
        # differential_mean_loss = differential_mean_pred_loss + differential_mean_loss_loss
        #
        # # add a multiplier to balancing losses to make them the top priority
        # differential_mean_loss = differential_mean_loss

        # remove the grads from the negative as it is only a balancing loss
        # pred_neg_loss = pred_neg_loss.detach()

        # loss = pred_loss + pred_neg_loss + differential_mean_loss
        loss = pred_loss + pred_neg_loss

        # loss = self.apply_snr(loss, timesteps)
        loss = loss.mean()
        loss.backward()

        # detach it so parent class can run backward on no grads without throwing error
        loss = loss.detach()
        loss.requires_grad_(True)

        return loss

    def get_prior_prediction(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            **kwargs
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        is_ip_adapter = False
        was_ip_adapter_active = False
        if self.adapter is not None and isinstance(self.adapter, IPAdapter):
            is_ip_adapter = True
            was_ip_adapter_active = self.adapter.is_active
            self.adapter.is_active = False

        # do a prediction here so we can match its output with network multiplier set to 0.0
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)

            embeds_to_use = conditional_embeds.clone().detach()
            # handle clip vision adapter by removing triggers from prompt and replacing with the class name
            if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
                prompt_list = batch.get_caption_list()
                for idx, prompt in enumerate(prompt_list):
                    prompt = self.adapter.inject_trigger_class_name_into_prompt(prompt)
                    prompt_list[idx] = prompt

                embeds_to_use = self.sd.encode_prompt(
                    prompt,
                    long_prompts=self.do_long_prompts).to(
                    self.device_torch,
                    dtype=dtype).detach()

            # dont use network on this
            # self.network.multiplier = 0.0
            self.sd.unet.eval()

            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                **pred_kwargs  # adapter residuals in here
            )
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']

            if is_ip_adapter:
                self.adapter.is_active = was_ip_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def hook_train_loop(self, batch: 'DataLoaderBatchDTO'):
        self.timer.start('preprocess_batch')
        batch = self.preprocess_batch(batch)
        dtype = get_torch_dtype(self.train_config.dtype)
        noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
        network_weight_list = batch.get_network_weight_list()
        if self.train_config.single_item_batching:
            network_weight_list = network_weight_list + network_weight_list

        has_adapter_img = batch.control_tensor is not None
        has_clip_image = batch.clip_image_tensor is not None

        if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not has_clip_image and has_adapter_img:
            raise ValueError("IPAdapter control image is now 'clip_image_path' instead of 'control_path'. Please update your dataset config ")

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

        is_reg = False
        with torch.no_grad():
            loss_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            for idx, file_item in enumerate(batch.file_items):
                if file_item.is_reg:
                    loss_multiplier[idx] = loss_multiplier[idx] * self.train_config.reg_weight
                    is_reg = True

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

            clip_images = None
            if has_clip_image:
                with self.timer('get_clip_images'):
                    # todo move this to data loader
                    if batch.clip_image_tensor is not None:
                        clip_images = batch.clip_image_tensor.to(self.device_torch, dtype=dtype).detach()


            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
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
                adapter_strength_min = 0.4
                adapter_strength_max = 0.7
                # adapter_strength_min = 0.9
                # adapter_strength_max = 1.1

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

            if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
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

        prompts_1 = conditioned_prompts
        prompts_2 = None
        if self.train_config.short_and_long_captions_encoder_split and self.sd.is_xl:
            prompts_1 = batch.get_caption_short_list()
            prompts_2 = conditioned_prompts

            # make the batch splits
        if self.train_config.single_item_batching:
            if self.model_config.refiner_name_or_path is not None:
                raise ValueError("Single item batching is not supported when training the refiner")
            batch_size = noisy_latents.shape[0]
            # chunk/split everything
            noisy_latents_list = torch.chunk(noisy_latents, batch_size, dim=0)
            noise_list = torch.chunk(noise, batch_size, dim=0)
            timesteps_list = torch.chunk(timesteps, batch_size, dim=0)
            conditioned_prompts_list = [[prompt] for prompt in prompts_1]
            if imgs is not None:
                imgs_list = torch.chunk(imgs, batch_size, dim=0)
            else:
                imgs_list = [None for _ in range(batch_size)]
            if adapter_images is not None:
                adapter_images_list = torch.chunk(adapter_images, batch_size, dim=0)
            else:
                adapter_images_list = [None for _ in range(batch_size)]
            if clip_images is not None:
                clip_images_list = torch.chunk(clip_images, batch_size, dim=0)
            else:
                clip_images_list = [None for _ in range(batch_size)]
            mask_multiplier_list = torch.chunk(mask_multiplier, batch_size, dim=0)
            if prompts_2 is None:
                prompt_2_list = [None for _ in range(batch_size)]
            else:
                prompt_2_list = [[prompt] for prompt in prompts_2]

        else:
            noisy_latents_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditioned_prompts_list = [prompts_1]
            imgs_list = [imgs]
            adapter_images_list = [adapter_images]
            clip_images_list = [clip_images]
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for noisy_latents, noise, timesteps, conditioned_prompts, imgs, adapter_images, clip_images, mask_multiplier, prompt_2 in zip(
                noisy_latents_list,
                noise_list,
                timesteps_list,
                conditioned_prompts_list,
                imgs_list,
                adapter_images_list,
                clip_images_list,
                mask_multiplier_list,
                prompt_2_list
        ):
            if self.train_config.negative_prompt is not None:
                # add negative prompt
                conditioned_prompts = conditioned_prompts + [self.train_config.negative_prompt for x in
                                                             range(len(conditioned_prompts))]
                if prompt_2 is not None:
                    prompt_2 = prompt_2 + [self.train_config.negative_prompt for x in range(len(prompt_2))]

            with network:
                # encode clip adapter here so embeds are active for tokenizer
                if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                    with self.timer('encode_clip_vision_embeds'):
                        if has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True
                            )
                        else:
                            # just do a blank one
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                torch.zeros(
                                    (noisy_latents.shape[0], 3, 512, 512),
                                    device=self.device_torch, dtype=dtype
                                ),
                                is_training=True
                            )
                        # it will be injected into the tokenizer when called
                        self.adapter(conditional_clip_embeds)

                with self.timer('encode_prompt'):
                    if grad_on_text_encoder:
                        with torch.set_grad_enabled(True):
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts).to(
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
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts).to(
                                self.device_torch,
                                dtype=dtype)

                        # detach the embeddings
                        conditional_embeds = conditional_embeds.detach()

                # flush()
                pred_kwargs = {}
                if has_adapter_img and (
                        (self.adapter and isinstance(self.adapter, T2IAdapter)) or self.assistant_adapter):
                    with torch.set_grad_enabled(self.adapter is not None):
                        adapter = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
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

                            pred_kwargs['down_intrablock_additional_residuals'] = down_block_additional_residuals


                if self.adapter and isinstance(self.adapter, IPAdapter):
                    with self.timer('encode_adapter_embeds'):
                        if has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True
                            )
                        elif is_reg:
                            # we will zero it out in the img embedder
                            clip_images = torch.zeros(
                                (noisy_latents.shape[0], 3, 512, 512),
                                device=self.device_torch, dtype=dtype
                            ).detach()
                            # drop will zero it out
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images,
                                drop=True,
                                is_training=True
                            )
                        else:
                            raise ValueError("Adapter images now must be loaded with dataloader or be a reg image")

                    if not self.adapter_config.train_image_encoder:
                        # we are not training the image encoder, so we need to detach the embeds
                        conditional_clip_embeds = conditional_clip_embeds.detach()


                    with self.timer('encode_adapter'):
                        conditional_embeds = self.adapter(conditional_embeds.detach(), conditional_clip_embeds)

                prior_pred = None

                do_reg_prior = False
                if is_reg and (self.network is not None or self.adapter is not None):
                    # we are doing a reg image and we have a network or adapter
                    do_reg_prior = True

                do_inverted_masked_prior = False
                if self.train_config.inverted_mask_prior and batch.mask_tensor is not None:
                    do_inverted_masked_prior = True

                if ((has_adapter_img and self.assistant_adapter and match_adapter_assist) or self.do_prior_prediction or do_reg_prior or do_inverted_masked_prior):
                    with self.timer('prior predict'):
                        prior_pred = self.get_prior_prediction(
                            noisy_latents=noisy_latents,
                            conditional_embeds=conditional_embeds,
                            match_adapter_assist=match_adapter_assist,
                            network_weight_list=network_weight_list,
                            timesteps=timesteps,
                            pred_kwargs=pred_kwargs,
                            noise=noise,
                            batch=batch,
                        )

                self.before_unet_predict()
                # do a prior pred if we have an unconditional image, we will swap out the giadance later
                if batch.unconditional_latents is not None or self.do_guided_loss:
                    # do guided loss
                    loss = self.get_guided_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                    )

                else:
                    with self.timer('predict_unet'):
                        noise_pred = self.sd.predict_noise(
                            latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                            timestep=timesteps,
                            guidance_scale=1.0,
                            **pred_kwargs
                        )
                    self.after_unet_predict()

                    with self.timer('calculate_loss'):
                        noise = noise.to(self.device_torch, dtype=dtype).detach()
                        loss = self.calculate_loss(
                            noise_pred=noise_pred,
                            noise=noise,
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=mask_multiplier,
                            prior_pred=prior_pred,
                        )
                # check if nan
                if torch.isnan(loss):
                    raise ValueError("loss is nan")

                with self.timer('backward'):
                    # todo we have multiplier seperated. works for now as res are not in same batch, but need to change
                    loss = loss * loss_multiplier.mean()
                    # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                    # it will destroy the gradients. This is because the network is a context manager
                    # and will change the multipliers back to 0.0 when exiting. They will be
                    # 0.0 for the backward pass and the gradients will be 0.0
                    # I spent weeks on fighting this. DON'T DO IT
                    # with fsdp_overlap_step_with_backward():
                    loss.backward()
        # flush()

        if not self.is_grad_accumulation_step:
            torch.nn.utils.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            # only step if we are not accumulating
            with self.timer('optimizer_step'):
                # apply gradients
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # gradient accumulation. Just a place for breakpoint
            pass

        # TODO Should we only step scheduler on grad step? If so, need to recalculate last step
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()
        if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
            with self.timer('restore_adapter'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.adapter.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )

        self.end_of_training_loop()

        return loss_dict
