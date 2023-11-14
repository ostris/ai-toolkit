from collections import OrderedDict
from typing import Union
from diffusers import T2IAdapter

from toolkit import train_tools
from toolkit.basic import value_map, adain
from toolkit.config_modules import GuidanceConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.ip_adapter import IPAdapter
from toolkit.prompt_utils import PromptEmbeds
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
        if self.train_config.inverted_mask_prior:
            self.do_prior_prediction = True

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

        if self.train_config.match_noise_norm:
            # match the norm of the noise
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)

        if self.train_config.inverted_mask_prior:
            # we need to make the noise prediction be a masked blending of noise and prior_pred
            prior_mask_multiplier = 1.0 - mask_multiplier
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

        if self.train_config.inverted_mask_prior:
            # to a loss to unmasked areas of the prior for unmasked regularization
            prior_loss = torch.nn.functional.mse_loss(
                prior_pred.float(),
                pred.float(),
                reduction="none"
            )
            prior_loss = prior_loss * prior_mask_multiplier * self.train_config.inverted_mask_prior_multiplier
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
        with torch.no_grad():
            # Perform targeted guidance (working title)
            conditional_noisy_latents = noisy_latents  # target images
            dtype = get_torch_dtype(self.train_config.dtype)

            if batch.unconditional_latents is not None:
                # unconditional latents are the "neutral" images. Add noise here identical to
                # the noise added to the conditional latents, at the same timesteps
                unconditional_noisy_latents = self.sd.noise_scheduler.add_noise(
                    batch.unconditional_latents, noise, timesteps
                )

            # calculate the differential between our conditional (target image) and out unconditional (neutral image)
            target_differential_noise = unconditional_noisy_latents - conditional_noisy_latents
            target_differential_noise = target_differential_noise.detach()

            # add the target differential to the target latents as if it were noise with the scheduler, scaled to
            # the current timestep. Scaling the noise here is important as it scales our guidance to the current
            # timestep. This is the key to making the guidance work.
            guidance_latents = self.sd.noise_scheduler.add_noise(
                conditional_noisy_latents,
                target_differential_noise,
                timesteps
            )

            # Disable the LoRA network so we can predict parent network knowledge without it
            self.network.is_active = False
            self.sd.unet.eval()

            # Predict noise to get a baseline of what the parent network wants to do with the latents + noise.
            # This acts as our control to preserve the unaltered parts of the image.
            baseline_prediction = self.sd.predict_noise(
                latents=guidance_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                **pred_kwargs  # adapter residuals in here
            ).detach()

        # turn the LoRA network back on.
        self.sd.unet.train()
        self.network.is_active = True
        self.network.multiplier = network_weight_list

        # do our prediction with LoRA active on the scaled guidance latents
        prediction = self.sd.predict_noise(
            latents=guidance_latents.to(self.device_torch, dtype=dtype).detach(),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        )

        # remove the baseline prediction from our prediction to get the differential between the two
        # all that should be left is the differential between the conditional and unconditional images
        pred_differential_noise = prediction - baseline_prediction

        # for loss, we target ONLY the unscaled differential between our conditional and unconditional latents
        # not the timestep scaled noise that was added. This is the diffusion training process.
        # This will guide the network to make identical predictions it previously did for everything EXCEPT our
        # differential between the conditional and unconditional images (target)
        loss = torch.nn.functional.mse_loss(
            pred_differential_noise.float(),
            target_differential_noise.float(),
            reduction="none"
        )

        loss = loss.mean([1, 2, 3])
        loss = self.apply_snr(loss, timesteps)
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
        # do a prediction here so we can match its output with network multiplier set to 0.0
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)
            # dont use network on this
            # self.network.multiplier = 0.0
            was_network_active = self.network.is_active
            self.network.is_active = False
            self.sd.unet.eval()
            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                **pred_kwargs  # adapter residuals in here
            )
            self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_block_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_block_additional_residuals']
            # restore network
            # self.network.multiplier = network_weight_list
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
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for noisy_latents, noise, timesteps, conditioned_prompts, imgs, adapter_images, mask_multiplier, prompt_2 in zip(
                noisy_latents_list,
                noise_list,
                timesteps_list,
                conditioned_prompts_list,
                imgs_list,
                adapter_images_list,
                mask_multiplier_list,
                prompt_2_list
        ):

            with network:
                with self.timer('encode_prompt'):
                    if grad_on_text_encoder:
                        with torch.set_grad_enabled(True):
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=True).to(
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
                                long_prompts=True).to(
                                self.device_torch,
                                dtype=dtype)

                        # detach the embeddings
                        conditional_embeds = conditional_embeds.detach()

                # flush()
                pred_kwargs = {}
                if has_adapter_img and (
                        (self.adapter and isinstance(self.adapter, T2IAdapter)) or self.assistant_adapter):
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

                prior_pred = None
                if (has_adapter_img and self.assistant_adapter and match_adapter_assist) or self.do_prior_prediction:
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

                if has_adapter_img and self.adapter and isinstance(self.adapter, IPAdapter):
                    with self.timer('encode_adapter'):
                        with torch.no_grad():
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(adapter_images)
                        conditional_embeds = self.adapter(conditional_embeds, conditional_clip_embeds)

                self.before_unet_predict()
                # do a prior pred if we have an unconditional image, we will swap out the giadance later
                if batch.unconditional_latents is not None:
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

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )

        self.end_of_training_loop()

        return loss_dict
