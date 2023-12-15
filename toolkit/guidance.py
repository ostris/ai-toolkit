import torch
from typing import Literal

from toolkit.basic import value_map
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype

GuidanceType = Literal["targeted", "polarity", "targeted_polarity", "direct"]

DIFFERENTIAL_SCALER = 0.2


# DIFFERENTIAL_SCALER = 0.25


def get_differential_mask(
        conditional_latents: torch.Tensor,
        unconditional_latents: torch.Tensor,
        threshold: float = 0.2,
        gradient: bool = False,
):
    # make a differential mask
    differential_mask = torch.abs(conditional_latents - unconditional_latents)
    max_differential = \
        differential_mask.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    differential_scaler = 1.0 / max_differential
    differential_mask = differential_mask * differential_scaler

    if gradient:
        # wew need to scale it to 0-1
        # differential_mask = differential_mask - differential_mask.min()
        # differential_mask = differential_mask / differential_mask.max()
        # add 0.2 threshold to both sides and clip
        differential_mask = value_map(
            differential_mask,
            differential_mask.min(),
            differential_mask.max(),
            0 - threshold,
            1 + threshold
        )
        differential_mask = torch.clamp(differential_mask, 0.0, 1.0)
    else:

        # make everything less than 0.2 be 0.0 and everything else be 1.0
        differential_mask = torch.where(
            differential_mask < threshold,
            torch.zeros_like(differential_mask),
            torch.ones_like(differential_mask)
        )
    return differential_mask


def get_targeted_polarity_loss(
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: 'DataLoaderBatchDTO',
        noise: torch.Tensor,
        sd: 'StableDiffusion',
        **kwargs
):
    dtype = get_torch_dtype(sd.torch_dtype)
    device = sd.device_torch
    with torch.no_grad():
        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

        # inputs_abs_mean = torch.abs(conditional_latents).mean(dim=[1, 2, 3], keepdim=True)
        # noise_abs_mean = torch.abs(noise).mean(dim=[1, 2, 3], keepdim=True)
        differential_scaler = DIFFERENTIAL_SCALER

        unconditional_diff = (unconditional_latents - conditional_latents)
        unconditional_diff_noise = unconditional_diff * differential_scaler
        conditional_diff = (conditional_latents - unconditional_latents)
        conditional_diff_noise = conditional_diff * differential_scaler
        conditional_diff_noise = conditional_diff_noise.detach().requires_grad_(False)
        unconditional_diff_noise = unconditional_diff_noise.detach().requires_grad_(False)
        #
        baseline_conditional_noisy_latents = sd.add_noise(
            conditional_latents,
            noise,
            timesteps
        ).detach()

        baseline_unconditional_noisy_latents = sd.add_noise(
            unconditional_latents,
            noise,
            timesteps
        ).detach()

        conditional_noise = noise + unconditional_diff_noise
        unconditional_noise = noise + conditional_diff_noise

        conditional_noisy_latents = sd.add_noise(
            conditional_latents,
            conditional_noise,
            timesteps
        ).detach()

        unconditional_noisy_latents = sd.add_noise(
            unconditional_latents,
            unconditional_noise,
            timesteps
        ).detach()

        # double up everything to run it through all at once
        cat_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])
        cat_latents = torch.cat([conditional_noisy_latents, unconditional_noisy_latents], dim=0)
        cat_timesteps = torch.cat([timesteps, timesteps], dim=0)
        # cat_baseline_noisy_latents = torch.cat(
        #     [baseline_conditional_noisy_latents, baseline_unconditional_noisy_latents],
        #     dim=0
        # )

        # Disable the LoRA network so we can predict parent network knowledge without it
        # sd.network.is_active = False
        # sd.unet.eval()

        # Predict noise to get a baseline of what the parent network wants to do with the latents + noise.
        # This acts as our control to preserve the unaltered parts of the image.
        # baseline_prediction = sd.predict_noise(
        #     latents=cat_baseline_noisy_latents.to(device, dtype=dtype).detach(),
        #     conditional_embeddings=cat_embeds.to(device, dtype=dtype).detach(),
        #     timestep=cat_timesteps,
        #     guidance_scale=1.0,
        #     **pred_kwargs  # adapter residuals in here
        # ).detach()

        # conditional_baseline_prediction, unconditional_baseline_prediction = torch.chunk(baseline_prediction, 2, dim=0)

        # negative_network_weights = [weight * -1.0 for weight in network_weight_list]
        # positive_network_weights = [weight * 1.0 for weight in network_weight_list]
        # cat_network_weight_list = positive_network_weights + negative_network_weights

        # turn the LoRA network back on.
        sd.unet.train()
        # sd.network.is_active = True

        # sd.network.multiplier = cat_network_weight_list

    # do our prediction with LoRA active on the scaled guidance latents
    prediction = sd.predict_noise(
        latents=cat_latents.to(device, dtype=dtype).detach(),
        conditional_embeddings=cat_embeds.to(device, dtype=dtype).detach(),
        timestep=cat_timesteps,
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )

    # prediction = prediction - baseline_prediction

    pred_pos, pred_neg = torch.chunk(prediction, 2, dim=0)
    # pred_pos = pred_pos - conditional_baseline_prediction
    # pred_neg = pred_neg - unconditional_baseline_prediction

    pred_loss = torch.nn.functional.mse_loss(
        pred_pos.float(),
        conditional_noise.float(),
        reduction="none"
    )
    pred_loss = pred_loss.mean([1, 2, 3])

    pred_neg_loss = torch.nn.functional.mse_loss(
        pred_neg.float(),
        unconditional_noise.float(),
        reduction="none"
    )
    pred_neg_loss = pred_neg_loss.mean([1, 2, 3])

    loss = pred_loss + pred_neg_loss

    loss = loss.mean()
    loss.backward()

    # detach it so parent class can run backward on no grads without throwing error
    loss = loss.detach()
    loss.requires_grad_(True)

    return loss

def get_direct_guidance_loss(
        noisy_latents: torch.Tensor,
        conditional_embeds: 'PromptEmbeds',
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: 'DataLoaderBatchDTO',
        noise: torch.Tensor,
        sd: 'StableDiffusion',
        **kwargs
):
    with torch.no_grad():
        # Perform targeted guidance (working title)
        dtype = get_torch_dtype(sd.torch_dtype)
        device = sd.device_torch


        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

        conditional_noisy_latents = sd.add_noise(
            conditional_latents,
            # target_noise,
            noise,
            timesteps
        ).detach()

        unconditional_noisy_latents = sd.add_noise(
            unconditional_latents,
            noise,
            timesteps
        ).detach()
        # turn the LoRA network back on.
        sd.unet.train()
        # sd.network.is_active = True

        # sd.network.multiplier = network_weight_list
    # do our prediction with LoRA active on the scaled guidance latents
    prediction = sd.predict_noise(
        latents=torch.cat([unconditional_noisy_latents, conditional_noisy_latents]).to(device, dtype=dtype).detach(),
        conditional_embeddings=concat_prompt_embeds([conditional_embeds,conditional_embeds]).to(device, dtype=dtype).detach(),
        timestep=torch.cat([timesteps, timesteps]),
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )

    noise_pred_uncond, noise_pred_cond = torch.chunk(prediction, 2, dim=0)

    guidance_scale = 1.0
    guidance_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
    )

    guidance_loss = torch.nn.functional.mse_loss(
        guidance_pred.float(),
        noise.detach().float(),
        reduction="none"
    )

    guidance_loss = guidance_loss.mean([1, 2, 3])

    guidance_loss = guidance_loss.mean()

    # loss = guidance_loss + masked_noise_loss
    loss = guidance_loss

    loss.backward()

    # detach it so parent class can run backward on no grads without throwing error
    loss = loss.detach()
    loss.requires_grad_(True)

    return loss


# targeted
def get_targeted_guidance_loss(
        noisy_latents: torch.Tensor,
        conditional_embeds: 'PromptEmbeds',
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: 'DataLoaderBatchDTO',
        noise: torch.Tensor,
        sd: 'StableDiffusion',
        **kwargs
):
    with torch.no_grad():
        dtype = get_torch_dtype(sd.torch_dtype)
        device = sd.device_torch

        # create the differential mask from the actual tensors
        conditional_imgs = batch.tensor.to(device, dtype=dtype).detach()
        unconditional_imgs = batch.unconditional_tensor.to(device, dtype=dtype).detach()
        differential_mask = torch.abs(conditional_imgs - unconditional_imgs)
        differential_mask = differential_mask - differential_mask.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        differential_mask = differential_mask / differential_mask.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        # differential_mask is (bs, 3, width, height)
        # latents are (bs, 4, width, height)
        # reduce the mean on dim 1 to get a single channel mask and stack it to match latents
        differential_mask = differential_mask.mean(dim=1, keepdim=True)
        differential_mask = torch.cat([differential_mask] * 4, dim=1)

        # scale the mask down to latent size
        differential_mask = torch.nn.functional.interpolate(
            differential_mask,
            size=noisy_latents.shape[2:],
            mode="nearest"
        )

        conditional_noisy_latents = noisy_latents

        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

        # unconditional_as_noise = unconditional_latents - conditional_latents
        # conditional_as_noise = conditional_latents - unconditional_latents

        # Encode the unconditional image into latents
        unconditional_noisy_latents = sd.noise_scheduler.add_noise(
            unconditional_latents,
            noise,
            timesteps
        )
        conditional_noisy_latents = sd.noise_scheduler.add_noise(
            conditional_latents,
            noise,
            timesteps
        )

        # was_network_active = self.network.is_active
        sd.network.is_active = False
        sd.unet.eval()


        # calculate the differential between our conditional (target image) and out unconditional ("bad" image)
        # target_differential = unconditional_noisy_latents - conditional_noisy_latents
        target_differential = unconditional_latents - conditional_latents
        # target_differential = conditional_latents - unconditional_latents

        # scale the target differential by the scheduler
        # todo, scale it the right way
        # target_differential = sd.noise_scheduler.add_noise(
        #     torch.zeros_like(target_differential),
        #     target_differential,
        #     timesteps
        # )

        # noise_abs_mean = torch.abs(noise + 1e-6).mean(dim=[1, 2, 3], keepdim=True)

        # target_differential = target_differential.detach()
        # target_differential_abs_mean = torch.abs(target_differential + 1e-6).mean(dim=[1, 2, 3], keepdim=True)
        # # determins scaler to adjust to same abs mean as noise
        # scaler = noise_abs_mean / target_differential_abs_mean


        target_differential_knowledge = target_differential
        target_differential_knowledge = target_differential_knowledge.detach()

        # add the target differential to the target latents as if it were noise with the scheduler scaled to
        # the current timestep. Scaling the noise here is IMPORTANT and will lead to a blurry targeted area if not done
        # properly
        # guidance_latents = sd.noise_scheduler.add_noise(
        #     conditional_noisy_latents,
        #     target_differential,
        #     timesteps
        # )

        # guidance_latents = conditional_noisy_latents + target_differential
        # target_noise = conditional_noisy_latents + target_differential

        # With LoRA network bypassed, predict noise to get a baseline of what the network
        # wants to do with the latents + noise. Pass our target latents here for the input.
        target_unconditional = sd.predict_noise(
            latents=unconditional_noisy_latents.to(device, dtype=dtype).detach(),
            conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        ).detach()
        # target_conditional = sd.predict_noise(
        #     latents=conditional_noisy_latents.to(device, dtype=dtype).detach(),
        #     conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
        #     timestep=timesteps,
        #     guidance_scale=1.0,
        #     **pred_kwargs  # adapter residuals in here
        # ).detach()

        # we calculate the networks current knowledge so we do not overlearn what we know
        # parent_knowledge = target_unconditional - target_conditional
        # parent_knowledge = parent_knowledge.detach()
        # del target_conditional
        # del target_unconditional

        # we now have the differential noise prediction needed to create our convergence target
        # target_unknown_knowledge = target_differential + parent_knowledge
        # del parent_knowledge
        prior_prediction_loss = torch.nn.functional.mse_loss(
            target_unconditional.float(),
            noise.float(),
            reduction="none"
        ).detach().clone()

    # turn the LoRA network back on.
    sd.unet.train()
    sd.network.is_active = True
    sd.network.multiplier = network_weight_list

    # with LoRA active, predict the noise with the scaled differential latents added. This will allow us
    # the opportunity to predict the differential + noise that was added to the latents.
    prediction_conditional = sd.predict_noise(
        latents=conditional_noisy_latents.to(device, dtype=dtype).detach(),
        conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
        timestep=timesteps,
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )



    # remove the baseline conditional prediction. This will leave only the divergence from the baseline and
    # the prediction of the added differential noise
    # prediction_positive = prediction_unconditional - target_unconditional
    # current_knowledge = target_unconditional - prediction_conditional
    # current_differential_knowledge = prediction_conditional - target_unconditional

    # current_unknown_knowledge = parent_knowledge - current_knowledge
    #
    # current_unknown_knowledge_abs_mean = torch.abs(current_unknown_knowledge + 1e-6).mean(dim=[1, 2, 3], keepdim=True)
    # current_unknown_knowledge_std = current_unknown_knowledge / current_unknown_knowledge_abs_mean


    # for loss, we target ONLY the unscaled differential between our conditional and unconditional latents
    # this is the diffusion training process.
    # This will guide the network to make identical predictions it previously did for everything EXCEPT our
    # differential between the conditional and unconditional images

    # positive_loss = torch.nn.functional.mse_loss(
    #     current_differential_knowledge.float(),
    #     target_differential_knowledge.float(),
    #     reduction="none"
    # )

    normal_loss = torch.nn.functional.mse_loss(
        prediction_conditional.float(),
        noise.float(),
        reduction="none"
    )
    #
    # # scale positive and neutral loss to the same scale
    # positive_loss_abs_mean = torch.abs(positive_loss + 1e-6).mean(dim=[1, 2, 3], keepdim=True)
    # normal_loss_abs_mean = torch.abs(normal_loss + 1e-6).mean(dim=[1, 2, 3], keepdim=True)
    # scaler = normal_loss_abs_mean / positive_loss_abs_mean
    # positive_loss = positive_loss * scaler

    # positive_loss = positive_loss * differential_mask
    # positive_loss = positive_loss
    # masked_normal_loss = normal_loss * differential_mask

    prior_loss = torch.abs(
        normal_loss.float() - prior_prediction_loss.float(),
    # ) * (1 - differential_mask)
    )

    decouple = True

    # positive_loss_full = positive_loss
    # prior_loss_full = prior_loss
    #
    # current_scaler = (prior_loss_full.max() / positive_loss_full.max())
    # # positive_loss = positive_loss * current_scaler
    # avg_scaler_arr.append(current_scaler.item())
    # avg_scaler = sum(avg_scaler_arr) / len(avg_scaler_arr)
    # print(f"avg scaler: {avg_scaler}, current scaler: {current_scaler.item()}")
    # # remove extra scalers more than 100
    # if len(avg_scaler_arr) > 100:
    #     avg_scaler_arr.pop(0)
    #
    # # positive_loss = positive_loss * avg_scaler
    # positive_loss = positive_loss * avg_scaler * 0.1

    if decouple:
        # positive_loss = positive_loss.mean([1, 2, 3])
        prior_loss = prior_loss.mean([1, 2, 3])
        # masked_normal_loss = masked_normal_loss.mean([1, 2, 3])
        positive_loss = prior_loss
        # positive_loss = positive_loss + prior_loss
    else:

        # positive_loss = positive_loss + prior_loss
        positive_loss = prior_loss
        positive_loss = positive_loss.mean([1, 2, 3])

    # positive_loss = positive_loss + adain_loss.mean([1, 2, 3])
    # send it backwards BEFORE switching network polarity
    # positive_loss = self.apply_snr(positive_loss, timesteps)
    positive_loss = positive_loss.mean()
    positive_loss.backward()
    # loss = positive_loss.detach() + negative_loss.detach()
    loss = positive_loss.detach()

    # add a grad so other backward does not fail
    loss.requires_grad_(True)

    # restore network
    sd.network.multiplier = network_weight_list

    return loss

def get_guided_loss_polarity(
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: 'DataLoaderBatchDTO',
        noise: torch.Tensor,
        sd: 'StableDiffusion',
        **kwargs
):
    dtype = get_torch_dtype(sd.torch_dtype)
    device = sd.device_torch
    with torch.no_grad():
        dtype = get_torch_dtype(dtype)

        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

        conditional_noisy_latents = sd.add_noise(
            conditional_latents,
            noise,
            timesteps
        ).detach()

        unconditional_noisy_latents = sd.add_noise(
            unconditional_latents,
            noise,
            timesteps
        ).detach()

        # double up everything to run it through all at once
        cat_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])
        cat_latents = torch.cat([conditional_noisy_latents, unconditional_noisy_latents], dim=0)
        cat_timesteps = torch.cat([timesteps, timesteps], dim=0)

        negative_network_weights = [weight * -1.0 for weight in network_weight_list]
        positive_network_weights = [weight * 1.0 for weight in network_weight_list]
        cat_network_weight_list = positive_network_weights + negative_network_weights

        # turn the LoRA network back on.
        sd.unet.train()
        sd.network.is_active = True

        sd.network.multiplier = cat_network_weight_list

    # do our prediction with LoRA active on the scaled guidance latents
    prediction = sd.predict_noise(
        latents=cat_latents.to(device, dtype=dtype).detach(),
        conditional_embeddings=cat_embeds.to(device, dtype=dtype).detach(),
        timestep=cat_timesteps,
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )

    pred_pos, pred_neg = torch.chunk(prediction, 2, dim=0)

    pred_loss = torch.nn.functional.mse_loss(
        pred_pos.float(),
        noise.float(),
        reduction="none"
    )
    # pred_loss = pred_loss.mean([1, 2, 3])

    pred_neg_loss = torch.nn.functional.mse_loss(
        pred_neg.float(),
        noise.float(),
        reduction="none"
    )

    loss = pred_loss + pred_neg_loss

    loss = loss.mean([1, 2, 3])
    loss = loss.mean()
    loss.backward()

    # detach it so parent class can run backward on no grads without throwing error
    loss = loss.detach()
    loss.requires_grad_(True)

    return loss


# this processes all guidance losses based on the batch information
def get_guidance_loss(
        noisy_latents: torch.Tensor,
        conditional_embeds: 'PromptEmbeds',
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: 'DataLoaderBatchDTO',
        noise: torch.Tensor,
        sd: 'StableDiffusion',
        **kwargs
):
    # TODO add others and process individual batch items separately
    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type

    if guidance_type == "targeted":
        return get_targeted_guidance_loss(
            noisy_latents,
            conditional_embeds,
            match_adapter_assist,
            network_weight_list,
            timesteps,
            pred_kwargs,
            batch,
            noise,
            sd,
            **kwargs
        )
    elif guidance_type == "polarity":
        return get_guided_loss_polarity(
            noisy_latents,
            conditional_embeds,
            match_adapter_assist,
            network_weight_list,
            timesteps,
            pred_kwargs,
            batch,
            noise,
            sd,
            **kwargs
        )

    elif guidance_type == "targeted_polarity":
        return get_targeted_polarity_loss(
            noisy_latents,
            conditional_embeds,
            match_adapter_assist,
            network_weight_list,
            timesteps,
            pred_kwargs,
            batch,
            noise,
            sd,
            **kwargs
        )
    elif guidance_type == "direct":
        return get_direct_guidance_loss(
            noisy_latents,
            conditional_embeds,
            match_adapter_assist,
            network_weight_list,
            timesteps,
            pred_kwargs,
            batch,
            noise,
            sd,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Guidance type {guidance_type} is not implemented")
