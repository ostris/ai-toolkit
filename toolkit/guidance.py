import torch
from typing import Literal
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype

GuidanceType = Literal["targeted", "polarity", "targeted_polarity"]

DIFFERENTIAL_SCALER = 0.2
# DIFFERENTIAL_SCALER = 0.25


def get_differential_mask(
        conditional_latents: torch.Tensor,
        unconditional_latents: torch.Tensor,
        threshold: float = 0.2
):
    # make a differential mask
    differential_mask = torch.abs(conditional_latents - unconditional_latents)
    max_differential = \
        differential_mask.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    differential_scaler = 1.0 / max_differential
    differential_mask = differential_mask * differential_scaler

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
        sd.network.is_active = False
        sd.unet.eval()

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

# This targets only the positive differential
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
        # Perform targeted guidance (working title)
        dtype = get_torch_dtype(sd.torch_dtype)
        device = sd.device_torch

        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

        unconditional_diff = (unconditional_latents - conditional_latents)

        diff_mask = get_differential_mask(
            conditional_latents,
            unconditional_latents,
            threshold=0.1
        )

        # this is a magic number I spent weeks deducing. It works and I have no idea why.
        # unconditional_diff_noise = unconditional_diff * DIFFERENTIAL_SCALER

        inputs_abs_mean = torch.abs(conditional_latents).mean(dim=[1, 2, 3], keepdim=True)
        noise_abs_mean = torch.abs(noise).mean(dim=[1, 2, 3], keepdim=True)
        diff_noise_scaler = noise_abs_mean / inputs_abs_mean
        unconditional_diff_noise = unconditional_diff * diff_noise_scaler

        baseline_noisy_latents = sd.add_noise(
            conditional_latents,
            noise,
            timesteps
        ).detach()

        noisy_latents = sd.add_noise(
            conditional_latents,
            # noise + unconditional_diff_noise,
            noise,
            timesteps
        ).detach()
        # Disable the LoRA network so we can predict parent network knowledge without it
        sd.network.is_active = False
        sd.unet.eval()

        # Predict noise to get a baseline of what the parent network wants to do with the latents + noise.
        # This acts as our control to preserve the unaltered parts of the image.
        baseline_prediction = sd.predict_noise(
            latents=baseline_noisy_latents.to(device, dtype=dtype).detach(),
            conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        ).detach()
        # turn the LoRA network back on.
        sd.unet.train()
        sd.network.is_active = True

        sd.network.multiplier = network_weight_list

        unmasked_baseline_prediction = baseline_prediction * (1.0 - diff_mask)
        masked_noise = noise * diff_mask
        # pred_target = unmasked_noise + unconditional_diff_noise

    # do our prediction with LoRA active on the scaled guidance latents
    prediction = sd.predict_noise(
        latents=noisy_latents.to(device, dtype=dtype).detach(),
        conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
        timestep=timesteps,
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )

    prediction = prediction - unmasked_baseline_prediction
    # prediction = prediction - baseline_prediction

    baseline_loss = torch.nn.functional.mse_loss(
        baseline_prediction.float(),
        noise.float(),
        reduction="none"
    )
    baseline_loss = baseline_loss * (1.0 - diff_mask)
    baseline_loss = baseline_loss.mean([1, 2, 3])

    # loss = torch.nn.functional.l1_loss(
    loss = torch.nn.functional.mse_loss(
        prediction.float(),
        masked_noise.float(),
        reduction="none"
    )
    loss = loss * diff_mask
    loss = loss.mean([1, 2, 3])
    primary_loss_scaler = 1.0

    loss = (loss * primary_loss_scaler) + baseline_loss

    loss = loss.mean()

    loss.backward()

    # detach it so parent class can run backward on no grads without throwing error
    loss = loss.detach()
    loss.requires_grad_(True)

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
    pred_loss = pred_loss.mean([1, 2, 3])

    pred_neg_loss = torch.nn.functional.mse_loss(
        pred_neg.float(),
        noise.float(),
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
    else:
        raise NotImplementedError(f"Guidance type {guidance_type} is not implemented")
