import torch
from typing import Literal, Optional

from toolkit.basic import value_map
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
from toolkit.config_modules import TrainConfig

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
    if len(differential_mask.shape) == 4:
        max_differential = \
            differential_mask.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    elif len(differential_mask.shape) == 5:
        max_differential = \
            differential_mask.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].max(dim=4, keepdim=True)[0]
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
        unconditional_embeds: Optional[PromptEmbeds] = None,
        mask_multiplier=None,
        prior_pred=None,
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
    if unconditional_embeds is not None:
        unconditional_embeds = unconditional_embeds.to(device, dtype=dtype).detach()
        unconditional_embeds = concat_prompt_embeds([unconditional_embeds, unconditional_embeds])

    prediction = sd.predict_noise(
        latents=torch.cat([unconditional_noisy_latents, conditional_noisy_latents]).to(device, dtype=dtype).detach(),
        conditional_embeddings=concat_prompt_embeds([conditional_embeds,conditional_embeds]).to(device, dtype=dtype).detach(),
        unconditional_embeddings=unconditional_embeds,
        timestep=torch.cat([timesteps, timesteps]),
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )

    noise_pred_uncond, noise_pred_cond = torch.chunk(prediction, 2, dim=0)

    guidance_scale = 1.1
    guidance_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
    )

    guidance_loss = torch.nn.functional.mse_loss(
        guidance_pred.float(),
        noise.detach().float(),
        reduction="none"
    )
    if mask_multiplier is not None:
        guidance_loss = guidance_loss * mask_multiplier

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

        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

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

        target_differential = unconditional_latents - conditional_latents
        # scale our loss by the differential scaler
        target_differential_abs = target_differential.abs()
        target_differential_abs_min = \
        target_differential_abs.min(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        target_differential_abs_max = \
            target_differential_abs.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        min_guidance = 1.0
        max_guidance = 2.0

        differential_scaler = value_map(
            target_differential_abs,
            target_differential_abs_min,
            target_differential_abs_max,
            min_guidance,
            max_guidance
        ).detach()


        # With LoRA network bypassed, predict noise to get a baseline of what the network
        # wants to do with the latents + noise. Pass our target latents here for the input.
        target_unconditional = sd.predict_noise(
            latents=unconditional_noisy_latents.to(device, dtype=dtype).detach(),
            conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        ).detach()
        prior_prediction_loss = torch.nn.functional.mse_loss(
            target_unconditional.float(),
            noise.float(),
            reduction="none"
        ).detach().clone()

    # turn the LoRA network back on.
    sd.unet.train()
    sd.network.is_active = True
    sd.network.multiplier = network_weight_list + [x + -1.0 for x in network_weight_list]

    # with LoRA active, predict the noise with the scaled differential latents added. This will allow us
    # the opportunity to predict the differential + noise that was added to the latents.
    prediction = sd.predict_noise(
        latents=torch.cat([conditional_noisy_latents, unconditional_noisy_latents], dim=0).to(device, dtype=dtype).detach(),
        conditional_embeddings=concat_prompt_embeds([conditional_embeds, conditional_embeds]).to(device, dtype=dtype).detach(),
        timestep=torch.cat([timesteps, timesteps], dim=0),
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )

    prediction_conditional, prediction_unconditional = torch.chunk(prediction, 2, dim=0)

    conditional_loss = torch.nn.functional.mse_loss(
        prediction_conditional.float(),
        noise.float(),
        reduction="none"
    )

    unconditional_loss = torch.nn.functional.mse_loss(
        prediction_unconditional.float(),
        noise.float(),
        reduction="none"
    )

    positive_loss = torch.abs(
        conditional_loss.float() - prior_prediction_loss.float(),
    )
    # scale our loss by the differential scaler
    positive_loss = positive_loss * differential_scaler

    positive_loss = positive_loss.mean([1, 2, 3])

    polar_loss = torch.abs(
        conditional_loss.float() - unconditional_loss.float(),
    ).mean([1, 2, 3])


    positive_loss = positive_loss.mean() + polar_loss.mean()


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
        train_config: 'TrainConfig',
        scaler=None,
        **kwargs
):
    dtype = get_torch_dtype(sd.torch_dtype)
    device = sd.device_torch
    with torch.no_grad():
        dtype = get_torch_dtype(dtype)
        noise = noise.to(device, dtype=dtype).detach()

        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

        target_pos = noise
        target_neg = noise

        if sd.is_flow_matching:
            linear_timesteps = any([
                train_config.linear_timesteps,
                train_config.linear_timesteps2,
                train_config.timestep_type == 'linear',
            ])
            
            timestep_type = 'linear' if linear_timesteps else None
            if timestep_type is None:
                timestep_type = train_config.timestep_type
            
            sd.noise_scheduler.set_train_timesteps(
                1000,
                device=device,
                timestep_type=timestep_type,
                latents=conditional_latents
            )
            target_pos = (noise - conditional_latents).detach()
            target_neg = (noise - unconditional_latents).detach()

        conditional_noisy_latents = sd.add_noise(
            conditional_latents,
            noise,
            timesteps
        ).detach()
        conditional_noisy_latents = sd.condition_noisy_latents(conditional_noisy_latents, batch)

        unconditional_noisy_latents = sd.add_noise(
            unconditional_latents,
            noise,
            timesteps
        ).detach()
        unconditional_noisy_latents = sd.condition_noisy_latents(unconditional_noisy_latents, batch)

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
        target_pos.float(),
        reduction="none"
    )
    # pred_loss = pred_loss.mean([1, 2, 3])

    pred_neg_loss = torch.nn.functional.mse_loss(
        pred_neg.float(),
        target_neg.float(),
        reduction="none"
    )

    loss = pred_loss + pred_neg_loss

    loss = loss.mean([1, 2, 3])
    loss = loss.mean()
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # detach it so parent class can run backward on no grads without throwing error
    loss = loss.detach()
    loss.requires_grad_(True)

    return loss



def get_guided_tnt(
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: 'DataLoaderBatchDTO',
        noise: torch.Tensor,
        sd: 'StableDiffusion',
        prior_pred: torch.Tensor = None,
        **kwargs
):
    dtype = get_torch_dtype(sd.torch_dtype)
    device = sd.device_torch
    with torch.no_grad():
        dtype = get_torch_dtype(dtype)
        noise = noise.to(device, dtype=dtype).detach()

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


        # turn the LoRA network back on.
        sd.unet.train()
        if sd.network is not None:
            cat_network_weight_list = [weight for weight in network_weight_list * 2]
            sd.network.multiplier = cat_network_weight_list
            sd.network.is_active = True


    prediction = sd.predict_noise(
        latents=cat_latents.to(device, dtype=dtype).detach(),
        conditional_embeddings=cat_embeds.to(device, dtype=dtype).detach(),
        timestep=cat_timesteps,
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )
    this_prediction, that_prediction = torch.chunk(prediction, 2, dim=0)

    this_loss = torch.nn.functional.mse_loss(
        this_prediction.float(),
        noise.float(),
        reduction="none"
    )

    that_loss = torch.nn.functional.mse_loss(
        that_prediction.float(),
        noise.float(),
        reduction="none"
    )

    this_loss = this_loss.mean([1, 2, 3])
    # negative loss on that
    that_loss = -that_loss.mean([1, 2, 3])

    with torch.no_grad():
        # match that loss with this loss so it is not a negative value and same scale
        that_loss_scaler = torch.abs(this_loss) / torch.abs(that_loss)

    that_loss = that_loss * that_loss_scaler * 0.01

    loss = this_loss + that_loss

    loss = loss.mean()

    loss.backward()

    # detach it so parent class can run backward on no grads without throwing error
    loss = loss.detach()
    loss.requires_grad_(True)

    return loss

def targeted_flow_guidance(
    noisy_latents: torch.Tensor,
    conditional_embeds: 'PromptEmbeds',
    match_adapter_assist: bool,
    network_weight_list: list,
    timesteps: torch.Tensor,
    pred_kwargs: dict,
    batch: 'DataLoaderBatchDTO',
    noise: torch.Tensor,
    sd: 'StableDiffusion',
    unconditional_embeds: Optional[PromptEmbeds] = None,
    mask_multiplier=None,
    prior_pred=None,
    scaler=None,
    train_config=None,
    **kwargs
):
    if not sd.is_flow_matching:
        raise ValueError("targeted_flow only works on flow matching models")
    dtype = get_torch_dtype(sd.torch_dtype)
    device = sd.device_torch
    with torch.no_grad():
        dtype = get_torch_dtype(dtype)
        noise = noise.to(device, dtype=dtype).detach()

        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()
        
        # get a mask on the differential of the latents
        # this will be scaled from 0.0-1.0 with 1.0 being the largest differential
        abs_differential_mask = get_differential_mask(
            conditional_latents,
            unconditional_latents,
            gradient=True
        )
        
        # get noisy latents for both conditional and unconditional predictions
        unconditional_noisy_latents = sd.add_noise(
            unconditional_latents,
            noise,
            timesteps
        ).detach()
        unconditional_noisy_latents = sd.condition_noisy_latents(unconditional_noisy_latents, batch)
        conditional_noisy_latents = sd.add_noise(
            conditional_latents,
            noise,
            timesteps
        ).detach()
        conditional_noisy_latents = sd.condition_noisy_latents(conditional_noisy_latents, batch)
        
        # disable the lora to get a baseline prediction
        sd.network.is_active = False
        sd.unet.eval()
        
        # get a baseline prediction of the model knowledge without the lora network
        # we do this with the unconditional noisy latents
        baseline_prediction = sd.predict_noise(
            latents=unconditional_noisy_latents.to(device, dtype=dtype).detach(),
            conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            **pred_kwargs
        ).detach()
        
        # This is our normal flowmatching target
        # target = noise - latents
        # we need to target the baseline noise but with our conditional latents
        # to do this we first have to determine the baseline_prediction noise by reversing the flowmatching target
        baseline_predicted_noise = baseline_prediction + unconditional_latents
        
        # baseline_predicted_noise is now the noise prediction our model would make with a the unconditional image.
        # we use this as our new noise target to preserve the existing knowledge of the image.
        # we apply a mask to this noise to only allow the differential of the conditional latents to be learned
        baseline_predicted_noise = (1 - abs_differential_mask) * baseline_predicted_noise
        masked_noise = abs_differential_mask * noise
        target_noise = masked_noise + baseline_predicted_noise
        
        # compute our new target prediction using our current knowledge noise with our conditional latents
        # this makes it so the only new information is the differential of our conditional and unconditional latents
        # forcing the network to preserve existing knowledge, but learn only our changes
        target_pred = (target_noise - conditional_latents).detach()
        
    # make a prediction with the lora network active
    sd.unet.train()
    sd.network.is_active = True
    sd.network.multiplier = network_weight_list
    prediction = sd.predict_noise(
        latents=conditional_noisy_latents.to(device, dtype=dtype).detach(),
        conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
        timestep=timesteps,
        guidance_scale=1.0,
        **pred_kwargs
    )
    
    # target our baseline + diffirential noise target
    pred_loss = torch.nn.functional.mse_loss(
        prediction.float(),
        target_pred.float()
    )
    
    return pred_loss


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
        unconditional_embeds: Optional[PromptEmbeds] = None,
        mask_multiplier=None,
        prior_pred=None,
        scaler=None,
        train_config=None,
        **kwargs
):
    # TODO add others and process individual batch items separately
    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type

    if guidance_type == "targeted":
        assert unconditional_embeds is None, "Unconditional embeds are not supported for targeted guidance"
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
        assert unconditional_embeds is None, "Unconditional embeds are not supported for polarity guidance"
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
            scaler=scaler,
            train_config=train_config,
            **kwargs
        )
    elif guidance_type == "tnt":
        assert unconditional_embeds is None, "Unconditional embeds are not supported for polarity guidance"
        return get_guided_tnt(
            noisy_latents,
            conditional_embeds,
            match_adapter_assist,
            network_weight_list,
            timesteps,
            pred_kwargs,
            batch,
            noise,
            sd,
            prior_pred=prior_pred,
            **kwargs
        )

    elif guidance_type == "targeted_polarity":
        assert unconditional_embeds is None, "Unconditional embeds are not supported for targeted polarity guidance"
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
            unconditional_embeds=unconditional_embeds,
            mask_multiplier=mask_multiplier,
            prior_pred=prior_pred,
            **kwargs
        )
    elif guidance_type == "targeted_flow":
        return targeted_flow_guidance(
            noisy_latents,
            conditional_embeds,
            match_adapter_assist,
            network_weight_list,
            timesteps,
            pred_kwargs,
            batch,
            noise,
            sd,
            unconditional_embeds=unconditional_embeds,
            mask_multiplier=mask_multiplier,
            prior_pred=prior_pred,
            scaler=scaler,
            train_config=train_config,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Guidance type {guidance_type} is not implemented")
