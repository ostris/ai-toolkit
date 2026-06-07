# ref https://huggingface.co/spaces/editing-images/ledits/blob/main/inversion_utils.py

import torch
import os
from tqdm import tqdm

from toolkit import train_tools
from toolkit.prompt_utils import PromptEmbeds
from toolkit.stable_diffusion_model import StableDiffusion


def mu_tilde(model, xt, x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev ** 0.5 * beta_t) / (1 - alpha_bar)) * x0 + (
            (alpha_t ** 0.5 * (1 - alpha_prod_t_prev)) / (1 - alpha_bar)) * xt


def sample_xts_from_x0(sd: StableDiffusion, sample: torch.Tensor, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = sd.noise_scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    alphas = sd.noise_scheduler.alphas
    betas = 1 - alphas
    # variance_noise_shape = (
    #     num_inference_steps,
    #     sd.unet.in_channels,
    #     sd.unet.sample_size,
    #     sd.unet.sample_size)
    variance_noise_shape = list(sample.shape)
    variance_noise_shape[0] = num_inference_steps

    timesteps = sd.noise_scheduler.timesteps.to(sd.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(sample.device, dtype=torch.float16)
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        xts[idx] = sample * (alpha_bar[t] ** 0.5) + torch.randn_like(sample, dtype=torch.float16) * sqrt_one_minus_alpha_bar[t]
    xts = torch.cat([xts, sample], dim=0)

    return xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def forward_step(sd: StableDiffusion, model_output, timestep, sample):
    next_timestep = min(
        sd.noise_scheduler.config['num_train_timesteps'] - 2,
        timestep + sd.noise_scheduler.config['num_train_timesteps'] // sd.noise_scheduler.num_inference_steps
    )

    # 2. compute alphas, betas
    alpha_prod_t = sd.noise_scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementation
    next_sample = sd.noise_scheduler.add_noise(
        pred_original_sample,
        model_output,
        torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(sd: StableDiffusion, timestep):  # , prev_timestep):
    prev_timestep = timestep - sd.noise_scheduler.config['num_train_timesteps'] // sd.noise_scheduler.num_inference_steps
    alpha_prod_t = sd.noise_scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = sd.noise_scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else sd.noise_scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def get_time_ids_from_latents(sd: StableDiffusion, latents: torch.Tensor):
    VAE_SCALE_FACTOR = 2 ** (len(sd.vae.config['block_out_channels']) - 1)
    if sd.is_xl:
        bs, ch, h, w = list(latents.shape)

        height = h * VAE_SCALE_FACTOR
        width = w * VAE_SCALE_FACTOR

        dtype = latents.dtype
        # just do it without any cropping nonsense
        target_size = (height, width)
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(latents.device, dtype=dtype)

        batch_time_ids = torch.cat(
            [add_time_ids for _ in range(bs)]
        )
        return batch_time_ids
    else:
        return None


def inversion_forward_process(
        sd: StableDiffusion,
        sample: torch.Tensor,
        conditional_embeddings: PromptEmbeds,
        unconditional_embeddings: PromptEmbeds,
        etas=None,
        prog_bar=False,
        cfg_scale=3.5,
        num_inference_steps=50, eps=None
):
    current_num_timesteps = len(sd.noise_scheduler.timesteps)
    sd.noise_scheduler.set_timesteps(num_inference_steps, device=sd.device)

    timesteps = sd.noise_scheduler.timesteps.to(sd.device)
    # variance_noise_shape = (
    #     num_inference_steps,
    #     sd.unet.in_channels,
    #     sd.unet.sample_size,
    #     sd.unet.sample_size
    # )
    variance_noise_shape = list(sample.shape)
    variance_noise_shape[0] = num_inference_steps
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas] * sd.noise_scheduler.num_inference_steps
        xts = sample_xts_from_x0(sd, sample, num_inference_steps=num_inference_steps)
        alpha_bar = sd.noise_scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=sd.device, dtype=torch.float16)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    noisy_sample = sample
    op = tqdm(reversed(timesteps), desc="Inverting...") if prog_bar else reversed(timesteps)

    for timestep in op:
        idx = t_to_idx[int(timestep)]
        # 1. predict noise residual
        if not eta_is_zero:
            noisy_sample = xts[idx][None]

        added_cond_kwargs = {}

        with torch.no_grad():
            text_embeddings = train_tools.concat_prompt_embeddings(
                unconditional_embeddings,  # negative embedding
                conditional_embeddings,  # positive embedding
                1,  # batch size
            )
            if sd.is_xl:
                add_time_ids = get_time_ids_from_latents(sd, noisy_sample)
                # add extra for cfg
                add_time_ids = torch.cat(
                    [add_time_ids] * 2, dim=0
                )

                added_cond_kwargs = {
                    "text_embeds": text_embeddings.pooled_embeds,
                    "time_ids": add_time_ids,
                }

            # double up for cfg
            latent_model_input = torch.cat(
                [noisy_sample] * 2, dim=0
            )

            noise_pred = sd.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings.text_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # out = sd.unet.forward(noisy_sample, timestep=timestep, encoder_hidden_states=uncond_embedding)
            # cond_out = sd.unet.forward(noisy_sample, timestep=timestep, encoder_hidden_states=text_embeddings)

        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            noisy_sample = forward_step(sd, noise_pred, timestep, noisy_sample)
            xts = None

        else:
            xtm1 = xts[idx + 1][None]
            # pred of x0
            pred_original_sample = (noisy_sample - (1 - alpha_bar[timestep]) ** 0.5 * noise_pred) / alpha_bar[
                timestep] ** 0.5

            # direction to xt
            prev_timestep = timestep - sd.noise_scheduler.config[
                'num_train_timesteps'] // sd.noise_scheduler.num_inference_steps
            alpha_prod_t_prev = sd.noise_scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else sd.noise_scheduler.final_alpha_cumprod

            variance = get_variance(sd, timestep)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx + 1] = xtm1

    if not zs is None:
        zs[-1] = torch.zeros_like(zs[-1])

    # restore timesteps
    sd.noise_scheduler.set_timesteps(current_num_timesteps, device=sd.device)

    return noisy_sample, zs, xts


#
# def inversion_forward_process(
#         model,
#         sample,
#         etas=None,
#         prog_bar=False,
#         prompt="",
#         cfg_scale=3.5,
#         num_inference_steps=50, eps=None
# ):
#     if not prompt == "":
#         text_embeddings = encode_text(model, prompt)
#     uncond_embedding = encode_text(model, "")
#     timesteps = model.scheduler.timesteps.to(model.device)
#     variance_noise_shape = (
#         num_inference_steps,
#         model.unet.in_channels,
#         model.unet.sample_size,
#         model.unet.sample_size)
#     if etas is None or (type(etas) in [int, float] and etas == 0):
#         eta_is_zero = True
#         zs = None
#     else:
#         eta_is_zero = False
#         if type(etas) in [int, float]: etas = [etas] * model.scheduler.num_inference_steps
#         xts = sample_xts_from_x0(model, sample, num_inference_steps=num_inference_steps)
#         alpha_bar = model.scheduler.alphas_cumprod
#         zs = torch.zeros(size=variance_noise_shape, device=model.device, dtype=torch.float16)
#
#     t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
#     noisy_sample = sample
#     op = tqdm(reversed(timesteps), desc="Inverting...") if prog_bar else reversed(timesteps)
#
#     for t in op:
#         idx = t_to_idx[int(t)]
#         # 1. predict noise residual
#         if not eta_is_zero:
#             noisy_sample = xts[idx][None]
#
#         with torch.no_grad():
#             out = model.unet.forward(noisy_sample, timestep=t, encoder_hidden_states=uncond_embedding)
#             if not prompt == "":
#                 cond_out = model.unet.forward(noisy_sample, timestep=t, encoder_hidden_states=text_embeddings)
#
#         if not prompt == "":
#             ## classifier free guidance
#             noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
#         else:
#             noise_pred = out.sample
#
#         if eta_is_zero:
#             # 2. compute more noisy image and set x_t -> x_t+1
#             noisy_sample = forward_step(model, noise_pred, t, noisy_sample)
#
#         else:
#             xtm1 = xts[idx + 1][None]
#             # pred of x0
#             pred_original_sample = (noisy_sample - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
#
#             # direction to xt
#             prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
#             alpha_prod_t_prev = model.scheduler.alphas_cumprod[
#                 prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
#
#             variance = get_variance(model, t)
#             pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * noise_pred
#
#             mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
#
#             z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
#             zs[idx] = z
#
#             # correction to avoid error accumulation
#             xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
#             xts[idx + 1] = xtm1
#
#     if not zs is None:
#         zs[-1] = torch.zeros_like(zs[-1])
#
#     return noisy_sample, zs, xts


def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep)  # , prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device, dtype=torch.float16)
        sigma_z = eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process(
        model,
        xT,
        etas=0,
        prompts="",
        cfg_scales=None,
        prog_bar=False,
        zs=None,
        controller=None,
        asyrp=False):
    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device, dtype=torch.float16)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = t_to_idx[int(t)]
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(xt, timestep=t,
                                            encoder_hidden_states=uncond_embedding)

            ## Conditional embedding
        if prompts:
            with torch.no_grad():
                cond_out = model.unet.forward(xt, timestep=t,
                                              encoder_hidden_states=text_embeddings)

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else:
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)
        if controller is not None:
            xt = controller.step_callback(xt)
    return xt, zs
