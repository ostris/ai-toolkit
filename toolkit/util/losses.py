import torch
import torch.nn.functional as F


_dwt = None


def _get_wavelet_loss(device, dtype):
    global _dwt
    if _dwt is not None:
        return _dwt

    # init wavelets
    from pytorch_wavelets import DWTForward

    # wave='db1'  wave='haar'
    dwt = DWTForward(J=1, mode="zero", wave="haar").to(device=device, dtype=dtype)
    _dwt = dwt
    return dwt


def wavelet_loss(model_pred, latents, noise):
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()
    dwt = _get_wavelet_loss(model_pred.device, model_pred.dtype)
    with torch.no_grad():
        model_input_xll, model_input_xh = dwt(latents)
        model_input_xlh, model_input_xhl, model_input_xhh = torch.unbind(
            model_input_xh[0], dim=2
        )
        model_input = torch.cat(
            [model_input_xll, model_input_xlh, model_input_xhl, model_input_xhh], dim=1
        )

    # reverse the noise to get the model prediction of the pure latents
    model_pred = noise - model_pred

    model_pred_xll, model_pred_xh = dwt(model_pred)
    model_pred_xlh, model_pred_xhl, model_pred_xhh = torch.unbind(
        model_pred_xh[0], dim=2
    )
    model_pred = torch.cat(
        [model_pred_xll, model_pred_xlh, model_pred_xhl, model_pred_xhh], dim=1
    )

    return torch.nn.functional.mse_loss(model_pred, model_input, reduction="none")


def stepped_loss(model_pred, latents, noise, noisy_latents, timesteps, scheduler):
    # this steps the on a 20 step timescale from the current step (50 idx steps ahead)
    # and then reconstructs the original image at that timestep. This should lessen the error
    # possible in high noise timesteps and make the flow smoother.
    bs = model_pred.shape[0]

    noise_pred_chunks = torch.chunk(model_pred, bs)
    timestep_chunks = torch.chunk(timesteps, bs)
    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
    noise_chunks = torch.chunk(noise, bs)

    x0_pred_chunks = []

    for idx in range(bs):
        model_output = noise_pred_chunks[idx]  # predicted noise (same shape as latent)
        timestep = timestep_chunks[idx]  # scalar tensor per sample (e.g., [t])
        sample = noisy_latent_chunks[idx].to(torch.float32)
        noise_i = noise_chunks[idx].to(sample.dtype).to(sample.device)

        # Initialize scheduler step index for this sample
        scheduler._step_index = None
        scheduler._init_step_index(timestep)

        # ---- Step +50 indices (or to the end) in sigma-space ----
        sigma = scheduler.sigmas[scheduler.step_index]
        target_idx = min(scheduler.step_index + 50, len(scheduler.sigmas) - 1)
        sigma_next = scheduler.sigmas[target_idx]

        # One-step update along the model-predicted direction
        stepped = sample + (sigma_next - sigma) * model_output

        # ---- Inverse-Gaussian recovery at the target timestep ----
        t_01 = (
            (scheduler.sigmas[target_idx]).to(stepped.device).to(stepped.dtype)
        )
        original_samples = (stepped - t_01 * noise_i) / (1.0 - t_01)
        x0_pred_chunks.append(original_samples)

    predicted_images = torch.cat(x0_pred_chunks, dim=0)

    return torch.nn.functional.mse_loss(
        predicted_images.float(),
        latents.float().to(device=predicted_images.device),
        reduction="none",
    )


def _band_lows(x: torch.Tensor, levels: int) -> list:
    """Gaussian-pyramid low-passes of (N, C, H, W) at the input size: lows[0] = x, lows[k] = x
    averaged over 2^k x 2^k cells and bilinearly resampled back."""
    lows = [x]
    cur = x
    for _ in range(levels):
        cur = F.avg_pool2d(cur, 2, ceil_mode=True)
        lows.append(F.interpolate(cur, size=x.shape[-2:], mode="bilinear", align_corners=False))
    return lows


def _bands(x: torch.Tensor, levels: int) -> list:
    """Laplacian bands at the input size, fine to coarse; they sum back to x."""
    lows = _band_lows(x, levels)
    return [lows[k] - lows[k + 1] for k in range(levels)] + [lows[levels]]


def band_loss(pred, target, latents, noise, timesteps, is_flow_matching, scheduler, levels: int = 3):
    """MSE on a per-sample frequency-filtered residual: each Laplacian band of the residual is
    weighted by how recoverable that band is at the sample's noise level, snr_k / (1 + snr_k) with
    snr_k = snr(t) * band power of the clean latents / band power of the noise (both measured on
    the batch). At t -> 0 every weight is 1 and this is the plain MSE; at high t the bands the
    noise has buried drop out instead of adding gradient noise. The filter is linear, so it
    applies to pred / target as they are (x0, velocity or epsilon residuals differ only by a
    scalar in t). Returns an elementwise loss, pred's shape."""
    pred, target = pred.float(), target.float()
    shape = pred.shape
    if pred.dim() not in (4, 5):
        return F.mse_loss(pred, target, reduction="none")
    n = shape[0]
    if pred.dim() == 5:  # video: bands over (H, W) of every frame
        flat = lambda x: x.float().transpose(1, 2).flatten(0, 1)
    else:
        flat = lambda x: x.float()
    with torch.no_grad():
        clean_b = _bands(flat(latents.to(pred.device)), levels)
        noise_b = _bands(flat(noise.to(pred.device)), levels)
        # band power over the whole batch: a prior-level quantity, not per image
        p_clean = torch.stack([b.pow(2).mean() for b in clean_b])
        p_noise = torch.stack([b.pow(2).mean() for b in noise_b]).clamp_min(1e-12)
        ts = timesteps.to(pred.device).float().flatten()
        if is_flow_matching:
            t = (ts / 1000.0).clamp(1e-4, 1 - 1e-4)
            snr = ((1 - t) / t) ** 2
        else:
            ac = scheduler.alphas_cumprod.to(pred.device, torch.float32)[ts.long().clamp(0, scheduler.alphas_cumprod.shape[0] - 1)]
            snr = ac / (1 - ac).clamp_min(1e-8)
        s = snr.view(n, 1) * (p_clean / p_noise).view(1, -1)  # (N, levels + 1)
        w = s / (1 + s)
        if pred.dim() == 5:
            w = w.repeat_interleave(shape[2], dim=0)  # per frame row
        w = w.t().reshape(levels + 1, -1, 1, 1, 1)
    pb, tb = _bands(flat(pred), levels), _bands(flat(target), levels)
    filtered = sum(w[k] * (pb[k] - tb[k]) for k in range(levels + 1))
    loss = filtered.pow(2)
    if pred.dim() == 5:
        loss = loss.view(n, shape[2], shape[1], *shape[-2:]).transpose(1, 2)
    return loss
