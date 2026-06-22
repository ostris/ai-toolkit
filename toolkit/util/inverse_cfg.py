import torch


def inverse_classifier_guidance(
        noise_pred_cond: torch.Tensor,
        noise_pred_uncond: torch.Tensor,
        guidance_scale: torch.Tensor
):
    """
    Adjust the noise_pred_cond for the classifier free guidance algorithm
    to ensure that the final noise prediction equals the original noise_pred_cond.
    """
    # To make noise_pred equal noise_pred_cond_orig, we adjust noise_pred_cond
    # based on the formula used in the algorithm.
    # We derive the formula to find the correct adjustment for noise_pred_cond:
    # noise_pred_cond = (noise_pred_cond_orig - noise_pred_uncond * guidance_scale) / (guidance_scale - 1)
    # It's important to check if guidance_scale is not 1 to avoid division by zero.
    if guidance_scale == 1:
        # If guidance_scale is 1, adjusting is not needed or possible in the same way,
        # since it would lead to division by zero. This also means the algorithm inherently
        # doesn't alter the noise_pred_cond in relation to noise_pred_uncond.
        # Thus, we return the original values, though this situation might need special handling.
        return noise_pred_cond
    adjusted_noise_pred_cond = (noise_pred_cond - noise_pred_uncond) / guidance_scale
    return adjusted_noise_pred_cond
