
import torch


_dwt = None


def _get_wavelet_loss(device, dtype):
    global _dwt
    if _dwt is not None:
        return _dwt

    # init wavelets
    from pytorch_wavelets import DWTForward
    # wave='db1'  wave='haar'
    dwt = DWTForward(J=1, mode='zero', wave='haar').to(
        device=device, dtype=dtype)
    _dwt = dwt
    return dwt


def wavelet_loss(model_pred, latents, noise):
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()
    dwt = _get_wavelet_loss(model_pred.device, model_pred.dtype)
    with torch.no_grad():
        model_input_xll, model_input_xh = dwt(latents)
        model_input_xlh, model_input_xhl, model_input_xhh = torch.unbind(model_input_xh[0], dim=2)
        model_input = torch.cat([model_input_xll, model_input_xlh, model_input_xhl, model_input_xhh], dim=1)
    
    # reverse the noise to get the model prediction of the pure latents
    model_pred = noise - model_pred

    model_pred_xll, model_pred_xh = dwt(model_pred)
    model_pred_xlh, model_pred_xhl, model_pred_xhh = torch.unbind(model_pred_xh[0], dim=2)
    model_pred = torch.cat([model_pred_xll, model_pred_xlh, model_pred_xhl, model_pred_xhh], dim=1)
    
    return torch.nn.functional.mse_loss(model_pred, model_input, reduction="none")