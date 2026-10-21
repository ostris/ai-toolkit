"""Preview sampler for Iris-3B: FlowDPM-Solver++ (multistep, order 2) on the
shifted rectified-flow grid, ported from the reference ``iris3b/flow/solver.py``.

Data-prediction formulation with ``alpha(t) = 1 - t``, ``sigma(t) = t``,
``lambda(t) = log((1 - t) / t)``. The model predicts velocity
(``x0_hat = x - t * v``). The time grid is
``sigma = 1 - linspace(1, 0.001, steps + 1)`` remapped by the training shift,
descending and ending exactly at t = 0 so the last (first-order) update is an
exact projection onto the predicted clean image. NFE == steps.

CFG is applied to the raw velocity (``uncond + s * (cond - uncond)``) and gated
to model times strictly inside ``cfg_interval``. Pixel space: the integrator
state IS the image in [-1, 1]; "decode" is a clamp + uint8 cast.
"""

import math
from typing import List, Optional, Tuple

import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def time_grid(steps: int, shift: float) -> List[float]:
    sigma = 1.0 - torch.linspace(1.0, 0.001, steps + 1, dtype=torch.float64)
    return shift_sigma(sigma, shift).flip(0).tolist()  # descending, last value exactly 0


def _lambda(t: float) -> float:
    if t <= 0.0:
        return math.inf
    return math.log((1.0 - t) / t)


def _first_order(x, s: float, t: float, x0):
    h = _lambda(t) - _lambda(s)
    phi1 = math.expm1(-h)
    return (t / s) * x - (1.0 - t) * phi1 * x0


def _second_order(x, prev1, prev0, t: float):
    (s1, x0_1), (s0, x0_0) = prev1, prev0
    lam_t, lam_0, lam_1 = _lambda(t), _lambda(s0), _lambda(s1)
    h, h0 = lam_t - lam_0, lam_0 - lam_1
    r0 = h0 / h
    d = (x0_0 - x0_1) / r0
    phi1 = math.expm1(-h)
    return (t / s0) * x - (1.0 - t) * phi1 * x0_0 - 0.5 * (1.0 - t) * phi1 * d


class Iris3BPipeline:
    """Lightweight sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # model: the Iris3BModel holder (transformer, devices, flow settings)
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    @staticmethod
    def _feats(embeds, device, dtype):
        feats = embeds.text_embeds.to(device, dtype=dtype)
        mask = getattr(embeds, "attention_mask", None)
        if mask is None:
            raise ValueError("Iris prompt embeds must carry an attention_mask")
        return feats, mask.to(device)

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        order: int = 2,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        shift: Optional[float] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer
        shift = model.flow_shift if shift is None else shift
        num_train_timesteps = model.num_train_timesteps

        do_cfg = unconditional_embeds is not None and guidance_scale != 1.0
        lo, hi = cfg_interval

        if latents is None:
            shape = (1, transformer.in_channels, height, width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        x = latents.to(device, dtype=torch.float32)
        batch = x.shape[0]

        cond_feats, cond_mask = self._feats(conditional_embeds, device, dtype)
        if do_cfg:
            uncond_feats, uncond_mask = self._feats(unconditional_embeds, device, dtype)
            cfg_feats = torch.cat([uncond_feats, cond_feats], dim=0)
            cfg_mask = torch.cat([uncond_mask, cond_mask], dim=0)

        def velocity(x_t: torch.Tensor, t: float) -> torch.Tensor:
            t_model = torch.full((batch,), t * num_train_timesteps, device=device, dtype=torch.float32)
            if do_cfg and lo < t < hi:
                out = model.model_velocity(
                    torch.cat([x_t, x_t], dim=0).to(dtype),
                    torch.cat([t_model, t_model], dim=0),
                    cfg_feats,
                    cfg_mask,
                )
                out_uncond, out_cond = out.float().chunk(2, dim=0)
                return out_uncond + guidance_scale * (out_cond - out_uncond)
            return model.model_velocity(x_t.to(dtype), t_model, cond_feats, cond_mask).float()

        grid = time_grid(num_inference_steps, shift)
        history: List[Tuple[float, torch.Tensor]] = []
        emit = getattr(model, "_emit_sample_step", None)
        for i in range(1, num_inference_steps + 1):
            s, t = grid[i - 1], grid[i]
            x0 = x - s * velocity(x, s)
            if emit is not None and getattr(model, "sample_step_hook", None) is not None:
                emit(x0, i - 1, num_inference_steps)
            history.append((s, x0))
            # order ramps up over the first steps and back down at the end so
            # the terminal t=0 update is first-order: an exact x <- x0 projection
            step_order = min(i, order, num_inference_steps + 1 - i)
            if step_order == 1:
                x = _first_order(x, s, t, x0)
            else:
                x = _second_order(x, history[-2], history[-1], t)
            if len(history) > 2:
                history.pop(0)

        images = model.decode_latents(x, device=device, dtype=torch.float32)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
