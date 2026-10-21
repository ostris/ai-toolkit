"""Per-step latent reporting for sample generation (inference engine).

A holder exposes `sample_step_hook(step_index, num_steps, latents)`; while it
is set, the sampling scheduler's `step` is wrapped on the instance so every
step reaches the hook without per-pipeline code. The latent handed to the
hook is the step's predicted CLEAN sample (x0) — what a preview should show
— falling back to the scheduler's next sample when x0 cannot be derived. Shared by
BaseModel and the legacy StableDiffusion monolith.
"""

from typing import Callable, List


def install_sample_step_hooks(holder, pipeline) -> Callable[[], None]:
    """Wrap scheduler.step on the sampling scheduler(s). Returns an unwrap
    callable. No-op without a hook. Instance-level wrap: the class method is
    untouched and a stale wrapper is replaced rather than stacked."""
    if getattr(holder, "sample_step_hook", None) is None:
        return lambda: None
    schedulers: List = []
    for cand in (
        getattr(pipeline, "scheduler", None),
        getattr(getattr(holder, "pipeline", None), "scheduler", None),
        getattr(holder, "noise_scheduler", None),
    ):
        if cand is not None and hasattr(cand, "step") and all(cand is not s for s in schedulers):
            schedulers.append(cand)
    unwraps = []
    for scheduler in schedulers:
        current = scheduler.step
        orig = getattr(current, "_aitk_orig_step", None) or current

        def make(orig, scheduler):
            def step(*args, **kwargs):
                # capture the pre-step state: x0 is derived from (model_output,
                # timestep, sample) with the sigma/alpha of THIS step, which
                # the scheduler advances past inside orig()
                model_output = args[0] if len(args) > 0 else kwargs.get("model_output")
                timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
                sample_in = args[2] if len(args) > 2 else kwargs.get("sample")
                coeffs = _x0_coefficients(scheduler, timestep)
                out = orig(*args, **kwargs)
                if isinstance(out, tuple):
                    prev_sample = out[0]
                    pred_x0 = None
                else:
                    prev_sample = getattr(out, "prev_sample", None)
                    pred_x0 = getattr(out, "pred_original_sample", None)
                hook = getattr(holder, "sample_step_hook", None)
                if hook is not None and prev_sample is not None:
                    if pred_x0 is None:
                        pred_x0 = _predict_x0(coeffs, model_output, sample_in)
                    timesteps = getattr(scheduler, "timesteps", None)
                    num_steps = len(timesteps) if timesteps is not None else None
                    idx = getattr(holder, "_sample_step_index", 0)
                    holder._sample_step_index = idx + 1
                    hook(idx, num_steps, pred_x0 if pred_x0 is not None else prev_sample)
                return out

            step._aitk_orig_step = orig
            return step

        scheduler.step = make(orig, scheduler)

        def unwrap(scheduler=scheduler):
            if "step" in scheduler.__dict__:
                del scheduler.__dict__["step"]

        unwraps.append(unwrap)
    return lambda: [u() for u in unwraps]


def _x0_coefficients(scheduler, timestep):
    """Per-step numbers needed to turn the model output into a clean-sample
    estimate, read BEFORE scheduler.step advances its index.
    Returns ("flow", sigma) | ("ddpm", prediction_type, alpha_bar) | None."""
    try:
        import torch

        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is not None and hasattr(scheduler, "index_for_timestep"):
            idx = getattr(scheduler, "step_index", None)
            if idx is None:
                idx = scheduler.index_for_timestep(timestep)
            return ("flow", float(sigmas[idx]))
        alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
        if alphas_cumprod is not None:
            t = timestep
            if isinstance(t, torch.Tensor):
                t = int(t.flatten()[0].item())
            prediction_type = getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon")
            return ("ddpm", prediction_type, float(alphas_cumprod[int(t)]))
    except Exception:
        return None
    return None


def _predict_x0(coeffs, model_output, sample):
    """Clean-sample estimate for the preview: flow matching x0 = x_t - sigma*v;
    DDPM-style eps / v / sample predictions via alpha_bar. None when unknown."""
    if coeffs is None or model_output is None or sample is None:
        return None
    try:
        x = sample.tensor if hasattr(sample, "tensor") else sample
        v = model_output.tensor if hasattr(model_output, "tensor") else model_output
        if v.shape != x.shape:
            return None
        v = v.to(x.dtype)
        if coeffs[0] == "flow":
            sigma = coeffs[1]
            return x - sigma * v
        _, prediction_type, alpha_bar = coeffs
        a = alpha_bar**0.5
        b = (1.0 - alpha_bar) ** 0.5
        if prediction_type == "epsilon":
            return (x - b * v) / a
        if prediction_type == "v_prediction":
            return a * x - b * v
        if prediction_type == "sample":
            return v
    except Exception:
        return None
    return None


def emit_sample_step(holder, latents, step_index=None, num_steps=None):
    """For sampling loops that bypass scheduler.step: report one denoised
    latent to the holder's hook (no-op when unset)."""
    hook = getattr(holder, "sample_step_hook", None)
    if hook is None:
        return
    idx = getattr(holder, "_sample_step_index", 0) if step_index is None else step_index
    holder._sample_step_index = idx + 1
    hook(idx, num_steps, latents)
