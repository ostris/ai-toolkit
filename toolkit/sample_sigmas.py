"""Explicit sigma schedules for sample generation.

A sample item may carry `sigmas: [1.0, 0.5]`: the exact noise levels the
sampler visits, descending on a 0..1 scale (1 = pure noise). The terminal clean
0.0 is implied and appended by every consumer, so `[1.0, 0.5]` is a 2-step run.
Values above 1 are read on the toolkit's 0..1000 timestep scale. When set, the
schedule replaces the pipeline's own grid (shift, dynamic shifting, terminal
stretch included) and `sample_steps` is ignored.

Pipelines that drive a diffusers-style scheduler are covered without per-model
code: `install_sample_sigma_override` wraps `set_timesteps` on the scheduler
classes the holder can reach (instances made inside a pipeline call included)
and rewrites `sigmas`/`timesteps` after the original has sized its state.
Pipelines that build their own grid take `sigmas=` directly.
"""

from typing import Callable, List, Optional, Sequence

import torch


def normalize_sample_sigmas(sigmas) -> Optional[List[float]]:
    if sigmas is None:
        return None
    if isinstance(sigmas, (int, float)):
        sigmas = [sigmas]
    values = [float(s) for s in sigmas]
    if len(values) == 0:
        return None
    if max(values) > 1.0:
        # 0..1000 timestep scale
        values = [v / 1000.0 for v in values]
    values = [min(max(v, 0.0), 1.0) for v in values]
    values = sorted(set(values), reverse=True)
    # the clean terminal is implied; consumers append it
    values = [v for v in values if v > 0.0]
    if len(values) == 0:
        raise ValueError(
            f"sample sigmas must contain at least one value above 0: {sigmas}"
        )
    return values


def sigmas_with_terminal(sigmas: Sequence[float]) -> List[float]:
    return [float(s) for s in sigmas] + [0.0]


def _is_flow_scheduler(scheduler) -> bool:
    config = getattr(scheduler, "config", None)
    if config is not None:
        if getattr(config, "use_flow_sigmas", False):
            return True
        if "use_dynamic_shifting" in config or "shift_terminal" in config:
            return True
    return "FlowMatch" in type(scheduler).__name__


def apply_sigma_override(scheduler, sigmas: Sequence[float]) -> None:
    """Rewrite a scheduler's schedule in place, after its own set_timesteps ran
    (so step-index, solver history and warmup state are already sized)."""
    sigmas = list(sigmas)
    num_train = int(
        getattr(getattr(scheduler, "config", None), "num_train_timesteps", 1000)
    )
    old_ts = getattr(scheduler, "timesteps", None)
    ts_device = old_ts.device if isinstance(old_ts, torch.Tensor) else "cpu"
    ts_dtype = old_ts.dtype if isinstance(old_ts, torch.Tensor) else torch.float32
    old_sig = getattr(scheduler, "sigmas", None)
    sig_device = old_sig.device if isinstance(old_sig, torch.Tensor) else ts_device
    sig_dtype = old_sig.dtype if isinstance(old_sig, torch.Tensor) else torch.float32

    if _is_flow_scheduler(scheduler):
        scheduler.sigmas = torch.tensor(
            sigmas_with_terminal(sigmas), dtype=sig_dtype, device=sig_device
        )
        scheduler.timesteps = (
            torch.tensor(sigmas, dtype=torch.float64) * num_train
        ).to(ts_device, ts_dtype)
    else:
        # alphas_cumprod schedulers index integer timesteps 0..T-1
        t = (
            (torch.tensor(sigmas, dtype=torch.float64) * num_train)
            .round()
            .clamp(0, num_train - 1)
        )
        scheduler.timesteps = t.to(ts_device, ts_dtype)
        alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
        if isinstance(old_sig, torch.Tensor) and alphas_cumprod is not None:
            ac = alphas_cumprod.detach().to("cpu", torch.float64)
            all_sig = ((1.0 - ac) / ac).sqrt()
            idx = t.long().cpu()
            scheduler.sigmas = torch.cat([all_sig[idx], all_sig.new_zeros(1)]).to(
                sig_device, sig_dtype
            )
    if hasattr(scheduler, "num_inference_steps"):
        scheduler.num_inference_steps = len(sigmas)
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None


def install_sample_sigma_override(
    holder, pipeline, image_configs
) -> Callable[[], None]:
    """Class-level wrap of set_timesteps on every scheduler class the holder can
    reach. The wrapper reads `holder.sample_sigmas` at call time (the sampling
    loop sets it per image) and is transparent when that is None. Returns an
    unwrap callable; no-op when no image config carries sigmas."""
    if not any(getattr(c, "sigmas", None) for c in image_configs):
        return lambda: None

    classes = []

    def add(obj):
        if obj is None or not hasattr(obj, "set_timesteps"):
            return
        cls = type(obj)
        if all(cls is not c for c in classes):
            classes.append(cls)

    add(getattr(pipeline, "scheduler", None))
    add(getattr(getattr(holder, "pipeline", None), "scheduler", None))
    add(getattr(holder, "noise_scheduler", None))
    for name in ("get_sample_scheduler", "get_train_scheduler"):
        fn = getattr(holder, name, None)
        if callable(fn):
            try:
                add(fn())
            except Exception:
                pass

    restores = []
    for cls in classes:
        if getattr(cls.__dict__.get("set_timesteps"), "_aitk_sigma_wrapper", False):
            continue
        own = "set_timesteps" in cls.__dict__
        orig = cls.__dict__["set_timesteps"] if own else cls.set_timesteps

        def make(orig):
            def set_timesteps(self, *args, **kwargs):
                out = orig(self, *args, **kwargs)
                sigmas = getattr(holder, "sample_sigmas", None)
                if sigmas:
                    apply_sigma_override(self, sigmas)
                return out

            set_timesteps._aitk_sigma_wrapper = True
            return set_timesteps

        cls.set_timesteps = make(orig)

        def restore(cls=cls, own=own, orig=orig):
            if own:
                cls.set_timesteps = orig
            elif "set_timesteps" in cls.__dict__:
                delattr(cls, "set_timesteps")

        restores.append(restore)

    def unwrap():
        holder.sample_sigmas = None
        for r in restores:
            r()

    return unwrap
