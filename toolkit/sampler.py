import copy

from diffusers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    LCMScheduler
)

from k_diffusion.external import CompVisDenoiser

from toolkit.samplers.custom_lcm_scheduler import CustomLCMScheduler

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

sdxl_sampler_config = {
    "_class_name": "EulerAncestralDiscreteScheduler",
    "_diffusers_version": "0.24.0.dev0",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": False,
    "steps_offset": 1,
    "timestep_spacing": "trailing",
    "trained_betas": None
}


def get_sampler(
        sampler: str,
        kwargs: dict = None,
):
    sched_init_args = {}
    if kwargs is not None:
        sched_init_args.update(kwargs)

    if sampler.startswith("k_"):
        sched_init_args["use_karras_sigmas"] = True

    if sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sampler == "ddpm":  # ddpm is not supported ?
        scheduler_cls = DDPMScheduler
    elif sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sampler == "lms" or sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sampler == "euler" or sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sampler == "euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sampler == "dpmsolver" or sampler == "dpmsolver++" or sampler == "k_dpmsolver" or sampler == "k_dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sampler.replace("k_", "")
    elif sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sampler == "dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sampler == "dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    elif sampler == "lcm":
        scheduler_cls = LCMScheduler
    elif sampler == "custom_lcm":
        scheduler_cls = CustomLCMScheduler

    config = copy.deepcopy(sdxl_sampler_config)
    config.update(sched_init_args)

    scheduler = scheduler_cls.from_config(config)

    return scheduler


# testing
if __name__ == "__main__":
    from diffusers import DiffusionPipeline

    from diffusers import StableDiffusionKDiffusionPipeline
    import torch
    import os

    inference_steps = 25

    pipe = StableDiffusionKDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    pipe = pipe.to("cuda")

    k_diffusion_model = CompVisDenoiser(model)

    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", custom_pipeline="sd_text2img_k_diffusion")
    pipe = pipe.to("cuda")

    prompt = "an astronaut riding a horse on mars"
    pipe.set_scheduler("sample_heun")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]

    image.save("./astronaut_heun_k_diffusion.png")
