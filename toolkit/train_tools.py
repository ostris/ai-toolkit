import argparse
import json
import os
import time

from diffusers import (
    StableDiffusionPipeline,
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
)
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import torch
import re

SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def get_torch_dtype(dtype_str):
    if dtype_str == "float" or dtype_str == "fp32" or dtype_str == "single" or dtype_str == "float32":
        return torch.float
    if dtype_str == "fp16" or dtype_str == "half" or dtype_str == "float16":
        return torch.float16
    if dtype_str == "bf16" or dtype_str == "bfloat16":
        return torch.bfloat16
    return None


def replace_filewords_prompt(prompt, args: argparse.Namespace):
    # if name_replace attr in args (may not be)
    if hasattr(args, "name_replace") and args.name_replace is not None:
        # replace [name] to args.name_replace
        prompt = prompt.replace("[name]", args.name_replace)
    if hasattr(args, "prepend") and args.prepend is not None:
        # prepend to every item in prompt file
        prompt = args.prepend + ' ' + prompt
    if hasattr(args, "append") and args.append is not None:
        # append to every item in prompt file
        prompt = prompt + ' ' + args.append
    return prompt


def replace_filewords_in_dataset_group(dataset_group, args: argparse.Namespace):
    # if name_replace attr in args (may not be)
    if hasattr(args, "name_replace") and args.name_replace is not None:
        if not len(dataset_group.image_data) > 0:
            # throw error
            raise ValueError("dataset_group.image_data is empty")
        for key in dataset_group.image_data:
            dataset_group.image_data[key].caption = dataset_group.image_data[key].caption.replace(
                "[name]", args.name_replace)

    return dataset_group


def get_seeds_from_latents(latents):
    # latents shape = (batch_size, 4, height, width)
    # for speed we only use 8x8 slice of the first channel
    seeds = []

    # split batch up
    for i in range(latents.shape[0]):
        # use only first channel, multiply by 255 and convert to int
        tensor = latents[i, 0, :, :] * 255.0  # shape = (height, width)
        # slice 8x8
        tensor = tensor[:8, :8]
        # clip to 0-255
        tensor = torch.clamp(tensor, 0, 255)
        # convert to 8bit int
        tensor = tensor.to(torch.uint8)
        # convert to bytes
        tensor_bytes = tensor.cpu().numpy().tobytes()
        # hash
        hash_object = hashlib.sha256(tensor_bytes)
        # get hex
        hex_dig = hash_object.hexdigest()
        # convert to int
        seed = int(hex_dig, 16) % (2 ** 32)
        # append
        seeds.append(seed)
    return seeds


def get_noise_from_latents(latents):
    seed_list = get_seeds_from_latents(latents)
    noise = []
    for seed in seed_list:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        noise.append(torch.randn_like(latents[0]))
    return torch.stack(noise)


# mix 0 is completely noise mean, mix 1 is completely target mean

def match_noise_to_target_mean_offset(noise, target, mix=0.5, dim=None):
    dim = dim or (1, 2, 3)
    # reduce mean of noise on dim 2, 3, keeping 0 and 1 intact
    noise_mean = noise.mean(dim=dim, keepdim=True)
    target_mean = target.mean(dim=dim, keepdim=True)

    new_noise_mean = mix * target_mean + (1 - mix) * noise_mean

    noise = noise - noise_mean + new_noise_mean
    return noise


def sample_images(
        accelerator,
        args: argparse.Namespace,
        epoch,
        steps,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
        prompt_replacement=None,
        force_sample=False
):
    """
    StableDiffusionLongPromptWeightingPipelineの改造版を使うようにしたので、clip skipおよびプロンプトの重みづけに対応した
    """
    if not force_sample:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            # sample_every_n_steps は無視する
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
                return

    is_sample_only = args.sample_only
    is_generating_only = hasattr(args, "is_generating_only") and args.is_generating_only

    print(f"\ngenerating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts):
        print(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    org_vae_device = vae.device  # CPUにいるはず
    vae.to(device)

    # read prompts

    # with open(args.sample_prompts, "rt", encoding="utf-8") as f:
    #     prompts = f.readlines()

    if args.sample_prompts.endswith(".txt"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif args.sample_prompts.endswith(".json"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # schedulerを用意する
    sched_init_args = {}
    if args.sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif args.sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif args.sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif args.sample_sampler == "lms" or args.sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif args.sample_sampler == "euler" or args.sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_sampler == "euler_a" or args.sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_sampler == "dpmsolver" or args.sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sample_sampler
    elif args.sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_sampler == "dpm_2" or args.sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif args.sample_sampler == "dpm_2_a" or args.sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # print("set clip_sample to True")
        scheduler.config.clip_sample = True

    pipeline = StableDiffusionLongPromptWeightingPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        clip_skip=args.clip_skip,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.to(device)

    if is_generating_only:
        save_dir = args.output_dir
    else:
        save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    with torch.no_grad():
        with accelerator.autocast():
            for i, prompt in enumerate(prompts):
                if not accelerator.is_main_process:
                    continue

                if isinstance(prompt, dict):
                    negative_prompt = prompt.get("negative_prompt")
                    sample_steps = prompt.get("sample_steps", 30)
                    width = prompt.get("width", 512)
                    height = prompt.get("height", 512)
                    scale = prompt.get("scale", 7.5)
                    seed = prompt.get("seed")
                    prompt = prompt.get("prompt")

                    prompt = replace_filewords_prompt(prompt, args)
                    negative_prompt = replace_filewords_prompt(negative_prompt, args)
                else:
                    prompt = replace_filewords_prompt(prompt, args)
                    # prompt = prompt.strip()
                    # if len(prompt) == 0 or prompt[0] == "#":
                    #     continue

                    # subset of gen_img_diffusers
                    prompt_args = prompt.split(" --")
                    prompt = prompt_args[0]
                    negative_prompt = None
                    sample_steps = 30
                    width = height = 512
                    scale = 7.5
                    seed = None
                    for parg in prompt_args:
                        try:
                            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                            if m:
                                width = int(m.group(1))
                                continue

                            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                            if m:
                                height = int(m.group(1))
                                continue

                            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
                            if m:
                                seed = int(m.group(1))
                                continue

                            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                            if m:  # steps
                                sample_steps = max(1, min(1000, int(m.group(1))))
                                continue

                            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # scale
                                scale = float(m.group(1))
                                continue

                            m = re.match(r"n (.+)", parg, re.IGNORECASE)
                            if m:  # negative prompt
                                negative_prompt = m.group(1)
                                continue

                        except ValueError as ex:
                            print(f"Exception in parsing / 解析エラー: {parg}")
                            print(ex)

                if seed is not None:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                if prompt_replacement is not None:
                    prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
                    if negative_prompt is not None:
                        negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

                height = max(64, height - height % 8)  # round to divisible by 8
                width = max(64, width - width % 8)  # round to divisible by 8
                print(f"prompt: {prompt}")
                print(f"negative_prompt: {negative_prompt}")
                print(f"height: {height}")
                print(f"width: {width}")
                print(f"sample_steps: {sample_steps}")
                print(f"scale: {scale}")
                image = pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=sample_steps,
                    guidance_scale=scale,
                    negative_prompt=negative_prompt,
                ).images[0]

                ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                seed_suffix = "" if seed is None else f"_{seed}"

                if is_generating_only:
                    img_filename = (
                        f"{'' if args.output_name is None else args.output_name + '_'}{ts_str}_{num_suffix}_{i:02d}{seed_suffix}.png"
                    )
                else:
                    img_filename = (
                        f"{'' if args.output_name is None else args.output_name + '_'}{ts_str}_{i:04d}{seed_suffix}.png"
                    )
                if is_sample_only:
                    # make prompt txt file
                    img_path_no_ext = os.path.join(save_dir, img_filename[:-4])
                    with open(img_path_no_ext + ".txt", "w") as f:
                        # put prompt in txt file
                        f.write(prompt)
                        # close file
                        f.close()

                image.save(os.path.join(save_dir, img_filename))

                # wandb有効時のみログを送信
                try:
                    wandb_tracker = accelerator.get_tracker("wandb")
                    try:
                        import wandb
                    except ImportError:  # 事前に一度確認するのでここはエラー出ないはず
                        raise ImportError("No wandb / wandb がインストールされていないようです")

                    wandb_tracker.log({f"sample_{i}": wandb.Image(image)})
                except:  # wandb 無効時
                    pass

    # clear pipeline and cache to reduce vram usage
    del pipeline
    torch.cuda.empty_cache()

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)
