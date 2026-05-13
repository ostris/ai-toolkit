from typing import List, Optional, Union

import einops
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as transforms

from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from dataclasses import dataclass


TIMESTEP_TOKEN_NUM = 1
DEFAULT_NOISE_SCALE = 8.0
T_EPS = 0.001
PATCH_SIZE = 32

TENSOR_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def round_to_patch(dim: int, patch: int = PATCH_SIZE) -> int:
    return max(patch, int(dim // patch * patch))


def _get_rope_index_t2i(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: torch.LongTensor,
    image_grid_thw: torch.LongTensor,
    skip_vision_start_token: List[int],
    fix_point: int = 4096,
):
    """Compute mrope position ids for the t2i case used by HiDream-O1."""
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    for i, ids_row in enumerate(input_ids):
        ids_row = ids_row[attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(ids_row == vision_start_token_id).squeeze(
            1
        )
        vision_tokens = ids_row[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum().item()
        video_nums = (vision_tokens == video_token_id).sum().item()
        input_tokens = ids_row.tolist()

        llm_pos_ids_list = []
        st = 0
        image_index = 0
        video_index = 0
        remain_images, remain_videos = image_nums, video_nums
        local_fix_point = fix_point

        for _ in range(image_nums + video_nums):
            ed_image = (
                input_tokens.index(image_token_id, st)
                if (image_token_id in input_tokens and remain_images > 0)
                else len(input_tokens) + 1
            )
            ed_video = (
                input_tokens.index(video_token_id, st)
                if (video_token_id in input_tokens and remain_videos > 0)
                else len(input_tokens) + 1
            )
            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index].tolist()
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = image_grid_thw[video_index].tolist()
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size

            text_len = ed - st - skip_vision_start_token[image_index - 1]
            text_len = max(0, text_len)

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )

            if skip_vision_start_token[image_index - 1]:
                if local_fix_point > 0:
                    local_fix_point = local_fix_point - st_idx
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + local_fix_point + st_idx
                )
                local_fix_point = 0
            else:
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
            position_ids.device
        )

    return position_ids


def _build_t2i_sample_from_input_ids(
    input_ids: torch.Tensor,
    height: int,
    width: int,
    model_config,
    attention_mask: Optional[torch.Tensor] = None,
):
    """Build the full conditioning sample (position_ids/token_types/vinput_mask)
    around an already-tokenized prompt."""
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    image_grid_thw = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)

    vision_tokens = (
        torch.zeros((1, image_len), dtype=input_ids.dtype, device=input_ids.device)
        + image_token_id
    )
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids = _get_rope_index_t2i(
        spatial_merge_size=1,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=image_grid_thw,
        skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn : bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3

    vinput_mask = token_types == 1
    token_types_bin = (token_types > 0).to(token_types.dtype)

    sample = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
    }
    if attention_mask is not None:
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        sample["attention_mask"] = attention_mask
    return sample


@dataclass
class HiDreamO1PipelineOutput(BaseOutput):
    images: List[Image.Image]


class HiDreamO1Pipeline(DiffusionPipeline):
    """
    Diffusers-style inference pipeline for HiDream-O1 (base model).

    HiDream-O1 is a unified text/vision/diffusion model with no VAE — the
    transformer directly predicts image patches in pixel space. This pipeline
    keeps only the components needed for text-to-image inference.
    """

    model_cpu_offload_seq = "model"

    def __init__(
        self,
        model,
        processor,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(model=model, processor=processor, scheduler=scheduler)

    @property
    def tokenizer(self):
        return (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )

    def _snap_resolution(self, width: int, height: int):
        w, h = round_to_patch(width), round_to_patch(height)
        if (w, h) != (width, height):
            print(f"[hidream-o1] Resolution rounded from {width}x{height} to {w}x{h}")
        return w, h

    def build_conditioning_sample(
        self,
        input_ids: torch.Tensor,
        height: int,
        width: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Build the per-sample conditioning dict (input_ids, position_ids,
        token_types, vinput_mask) around already-tokenized text. Useful when
        a training loop needs to batch samples manually."""
        return _build_t2i_sample_from_input_ids(
            input_ids,
            height,
            width,
            self.model.config,
            attention_mask=attention_mask,
        )

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Apply the chat template + boi/tms suffix and tokenize.
        Returns input_ids of shape (1, seq_len). Use these to precompute and
        pass back into __call__ via `prompt_input_ids` / `negative_prompt_input_ids`."""
        tokenizer = self.tokenizer
        boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
        tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

        messages = [{"role": "user", "content": prompt}]
        template_caption = (
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + boi_token
            + tms_token * TIMESTEP_TOKEN_NUM
        )
        return tokenizer.encode(
            template_caption, return_tensors="pt", add_special_tokens=False
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = " ",
        prompt_input_ids: Optional[torch.Tensor] = None,
        negative_prompt_input_ids: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        height: int = 1440,
        width: int = 2560,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 3.0,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        noise_scale: float = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        if noise_scale is None:
            noise_scale = DEFAULT_NOISE_SCALE
        if prompt is None and prompt_input_ids is None:
            raise ValueError("Provide either `prompt` or `prompt_input_ids`.")

        def _unwrap_str(x):
            if isinstance(x, list):
                if len(x) != 1:
                    raise ValueError(
                        "HiDreamO1Pipeline currently supports batch size 1."
                    )
                return x[0]
            return x

        prompt = _unwrap_str(prompt)
        negative_prompt = _unwrap_str(negative_prompt)

        device = self._execution_device
        dtype = torch.bfloat16
        model_config = self.model.config

        width, height = self._snap_resolution(width, height)
        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE

        do_cfg = guidance_scale > 1.0

        if prompt_input_ids is None:
            prompt_input_ids = self.encode_prompt(prompt)
        if do_cfg and negative_prompt_input_ids is None:
            if negative_prompt is None:
                negative_prompt = " "
            negative_prompt_input_ids = self.encode_prompt(negative_prompt)

        cond_sample = _build_t2i_sample_from_input_ids(
            prompt_input_ids,
            height,
            width,
            model_config,
            attention_mask=prompt_attention_mask,
        )
        uncond_sample = (
            _build_t2i_sample_from_input_ids(
                negative_prompt_input_ids,
                height,
                width,
                model_config,
                attention_mask=negative_prompt_attention_mask,
            )
            if do_cfg
            else None
        )

        def _to_device(s):
            return {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s.items()
            }

        cond_sample = _to_device(cond_sample)
        if uncond_sample is not None:
            uncond_sample = _to_device(uncond_sample)

        if generator is None:
            if seed is None:
                seed = 0
            generator = torch.Generator(device="cpu").manual_seed(seed + 1)
            torch.manual_seed(seed + 1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + 1)

        noise = noise_scale * torch.randn(
            (1, 3, height, width), generator=generator
        ).to(device, dtype)
        z = einops.rearrange(
            noise,
            "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )

        if shift is not None and hasattr(self.scheduler, "set_shift"):
            self.scheduler.set_shift(shift)
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        def _forward_once(sample, z_in, t_pixeldit):
            with torch.autocast(device.type, dtype=dtype):
                kwargs = {
                    "input_ids": sample["input_ids"],
                    "position_ids": sample["position_ids"],
                    "vinputs": z_in,
                    "timestep": t_pixeldit.reshape(-1).to(device),
                    "token_types": sample["token_types"],
                    "use_flash_attn": True,
                }
                if "attention_mask" in sample:
                    kwargs["attention_mask"] = sample["attention_mask"]
                outputs = self.model(**kwargs)
            x_pred = outputs.x_pred
            return x_pred[0, sample["vinput_mask"][0]].unsqueeze(0)

        for step_t in self.progress_bar(timesteps):
            t_pixeldit = 1.0 - step_t.float() / 1000.0
            sigma = (step_t.float() / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)

            x_pred_cond = _forward_once(cond_sample, z.clone(), t_pixeldit)
            v_cond = (x_pred_cond.float() - z.float()) / sigma

            if do_cfg:
                x_pred_uncond = _forward_once(uncond_sample, z.clone(), t_pixeldit)
                v_uncond = (x_pred_uncond.float() - z.float()) / sigma
                v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_guided = v_cond

            model_output = -v_guided
            z = self.scheduler.step(
                model_output.float(),
                step_t.to(dtype=torch.float32),
                z.float(),
                return_dict=False,
            )[0].to(dtype)

        img = (z + 1) / 2
        img = einops.rearrange(
            img.cpu().float(),
            "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
            H=h_patches,
            W=w_patches,
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )

        if output_type == "pt":
            images = img.clamp(0, 1)
        elif output_type == "np":
            images = np.clip(img.numpy().transpose(0, 2, 3, 1), 0, 1)
        else:
            arr = np.round(
                np.clip(img[0].numpy().transpose(1, 2, 0) * 255, 0, 255)
            ).astype(np.uint8)
            images = [Image.fromarray(arr).convert("RGB")]

        if not return_dict:
            return (images,)
        return HiDreamO1PipelineOutput(images=images)
