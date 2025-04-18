import argparse
import hashlib
import json
import os
import time
from typing import TYPE_CHECKING, Union, List
import sys


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
    KDPM2AncestralDiscreteScheduler
)
import torch
import re
from transformers import T5Tokenizer, T5EncoderModel, UMT5EncoderModel

SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816


def get_torch_dtype(dtype_str):
    # if it is a torch dtype, return it
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    if dtype_str == "float" or dtype_str == "fp32" or dtype_str == "single" or dtype_str == "float32":
        return torch.float
    if dtype_str == "fp16" or dtype_str == "half" or dtype_str == "float16":
        return torch.float16
    if dtype_str == "bf16" or dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "8bit" or dtype_str == "e4m3fn" or dtype_str == "float8":
        return torch.float8_e4m3fn
    return dtype_str


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


# https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(noise, noise_offset):
    if noise_offset is None or (noise_offset < 0.000001 and noise_offset > -0.000001):
        return noise
    if len(noise.shape) > 4:
        raise ValueError("Applying noise offset not supported for video models at this time.")
    noise = noise + noise_offset * torch.randn((noise.shape[0], noise.shape[1], 1, 1), device=noise.device)
    return noise


if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import PromptEmbeds


def concat_prompt_embeddings(
        unconditional: 'PromptEmbeds',
        conditional: 'PromptEmbeds',
        n_imgs: int,
):
    from toolkit.stable_diffusion_model import PromptEmbeds
    text_embeds = torch.cat(
        [unconditional.text_embeds, conditional.text_embeds]
    ).repeat_interleave(n_imgs, dim=0)
    pooled_embeds = None
    if unconditional.pooled_embeds is not None and conditional.pooled_embeds is not None:
        pooled_embeds = torch.cat(
            [unconditional.pooled_embeds, conditional.pooled_embeds]
        ).repeat_interleave(n_imgs, dim=0)
    return PromptEmbeds([text_embeds, pooled_embeds])


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


if TYPE_CHECKING:
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection


def text_tokenize(
        tokenizer: 'CLIPTokenizer',
        prompts: list[str],
        truncate: bool = True,
        max_length: int = None,
        max_length_multiplier: int = 4,
):
    # allow fo up to 4x the max length for long prompts
    if max_length is None:
        if truncate:
            max_length = tokenizer.model_max_length
        else:
            # allow up to 4x the max length for long prompts
            max_length = tokenizer.model_max_length * max_length_multiplier

    input_ids = tokenizer(
        prompts,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    if truncate or max_length == tokenizer.model_max_length:
        return input_ids
    else:
        # remove additional padding
        num_chunks = input_ids.shape[1] // tokenizer.model_max_length
        chunks = torch.chunk(input_ids, chunks=num_chunks, dim=1)

        # New list to store non-redundant chunks
        non_redundant_chunks = []

        for chunk in chunks:
            if not chunk.eq(chunk[0, 0]).all():  # Check if all elements in the chunk are the same as the first element
                non_redundant_chunks.append(chunk)

        input_ids = torch.cat(non_redundant_chunks, dim=1)
        return input_ids


# https://github.com/huggingface/diffusers/blob/78922ed7c7e66c20aa95159c7b7a6057ba7d590d/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L334-L348
def text_encode_xl(
        text_encoder: Union['CLIPTextModel', 'CLIPTextModelWithProjection'],
        tokens: torch.FloatTensor,
        num_images_per_prompt: int = 1,
        max_length: int = 77,  # not sure what default to put here, always pass one?
        truncate: bool = True,
):
    if truncate:
        # normal short prompt 77 tokens max
        prompt_embeds = text_encoder(
            tokens.to(text_encoder.device), output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer
    else:
        # handle long prompts
        prompt_embeds_list = []
        tokens = tokens.to(text_encoder.device)
        pooled_prompt_embeds = None
        for i in range(0, tokens.shape[-1], max_length):
            # todo run it through the in a single batch
            section_tokens = tokens[:, i: i + max_length]
            embeds = text_encoder(section_tokens, output_hidden_states=True)
            pooled_prompt_embed = embeds[0]
            if pooled_prompt_embeds is None:
                # we only want the first ( I think??)
                pooled_prompt_embeds = pooled_prompt_embed
            prompt_embed = embeds.hidden_states[-2]  # always penultimate layer
            prompt_embeds_list.append(prompt_embed)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompts_xl(
        tokenizers: list['CLIPTokenizer'],
        text_encoders: list[Union['CLIPTextModel', 'CLIPTextModelWithProjection']],
        prompts: list[str],
        prompts2: Union[list[str], None],
        num_images_per_prompt: int = 1,
        use_text_encoder_1: bool = True,  # sdxl
        use_text_encoder_2: bool = True,  # sdxl
        truncate: bool = True,
        max_length=None,
        dropout_prob=0.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool
    if prompts2 is None:
        prompts2 = prompts

    for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
        # todo, we are using a blank string to ignore that encoder for now.
        # find a better way to do this (zeroing?, removing it from the unet?)
        prompt_list_to_use = prompts if idx == 0 else prompts2
        if idx == 0 and not use_text_encoder_1:
            prompt_list_to_use = ["" for _ in prompts]
        if idx == 1 and not use_text_encoder_2:
            prompt_list_to_use = ["" for _ in prompts]

        if dropout_prob > 0.0:
            # randomly drop out prompts
            prompt_list_to_use = [
                prompt if torch.rand(1).item() > dropout_prob else "" for prompt in prompt_list_to_use
            ]

        text_tokens_input_ids = text_tokenize(tokenizer, prompt_list_to_use, truncate=truncate, max_length=max_length)
        # set the max length for the next one
        if idx == 0:
            max_length = text_tokens_input_ids.shape[-1]

        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, num_images_per_prompt, max_length=tokenizer.model_max_length,
            truncate=truncate
        )

        text_embeds_list.append(text_embeds)

    bs_embed = pooled_text_embeds.shape[0]
    pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds

def encode_prompts_sd3(
        tokenizers: list['CLIPTokenizer'],
        text_encoders: list[Union['CLIPTextModel', 'CLIPTextModelWithProjection', T5EncoderModel]],
        prompts: list[str],
        num_images_per_prompt: int = 1,
        truncate: bool = True,
        max_length=None,
        dropout_prob=0.0,
        pipeline = None,
):
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool

    prompt_2 = prompts
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    prompt_3 = prompts
    prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

    device = text_encoders[0].device

    prompt_embed, pooled_prompt_embed = pipeline._get_clip_prompt_embeds(
        prompt=prompts,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=None,
        clip_model_index=0,
    )
    prompt_2_embed, pooled_prompt_2_embed = pipeline._get_clip_prompt_embeds(
        prompt=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=None,
        clip_model_index=1,
    )
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

    t5_prompt_embed = pipeline._get_t5_prompt_embeds(
        prompt=prompt_3,
        num_images_per_prompt=num_images_per_prompt,
        device=device
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

    return prompt_embeds, pooled_prompt_embeds


# ref for long prompts https://github.com/huggingface/diffusers/issues/2136
def text_encode(text_encoder: 'CLIPTextModel', tokens, truncate: bool = True, max_length=None):
    if max_length is None and not truncate:
        raise ValueError("max_length must be set if truncate is True")
    try:
        tokens = tokens.to(text_encoder.device)
    except Exception as e:
        print(e)
        print("tokens.device", tokens.device)
        print("text_encoder.device", text_encoder.device)
        raise e

    if truncate:
        return text_encoder(tokens)[0]
    else:
        # handle long prompts
        prompt_embeds_list = []
        for i in range(0, tokens.shape[-1], max_length):
            prompt_embeds = text_encoder(tokens[:, i: i + max_length])[0]
            prompt_embeds_list.append(prompt_embeds)

        return torch.cat(prompt_embeds_list, dim=1)


def encode_prompts(
        tokenizer: 'CLIPTokenizer',
        text_encoder: 'CLIPTextModel',
        prompts: list[str],
        truncate: bool = True,
        max_length=None,
        dropout_prob=0.0,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if dropout_prob > 0.0:
        # randomly drop out prompts
        prompts = [
            prompt if torch.rand(1).item() > dropout_prob else "" for prompt in prompts
        ]

    text_tokens = text_tokenize(tokenizer, prompts, truncate=truncate, max_length=max_length)
    text_embeddings = text_encode(text_encoder, text_tokens, truncate=truncate, max_length=max_length)

    return text_embeddings


def encode_prompts_pixart(
        tokenizer: 'T5Tokenizer',
        text_encoder: 'T5EncoderModel',
        prompts: list[str],
        truncate: bool = True,
        max_length=None,
        dropout_prob=0.0,
):
    if max_length is None:
        # See Section 3.1. of the paper.
        max_length = 120

    if dropout_prob > 0.0:
        # randomly drop out prompts
        prompts = [
            prompt if torch.rand(1).item() > dropout_prob else "" for prompt in prompts
        ]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_length - 1: -1])

    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.to(text_encoder.device)

    text_input_ids = text_input_ids.to(text_encoder.device)

    prompt_embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)

    return prompt_embeds.last_hidden_state, prompt_attention_mask


def encode_prompts_auraflow(
        tokenizer: 'T5Tokenizer',
        text_encoder: 'UMT5EncoderModel',
        prompts: list[str],
        truncate: bool = True,
        max_length=None,
        dropout_prob=0.0,
):
    if max_length is None:
        max_length = 256

    if dropout_prob > 0.0:
        # randomly drop out prompts
        prompts = [
            prompt if torch.rand(1).item() > dropout_prob else "" for prompt in prompts
        ]

    device = text_encoder.device

    text_inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    text_input_ids = text_inputs["input_ids"]
    untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_length - 1: -1])

    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    prompt_embeds = text_encoder(**text_inputs)[0]
    prompt_attention_mask = text_inputs["attention_mask"].unsqueeze(-1).expand(prompt_embeds.shape)
    prompt_embeds = prompt_embeds * prompt_attention_mask

    return prompt_embeds, prompt_attention_mask

def encode_prompts_flux(
        tokenizer: List[Union['CLIPTokenizer','T5Tokenizer']],
        text_encoder: List[Union['CLIPTextModel', 'T5EncoderModel']],
        prompts: list[str],
        truncate: bool = True,
        max_length=None,
        dropout_prob=0.0,
        attn_mask: bool = False,
):
    if max_length is None:
        max_length = 512

    if dropout_prob > 0.0:
        # randomly drop out prompts
        prompts = [
            prompt if torch.rand(1).item() > dropout_prob else "" for prompt in prompts
        ]

    device = text_encoder[0].device
    dtype = text_encoder[0].dtype

    batch_size = len(prompts)

    # clip
    text_inputs = tokenizer[0](
        prompts,
        padding="max_length",
        max_length=tokenizer[0].model_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder[0](text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    pooled_prompt_embeds = prompt_embeds.pooler_output
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype, device=device)

    # T5
    text_inputs = tokenizer[1](
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder[1](text_input_ids.to(device), output_hidden_states=False)[0]

    dtype = text_encoder[1].dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if attn_mask:
        prompt_attention_mask = text_inputs["attention_mask"].unsqueeze(-1).expand(prompt_embeds.shape)
        prompt_embeds = prompt_embeds * prompt_attention_mask.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)

    return prompt_embeds, pooled_prompt_embeds


# for XL
def get_add_time_ids(
        height: int,
        width: int,
        dynamic_crops: bool = False,
        dtype: torch.dtype = torch.float32,
):
    if dynamic_crops:
        # random float scale between 1 and 3
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        # random position
        crops_coords_top_left = (
            torch.randint(0, original_size[0] - height, (1,)).item(),
            torch.randint(0, original_size[1] - width, (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    # this is expected as 6
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # this is expected as 2816
    passed_add_embed_dim = (
            UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)  # 256 * 6
            + TEXT_ENCODER_2_PROJECTION_DIM  # + 1280
    )
    if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
        raise ValueError(
            f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def concat_embeddings(
        unconditional: torch.FloatTensor,
        conditional: torch.FloatTensor,
        n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


def add_all_snr_to_noise_scheduler(noise_scheduler, device):
    try:
        if hasattr(noise_scheduler, "all_snr"):
            return
        # compute it
        with torch.no_grad():
            alphas_cumprod = noise_scheduler.alphas_cumprod
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
            alpha = sqrt_alphas_cumprod
            sigma = sqrt_one_minus_alphas_cumprod
            all_snr = (alpha / sigma) ** 2
            all_snr.requires_grad = False
        noise_scheduler.all_snr = all_snr.to(device)
    except Exception as e:
        # just move on
        pass


def get_all_snr(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return noise_scheduler.all_snr.to(device)
    # compute it
    with torch.no_grad():
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        alpha = sqrt_alphas_cumprod
        sigma = sqrt_one_minus_alphas_cumprod
        all_snr = (alpha / sigma) ** 2
        all_snr.requires_grad = False
    return all_snr.to(device)

class LearnableSNRGamma:
    """
    This is a trainer for learnable snr gamma
    It will adapt to the dataset and attempt to adjust the snr multiplier to balance the loss over the timesteps
    """
    def __init__(self, noise_scheduler: Union['DDPMScheduler'], device='cuda'):
        self.device = device
        self.noise_scheduler: Union['DDPMScheduler'] = noise_scheduler
        self.offset_1 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
        self.offset_2 = torch.nn.Parameter(torch.tensor(0.777, dtype=torch.float32, device=device))
        self.scale = torch.nn.Parameter(torch.tensor(4.14, dtype=torch.float32, device=device))
        self.gamma = torch.nn.Parameter(torch.tensor(2.03, dtype=torch.float32, device=device))
        self.optimizer = torch.optim.AdamW([self.offset_1, self.offset_2, self.gamma, self.scale], lr=0.01)
        self.buffer = []
        self.max_buffer_size = 20

    def forward(self, loss, timesteps):
        # do a our train loop for lsnr here and return our values detached
        loss = loss.detach()
        with torch.no_grad():
            loss_chunks = torch.chunk(loss, loss.shape[0], dim=0)
            for loss_chunk in loss_chunks:
                self.buffer.append(loss_chunk.mean().detach())
                if len(self.buffer) > self.max_buffer_size:
                    self.buffer.pop(0)
            all_snr = get_all_snr(self.noise_scheduler, loss.device)
            snr: torch.Tensor = torch.stack([all_snr[t] for t in timesteps]).detach().float().to(loss.device)
        base_snrs = snr.clone().detach()
        snr.requires_grad = True
        snr = (snr + self.offset_1) * self.scale + self.offset_2

        gamma_over_snr = torch.div(torch.ones_like(snr) * self.gamma, snr)
        snr_weight = torch.abs(gamma_over_snr).float().to(loss.device)  # directly using gamma over snr
        snr_adjusted_loss = loss * snr_weight
        with torch.no_grad():
            target = torch.mean(torch.stack(self.buffer)).detach()

        # local_loss = torch.mean(torch.abs(snr_adjusted_loss - target))
        squared_differences = (snr_adjusted_loss - target) ** 2
        local_loss = torch.mean(squared_differences)
        local_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return base_snrs, self.gamma.detach(), self.offset_1.detach(), self.offset_2.detach(), self.scale.detach()


def apply_learnable_snr_gos(
        loss,
        timesteps,
        learnable_snr_trainer: LearnableSNRGamma
):

    snr, gamma, offset_1, offset_2, scale = learnable_snr_trainer.forward(loss, timesteps)

    snr = (snr + offset_1) * scale + offset_2

    gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)
    snr_weight = torch.abs(gamma_over_snr).float().to(loss.device)  # directly using gamma over snr
    snr_adjusted_loss = loss * snr_weight

    return snr_adjusted_loss


def apply_snr_weight(
        loss,
        timesteps,
        noise_scheduler: Union['DDPMScheduler'],
        gamma,
        fixed=False,
):
    # will get it from noise scheduler if exist or will calculate it if not
    all_snr = get_all_snr(noise_scheduler, loss.device)
    # step_indices = []
    # for t in timesteps:
    #     for i, st in enumerate(noise_scheduler.timesteps):
    #         if st == t:
    #             step_indices.append(i)
    #             break
    # this breaks on some schedulers
    # step_indices = [(noise_scheduler.timesteps == t).nonzero().item() for t in timesteps]

    offset = 0
    if noise_scheduler.timesteps[0] == 1000:
        offset = 1
    snr = torch.stack([all_snr[(t - offset).int()] for t in timesteps])
    gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)
    if fixed:
        snr_weight = gamma_over_snr.float().to(loss.device)  # directly using gamma over snr
    else:
        snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float().to(loss.device)
    snr_adjusted_loss = loss * snr_weight

    return snr_adjusted_loss


def precondition_model_outputs_flow_match(model_output, model_input, timestep_tensor, noise_scheduler):
    mo_chunks = torch.chunk(model_output, model_output.shape[0], dim=0)
    mi_chunks = torch.chunk(model_input, model_input.shape[0], dim=0)
    timestep_chunks = torch.chunk(timestep_tensor, timestep_tensor.shape[0], dim=0)
    out_chunks = []
    # unsqueeze if timestep is zero dim
    for idx in range(model_output.shape[0]):
        sigmas = noise_scheduler.get_sigmas(timestep_chunks[idx], n_dim=model_output.ndim,
                                                 dtype=model_output.dtype, device=model_output.device)
        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        out = mo_chunks[idx] * (-sigmas) + mi_chunks[idx]
        out_chunks.append(out)
    return torch.cat(out_chunks, dim=0)
