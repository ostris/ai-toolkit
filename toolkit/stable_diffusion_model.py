from typing import Union, OrderedDict
import sys
import os

from safetensors.torch import save_file

from toolkit.paths import REPOS_ROOT
from toolkit.train_tools import get_torch_dtype

sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))
from leco import train_util
import torch
from library import model_util
from library.sdxl_model_util import convert_text_encoder_2_state_dict_to_sdxl


class PromptEmbeds:
    text_embeds: torch.FloatTensor
    pooled_embeds: Union[torch.FloatTensor, None]

    def __init__(self, args) -> None:
        if isinstance(args, list) or isinstance(args, tuple):
            # xl
            self.text_embeds = args[0]
            self.pooled_embeds = args[1]
        else:
            # sdv1.x, sdv2.x
            self.text_embeds = args
            self.pooled_embeds = None

    def to(self, **kwargs):
        self.text_embeds = self.text_embeds.to(**kwargs)
        if self.pooled_embeds is not None:
            self.pooled_embeds = self.pooled_embeds.to(**kwargs)
        return self


class StableDiffusion:
    def __init__(
            self,
            vae,
            tokenizer,
            text_encoder,
            unet,
            noise_scheduler,
            is_xl=False
    ):
        # text encoder has a list of 2 for xl
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.is_xl = is_xl

    def encode_prompt(self, prompt, num_images_per_prompt=1) -> PromptEmbeds:
        prompt = prompt
        # if it is not a list, make it one
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.is_xl:
            return PromptEmbeds(
                train_util.encode_prompts_xl(
                    self.tokenizer,
                    self.text_encoder,
                    prompt,
                    num_images_per_prompt=num_images_per_prompt,
                )
            )
        else:
            return PromptEmbeds(
                train_util.encode_prompts(
                    self.tokenizer, self.text_encoder, prompt
                )
            )

    def save(self, output_file: str, meta: OrderedDict, save_dtype=get_torch_dtype('fp16'), logit_scale=None):
        # todo see what logit scale is
        if self.is_xl:

            state_dict = {}

            def update_sd(prefix, sd):
                for k, v in sd.items():
                    key = prefix + k
                    v = v.detach().clone().to("cpu").to(get_torch_dtype(save_dtype))
                    state_dict[key] = v

            # Convert the UNet model
            update_sd("model.diffusion_model.", self.unet.state_dict())

            # Convert the text encoders
            update_sd("conditioner.embedders.0.transformer.", self.text_encoder[0].state_dict())

            text_enc2_dict = convert_text_encoder_2_state_dict_to_sdxl(self.text_encoder[1].state_dict(), logit_scale)
            update_sd("conditioner.embedders.1.model.", text_enc2_dict)

            # Convert the VAE
            vae_dict = model_util.convert_vae_state_dict(self.vae.state_dict())
            update_sd("first_stage_model.", vae_dict)

            # Put together new checkpoint
            key_count = len(state_dict.keys())
            new_ckpt = {"state_dict": state_dict}

            if model_util.is_safetensors(output_file):
                save_file(state_dict, output_file)
            else:
                torch.save(new_ckpt, output_file, meta)

            return key_count
        else:
            raise NotImplementedError("sdv1.x, sdv2.x is not implemented yet")
