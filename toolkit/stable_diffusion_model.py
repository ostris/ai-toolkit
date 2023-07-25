from typing import Union
import sys
import os
from toolkit.paths import REPOS_ROOT
sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))
from leco import train_util
import torch


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
