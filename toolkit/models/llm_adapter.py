from functools import partial
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from typing import List, Optional, Tuple, Union, TYPE_CHECKING


from transformers import AutoModel, AutoTokenizer, Qwen2Model, LlamaModel, Qwen2Tokenizer, LlamaTokenizer

from toolkit import train_tools
from toolkit.prompt_utils import PromptEmbeds
from diffusers import Transformer2DModel


if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion, PixArtSigmaPipeline
    from toolkit.custom_adapter import CustomAdapter

LLM = Union[Qwen2Model, LlamaModel]
LLMTokenizer = Union[Qwen2Tokenizer, LlamaTokenizer]


def new_context_embedder_forward(self, x):
    if self._adapter_ref().is_active:
        x = self._context_embedder_ref()(x)
    else:
        x = self._orig_forward(x)
    return x


class LLMAdapter(torch.nn.Module):
    def __init__(
            self,
            adapter: 'CustomAdapter',
            sd: 'StableDiffusion',
            llm: LLM,
            tokenizer: LLMTokenizer,
    ):
        super(LLMAdapter, self).__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.llm_ref: weakref.ref = weakref.ref(llm)
        self.tokenizer_ref: weakref.ref = weakref.ref(tokenizer)

        self.system_prompt = ""
        # self.system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> "

        self.hidden_size = llm.config.hidden_size

        if sd.is_flux:
            self.context_embedder = nn.Linear(
                self.hidden_size, sd.unet.inner_dim)
            self.sequence_length = 512
            sd.unet.context_embedder._orig_forward = sd.unet.context_embedder.forward
            sd.unet.context_embedder.forward = partial(
                new_context_embedder_forward, sd.unet.context_embedder)
            sd.unet.context_embedder._context_embedder_ref = weakref.ref(self.context_embedder)
            # add a is active property to the context embedder
            sd.unet.context_embedder._adapter_ref = self.adapter_ref
            
        elif sd.is_lumina2:
            self.context_embedder = nn.Linear(
                self.hidden_size, sd.unet.hidden_size)
            self.sequence_length = 256
        else:
            raise ValueError(
                "llm adapter currently only supports flux or lumina2")

    def _get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizer_ref()
        text_encoder = self.llm_ref()
        device = text_encoder.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(
            text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        prompt_embeds = prompt_embeds.hidden_states[-2]

        dtype = text_encoder.dtype

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask

    # make a getter to see if is active

    @property
    def is_active(self):
        return self.adapter_ref().is_active

    def encode_text(self, prompt):

        prompt = prompt if isinstance(prompt, list) else [prompt]

        # prompt = [self.system_prompt + " <Prompt Start> " + p for p in prompt]
        prompt = [self.system_prompt + p for p in prompt]

        prompt_embeds, prompt_attention_mask = self._get_prompt_embeds(
            prompt=prompt,
            max_sequence_length=self.sequence_length,
        )

        prompt_embeds = PromptEmbeds(
            prompt_embeds,
            attention_mask=prompt_attention_mask,
        ).detach()

        return prompt_embeds

    def forward(self, input):
        return input
