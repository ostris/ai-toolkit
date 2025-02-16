from functools import partial
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from transformers import AutoModel, AutoTokenizer, Qwen2Model, LlamaModel, Qwen2Tokenizer, LlamaTokenizer

from toolkit import train_tools
from toolkit.prompt_utils import PromptEmbeds
from diffusers import Transformer2DModel
from toolkit.dequantize import patch_dequantization_on_save


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

def new_block_forward(
    self: FluxTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if self._adapter_ref().is_active:
        return self._new_block_ref()(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
    else:
        return self._orig_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)


class LLMAdapter(torch.nn.Module):
    def __init__(
            self,
            adapter: 'CustomAdapter',
            sd: 'StableDiffusion',
            llm: LLM,
            tokenizer: LLMTokenizer,
            num_cloned_blocks: int = 0,
    ):
        super(LLMAdapter, self).__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.llm_ref: weakref.ref = weakref.ref(llm)
        self.tokenizer_ref: weakref.ref = weakref.ref(tokenizer)
        self.num_cloned_blocks = num_cloned_blocks
        self.apply_embedding_mask = False
        # make sure we can pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # self.system_prompt = ""
        self.system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> "
        
        # determine length of system prompt
        sys_prompt_tokenized = tokenizer(
            [self.system_prompt],
            padding="longest",
            return_tensors="pt",
        )

        sys_prompt_tokenized_ids = sys_prompt_tokenized.input_ids[0]
        
        self.system_prompt_length = sys_prompt_tokenized_ids.shape[0]
        
        print(f"System prompt length: {self.system_prompt_length}")

        self.hidden_size = llm.config.hidden_size
        
        blocks = []

        if sd.is_flux:
            self.apply_embedding_mask = True
            self.context_embedder = nn.Linear(
                self.hidden_size, sd.unet.inner_dim)
            self.sequence_length = 512
            sd.unet.context_embedder._orig_forward = sd.unet.context_embedder.forward
            sd.unet.context_embedder.forward = partial(
                new_context_embedder_forward, sd.unet.context_embedder)
            sd.unet.context_embedder._context_embedder_ref = weakref.ref(self.context_embedder)
            # add a is active property to the context embedder
            sd.unet.context_embedder._adapter_ref = self.adapter_ref
            
            for idx in range(self.num_cloned_blocks):
                block = FluxTransformerBlock(
                    dim=sd.unet.inner_dim,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                # patch it in case it is quantized
                patch_dequantization_on_save(sd.unet.transformer_blocks[idx])
                state_dict = sd.unet.transformer_blocks[idx].state_dict()
                for key, value in state_dict.items():
                    block.state_dict()[key].copy_(value)
                blocks.append(block)
                orig_block = sd.unet.transformer_blocks[idx]
                orig_block._orig_forward = orig_block.forward
                orig_block.forward = partial(
                    new_block_forward, orig_block)
                orig_block._new_block_ref = weakref.ref(block)
                orig_block._adapter_ref = self.adapter_ref
            
        elif sd.is_lumina2:
            self.context_embedder = nn.Linear(
                self.hidden_size, sd.unet.hidden_size)
            self.sequence_length = 256
        else:
            raise ValueError(
                "llm adapter currently only supports flux or lumina2")
        
        self.blocks = nn.ModuleList(blocks)

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
            max_length=max_sequence_length + self.system_prompt_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)
        
        # remove the system prompt from the input and attention mask
        
        prompt_embeds = text_encoder(
            text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        prompt_embeds = prompt_embeds.hidden_states[-1]
        
        prompt_embeds = prompt_embeds[:, self.system_prompt_length:]
        prompt_attention_mask = prompt_attention_mask[:, self.system_prompt_length:]

        dtype = text_encoder.dtype

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask

    # make a getter to see if is active

    @property
    def is_active(self):
        return self.adapter_ref().is_active

    def encode_text(self, prompt):

        prompt = prompt if isinstance(prompt, list) else [prompt]

        prompt = [self.system_prompt + p for p in prompt]
        # prompt = [self.system_prompt + p for p in prompt]

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
