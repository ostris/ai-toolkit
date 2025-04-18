import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from typing import Union, TYPE_CHECKING
from transformers import T5EncoderModel, CLIPTokenizer, T5Tokenizer, CLIPTextModelWithProjection
from toolkit import train_tools
from toolkit.prompt_utils import PromptEmbeds
from diffusers import Transformer2DModel
from toolkit.util.ip_adapter_utils import AttnProcessor2_0


if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.custom_adapter import CustomAdapter


class TEAdapterCaptionProjection(nn.Module):
    def __init__(self, caption_channels, adapter: 'TEAdapter'):
        super().__init__()
        in_features = caption_channels
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        sd = adapter.sd_ref()
        self.parent_module_ref = weakref.ref(sd.unet.caption_projection)
        parent_module = self.parent_module_ref()
        self.linear_1 = nn.Linear(
            in_features=in_features,
            out_features=parent_module.linear_1.out_features,
            bias=True
        )
        self.linear_2 = nn.Linear(
            in_features=parent_module.linear_2.in_features,
            out_features=parent_module.linear_2.out_features,
            bias=True
        )

        # save the orig forward
        parent_module.linear_1.orig_forward = parent_module.linear_1.forward
        parent_module.linear_2.orig_forward = parent_module.linear_2.forward

        # replace original forward
        parent_module.orig_forward = parent_module.forward
        parent_module.forward = self.forward


    @property
    def is_active(self):
        return self.adapter_ref().is_active

    @property
    def unconditional_embeds(self):
        return self.adapter_ref().adapter_ref().unconditional_embeds

    @property
    def conditional_embeds(self):
        return self.adapter_ref().adapter_ref().conditional_embeds

    def forward(self, caption):
        if self.is_active and self.conditional_embeds is not None:
            adapter_hidden_states = self.conditional_embeds.text_embeds
            # check if we are doing unconditional
            if self.unconditional_embeds is not None and adapter_hidden_states.shape[0] != caption.shape[0]:
                # concat unconditional to match the hidden state batch size
                if self.unconditional_embeds.text_embeds.shape[0] == 1 and adapter_hidden_states.shape[0] != 1:
                    unconditional = torch.cat([self.unconditional_embeds.text_embeds] * adapter_hidden_states.shape[0], dim=0)
                else:
                    unconditional = self.unconditional_embeds.text_embeds
                adapter_hidden_states = torch.cat([unconditional, adapter_hidden_states], dim=0)
            hidden_states = self.linear_1(adapter_hidden_states)
            hidden_states = self.parent_module_ref().act_1(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            return hidden_states
        else:
            return self.parent_module_ref().orig_forward(caption)


class TEAdapterAttnProcessor(nn.Module):
    r"""
    Attention processor for Custom TE for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
        adapter
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, adapter=None,
                 adapter_hidden_size=None, layer_name=None):
        super().__init__()
        self.layer_name = layer_name

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.adapter_ref: weakref.ref = weakref.ref(adapter)

        self.hidden_size = hidden_size
        self.adapter_hidden_size = adapter_hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_adapter = nn.Linear(adapter_hidden_size, hidden_size, bias=False)
        self.to_v_adapter = nn.Linear(adapter_hidden_size, hidden_size, bias=False)

    @property
    def is_active(self):
        return self.adapter_ref().is_active

    @property
    def unconditional_embeds(self):
        return self.adapter_ref().adapter_ref().unconditional_embeds

    @property
    def conditional_embeds(self):
        return self.adapter_ref().adapter_ref().conditional_embeds

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        is_active = self.adapter_ref().is_active
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # will be none if disabled
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # only use one TE or the other. If our adapter is active only use ours
        if self.is_active and self.conditional_embeds is not None:
            adapter_hidden_states = self.conditional_embeds.text_embeds
            # check if we are doing unconditional
            if self.unconditional_embeds is not None and adapter_hidden_states.shape[0] != encoder_hidden_states.shape[0]:
                # concat unconditional to match the hidden state batch size
                if self.unconditional_embeds.text_embeds.shape[0] == 1 and adapter_hidden_states.shape[0] != 1:
                    unconditional = torch.cat([self.unconditional_embeds.text_embeds] * adapter_hidden_states.shape[0], dim=0)
                else:
                    unconditional = self.unconditional_embeds.text_embeds
                adapter_hidden_states = torch.cat([unconditional, adapter_hidden_states], dim=0)
            # for ip-adapter
            key = self.to_k_adapter(adapter_hidden_states)
            value = self.to_v_adapter(adapter_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        try:
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        except RuntimeError:
            raise RuntimeError(f"key shape: {key.shape}, value shape: {value.shape}")

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # remove attn mask if doing clip
        if self.adapter_ref().adapter_ref().config.text_encoder_arch == "clip":
            attention_mask = None

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class TEAdapter(torch.nn.Module):
    def __init__(
            self,
            adapter: 'CustomAdapter',
            sd: 'StableDiffusion',
            te: Union[T5EncoderModel],
            tokenizer: CLIPTokenizer
    ):
        super(TEAdapter, self).__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.te_ref: weakref.ref = weakref.ref(te)
        self.tokenizer_ref: weakref.ref = weakref.ref(tokenizer)
        self.adapter_modules = []
        self.caption_projection = None
        self.embeds_store = []
        is_pixart = sd.is_pixart

        if self.adapter_ref().config.text_encoder_arch == "t5" or self.adapter_ref().config.text_encoder_arch == "pile-t5":
            self.token_size = self.te_ref().config.d_model
        else:
            self.token_size = self.te_ref().config.hidden_size

        # add text projection if is sdxl
        self.text_projection = None
        if sd.is_xl:
            clip_with_projection: CLIPTextModelWithProjection = sd.text_encoder[0]
            self.text_projection = nn.Linear(te.config.hidden_size, clip_with_projection.config.projection_dim, bias=False)

        # init adapter modules
        attn_procs = {}
        unet_sd = sd.unet.state_dict()
        attn_dict_map = {

        }
        module_idx = 0
        # init adapter modules
        attn_procs = {}
        unet_sd = sd.unet.state_dict()
        attn_processor_keys = []
        if is_pixart:
            transformer: Transformer2DModel = sd.unet
            for i, module in transformer.transformer_blocks.named_children():
                attn_processor_keys.append(f"transformer_blocks.{i}.attn1")

                # cross attention
                attn_processor_keys.append(f"transformer_blocks.{i}.attn2")

        else:
            attn_processor_keys = list(sd.unet.attn_processors.keys())

        attn_processor_names = []

        blocks = []
        transformer_blocks = []
        for name in attn_processor_keys:
            cross_attention_dim = None if name.endswith("attn1.processor") or name.endswith("attn.1") or name.endswith("attn1") else \
                sd.unet.config['cross_attention_dim']
            if name.startswith("mid_block"):
                hidden_size = sd.unet.config['block_out_channels'][-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(sd.unet.config['block_out_channels']))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = sd.unet.config['block_out_channels'][block_id]
            elif name.startswith("transformer"):
                hidden_size = sd.unet.config['cross_attention_dim']
            else:
                # they didnt have this, but would lead to undefined below
                raise ValueError(f"unknown attn processor name: {name}")
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                layer_name = name.split(".processor")[0]
                to_k_adapter = unet_sd[layer_name + ".to_k.weight"]
                to_v_adapter = unet_sd[layer_name + ".to_v.weight"]

                # add zero padding to the adapter
                if to_k_adapter.shape[1] < self.token_size:
                    to_k_adapter = torch.cat([
                        to_k_adapter,
                        torch.randn(to_k_adapter.shape[0], self.token_size - to_k_adapter.shape[1]).to(
                            to_k_adapter.device, dtype=to_k_adapter.dtype) * 0.01
                    ],
                        dim=1
                    )
                    to_v_adapter = torch.cat([
                        to_v_adapter,
                        torch.randn(to_v_adapter.shape[0], self.token_size - to_v_adapter.shape[1]).to(
                            to_k_adapter.device, dtype=to_k_adapter.dtype) * 0.01
                    ],
                        dim=1
                    )
                elif to_k_adapter.shape[1] > self.token_size:
                    to_k_adapter = to_k_adapter[:, :self.token_size]
                    to_v_adapter = to_v_adapter[:, :self.token_size]
                else:
                    to_k_adapter = to_k_adapter
                    to_v_adapter = to_v_adapter

                # todo resize to the TE hidden size
                weights = {
                    "to_k_adapter.weight": to_k_adapter,
                    "to_v_adapter.weight": to_v_adapter,
                }
                
                if self.sd_ref().is_pixart:
                    # pixart is much more sensitive
                    weights = {
                        "to_k_adapter.weight": weights["to_k_adapter.weight"] * 0.01,
                        "to_v_adapter.weight": weights["to_v_adapter.weight"] * 0.01,
                    }

                attn_procs[name] = TEAdapterAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.adapter_ref().config.num_tokens,
                    adapter=self,
                    adapter_hidden_size=self.token_size,
                    layer_name=layer_name
                )
                attn_procs[name].load_state_dict(weights)
                self.adapter_modules.append(attn_procs[name])
        if self.sd_ref().is_pixart:
            # we have to set them ourselves
            transformer: Transformer2DModel = sd.unet
            for i, module in transformer.transformer_blocks.named_children():
                module.attn1.processor = attn_procs[f"transformer_blocks.{i}.attn1"]
                module.attn2.processor = attn_procs[f"transformer_blocks.{i}.attn2"]
            self.adapter_modules = torch.nn.ModuleList(
                [
                    transformer.transformer_blocks[i].attn2.processor for i in
                    range(len(transformer.transformer_blocks))
                ])
            self.caption_projection = TEAdapterCaptionProjection(
                caption_channels=self.token_size,
                adapter=self,
            )

        else:
            sd.unet.set_attn_processor(attn_procs)
            self.adapter_modules = torch.nn.ModuleList(sd.unet.attn_processors.values())

    # make a getter to see if is active
    @property
    def is_active(self):
        return self.adapter_ref().is_active

    def encode_text(self, text):
        te: T5EncoderModel = self.te_ref()
        tokenizer: T5Tokenizer = self.tokenizer_ref()
        attn_mask_float = None

        # input_ids = tokenizer(
        #     text,
        #     max_length=77,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt",
        # ).input_ids.to(te.device)
        # outputs = te(input_ids=input_ids)
        # outputs = outputs.last_hidden_state
        if self.adapter_ref().config.text_encoder_arch == "clip":
            embeds = train_tools.encode_prompts(
                tokenizer,
                te,
                text,
                truncate=True,
                max_length=self.adapter_ref().config.num_tokens,
            )
            attention_mask = torch.ones(embeds.shape[:2], device=embeds.device)

        elif self.adapter_ref().config.text_encoder_arch == "pile-t5":
            # just use aura pile
            embeds, attention_mask = train_tools.encode_prompts_auraflow(
                tokenizer,
                te,
                text,
                truncate=True,
                max_length=self.adapter_ref().config.num_tokens,
            )

        else:
            embeds, attention_mask = train_tools.encode_prompts_pixart(
                tokenizer,
                te,
                text,
                truncate=True,
                max_length=self.adapter_ref().config.num_tokens,
            )
        if attention_mask is not None:
            attn_mask_float = attention_mask.to(embeds.device, dtype=embeds.dtype)
        if self.text_projection is not None:
            # pool the output of embeds ignoring 0 in the attention mask
            if attn_mask_float is not None:
                pooled_output = embeds * attn_mask_float.unsqueeze(-1)
            else:
                pooled_output = embeds

            # reduce along dim 1 while maintaining batch and dim 2
            pooled_output_sum = pooled_output.sum(dim=1)

            if attn_mask_float is not None:
                attn_mask_sum = attn_mask_float.sum(dim=1).unsqueeze(-1)

                pooled_output = pooled_output_sum / attn_mask_sum

            pooled_embeds = self.text_projection(pooled_output)

            prompt_embeds = PromptEmbeds(
                (embeds, pooled_embeds),
                attention_mask=attention_mask,
            ).detach()

        else:

            prompt_embeds = PromptEmbeds(
                embeds,
                attention_mask=attention_mask,
            ).detach()

        return prompt_embeds



    def forward(self, input):
        return input
