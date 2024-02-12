import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from typing import Union, TYPE_CHECKING

from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5Tokenizer, CLIPVisionModelWithProjection
from toolkit.paths import REPOS_ROOT
sys.path.append(REPOS_ROOT)

from ipadapter.ip_adapter.attention_processor import AttnProcessor2_0

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.custom_adapter import CustomAdapter


class VisionDirectAdapterAttnProcessor(nn.Module):
    r"""
    Attention processor for Custom TE for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        adapter
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, adapter=None,
                 adapter_hidden_size=None):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.adapter_ref: weakref.ref = weakref.ref(adapter)

        self.hidden_size = hidden_size
        self.adapter_hidden_size = adapter_hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

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

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # only use one TE or the other. If our adapter is active only use ours
        if self.is_active and self.conditional_embeds is not None:

            adapter_hidden_states = self.conditional_embeds
            if adapter_hidden_states.shape[0] < batch_size:
                adapter_hidden_states = torch.cat([
                    self.unconditional_embeds,
                    adapter_hidden_states
                ])
            # conditional_batch_size = adapter_hidden_states.shape[0]
            # conditional_query = query

            # for ip-adapter
            vd_key = self.to_k_adapter(adapter_hidden_states)
            vd_value = self.to_v_adapter(adapter_hidden_states)

            vd_key = vd_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            vd_value = vd_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            vd_hidden_states = F.scaled_dot_product_attention(
                query, vd_key, vd_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            vd_hidden_states = vd_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            vd_hidden_states = vd_hidden_states.to(query.dtype)

            hidden_states = hidden_states + self.scale * vd_hidden_states


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


class VisionDirectAdapter(torch.nn.Module):
    def __init__(
            self,
            adapter: 'CustomAdapter',
            sd: 'StableDiffusion',
            vision_model: Union[CLIPVisionModelWithProjection],
    ):
        super(VisionDirectAdapter, self).__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.vision_model_ref: weakref.ref = weakref.ref(vision_model)

        self.token_size = vision_model.config.hidden_size

        # init adapter modules
        attn_procs = {}
        unet_sd = sd.unet.state_dict()
        for name in sd.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else sd.unet.config['cross_attention_dim']
            if name.startswith("mid_block"):
                hidden_size = sd.unet.config['block_out_channels'][-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(sd.unet.config['block_out_channels']))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = sd.unet.config['block_out_channels'][block_id]
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

                attn_procs[name] = VisionDirectAdapterAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    adapter=self,
                    adapter_hidden_size=self.token_size
                )
                attn_procs[name].load_state_dict(weights)
        sd.unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(sd.unet.attn_processors.values())

    # make a getter to see if is active
    @property
    def is_active(self):
        return self.adapter_ref().is_active

    def forward(self, input):
        return input
