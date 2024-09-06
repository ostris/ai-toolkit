import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from typing import Union, TYPE_CHECKING, Optional
from collections import OrderedDict

from diffusers import Transformer2DModel, FluxTransformer2DModel
from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5Tokenizer, CLIPVisionModelWithProjection

from toolkit.config_modules import AdapterConfig
from toolkit.paths import REPOS_ROOT
sys.path.append(REPOS_ROOT)


if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.custom_adapter import CustomAdapter


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.1, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x

class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
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
                 adapter_hidden_size=None, has_bias=False, **kwargs):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.adapter_ref: weakref.ref = weakref.ref(adapter)

        self.hidden_size = hidden_size
        self.adapter_hidden_size = adapter_hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        self.to_k_adapter = nn.Linear(adapter_hidden_size, hidden_size, bias=has_bias)
        self.to_v_adapter = nn.Linear(adapter_hidden_size, hidden_size, bias=has_bias)

    @property
    def is_active(self):
        return self.adapter_ref().is_active
        # return False

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
                ], dim=0)
                # if it is image embeds, we need to add a 1 dim at inx 1
            if len(adapter_hidden_states.shape) == 2:
                adapter_hidden_states = adapter_hidden_states.unsqueeze(1)
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


class CustomFluxVDAttnProcessor2_0(torch.nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, adapter=None,
                 adapter_hidden_size=None, has_bias=False, block_idx=0, **kwargs):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.adapter_ref: weakref.ref = weakref.ref(adapter)

        self.hidden_size = hidden_size
        self.adapter_hidden_size = adapter_hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.block_idx = block_idx

        self.to_k_adapter = nn.Linear(adapter_hidden_size, hidden_size, bias=has_bias)
        self.to_v_adapter = nn.Linear(adapter_hidden_size, hidden_size, bias=has_bias)

    @property
    def is_active(self):
        return self.adapter_ref().is_active
        # return False

    @property
    def unconditional_embeds(self):
        return self.adapter_ref().adapter_ref().unconditional_embeds

    @property
    def conditional_embeds(self):
        return self.adapter_ref().adapter_ref().conditional_embeds

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # begin ip adapter
        if self.is_active and self.conditional_embeds is not None:
            adapter_hidden_states = self.conditional_embeds
            block_scaler = self.adapter_ref().block_scaler
            if block_scaler is not None:
                block_scaler = block_scaler[self.block_idx]

            if adapter_hidden_states.shape[0] < batch_size:
                adapter_hidden_states = torch.cat([
                    self.unconditional_embeds,
                    adapter_hidden_states
                ], dim=0)
                # if it is image embeds, we need to add a 1 dim at inx 1
            if len(adapter_hidden_states.shape) == 2:
                adapter_hidden_states = adapter_hidden_states.unsqueeze(1)
            # conditional_batch_size = adapter_hidden_states.shape[0]
            # conditional_query = query

            # for ip-adapter
            vd_key = self.to_k_adapter(adapter_hidden_states)
            vd_value = self.to_v_adapter(adapter_hidden_states)

            vd_key = vd_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            vd_value = vd_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            vd_hidden_states = F.scaled_dot_product_attention(
                query, vd_key, vd_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            vd_hidden_states = vd_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            vd_hidden_states = vd_hidden_states.to(query.dtype)

            # scale to block scaler
            if block_scaler is not None:
                orig_dtype = vd_hidden_states.dtype
                if block_scaler.dtype != vd_hidden_states.dtype:
                    vd_hidden_states = vd_hidden_states.to(block_scaler.dtype)
                vd_hidden_states = vd_hidden_states * block_scaler
                if block_scaler.dtype != orig_dtype:
                    vd_hidden_states = vd_hidden_states.to(orig_dtype)

            hidden_states = hidden_states + self.scale * vd_hidden_states

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

class VisionDirectAdapter(torch.nn.Module):
    def __init__(
            self,
            adapter: 'CustomAdapter',
            sd: 'StableDiffusion',
            vision_model: Union[CLIPVisionModelWithProjection],
    ):
        super(VisionDirectAdapter, self).__init__()
        is_pixart = sd.is_pixart
        is_flux = sd.is_flux
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.config: AdapterConfig = adapter.config
        self.vision_model_ref: weakref.ref = weakref.ref(vision_model)

        if adapter.config.clip_layer == "image_embeds":
            self.token_size = vision_model.config.projection_dim
        else:
            self.token_size = vision_model.config.hidden_size

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

        elif is_flux:
            transformer: FluxTransformer2DModel = sd.unet
            for i, module in transformer.transformer_blocks.named_children():
                attn_processor_keys.append(f"transformer_blocks.{i}.attn")

            # single transformer blocks do not have cross attn, but we will do them anyway
            for i, module in transformer.single_transformer_blocks.named_children():
                attn_processor_keys.append(f"single_transformer_blocks.{i}.attn")
        else:
            attn_processor_keys = list(sd.unet.attn_processors.keys())

        current_idx = 0

        for name in attn_processor_keys:
            if is_flux:
                cross_attention_dim = None
            else:
                cross_attention_dim = None if name.endswith("attn1.processor") or name.endswith("attn.1") else sd.unet.config['cross_attention_dim']
            if name.startswith("mid_block"):
                hidden_size = sd.unet.config['block_out_channels'][-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(sd.unet.config['block_out_channels']))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = sd.unet.config['block_out_channels'][block_id]
            elif name.startswith("transformer") or name.startswith("single_transformer"):
                if is_flux:
                    hidden_size = 3072
                else:
                    hidden_size = sd.unet.config['cross_attention_dim']
            else:
                # they didnt have this, but would lead to undefined below
                raise ValueError(f"unknown attn processor name: {name}")
            if cross_attention_dim is None and not is_flux:
                attn_procs[name] = AttnProcessor2_0()
            else:
                layer_name = name.split(".processor")[0]
                if f"{layer_name}.to_k.weight._data" in unet_sd and is_flux:
                    # is quantized

                    to_k_adapter = torch.randn(hidden_size, hidden_size) * 0.01
                    to_v_adapter = torch.randn(hidden_size, hidden_size) * 0.01
                    to_k_adapter = to_k_adapter.to(self.sd_ref().torch_dtype)
                    to_v_adapter = to_v_adapter.to(self.sd_ref().torch_dtype)
                else:
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
                    # if is_pixart:
                    #     to_k_bias = to_k_bias[:self.token_size]
                    #     to_v_bias = to_v_bias[:self.token_size]
                else:
                    to_k_adapter = to_k_adapter
                    to_v_adapter = to_v_adapter
                    # if is_pixart:
                    #     to_k_bias = to_k_bias
                    #     to_v_bias = to_v_bias

                weights = {
                    "to_k_adapter.weight": to_k_adapter * 0.01,
                    "to_v_adapter.weight": to_v_adapter * 0.01,
                }
                # if is_pixart:
                #     weights["to_k_adapter.bias"] = to_k_bias
                #     weights["to_v_adapter.bias"] = to_v_bias\

                if is_flux:
                    attn_procs[name] = CustomFluxVDAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        adapter=self,
                        adapter_hidden_size=self.token_size,
                        has_bias=False,
                        block_idx=current_idx
                    )
                else:
                    attn_procs[name] = VisionDirectAdapterAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        adapter=self,
                        adapter_hidden_size=self.token_size,
                        has_bias=False,
                    )
                current_idx += 1
                attn_procs[name].load_state_dict(weights)

        if self.sd_ref().is_pixart:
            # we have to set them ourselves
            transformer: Transformer2DModel = sd.unet
            for i, module in transformer.transformer_blocks.named_children():
                module.attn1.processor = attn_procs[f"transformer_blocks.{i}.attn1"]
                module.attn2.processor = attn_procs[f"transformer_blocks.{i}.attn2"]
            self.adapter_modules = torch.nn.ModuleList([
                transformer.transformer_blocks[i].attn1.processor for i in range(len(transformer.transformer_blocks))
            ] + [
                transformer.transformer_blocks[i].attn2.processor for i in range(len(transformer.transformer_blocks))
            ])
        elif self.sd_ref().is_flux:
            # we have to set them ourselves
            transformer: FluxTransformer2DModel = sd.unet
            for i, module in transformer.transformer_blocks.named_children():
                module.attn.processor = attn_procs[f"transformer_blocks.{i}.attn"]

            # do single blocks too even though they dont have cross attn
            for i, module in transformer.single_transformer_blocks.named_children():
                module.attn.processor = attn_procs[f"single_transformer_blocks.{i}.attn"]

            self.adapter_modules = torch.nn.ModuleList(
                [
                    transformer.transformer_blocks[i].attn.processor for i in
                    range(len(transformer.transformer_blocks))
                ] + [
                    transformer.single_transformer_blocks[i].attn.processor for i in
                    range(len(transformer.single_transformer_blocks))
                ]
            )
        else:
            sd.unet.set_attn_processor(attn_procs)
            self.adapter_modules = torch.nn.ModuleList(sd.unet.attn_processors.values())

        num_modules = len(self.adapter_modules)
        if self.config.train_scaler:
            self.block_scaler = torch.nn.Parameter(torch.tensor([1.0] * num_modules).to(
                dtype=torch.float32,
                device=self.sd_ref().device_torch
            ))
            self.block_scaler.data = self.block_scaler.data.to(torch.float32)
            self.block_scaler.requires_grad = True
        else:
            self.block_scaler = None

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if self.config.train_scaler:
            # only return the block scaler
            if destination is None:
                destination = OrderedDict()
            destination[prefix + 'block_scaler'] = self.block_scaler
            return destination
        return super().state_dict(destination, prefix, keep_vars)

    # make a getter to see if is active
    @property
    def is_active(self):
        return self.adapter_ref().is_active

    def forward(self, input):
        # block scaler keeps moving dtypes. make sure it is float32 here
        # todo remove this when we have a real solution
        if self.block_scaler is not None and self.block_scaler.dtype != torch.float32:
            self.block_scaler.data = self.block_scaler.data.to(torch.float32)
        return input

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.block_scaler is not None:
            if self.block_scaler.dtype != torch.float32:
                self.block_scaler.data = self.block_scaler.data.to(torch.float32)
        return self

    def post_weight_update(self):
        # force block scaler to be mean of 1
        # if self.block_scaler is not None:
        #     self.block_scaler.data = self.block_scaler.data / self.block_scaler.data.mean()
        pass
