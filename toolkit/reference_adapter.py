import math

import torch
import sys

from PIL import Image
from torch.nn import Parameter
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from toolkit.basic import adain
from toolkit.saving import load_ip_adapter_model
from toolkit.train_tools import get_torch_dtype
from typing import TYPE_CHECKING, Union, Iterator, Mapping, Any, Tuple, List, Optional, Dict
from collections import OrderedDict
from toolkit.config_modules import AdapterConfig
from toolkit.prompt_utils import PromptEmbeds
import weakref

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

from diffusers import (
    EulerDiscreteScheduler,
    DDPMScheduler,
)

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection
)
from toolkit.models.size_agnostic_feature_encoder import SAFEImageProcessor, SAFEVisionModel

from transformers import ViTHybridImageProcessor, ViTHybridForImageClassification

from transformers import ViTFeatureExtractor, ViTForImageClassification

import torch.nn.functional as F
import torch.nn as nn


class ReferenceAttnProcessor2_0(torch.nn.Module):
    r"""
        Attention processor for IP-Adapater for PyTorch 2.0.
        Args:
            hidden_size (`int`):
                The hidden size of the attention layer.
            cross_attention_dim (`int`):
                The number of channels in the `encoder_hidden_states`.
            scale (`float`, defaults to 1.0):
                the weight scale of image prompt.
            num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
                The context length of the image features.
        """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, adapter=None):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.ref_net = nn.Linear(hidden_size, hidden_size)
        self.blend = nn.Parameter(torch.zeros(hidden_size))
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self._memory = None

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

        if self.adapter_ref().is_active:
            if self.adapter_ref().reference_mode == "write":
                # write_mode
                memory_ref = self.ref_net(hidden_states)
                self._memory = memory_ref
            elif self.adapter_ref().reference_mode == "read":
                # read_mode
                if self._memory is None:
                    print("Warning: no memory to read from")
                else:

                    saved_hidden_states = self._memory
                    try:
                        new_hidden_states = saved_hidden_states
                        blend = self.blend
                        # expand the blend buyt keep dim 0 the same (batch)
                        while blend.ndim < new_hidden_states.ndim:
                            blend = blend.unsqueeze(0)
                        # expand batch
                        blend = torch.cat([blend] * new_hidden_states.shape[0], dim=0)
                        hidden_states = blend * new_hidden_states + (1 - blend) * hidden_states
                    except Exception as e:
                        raise Exception(f"Error blending: {e}")

        return hidden_states


class ReferenceAdapter(torch.nn.Module):

    def __init__(self, sd: 'StableDiffusion', adapter_config: 'AdapterConfig'):
        super().__init__()
        self.config = adapter_config
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.device = self.sd_ref().unet.device
        self.reference_mode = "read"
        self.current_scale = 1.0
        self.is_active = True
        self._reference_images = None
        self._reference_latents = None
        self.has_memory = False

        self.noise_scheduler: Union[DDPMScheduler, EulerDiscreteScheduler] = None

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
                # layer_name = name.split(".processor")[0]
                # weights = {
                #     "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                #     "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                # }

                attn_procs[name] = ReferenceAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.config.num_tokens,
                    adapter=self
                )
                # attn_procs[name].load_state_dict(weights)
        sd.unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(sd.unet.attn_processors.values())

        sd.adapter = self
        self.unet_ref: weakref.ref = weakref.ref(sd.unet)
        self.adapter_modules = adapter_modules
        # load the weights if we have some
        if self.config.name_or_path:
            loaded_state_dict = load_ip_adapter_model(
                self.config.name_or_path,
                device='cpu',
                dtype=sd.torch_dtype
            )
            self.load_state_dict(loaded_state_dict)

        self.set_scale(1.0)
        self.attach()
        self.to(self.device, self.sd_ref().torch_dtype)

        # if self.config.train_image_encoder:
        #     self.image_encoder.train()
        #     self.image_encoder.requires_grad_(True)


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # self.image_encoder.to(*args, **kwargs)
        # self.image_proj_model.to(*args, **kwargs)
        self.adapter_modules.to(*args, **kwargs)
        return self

    def load_reference_adapter(self, state_dict: Union[OrderedDict, dict]):
        reference_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        reference_layers.load_state_dict(state_dict["reference_adapter"])

    # def load_state_dict(self, state_dict: Union[OrderedDict, dict]):
    #     self.load_ip_adapter(state_dict)

    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        state_dict["reference_adapter"] = self.adapter_modules.state_dict()
        return state_dict

    def get_scale(self):
        return self.current_scale

    def set_reference_images(self, reference_images: Optional[torch.Tensor]):
        self._reference_images = reference_images.clone().detach()
        self._reference_latents = None
        self.clear_memory()

    def set_blank_reference_images(self, batch_size):
        self._reference_images = torch.zeros((batch_size, 3, 512, 512), device=self.device, dtype=self.sd_ref().torch_dtype)
        self._reference_latents = torch.zeros((batch_size, 4, 64, 64), device=self.device, dtype=self.sd_ref().torch_dtype)
        self.clear_memory()


    def set_scale(self, scale):
        self.current_scale = scale
        for attn_processor in self.sd_ref().unet.attn_processors.values():
            if isinstance(attn_processor, ReferenceAttnProcessor2_0):
                attn_processor.scale = scale


    def attach(self):
        unet = self.sd_ref().unet
        self._original_unet_forward = unet.forward
        unet.forward = lambda *args, **kwargs: self.unet_forward(*args, **kwargs)
        if self.sd_ref().network is not None:
            # set network to not merge in
            self.sd_ref().network.can_merge_in = False

    def unet_forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        skip = False
        if self._reference_images is None and self._reference_latents is None:
            skip = True
        if not self.is_active:
            skip = True

        if self.has_memory:
            skip = True

        if not skip:
            if self.sd_ref().network is not None:
                self.sd_ref().network.is_active = True
            if self.sd_ref().network.is_merged_in:
                raise ValueError("network is merged in, but we are not supposed to be merged in")
                # send it through our forward first
            self.forward(sample, timestep, encoder_hidden_states, *args, **kwargs)

        if self.sd_ref().network is not None:
            self.sd_ref().network.is_active = False

        # Send it through the original unet forward
        return self._original_unet_forward(sample, timestep, encoder_hidden_states, args, **kwargs)


    # use drop for prompt dropout, or negatives
    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        if not self.noise_scheduler:
            raise ValueError("noise scheduler not set")
        if not self.is_active or (self._reference_images is None and self._reference_latents is None):
            raise ValueError("reference adapter not active or no reference images set")
        # todo may need to handle cfg?
        self.reference_mode = "write"

        if self._reference_latents is None:
            self._reference_latents = self.sd_ref().encode_images(self._reference_images.to(
                self.device, self.sd_ref().torch_dtype
            )).detach()
        # create a sample from our reference images
        reference_latents = self._reference_latents.clone().detach().to(self.device, self.sd_ref().torch_dtype)
        # if our num of samples are half of incoming, we are doing cfg. Zero out the first half (unconditional)
        if reference_latents.shape[0] * 2 == sample.shape[0]:
            # we are doing cfg
            # Unconditional goes first
            reference_latents = torch.cat([torch.zeros_like(reference_latents), reference_latents], dim=0).detach()

        # resize it so reference_latents will fit inside sample in the center
        width_scale = sample.shape[2] / reference_latents.shape[2]
        height_scale = sample.shape[3] / reference_latents.shape[3]
        scale = min(width_scale, height_scale)
        # resize the reference latents

        mode = "bilinear" if scale > 1.0 else "bicubic"

        reference_latents = F.interpolate(
            reference_latents,
            size=(int(reference_latents.shape[2] * scale), int(reference_latents.shape[3] * scale)),
            mode=mode,
            align_corners=False
        )

        # add 0 padding if needed
        width_pad = (sample.shape[2] - reference_latents.shape[2]) / 2
        height_pad = (sample.shape[3] - reference_latents.shape[3]) / 2
        reference_latents = F.pad(
            reference_latents,
            (math.floor(width_pad), math.floor(width_pad), math.ceil(height_pad), math.ceil(height_pad)),
            mode="constant",
            value=0
        )

        # resize again just to make sure it is exact same size
        reference_latents = F.interpolate(
            reference_latents,
            size=(sample.shape[2], sample.shape[3]),
            mode="bicubic",
            align_corners=False
        )

        # todo maybe add same noise to the sample? For now we will send it through with no noise
        # sample_imgs = self.noise_scheduler.add_noise(sample_imgs, timestep)
        self._original_unet_forward(reference_latents, timestep, encoder_hidden_states, *args, **kwargs)
        self.reference_mode = "read"
        self.has_memory = True
        return None

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for attn_processor in self.adapter_modules:
            yield from attn_processor.parameters(recurse)
        # yield from self.image_proj_model.parameters(recurse)
        # if self.config.train_image_encoder:
        #     yield from self.image_encoder.parameters(recurse)
        # if self.config.train_image_encoder:
        #     yield from self.image_encoder.parameters(recurse)
        #     self.image_encoder.train()
        # else:
        #     for attn_processor in self.adapter_modules:
        #         yield from attn_processor.parameters(recurse)
        #     yield from self.image_proj_model.parameters(recurse)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        strict = False
        # self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict)
        self.adapter_modules.load_state_dict(state_dict["reference_adapter"], strict=strict)

    def enable_gradient_checkpointing(self):
        self.image_encoder.gradient_checkpointing = True

    def clear_memory(self):
        for attn_processor in self.adapter_modules:
            if isinstance(attn_processor, ReferenceAttnProcessor2_0):
                attn_processor._memory = None
        self.has_memory = False
