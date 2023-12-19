import torch
import sys

from PIL import Image
from torch.nn import Parameter
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from toolkit.paths import REPOS_ROOT
from toolkit.saving import load_ip_adapter_model
from toolkit.train_tools import get_torch_dtype

sys.path.append(REPOS_ROOT)
from typing import TYPE_CHECKING, Union, Iterator, Mapping, Any, Tuple, List
from collections import OrderedDict
from ipadapter.ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor, IPAttnProcessor2_0, AttnProcessor2_0
from ipadapter.ip_adapter.ip_adapter import ImageProjModel
from ipadapter.ip_adapter.resampler import Resampler
from toolkit.config_modules import AdapterConfig
from toolkit.prompt_utils import PromptEmbeds
import weakref

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

import torch.nn.functional as F


class CustomIPAttentionProcessor(IPAttnProcessor2_0):
    def __init__(self, hidden_size, cross_attention_dim, scale=1.0, num_tokens=4, adapter=None):
        super().__init__(hidden_size, cross_attention_dim, scale=scale, num_tokens=num_tokens)
        self.adapter_ref: weakref.ref = weakref.ref(adapter)

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
        if not is_active:
            ip_hidden_states = None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
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

        # will be none if disabled
        if ip_hidden_states is not None:
            # for ip-adapter
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            hidden_states = hidden_states + self.scale * ip_hidden_states

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


# loosely based on # ref https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train.py
class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, sd: 'StableDiffusion', adapter_config: 'AdapterConfig'):
        super().__init__()
        self.config = adapter_config
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.clip_image_processor = CLIPImageProcessor()
        self.device = self.sd_ref().unet.device
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(adapter_config.image_encoder_path)
        self.current_scale = 1.0
        self.is_active = True
        if adapter_config.type == 'ip':
            # ip-adapter
            image_proj_model = ImageProjModel(
                cross_attention_dim=sd.unet.config['cross_attention_dim'],
                clip_embeddings_dim=self.image_encoder.config.projection_dim,
                clip_extra_context_tokens=4,
            )
        elif adapter_config.type == 'ip+':
            # ip-adapter-plus
            num_tokens = 16
            image_proj_model = Resampler(
                dim=sd.unet.config['cross_attention_dim'],
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim=self.image_encoder.config.hidden_size,
                output_dim=sd.unet.config['cross_attention_dim'],
                ff_mult=4
            )
        else:
            raise ValueError(f"unknown adapter type: {adapter_config.type}")

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
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                if adapter_config.type == 'ip':
                    # ip-adapter
                    num_tokens = 4
                elif adapter_config.type == 'ip+':
                    # ip-adapter-plus
                    num_tokens = 16
                else:
                    raise ValueError(f"unknown adapter type: {adapter_config.type}")

                attn_procs[name] = CustomIPAttentionProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_tokens,
                    adapter=self
                )
                attn_procs[name].load_state_dict(weights)
        sd.unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(sd.unet.attn_processors.values())

        sd.adapter = self
        self.unet_ref: weakref.ref = weakref.ref(sd.unet)
        self.image_proj_model = image_proj_model
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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.image_encoder.to(*args, **kwargs)
        self.image_proj_model.to(*args, **kwargs)
        self.adapter_modules.to(*args, **kwargs)
        return self

    def load_ip_adapter(self, state_dict: Union[OrderedDict, dict]):
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    # def load_state_dict(self, state_dict: Union[OrderedDict, dict]):
    #     self.load_ip_adapter(state_dict)

    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        state_dict["image_proj"] = self.image_proj_model.state_dict()
        state_dict["ip_adapter"] = self.adapter_modules.state_dict()
        return state_dict

    def get_scale(self):
        return self.current_scale

    def set_scale(self, scale):
        self.current_scale = scale
        for attn_processor in self.sd_ref().unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    @torch.no_grad()
    def get_clip_image_embeds_from_pil(self, pil_image: Union[Image.Image, List[Image.Image]],
                                       drop=False) -> torch.Tensor:
        # todo: add support for sdxl
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        if drop:
            clip_image = clip_image * 0
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        return clip_image_embeds

    @torch.no_grad()
    def get_clip_image_embeds_from_tensors(self, tensors_0_1: torch.Tensor, drop=False) -> torch.Tensor:
        # tensors should be 0-1
        # todo: add support for sdxl
        if tensors_0_1.ndim == 3:
            tensors_0_1 = tensors_0_1.unsqueeze(0)
        # training tensors are 0 - 1
        tensors_0_1 = tensors_0_1.to(self.device, dtype=torch.float16)
        # if images are out of this range throw error
        if tensors_0_1.min() < -0.3 or tensors_0_1.max() > 1.3:
            raise ValueError("image tensor values must be between 0 and 1. Got min: {}, max: {}".format(
                tensors_0_1.min(), tensors_0_1.max()
            ))

        clip_image = self.clip_image_processor(
            images=tensors_0_1,
            return_tensors="pt",
            do_resize=True,
            do_rescale=False,
        ).pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16).detach()
        if drop:
            clip_image = clip_image * 0
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        return clip_image_embeds

    # use drop for prompt dropout, or negatives
    def forward(self, embeddings: PromptEmbeds, clip_image_embeds: torch.Tensor) -> PromptEmbeds:
        clip_image_embeds = clip_image_embeds.detach()
        clip_image_embeds = clip_image_embeds.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        image_prompt_embeds = self.image_proj_model(clip_image_embeds.detach())
        embeddings.text_embeds = torch.cat([embeddings.text_embeds, image_prompt_embeds], dim=1)
        return embeddings

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for attn_processor in self.adapter_modules:
            yield from attn_processor.parameters(recurse)
        yield from self.image_proj_model.parameters(recurse)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=strict)
