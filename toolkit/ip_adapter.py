import random

import torch
import sys

from diffusers import Transformer2DModel
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.module import T
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from toolkit.models.clip_pre_processor import CLIPImagePreProcessor
from toolkit.models.zipper_resampler import ZipperResampler
from toolkit.saving import load_ip_adapter_model
from toolkit.train_tools import get_torch_dtype
from toolkit.util.inverse_cfg import inverse_classifier_guidance

from typing import TYPE_CHECKING, Union, Iterator, Mapping, Any, Tuple, List, Optional
from collections import OrderedDict
from toolkit.util.ip_adapter_utils import AttnProcessor2_0, IPAttnProcessor2_0, ImageProjModel
from toolkit.resampler import Resampler
from toolkit.config_modules import AdapterConfig
from toolkit.prompt_utils import PromptEmbeds
import weakref
from diffusers import FluxTransformer2DModel

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    AutoImageProcessor,
    ConvNextV2ForImageClassification,
    ConvNextForImageClassification,
    ConvNextImageProcessor
)
from toolkit.models.size_agnostic_feature_encoder import SAFEImageProcessor, SAFEVisionModel

from transformers import ViTHybridImageProcessor, ViTHybridForImageClassification

from transformers import ViTFeatureExtractor, ViTForImageClassification

import torch.nn.functional as F


class MLPProjModelClipFace(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.norm = torch.nn.LayerNorm(id_embeddings_dim)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        # Initialize the last linear layer weights near zero
        torch.nn.init.uniform_(self.proj[2].weight, a=-0.01, b=0.01)
        torch.nn.init.zeros_(self.proj[2].bias)
        # # Custom initialization for LayerNorm to output near zero
        # torch.nn.init.constant_(self.norm.weight, 0.1)  # Small weights near zero
        # torch.nn.init.zeros_(self.norm.bias)  # Bias to zero

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return x


class CustomIPAttentionProcessor(IPAttnProcessor2_0):
    def __init__(self, hidden_size, cross_attention_dim, scale=1.0, num_tokens=4, adapter=None, train_scaler=False, full_token_scaler=False):
        super().__init__(hidden_size, cross_attention_dim, scale=scale, num_tokens=num_tokens)
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.train_scaler = train_scaler
        if train_scaler:
            if full_token_scaler:
                self.ip_scaler = torch.nn.Parameter(torch.ones([num_tokens], dtype=torch.float32) * 0.999)
            else:
                self.ip_scaler = torch.nn.Parameter(torch.ones([1], dtype=torch.float32) * 0.999)
            # self.ip_scaler = torch.nn.Parameter(torch.ones([1], dtype=torch.float32) * 0.9999)
            self.ip_scaler.requires_grad_(True)

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

        if is_active:
            # since we are removing tokens, we need to adjust the sequence length
            sequence_length = sequence_length - self.num_tokens

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
        try:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        except Exception as e:
            print(e)
            raise e

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # will be none if disabled
        if ip_hidden_states is not None:
            # apply scaler
            if self.train_scaler:
                weight = self.ip_scaler
                # reshape to (1, self.num_tokens, 1)
                weight = weight.view(1, -1, 1)
                ip_hidden_states = ip_hidden_states * weight

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

            scale = self.scale
            hidden_states = hidden_states + scale * ip_hidden_states

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

    # this ensures that the ip_scaler is not changed when we load the model
    # def _apply(self, fn):
    #     if hasattr(self, "ip_scaler"):
    #         # Overriding the _apply method to prevent the special_parameter from changing dtype
    #         self.ip_scaler = fn(self.ip_scaler)
    #         # Temporarily set the special_parameter to None to exclude it from default _apply processing
    #         ip_scaler = self.ip_scaler
    #         self.ip_scaler = None
    #         super(CustomIPAttentionProcessor, self)._apply(fn)
    #         # Restore the special_parameter after the default _apply processing
    #         self.ip_scaler = ip_scaler
    #         return self
    #     else:
    #         return super(CustomIPAttentionProcessor, self)._apply(fn)


class CustomIPFluxAttnProcessor2_0(torch.nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, hidden_size, cross_attention_dim, scale=1.0, num_tokens=4, adapter=None, train_scaler=False,
                 full_token_scaler=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.train_scaler = train_scaler
        self.num_tokens = num_tokens
        if train_scaler:
            if full_token_scaler:
                self.ip_scaler = torch.nn.Parameter(torch.ones([num_tokens], dtype=torch.float32) * 0.999)
            else:
                self.ip_scaler = torch.nn.Parameter(torch.ones([1], dtype=torch.float32) * 0.999)
            # self.ip_scaler = torch.nn.Parameter(torch.ones([1], dtype=torch.float32) * 0.9999)
            self.ip_scaler.requires_grad_(True)

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        is_active = self.adapter_ref().is_active
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
        if not is_active:
            ip_hidden_states = None
        else:
            # get ip hidden states. Should be stored
            ip_hidden_states = self.adapter_ref().last_conditional
            # add unconditional to front if it exists
            if ip_hidden_states.shape[0] * 2 == batch_size:
                if self.adapter_ref().last_unconditional is None:
                    raise ValueError("Unconditional is None but should not be")
                ip_hidden_states = torch.cat([self.adapter_ref().last_unconditional, ip_hidden_states], dim=0)

        if ip_hidden_states is not None:
            # apply scaler
            if self.train_scaler:
                weight = self.ip_scaler
                # reshape to (1, self.num_tokens, 1)
                weight = weight.view(1, -1, 1)
                ip_hidden_states = ip_hidden_states * weight

            # for ip-adapter
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            scale = self.scale
            hidden_states = hidden_states + scale * ip_hidden_states
        # end ip adapter

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

# loosely based on # ref https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train.py
class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, sd: 'StableDiffusion', adapter_config: 'AdapterConfig'):
        super().__init__()
        self.config = adapter_config
        self.sd_ref: weakref.ref = weakref.ref(sd)
        self.device = self.sd_ref().unet.device
        self.preprocessor: Optional[CLIPImagePreProcessor] = None
        self.input_size = 224
        self.clip_noise_zero = True
        self.unconditional: torch.Tensor = None

        self.last_conditional: torch.Tensor = None
        self.last_unconditional: torch.Tensor = None

        self.additional_loss = None
        if self.config.image_encoder_arch.startswith("clip"):
            try:
                self.clip_image_processor = CLIPImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                self.clip_image_processor = CLIPImageProcessor()
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                adapter_config.image_encoder_path,
                ignore_mismatched_sizes=True).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'siglip':
            from transformers import SiglipImageProcessor, SiglipVisionModel
            try:
                self.clip_image_processor = SiglipImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                self.clip_image_processor = SiglipImageProcessor()
            self.image_encoder = SiglipVisionModel.from_pretrained(
                adapter_config.image_encoder_path,
                ignore_mismatched_sizes=True).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'vit':
            try:
                self.clip_image_processor = ViTFeatureExtractor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                self.clip_image_processor = ViTFeatureExtractor()
            self.image_encoder = ViTForImageClassification.from_pretrained(adapter_config.image_encoder_path).to(
                self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'safe':
            try:
                self.clip_image_processor = SAFEImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                self.clip_image_processor = SAFEImageProcessor()
            self.image_encoder = SAFEVisionModel(
                in_channels=3,
                num_tokens=self.config.safe_tokens,
                num_vectors=sd.unet.config['cross_attention_dim'],
                reducer_channels=self.config.safe_reducer_channels,
                channels=self.config.safe_channels,
                downscale_factor=8
            ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'convnext':
            try:
                self.clip_image_processor = ConvNextImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                print(f"could not load image processor from {adapter_config.image_encoder_path}")
                self.clip_image_processor = ConvNextImageProcessor(
                    size=320,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711],
                )
            self.image_encoder = ConvNextForImageClassification.from_pretrained(
                adapter_config.image_encoder_path,
                use_safetensors=True,
            ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'convnextv2':
            try:
                self.clip_image_processor = AutoImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                print(f"could not load image processor from {adapter_config.image_encoder_path}")
                self.clip_image_processor = ConvNextImageProcessor(
                    size=512,
                    image_mean=[0.485, 0.456, 0.406],
                    image_std=[0.229, 0.224, 0.225],
                )
            self.image_encoder = ConvNextV2ForImageClassification.from_pretrained(
                adapter_config.image_encoder_path,
                use_safetensors=True,
            ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        elif self.config.image_encoder_arch == 'vit-hybrid':
            try:
                self.clip_image_processor = ViTHybridImageProcessor.from_pretrained(adapter_config.image_encoder_path)
            except EnvironmentError:
                print(f"could not load image processor from {adapter_config.image_encoder_path}")
                self.clip_image_processor = ViTHybridImageProcessor(
                    size=320,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711],
                )
            self.image_encoder = ViTHybridForImageClassification.from_pretrained(
                adapter_config.image_encoder_path,
                use_safetensors=True,
            ).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        else:
            raise ValueError(f"unknown image encoder arch: {adapter_config.image_encoder_arch}")

        if not self.config.train_image_encoder:
            # compile it
            print('Compiling image encoder')
            #torch.compile(self.image_encoder, fullgraph=True)

        self.input_size = self.image_encoder.config.image_size

        if self.config.quad_image:  # 4x4 image
            # self.clip_image_processor.config
            # We do a 3x downscale of the image, so we need to adjust the input size
            preprocessor_input_size = self.image_encoder.config.image_size * 2

            # update the preprocessor so images come in at the right size
            if 'height' in self.clip_image_processor.size:
                self.clip_image_processor.size['height'] = preprocessor_input_size
                self.clip_image_processor.size['width'] = preprocessor_input_size
            elif hasattr(self.clip_image_processor, 'crop_size'):
                self.clip_image_processor.size['shortest_edge'] = preprocessor_input_size
                self.clip_image_processor.crop_size['height'] = preprocessor_input_size
                self.clip_image_processor.crop_size['width'] = preprocessor_input_size

        if self.config.image_encoder_arch == 'clip+':
            # self.clip_image_processor.config
            # We do a 3x downscale of the image, so we need to adjust the input size
            preprocessor_input_size = self.image_encoder.config.image_size * 4

            # update the preprocessor so images come in at the right size
            self.clip_image_processor.size['shortest_edge'] = preprocessor_input_size
            self.clip_image_processor.crop_size['height'] = preprocessor_input_size
            self.clip_image_processor.crop_size['width'] = preprocessor_input_size

            self.preprocessor = CLIPImagePreProcessor(
                input_size=preprocessor_input_size,
                clip_input_size=self.image_encoder.config.image_size,
            )
        if not self.config.image_encoder_arch == 'safe':
            if 'height' in self.clip_image_processor.size:
                self.input_size = self.clip_image_processor.size['height']
            elif hasattr(self.clip_image_processor, 'crop_size'):
                self.input_size = self.clip_image_processor.crop_size['height']
            elif 'shortest_edge' in self.clip_image_processor.size.keys():
                self.input_size = self.clip_image_processor.size['shortest_edge']
            else:
                raise ValueError(f"unknown image processor size: {self.clip_image_processor.size}")
        self.current_scale = 1.0
        self.is_active = True
        is_pixart = sd.is_pixart
        is_flux = sd.is_flux
        if adapter_config.type == 'ip':
            # ip-adapter
            image_proj_model = ImageProjModel(
                cross_attention_dim=sd.unet.config['cross_attention_dim'],
                clip_embeddings_dim=self.image_encoder.config.projection_dim,
                clip_extra_context_tokens=self.config.num_tokens,  # usually 4
            )
        elif adapter_config.type == 'ip_clip_face':
            cross_attn_dim = 4096 if is_pixart else sd.unet.config['cross_attention_dim']
            image_proj_model = MLPProjModelClipFace(
                cross_attention_dim=cross_attn_dim,
                id_embeddings_dim=self.image_encoder.config.projection_dim,
                num_tokens=self.config.num_tokens,  # usually 4
            )
        elif adapter_config.type == 'ip+':
            heads = 12 if not sd.is_xl else 20
            if is_flux:
                dim = 1280
            else:
                dim = sd.unet.config['cross_attention_dim'] if not sd.is_xl else 1280
            embedding_dim = self.image_encoder.config.hidden_size if not self.config.image_encoder_arch.startswith(
                'convnext') else \
                self.image_encoder.config.hidden_sizes[-1]

            image_encoder_state_dict = self.image_encoder.state_dict()
            # max_seq_len = CLIP tokens + CLS token
            max_seq_len = 257
            if "vision_model.embeddings.position_embedding.weight" in image_encoder_state_dict:
                # clip
                max_seq_len = int(
                    image_encoder_state_dict["vision_model.embeddings.position_embedding.weight"].shape[0])

            if is_pixart:
                heads = 20
                dim = 1280
                output_dim = 4096
            elif is_flux:
                heads = 20
                dim = 1280
                output_dim = 3072
            else:
                output_dim = sd.unet.config['cross_attention_dim']

            if self.config.image_encoder_arch.startswith('convnext'):
                in_tokens = 16 * 16
                embedding_dim = self.image_encoder.config.hidden_sizes[-1]

            # ip-adapter-plus
            image_proj_model = Resampler(
                dim=dim,
                depth=4,
                dim_head=64,
                heads=heads,
                num_queries=self.config.num_tokens if self.config.num_tokens > 0 else max_seq_len,
                embedding_dim=embedding_dim,
                max_seq_len=max_seq_len,
                output_dim=output_dim,
                ff_mult=4
            )
        elif adapter_config.type == 'ipz':
            dim = sd.unet.config['cross_attention_dim']
            if hasattr(self.image_encoder.config, 'hidden_sizes'):
                embedding_dim = self.image_encoder.config.hidden_sizes[-1]
            else:
                embedding_dim = self.image_encoder.config.target_hidden_size

            image_encoder_state_dict = self.image_encoder.state_dict()
            # max_seq_len = CLIP tokens + CLS token
            in_tokens = 257
            if "vision_model.embeddings.position_embedding.weight" in image_encoder_state_dict:
                # clip
                in_tokens = int(image_encoder_state_dict["vision_model.embeddings.position_embedding.weight"].shape[0])

            if self.config.image_encoder_arch.startswith('convnext'):
                in_tokens = 16 * 16
                embedding_dim = self.image_encoder.config.hidden_sizes[-1]

            is_conv_next = self.config.image_encoder_arch.startswith('convnext')

            out_tokens = self.config.num_tokens if self.config.num_tokens > 0 else in_tokens
            # ip-adapter-plus
            image_proj_model = ZipperResampler(
                in_size=embedding_dim,
                in_tokens=in_tokens,
                out_size=dim,
                out_tokens=out_tokens,
                hidden_size=embedding_dim,
                hidden_tokens=in_tokens,
                # num_blocks=1 if not is_conv_next else 2,
                num_blocks=1 if not is_conv_next else 2,
                is_conv_input=is_conv_next
            )
        elif adapter_config.type == 'ilora':
            # we apply the clip encodings to the LoRA
            image_proj_model = None
        else:
            raise ValueError(f"unknown adapter type: {adapter_config.type}")

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

        attn_processor_names = []

        blocks = []
        transformer_blocks = []
        for name in attn_processor_keys:
            name_split = name.split(".")
            block_name = f"{name_split[0]}.{name_split[1]}"
            transformer_idx = name_split.index("transformer_blocks") if "transformer_blocks" in name_split else -1
            if transformer_idx >= 0:
                transformer_name = ".".join(name_split[:2])
                transformer_name += "." + ".".join(name_split[transformer_idx:transformer_idx + 2])
                if transformer_name not in transformer_blocks:
                    transformer_blocks.append(transformer_name)


            if block_name not in blocks:
                blocks.append(block_name)
            if is_flux:
                cross_attention_dim = None
            else:
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

                # if quantized, we need to scale the weights
                if f"{layer_name}.to_k.weight._data" in unet_sd and is_flux:
                    # is quantized

                    k_weight = torch.randn(hidden_size, hidden_size) * 0.01
                    v_weight = torch.randn(hidden_size, hidden_size) * 0.01
                    k_weight = k_weight.to(self.sd_ref().torch_dtype)
                    v_weight = v_weight.to(self.sd_ref().torch_dtype)
                else:
                    k_weight = unet_sd[layer_name + ".to_k.weight"]
                    v_weight = unet_sd[layer_name + ".to_v.weight"]

                weights = {
                    "to_k_ip.weight": k_weight,
                    "to_v_ip.weight": v_weight
                }

                if is_flux:
                    attn_procs[name] = CustomIPFluxAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.config.num_tokens,
                        adapter=self,
                        train_scaler=self.config.train_scaler or self.config.merge_scaler,
                        full_token_scaler=False
                    )
                else:
                    attn_procs[name] = CustomIPAttentionProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.config.num_tokens,
                        adapter=self,
                        train_scaler=self.config.train_scaler or self.config.merge_scaler,
                        # full_token_scaler=self.config.train_scaler # full token cannot be merged in, only use if training an actual scaler
                        full_token_scaler=False
                    )
                if self.sd_ref().is_pixart or self.sd_ref().is_flux:
                    # pixart is much more sensitive
                    weights = {
                        "to_k_ip.weight": weights["to_k_ip.weight"] * 0.01,
                        "to_v_ip.weight": weights["to_v_ip.weight"] * 0.01,
                    }

                attn_procs[name].load_state_dict(weights, strict=False)
                attn_processor_names.append(name)
        print(f"Attn Processors")
        print(attn_processor_names)
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

        sd.adapter = self
        self.unet_ref: weakref.ref = weakref.ref(sd.unet)
        self.image_proj_model = image_proj_model
        # load the weights if we have some
        if self.config.name_or_path:
            loaded_state_dict = load_ip_adapter_model(
                self.config.name_or_path,
                device='cpu',
                dtype=sd.torch_dtype
            )
            self.load_state_dict(loaded_state_dict)

        self.set_scale(1.0)

        if self.config.train_image_encoder:
            self.image_encoder.train()
            self.image_encoder.requires_grad_(True)

        # premake a unconditional
        zerod = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device, dtype=torch.float16)
        self.unconditional = self.clip_image_processor(
            images=zerod,
            return_tensors="pt",
            do_resize=True,
            do_rescale=False,
        ).pixel_values

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.image_encoder.to(*args, **kwargs)
        self.image_proj_model.to(*args, **kwargs)
        self.adapter_modules.to(*args, **kwargs)
        if self.preprocessor is not None:
            self.preprocessor.to(*args, **kwargs)
        return self

    # def load_ip_adapter(self, state_dict: Union[OrderedDict, dict]):
    #     self.image_proj_model.load_state_dict(state_dict["image_proj"])
    #     ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
    #     ip_layers.load_state_dict(state_dict["ip_adapter"])
    #     if self.config.train_image_encoder and 'image_encoder' in state_dict:
    #         self.image_encoder.load_state_dict(state_dict["image_encoder"])
    #     if self.preprocessor is not None and 'preprocessor' in state_dict:
    #         self.preprocessor.load_state_dict(state_dict["preprocessor"])

    # def load_state_dict(self, state_dict: Union[OrderedDict, dict]):
    #     self.load_ip_adapter(state_dict)

    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        if self.config.train_only_image_encoder:
            return self.image_encoder.state_dict()
        if self.config.train_scaler:
            state_dict["ip_scale"] = self.adapter_modules.state_dict()
            # remove items that are not scalers
            for key in list(state_dict["ip_scale"].keys()):
                if not key.endswith("ip_scaler"):
                    del state_dict["ip_scale"][key]
            return state_dict

        state_dict["image_proj"] = self.image_proj_model.state_dict()
        state_dict["ip_adapter"] = self.adapter_modules.state_dict()
        # handle merge scaler training
        if self.config.merge_scaler:
            for key in list(state_dict["ip_adapter"].keys()):
                if key.endswith("ip_scaler"):
                    # merge in the scaler so we dont have to save it and it will be compatible with other ip adapters
                    scale = state_dict["ip_adapter"][key].clone()

                    key_start = key.split(".")[-2]
                    # reshape to (1, 1)
                    scale = scale.view(1, 1)
                    del state_dict["ip_adapter"][key]
                    # find the to_k_ip and to_v_ip keys
                    for key2 in list(state_dict["ip_adapter"].keys()):
                        if key2.endswith(f"{key_start}.to_k_ip.weight"):
                            state_dict["ip_adapter"][key2] = state_dict["ip_adapter"][key2].clone() * scale
                        if key2.endswith(f"{key_start}.to_v_ip.weight"):
                            state_dict["ip_adapter"][key2] = state_dict["ip_adapter"][key2].clone() * scale

        if self.config.train_image_encoder:
            state_dict["image_encoder"] = self.image_encoder.state_dict()
        if self.preprocessor is not None:
            state_dict["preprocessor"] = self.preprocessor.state_dict()
        return state_dict

    def get_scale(self):
        return self.current_scale

    def set_scale(self, scale):
        self.current_scale = scale
        if not self.sd_ref().is_pixart and not self.sd_ref().is_flux:
            for attn_processor in self.sd_ref().unet.attn_processors.values():
                if isinstance(attn_processor, CustomIPAttentionProcessor):
                    attn_processor.scale = scale

    # @torch.no_grad()
    # def get_clip_image_embeds_from_pil(self, pil_image: Union[Image.Image, List[Image.Image]],
    #                                    drop=False) -> torch.Tensor:
    #     # todo: add support for sdxl
    #     if isinstance(pil_image, Image.Image):
    #         pil_image = [pil_image]
    #     clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
    #     clip_image = clip_image.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
    #     if drop:
    #         clip_image = clip_image * 0
    #     clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
    #     return clip_image_embeds

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.image_encoder.to(*args, **kwargs)
        self.image_proj_model.to(*args, **kwargs)
        self.adapter_modules.to(*args, **kwargs)
        if self.preprocessor is not None:
            self.preprocessor.to(*args, **kwargs)
        return self

    def parse_clip_image_embeds_from_cache(
            self,
            image_embeds_list: List[dict],  # has ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
            quad_count=4,
    ):
        with torch.no_grad():
            device = self.sd_ref().unet.device
            clip_image_embeds = torch.cat([x[self.config.clip_layer] for x in image_embeds_list], dim=0)

            if self.config.quad_image:
                # get the outputs of the quat
                chunks = clip_image_embeds.chunk(quad_count, dim=0)
                chunk_sum = torch.zeros_like(chunks[0])
                for chunk in chunks:
                    chunk_sum = chunk_sum + chunk
                # get the mean of them

                clip_image_embeds = chunk_sum / quad_count

            clip_image_embeds = clip_image_embeds.to(device, dtype=get_torch_dtype(self.sd_ref().dtype)).detach()
        return clip_image_embeds

    def get_empty_clip_image(self, batch_size: int) -> torch.Tensor:
        with torch.no_grad():
            tensors_0_1 = torch.rand([batch_size, 3, self.input_size, self.input_size], device=self.device)
            noise_scale = torch.rand([tensors_0_1.shape[0], 1, 1, 1], device=self.device,
                                     dtype=get_torch_dtype(self.sd_ref().dtype))
            tensors_0_1 = tensors_0_1 * noise_scale
            # tensors_0_1 = tensors_0_1 * 0
            mean = torch.tensor(self.clip_image_processor.image_mean).to(
                self.device, dtype=get_torch_dtype(self.sd_ref().dtype)
            ).detach()
            std = torch.tensor(self.clip_image_processor.image_std).to(
                self.device, dtype=get_torch_dtype(self.sd_ref().dtype)
            ).detach()
            tensors_0_1 = torch.clip((255. * tensors_0_1), 0, 255).round() / 255.0
            clip_image = (tensors_0_1 - mean.view([1, 3, 1, 1])) / std.view([1, 3, 1, 1])
        return clip_image.detach()

    def get_clip_image_embeds_from_tensors(
            self,
            tensors_0_1: torch.Tensor,
            drop=False,
            is_training=False,
            has_been_preprocessed=False,
            quad_count=4,
            cfg_embed_strength=None, # perform CFG on embeds with unconditional as negative
    ) -> torch.Tensor:
        if self.sd_ref().unet.device != self.device:
            self.to(self.sd_ref().unet.device)
        if self.sd_ref().unet.device != self.image_encoder.device:
            self.to(self.sd_ref().unet.device)
        if not self.config.train:
            is_training = False
        uncond_clip = None
        with torch.no_grad():
            # on training the clip image is created in the dataloader
            if not has_been_preprocessed:
                # tensors should be 0-1
                if tensors_0_1.ndim == 3:
                    tensors_0_1 = tensors_0_1.unsqueeze(0)
                # training tensors are 0 - 1
                tensors_0_1 = tensors_0_1.to(self.device, dtype=torch.float16)

                # if images are out of this range throw error
                if tensors_0_1.min() < -0.3 or tensors_0_1.max() > 1.3:
                    raise ValueError("image tensor values must be between 0 and 1. Got min: {}, max: {}".format(
                        tensors_0_1.min(), tensors_0_1.max()
                    ))
                # unconditional
                if drop:
                    if self.clip_noise_zero:
                        tensors_0_1 = torch.rand_like(tensors_0_1).detach()
                        noise_scale = torch.rand([tensors_0_1.shape[0], 1, 1, 1], device=self.device,
                                                 dtype=get_torch_dtype(self.sd_ref().dtype))
                        tensors_0_1 = tensors_0_1 * noise_scale
                    else:
                        tensors_0_1 = torch.zeros_like(tensors_0_1).detach()
                    # tensors_0_1 = tensors_0_1 * 0
                clip_image = self.clip_image_processor(
                    images=tensors_0_1,
                    return_tensors="pt",
                    do_resize=True,
                    do_rescale=False,
                ).pixel_values
            else:
                if drop:
                    # scale the noise down
                    if self.clip_noise_zero:
                        tensors_0_1 = torch.rand_like(tensors_0_1).detach()
                        noise_scale = torch.rand([tensors_0_1.shape[0], 1, 1, 1], device=self.device,
                                                 dtype=get_torch_dtype(self.sd_ref().dtype))
                        tensors_0_1 = tensors_0_1 * noise_scale
                    else:
                        tensors_0_1 = torch.zeros_like(tensors_0_1).detach()
                    # tensors_0_1 = tensors_0_1 * 0
                    mean = torch.tensor(self.clip_image_processor.image_mean).to(
                        self.device, dtype=get_torch_dtype(self.sd_ref().dtype)
                    ).detach()
                    std = torch.tensor(self.clip_image_processor.image_std).to(
                        self.device, dtype=get_torch_dtype(self.sd_ref().dtype)
                    ).detach()
                    tensors_0_1 = torch.clip((255. * tensors_0_1), 0, 255).round() / 255.0
                    clip_image = (tensors_0_1 - mean.view([1, 3, 1, 1])) / std.view([1, 3, 1, 1])

                else:
                    clip_image = tensors_0_1
            clip_image = clip_image.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype)).detach()

            if self.config.quad_image:
                # split the 4x4 grid and stack on batch
                ci1, ci2 = clip_image.chunk(2, dim=2)
                ci1, ci3 = ci1.chunk(2, dim=3)
                ci2, ci4 = ci2.chunk(2, dim=3)
                to_cat = []
                for i, ci in enumerate([ci1, ci2, ci3, ci4]):
                    if i < quad_count:
                        to_cat.append(ci)
                    else:
                        break

                clip_image = torch.cat(to_cat, dim=0).detach()

            # if drop:
            #     clip_image = clip_image * 0
        with torch.set_grad_enabled(is_training):
            if is_training and self.config.train_image_encoder:
                self.image_encoder.train()
                clip_image = clip_image.requires_grad_(True)
                if self.preprocessor is not None:
                    clip_image = self.preprocessor(clip_image)
                clip_output = self.image_encoder(
                    clip_image,
                    output_hidden_states=True
                )
            else:
                self.image_encoder.eval()
                if self.preprocessor is not None:
                    clip_image = self.preprocessor(clip_image)
                clip_output = self.image_encoder(
                    clip_image, output_hidden_states=True
                )

            if self.config.clip_layer == 'penultimate_hidden_states':
                # they skip last layer for ip+
                # https://github.com/tencent-ailab/IP-Adapter/blob/f4b6742db35ea6d81c7b829a55b0a312c7f5a677/tutorial_train_plus.py#L403C26-L403C26
                clip_image_embeds = clip_output.hidden_states[-2]
            elif self.config.clip_layer == 'last_hidden_state':
                clip_image_embeds = clip_output.hidden_states[-1]
            else:
                clip_image_embeds = clip_output.image_embeds

                if self.config.adapter_type == "clip_face":
                    l2_norm = torch.norm(clip_image_embeds, p=2)
                    clip_image_embeds = clip_image_embeds / l2_norm

            if self.config.image_encoder_arch.startswith('convnext'):
                # flatten the width height layers to make the token space
                clip_image_embeds = clip_image_embeds.view(clip_image_embeds.size(0), clip_image_embeds.size(1), -1)
                # rearrange to (batch, tokens, size)
                clip_image_embeds = clip_image_embeds.permute(0, 2, 1)

            # apply unconditional if doing cfg on embeds
            with torch.no_grad():
                if cfg_embed_strength is not None:
                    uncond_clip = self.get_empty_clip_image(tensors_0_1.shape[0]).to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
                    if self.config.quad_image:
                        # split the 4x4 grid and stack on batch
                        ci1, ci2 = uncond_clip.chunk(2, dim=2)
                        ci1, ci3 = ci1.chunk(2, dim=3)
                        ci2, ci4 = ci2.chunk(2, dim=3)
                        to_cat = []
                        for i, ci in enumerate([ci1, ci2, ci3, ci4]):
                            if i < quad_count:
                                to_cat.append(ci)
                            else:
                                break

                        uncond_clip = torch.cat(to_cat, dim=0).detach()
                    uncond_clip_output = self.image_encoder(
                        uncond_clip, output_hidden_states=True
                    )

                    if self.config.clip_layer == 'penultimate_hidden_states':
                        uncond_clip_output_embeds = uncond_clip_output.hidden_states[-2]
                    elif self.config.clip_layer == 'last_hidden_state':
                        uncond_clip_output_embeds = uncond_clip_output.hidden_states[-1]
                    else:
                        uncond_clip_output_embeds = uncond_clip_output.image_embeds
                        if self.config.adapter_type == "clip_face":
                            l2_norm = torch.norm(uncond_clip_output_embeds, p=2)
                            uncond_clip_output_embeds = uncond_clip_output_embeds / l2_norm

                    uncond_clip_output_embeds = uncond_clip_output_embeds.detach()


                    # apply inverse cfg
                    clip_image_embeds = inverse_classifier_guidance(
                        clip_image_embeds,
                        uncond_clip_output_embeds,
                        cfg_embed_strength
                    )


            if self.config.quad_image:
                # get the outputs of the quat
                chunks = clip_image_embeds.chunk(quad_count, dim=0)
                if self.config.train_image_encoder and is_training:
                    # perform a loss across all chunks this will teach the vision encoder to
                    # identify similarities in our pairs of images and ignore things that do not make them similar
                    num_losses = 0
                    total_loss = None
                    for chunk in chunks:
                        for chunk2 in chunks:
                            if chunk is not chunk2:
                                loss = F.mse_loss(chunk, chunk2)
                                if total_loss is None:
                                    total_loss = loss
                                else:
                                    total_loss = total_loss + loss
                                num_losses += 1
                    if total_loss is not None:
                        total_loss = total_loss / num_losses
                        total_loss = total_loss * 1e-2
                        if self.additional_loss is not None:
                            total_loss = total_loss + self.additional_loss
                        self.additional_loss = total_loss

                chunk_sum = torch.zeros_like(chunks[0])
                for chunk in chunks:
                    chunk_sum = chunk_sum + chunk
                # get the mean of them

                clip_image_embeds = chunk_sum / quad_count

        if not is_training or not self.config.train_image_encoder:
            clip_image_embeds = clip_image_embeds.detach()

        return clip_image_embeds

    # use drop for prompt dropout, or negatives
    def forward(self, embeddings: PromptEmbeds, clip_image_embeds: torch.Tensor, is_unconditional=False) -> PromptEmbeds:
        clip_image_embeds = clip_image_embeds.to(self.device, dtype=get_torch_dtype(self.sd_ref().dtype))
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        if self.sd_ref().is_flux:
            # do not attach to text embeds for flux, we will save and grab them as it messes
            # with the RoPE to have them in the same tensor
            if is_unconditional:
                self.last_unconditional = image_prompt_embeds
            else:
                self.last_conditional = image_prompt_embeds
        else:
            embeddings.text_embeds = torch.cat([embeddings.text_embeds, image_prompt_embeds], dim=1)
        return embeddings

    def train(self: T, mode: bool = True) -> T:
        if self.config.train_image_encoder:
            self.image_encoder.train(mode)
        if not self.config.train_only_image_encoder:
            for attn_processor in self.adapter_modules:
                attn_processor.train(mode)
        if self.image_proj_model is not None:
            self.image_proj_model.train(mode)
        return super().train(mode)

    def get_parameter_groups(self, adapter_lr):
        param_groups = []
        # when training just scaler, we do not train anything else
        if not self.config.train_scaler:
            param_groups.append({
                "params": list(self.get_non_scaler_parameters()),
                "lr": adapter_lr,
            })
        if self.config.train_scaler or self.config.merge_scaler:
            scaler_lr = adapter_lr if self.config.scaler_lr is None else self.config.scaler_lr
            param_groups.append({
                "params": list(self.get_scaler_parameters()),
                "lr": scaler_lr,
            })
        return param_groups

    def get_scaler_parameters(self):
        # only get the scalera from the adapter modules
        for attn_processor in self.adapter_modules:
            # only get the scaler
            # check if it has ip_scaler attribute
            if hasattr(attn_processor, "ip_scaler"):
                scaler_param = attn_processor.ip_scaler
                yield scaler_param

    def get_non_scaler_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.config.train_only_image_encoder:
            if self.config.train_only_image_encoder_positional_embedding:
                yield from self.image_encoder.vision_model.embeddings.position_embedding.parameters(recurse)
            else:
                yield from self.image_encoder.parameters(recurse)
            return
        if self.config.train_scaler:
            # no params
            return

        for attn_processor in self.adapter_modules:
            if self.config.train_scaler or self.config.merge_scaler:
                # todo remove scaler
                if hasattr(attn_processor, "to_k_ip"):
                    # yield the linear layer
                    yield from attn_processor.to_k_ip.parameters(recurse)
                if hasattr(attn_processor, "to_v_ip"):
                    # yield the linear layer
                    yield from attn_processor.to_v_ip.parameters(recurse)
            else:
                yield from attn_processor.parameters(recurse)
        yield from self.image_proj_model.parameters(recurse)
        if self.config.train_image_encoder:
            yield from self.image_encoder.parameters(recurse)
        if self.preprocessor is not None:
            yield from self.preprocessor.parameters(recurse)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from self.get_non_scaler_parameters(recurse)
        if self.config.train_scaler or self.config.merge_scaler:
            yield from self.get_scaler_parameters()

    def merge_in_weights(self, state_dict: Mapping[str, Any]):
        # merge in img_proj weights
        current_img_proj_state_dict = self.image_proj_model.state_dict()
        for key, value in state_dict["image_proj"].items():
            if key in current_img_proj_state_dict:
                current_shape = current_img_proj_state_dict[key].shape
                new_shape = value.shape
                if current_shape != new_shape:
                    try:
                        # merge in what we can and leave the other values as they are
                        if len(current_shape) == 1:
                            current_img_proj_state_dict[key][:new_shape[0]] = value
                        elif len(current_shape) == 2:
                            current_img_proj_state_dict[key][:new_shape[0], :new_shape[1]] = value
                        elif len(current_shape) == 3:
                            current_img_proj_state_dict[key][:new_shape[0], :new_shape[1], :new_shape[2]] = value
                        elif len(current_shape) == 4:
                            current_img_proj_state_dict[key][:new_shape[0], :new_shape[1], :new_shape[2],
                            :new_shape[3]] = value
                        else:
                            raise ValueError(f"unknown shape: {current_shape}")
                    except RuntimeError as e:
                        print(e)
                        print(
                            f"could not merge in {key}: {list(current_shape)} <<< {list(new_shape)}. Trying other way")

                        if len(current_shape) == 1:
                            current_img_proj_state_dict[key][:current_shape[0]] = value[:current_shape[0]]
                        elif len(current_shape) == 2:
                            current_img_proj_state_dict[key][:current_shape[0], :current_shape[1]] = value[
                                                                                                     :current_shape[0],
                                                                                                     :current_shape[1]]
                        elif len(current_shape) == 3:
                            current_img_proj_state_dict[key][:current_shape[0], :current_shape[1],
                            :current_shape[2]] = value[:current_shape[0], :current_shape[1], :current_shape[2]]
                        elif len(current_shape) == 4:
                            current_img_proj_state_dict[key][:current_shape[0], :current_shape[1], :current_shape[2],
                            :current_shape[3]] = value[:current_shape[0], :current_shape[1], :current_shape[2],
                                                 :current_shape[3]]
                        else:
                            raise ValueError(f"unknown shape: {current_shape}")
                        print(f"Force merged in {key}: {list(current_shape)} <<< {list(new_shape)}")
                else:
                    current_img_proj_state_dict[key] = value
        self.image_proj_model.load_state_dict(current_img_proj_state_dict)

        # merge in ip adapter weights
        current_ip_adapter_state_dict = self.adapter_modules.state_dict()
        for key, value in state_dict["ip_adapter"].items():
            if key in current_ip_adapter_state_dict:
                current_shape = current_ip_adapter_state_dict[key].shape
                new_shape = value.shape
                if current_shape != new_shape:
                    try:
                        # merge in what we can and leave the other values as they are
                        if len(current_shape) == 1:
                            current_ip_adapter_state_dict[key][:new_shape[0]] = value
                        elif len(current_shape) == 2:
                            current_ip_adapter_state_dict[key][:new_shape[0], :new_shape[1]] = value
                        elif len(current_shape) == 3:
                            current_ip_adapter_state_dict[key][:new_shape[0], :new_shape[1], :new_shape[2]] = value
                        elif len(current_shape) == 4:
                            current_ip_adapter_state_dict[key][:new_shape[0], :new_shape[1], :new_shape[2],
                            :new_shape[3]] = value
                        else:
                            raise ValueError(f"unknown shape: {current_shape}")
                        print(f"Force merged in {key}: {list(current_shape)} <<< {list(new_shape)}")
                    except RuntimeError as e:
                        print(e)
                        print(
                            f"could not merge in {key}: {list(current_shape)} <<< {list(new_shape)}. Trying other way")

                        if (len(current_shape) == 1):
                            current_ip_adapter_state_dict[key][:current_shape[0]] = value[:current_shape[0]]
                        elif (len(current_shape) == 2):
                            current_ip_adapter_state_dict[key][:current_shape[0], :current_shape[1]] = value[
                                                                                                       :current_shape[
                                                                                                           0],
                                                                                                       :current_shape[
                                                                                                           1]]
                        elif (len(current_shape) == 3):
                            current_ip_adapter_state_dict[key][:current_shape[0], :current_shape[1],
                            :current_shape[2]] = value[:current_shape[0], :current_shape[1], :current_shape[2]]
                        elif (len(current_shape) == 4):
                            current_ip_adapter_state_dict[key][:current_shape[0], :current_shape[1], :current_shape[2],
                            :current_shape[3]] = value[:current_shape[0], :current_shape[1], :current_shape[2],
                                                 :current_shape[3]]
                        else:
                            raise ValueError(f"unknown shape: {current_shape}")
                        print(f"Force merged in {key}: {list(current_shape)} <<< {list(new_shape)}")

                else:
                    current_ip_adapter_state_dict[key] = value
        self.adapter_modules.load_state_dict(current_ip_adapter_state_dict)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        strict = False
        if self.config.train_scaler and 'ip_scale' in state_dict:
            self.adapter_modules.load_state_dict(state_dict["ip_scale"], strict=False)
        if 'ip_adapter' in state_dict:
            try:
                self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict)
                self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=strict)
            except Exception as e:
                print(e)
                print("could not load ip adapter weights, trying to merge in weights")
                self.merge_in_weights(state_dict)
        if self.config.train_image_encoder and 'image_encoder' in state_dict:
            self.image_encoder.load_state_dict(state_dict["image_encoder"], strict=strict)
        if self.preprocessor is not None and 'preprocessor' in state_dict:
            self.preprocessor.load_state_dict(state_dict["preprocessor"], strict=strict)

        if self.config.train_only_image_encoder and 'ip_adapter' not in state_dict:
            # we are loading pure clip weights.
            self.image_encoder.load_state_dict(state_dict, strict=strict)

    def enable_gradient_checkpointing(self):
        if hasattr(self.image_encoder, "enable_gradient_checkpointing"):
            self.image_encoder.enable_gradient_checkpointing()
        elif hasattr(self.image_encoder, 'gradient_checkpointing'):
            self.image_encoder.gradient_checkpointing = True
