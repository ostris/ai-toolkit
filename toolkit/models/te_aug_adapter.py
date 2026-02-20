import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from typing import Union, TYPE_CHECKING, Optional, Tuple

from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5Tokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPAttention

from toolkit.models.zipper_resampler import ZipperResampler, ZipperModule

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.custom_adapter import CustomAdapter


class TEAugAdapterCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attn_module: 'CLIPAttention', adapter: 'TEAugAdapter'):
        super().__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.attn_module_ref: weakref.ref = weakref.ref(attn_module)
        self.k_proj_adapter = nn.Linear(attn_module.embed_dim, attn_module.embed_dim)
        self.v_proj_adapter = nn.Linear(attn_module.embed_dim, attn_module.embed_dim)
        # copy the weights from the original module
        self.k_proj_adapter.weight.data = attn_module.k_proj.weight.data.clone() * 0.01
        self.v_proj_adapter.weight.data = attn_module.v_proj.weight.data.clone() * 0.01
        #reset the bias
        self.k_proj_adapter.bias.data = attn_module.k_proj.bias.data.clone() * 0.001
        self.v_proj_adapter.bias.data = attn_module.v_proj.bias.data.clone() * 0.001

        self.zipper = ZipperModule(
            in_size=attn_module.embed_dim,
            in_tokens=77 * 2,
            out_size=attn_module.embed_dim,
            out_tokens=77,
            hidden_size=attn_module.embed_dim,
            hidden_tokens=77,
        )
        # self.k_proj_adapter.weight.data = torch.zeros_like(attn_module.k_proj.weight.data)
        # self.v_proj_adapter.weight.data = torch.zeros_like(attn_module.v_proj.weight.data)
        # #reset the bias
        # self.k_proj_adapter.bias.data = torch.zeros_like(attn_module.k_proj.bias.data)
        # self.v_proj_adapter.bias.data = torch.zeros_like(attn_module.v_proj.bias.data)

        # replace the original forward with our forward
        self.original_forward = attn_module.forward
        attn_module.forward = self.forward


    @property
    def is_active(self):
        return self.adapter_ref().is_active

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        attn_module = self.attn_module_ref()

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = attn_module.q_proj(hidden_states) * attn_module.scale
        key_states = attn_module._shape(attn_module.k_proj(hidden_states), -1, bsz)
        value_states = attn_module._shape(attn_module.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * attn_module.num_heads, -1, attn_module.head_dim)
        query_states = attn_module._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * attn_module.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * attn_module.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, attn_module.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * attn_module.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, attn_module.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * attn_module.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, attn_module.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * attn_module.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=attn_module.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * attn_module.num_heads, tgt_len, attn_module.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn_module.num_heads, tgt_len, attn_module.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, attn_module.num_heads, tgt_len, attn_module.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        adapter: 'CustomAdapter' = self.adapter_ref().adapter_ref()
        if self.adapter_ref().is_active and adapter.conditional_embeds is not None:
            # apply the adapter

            if adapter.is_unconditional_run:
                embeds = adapter.unconditional_embeds
            else:
                embeds = adapter.conditional_embeds
                # if the shape is not the same on batch, we are doing cfg and need to concat unconditional as well
                if embeds.size(0) != bsz:
                    embeds = torch.cat([adapter.unconditional_embeds, embeds], dim=0)

            key_states_raw = self.k_proj_adapter(embeds)
            key_states = attn_module._shape(key_states_raw, -1, bsz)
            value_states_raw = self.v_proj_adapter(embeds)
            value_states = attn_module._shape(value_states_raw, -1, bsz)
            key_states = key_states.view(*proj_shape)
            value_states = value_states.view(*proj_shape)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(attn_weights, p=attn_module.dropout, training=self.training)
            attn_output_adapter = torch.bmm(attn_probs, value_states)

            if attn_output_adapter.size() != (bsz * attn_module.num_heads, tgt_len, attn_module.head_dim):
                raise ValueError(
                    f"`attn_output_adapter` should be of size {(bsz, attn_module.num_heads, tgt_len, attn_module.head_dim)}, but is"
                    f" {attn_output_adapter.size()}"
                )

            attn_output_adapter = attn_output_adapter.view(bsz, attn_module.num_heads, tgt_len, attn_module.head_dim)
            attn_output_adapter = attn_output_adapter.transpose(1, 2)
            attn_output_adapter = attn_output_adapter.reshape(bsz, tgt_len, embed_dim)

            attn_output_adapter = self.zipper(torch.cat([attn_output_adapter, attn_output], dim=1))

            # attn_output_adapter = attn_module.out_proj(attn_output_adapter)
            attn_output = attn_output + attn_output_adapter

        attn_output = attn_module.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

class TEAugAdapter(torch.nn.Module):
    def __init__(
            self,
            adapter: 'CustomAdapter',
            sd: 'StableDiffusion',
    ):
        super(TEAugAdapter, self).__init__()
        self.adapter_ref: weakref.ref = weakref.ref(adapter)
        self.sd_ref: weakref.ref = weakref.ref(sd)

        if isinstance(sd.text_encoder, list):
            raise ValueError("Dual text encoders is not yet supported")

        # dim will come from text encoder
        # dim = sd.unet.config['cross_attention_dim']
        text_encoder: CLIPTextModel = sd.text_encoder
        dim = text_encoder.config.hidden_size

        clip_encoder: CLIPEncoder = text_encoder.text_model.encoder
        # dim = clip_encoder.layers[-1].self_attn

        if hasattr(adapter.vision_encoder.config, 'hidden_sizes'):
            embedding_dim = adapter.vision_encoder.config.hidden_sizes[-1]
        else:
            embedding_dim = adapter.vision_encoder.config.hidden_size

        image_encoder_state_dict = adapter.vision_encoder.state_dict()
        # max_seq_len = CLIP tokens + CLS token
        in_tokens = 257
        if "vision_model.embeddings.position_embedding.weight" in image_encoder_state_dict:
            # clip
            in_tokens = int(image_encoder_state_dict["vision_model.embeddings.position_embedding.weight"].shape[0])

        if adapter.config.image_encoder_arch.startswith('convnext'):
            in_tokens = 16 * 16
            embedding_dim = adapter.vision_encoder.config.hidden_sizes[-1]

        out_tokens = adapter.config.num_tokens if adapter.config.num_tokens > 0 else in_tokens
        self.image_proj_model = ZipperModule(
            in_size=embedding_dim,
            in_tokens=in_tokens,
            out_size=dim,
            out_tokens=out_tokens,
            hidden_size=dim,
            hidden_tokens=out_tokens,
        )
        # init adapter modules
        attn_procs = {}
        for idx, layer in enumerate(clip_encoder.layers):
            name = f"clip_attention.{idx}"
            attn_procs[name] = TEAugAdapterCLIPAttention(
                layer.self_attn,
                self
            )

        self.adapter_modules = torch.nn.ModuleList(list(attn_procs.values()))

    # make a getter to see if is active
    @property
    def is_active(self):
        return self.adapter_ref().is_active


    def forward(self, input):
        # # apply the adapter
        input = self.image_proj_model(input)
        # self.embeds = input
        return input
