import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb as diffusers_apply_rotary_emb
from diffusers.models.transformers.transformer_wan import (
    _get_qkv_projections,
    _get_added_kv_projections,
)
from toolkit.print import print_acc

HAS_LOGGED_ROTARY_SHAPES = False


class WanSageAttnProcessor2_0:
    """
    SageAttention processor for Wan models (T2V and I2V).
    Based on WanAttnProcessor2_0 but using sageattn for 2-3x speedup.
    """

    def __init__(self, num_img_tokens: int = 257):
        self.num_img_tokens = num_img_tokens
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanSageAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        from sageattention import sageattn

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:,
                                                              :self.num_img_tokens]
            encoder_hidden_states = encoder_hidden_states[:,
                                                          self.num_img_tokens:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            global HAS_LOGGED_ROTARY_SHAPES
            if not HAS_LOGGED_ROTARY_SHAPES:
                try:
                    if isinstance(rotary_emb, tuple):
                        cos, sin = rotary_emb
                        print_acc(f"[WanSageAttn] rotary tuple shapes query={query.shape}, cos={cos.shape}, sin={sin.shape}")
                    else:
                        print_acc(f"[WanSageAttn] rotary tensor shapes query={query.shape}, rotary={rotary_emb.shape}")
                except Exception:
                    pass
                HAS_LOGGED_ROTARY_SHAPES = True
            # Support both tuple(rotary_cos, rotary_sin) and complex-valued rotary embeddings
            if isinstance(rotary_emb, tuple):
                freqs_cos, freqs_sin = rotary_emb

                def apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
                    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                    cos = cos[..., 0::2]
                    sin = sin[..., 1::2]
                    out = torch.empty_like(hidden_states)
                    out[..., 0::2] = x1 * cos - x2 * sin
                    out[..., 1::2] = x1 * sin + x2 * cos
                    return out.type_as(hidden_states)

                query = apply_rotary_emb(query, freqs_cos, freqs_sin)
                key = apply_rotary_emb(key, freqs_cos, freqs_sin)
            else:
                # Fallback path for complex rotary embeddings; temporarily permute to (B, H, S, D)
                query_hnd = query.permute(0, 2, 1, 3)
                key_hnd = key.permute(0, 2, 1, 3)
                query_hnd = diffusers_apply_rotary_emb(query_hnd, rotary_emb, use_real=False)
                key_hnd = diffusers_apply_rotary_emb(key_hnd, rotary_emb, use_real=False)
                query = query_hnd.permute(0, 2, 1, 3)
                key = key_hnd.permute(0, 2, 1, 3)

        # I2V task - process image conditioning separately
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            # Use SageAttention for image conditioning
            hidden_states_img = sageattn(
                query, key_img, value_img, attn_mask=None, is_causal=False, tensor_layout="NHD"
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Main attention with SageAttention
        hidden_states = sageattn(
            query, key, value, attn_mask=attention_mask, is_causal=False, tensor_layout="NHD"
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # Combine image conditioning if present
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
