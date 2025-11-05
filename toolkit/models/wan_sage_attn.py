import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb as diffusers_apply_rotary_emb
from diffusers.models.transformers.transformer_wan import (
    _get_qkv_projections,
    _get_added_kv_projections,
)
from diffusers.models.attention_dispatch import dispatch_attention_fn
from toolkit.print import print_acc

HAS_LOGGED_ROTARY_SHAPES = False


class WanSageAttnProcessor2_0:
    """
    SageAttention processor for Wan models (T2V and I2V).
    Based on WanAttnProcessor2_0 but using sageattn for 2-3x speedup.
    """

    def __init__(self, num_img_tokens: int = 257):
        # Fallback only; we prefer computing image context length dynamically
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
            # Match Diffusers reference: reserve last 512 tokens for text, remaining (front) for image
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            img_ctx_len = max(encoder_hidden_states.shape[1] - 512, 0)
            if img_ctx_len > 0:
                encoder_hidden_states_img = encoder_hidden_states[:, :img_ctx_len]
                encoder_hidden_states = encoder_hidden_states[:, img_ctx_len:]
            else:
                encoder_hidden_states_img = None  # text-only context; no image tokens
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
            # Apply via diffusers helper in a consistent layout for both tuple and tensor rotary
            query_hnd = query.permute(0, 2, 1, 3)  # (B, H, S, D) -> (B, S, H, D)
            key_hnd = key.permute(0, 2, 1, 3)
            query_hnd = diffusers_apply_rotary_emb(query_hnd, rotary_emb, use_real=False)
            key_hnd = diffusers_apply_rotary_emb(key_hnd, rotary_emb, use_real=False)
            query = query_hnd.permute(0, 2, 1, 3)
            key = key_hnd.permute(0, 2, 1, 3)

        # I2V task - process image conditioning separately
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            if hasattr(attn, "norm_added_k") and attn.norm_added_k is not None:
                key_img = attn.norm_added_k(key_img)
            if hasattr(attn, "norm_added_v") and attn.norm_added_v is not None:
                value_img = attn.norm_added_v(value_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))  # (B, S_img, H, D)
            value_img = value_img.unflatten(2, (attn.heads, -1))

            # Permute to HND layout expected by sageattn
            q_hnd = query.permute(0, 2, 1, 3)
            k_img_hnd = key_img.permute(0, 2, 1, 3)
            v_img_hnd = value_img.permute(0, 2, 1, 3)
            sm_scale = getattr(attn, "scale", None)
            if sm_scale is None:
                sm_scale = 1.0 / (q_hnd.shape[-1] ** 0.5)

            hs_img_hnd = sageattn(q_hnd, k_img_hnd, v_img_hnd, tensor_layout="HND", is_causal=False, sm_scale=sm_scale)
            # Back to (B, S, H, D), then flatten heads
            hidden_states_img = hs_img_hnd.permute(0, 2, 1, 3).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Main attention; if an attention mask is provided, fall back to reference backend for correctness
        if attention_mask is not None:
            hs = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, backend=None
            )
            hidden_states = hs.flatten(2, 3)
            hidden_states = hidden_states.type_as(query)
        else:
            q_hnd = query.permute(0, 2, 1, 3)
            k_hnd = key.permute(0, 2, 1, 3)
            v_hnd = value.permute(0, 2, 1, 3)
            sm_scale = getattr(attn, "scale", None)
            if sm_scale is None:
                sm_scale = 1.0 / (q_hnd.shape[-1] ** 0.5)
            hs_hnd = sageattn(q_hnd, k_hnd, v_hnd, tensor_layout="HND", is_causal=False, sm_scale=sm_scale)
            hidden_states = hs_hnd.permute(0, 2, 1, 3).flatten(2, 3)
            hidden_states = hidden_states.type_as(query)

        # Combine image conditioning if present
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
