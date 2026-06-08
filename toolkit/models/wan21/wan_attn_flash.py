import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention

# Try to import Flash Attention
flash_attn_available = False
USE_FLASH_ATTN3 = False
try:
    from flash_attn_interface import flash_attn_func
    USE_FLASH_ATTN3 = True
    flash_attn_available = True
except ImportError:
    try:
        from flash_attn import flash_attn_func
        USE_FLASH_ATTN3 = False
        flash_attn_available = True
    except ImportError:
        flash_attn_available = False


# modified to set the image embedder size with flash attention support
class WanAttnProcessor2_0Flash:
    def __init__(self, num_img_tokens: int = 257, use_flash: bool = True):
        self.num_img_tokens = num_img_tokens
        self.use_flash = use_flash and flash_attn_available
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0Flash requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        if self.use_flash and not flash_attn_available:
            import warnings
            warnings.warn(
                "Flash attention requested but not available. Falling back to SDP. "
                "Install flash-attn for better performance: pip install flash-attn"
            )
            self.use_flash = False

    def _apply_flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply flash attention. Expects tensors in [batch, heads, seq_len, head_dim] format.
        Flash attention expects [batch, seq_len, heads, head_dim] format.
        """
        # Flash attention expects [batch, seq_len, heads, head_dim]
        # Current format is [batch, heads, seq_len, head_dim]
        batch, heads, seq_len, head_dim = query.shape
        
        # Transpose to [batch, seq_len, heads, head_dim] and make contiguous
        q_contiguous = query.transpose(1, 2).contiguous()
        k_contiguous = key.transpose(1, 2).contiguous()
        v_contiguous = value.transpose(1, 2).contiguous()
        
        if USE_FLASH_ATTN3:
            # Flash Attention 3 API
            hidden_states = flash_attn_func(
                q_contiguous, k_contiguous, v_contiguous,
                causal=False,
                deterministic=False
            )[0]
        else:
            # Flash Attention 2 API
            hidden_states = flash_attn_func(
                q_contiguous, k_contiguous, v_contiguous,
                dropout_p=0.0,
                causal=False
            )
        
        # Transpose back to [batch, heads, seq_len, head_dim]
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:,
                                                              :self.num_img_tokens]
            encoder_hidden_states = encoder_hidden_states[:,
                                                          self.num_img_tokens:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(
                    hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task - handle image conditioning separately
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(
                2, (attn.heads, -1)).transpose(1, 2)

            # For image attention, use flash if available and on GPU (CUDA or ROCm)
            if self.use_flash and query.device.type in ['cuda', 'hip'] and attention_mask is None:
                hidden_states_img = self._apply_flash_attention(
                    query, key_img, value_img
                )
            else:
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Main attention - use flash if available and on GPU (CUDA or ROCm)
        if self.use_flash and query.device.type in ['cuda', 'hip'] and attention_mask is None:
            hidden_states = self._apply_flash_attention(query, key, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


