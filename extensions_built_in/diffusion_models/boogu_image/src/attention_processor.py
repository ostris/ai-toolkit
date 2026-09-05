# Vendored from the Boogu-Image repository (boogu/models/attention_processor.py).
# Original work: Copyright 2025 BAAI / OmniGen2 / HuggingFace. Apache-2.0.
#
# Attention here defaults to torch's ``scaled_dot_product_attention`` (the
# "native" backend) so the model has NO hard dependency on flash-attn. Flash
# Attention 2 is an OPTIONAL backend: each processor carries an
# ``attention_backend`` flag (set in bulk via
# ``BooguImageTransformer2DModel.set_attention_backend``) and only the "flash"
# branch touches the ``flash_attn`` package, so importing it stays lazy/guarded.
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import repeat

from .embeddings import apply_rotary_emb

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except ImportError:  # flash-attn is optional; "native" SDPA needs none of this.
    flash_attn_varlen_func = None
    index_first_axis = pad_input = unpad_input = None
    _FLASH_ATTN_AVAILABLE = False

# Supported attention backends. "native" -> SDPA, "flash" -> Flash Attention 2.
ATTENTION_BACKENDS = ("native", "flash")


def _get_unpad_data(mask_2d: torch.Tensor):
    """Indices / cu_seqlens / max_seqlen from a 2D padding mask [B, L]."""
    seqlens_in_batch = mask_2d.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask_2d.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def _upad_input(query, key, value, attention_mask, query_length, num_heads):
    """Unpad q/k/v for ``flash_attn_varlen_func`` given a [B, L] padding mask."""
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key.shape

    key = index_first_axis(
        key.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    value = index_first_axis(
        value.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )

    if query_length == kv_seq_len:
        query = index_first_axis(
            query.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query.device
        )
        indices_q = cu_seqlens_q[:-1]
        query = query.squeeze(1)
    else:
        q_mask = attention_mask[:, -query_length:]
        query, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
            query, q_mask
        )

    return (
        query,
        key,
        value,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def _flash_varlen_attention(query, key, value, attention_mask, attn, softmax_scale):
    """Run flash-attn varlen over a [B, L, heads, head_dim] q/k/v with a 2D mask.

    Returns the attention output flattened back to [B, L, heads * head_dim].
    """
    batch_size, sequence_length = query.shape[0], query.shape[1]
    kv_heads = key.shape[2]

    mask_2d = attention_mask.bool() if attention_mask is not None else None
    (
        query_states,
        key_states,
        value_states,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_q, max_seqlen_k),
    ) = _upad_input(query, key, value, mask_2d, sequence_length, attn.heads)

    if kv_heads < attn.heads:
        key_states = repeat(key_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)
        value_states = repeat(
            value_states, "l h c -> l (h k) c", k=attn.heads // kv_heads
        )

    attn_output_unpad = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        causal=False,
        softmax_scale=softmax_scale,
    )
    hidden_states = pad_input(attn_output_unpad, indices_q, batch_size, sequence_length)
    return hidden_states.flatten(-2)


class BooguImageDoubleStreamSelfAttnProcessor(nn.Module):
    """
    Double-stream self-attention processor.

    Instruction and image features each get their own q/k/v projections; the two
    streams are concatenated (instruction first), attended jointly, then split
    back and projected with separate output heads. Uses torch SDPA by default;
    set ``attention_backend = "flash"`` for Flash Attention 2.
    """

    def __init__(
        self,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "BooguImageDoubleStreamSelfAttnProcessor requires PyTorch 2.0+."
            )

        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.attention_backend = "native"

        query_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_kv_heads

        self.img_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.img_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        self.instruct_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.instruct_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.instruct_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        self.instruct_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.img_to_q.weight)
        nn.init.xavier_uniform_(self.img_to_k.weight)
        nn.init.xavier_uniform_(self.img_to_v.weight)
        nn.init.xavier_uniform_(self.instruct_to_q.weight)
        nn.init.xavier_uniform_(self.instruct_to_k.weight)
        nn.init.xavier_uniform_(self.instruct_to_v.weight)
        nn.init.xavier_uniform_(self.instruct_out.weight)
        nn.init.xavier_uniform_(self.img_out.weight)

        if self.img_to_q.bias is not None:
            nn.init.zeros_(self.img_to_q.bias)
            nn.init.zeros_(self.img_to_k.bias)
            nn.init.zeros_(self.img_to_v.bias)
            nn.init.zeros_(self.instruct_to_q.bias)
            nn.init.zeros_(self.instruct_to_k.bias)
            nn.init.zeros_(self.instruct_to_v.bias)
            nn.init.zeros_(self.instruct_out.bias)
            nn.init.zeros_(self.img_out.bias)

    def _concat_instruction_image_features(
        self,
        img_hidden_states_list: List[torch.Tensor],
        instruct_hidden_states_list: List[torch.Tensor],
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
    ) -> List[torch.Tensor]:
        """Concatenate instruction then image features into one joint sequence."""
        batch_size = img_hidden_states_list[0].shape[0]
        max_seq_len = max(seq_lengths)

        concatenated_list = []
        for img_tensor, instruct_tensor in zip(
            img_hidden_states_list, instruct_hidden_states_list
        ):
            device = img_tensor.device
            if instruct_tensor.device != device:
                instruct_tensor = instruct_tensor.to(device)

            feature_dim = img_tensor.shape[-1]
            concatenated = img_tensor.new_zeros(batch_size, max_seq_len, feature_dim)

            for i, (encoder_seq_len, seq_len) in enumerate(
                zip(encoder_seq_lengths, seq_lengths)
            ):
                concatenated[i, :encoder_seq_len] = instruct_tensor[i, :encoder_seq_len]
                concatenated[i, encoder_seq_len:seq_len] = img_tensor[
                    i, : seq_len - encoder_seq_len
                ]

            concatenated_list.append(concatenated)

        return concatenated_list

    def _split_instruction_image_features(
        self,
        hidden_states_list: List[torch.Tensor],
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Inverse of ``_concat_instruction_image_features``."""
        result_list = []
        for hidden_states in hidden_states_list:
            batch_size = hidden_states.shape[0]
            feature_dim = hidden_states.shape[-1]

            max_instruct_len = max(encoder_seq_lengths)
            max_img_len = max(
                seq_len - encoder_seq_len
                for seq_len, encoder_seq_len in zip(seq_lengths, encoder_seq_lengths)
            )

            instruct_hidden_states = hidden_states.new_zeros(
                batch_size, max_instruct_len, feature_dim
            )
            img_hidden_states = hidden_states.new_zeros(
                batch_size, max_img_len, feature_dim
            )

            for i, (encoder_seq_len, seq_len) in enumerate(
                zip(encoder_seq_lengths, seq_lengths)
            ):
                img_len = seq_len - encoder_seq_len
                instruct_hidden_states[i, :encoder_seq_len] = hidden_states[
                    i, :encoder_seq_len
                ]
                img_hidden_states[i, :img_len] = hidden_states[
                    i, encoder_seq_len:seq_len
                ]

            result_list.append((instruct_hidden_states, img_hidden_states))

        return result_list

    def __call__(
        self,
        attn: Attention,
        img_hidden_states: torch.Tensor,
        instruct_hidden_states: torch.Tensor,
        joint_attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        encoder_seq_lengths: List[int] = None,
        seq_lengths: List[int] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = img_hidden_states.shape[0]

        img_query = self.img_to_q(img_hidden_states)
        img_key = self.img_to_k(img_hidden_states)
        img_value = self.img_to_v(img_hidden_states)

        instruct_query = self.instruct_to_q(instruct_hidden_states)
        instruct_key = self.instruct_to_k(instruct_hidden_states)
        instruct_value = self.instruct_to_v(instruct_hidden_states)

        img_list = [img_query, img_key, img_value]
        instruct_list = [instruct_query, instruct_key, instruct_value]
        concatenated_list = self._concat_instruction_image_features(
            img_list, instruct_list, encoder_seq_lengths, seq_lengths
        )
        query, key, value = concatenated_list

        sequence_length = max(seq_lengths)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        kv_heads = inner_dim // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb, use_real=False)
            key = apply_rotary_emb(key, rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        if base_sequence_length is not None:
            softmax_scale = (
                math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            )
        else:
            softmax_scale = attn.scale

        if self.attention_backend == "flash":
            # q/k/v are [B, L, heads, head_dim]; the joint padding mask is 2D.
            hidden_states = _flash_varlen_attention(
                query, key, value, joint_attention_mask, attn, softmax_scale
            )
            hidden_states = hidden_states.type_as(query)
        else:
            if joint_attention_mask is not None:
                joint_attention_mask = joint_attention_mask.bool()
                if joint_attention_mask.dim() == 2:
                    joint_attention_mask = joint_attention_mask.view(
                        batch_size, 1, 1, -1
                    )
                elif joint_attention_mask.dim() == 3:
                    joint_attention_mask = joint_attention_mask.unsqueeze(1)
                else:
                    raise ValueError(
                        f"Unsupported joint_attention_mask shape: {joint_attention_mask.shape}"
                    )

            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            # explicitly repeat key/value to avoid the slow MATH SDPA backend that
            # enable_gqa triggers on some torch builds
            k = k.repeat_interleave(q.size(-3) // k.size(-3), -3)
            v = v.repeat_interleave(q.size(-3) // v.size(-3), -3)

            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=joint_attention_mask, scale=softmax_scale
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.type_as(query)

        split_results = self._split_instruction_image_features(
            [hidden_states], encoder_seq_lengths, seq_lengths
        )
        instruct_hidden_states, img_hidden_states = split_results[0]

        instruct_projected = self.instruct_out(instruct_hidden_states)
        img_projected = self.img_out(img_hidden_states)

        merged_list = self._concat_instruction_image_features(
            [img_projected], [instruct_projected], encoder_seq_lengths, seq_lengths
        )
        hidden_states = merged_list[0]

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class BooguImageAttnProcessor:
    """
    Single-stream self-attention processor with RoPE + QK norm.

    Uses torch SDPA by default; set ``attention_backend = "flash"`` for Flash
    Attention 2 (requires the ``flash_attn`` package).
    """

    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("BooguImageAttnProcessor requires PyTorch 2.0+.")
        self.attention_backend = "native"

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        kv_heads = inner_dim // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        if base_sequence_length is not None:
            softmax_scale = (
                math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            )
        else:
            softmax_scale = attn.scale

        if self.attention_backend == "flash" and (
            attention_mask is None or attention_mask.dim() == 2
        ):
            mask = (
                attention_mask
                if attention_mask is not None
                else query.new_ones(batch_size, sequence_length, dtype=torch.bool)
            )
            hidden_states = _flash_varlen_attention(
                query, key, value, mask, attn, softmax_scale
            )
            hidden_states = hidden_states.type_as(query)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.view(batch_size, 1, 1, -1)
            elif attention_mask.dim() == 3:
                B, L, _ = attention_mask.shape
                diag_valid = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
                lengths = diag_valid.sum(dim=-1)
                arange_L = torch.arange(L, device=attention_mask.device)
                q_valid = arange_L.unsqueeze(0) < lengths.unsqueeze(1)
                k_valid = q_valid
                causal = torch.tril(
                    torch.ones(L, L, dtype=torch.bool, device=attention_mask.device)
                )
                combined = causal & q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)
                attention_mask = combined.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unsupported attention_mask shape: {attention_mask.shape}"
                )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
