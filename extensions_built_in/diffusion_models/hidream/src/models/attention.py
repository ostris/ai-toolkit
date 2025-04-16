import torch
from torch import nn
from typing import Optional
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import maybe_allow_in_graph

@maybe_allow_in_graph
class HiDreamAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor = None,
        out_dim: int = None,
        single: bool = False
    ):
        super(Attention, self).__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        linear_cls = nn.Linear
        self.linear_cls = linear_cls
        self.to_q = linear_cls(query_dim, self.inner_dim)
        self.to_k = linear_cls(self.inner_dim, self.inner_dim)
        self.to_v = linear_cls(self.inner_dim, self.inner_dim)
        self.to_out = linear_cls(self.inner_dim, self.out_dim)
        self.q_rms_norm = nn.RMSNorm(self.inner_dim, eps)
        self.k_rms_norm = nn.RMSNorm(self.inner_dim, eps)

        if not single:
            self.to_q_t = linear_cls(query_dim, self.inner_dim)
            self.to_k_t = linear_cls(self.inner_dim, self.inner_dim)
            self.to_v_t = linear_cls(self.inner_dim, self.inner_dim)
            self.to_out_t = linear_cls(self.inner_dim, self.out_dim)
            self.q_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)
            self.k_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)

        self.set_processor(processor)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        norm_image_tokens: torch.FloatTensor,
        image_tokens_masks: torch.FloatTensor = None,
        norm_text_tokens: torch.FloatTensor = None,
        rope: torch.FloatTensor = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            image_tokens = norm_image_tokens,
            image_tokens_masks = image_tokens_masks,
            text_tokens = norm_text_tokens,
            rope = rope,
        )

class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))