# orig code provided by lodestones, altered for ai-toolkit

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import torch.utils.checkpoint as ckpt


@dataclass
class ZImageDCTParams:
    patch_size: int = 1
    f_patch_size: int = 1
    in_channels: int = 128
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    rope_theta: int = 256
    t_scale: float = 1000.0
    axes_dims: list = field(default_factory=lambda: [32, 48, 48])
    axes_lens: list = field(default_factory=lambda: [1536, 512, 512])
    adaln_embed_dim: int = 256
    use_x0: bool = True
    # DCT decoder params
    decoder_hidden_size: int = 3840
    decoder_num_res_blocks: int = 4
    decoder_max_freqs: int = 8


class FakeConfig:
    # for diffusers compatability
    def __init__(self):
        self.patch_size = 1


def _process_mask(attn_mask: Optional[torch.Tensor], dtype: torch.dtype):
    if attn_mask is None:
        return None
    if attn_mask.ndim == 2:
        attn_mask = attn_mask[:, None, None, :]
    if attn_mask.dtype == torch.bool:
        new_mask = torch.zeros_like(attn_mask, dtype=dtype)
        new_mask.masked_fill_(~attn_mask, float("-inf"))
        return new_mask
    return attn_mask


def _native_attention_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn_mask = _process_mask(attn_mask, query.dtype)
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return out.transpose(1, 2).contiguous()


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
                / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat(
                    [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
                )
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList(
            [nn.Linear(n_heads * self.head_dim, dim, bias=False)]
        )

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_kv_heads, -1))
        value = value.unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        hidden_states = _native_attention_wrapper(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        return self.to_out[0](hidden_states)


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        adaln_embed_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList(
                [nn.Linear(min(dim, adaln_embed_dim), 4 * dim, bias=True)]
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            )
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256,
        axes_dims: List[int] = None,
        axes_lens: List[int] = None,
    ):
        self.theta = theta
        self.axes_dims = axes_dims or [32, 48, 48]
        self.axes_lens = axes_lens or [1536, 512, 512]
        assert len(self.axes_dims) == len(self.axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (
                    theta
                    ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
                )
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(
                    torch.complex64
                )
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim >= 2 and ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(
                self.axes_dims, self.axes_lens, theta=self.theta
            )
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]

        return torch.cat(
            [self.freqs_cis[i][ids[..., i]] for i in range(len(self.axes_dims))], dim=-1
        )


# --- Decoder components ---


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input)
        )

    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size, device, dtype):
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(
            0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device
        )
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]

        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs**2)
        return dct

    def forward(self, inputs):
        B, P2, C = inputs.shape
        original_dtype = inputs.dtype
        with torch.autocast("cuda", enabled=False):
            patch_size = int(P2**0.5)
            inputs = inputs.float()
            dct = self.fetch_pos(patch_size, inputs.device, torch.float32)
            dct = dct.repeat(B, 1, 1)
            inputs = torch.cat([inputs, dct], dim=-1)
            inputs = self.embedder.float()(inputs)
        return inputs.to(original_dtype)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class DCTFinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
        max_freqs=8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)
        self.input_embedder = NerfEmbedder(
            in_channels=in_channels,
            hidden_size_input=model_channels,
            max_freqs=max_freqs,
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.final_layer = DCTFinalLayer(model_channels, out_channels)
        nn.init.xavier_uniform_(self.cond_embed.weight)
        nn.init.constant_(self.cond_embed.bias, 0)

    def forward(self, x, c):
        x = self.input_embedder(x)
        c = self.cond_embed(c)
        y = c.reshape(c.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x)


class ZImageDCT(nn.Module):
    def __init__(self, params: ZImageDCTParams):
        super().__init__()
        self.config = FakeConfig()
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.patch_size = params.patch_size
        self.f_patch_size = params.f_patch_size
        self.dim = params.dim
        self.n_heads = params.n_heads
        self.rope_theta = params.rope_theta
        self.t_scale = params.t_scale
        self.adaln_embed_dim = params.adaln_embed_dim

        self.x_embedder = nn.Linear(
            self.f_patch_size * self.patch_size * self.patch_size * params.in_channels,
            params.dim,
            bias=True,
        )

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + i,
                    params.dim,
                    params.n_heads,
                    params.n_kv_heads,
                    params.norm_eps,
                    params.qk_norm,
                    modulation=True,
                    adaln_embed_dim=params.adaln_embed_dim,
                )
                for i in range(params.n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    i,
                    params.dim,
                    params.n_heads,
                    params.n_kv_heads,
                    params.norm_eps,
                    params.qk_norm,
                    modulation=False,
                )
                for i in range(params.n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(
            min(params.dim, params.adaln_embed_dim), mid_size=1024
        )

        self.cap_embedder = nn.Sequential(
            RMSNorm(params.cap_feat_dim, eps=params.norm_eps),
            nn.Linear(params.cap_feat_dim, params.dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, params.dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, params.dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    i,
                    params.dim,
                    params.n_heads,
                    params.n_kv_heads,
                    params.norm_eps,
                    params.qk_norm,
                    modulation=True,
                    adaln_embed_dim=params.adaln_embed_dim,
                )
                for i in range(params.n_layers)
            ]
        )

        head_dim = params.dim // params.n_heads
        assert head_dim == sum(params.axes_dims)
        self.axes_dims = params.axes_dims
        self.axes_lens = params.axes_lens

        self.rope_embedder = RopeEmbedder(
            theta=params.rope_theta,
            axes_dims=params.axes_dims,
            axes_lens=params.axes_lens,
        )

        self.dec_net = SimpleMLPAdaLN(
            in_channels=params.in_channels,
            model_channels=params.decoder_hidden_size,
            out_channels=params.in_channels,
            z_channels=params.dim,
            num_res_blocks=params.decoder_num_res_blocks,
            patch_size=self.patch_size,
            max_freqs=params.decoder_max_freqs,
        )

        if params.use_x0:
            self.register_buffer("__x0__", torch.tensor([]))

        self.gradient_checkpointing = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def _forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        img_mask: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
    ):
        B = img.shape[0]
        num_patches = img.shape[1]

        pixel_values = img.reshape(
            B * num_patches, self.patch_size**2, self.in_channels
        )

        timesteps = (1 - timesteps) * self.t_scale
        timesteps_embedding = self.t_embedder(timesteps)

        img_hidden = self.x_embedder(img)
        txt_hidden = self.cap_embedder(txt)

        img_pe = self.rope_embedder(img_ids)
        txt_pe = self.rope_embedder(txt_ids)

        for layer in self.noise_refiner:
            img_hidden = layer(img_hidden, img_mask, img_pe, timesteps_embedding)

        for layer in self.context_refiner:
            txt_hidden = layer(txt_hidden, txt_mask, txt_pe)

        mixed_hidden = torch.cat((txt_hidden, img_hidden), 1)
        mixed_mask = torch.cat((txt_mask, img_mask), 1)
        mixed_pe = torch.cat((txt_pe, img_pe), 1)

        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                mixed_hidden = ckpt.checkpoint(
                    layer,
                    mixed_hidden,
                    mixed_mask,
                    mixed_pe,
                    timesteps_embedding,
                    use_reentrant=False,
                )
            else:
                mixed_hidden = layer(
                    mixed_hidden, mixed_mask, mixed_pe, timesteps_embedding
                )

        img_hidden = mixed_hidden[:, txt.shape[1] :, ...]

        decoder_condition = img_hidden.reshape(B * num_patches, self.dim)
        output = self.dec_net(pixel_values, decoder_condition)
        output = output.reshape(B, num_patches, -1)

        return -output

    def _apply_x0_residual(self, predicted, noisy, timesteps):
        return (noisy - predicted) / timesteps.view(-1, 1, 1)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        img_mask: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
    ):
        out = self._forward(
            img=img,
            img_ids=img_ids,
            img_mask=img_mask,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=timesteps,
        )
        if hasattr(self, "__x0__"):
            return self._apply_x0_residual(out, img, timesteps)
        return out


def vae_flatten(latents, patch_size=2):
    """Patchify: [N, C, H, W] -> ([N, num_patches, patch_size*patch_size*C], original_shape)"""
    return (
        rearrange(
            latents,
            "n c (h dh) (w dw) -> n (h w) (dh dw c)",
            dh=patch_size,
            dw=patch_size,
        ),
        latents.shape,
    )


def vae_unflatten(latents, shape, patch_size=2):
    """Unpatchify: [N, num_patches, patch_size*patch_size*C] -> [N, C, H, W]"""
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (dh dw c) -> n c (h dh) (w dw)",
        dh=patch_size,
        dw=patch_size,
        c=c,
        h=h // patch_size,
        w=w // patch_size,
    )


def prepare_latent_image_ids(start_indices, height, width, patch_size=2, max_offset=0):
    """Generate 3D positional IDs for image patches."""
    if isinstance(start_indices, list):
        start_indices = torch.tensor(start_indices)

    batch_size = len(start_indices)
    latent_image_ids = torch.zeros(height // patch_size, width // patch_size, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // patch_size)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // patch_size)[None, :]
    )

    h, w, ch = latent_image_ids.shape
    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)

    for i, start_idx in enumerate(start_indices):
        latent_image_ids[i, :, :, 0] = start_idx

    return latent_image_ids.reshape(batch_size, h * w, ch).int()


def make_text_position_ids(valid_len, max_sequence_length, extra_padding=0):
    """Generate 3D positional IDs for text tokens."""
    device = valid_len.device
    valid_len = valid_len + extra_padding
    B = valid_len.shape[0]
    seq = (
        torch.arange(1, max_sequence_length + 1, device=device)
        .unsqueeze(0)
        .expand(B, -1)
    )
    increment_then_repeat = torch.minimum(seq, valid_len.unsqueeze(1))
    pos_ids = torch.zeros((B, max_sequence_length, 3), device=device)
    pos_ids[:, :, 0] = increment_then_repeat
    return pos_ids.int()



def time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list:
    """Build a shifted cosine timestep schedule from t=1 (noise) to t=0 (clean)."""
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        m = (max_shift - base_shift) / (4096 - 256)
        b = base_shift - m * 256
        mu = m * image_seq_len + b
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()
