#!/usr/bin/env python3
"""
ACE-Step v1.5 — Standalone single-file inference.

Generates music from text + lyrics. All model code inlined — no project imports,
no trust_remote_code. Uses ComfyUI-style architecture for AIO checkpoint compat.

Requirements:
    pip install torch torchaudio transformers safetensors

Usage:
    python simple_inference.py --prompt "indie folk, warm female vocal, 100 bpm" \
        --lyrics "[Verse]\\nSunlight through the window pane" --duration 30
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer
import torch.utils.checkpoint as ckpt

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATHS = {
    "base": os.path.join(MODELS_DIR, "ace_step_1.5_xl_base_aio.safetensors"),
    "turbo": os.path.join(MODELS_DIR, "ace_step_1.5_turbo_aio.safetensors"),
}
SAMPLE_RATE = 48000
LATENT_RATE = 25  # 48000 / 1920

SFT_PROMPT = """# Instruction
{instruction}

# Caption
{caption}

# Metas
{metas}<|endoftext|>
"""

TURBO_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.933, 0.857, 0.769, 0.667, 0.545, 0.4, 0.222],
    3.0: [
        1.0,
        0.9545454545454546,
        0.9,
        0.8333333333333334,
        0.75,
        0.6428571428571429,
        0.5,
        0.3,
    ],
}


def compute_timesteps(num_steps, shift=3.0):
    """Compute flow-matching timestep schedule with shifting."""
    import numpy as np

    sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]  # exclude final 0
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Silence latent (hardcoded, from ComfyUI)
# ═══════════════════════════════════════════════════════════════════════════════


def get_silence_latent(length, device, dtype=torch.bfloat16):
    head = torch.tensor(
        [
            [
                [
                    0.5707,
                    0.0982,
                    0.6909,
                    -0.5658,
                    0.6266,
                    0.6996,
                    -0.1365,
                    -0.1291,
                    -0.0776,
                    -0.1171,
                    -0.2743,
                    -0.8422,
                    -0.1168,
                    1.5539,
                    -4.6936,
                    0.7436,
                    -1.1846,
                    -0.2637,
                    0.6933,
                    -6.7266,
                    0.0966,
                    -0.1187,
                    -0.3501,
                    -1.1736,
                    0.0587,
                    -2.0517,
                    -1.3651,
                    0.7508,
                    -0.2490,
                    -1.3548,
                    -0.1290,
                    -0.7261,
                    1.1132,
                    -0.3249,
                    0.2337,
                    0.3004,
                    0.6605,
                    -0.0298,
                    -0.1989,
                    -0.4041,
                    0.2843,
                    -1.0963,
                    -0.5519,
                    0.2639,
                    -1.0436,
                    -0.1183,
                    0.0640,
                    0.4460,
                    -1.1001,
                    -0.6172,
                    -1.3241,
                    1.1379,
                    0.5623,
                    -0.1507,
                    -0.1963,
                    -0.4742,
                    -2.4697,
                    0.5302,
                    0.5381,
                    0.4636,
                    -0.1782,
                    -0.0687,
                    1.0333,
                    0.4202,
                ],
                [
                    0.3040,
                    -0.1367,
                    0.6200,
                    0.0665,
                    -0.0642,
                    0.4655,
                    -0.1187,
                    -0.0440,
                    0.2941,
                    -0.2753,
                    0.0173,
                    -0.2421,
                    -0.0147,
                    1.5603,
                    -2.7025,
                    0.7907,
                    -0.9736,
                    -0.0682,
                    0.1294,
                    -5.0707,
                    -0.2167,
                    0.3302,
                    -0.1513,
                    -0.8100,
                    -0.3894,
                    -0.2884,
                    -0.3149,
                    0.8660,
                    -0.3817,
                    -1.7061,
                    0.5824,
                    -0.4840,
                    0.6938,
                    0.1859,
                    0.1753,
                    0.3081,
                    0.0195,
                    0.1403,
                    -0.0754,
                    -0.2091,
                    0.1251,
                    -0.1578,
                    -0.4968,
                    -0.1052,
                    -0.4554,
                    -0.0320,
                    0.1284,
                    0.4974,
                    -1.1889,
                    -0.0344,
                    -0.8313,
                    0.2953,
                    0.5445,
                    -0.6249,
                    -0.1595,
                    -0.0682,
                    -3.1412,
                    0.0484,
                    0.4153,
                    0.8260,
                    -0.1526,
                    -0.0625,
                    0.5366,
                    0.8473,
                ],
                [
                    5.3524e-02,
                    -1.7534e-01,
                    5.4443e-01,
                    -4.3501e-01,
                    -2.1317e-03,
                    3.7200e-01,
                    -4.0143e-03,
                    -1.5516e-01,
                    -1.2968e-01,
                    -1.5375e-01,
                    -7.7107e-02,
                    -2.0593e-01,
                    -3.2780e-01,
                    1.5142e00,
                    -2.6101e00,
                    5.8698e-01,
                    -1.2716e00,
                    -2.4773e-01,
                    -2.7933e-02,
                    -5.0799e00,
                    1.1601e-01,
                    4.0987e-01,
                    -2.2030e-02,
                    -6.6495e-01,
                    -2.0995e-01,
                    -6.3474e-01,
                    -1.5893e-01,
                    8.2745e-01,
                    -2.2992e-01,
                    -1.6816e00,
                    5.4440e-01,
                    -4.9579e-01,
                    5.5128e-01,
                    3.0477e-01,
                    8.3052e-02,
                    -6.1782e-02,
                    5.9036e-03,
                    2.9553e-01,
                    -8.0645e-02,
                    -1.0060e-01,
                    1.9144e-01,
                    -3.8124e-01,
                    -7.2949e-01,
                    2.4520e-02,
                    -5.0814e-01,
                    2.3977e-01,
                    9.2943e-02,
                    3.9256e-01,
                    -1.1993e00,
                    -3.2752e-01,
                    -7.2707e-01,
                    2.9476e-01,
                    4.3542e-01,
                    -8.8597e-01,
                    -4.1686e-01,
                    -8.5390e-02,
                    -2.9018e00,
                    6.4988e-02,
                    5.3945e-01,
                    9.1988e-01,
                    5.8762e-02,
                    -7.0098e-02,
                    6.4772e-01,
                    8.9118e-01,
                ],
                [
                    -3.2225e-02,
                    -1.3195e-01,
                    5.6411e-01,
                    -5.4766e-01,
                    -5.2170e-03,
                    3.1425e-01,
                    -5.4367e-02,
                    -1.9419e-01,
                    -1.3059e-01,
                    -1.3660e-01,
                    -9.0984e-02,
                    -1.9540e-01,
                    -2.5590e-01,
                    1.5440e00,
                    -2.6349e00,
                    6.8273e-01,
                    -1.2532e00,
                    -1.9810e-01,
                    -2.2793e-02,
                    -5.0506e00,
                    1.8818e-01,
                    5.0109e-01,
                    7.3546e-03,
                    -6.8771e-01,
                    -3.0676e-01,
                    -7.3257e-01,
                    -1.6687e-01,
                    9.2232e-01,
                    -1.8987e-01,
                    -1.7267e00,
                    5.3355e-01,
                    -5.3179e-01,
                    4.4953e-01,
                    2.8820e-01,
                    1.3012e-01,
                    -2.0943e-01,
                    -1.1348e-01,
                    3.3929e-01,
                    -1.5069e-01,
                    -1.2919e-01,
                    1.8929e-01,
                    -3.6166e-01,
                    -8.0756e-01,
                    6.6387e-02,
                    -5.8867e-01,
                    1.6978e-01,
                    1.0134e-01,
                    3.3877e-01,
                    -1.2133e00,
                    -3.2492e-01,
                    -8.1237e-01,
                    3.8101e-01,
                    4.3765e-01,
                    -8.0596e-01,
                    -4.4531e-01,
                    -4.7513e-02,
                    -2.9266e00,
                    1.1741e-03,
                    4.5123e-01,
                    9.3075e-01,
                    5.3688e-02,
                    -1.9621e-01,
                    6.4530e-01,
                    9.3870e-01,
                ],
            ]
        ],
        device=device,
    ).movedim(-1, 1)
    body = (
        torch.tensor(
            [
                [
                    [
                        -1.3672e-01,
                        -1.5820e-01,
                        5.8594e-01,
                        -5.7422e-01,
                        3.0273e-02,
                        2.7930e-01,
                        -2.5940e-03,
                        -2.0703e-01,
                        -1.6113e-01,
                        -1.4746e-01,
                        -2.7710e-02,
                        -1.8066e-01,
                        -2.9688e-01,
                        1.6016e00,
                        -2.6719e00,
                        7.7734e-01,
                        -1.3516e00,
                        -1.9434e-01,
                        -7.1289e-02,
                        -5.0938e00,
                        2.4316e-01,
                        4.7266e-01,
                        4.6387e-02,
                        -6.6406e-01,
                        -2.1973e-01,
                        -6.7578e-01,
                        -1.5723e-01,
                        9.5312e-01,
                        -2.0020e-01,
                        -1.7109e00,
                        5.8984e-01,
                        -5.7422e-01,
                        5.1562e-01,
                        2.8320e-01,
                        1.4551e-01,
                        -1.8750e-01,
                        -5.9814e-02,
                        3.6719e-01,
                        -1.0059e-01,
                        -1.5723e-01,
                        2.0605e-01,
                        -4.3359e-01,
                        -8.2812e-01,
                        4.5654e-02,
                        -6.6016e-01,
                        1.4844e-01,
                        9.4727e-02,
                        3.8477e-01,
                        -1.2578e00,
                        -3.3203e-01,
                        -8.5547e-01,
                        4.3359e-01,
                        4.2383e-01,
                        -8.9453e-01,
                        -5.0391e-01,
                        -5.6152e-02,
                        -2.9219e00,
                        -2.4658e-02,
                        5.0391e-01,
                        9.8438e-01,
                        7.2754e-02,
                        -2.1582e-01,
                        6.3672e-01,
                        1.0000e00,
                    ]
                ]
            ],
            device=device,
        )
        .movedim(-1, 1)
        .repeat(1, 1, length)
    )
    body[:, :, : head.shape[-1]] = head
    return body.to(dtype)  # [1, 64, T]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos = None
        self._sin = None
        self._cached_len = 0

    def _build_cache(self, seq_len, device, dtype):
        if (
            seq_len <= self._cached_len
            and self._cos is not None
            and self._cos.device == device
        ):
            return
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._cached_len = seq_len

    def forward(self, x, seq_len):
        self._build_cache(seq_len, x.device, x.dtype)
        return self._cos[:seq_len], self._sin[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


class MLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def pack_sequences(h1, h2, m1, m2):
    h = torch.cat([h1, h2], dim=1)
    if m1 is not None and m2 is not None:
        m = torch.cat([m1, m2], dim=1)
        B, L, D = h.shape
        idx = m.argsort(dim=1, descending=True, stable=True)
        h = torch.gather(h, 1, idx.unsqueeze(-1).expand(B, L, D))
        lengths = m.sum(dim=1)
        m = torch.arange(L, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
    else:
        m = None
    return h, m


def timestep_embedding(t, dim, scale=1000, max_period=10000):
    t = t * scale
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# DiT model components (ComfyUI-style, matches AIO weight keys)
# ═══════════════════════════════════════════════════════════════════════════════


class TimestepEmbed(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(hidden, hidden)
        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(hidden, hidden * 6)
        self.scale = 1000

    def forward(self, t, dtype=None):
        emb = timestep_embedding(t, 256, self.scale)
        temb = self.act1(self.linear_1(emb.to(dtype=dtype)))
        temb = self.linear_2(temb)
        proj = self.time_proj(self.act2(temb)).view(-1, 6, temb.shape[-1])
        return temb, proj


class Attention(nn.Module):
    def __init__(
        self,
        hidden,
        num_heads,
        num_kv,
        head_dim,
        eps=1e-6,
        is_cross=False,
        sliding_window=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv = num_kv
        self.head_dim = head_dim
        self.is_cross = is_cross
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def forward(self, x, encoder_hidden_states=None, position_embeddings=None):
        B, L, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        ).transpose(1, 2)

        src = (
            encoder_hidden_states
            if (self.is_cross and encoder_hidden_states is not None)
            else x
        )
        sL = src.shape[1]
        k = self.k_norm(
            self.k_proj(src).view(B, sL, self.num_kv, self.head_dim)
        ).transpose(1, 2)
        v = self.v_proj(src).view(B, sL, self.num_kv, self.head_dim).transpose(1, 2)

        if position_embeddings is not None and not (
            self.is_cross and encoder_hidden_states is not None
        ):
            q, k = apply_rotary(q, k, *position_embeddings)

        n_rep = self.num_heads // self.num_kv
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        attn_bias = None
        if self.sliding_window is not None and not self.is_cross:
            idx = torch.arange(L, device=q.device)
            in_win = (
                torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)) <= self.sliding_window
            )
            attn_bias = torch.zeros(L, sL, device=q.device, dtype=q.dtype)
            attn_bias.masked_fill_(~in_win, torch.finfo(q.dtype).min)
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))


class EncoderLayer(nn.Module):
    def __init__(self, hidden, heads, kv, head_dim, inter, eps=1e-6):
        super().__init__()
        self.self_attn = Attention(hidden, heads, kv, head_dim, eps)
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, inter)

    def forward(self, x, position_embeddings):
        x = x + self.self_attn(
            self.input_layernorm(x), position_embeddings=position_embeddings
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DiTLayer(nn.Module):
    def __init__(
        self, hidden, heads, kv, head_dim, inter, eps=1e-6, sliding_window=None
    ):
        super().__init__()
        self.self_attn_norm = RMSNorm(hidden, eps)
        self.self_attn = Attention(
            hidden, heads, kv, head_dim, eps, sliding_window=sliding_window
        )
        self.cross_attn_norm = RMSNorm(hidden, eps)
        self.cross_attn = Attention(hidden, heads, kv, head_dim, eps, is_cross=True)
        self.mlp_norm = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, inter)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, hidden))

    def forward(self, x, temb, enc, position_embeddings):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = (
            self.scale_shift_table.to(temb) + temb
        ).chunk(6, dim=1)
        x = (
            x
            + self.self_attn(
                self.self_attn_norm(x) * (1 + sc_msa) + s_msa,
                position_embeddings=position_embeddings,
            )
            * g_msa
        )
        x = x + self.cross_attn(self.cross_attn_norm(x), encoder_hidden_states=enc)
        x = x + self.mlp(self.mlp_norm(x) * (1 + sc_mlp) + s_mlp) * g_mlp
        return x


# ── Encoders ──


class LyricEncoder(nn.Module):
    def __init__(
        self, text_dim, hidden, n_layers, heads, kv, head_dim, inter, eps=1e-6
    ):
        super().__init__()
        self.embed_tokens = nn.Linear(text_dim, hidden)
        self.norm = RMSNorm(hidden, eps)
        self.rotary_emb = RotaryEmbedding(head_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden, heads, kv, head_dim, inter, eps)
                for _ in range(n_layers)
            ]
        )

    def forward(self, embeds):
        x = self.embed_tokens(embeds)
        cos, sin = self.rotary_emb(x, x.shape[1])
        for layer in self.layers:
            x = layer(x, (cos, sin))
        return self.norm(x)


class TimbreEncoder(nn.Module):
    def __init__(
        self, timbre_dim, hidden, n_layers, heads, kv, head_dim, inter, eps=1e-6
    ):
        super().__init__()
        self.embed_tokens = nn.Linear(timbre_dim, hidden)
        self.norm = RMSNorm(hidden, eps)
        self.rotary_emb = RotaryEmbedding(head_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden, heads, kv, head_dim, inter, eps)
                for _ in range(n_layers)
            ]
        )
        self.special_token = nn.Parameter(torch.empty(1, 1, hidden))

    def forward(self, packed, order_mask):
        x = self.embed_tokens(packed)
        cos, sin = self.rotary_emb(x, x.shape[1])
        for layer in self.layers:
            x = layer(x, (cos, sin))
        x = self.norm(x)
        cls = x[:, 0, :]
        # Unpack to batch
        N, D = cls.shape
        B = int(order_mask.max().item() + 1)
        counts = torch.bincount(order_mask, minlength=B)
        mc = counts.max().item()
        result = torch.zeros(B, mc, D, device=cls.device, dtype=cls.dtype)
        mask = torch.zeros(B, mc, device=cls.device, dtype=torch.long)
        for i in range(N):
            b = order_mask[i].item()
            pos = (order_mask[:i] == b).sum().item()
            result[b, pos] = cls[i]
            mask[b, pos] = 1
        return result, mask


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        text_dim,
        timbre_dim,
        hidden,
        n_lyric,
        n_timbre,
        heads,
        kv,
        head_dim,
        inter,
        eps=1e-6,
    ):
        super().__init__()
        self.text_projector = nn.Linear(text_dim, hidden, bias=False)
        self.lyric_encoder = LyricEncoder(
            text_dim, hidden, n_lyric, heads, kv, head_dim, inter, eps
        )
        self.timbre_encoder = TimbreEncoder(
            timbre_dim, hidden, n_timbre, heads, kv, head_dim, inter, eps
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, text_h, text_m, lyric_h, lyric_m, refer_packed, refer_order):
        text_proj = self.text_projector(text_h)
        lyric_enc = self.lyric_encoder(lyric_h)
        timbre_enc, timbre_mask = self.timbre_encoder(refer_packed, refer_order)
        merged, merged_m = pack_sequences(lyric_enc, timbre_enc, lyric_m, timbre_mask)
        final, final_m = pack_sequences(merged, text_proj, merged_m, text_m)
        return final, final_m


# ── DiT ──


class DiTModel(nn.Module):
    def __init__(
        self,
        in_ch,
        hidden,
        n_layers,
        heads,
        kv,
        head_dim,
        inter,
        patch,
        out_ch,
        layer_types=None,
        sliding_window=128,
        eps=1e-6,
        cond_dim=None,
    ):
        super().__init__()
        self.patch_size = patch
        self.rotary_emb = RotaryEmbedding(head_dim)
        self.proj_in = nn.Sequential(
            nn.Identity(), nn.Conv1d(in_ch, hidden, kernel_size=patch, stride=patch)
        )
        self.time_embed = TimestepEmbed(hidden)
        self.time_embed_r = TimestepEmbed(hidden)
        self.condition_embedder = nn.Linear(cond_dim or hidden, hidden)
        lt = layer_types or [
            "sliding_attention" if i % 2 == 0 else "full_attention"
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(
            [
                DiTLayer(
                    hidden,
                    heads,
                    kv,
                    head_dim,
                    inter,
                    eps,
                    sliding_window=sliding_window
                    if lt[i] == "sliding_attention"
                    else None,
                )
                for i in range(n_layers)
            ]
        )
        self.norm_out = RMSNorm(hidden, eps)
        self.proj_out = nn.Sequential(
            nn.Identity(),
            nn.ConvTranspose1d(hidden, out_ch, kernel_size=patch, stride=patch),
        )
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, hidden))
        self.gradient_checkpointing = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, x, timestep, timestep_r, attention_mask, enc_h, enc_m, context):
        temb_t, proj_t = self.time_embed(timestep, dtype=x.dtype)
        temb_r, proj_r = self.time_embed_r(timestep - timestep_r, dtype=x.dtype)
        temb = temb_t + temb_r
        tproj = proj_t + proj_r

        h = torch.cat([context, x], dim=-1)
        orig_len = h.shape[1]
        if h.shape[1] % self.patch_size != 0:
            h = F.pad(h, (0, 0, 0, self.patch_size - h.shape[1] % self.patch_size))
        h = self.proj_in(h.transpose(1, 2)).transpose(1, 2)
        enc = self.condition_embedder(enc_h)
        cos, sin = self.rotary_emb(h, h.shape[1])
        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                h = ckpt.checkpoint(
                    layer, h, tproj, enc, (cos, sin), use_reentrant=False
                )
            else:
                h = layer(h, tproj, enc, (cos, sin))
        shift, scale = (self.scale_shift_table.to(temb) + temb.unsqueeze(1)).chunk(
            2, dim=1
        )
        h = self.norm_out(h) * (1 + scale) + shift
        h = self.proj_out(h.transpose(1, 2)).transpose(1, 2)
        return h[:, :orig_len, :]


# ── Top-level model ──


class AceStep15(nn.Module):
    def __init__(
        self,
        hidden=2048,
        text_dim=1024,
        timbre_dim=64,
        out_ch=64,
        n_dit=24,
        n_lyric=8,
        n_timbre=4,
        heads=16,
        kv=8,
        head_dim=128,
        inter=6144,
        patch=2,
        in_ch=192,
        sliding_window=128,
        eps=1e-6,
        layer_types=None,
        # Encoder can have different size than decoder (XL models)
        enc_hidden=None,
        enc_heads=None,
        enc_kv=None,
        enc_inter=None,
    ):
        super().__init__()
        eh = enc_hidden or hidden
        eheads = enc_heads or heads
        ekv = enc_kv or kv
        einter = enc_inter or inter

        self.decoder = DiTModel(
            in_ch,
            hidden,
            n_dit,
            heads,
            kv,
            head_dim,
            inter,
            patch,
            out_ch,
            layer_types,
            sliding_window,
            eps,
            cond_dim=eh,
        )
        self.encoder = ConditionEncoder(
            text_dim,
            timbre_dim,
            eh,
            n_lyric,
            n_timbre,
            eheads,
            ekv,
            head_dim,
            einter,
            eps,
        )
        self.null_condition_emb = nn.Parameter(torch.empty(1, 1, eh))
        self._gradient_checkpointing = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._gradient_checkpointing = value
        self.decoder.gradient_checkpointing = value

    def prepare_condition(
        self,
        text_h,
        text_m,
        lyric_h,
        lyric_m,
        refer_packed,
        refer_order,
        src_latents,
        chunk_masks,
    ):
        enc_h, enc_m = self.encoder(
            text_h, text_m, lyric_h, lyric_m, refer_packed, refer_order
        )
        context = torch.cat([src_latents, chunk_masks.to(src_latents.dtype)], dim=-1)
        return enc_h, enc_m, context


# ═══════════════════════════════════════════════════════════════════════════════
# VAE (ComfyUI Oobleck style — uses parametrizations.weight_norm)
# ═══════════════════════════════════════════════════════════════════════════════


def WNConv1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvT1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(
        nn.ConvTranspose1d(*args, **kwargs)
    )


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        a = self.alpha.unsqueeze(0).unsqueeze(-1).exp().to(x.device)
        b = self.beta.unsqueeze(0).unsqueeze(-1).exp().to(x.device)
        return x + (1.0 / (b + 1e-9)) * torch.sin(x * a).pow(2)


class ResUnit(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            SnakeBeta(ch),
            WNConv1d(ch, ch, 7, dilation=dilation, padding=(dilation * 6) // 2),
            SnakeBeta(ch),
            WNConv1d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.layers = nn.Sequential(
            ResUnit(in_ch, 1),
            ResUnit(in_ch, 3),
            ResUnit(in_ch, 9),
            SnakeBeta(in_ch),
            WNConv1d(
                in_ch, out_ch, 2 * stride, stride=stride, padding=math.ceil(stride / 2)
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.layers = nn.Sequential(
            SnakeBeta(in_ch),
            WNConvT1d(
                in_ch, out_ch, 2 * stride, stride=stride, padding=math.ceil(stride / 2)
            ),
            ResUnit(out_ch, 1),
            ResUnit(out_ch, 3),
            ResUnit(out_ch, 9),
        )

    def forward(self, x):
        return self.layers(x)


class VAEBottleneck(nn.Module):
    def encode(self, x):
        mean, scale = x.chunk(2, dim=1)
        return mean

    def decode(self, x):
        return x


class _SeqWrap(nn.Module):
    """Wraps Sequential as .layers so state_dict keys match AIO format."""

    def __init__(self, *modules):
        super().__init__()
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class OobleckVAE(nn.Module):
    def __init__(
        self,
        in_ch=2,
        channels=128,
        latent_dim=64,
        c_mults=(1, 2, 4, 8, 16),
        strides=(2, 4, 4, 6, 10),
    ):
        super().__init__()
        cm = [1] + list(c_mults)
        # Encoder
        enc = [WNConv1d(in_ch, cm[0] * channels, 7, padding=3)]
        for i in range(len(cm) - 1):
            enc.append(EncBlock(cm[i] * channels, cm[i + 1] * channels, strides[i]))
        enc += [
            SnakeBeta(cm[-1] * channels),
            WNConv1d(cm[-1] * channels, latent_dim * 2, 3, padding=1),
        ]
        self.encoder = _SeqWrap(*enc)
        # Decoder
        dec = [WNConv1d(latent_dim, cm[-1] * channels, 7, padding=3)]
        for i in range(len(cm) - 1, 0, -1):
            dec.append(DecBlock(cm[i] * channels, cm[i - 1] * channels, strides[i - 1]))
        dec += [
            SnakeBeta(cm[0] * channels),
            WNConv1d(cm[0] * channels, in_ch, 7, padding=3, bias=False),
        ]
        self.decoder = _SeqWrap(*dec)
        self.bottleneck = VAEBottleneck()

    def encode(self, x):
        return self.bottleneck.encode(self.encoder(x))

    def decode(self, x):
        return self.decoder(self.bottleneck.decode(x))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# ═══════════════════════════════════════════════════════════════════════════════
# Text encoder (Qwen3-Embedding, just need embed_tokens + model)
# ═══════════════════════════════════════════════════════════════════════════════


class TextEncoder(nn.Module):
    """Wraps Qwen3 weights loaded from AIO. Forward returns last_hidden_state."""

    def __init__(self, qwen_model):
        super().__init__()
        self.model = qwen_model  # the inner model (layers, norm, embed_tokens)

    def encode_text(self, input_ids):
        return self.model(input_ids=input_ids).last_hidden_state

    def encode_lyrics(self, input_ids):
        return self.model.embed_tokens(input_ids)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════


def infer_dit_config(dit_sd):
    """Infer model config from DiT state dict tensor shapes."""
    # hidden_size from decoder norm
    hidden = dit_sd["decoder.scale_shift_table"].shape[2]
    # intermediate_size from MLP gate_proj
    inter = dit_sd["decoder.layers.0.mlp.gate_proj.weight"].shape[0]
    # num_heads from q_proj: q_proj.weight is [num_heads * head_dim, hidden]
    q_size = dit_sd["decoder.layers.0.self_attn.q_proj.weight"].shape[0]
    # head_dim from q_norm
    head_dim = dit_sd["decoder.layers.0.self_attn.q_norm.weight"].shape[0]
    heads = q_size // head_dim
    # num_kv_heads from k_proj
    k_size = dit_sd["decoder.layers.0.self_attn.k_proj.weight"].shape[0]
    kv = k_size // head_dim
    # num_dit_layers: count unique layer indices
    n_dit = (
        max(int(k.split(".")[2]) for k in dit_sd if k.startswith("decoder.layers.")) + 1
    )
    # encoder hidden (may differ from decoder hidden for XL models)
    enc_hidden = dit_sd["encoder.text_projector.weight"].shape[0]
    # encoder layers
    n_lyric = (
        max(
            int(k.split(".")[3])
            for k in dit_sd
            if k.startswith("encoder.lyric_encoder.layers.")
        )
        + 1
    )
    n_timbre = (
        max(
            int(k.split(".")[3])
            for k in dit_sd
            if k.startswith("encoder.timbre_encoder.layers.")
        )
        + 1
    )
    # encoder attention config
    enc_heads = (
        dit_sd["encoder.lyric_encoder.layers.0.self_attn.q_proj.weight"].shape[0]
        // head_dim
    )
    enc_kv = (
        dit_sd["encoder.lyric_encoder.layers.0.self_attn.k_proj.weight"].shape[0]
        // head_dim
    )
    enc_inter = dit_sd["encoder.lyric_encoder.layers.0.mlp.gate_proj.weight"].shape[0]
    config = dict(
        hidden=hidden,
        inter=inter,
        heads=heads,
        kv=kv,
        head_dim=head_dim,
        n_dit=n_dit,
        n_lyric=n_lyric,
        n_timbre=n_timbre,
        enc_hidden=enc_hidden,
        enc_heads=enc_heads,
        enc_kv=enc_kv,
        enc_inter=enc_inter,
    )
    print(
        f"    Detected config: hidden={hidden}, inter={inter}, heads={heads}, kv={kv}, "
        f"n_dit={n_dit}, enc_hidden={enc_hidden}"
    )
    return config


def load_models(checkpoint_path, device="cuda", dtype=torch.bfloat16):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading from: {checkpoint_path}")

    sd = load_file(checkpoint_path)

    # --- DiT ---
    print("  Loading DiT...")
    dit_sd = {
        k.removeprefix("model.diffusion_model."): v
        for k, v in sd.items()
        if k.startswith("model.diffusion_model.")
    }
    cfg = infer_dit_config(dit_sd)
    model = AceStep15(
        hidden=cfg["hidden"],
        inter=cfg["inter"],
        heads=cfg["heads"],
        kv=cfg["kv"],
        head_dim=cfg["head_dim"],
        n_dit=cfg["n_dit"],
        n_lyric=cfg["n_lyric"],
        n_timbre=cfg["n_timbre"],
        enc_hidden=cfg["enc_hidden"],
        enc_heads=cfg["enc_heads"],
        enc_kv=cfg["enc_kv"],
        enc_inter=cfg["enc_inter"],
    )
    missing, unexpected = model.load_state_dict(dit_sd, strict=False)
    # tokenizer/detokenizer keys are expected to be unused (cover mode only)
    unexpected = [
        k for k in unexpected if not k.startswith(("tokenizer.", "detokenizer."))
    ]
    if missing:
        print(f"    DiT missing: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"    DiT unexpected: {len(unexpected)} (first 3: {unexpected[:3]})")
    model = model.to(device).to(dtype).eval()

    # --- VAE ---
    print("  Loading VAE...")
    vae_sd = {k.removeprefix("vae."): v for k, v in sd.items() if k.startswith("vae.")}
    vae = OobleckVAE()
    m, u = vae.load_state_dict(vae_sd, strict=False)
    if m:
        print(f"    VAE missing: {len(m)} (first 3: {m[:3]})")
    if u:
        print(f"    VAE unexpected: {len(u)}")
    vae = vae.to(device).to(dtype).eval()

    # --- Text encoder (Qwen3-Embedding from AIO) ---
    print("  Loading text encoder...")
    te_sd = {
        k.removeprefix("text_encoders.qwen3_06b.transformer.model."): v
        for k, v in sd.items()
        if k.startswith("text_encoders.qwen3_06b.transformer.model.")
    }
    # Load Qwen3 model structure from transformers, then override weights
    from transformers import Qwen3Model, Qwen3Config

    qwen_cfg = Qwen3Config(
        vocab_size=151669,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
    )
    qwen = Qwen3Model(qwen_cfg)
    m2, u2 = qwen.load_state_dict(te_sd, strict=False)
    if m2:
        print(f"    TE missing: {len(m2)} (first 3: {m2[:3]})")
    te = TextEncoder(qwen).to(device).to(dtype).eval()

    # Tokenizer — download from HF
    print("  Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=False
    )

    del sd  # free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  Done.\n")
    return dict(
        model=model, vae=vae, text_encoder=te, tokenizer=tok, device=device, dtype=dtype
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════


@torch.inference_mode()
def get_latent(audio_path, models):
    """Encode audio file to VAE latent. Returns [1, 64, T] tensor."""
    vae, device, dtype = models["vae"], models["device"], models["dtype"]
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return vae.encode(wav.unsqueeze(0).to(device, dtype))  # [1, 64, T]


@torch.inference_mode()
def generate(
    models,
    prompt,
    lyrics="",
    duration=30.0,
    seed=42,
    bpm="N/A",
    key="N/A",
    time_sig="N/A",
    language="en",
    timesteps=None,
    guidance_scale=1.0,
):
    model = models["model"]
    vae = models["vae"]
    te = models["text_encoder"]
    tok = models["tokenizer"]
    device = models["device"]
    dtype = models["dtype"]

    t_sched = timesteps
    latent_len = int(duration * LATENT_RATE)
    print(
        f"Duration: {duration}s -> {latent_len} latent frames, {len(t_sched)} steps"
        + (f", CFG={guidance_scale}" if guidance_scale > 1.0 else "")
    )

    # Silence as source latent [1, 64, T] -> [1, T, 64] for DiT
    sil = get_silence_latent(latent_len, device, dtype)  # [1, 64, T]
    src = sil.transpose(1, 2)  # [1, T, 64]
    chunk_masks = torch.ones_like(src)

    # Text encoding
    metas = f"- bpm: {bpm}\n- timesignature: {time_sig}\n- keyscale: {key}\n- duration: {int(duration)} seconds\n"
    caption = SFT_PROMPT.format(
        instruction="Fill the audio semantic mask based on the given conditions:",
        caption=prompt,
        metas=metas,
    )
    lyrics_text = f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    cap_tok = tok(caption, truncation=True, max_length=256, return_tensors="pt")
    lyr_tok = tok(lyrics_text, truncation=True, max_length=2048, return_tensors="pt")

    text_h = te.encode_text(cap_tok.input_ids.to(device)).to(dtype)
    text_m = cap_tok.attention_mask.to(device).bool()
    lyric_h = te.encode_lyrics(lyr_tok.input_ids.to(device)).to(dtype)
    lyric_m = lyr_tok.attention_mask.to(device).bool()

    # Reference audio (silence)
    ref = sil[:, :, :750].transpose(1, 2)  # [1, 750, 64]
    ref_order = torch.zeros(1, device=device, dtype=torch.long)

    # Prepare conditions (conditional)
    print("Preparing conditions...")
    enc_h, enc_m, ctx = model.prepare_condition(
        text_h, text_m, lyric_h, lyric_m, ref, ref_order, src, chunk_masks
    )

    # Prepare unconditional conditions for CFG
    use_cfg = guidance_scale > 1.0
    enc_h_uncond = None
    if use_cfg:
        enc_h_uncond = model.null_condition_emb.expand_as(enc_h)

    # Noise
    gen = torch.Generator(device=device).manual_seed(seed)
    noise_ch = ctx.shape[-1] // 2
    xt = torch.randn(1, latent_len, noise_ch, generator=gen, device=device, dtype=dtype)

    # Diffusion
    print("Running diffusion...")
    t0 = time.time()
    t_sched_t = torch.tensor(t_sched, device=device, dtype=dtype)
    attn = torch.ones(1, latent_len, device=device, dtype=dtype)

    for i in range(len(t_sched_t)):
        tv = t_sched_t[i].item()
        tt = torch.full((1,), tv, device=device, dtype=dtype)

        vt_cond = model.decoder(xt, tt, tt, attn, enc_h, enc_m, ctx)

        if use_cfg:
            vt_uncond = model.decoder(xt, tt, tt, attn, enc_h_uncond, enc_m, ctx)
            vt = vt_uncond + guidance_scale * (vt_cond - vt_uncond)
        else:
            vt = vt_cond

        if i == len(t_sched_t) - 1:
            xt = xt - vt * tv
        else:
            xt = xt - vt * (tv - t_sched_t[i + 1].item())

    print(f"Diffusion: {time.time() - t0:.2f}s")

    # VAE decode
    print("Decoding audio...")
    t0 = time.time()
    wav = vae.decode(xt.transpose(1, 2))  # [1, 2, samples]
    wav = wav[0, :, : int(duration * SAMPLE_RATE)]
    print(f"VAE decode: {time.time() - t0:.2f}s")
    return wav.cpu().float()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description="ACE-Step v1.5 standalone inference")
    p.add_argument("--prompt", required=True)
    p.add_argument("--lyrics", default="")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--output", default="output.wav")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        default="base",
        choices=["base", "turbo"],
        help="Model variant (default: base)",
    )
    p.add_argument(
        "--checkpoint", default=None, help="Override path to AIO .safetensors"
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    p.add_argument("--bpm", default="N/A")
    p.add_argument("--key", default="N/A")
    p.add_argument("--time-sig", default="N/A")
    p.add_argument("--language", default="en")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Diffusion steps (default: 30 for base, 8 for turbo)",
    )
    p.add_argument(
        "--shift", type=float, default=3.0, help="Timestep shift (default: 3.0)"
    )
    p.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="CFG guidance scale (default: 3.5 for base, 1.0 for turbo)",
    )
    args = p.parse_args()

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if device == "mps":
        dtype = torch.float32

    lyrics = args.lyrics
    if lyrics.startswith("@") and os.path.isfile(lyrics[1:]):
        lyrics = open(lyrics[1:]).read()
    else:
        lyrics = lyrics.replace("\\n", "\n")

    # Model-specific defaults
    is_turbo = args.model == "turbo"
    ckpt = args.checkpoint or MODEL_PATHS[args.model]
    steps = args.steps or (8 if is_turbo else 30)
    cfg = args.cfg if args.cfg is not None else (1.0 if is_turbo else 3.5)

    # Timestep schedule
    if is_turbo and steps == 8:
        ts = TURBO_TIMESTEPS.get(args.shift, TURBO_TIMESTEPS[3.0])
    else:
        ts = compute_timesteps(steps, args.shift)

    print(
        f"ACE-Step v1.5 ({args.model}) | {device} ({dtype}) | seed={args.seed} | {args.duration}s | {steps} steps | CFG={cfg}"
    )
    models = load_models(ckpt, device, dtype)
    wav = generate(
        models,
        args.prompt,
        lyrics,
        args.duration,
        args.seed,
        args.bpm,
        args.key,
        args.time_sig,
        args.language,
        ts,
        cfg,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torchaudio.save(args.output, wav, SAMPLE_RATE)
    print(f"Saved: {args.output} ({wav.shape[1] / SAMPLE_RATE:.1f}s stereo)")


if __name__ == "__main__":
    main()
