# originally from https://github.com/fal-ai/f-lite/blob/main/f_lite/model.py but modified slightly

import math

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=t.device
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, trainable=False):
        super().__init__()
        self.eps = eps
        if trainable:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            return (x * norm * self.weight).to(dtype=x_dtype)
        else:
            return (x * norm).to(dtype=x_dtype)


class QKNorm(nn.Module):
    """Normalizing the query and the key independently, as Flux proposes"""

    def __init__(self, dim, trainable=False):
        super().__init__()
        self.query_norm = RMSNorm(dim, trainable=trainable)
        self.key_norm = RMSNorm(dim, trainable=trainable)

    def forward(self, q, k):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        is_self_attn=True,
        cross_attn_input_size=None,
        residual_v=False,
        dynamic_softmax_temperature=False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_self_attn = is_self_attn
        self.residual_v = residual_v
        self.dynamic_softmax_temperature = dynamic_softmax_temperature

        if is_self_attn:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.context_kv = nn.Linear(cross_attn_input_size, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim, bias=False)

        if residual_v:
            self.lambda_param = nn.Parameter(torch.tensor(0.5).reshape(1))

        self.qk_norm = QKNorm(self.head_dim)

    def forward(self, x, context=None, v_0=None, rope=None):
        if self.is_self_attn:
            qkv = self.qkv(x)
            qkv = rearrange(qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
            q, k, v = qkv.unbind(0)

            if self.residual_v and v_0 is not None:
                v = self.lambda_param * v + (1 - self.lambda_param) * v_0

            if rope is not None:
                # print(q.shape, rope[0].shape, rope[1].shape)
                q = apply_rotary_emb(q, rope[0], rope[1])
                k = apply_rotary_emb(k, rope[0], rope[1])

                # https://arxiv.org/abs/2306.08645
                # https://arxiv.org/abs/2410.01104
                # ratioonale is that if tokens get larger, categorical distribution get more uniform
                # so you want to enlargen entropy.

                token_length = q.shape[2]
                if self.dynamic_softmax_temperature:
                    ratio = math.sqrt(math.log(token_length) / math.log(1040.0))  # 1024 + 16
                    k = k * ratio
            q, k = self.qk_norm(q, k)

        else:
            q = rearrange(self.q(x), "b l (h d) -> b h l d", h=self.num_heads)
            kv = rearrange(
                self.context_kv(context),
                "b l (k h d) -> k b h l d",
                k=2,
                h=self.num_heads,
            )
            k, v = kv.unbind(0)
            q, k = self.qk_norm(q, k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.proj(x)
        return x, v if self.is_self_attn else None


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        cross_attn_input_size,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        residual_v=False,
        dynamic_softmax_temperature=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = RMSNorm(hidden_size, trainable=qkv_bias)
        self.self_attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            is_self_attn=True,
            residual_v=residual_v,
            dynamic_softmax_temperature=dynamic_softmax_temperature,
        )

        if cross_attn_input_size is not None:
            self.norm2 = RMSNorm(hidden_size, trainable=qkv_bias)
            self.cross_attn = Attention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                is_self_attn=False,
                cross_attn_input_size=cross_attn_input_size,
                dynamic_softmax_temperature=dynamic_softmax_temperature,
            )
        else:
            self.norm2 = None
            self.cross_attn = None

        self.norm3 = RMSNorm(hidden_size, trainable=qkv_bias)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

        self.adaLN_modulation[-1].weight.data.zero_()
        self.adaLN_modulation[-1].bias.data.zero_()

    # @torch.compile(mode='reduce-overhead')
    def forward(self, x, context, c, v_0=None, rope=None):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)

        scale_sa = scale_sa[:, None, :]
        scale_ca = scale_ca[:, None, :]
        scale_mlp = scale_mlp[:, None, :]

        shift_sa = shift_sa[:, None, :]
        shift_ca = shift_ca[:, None, :]
        shift_mlp = shift_mlp[:, None, :]

        gate_sa = gate_sa[:, None, :]
        gate_ca = gate_ca[:, None, :]
        gate_mlp = gate_mlp[:, None, :]

        norm_x = self.norm1(x.clone())
        norm_x = norm_x * (1 + scale_sa) + shift_sa
        attn_out, v = self.self_attn(norm_x, v_0=v_0, rope=rope)
        x = x + attn_out * gate_sa

        if self.norm2 is not None:
            norm_x = self.norm2(x)
            norm_x = norm_x * (1 + scale_ca) + shift_ca
            x = x + self.cross_attn(norm_x, context)[0] * gate_ca

        norm_x = self.norm3(x)
        norm_x = norm_x * (1 + scale_mlp) + shift_mlp
        x = x + self.mlp(norm_x) * gate_mlp

        return x, v


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class TwoDimRotary(torch.nn.Module):
    def __init__(self, dim, base=10000, h=256, w=256):
        super().__init__()
        self.inv_freq = torch.FloatTensor([1.0 / (base ** (i / dim)) for i in range(0, dim, 2)])
        self.h = h
        self.w = w

        t_h = torch.arange(h, dtype=torch.float32)
        t_w = torch.arange(w, dtype=torch.float32)

        freqs_h = torch.outer(t_h, self.inv_freq).unsqueeze(1)  # h, 1, d / 2
        freqs_w = torch.outer(t_w, self.inv_freq).unsqueeze(0)  # 1, w, d / 2
        freqs_h = freqs_h.repeat(1, w, 1)  # h, w, d / 2
        freqs_w = freqs_w.repeat(h, 1, 1)  # h, w, d / 2
        freqs_hw = torch.cat([freqs_h, freqs_w], 2)  # h, w, d

        self.register_buffer("freqs_hw_cos", freqs_hw.cos())
        self.register_buffer("freqs_hw_sin", freqs_hw.sin())

    def forward(self, x, height_width=None, extend_with_register_tokens=0):
        if height_width is not None:
            this_h, this_w = height_width
        else:
            this_hw = x.shape[1]
            this_h, this_w = int(this_hw**0.5), int(this_hw**0.5)

        cos = self.freqs_hw_cos[0 : this_h, 0 : this_w]
        sin = self.freqs_hw_sin[0 : this_h, 0 : this_w]

        cos = cos.clone().reshape(this_h * this_w, -1)
        sin = sin.clone().reshape(this_h * this_w, -1)

        # append N of zero-attn tokens
        if extend_with_register_tokens > 0:
            cos = torch.cat(
                [
                    torch.ones(extend_with_register_tokens, cos.shape[1]).to(cos.device),
                    cos,
                ],
                0,
            )
            sin = torch.cat(
                [
                    torch.zeros(extend_with_register_tokens, sin.shape[1]).to(sin.device),
                    sin,
                ],
                0,
            )

        return cos[None, None, :, :], sin[None, None, :, :]  # [1, 1, T + N, Attn-dim]


def apply_rotary_emb(x, cos, sin):
    orig_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(dtype=orig_dtype)


class DiT(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):  # type: ignore[misc]
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        in_channels=4,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_input_size=128,
        residual_v=False,
        train_bias_and_rms=True,
        use_rope=True,
        gradient_checkpoint=False,
        dynamic_softmax_temperature=False,
        rope_base=10000,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)

        if use_rope:
            self.rope = TwoDimRotary(hidden_size // (2 * num_heads), base=rope_base, h=512, w=512)
        else:
            self.positional_embedding = nn.Parameter(torch.zeros(1, 2048, hidden_size))

        self.register_tokens = nn.Parameter(torch.randn(1, 16, hidden_size))

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cross_attn_input_size=cross_attn_input_size,
                    residual_v=residual_v,
                    qkv_bias=train_bias_and_rms,
                    dynamic_softmax_temperature=dynamic_softmax_temperature,
                )
                for _ in range(depth)
            ]
        )

        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        self.final_norm = RMSNorm(hidden_size, trainable=train_bias_and_rms)
        self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        self.paramstatus = {}
        for n, p in self.named_parameters():
            self.paramstatus[n] = {
                "shape": p.shape,
                "requires_grad": p.requires_grad,
            }
        self.gradient_checkpointing = False

    def save_lora_weights(self, save_directory):
        """Save LoRA weights to a file"""
        lora_state_dict = get_peft_model_state_dict(self)
        torch.save(lora_state_dict, f"{save_directory}/lora_weights.pt")

    def load_lora_weights(self, load_directory):
        """Load LoRA weights from a file"""
        lora_state_dict = torch.load(f"{load_directory}/lora_weights.pt")
        set_peft_model_state_dict(self, lora_state_dict)

    @apply_forward_hook
    def forward(self, x, context, timesteps):
        b, c, h, w = x.shape
        x = self.patch_embed(x)  # b, T, d

        x = torch.cat([self.register_tokens.repeat(b, 1, 1), x], 1)  # b, T + N, d

        if self.config.use_rope:
            cos, sin = self.rope(
                x,
                extend_with_register_tokens=16,
                height_width=(h // self.config.patch_size, w // self.config.patch_size),
            )
        else:
            x = x + self.positional_embedding.repeat(b, 1, 1)[:, : x.shape[1], :]
            cos, sin = None, None

        t_emb = timestep_embedding(timesteps * 1000, self.config.hidden_size).to(x.device, dtype=x.dtype)
        t_emb = self.time_embed(t_emb)

        v_0 = None

        for _idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x, v = self._gradient_checkpointing_func(
                    block,
                    x,
                    context,
                    t_emb,
                    v_0,
                    (cos, sin)
                )
            else:
                x, v = block(x, context, t_emb, v_0, (cos, sin))
            if v_0 is None:
                v_0 = v

        x = x[:, 16:, :]
        final_shift, final_scale = self.final_modulation(t_emb).chunk(2, dim=1)
        x = self.final_norm(x)
        x = x * (1 + final_scale[:, None, :]) + final_shift[:, None, :]
        x = self.final_proj(x)

        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h // self.config.patch_size,
            w=w // self.config.patch_size,
            p1=self.config.patch_size,
            p2=self.config.patch_size,
        )
        return x


if __name__ == "__main__":
    model = DiT(
        in_channels=4,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_input_size=128,
        residual_v=False,
        train_bias_and_rms=True,
        use_rope=True,
    ).cuda()
    print(
        model(
            torch.randn(1, 4, 64, 64).cuda(),
            torch.randn(1, 37, 128).cuda(),
            torch.tensor([1.0]).cuda(),
        )
    )