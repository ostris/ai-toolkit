"""MiniMax-H3 audio VAE -- frozen DAC/BigVGAN waveform autoencoder.

32 kHz mono waveform <-> 32-channel latents at 40 latents/second (hop 800,
no mel front-end and no separate vocoder). Stereo is carried by the CALLER
as two batch items. Runs in fp32 (the BigVGAN decoder degrades audibly
under bf16).

Weight-compatible with the ``MiniMaxAI/MiniMax-H3`` checkpoint
``FL2VA/audio_vae/model.safetensors`` once ``fold_audio_vae_weight_norm``
collapses its ``weight_g``/``weight_v`` weight-norm pairs into plain conv
weights::

    vae = MiniMaxH3AudioVAE()
    vae.load_state_dict(fold_audio_vae_weight_norm(load_file(path)), strict=True)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# fmt: off
# Per-channel latent statistics from the released FL2VA/audio_vae/config.json.
_LATENTS_MEAN = [
    -0.020211687488382354, 0.3876466479950502, -0.04398279799186767,
    -0.28591514936373, 0.08179686214561671, -0.35782641352446604,
    0.040623809960919084, -0.01552534501956604, -0.223362481667332,
    0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
    -0.056815072777201, -0.3699028221860095, -0.31616315591624855,
    0.5905951377425391, -0.052139568068853864, 0.013673160263486295,
    -0.03691647864630577, 0.09732660653298163, -0.3394662328788498,
    -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538, -0.21217779266454084, -0.1678263169941987,
    0.3221287889040614, -0.1223055851554907, 0.4356604928128464,
    -0.0502599202236253, 0.3979258376211797,
]
_LATENTS_STD = [
    1.6895524230479284, 2.76263727217653, 1.7945344281264435,
    1.6801681847309828, 1.6390226546605453, 2.7788298348882177,
    1.7659090095747236, 1.6199757612137327, 2.6336525640336896,
    1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364,
    3.298670198067373, 1.9491804496832168, 1.8720003270431442,
    1.8334080103291832, 1.6488070416529093, 1.6176957696319716,
    1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526,
    1.599198216109855, 1.718045989838149, 1.6307219190837705,
    1.8661226051202384, 1.5613768203168363,
]
# fmt: on


def fold_audio_vae_weight_norm(state_dict: dict) -> dict:
    """Fold weight-norm ``weight_g``/``weight_v`` pairs into plain ``.weight``
    tensors, passing every other key through untouched.

    The checkpoint uses ``nn.utils.weight_norm`` with the default ``dim=0``,
    so ``w = g * v / ||v||`` with the norm taken over every dim except 0.
    That also holds for the decoder's ConvTranspose1d layers: their
    ``weight_v`` is laid out ``[in, out, k]`` and ``weight_g`` is
    ``[in, 1, 1]``, i.e. the reduction still runs over dims (1, 2).
    """
    folded = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            g = state_dict[name[: -len("weight_v")] + "weight_g"]
            norm = tensor.norm(p=2, dim=tuple(range(1, tensor.ndim)), keepdim=True)
            folded[name[: -len("weight_v")] + "weight"] = g * tensor / norm
        else:
            folded[name] = tensor
    return folded


def _kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> Tensor:
    # Kaiser-windowed sinc low-pass, arithmetically identical to
    # alias-free-torch. The result is also stored in the checkpoint as a
    # buffer, so this init only matters for randomly initialized models.
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size)


class Snake1d(nn.Module):
    """``x + (alpha + 1e-9)^-1 * sin(alpha * x)^2`` with per-channel alpha
    stored as ``[1, C, 1]``. The DAC encoder's activation."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class SnakeBeta(nn.Module):
    """BigVGAN activation with separate frequency/magnitude parameters, both
    stored log-scale as ``[C]`` vectors:
    ``x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2``."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        alpha = torch.exp(self.alpha)[None, :, None]
        beta = torch.exp(self.beta)[None, :, None]
        return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class UpSample1d(nn.Module):
    """Anti-aliased ratio-x upsampler (depthwise transposed Kaiser-sinc conv)."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter",
            _kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.ratio, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer(
            "filter", _kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )


class DownSample1d(nn.Module):
    """Anti-aliased ratio-x downsampler."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.lowpass = LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.lowpass(x)


class AliasFreeActivation1d(nn.Module):
    """2x upsample -> activation -> 2x downsample (BigVGAN's alias-free wrapper)."""

    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(ratio, kernel_size)
        self.downsample = DownSample1d(ratio, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(self.act(self.upsample(x)))


class ResidualUnit(nn.Module):
    """DAC residual unit: Snake -> dilated conv k7 -> Snake -> conv k1."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(
                dim, dim, 7, dilation=dilation, padding=((7 - 1) * dilation) // 2
            ),
            Snake1d(dim),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        crop = (x.shape[-1] - y.shape[-1]) // 2
        if crop > 0:
            x = x[..., crop:-crop]
        return x + y


class EncoderBlock(nn.Module):
    """Three residual units (dilations 1/3/9), Snake, then a strided
    channel-doubling conv (kernel 2*stride, padding ceil(stride/2))."""

    def __init__(self, out_dim: int, stride: int):
        super().__init__()
        in_dim = out_dim // 2
        self.block = nn.Sequential(
            ResidualUnit(in_dim, dilation=1),
            ResidualUnit(in_dim, dilation=3),
            ResidualUnit(in_dim, dilation=9),
            Snake1d(in_dim),
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DacEncoder(nn.Module):
    """``[B, 1, samples] -> [B, d_latent, samples / prod(strides)]``."""

    def __init__(self, d_model: int, strides: tuple, d_latent: int):
        super().__init__()
        layers = [nn.Conv1d(1, d_model, 7, padding=3)]
        for stride in strides:
            d_model *= 2
            layers.append(EncoderBlock(d_model, stride))
        layers += [Snake1d(d_model), nn.Conv1d(d_model, d_latent, 3, padding=1)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)) * self.w1(x))


class CausalAttention(nn.Module):
    """Causal self-attention that narrows the width from in_dim to out_dim.

    QKV is one bias-less linear; the checkpoint stores separate q/v bias
    parameters and a frozen all-zero key bias buffer. The heads are
    MEAN-pooled away (not concatenated) and the surviving head dim (256) is
    adaptively average-pooled down to out_dim."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        q, k, v = (
            qkv.view(b, t, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.mean(dim=1)
        x = F.adaptive_avg_pool1d(x, self.out_dim)
        return self.proj(x)


class AttnProjection(nn.Module):
    """``pre_block``: rewires the 2048-wide encoder trunk to the 32-channel
    latent width. Operates on ``[B, T, C]``."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, out_dim * mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AMPBlock(nn.Module):
    """BigVGAN AMPBlock1: per dilation a (dilated conv, dilation-1 conv)
    pair, each conv preceded by its own alias-free SnakeBeta."""

    def __init__(self, channels: int, kernel_size: int, dilations: tuple):
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=d,
                padding=(kernel_size * d - d) // 2,
            )
            for d in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in dilations
        )
        self.activations = nn.ModuleList(
            AliasFreeActivation1d(SnakeBeta(channels))
            for _ in range(2 * len(dilations))
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            y = conv1(self.activations[2 * i](x))
            y = conv2(self.activations[2 * i + 1](y))
            x = x + y
        return x


class BigVGANDecoder(nn.Module):
    """``[B, in_channels, T] -> [B, 1, T * prod(rates)]``, clamped to [-1, 1]."""

    def __init__(
        self,
        in_channels: int = 2048,
        initial_channels: int = 1024,
        rates: tuple = (5, 5, 2, 2, 2, 2, 2),
        kernel_sizes: tuple = (9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple = (3, 7, 11),
        resblock_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(in_channels, initial_channels, 7, padding=3)

        # each upsampler is nested in a one-element ModuleList so the state
        # dict keys stay `ups.<i>.0.*` like the checkpoint
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = initial_channels
        for rate, kernel in zip(rates, kernel_sizes):
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose1d(
                            channels,
                            channels // 2,
                            kernel,
                            stride=rate,
                            padding=(kernel - rate) // 2,
                        )
                    ]
                )
            )
            channels //= 2
            for k, dils in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(AMPBlock(channels, k, dils))

        self.activation_post = AliasFreeActivation1d(SnakeBeta(channels))
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3, bias=False)

        self.gradient_checkpointing = True

    def _stage(self, i: int, x: Tensor) -> Tensor:
        x = self.ups[i][0](x)
        acc = None
        for j in range(self.num_kernels):
            y = self.resblocks[i * self.num_kernels + j](x)
            acc = y if acc is None else acc + y
        return acc / self.num_kernels

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_pre(x)
        for i in range(len(self.ups)):
            # gated on is_grad_enabled (not train mode): the VAE stays eval
            # but differentiable decodes during training should checkpoint
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(self._stage, i, x, use_reentrant=False)
            else:
                x = self._stage(i, x)
        x = self.activation_post(x)
        x = self.conv_post(x)
        return torch.clamp(x, min=-1.0, max=1.0)


class MiniMaxH3AudioVAE(nn.Module):
    """Frozen MiniMax-H3 audio autoencoder with diagonal latent normalization.

    - ``encode``: waveform ``(B, 1, samples)`` at 32 kHz -> latents ``(B, 32, T)``
    - ``decode``: latents ``(B, 32, T)`` -> waveform ``(B, 1, T * 800)``
    """

    SAMPLE_RATE = 32000
    HOP_LENGTH = 800
    LATENT_CHANNELS = 32
    LATENTS_PER_SECOND = 40.0

    def __init__(self):
        super().__init__()
        self.encoder = DacEncoder(d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048)
        self.pre_block = AttnProjection(2048, 32, num_heads=8)
        self.mean_proj = nn.Conv1d(32, 32, 1)
        # log-STD head (not log-var); unused by encode(), kept for weight parity
        self.logs_proj = nn.Conv1d(32, 32, 1)
        self.dec_in_proj = nn.Conv1d(32, 2048, 1)
        self.decoder = BigVGANDecoder()

        self.register_buffer(
            "latents_mean",
            torch.tensor(_LATENTS_MEAN, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(_LATENTS_STD, dtype=torch.float32),
            persistent=False,
        )

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def downsampling_ratio(self) -> int:
        return self.HOP_LENGTH

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.decoder.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.enable_gradient_checkpointing(False)

    def _apply(self, fn, recurse=True):
        # This VAE is pinned to fp32 (bf16 decodes are audibly degraded).
        # Device moves pass through, but any cast that would land a float
        # tensor in a non-fp32 dtype is rewritten to a device-move of the
        # ORIGINAL tensor kept at fp32 -- the weights never round-trip
        # through a lower precision.
        def guarded(t):
            out = fn(t)
            if (
                torch.is_tensor(out)
                and out.is_floating_point()
                and out.dtype != torch.float32
            ):
                return t.to(device=out.device, dtype=torch.float32)
            return out

        return super()._apply(guarded, recurse)

    def encode(self, waveform: Tensor) -> Tensor:
        """Mono waveform ``(B, 1, samples)`` at 32 kHz -> normalized latents
        ``(B, 32, T)``. Uses the posterior mean (mode), like the released
        pipeline; the tail is zero-padded to a multiple of 800 samples."""
        waveform = waveform.float()
        remainder = waveform.shape[-1] % self.HOP_LENGTH
        if remainder:
            waveform = F.pad(waveform, (0, self.HOP_LENGTH - remainder))
        h = self.encoder(waveform)
        h = self.pre_block(h.transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(h)
        return (z - self.latents_mean[:, None]) / self.latents_std[:, None]

    def decode(self, latents: Tensor) -> Tensor:
        """Normalized latents ``(B, 32, T)`` -> waveform ``(B, 1, T * 800)``
        in [-1, 1], fp32."""
        latents = latents.float()
        z = latents * self.latents_std[:, None] + self.latents_mean[:, None]
        return self.decoder(self.dec_in_proj(z))

    def forward(self, waveform: Tensor) -> Tensor:
        return self.decode(self.encode(waveform))
