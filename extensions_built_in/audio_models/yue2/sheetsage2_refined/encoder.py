"""MERT-v2-FullSong audio encoder: 24 kHz waveform -> 25 Hz frame features.

Re-implemented from m-a-p/MERT-v2-FullSong (weights: CC BY-NC 4.0).

    waveform -> log-mel (100 Hz, 128 bins, fixed normalisation)
             -> 3 ConvNeXt stages (strides 1, 2, 2 -> 25 Hz, 1024 channels)
             -> 24 Conformer blocks with rotary self-attention
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EncoderConfig:
    sample_rate: int = 24_000
    n_fft: int = 2048
    hop_length: int = 240
    n_mels: int = 128
    stage_channels: tuple[int, ...] = (128, 512, 1024)
    stage_depths: tuple[int, ...] = (3, 4, 5)
    stage_strides: tuple[int, ...] = (1, 2, 2)
    stage_norm_eps: float = 1e-6
    width: int = 1024
    ffn_width: int = 4096
    layers: int = 24
    heads: int = 16
    conv_kernel: int = 31
    rope_base: float = 10_000.0
    norm_eps: float = 1e-5

    @property
    def samples_per_frame(self) -> int:
        return self.hop_length * 4  # two stride-2 stages after the 100 Hz mel


class MelFrontend(nn.Module):
    """Power log-mel spectrogram with the checkpoint's per-bin mean and std."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.n_fft, self.hop_length = config.n_fft, config.hop_length
        self.register_buffer("window", torch.ones(config.n_fft))
        self.register_buffer("mel_fb", torch.zeros(config.n_fft // 2 + 1, config.n_mels))
        self.register_buffer("mel_mean", torch.zeros(config.n_mels))
        self.register_buffer("mel_std", torch.ones(config.n_mels))

    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        """``[B, samples]`` -> ``[B, frames, n_mels]``, always computed in float32."""

        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = (
                torch.stft(
                    waveform.float(),
                    self.n_fft,
                    self.hop_length,
                    window=self.window.float(),
                    center=True,
                    pad_mode="reflect",
                    return_complex=True,
                )
                .abs()
                .pow(2.0)
            )
            mel = torch.matmul(spectrum.transpose(-1, -2), self.mel_fb.float()).transpose(-1, -2)
            mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
            mel = mel[..., :-1].transpose(-1, -2)  # drop the trailing centre-padded frame
            return (mel - self.mel_mean.float()) / self.mel_std.float().clamp_min(1e-5)


class ConvNeXtLayer(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.grn_gamma = nn.Parameter(torch.zeros(4 * dim))
        self.grn_beta = nn.Parameter(torch.zeros(4 * dim))
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(self.fc1(self.norm(y)))
        # Global response norm: per-channel L2 norm over the whole time axis.
        magnitude = torch.norm(y, p=2, dim=1, keepdim=True)
        y = self.grn_gamma * (y * (magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6))) + self.grn_beta + y
        return x + self.fc2(y)


class ConvNeXtStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int, depth: int, eps: float):
        super().__init__()
        self.downsample = in_dim != out_dim or stride > 1
        if self.downsample:
            self.down_norm = nn.LayerNorm(in_dim, eps=eps)
            self.down_conv = nn.Conv1d(in_dim, out_dim, 2, stride=stride)
        self.layers = nn.Sequential(*[ConvNeXtLayer(out_dim, eps) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample:
            x = self.down_conv(self.down_norm(x).transpose(1, 2)).transpose(1, 2)
        return self.layers(x)


@torch.compiler.disable  # inductor would drop the bf16 round trip that the model depends on
def rotary_tables(frames: int, head_dim: int, base: float, device, dtype) -> tuple[Tensor, Tensor]:
    """``cos, sin`` of shape ``[1, frames, 1, head_dim]``, with bfloat16-rounded angles.

    The original implementation computed the angles with an einsum under bf16
    autocast, i.e. a bf16 matmul: positions, frequencies and their products are
    rounded to bfloat16, so frame positions above 256 collapse to steps of up to
    32 frames (1.28 s).  SheetSage2 was trained this way and breaks with exact
    angles (beat F 0.985 -> 0.447), so the rounding is reproduced explicitly here
    and holds whatever precision the caller runs in.  The frequencies are computed
    on the CPU because CUDA's pow rounds differently.
    """

    inverse_frequency = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(frames, dtype=torch.float32, device=device)
    angles = torch.outer(positions.bfloat16().float(), inverse_frequency.to(device).bfloat16().float()).bfloat16()
    angles = torch.cat((angles, angles), dim=-1)[None, :, None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotarySelfAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, x: Tensor, rotary: tuple[Tensor, Tensor]) -> Tensor:
        batch, frames, width = x.shape
        shape = (batch, frames, self.heads, width // self.heads)
        cos, sin = rotary
        q, k, v = self.q(x).reshape(shape), self.k(x).reshape(shape), self.v(x).reshape(shape)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out(y.transpose(1, 2).reshape(batch, frames, width))


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, width: int, kernel: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=eps)
        self.pointwise_in = nn.Conv1d(width, 2 * width, 1, bias=False)
        self.depthwise = nn.Conv1d(width, width, kernel, padding=(kernel - 1) // 2, groups=width, bias=False)
        self.depthwise_norm = nn.LayerNorm(width, eps=eps)
        self.pointwise_out = nn.Conv1d(width, width, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        y = F.glu(self.pointwise_in(self.norm(x).transpose(1, 2)), dim=1)
        y = self.depthwise_norm(self.depthwise(y).transpose(1, 2)).transpose(1, 2)
        return self.pointwise_out(F.gelu(y)).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        width, eps = config.width, config.norm_eps
        self.ffn1_norm = nn.LayerNorm(width, eps=eps)
        self.ffn1 = FeedForward(width, config.ffn_width)
        self.attn_norm = nn.LayerNorm(width, eps=eps)
        self.attn = RotarySelfAttention(width, config.heads)
        self.conv = ConvolutionModule(width, config.conv_kernel, eps)
        self.ffn2_norm = nn.LayerNorm(width, eps=eps)
        self.ffn2 = FeedForward(width, config.ffn_width)
        self.final_norm = nn.LayerNorm(width, eps=eps)

    def forward(self, x: Tensor, rotary: tuple[Tensor, Tensor]) -> Tensor:
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        x = self.attn(self.attn_norm(x), rotary) + x
        x = self.conv(x) + x
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


class MERTEncoder(nn.Module):
    def __init__(self, config: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.config = config
        self.frontend = MelFrontend(config)
        channels = (config.n_mels, *config.stage_channels)
        self.stages = nn.Sequential(
            *[
                ConvNeXtStage(
                    channels[i], channels[i + 1], config.stage_strides[i], config.stage_depths[i], config.stage_norm_eps
                )
                for i in range(len(config.stage_channels))
            ]
        )
        self.layers = nn.ModuleList([ConformerBlock(config) for _ in range(config.layers)])

    def forward(self, waveform: Tensor) -> list[Tensor]:
        """Return ``[input, block_1, ..., block_24]`` hidden states, each ``[B, frames, 1024]``."""

        mel = self.frontend(waveform)
        hidden = self.stages(mel.to(self.stages[0].layers[0].depthwise.weight.dtype))
        rotary = rotary_tables(
            hidden.shape[1],
            self.config.width // self.config.heads,
            self.config.rope_base,
            hidden.device,
            hidden.dtype,
        )
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden, rotary)
            states.append(hidden)
        return states
