"""YuE2 Oobleck-style waveform VAE (48 kHz stereo, 64-ch latents at 25 Hz).

Matches the ``vae.*`` tensors of the Comfy-Org repack: channels 64, channel
multipliers (1, 2, 4, 8, 16, 32), strides (2, 2, 4, 4, 5, 6) -> hop 1920.
Encoding is deterministic (mean half of the bottleneck). Checkpoints carry
old-style ``weight_g`` / ``weight_v`` weight-norm keys; they are remapped to
torch parametrizations on load.
"""

import math

import torch
from torch import nn

from toolkit.models.v2._mixin import OstrisModelMixin

SAMPLE_RATE = 48000
HOP = 1920
LATENT_RATE = SAMPLE_RATE // HOP  # 25


def WNConv1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvT1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        a = self.alpha.unsqueeze(0).unsqueeze(-1).exp()
        b = self.beta.unsqueeze(0).unsqueeze(-1).exp()
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
            WNConv1d(in_ch, out_ch, 2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.layers(x)


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.layers = nn.Sequential(
            SnakeBeta(in_ch),
            # odd strides need output_padding to land on exactly stride * T
            WNConvT1d(in_ch, out_ch, 2 * stride, stride=stride, padding=math.ceil(stride / 2), output_padding=stride % 2),
            ResUnit(out_ch, 1),
            ResUnit(out_ch, 3),
            ResUnit(out_ch, 9),
        )

    def forward(self, x):
        return self.layers(x)


class _SeqWrap(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


def convert_weight_norm_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_g"):
            k = k[: -len(".weight_g")] + ".parametrizations.weight.original0"
        elif k.endswith(".weight_v"):
            k = k[: -len(".weight_v")] + ".parametrizations.weight.original1"
        out[k] = v
    return out


class YuE2VAE(nn.Module, OstrisModelMixin):
    @classmethod
    def load_from_state_dict(cls, state_dict, dtype=torch.float32):
        vae = cls()
        sd = convert_weight_norm_keys(state_dict)
        missing, unexpected = vae.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
        if missing:
            raise ValueError(f"YuE2 VAE missing keys: {missing[:5]} (+{max(0, len(missing) - 5)})")
        if unexpected:
            print(f"    YuE2 VAE unexpected: {len(unexpected)} (first 3: {unexpected[:3]})")
        return vae.to(dtype)

    def __init__(
        self,
        in_ch=2,
        channels=64,
        latent_dim=64,
        c_mults=(1, 2, 4, 8, 16, 32),
        strides=(2, 2, 4, 4, 5, 6),
    ):
        super().__init__()
        cm = [1] + list(c_mults)
        enc = [WNConv1d(in_ch, cm[0] * channels, 7, padding=3)]
        for i in range(len(cm) - 1):
            enc.append(EncBlock(cm[i] * channels, cm[i + 1] * channels, strides[i]))
        enc += [SnakeBeta(cm[-1] * channels), WNConv1d(cm[-1] * channels, latent_dim * 2, 3, padding=1)]
        self.encoder = _SeqWrap(*enc)
        dec = [WNConv1d(latent_dim, cm[-1] * channels, 7, padding=3)]
        for i in range(len(cm) - 1, 0, -1):
            dec.append(DecBlock(cm[i] * channels, cm[i - 1] * channels, strides[i - 1]))
        dec += [SnakeBeta(cm[0] * channels), WNConv1d(cm[0] * channels, in_ch, 7, padding=3, bias=False)]
        self.decoder = _SeqWrap(*dec)
        self.latent_dim = latent_dim
        self.upscale_factor = math.prod(strides)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode(self, x: torch.Tensor, chunk_seconds: float = 60.0) -> torch.Tensor:
        """[B, 2, samples] -> [B, 64, T]. Encodes in whole-second chunks that
        are multiples of the hop, matching the community tokenizer's prep."""
        chunk = int(chunk_seconds * SAMPLE_RATE)
        chunk -= chunk % HOP
        outs = []
        for s in range(0, x.shape[-1], chunk):
            seg = x[..., s : s + chunk]
            if seg.shape[-1] < HOP:
                break
            outs.append(self.encoder(seg).chunk(2, dim=1)[0])
        return torch.cat(outs, dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def tiled_decode(self, z: torch.Tensor, core_frames: int = 750, halo_frames: int = 16) -> torch.Tensor:
        """Halo-crop tiled decode: decode core + halo, keep the core."""
        total = z.shape[-1]
        if total <= core_frames + 2 * halo_frames:
            return self.decoder(z)
        pieces = []
        for start in range(0, total, core_frames):
            end = min(start + core_frames, total)
            lo = max(0, start - halo_frames)
            hi = min(total, end + halo_frames)
            audio = self.decoder(z[..., lo:hi])
            pieces.append(audio[..., (start - lo) * HOP : (start - lo + end - start) * HOP])
        return torch.cat(pieces, dim=-1)
