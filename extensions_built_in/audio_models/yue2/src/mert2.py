"""MERT2 encoder (m-a-p/MERT-v2-30s, m-a-p/MERT-v2-FullSong) in plain PyTorch: no transformers remote code.

Module names match upstream ``modeling_mert2.py`` so the released ``model.safetensors`` loads strictly. Two switches
reproduce the YuE2 semantic-tokenizer branch (YuE2 report, arXiv 2609.33757, sec. 3.3 and App. B):

  causal   self-attention sees only the current and earlier frames. Everything else stays as released: the Conformer
           depthwise convolutions are symmetric (kernel 31 = +-15 frames per block) and the ConvNeXt GlobalResponseNorm
           takes an L2 norm over the WHOLE clip's time axis -- the paper's "causal self-attention with noncausal
           convolution and temporal normalization", which is why tokenization runs offline over the full recording.
  layers   build and run only the first N Conformer blocks (the deployed tokenizer keeps blocks 0-13).

Block outputs are indexed like upstream ``hidden_states``: 0 = output of block 0 (upstream README: "hidden_states[0]
is block 1"). Frames are 25 Hz: 24 kHz mono -> 100 Hz log-mel (hop 240) -> ConvNeXt x4 subsampling.
"""
import json
import os
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torchaudio.transforms import AmplitudeToDB, MelScale, Spectrogram

SAMPLE_RATE = 24000
HOP = 960                      # input samples per 25 Hz frame


class MelFrontend(nn.Module):
    """Power log-mel with the checkpoint's fixed per-bin normalisation, always computed in fp32."""

    def __init__(self, cfg: Dict):
        super().__init__()
        bins = cfg["num_mel_bins"]
        self.register_buffer("mel_mean", torch.zeros(bins))
        self.register_buffer("mel_std", torch.ones(bins))
        self.spectrogram = Spectrogram(n_fft=cfg["n_fft"], win_length=cfg["win_length"], hop_length=cfg["hop_length"], power=2.0)
        self.mel_scale = MelScale(n_mels=bins, sample_rate=cfg["sampling_rate"], n_stft=cfg["n_fft"] // 2 + 1)
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=None)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            mel = self.amplitude_to_db(self.mel_scale(self.spectrogram(wav.float())))
            mel = mel[..., :-1].transpose(-1, -2)
            return (mel - self.mel_mean) / self.mel_std.clamp_min(1e-5)


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class GlobalResponseNorm(nn.Module):
    """ConvNeXt-V2 GRN over [B, T, C]: the L2 norm runs over TIME (dim 1), i.e. over the whole clip."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        magnitude = torch.norm(x, p=2, dim=1, keepdim=True)
        return self.weight * (x * (magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6))) + self.bias + x


class ConvNextLayer(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.depthwise_block = nn.Sequential(Transpose(), nn.Conv1d(dim, dim, 7, padding=3, groups=dim), Transpose())
        self.pointwise_block = nn.Sequential(nn.LayerNorm(dim, eps=eps), nn.Linear(dim, 4 * dim), nn.GELU(),
                                             GlobalResponseNorm(4 * dim), nn.Linear(4 * dim, dim))

    def forward(self, x):
        return x + self.pointwise_block(self.depthwise_block(x))


class ConvNextBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, depth: int, eps: float):
        super().__init__()
        self.resampling_layer = (nn.Sequential(nn.LayerNorm(cin, eps=eps), Transpose(), nn.Conv1d(cin, cout, 2, stride=stride), Transpose())
                                 if cin != cout or stride > 1 else nn.Identity())
        self.convnext_layers = nn.Sequential(*[ConvNextLayer(cout, eps) for _ in range(depth)])

    def forward(self, x):
        return self.convnext_layers(self.resampling_layer(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        inv = 1.0 / (cfg["rotary_embedding_base"] ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x):
        """-> (cos, sin) [1, T, 1, head_dim] in x's dtype (angles in fp32)."""
        pos = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        ang = torch.einsum("i,j->ij", pos, self.inv_freq.float())
        ang = torch.cat((ang, ang), dim=-1)[None, :, None, :]
        return ang.cos().to(x.dtype), ang.sin().to(x.dtype)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, cfg: Dict, causal: bool):
        super().__init__()
        width = cfg["hidden_size"]
        self.num_heads, self.head_dim, self.causal = cfg["num_attention_heads"], width // cfg["num_attention_heads"], causal
        self.query_proj, self.key_proj = nn.Linear(width, width), nn.Linear(width, width)
        self.value_proj, self.out_proj = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, x, rope):
        b, t, w = x.shape
        shape = (b, t, self.num_heads, self.head_dim)
        q, k, v = (proj(x).reshape(shape) for proj in (self.query_proj, self.key_proj, self.value_proj))
        cos, sin = rope
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal)
        return self.out_proj(out.transpose(1, 2).reshape(b, t, w))


class FeedForward(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.w_1 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"])
        self.w_2 = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"])

    def forward(self, x):
        return self.w_2(F.gelu(self.w_1(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        width, kernel, eps = cfg["hidden_size"], cfg["conv_depthwise_kernel_size"], cfg["layer_norm_eps"]
        self.layer_norm = nn.LayerNorm(width, eps=eps)
        self.conv_block = nn.Sequential(
            Transpose(), nn.Conv1d(width, 2 * width, 1, bias=False), nn.GLU(dim=1),
            nn.Conv1d(width, width, kernel, padding=(kernel - 1) // 2, groups=width, bias=False),
            nn.Sequential(Transpose(), nn.LayerNorm(width, eps=eps), Transpose()), nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False), Transpose(),
        )

    def forward(self, x):
        return self.conv_block(self.layer_norm(x))


class ConformerBlock(nn.Module):
    def __init__(self, cfg: Dict, causal: bool):
        super().__init__()
        width, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.ffn1_layer_norm, self.ffn1 = nn.LayerNorm(width, eps=eps), FeedForward(cfg)
        self.attn_layer_norm, self.attn = nn.LayerNorm(width, eps=eps), SelfAttention(cfg, causal)
        self.conv_module = ConvolutionModule(cfg)
        self.ffn2_layer_norm, self.ffn2 = nn.LayerNorm(width, eps=eps), FeedForward(cfg)
        self.final_layer_norm = nn.LayerNorm(width, eps=eps)

    def forward(self, x, rope):
        x = x + 0.5 * self.ffn1(self.ffn1_layer_norm(x))
        x = self.attn(self.attn_layer_norm(x), rope) + x
        x = self.conv_module(x) + x
        x = x + 0.5 * self.ffn2(self.ffn2_layer_norm(x))
        return self.final_layer_norm(x)


class MERT2Encoder(nn.Module):
    """waveform [B, S] mono 24 kHz -> {block index: [B, S // 960, 1024]} for the requested block outputs."""

    def __init__(self, cfg: Dict, causal: bool = False, layers: Optional[int] = None):
        super().__init__()
        self.cfg, self.causal = cfg, causal
        self.checkpointing = False        # training: recompute each block in backward (full songs are ~9,000 frames)
        self.feature_extractor = MelFrontend(cfg)
        ch = [cfg["num_mel_bins"]] + list(cfg["subsampling_channels"])
        self.subsampling_module = nn.Sequential(*[
            ConvNextBlock(ch[i], ch[i + 1], (1, 2, 2)[i], cfg["subsampling_depths"][i], cfg["subsampling_layer_norm_eps"]) for i in range(3)
        ])
        self.layers = nn.ModuleList([ConformerBlock(cfg, causal) for _ in range(cfg["num_hidden_layers"] if layers is None else layers)])
        self.embed_positions = RotaryEmbedding(cfg)

    @classmethod
    def from_pretrained(cls, path: str, causal: bool = False, layers: Optional[int] = None) -> "MERT2Encoder":
        """A local snapshot dir (config.json + model.safetensors), or a unified tokenizer file: one safetensors whose
        metadata carries ``encoder.config`` and whose ``encoder.*`` tensors are this encoder (head.export_unified writes
        it, the head's tensors ride along under ``head.*``). ``layers`` builds only blocks 0..layers-1; the dropped
        blocks' tensors are the only ones allowed to go unused, everything built must load."""
        if os.path.isfile(path):
            from safetensors import safe_open

            with safe_open(path, "pt") as f:
                meta = f.metadata() or {}
                if "encoder.config" not in meta:
                    raise ValueError(f"{path} is not a unified tokenizer file (no encoder.config in its metadata)")
                cfg = json.loads(meta["encoder.config"])
                sd = {k[len("encoder."):]: f.get_tensor(k) for k in f.keys() if k.startswith("encoder.")}
        else:
            with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            sd = load_file(os.path.join(path, "model.safetensors"))
        enc = cls(cfg, causal=causal, layers=layers)
        n = len(enc.layers)
        sd = {k: v for k, v in sd.items() if not (k.startswith("layers.") and int(k.split(".")[1]) >= n)}
        enc.load_state_dict(sd, strict=True)
        return enc.eval().requires_grad_(False)

    def save_pretrained(self, path: str, **note) -> str:
        """A snapshot from_pretrained reads back: config.json (num_hidden_layers = the blocks this encoder has, plus
        ``note`` keys such as lineage) + model.safetensors in the upstream key layout."""
        from safetensors.torch import save_file

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({**self.cfg, "num_hidden_layers": len(self.layers), "causal_attention": self.causal, **note}, f, indent=2)
        save_file({k: v.detach().float().cpu().contiguous() for k, v in self.state_dict().items()}, os.path.join(path, "model.safetensors"))
        return path

    def forward(self, wav: torch.Tensor, outputs: Sequence[int] = ()) -> Dict[int, torch.Tensor]:
        """Run the frontend and blocks up to the deepest requested output; ``outputs`` = block indices (0-based).
        The frontend (log-mel + ConvNeXt) runs without grad unless its weights require it."""
        mel = self.feature_extractor(wav)
        with torch.set_grad_enabled(torch.is_grad_enabled() and any(p.requires_grad for p in self.subsampling_module.parameters())):
            x = self.subsampling_module(mel.to(self.subsampling_module[0].convnext_layers[0].depthwise_block[1].weight.dtype))
        rope = self.embed_positions(x)
        want, got = set(outputs), {}
        for i, layer in enumerate(self.layers[: max(want) + 1]):
            if self.checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(layer, x, rope, use_reentrant=False)
            else:
                x = layer(x, rope)
            if i in want:
                got[i] = x
        return got

    @torch.no_grad()
    def features(self, mono24: torch.Tensor, outputs: Sequence[int], chunk: float = 0) -> Dict[int, torch.Tensor]:
        """One song: mono24 [S] (24 kHz, on the encoder's device) -> {block: [T, 1024] fp32}, bf16 autocast on CUDA.
        ``chunk`` 0 = the whole song in one pass (T = S // 960: what a full-recording tokenizer sees; GRN normalises over
        the song). ``chunk`` > 0 = the community convention: ``chunk``-second pieces (full ones batched, pieces under 1 s
        dropped), concatenated and linearly resampled to round(S / 960) frames."""
        with torch.autocast(device_type=mono24.device.type, dtype=torch.bfloat16, enabled=mono24.device.type == "cuda"):
            if not chunk:
                return {k: v[0].float() for k, v in self(mono24[None], outputs).items()}
            n = int(SAMPLE_RATE * chunk)
            pieces = [p for p in (mono24[s : s + n] for s in range(0, mono24.shape[0], n)) if p.shape[0] >= SAMPLE_RATE]
            full = [p for p in pieces if p.shape[0] == n]
            groups = ([torch.stack(full)] if full else []) + [p[None] for p in pieces if p.shape[0] < n]
            parts = [self(g, outputs) for g in groups]
        t25 = int(round(mono24.shape[0] / HOP))
        return {k: F.interpolate(torch.cat([p[k].reshape(-1, p[k].shape[-1]) for p in parts]).float().T[None], size=t25,
                                 mode="linear", align_corners=False)[0].T for k in outputs}
