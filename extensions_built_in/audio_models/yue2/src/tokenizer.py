"""Text side and audio side of YuE2 conditioning.

- ``YuE2TextTokenizer``: the Qwen BPE that ComfyUI embeds in the checkpoint
  (``text_encoders.yue2_tokenizer_json``), plus the prompt/prefix protocol.
- ``SemanticTokenizer``: the community audio -> codec-token head
  (Mothersuperior v4): MERT-v2-FullSong layer-20 features at 25 Hz,
  per-track instance normalised, through an 8-layer transformer classifier
  over 512-frame windows.
"""

import unicodedata
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .model import ABC_END, ABC_START, CODEC_SIZE, EOD, INSTRUCTIONS, MUSIC_START

MERT_REPO = "m-a-p/MERT-v2-FullSong"
MERT_SAMPLE_RATE = 24000
MERT_LAYER = 20
HEAD_REPO = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
HEAD_FILE = "tokenizer_head_joint_v4.pt"
NAR_LORA_FILE = "nar_lora_joint_v4.pt"
HEAD_WINDOW = 512


class YuE2TextTokenizer:
    def __init__(self, tokenizer_json: bytes):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(unicodedata.normalize("NFC", text)).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def prefix_ids(self, style: str, lyrics: str, cot: str = "off", abc: str = "") -> List[int]:
        """Everything the AR expert sees before the first codec token."""
        prompt = f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"
        abc_ids = [] if cot == "off" else self.encode(abc)
        return [EOD] + self.encode(prompt) + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]


class TokenizerHead(nn.Module):
    def __init__(self, din=1024, d=512, layers=8, heads=8, vocab=CODEC_SIZE, window=HEAD_WINDOW):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, : x.shape[1]])))


def _rebuild_rotary(model: nn.Module) -> int:
    """MERT2's RotaryEmbedding keeps ``inv_freq`` as a non-persistent buffer; transformers'
    meta-device loading materializes it uninitialized (zeros on a fresh GPU, garbage or NaN
    otherwise). Recompute it from the module's own formula and drop the cos/sin cache."""
    n = 0
    for m in model.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "head_dim") and hasattr(m, "base"):
            inv = 1.0 / (m.base ** (torch.arange(0, m.head_dim, 2, dtype=torch.float32) / m.head_dim))
            m.inv_freq = inv.to(device=m.inv_freq.device)
            for attr, val in (("_cos", None), ("_sin", None), ("_sequence_length", 0), ("_cache_device", None)):
                if hasattr(m, attr):
                    setattr(m, attr, val)
            n += 1
    return n


class SemanticTokenizer(nn.Module):
    """Waveform -> per-frame codec ids (0..32767) at 25 Hz."""

    def __init__(self, head_path: str, mert_repo: str = MERT_REPO):
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel

        self.processor = AutoFeatureExtractor.from_pretrained(mert_repo, trust_remote_code=True)
        self.mert = AutoModel.from_pretrained(mert_repo, trust_remote_code=True).eval()
        self.mert.requires_grad_(False)
        _rebuild_rotary(self.mert)
        self.head = TokenizerHead()
        ck = torch.load(head_path, map_location="cpu", weights_only=False)
        self.head.load_state_dict(ck["model"])
        self.head.eval().requires_grad_(False)

    @property
    def device(self):
        return self.head.head.weight.device

    @torch.no_grad()
    def mert_features(self, mono24: torch.Tensor) -> torch.Tensor:
        """mono24 [samples] at 24 kHz -> [T25, 1024] fp32 layer-20 features."""
        device = self.device
        chunk = MERT_SAMPLE_RATE * 30
        chunks = [mono24[s : s + chunk] for s in range(0, mono24.shape[0], chunk)]
        chunks = [c for c in chunks if c.shape[0] >= MERT_SAMPLE_RATE]
        full = [c for c in chunks if c.shape[0] == chunk]
        tail = [c for c in chunks if c.shape[0] < chunk]
        feats = []
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for group in ([full] if full else []) + [[c] for c in tail]:
                inp = self.processor([c.cpu().numpy() for c in group], sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                out = self.mert(**inp, output_hidden_states=True)
                feats.append(out.hidden_states[MERT_LAYER].reshape(-1, 1024))
        h = torch.cat(feats, 0).float()
        t25 = int(round(mono24.shape[0] / MERT_SAMPLE_RATE * 25))
        return F.interpolate(h.T[None], size=t25, mode="linear", align_corners=False)[0].T

    @torch.no_grad()
    def tokens_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """[T, 1024] -> [T] long. Windowed inference with edge trimming."""
        x = feats.float()
        x = (x - x.mean(0)) / (x.std(0) + 1e-5)
        total = x.shape[0]
        win = HEAD_WINDOW
        out = torch.zeros(total, dtype=torch.long, device=x.device)
        starts = list(range(0, max(1, total - win + 1), win // 2))
        if starts[-1] + win < total:
            starts.append(max(0, total - win))
        device = self.device
        for s0 in starts:
            xw = x[s0 : s0 + win]
            n = xw.shape[0]
            if n < win:
                xw = F.pad(xw, (0, 0, 0, win - n))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = self.head(xw[None].to(device))[0, :n].float().argmax(-1)
            lo = s0 + (0 if s0 == 0 else win // 4)
            hi = s0 + n - (0 if s0 + n >= total else win // 4)
            out[lo:hi] = pred[lo - s0 : hi - s0].to(out.device)
        return out

    @torch.no_grad()
    def tokenize(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """waveform [C, samples] -> codec ids [T] (long, no vocab offset)."""
        import torchaudio

        mono = waveform.float().mean(0) if waveform.dim() == 2 else waveform.float()
        if sample_rate != MERT_SAMPLE_RATE:
            mono = torchaudio.functional.resample(mono, sample_rate, MERT_SAMPLE_RATE)
        feats = self.mert_features(mono.to(self.device))
        out = self.tokens_from_features(feats)
        if out.unique().numel() < 8:
            print(f"YuE2 tokenizer: degenerate token stream (unique {out.unique().numel()}); MERT features nan {torch.isnan(feats).sum().item()}")
        return out
