"""Sections (phrases with their labels) of whole songs without the decoder.

The section model reads per-beat inputs of the frozen encoder, the beat tracker and the melody head
(``beat_features``: the encoder states averaged per beat over three groups of blocks, the melody line
folded onto pitch classes per sixteenth, how much of the beat is sung), turns them into one vector per bar of the
tracked grid (a few convolutions over the beats, then the mean over each bar's beats) and runs a Transformer over all
the bars of the song, so that a phrase can be compared with every other one (a chorus sounds like the other choruses).
Optionally the bars' attention is told how alike two bars are (``similarity``: cosine similarities of their melody and
of their pooled states, weighted per head).  Per bar it says:

* ``label``: the section label (``LABELS``, SheetSage2's that it writes: silence, intro, verse, pre-chorus, chorus,
  bridge, interlude, outro);
* ``boundary``: whether a phrase starts on this bar (SheetSage2 writes a section on every phrase, mostly 8 bars:
  verse, verse, pre-chorus, chorus, chorus);
* ``phrase``: the bars since the phrase started (0..15, the last class 15 or more).

``segments`` turns them into phrases with a semi-Markov Viterbi: a phrase's score is its bars' label log-probability,
the boundary's log-probability where it starts and the log-probability of no boundary inside, a prior on its length
given its label and a prior on its label given the one before (``Priors``, counted on the training songs).

A model folder names its section model in ``config.json`` (the file holds the model, its ``config`` and its priors)::

    "section_tracker": {"head": "heads/section-s3.safetensors",
                        "decoding": {"boundary_weight": 8.0, "length_weight": 1.0, "transition_weight": 0.0}}

``HeadSections`` runs the encoder once per window for the beat head, the melody head and the per-beat states, tracks
the beats (``beats.HeadBeats``), pools the inputs (``beat_features``) and decodes the phrases.

    python -m sheetsage2_refined.sections song.mp3 [more.wav ...] [--model REPO_OR_FOLDER] [--out-dir DIR]

writes ``<out-dir>/<song>.sections.json``: the beat tracker's dict (``beats``, ``downbeats``, ``bars``, ``meters``) and
``"sections": [{"start": s, "end": s, "label": "chorus", "bars": n}, ...]`` (a section starts on a bar of the tracked
grid; the last ends with the song).

From Python::

    from sheetsage2_refined.sections import HeadSections

    sections = HeadSections.load("models/sheetsage2-refined")  # SheetSage2's encoder + the beat, melody and section heads
    result = sections.transcribe("song.mp3")                    # the dict above

The model ``section-s3`` (``training.train_sections --similarity --shuffle-prob 0.5``) learnt SheetSage2-Refined's
own sections on 16k songs (the songs whose sections look sound).  It agrees with them on 0.96 of the phrase starts
(dev / valid; test 0.97) and 0.94 of the bars' labels.  On SongFormBench-HarmonixSet (200 songs, seven classes) it
scores label accuracy 0.812 and boundary F 0.682 / 0.835 (0.5 / 3 s): above SheetSage2-Refined's decoder (0.811,
0.671 / 0.823) and level with SheetSage2's (0.813, 0.680 / 0.830).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .beats import HeadBeats, encode_windows, load_encoder
from .heads import CHUNK, FRAME_RATE, MELODY_KEYS, MELODY_LOWEST, FrameHead, load_head
from .pipeline import DEFAULT_MODEL, model_folder
from .transcribe import autocast, load_waveform

LABELS = ("silence", "intro", "verse", "pre-chorus", "chorus", "bridge", "interlude", "outro")
BLOCKS = ((0, 9), (9, 17), (17, 25))  # encoder states [lo, hi) averaged into the per-beat states: low, middle, high
GROUPS, WIDTH, SIXTEENTHS, CLASSES = len(BLOCKS), 1024, 4, 13  # the per-beat inputs (``beat_features``)
PHRASE = 16  # classes of ``phrase``
MAX_BAR_BEATS = 16
LONGEST = 64  # the longest phrase in bars that ``segments`` considers


def bar_layout(beats: np.ndarray, downbeats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each beat's bar (beats before the first downbeat make a bar of their own) and each bar's first beat."""

    index = np.searchsorted(downbeats, beats + 1e-3, side="right") - 1
    index = index - index.min()
    starts = np.flatnonzero(np.diff(np.append(-1, index)))
    return index, starts


class ConvBlock(nn.Module):
    def __init__(self, width: int, kernel: int = 9, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depth = nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width)
        self.point = nn.Linear(width, 2 * width)
        self.out = nn.Linear(2 * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        y = self.depth((self.norm(x) * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(self.out(F.gelu(self.point(y))))


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        b, n, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(b, n, w)))
        return x + self.drop(self.ff(self.norm2(x)))


class SectionModel(nn.Module):
    def __init__(self, width: int = 256, heads: int = 4, beat_layers: int = 2, bar_layers: int = 4, relative: int = 64,
                 similarity: bool = False, melody: bool = True, dropout: float = 0.1):  # fmt: skip
        super().__init__()
        self.config = {"width": width, "heads": heads, "beat_layers": beat_layers, "bar_layers": bar_layers,
                       "relative": relative, "similarity": similarity, "melody": melody, "dropout": dropout}  # fmt: skip
        self.states = nn.ModuleList(nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, width)) for _ in range(GROUPS))
        self.melody = nn.Linear(SIXTEENTHS * CLASSES + 1, width) if melody else None  # + the sung share
        self.beat_position = nn.Embedding(MAX_BAR_BEATS, width)  # the beat's place in its bar
        self.beat_layers = nn.ModuleList(ConvBlock(width, dropout=dropout) for _ in range(beat_layers))
        self.bar_beats = nn.Embedding(MAX_BAR_BEATS + 1, width)  # beats in the bar
        self.song_position = nn.Linear(8, width)  # where the bar lies in the song (Fourier features of 0..1)
        self.bar_layers = nn.ModuleList(AttentionBlock(width, heads, dropout) for _ in range(bar_layers))
        self.relative = relative
        self.distance = nn.Parameter(torch.zeros(bar_layers, heads, 2 * relative + 1))  # per layer and head, by bar distance
        self.similar = nn.Parameter(torch.zeros(bar_layers, heads, 2)) if similarity else None  # melody, states
        self.norm = nn.LayerNorm(width)
        self.label = nn.Linear(width, len(LABELS))
        self.boundary = nn.Linear(width, 1)
        self.phrase = nn.Linear(width, PHRASE)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """``batch``: ``states`` [B, T, groups, 1024], ``melody`` [B, T, 4, 13], ``vocal`` [B, T], ``beat_bar`` [B, T]
        (the beat's bar, -1 for padding), ``beat_position`` [B, T], ``bar_mask`` [B, N] (bool), ``bar_beats`` [B, N],
        ``bar_melody`` [B, N, 16 * 12] (the pitch classes of the bar's first four beats' sixteenths).  Per bar
        logits ``label`` [B, N, labels], ``boundary`` [B, N], ``phrase`` [B, N, 16]."""

        y = self.encode(batch)
        return {"label": self.label(y), "boundary": self.boundary(y)[..., 0], "phrase": self.phrase(y)}

    def beat_inputs(self, batch: dict[str, Tensor]) -> Tensor:
        states = batch["states"].float()
        x = sum(project(states[:, :, g]) for g, project in enumerate(self.states))
        if self.melody is not None:
            x = x + self.melody(torch.cat([batch["melody"].float().flatten(2), batch["vocal"].float()[..., None]], -1))
        return x

    def bar_inputs(self, batch: dict[str, Tensor]) -> Tensor | float:
        """Per-bar inputs added to the pooled beats (none here)."""

        return 0.0

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        """One vector per bar ``[B, N, width]`` (``forward``'s batch)."""

        x = self.beat_inputs(batch)
        beat_mask = (batch["beat_bar"] >= 0).float()
        x = x + self.beat_position(batch["beat_position"].clamp(0, MAX_BAR_BEATS - 1))
        for layer in self.beat_layers:
            x = layer(x, beat_mask)
        bars = batch["bar_mask"].shape[1]
        member = (batch["beat_bar"][:, None, :] == torch.arange(bars, device=x.device)[None, :, None]).float()
        pooled = member @ x / member.sum(-1, keepdim=True).clamp(min=1)
        mask = batch["bar_mask"]
        count = mask.sum(1, keepdim=True).clamp(min=1)
        where = torch.arange(bars, device=x.device)[None] / count  # 0..1 over the song's bars
        fourier = torch.cat([f(math.pi * k * where[..., None]) for k in (1, 2, 4, 8) for f in (torch.sin, torch.cos)], -1)
        y = pooled + self.bar_beats(batch["bar_beats"].clamp(0, MAX_BAR_BEATS)) + self.song_position(fourier)
        y = y + self.bar_inputs(batch)
        offsets = torch.arange(bars, device=x.device)
        distance = (offsets[None, :] - offsets[:, None]).clamp(-self.relative, self.relative) + self.relative
        padding = torch.where(mask, 0.0, -1e9)[:, None, None, :]  # [B, 1, 1, N]
        similar = None
        if self.similar is not None:
            similar = torch.stack([cosine(batch["bar_melody"].float()), cosine(pooled.detach())], 1)  # [B, 2, N, N]
        for i, layer in enumerate(self.bar_layers):
            bias = self.distance[i][:, distance][None] + padding  # [B, H, N, N]
            if similar is not None:
                bias = bias + torch.einsum("hk,bkij->bhij", self.similar[i], similar)
            y = layer(y, bias)
        return self.norm(y)


def cosine(x: Tensor) -> Tensor:
    x = F.normalize(x, dim=-1)
    return x @ x.transpose(1, 2)


def bar_melody(melody: np.ndarray, beat_bar: np.ndarray, bars: int) -> np.ndarray:
    """Per bar, the 12 pitch classes of the sixteenths of its first four beats (``[bars, 16 * 12]``)."""

    out = np.zeros((bars, 4, SIXTEENTHS, 12), dtype=np.float32)
    first = np.searchsorted(beat_bar, np.arange(bars))
    for k in range(4):
        beat = first + k
        ok = (beat < len(beat_bar)) & (beat_bar[np.minimum(beat, len(beat_bar) - 1)] == np.arange(bars))
        out[ok, k] = melody[beat[ok], :, :12]
    return out.reshape(bars, -1)


def model_inputs(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One song's model inputs (unbatched) from its section features."""

    beat_bar, starts = bar_layout(features["beats"], features["downbeats"])
    bars = len(starts)
    position = np.arange(len(beat_bar)) - starts[beat_bar]
    melody = features["melody"].astype(np.float32)
    return {
        "states": features["states"], "melody": melody, "vocal": features["vocal"].astype(np.float32),
        "beat_bar": beat_bar, "beat_position": position, "bar_mask": np.ones(bars, dtype=bool),
        "bar_beats": np.bincount(beat_bar, minlength=bars), "bar_melody": bar_melody(melody, beat_bar, bars),
    }  # fmt: skip


def collate(items: list[dict[str, np.ndarray]]) -> dict[str, Tensor]:
    """Pad songs to the longest (beats with -1 bars, bars masked)."""

    beats = max(len(i["beat_bar"]) for i in items)
    bars = max(len(i["bar_mask"]) for i in items)
    out = {}
    for name in items[0]:
        size = bars if name.startswith("bar") or name in ("label", "boundary", "phrase", "loss_mask") else beats
        fill = -1 if name in ("beat_bar", "label", "phrase") else 0
        rows = []
        for item in items:
            value = torch.as_tensor(np.asarray(item[name]))
            pad = [0, 0] * (value.dim() - 1) + [0, size - len(value)]
            rows.append(F.pad(value, pad, value=fill) if value.dtype != torch.bool else F.pad(value, pad, value=False))
        out[name] = torch.stack(rows)
    return out


@dataclass
class Priors:
    """log P(first label), log P(label | previous label), log P(length in bars | label) (1..LONGEST), log P(last label)."""

    first: np.ndarray  # [labels]
    transition: np.ndarray  # [labels, labels]
    length: np.ndarray  # [labels, LONGEST + 1] (index 0 unused)
    last: np.ndarray  # [labels]

    @classmethod
    def count(cls, songs: list[tuple[np.ndarray, np.ndarray]], smoothing: float = 1.0) -> "Priors":
        """From the songs' bar labels and boundaries (``[bars]`` each)."""

        n = len(LABELS)
        first, last, trans = np.full(n, smoothing), np.full(n, smoothing), np.full((n, n), smoothing)
        length = np.full((n, LONGEST + 1), smoothing / 10)
        for labels, boundary in songs:
            phrases = phrase_list(labels, boundary)
            first[phrases[0][2]] += 1
            last[phrases[-1][2]] += 1
            for (_, _, a), (_, _, b) in zip(phrases[:-1], phrases[1:]):
                trans[a, b] += 1
            for start, end, label in phrases:
                length[label, min(end - start, LONGEST)] += 1
        length[:, 0] = 0
        norm = lambda x: np.log(x / x.sum(-1, keepdims=True))  # noqa: E731
        with np.errstate(divide="ignore"):
            return cls(norm(first), norm(trans), norm(length), norm(last))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"first": self.first, "transition": self.transition, "length": self.length, "last": self.last}


def phrase_list(labels: np.ndarray, boundary: np.ndarray) -> list[tuple[int, int, int]]:
    """``(first bar, end bar, label)`` of the phrases: a phrase starts on bar 0, on each boundary and where the label
    changes; its label is its bars' commonest."""

    starts = sorted({0, *np.flatnonzero(boundary).tolist(), *(np.flatnonzero(np.diff(labels)) + 1).tolist()})
    ends = [*starts[1:], len(labels)]
    return [(s, e, int(np.bincount(labels[s:e]).argmax())) for s, e in zip(starts, ends)]


def segments(label_logp: np.ndarray, boundary_p: np.ndarray, priors: Priors, boundary_weight: float = 1.0,
             length_weight: float = 1.0, transition_weight: float = 1.0) -> list[tuple[int, int, int]]:  # fmt: skip
    """The likeliest phrases ``(first bar, end bar, label)`` (semi-Markov Viterbi, see the module's docstring) from
    per-bar label log-probabilities ``[bars, labels]`` and boundary probabilities ``[bars]``."""

    bars, n = label_logp.shape
    p = np.clip(boundary_p, 1e-6, 1 - 1e-6)
    start_score, inside = boundary_weight * np.log(p), boundary_weight * np.log1p(-p)
    start_score[0] = 0.0
    cum_label = np.vstack([np.zeros(n), np.cumsum(label_logp, 0)])
    cum_inside = np.append(0.0, np.cumsum(inside))
    best = np.full((bars + 1, n), -np.inf)
    back = np.zeros((bars + 1, n, 2), dtype=np.int64)  # (length, previous label)
    trans = transition_weight * priors.transition
    for end in range(1, bars + 1):
        for length in range(1, min(LONGEST, end) + 1):
            start = end - length
            score = (cum_label[end] - cum_label[start] + start_score[start] + cum_inside[end] - cum_inside[start + 1]
                     + length_weight * priors.length[:, length])  # fmt: skip
            if start == 0:
                total, previous = score + transition_weight * priors.first, np.full(n, -1)
            else:
                options = best[start][:, None] + trans  # [previous, label]
                previous = options.argmax(0)
                total = score + options[previous, np.arange(n)]
            better = total > best[end]
            best[end][better] = total[better]
            back[end][better] = np.stack([np.full(n, length), previous], 1)[better]
    label = int(np.argmax(best[bars] + transition_weight * priors.last))
    out, end = [], bars
    while end > 0:
        length, previous = back[end][label]
        out.append((end - length, end, label))
        end, label = end - length, previous
    return out[::-1]


# ------------------------------------------------------------------ per-beat inputs


def beat_spans(beats: np.ndarray, frames: int) -> np.ndarray:
    """Each beat's first and last-plus-one frame (``[B, 2]``), at least one frame each."""

    step = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    edges = np.append(beats, beats[-1] + step) * FRAME_RATE - 0.5  # a frame belongs to a beat if its centre does
    lo = np.clip(np.ceil(edges[:-1]), 0, frames - 1).astype(int)
    hi = np.clip(np.ceil(edges[1:]), 1, frames).astype(int)
    return np.stack([lo, np.maximum(hi, lo + 1)], 1)


def pool(values: Tensor, spans: np.ndarray) -> Tensor:
    """Means of ``values`` [frames, ...] over each span (cumulative sums: one pass)."""

    total = torch.cat([torch.zeros_like(values[:1]), values.cumsum(0)])
    lo, hi = torch.as_tensor(spans[:, 0]), torch.as_tensor(spans[:, 1])
    shape = (-1,) + (1,) * (values.dim() - 1)
    return (total[hi] - total[lo]) / (hi - lo).reshape(shape).to(values.dtype)


def pitch_classes(pitch: np.ndarray) -> np.ndarray:
    """``melody_pitch`` [frames, 89] (rest, MIDI 21..108) folded onto [frames, 13] (12 pitch classes, rest)."""

    out = np.zeros((len(pitch), CLASSES), dtype=np.float32)
    classes = (np.arange(MELODY_KEYS) + MELODY_LOWEST) % 12
    np.add.at(out.T, classes, pitch[:, 1:].T)
    out[:, 12] = pitch[:, 0]
    return out


def melody_slots(probs: np.ndarray, beats: np.ndarray, frames: int) -> np.ndarray:
    """Per beat and sixteenth, the mean of ``probs`` [frames, n] over the slot's frames (``[B, 4, n]``)."""

    step = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    edges = np.append(beats, beats[-1] + step)
    slots = edges[:-1, None] + np.diff(edges)[:, None] * np.arange(SIXTEENTHS + 1)[None] / SIXTEENTHS
    flat = np.stack([slots[:, :-1].ravel(), slots[:, 1:].ravel()], 1) * FRAME_RATE - 0.5
    lo = np.clip(np.ceil(flat[:, 0]), 0, frames - 1).astype(int)
    hi = np.maximum(np.clip(np.ceil(flat[:, 1]), 1, frames).astype(int), lo + 1)
    pooled = pool(torch.from_numpy(probs), np.stack([lo, hi], 1)).numpy()
    return pooled.reshape(len(beats), SIXTEENTHS, -1)


def beat_features(frames: dict, beats: np.ndarray) -> dict[str, np.ndarray]:
    """The section model's per-beat inputs from ``HeadSections.frames`` on the tracked ``beats`` (at least two):
    ``states`` [B, groups, 1024] (the encoder states averaged over the beat's frames, those whose centre lies from the
    beat to the next; the last beat lasts one median beat), ``melody`` [B, 4, 13] (the melody line folded onto the
    pitch classes and the rest, averaged over each sixteenth of the beat) and ``vocal`` [B] (how much of the beat a
    sung note sounds), float16 as the training features."""

    states = frames["states"]
    spans = beat_spans(beats, len(states))
    return {
        "states": pool(states.float(), spans).numpy().astype(np.float16),
        "melody": melody_slots(pitch_classes(frames["pitch"]), beats, len(states)).astype(np.float16),
        "vocal": pool(torch.from_numpy(frames["sung"]), spans).numpy().astype(np.float16),
    }


# ------------------------------------------------------------------ whole songs


def section_files(config: dict) -> list[str]:
    """The files of a model folder the sections need: SheetSage2's weights, the beat head, the melody head and the
    section model (``config.json``'s ``section_tracker.head``)."""

    if not (config.get("section_tracker") or {}).get("head"):
        raise ValueError("this model has no section model (config.json has no section_tracker.head)")
    if not (config.get("beat_tracker") or {}).get("head") or not (config.get("melody_tracker") or {}).get("head"):
        raise ValueError("the sections need the model's beat_tracker and melody_tracker heads")
    heads = (config["beat_tracker"]["head"], config["melody_tracker"]["head"], config["section_tracker"]["head"])
    return [config["base_weights"], *heads]


def save_sections(path: Path, model: SectionModel, priors: Priors) -> None:
    """One file with the model's weights, its ``config`` (metadata) and its priors (``priors.*``)."""

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    state |= {f"priors.{k}": torch.from_numpy(np.asarray(v, dtype=np.float32).copy()) for k, v in priors.arrays().items()}
    save_file(state, str(path), {"config": json.dumps(model.config)})


def load_sections(path: Path, device: str | torch.device = "cpu") -> tuple[SectionModel, Priors]:
    with safe_open(str(path), "pt") as handle:
        config = json.loads(handle.metadata()["config"])
    state = load_file(str(path))
    names = ("first", "transition", "length", "last")
    priors = Priors(**{k: state.pop(f"priors.{k}").numpy().astype(np.float64) for k in names})
    model = SectionModel(**config)
    model.load_state_dict(state)
    return model.to(device).eval(), priors


class HeadSections:
    def __init__(self, beats: HeadBeats, melody: FrameHead, model: SectionModel, priors: Priors, device: torch.device,
                 **decoding):  # fmt: skip
        self.beats, self.melody, self.model, self.priors, self.device = beats, melody, model, priors, device
        self.decoding = decoding  # ``segments``' weights

    @classmethod
    def load(cls, model: Path | str = DEFAULT_MODEL, device: str | None = None, revision: str | None = None,
             head: Path | None = None, **decoding) -> "HeadSections":  # fmt: skip
        """The encoder, the beat, melody and section heads of a model folder or Hugging Face repo id (at
        ``revision``; from the Hub only those files are downloaded) on ``device`` (default: CUDA if there is one).
        ``head`` (a section model file) replaces the model's; ``decoding`` goes to ``segments`` over the model's
        ``section_tracker.decoding``."""

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        folder = model_folder(model, revision, section_files)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        encoder = load_encoder(folder / config["base_weights"], device)
        beats = HeadBeats(encoder, load_head(folder / config["beat_tracker"]["head"], device), device)
        melody = load_head(folder / config["melody_tracker"]["head"], device)
        if melody.line is None:
            raise ValueError("the sections need a line melody head (melody_pitch)")
        sections, priors = load_sections(Path(head) if head else folder / config["section_tracker"]["head"], device)
        decoding = (config["section_tracker"].get("decoding") or {}) | decoding
        return cls(beats, melody, sections, priors, device, **decoding)

    @torch.no_grad()
    def frames(self, wave: Tensor, chord: FrameHead | None = None) -> dict:
        """One encoder pass per window for a 24 kHz waveform: the beat head's probabilities (``probs``, as
        ``HeadBeats.probabilities``), the states averaged over ``BLOCKS`` (``states`` [frames, groups, 1024] float16),
        the melody line (``pitch`` [frames, 89]) and how sung each frame is (``sung`` [frames]); with a ``chord`` head
        also its chord (``chord`` [frames, 38]: the root and bass softmaxes, 12 + N, and the 12 tones' sigmoids)."""

        encoder, beat_head = self.beats.encoder, self.beats.head
        groups, beat_out, melody_out, chord_out = [], [], [], []
        for states, keep in encode_windows(encoder, wave, self.device):  # as beats.frame_outputs, keeping the states
            groups.append(torch.stack([torch.stack(states[lo:hi]).float().mean(0)[0] for lo, hi in BLOCKS], 1)[:keep].half())
            beat_out.append({k: v[:, :keep] for k, v in beat_head(states, CHUNK).items()})
            melody_out.append({k: v[:, :keep] for k, v in self.melody(states, CHUNK).items()})
            if chord is not None:
                c = chord(states, CHUNK)
                chord_out.append(torch.cat([torch.softmax(c["root"].float(), -1), torch.softmax(c["bass"].float(), -1),
                                            torch.sigmoid(c["tones"].float())], -1)[0, :keep].cpu())  # fmt: skip
            del states
        out = {k: torch.cat([o[k] for o in beat_out], 1)[0].float().cpu() for k in beat_out[0]}
        probs = {k: torch.sigmoid(out[k]).numpy() for k in ("beat", "downbeat")}
        probs |= {k: torch.softmax(out[k], -1).numpy() for k in ("bar_beats", "beats_left", "bar_unit") if k in out}
        probs |= {k: out[k].numpy() for k in ("beat_offset", "downbeat_offset") if k in out}
        mel = {k: torch.cat([o[k] for o in melody_out], 1)[0].float() for k in ("melody_pitch", "melody_vocal")}
        pitch = torch.softmax(mel["melody_pitch"], -1)
        sung = (1 - pitch[:, 0]) * torch.sigmoid(mel["melody_vocal"])
        result = {"probs": probs, "states": torch.cat(groups).cpu(), "pitch": pitch.cpu().numpy(), "sung": sung.cpu().numpy(),
                  "duration": len(wave) / encoder.config.sample_rate}  # fmt: skip
        if chord is not None:
            result["chord"] = torch.cat(chord_out)
        return result

    @torch.no_grad()
    def sections(self, frames: dict) -> dict:
        """The tracked beats (``HeadBeats.track_probabilities``' dict) and their ``sections``."""

        tracked = self.beats.track_probabilities(frames["probs"])
        beats, downbeats = np.asarray(tracked["beats"]), np.asarray(tracked["downbeats"])
        if len(beats) < 2:
            return tracked | {"sections": []}
        inputs = model_inputs({"beats": beats, "downbeats": downbeats, **beat_features(frames, beats)})
        batch = {k: v.to(self.device) for k, v in collate([inputs]).items()}
        with autocast(self.device):
            out = self.model(batch)
        label_logp = torch.log_softmax(out["label"][0].float(), -1).cpu().numpy()
        boundary_p = torch.sigmoid(out["boundary"][0].float()).cpu().numpy()
        bar_start = beats[bar_layout(beats, downbeats)[1]]
        edges = np.append(bar_start, max(float(frames["duration"]), float(beats[-1])))
        found = [{"start": round(float(edges[s]), 4), "end": round(float(edges[e]), 4), "label": LABELS[label], "bars": int(e - s)}
                 for s, e, label in segments(label_logp, boundary_p, self.priors, **self.decoding)]  # fmt: skip
        return tracked | {"sections": found}

    def transcribe(self, audio: Path | str) -> dict:
        wave = load_waveform(Path(audio), self.beats.encoder.config.sample_rate)
        duration = len(wave) / self.beats.encoder.config.sample_rate
        return {"audio": str(audio), "duration": duration} | self.sections(self.frames(wave))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local model folder")
    parser.add_argument("--revision", help="the Hugging Face repo's branch, tag or commit")
    parser.add_argument("--head", type=Path, help="a section model file (default: the model's)")
    decoding = parser.add_argument_group("decoding (default: the model's section_tracker.decoding, else segments')")
    decoding.add_argument("--boundary-weight", type=float, help="weight of the boundary probabilities")
    decoding.add_argument("--length-weight", type=float, help="weight of the phrase-length prior")
    decoding.add_argument("--transition-weight", type=float, help="weight of the label-transition prior")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "sections")
    parser.add_argument("--memory-fraction", type=float, default=0.6, help="cap on this process's share of GPU memory")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device).index or 0)
    names = ("boundary_weight", "length_weight", "transition_weight")
    given = {k: getattr(args, k) for k in names if getattr(args, k) is not None}
    tracker = HeadSections.load(args.model, args.device, args.revision, args.head, **given)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for audio in args.audio:
        started = time.perf_counter()
        result = tracker.transcribe(audio)
        path = args.out_dir / f"{audio.stem}.sections.json"
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        labels = " ".join(s["label"] for s in result["sections"])
        print(f"{audio.name}: {len(result['sections'])} sections ({labels}), "
              f"{time.perf_counter() - started:.1f} s -> {path}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
