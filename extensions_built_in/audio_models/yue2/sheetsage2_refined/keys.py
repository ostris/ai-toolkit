"""Keys with modes of whole songs without the decoder.

The key model is the section model (``sections.SectionModel``: per-beat inputs, a few convolutions over the beats,
the mean over each bar's beats, a Transformer over all the bars of the song) with one more per-beat input, the chord
head's chord (``chord``: its root and bass softmaxes and its tones' sigmoids, averaged over the beat), optionally
one per-bar input, the section model's view of the bar (``bar_section``: its label and phrase softmaxes, boundary
probability and decoded section starts, ``data_tools.key_sections``), and per bar:

* ``key``: the bar's key, one of ``KEYS`` (12 tonics x the 9 modes of ``theory.MODES``);
* ``boundary``: whether the key changes on this bar (keys change on a downbeat).

With ``local`` a second, local reading (``LocalKeys``) scores each bar from the chords and melody of the bars around
it only, read from each of the 12 candidate tonics with one shared network (the same progression or resolution means
the same in every key: transposition-equivariant).  It has its own loss; the model's ``key`` is the whole-song
reading's log-probabilities plus ``local_weight`` times the local one's (the whole-song reading leans on the encoder
states, which already carry the surrounding key, and smooths brief changes away).

``decode`` turns them into key segments with a Viterbi over the bars: a bar's score is its key's log-probability, a
change costs the boundary's log-probability and staying the log-probability of no boundary (``boundary_weight``).
With ``min_bars`` > 1 it is a semi-Markov Viterbi over whole segments of at least ``min_bars`` bars (the first and
the last segment may be shorter): the boundary head often fires on two or three neighbouring bars around one change,
which the bar Viterbi follows with one-bar back-and-forth keys (E G E G).

A model folder names its key model in ``config.json`` (the file holds the model and its ``config``)::

    "key_tracker": {"head": "heads/key-k10.safetensors",
                    "decoding": {"local_weight": 2.0, "boundary_weight": 4.0, "mode_tau": 0.5, "min_bars": 4}}

``local_weight`` replaces the file's own, ``mode_tau`` adds the decoder's mode bias (``pipeline.mode_bias``, from the
folder's ``mode_counts``).  ``HeadKeys`` runs the encoder once per window for the beat head, the melody head, the
chord head (``rescoring.quality``) and the per-beat states, tracks the beats (``beats.HeadBeats``), pools the inputs
per beat as the training features (``data_tools.key_features``) and decodes the keys.

    python -m sheetsage2_refined.keys song.mp3 [more.wav ...] [--model REPO_OR_FOLDER] [--out-dir DIR]

writes ``<out-dir>/<song>.keys.json``: the beat tracker's dict (``beats``, ``downbeats``, ``bars``, ``meters``) and
``"keys": [{"start": s, "end": s, "key": "A:major", "bars": n}, ...]`` (a key starts on a bar of the tracked grid; the
first from 0 s, the last ends with the song).

From Python::

    from sheetsage2_refined.keys import HeadKeys

    keys = HeadKeys.load("models/sheetsage2-refined")  # SheetSage2's encoder + the beat, melody, chord and key heads
    result = keys.transcribe("song.mp3")              # the dict above

The model ``key-k10`` (``training.train_keys --local --ema 0.999``) learnt from the hand-labelled chord songs (keys
reviewed by listening), Hooktheory clips, synthetic songs with modes and key changes and pitch-shifted chord songs
with spliced brief changes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor, nn

from .beats import HeadBeats, load_encoder
from .heads import FrameHead, load_head
from .pipeline import DEFAULT_MODEL, mode_bias, model_folder
from .sections import (MAX_BAR_BEATS, ConvBlock, HeadSections, SectionModel, bar_layout, beat_features, beat_spans, collate,
                       model_inputs, pool)  # fmt: skip
from .theory import MODES, SHARP_NAMES
from .transcribe import load_waveform

KEYS = tuple((tonic, mode) for mode in MODES for tonic in range(12))  # index: 12 * MODES.index(mode) + tonic
CHORD = 38  # the chord head's root (13), bass (13) and tones (12) per beat
CONTEXT = 2 * CHORD  # the previous and the next chord of each beat (``chord_context``)
SECTION = 26  # the section model's label (8), boundary (1), phrase (16) and decoded start (1) per bar


def key_index(tonic: int, mode: str) -> int:
    return 12 * MODES.index(mode) + tonic % 12


def key_name(index: int) -> str:
    tonic, mode = KEYS[index]
    return f"{SHARP_NAMES[tonic]}:{mode}"


def chord_context(chord: np.ndarray) -> np.ndarray:
    """Per beat ``[T, 76]``: the chord before the beat's chord and the one after it (a chord: the beats with one
    likeliest root and one set of tones over 0.5; zeros at the song's ends)."""

    signature = np.concatenate([chord[:, :13].argmax(1)[:, None], chord[:, 26:38] > 0.5], 1)
    start = np.append(True, np.any(signature[1:] != signature[:-1], axis=1))
    run = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    last = np.append(first[1:] - 1, len(chord) - 1)
    out = np.zeros((len(chord), CONTEXT), dtype=np.float32)
    has_prev, has_next = run > 0, run + 1 < len(first)
    out[has_prev, :CHORD] = chord[last[run[has_prev] - 1]]
    out[has_next, CHORD:] = chord[first[run[has_next] + 1]]
    return out


def rolled(x: Tensor, perm: Tensor) -> Tensor:
    """``x`` [B, T, ..., 12] read from each candidate tonic (``perm`` [12, 12], row t: t, t + 1, ...): [B, 12, T, ..., 12]."""

    return x[..., perm].movedim(-2, 1)


class LocalKeys(nn.Module):
    """Per bar key logits ``[B, N, keys]`` from the chords and melody of the bars around it (a few convolutions over
    the beats, the mean over each bar's beats, one over the bars), the pitch classes rolled so that each candidate
    tonic is 0 and one network scoring the modes per candidate.  Per beat it reads the chord head's chord, the chords
    before and after it, the root (and bass) moves into and out of it, and the melody's pitch classes."""

    FEATURES = 3 * CHORD + 3 * 144 + 4 * 13 + 1

    def __init__(self, width: int = 128, layers: int = 3, bar_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(self.FEATURES, width)
        self.beat_position = nn.Embedding(MAX_BAR_BEATS, width)
        self.beat_layers = nn.ModuleList(ConvBlock(width, 5, dropout) for _ in range(layers))
        self.bar_layers = nn.ModuleList(ConvBlock(width, 5, dropout) for _ in range(bar_layers))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(MODES)))
        self.register_buffer("perm", torch.stack([(torch.arange(12) + t) % 12 for t in range(12)]), persistent=False)

    def chord(self, chord: Tensor) -> list[Tensor]:
        """The chord's root, bass and tones rolled per candidate, the no-chord flags as they are."""

        flags = lambda i: chord[:, None, :, i:i + 1].expand(-1, 12, -1, -1)  # noqa: E731
        return [rolled(chord[..., 0:12], self.perm), flags(12), rolled(chord[..., 13:25], self.perm), flags(25),
                rolled(chord[..., 26:38], self.perm)]  # fmt: skip

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        chord, context = batch["chord"].float(), batch["chord_context"].float()
        cur, prev, nxt = self.chord(chord), self.chord(context[..., :CHORD]), self.chord(context[..., CHORD:])
        pair = lambda a, b: (a[..., :, None] * b[..., None, :]).flatten(-2)  # noqa: E731
        melody = batch["melody"].float()
        every = lambda v: v[:, None].expand(-1, 12, *v.shape[1:])  # noqa: E731
        x = torch.cat([*cur, *prev, *nxt, pair(prev[0], cur[0]), pair(cur[0], nxt[0]), pair(prev[2], cur[2]),
                       rolled(melody[..., :12], self.perm).flatten(-2), every(melody[..., 12]),
                       every(batch["vocal"].float()[..., None])], -1)  # fmt: skip
        b, r = x.shape[:2]
        position = self.beat_position(batch["beat_position"].clamp(0, MAX_BAR_BEATS - 1))
        x = self.input(x.flatten(0, 1)) + position.repeat_interleave(r, 0)
        beat_mask = (batch["beat_bar"] >= 0).float().repeat_interleave(r, 0)
        for layer in self.beat_layers:
            x = layer(x, beat_mask)
        bars = batch["bar_mask"].shape[1]
        member = (batch["beat_bar"][:, None, :] == torch.arange(bars, device=x.device)[None, :, None]).to(x.dtype)
        y = torch.stack([member[i] @ x[i * r:(i + 1) * r] for i in range(b)]).flatten(0, 1)  # [B * 12, N, width] sums
        y = y / member.sum(-1, keepdim=True).clamp(min=1).repeat_interleave(r, 0)
        bar_mask = batch["bar_mask"].float().repeat_interleave(r, 0)
        for layer in self.bar_layers:
            y = layer(y, bar_mask)
        logits = self.head(y).view(b, r, bars, len(MODES))  # [B, tonic, N, mode]
        return logits.permute(0, 2, 3, 1).flatten(2)  # index 12 * mode + tonic


class KeyModel(SectionModel):
    def __init__(self, width: int = 256, heads: int = 4, beat_layers: int = 2, bar_layers: int = 4, relative: int = 64,
                 melody: bool = True, chord: bool = True, dropout: float = 0.1, sections: bool = False,
                 local: bool = False, local_weight: float = 1.0):  # fmt: skip
        super().__init__(width, heads, beat_layers, bar_layers, relative, False, melody, dropout)
        del self.config["similarity"], self.label, self.phrase
        self.config["chord"] = chord
        self.config["sections"] = sections
        self.config["local"], self.config["local_weight"] = local, local_weight
        self.local = LocalKeys(dropout=dropout) if local else None
        self.chord = nn.Linear(CHORD, width) if chord else None
        self.section = nn.Linear(SECTION, width) if sections else None
        self.key = nn.Linear(width, len(KEYS))

    def beat_inputs(self, batch: dict[str, Tensor]) -> Tensor:
        x = super().beat_inputs(batch)
        if self.chord is not None:
            x = x + self.chord(batch["chord"].float())
        return x

    def bar_inputs(self, batch: dict[str, Tensor]) -> Tensor | float:
        return 0.0 if self.section is None else self.section(batch["bar_section"].float())

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """``batch``: ``SectionModel.forward``'s, ``chord`` [B, T, 38] and with ``sections`` ``bar_section``
        [B, N, 26], with ``local`` ``chord_context`` [B, T, 76].  Per bar logits ``key`` [B, N, keys] and
        ``boundary`` [B, N]; with ``local`` also the two readings' logits ``whole`` and ``local`` (``key``: their
        log-probabilities, the local one times ``local_weight``)."""

        y = self.encode(batch)
        out = {"key": self.key(y), "boundary": self.boundary(y)[..., 0]}
        if self.local is not None:
            local = self.local(batch)
            key = torch.log_softmax(out["key"].float(), -1) + self.config["local_weight"] * torch.log_softmax(local.float(), -1)
            out |= {"whole": out["key"], "local": local, "key": key}
        return out


def key_inputs(features: dict[str, np.ndarray], section: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """One song's model inputs (unbatched) from its key features (``data_tools.key_features``) and optionally its
    sections (``data_tools.key_sections``)."""

    chord = features["chord"].astype(np.float32)
    inputs = model_inputs(features) | {"chord": chord, "chord_context": chord_context(chord)}
    return inputs if section is None else inputs | {"bar_section": section.astype(np.float32)}


def decode(key_logp: np.ndarray, boundary_p: np.ndarray, boundary_weight: float = 1.0, min_bars: int = 1) -> np.ndarray:
    """The likeliest key per bar (``[bars]`` indices into ``KEYS``) from per-bar key log-probabilities ``[bars, keys]``
    and change probabilities ``[bars]``; with ``min_bars`` > 1 no key segment but the first and the last is shorter."""

    bars, n = key_logp.shape
    p = np.clip(boundary_p, 1e-6, 1 - 1e-6)
    change, stay = boundary_weight * np.log(p), boundary_weight * np.log1p(-p)
    if min_bars > 1:
        return decode_segments(key_logp, change, stay, min_bars)
    best = key_logp[0].copy()
    back = np.zeros((bars, n), dtype=np.int64)
    for i in range(1, bars):
        order = np.argsort(best)[::-1][:2]  # the best other key: the best, or the second best for the best itself
        other = np.where(np.arange(n) == order[0], order[1], order[0])
        moved = best[other] + change[i]
        kept = best + stay[i]
        back[i] = np.where(kept >= moved, np.arange(n), other)
        best = np.maximum(kept, moved) + key_logp[i]
    out = np.zeros(bars, dtype=np.int64)
    out[-1] = int(best.argmax())
    for i in range(bars - 1, 0, -1):
        out[i - 1] = back[i][out[i]]
    return out


def decode_segments(key_logp: np.ndarray, change: np.ndarray, stay: np.ndarray, min_bars: int) -> np.ndarray:
    """``decode``'s semi-Markov Viterbi: segments ``[s, e)`` of one key, each scoring its bars' key log-probabilities,
    ``stay`` on its bars after the first and ``change`` on its first bar (not the song's); a segment differs from the
    one before and is at least ``min_bars`` long unless it starts or ends the song."""

    bars, n = key_logp.shape
    keys = np.arange(n)
    summed = np.vstack([np.zeros(n), np.cumsum(key_logp, 0)])
    stayed = np.concatenate([[0.0], np.cumsum(stay)])
    best = np.full((bars + 1, n), -np.inf)  # best[e, k]: the best decoding of bars [0, e) ending with key k
    start = np.zeros((bars + 1, n), dtype=np.int64)
    before = np.full((bars + 1, n), -1)
    for e in range(1, bars + 1):
        for s in range(e - 1, -1, -1):
            if e - s < min_bars and 0 < s and e < bars:
                continue
            score = summed[e] - summed[s] + stayed[e] - stayed[s + 1]
            if s == 0:
                total, prev = score, np.full(n, -1)
            else:
                order = np.argsort(best[s])[::-1][:2]  # the best other key, as in ``decode``
                prev = np.where(keys == order[0], order[1], order[0])
                total = best[s][prev] + change[s] + score
            better = total > best[e]
            best[e][better], start[e][better], before[e][better] = total[better], s, prev[better]
    out = np.zeros(bars, dtype=np.int64)
    e, key = bars, int(best[bars].argmax())
    while e > 0:
        s = start[e, key]
        out[s:e] = key
        e, key = s, before[e, key]
    return out


# ------------------------------------------------------------------ whole songs


def key_files(config: dict) -> list[str]:
    """The files of a model folder the keys need: SheetSage2's weights, the beat, melody and chord heads and the key
    model (``config.json``'s ``key_tracker.head``)."""

    if not (config.get("key_tracker") or {}).get("head"):
        raise ValueError("this model has no key model (config.json has no key_tracker.head)")
    if not (config.get("beat_tracker") or {}).get("head") or not (config.get("melody_tracker") or {}).get("head"):
        raise ValueError("the keys need the model's beat_tracker and melody_tracker heads")
    if not ((config.get("rescoring") or {}).get("quality") or {}).get("head"):
        raise ValueError("the keys need the model's chord head (rescoring.quality)")
    heads = (config["beat_tracker"]["head"], config["melody_tracker"]["head"], config["rescoring"]["quality"]["head"],
             config["key_tracker"]["head"])  # fmt: skip
    return [config["base_weights"], *heads]


def load_keys(path: Path, device: str | torch.device = "cpu") -> KeyModel:
    with safe_open(str(path), "pt") as handle:
        config = json.loads(handle.metadata()["config"])
    model = KeyModel(**config)
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval()


class HeadKeys:
    def __init__(self, heads: HeadSections, chord: FrameHead, model: KeyModel, device: torch.device,
                 boundary_weight: float = 1.0, bias: np.ndarray | None = None, min_bars: int = 1):  # fmt: skip
        self.heads, self.chord, self.model, self.device = heads, chord, model, device
        self.boundary_weight, self.bias = boundary_weight, bias  # ``decode``'s weight, the per-key mode bias
        self.min_bars = min_bars  # ``decode``'s shortest key segment

    @classmethod
    def load(cls, model: Path | str = DEFAULT_MODEL, device: str | None = None, revision: str | None = None,
             head: Path | None = None, **decoding) -> "HeadKeys":  # fmt: skip
        """The encoder, the beat, melody, chord and key heads of a model folder or Hugging Face repo id (at
        ``revision``; from the Hub only those files are downloaded) on ``device`` (default: CUDA if there is one).
        ``head`` (a key model file) replaces the model's; ``decoding`` (``local_weight``, ``boundary_weight``,
        ``mode_tau``, ``min_bars``) goes over the model's ``key_tracker.decoding``."""

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        folder = model_folder(model, revision, key_files)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        encoder = load_encoder(folder / config["base_weights"], device)
        beats = HeadBeats(encoder, load_head(folder / config["beat_tracker"]["head"], device), device)
        melody = load_head(folder / config["melody_tracker"]["head"], device)
        if melody.line is None:
            raise ValueError("the keys need a line melody head (melody_pitch)")
        chord = load_head(folder / config["rescoring"]["quality"]["head"], device)
        keys = load_keys(Path(head) if head else folder / config["key_tracker"]["head"], device)
        if keys.config.get("sections"):
            raise ValueError("this key model reads the section model's bars, which HeadKeys does not compute")
        decoding = (config["key_tracker"].get("decoding") or {}) | decoding
        if decoding.get("local_weight") is not None:
            keys.config["local_weight"] = decoding["local_weight"]
        per_mode = mode_bias(decoding.get("mode_tau", 0.0), config.get("mode_counts"))
        bias = None if per_mode is None else np.array([per_mode[MODES.index(mode)] for _, mode in KEYS])
        heads = HeadSections(beats, melody, None, None, device)  # for its encoder pass
        return cls(heads, chord, keys, device, decoding.get("boundary_weight", 1.0), bias, int(decoding.get("min_bars") or 1))

    @torch.no_grad()
    def keys(self, frames: dict) -> dict:
        """The tracked beats (``HeadBeats.track_probabilities``' dict) and their ``keys`` from ``HeadSections.frames``
        with the chord head."""

        tracked = self.heads.beats.track_probabilities(frames["probs"])
        beats, downbeats = np.asarray(tracked["beats"]), np.asarray(tracked["downbeats"])
        if len(beats) < 2:
            return tracked | {"keys": []}
        chord = pool(frames["chord"], beat_spans(beats, len(frames["chord"]))).numpy().astype(np.float16)
        features = {"beats": beats, "downbeats": downbeats, "chord": chord} | beat_features(frames, beats)
        batch = {k: v.to(self.device) for k, v in collate([key_inputs(features)]).items()}
        out = self.model(batch)
        key_logp = torch.log_softmax(out["key"][0].float(), -1).cpu().numpy()
        if self.bias is not None:
            key_logp = key_logp + self.bias
        boundary_p = torch.sigmoid(out["boundary"][0].float()).cpu().numpy()
        bar_start = beats[bar_layout(beats, downbeats)[1]]
        edges = np.append(bar_start, max(float(frames["duration"]), float(beats[-1])))
        edges[0] = 0.0
        found = []
        for i, key in enumerate(decode(key_logp, boundary_p, self.boundary_weight, self.min_bars)):
            if found and found[-1]["key"] == key_name(key):
                found[-1]["end"], found[-1]["bars"] = round(float(edges[i + 1]), 4), found[-1]["bars"] + 1
            else:
                found.append({"start": round(float(edges[i]), 4), "end": round(float(edges[i + 1]), 4), "key": key_name(key), "bars": 1})
        return tracked | {"keys": found}

    def transcribe(self, audio: Path | str) -> dict:
        sample_rate = self.heads.beats.encoder.config.sample_rate
        wave = load_waveform(Path(audio), sample_rate)
        return {"audio": str(audio), "duration": len(wave) / sample_rate} | self.keys(self.heads.frames(wave, self.chord))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local model folder")
    parser.add_argument("--revision", help="the Hugging Face repo's branch, tag or commit")
    parser.add_argument("--head", type=Path, help="a key model file (default: the model's)")
    decoding = parser.add_argument_group("decoding (default: the model's key_tracker.decoding)")
    decoding.add_argument("--local-weight", type=float, help="weight of the local reading")
    decoding.add_argument("--boundary-weight", type=float, help="weight of the key-change probabilities")
    decoding.add_argument("--mode-tau", type=float, help="the mode bias (0: none)")
    decoding.add_argument("--min-bars", type=int, help="the shortest key segment in bars (1: the bar Viterbi)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "keys")
    parser.add_argument("--memory-fraction", type=float, default=0.6, help="cap on this process's share of GPU memory")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device).index or 0)
    names = ("local_weight", "boundary_weight", "mode_tau", "min_bars")
    given = {k: getattr(args, k) for k in names if getattr(args, k) is not None}
    tracker = HeadKeys.load(args.model, args.device, args.revision, args.head, **given)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for audio in args.audio:
        started = time.perf_counter()
        result = tracker.transcribe(audio)
        path = args.out_dir / f"{audio.stem}.keys.json"
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        keys = " ".join(k["key"] for k in result["keys"])
        print(f"{audio.name}: {len(result['keys'])} keys ({keys}), {time.perf_counter() - started:.1f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
