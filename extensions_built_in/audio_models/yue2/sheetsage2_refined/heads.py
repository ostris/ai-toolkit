"""Frame heads: per-frame predictions from the frozen encoder, used two ways at inference.

A ``FrameHead`` reads one or more learned mixes of all 25 encoder states (25 Hz, 1024-d; several mixes
let tensions come from the lower blocks while roots come from the upper ones) and predicts, per
frame, the chord's ``root`` / ``bass`` (12 + N) and ``tones`` (12), the ``key`` (24: tonic x
major / minor), where the chord, the key, the beat, the bar or a beat group starts (``BINARY``),
and, for a head trained with bars (``BAR``), facts about the bar the frame lies in: its beats, the
beats left until the next bar head and its groups.  Heads are trained by ``training/train_heads.py``.
A head with ``offset`` also says where the beat and the bar head lie within their 40 ms frame (``OFFSETS``, in
frames from the frame centre), for beat times finer than the frame grid.  A head with ``unit`` also predicts the
denominator of the bar the frame lies in (``bar_unit`` over ``UNITS``).  A head with ``melody`` transcribes the melody
(one line, as SheetSage2-Refined writes it: its vocal and instrument tracks almost never sound together): per frame
and key of ``MELODY_KEYS`` from ``MELODY_LOWEST`` whether a note sounds (``melody``) or starts (``melody_on``), where
the starting note lies from the frame centre (``melody_offset``, frames) and whether the sounding note is sung
(``melody_vocal``: track 0) or played (track 1).  A head with ``line`` reads the melody as one line instead: per frame
the sounding key or a rest (``melody_pitch``, a softmax over the rest and the 88 keys, so one note at a time), whether
a note starts (``melody_on``, one bit), the offset and the vocal call as above, and facts about the note the frame
lies in, read on the spot as the bar facts: its length (``note_steps``, 1..16 sixteenths), the sixteenths left until
it ends (``steps_left``, 1..16) and where in its beat it starts (``onset_pos``, 0..3 sixteenths).  A head with
``extra`` adds a projection of more per-frame inputs (``BeatInformed``: a beat head's bar facts) to its features.

* ``beats.HeadBeats`` tracks beats, bars and meters from a bar head alone (no decoder).
* ``melody.HeadMelody`` writes the melody's notes from a melody head alone (no decoder).
* ``BarFeatures`` adds a projection of a pair of bar heads' bar facts to the decoder memory (the
  decoder was trained with this input).
* ``transcribe_many(rescoring=...)`` rescores the decoder's tokens with heads' predictions
  (see ``engine.BatchDecoder``).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

FRAME_RATE = 25
DETAIL = 745  # root x quality in chord_detail order (12 * quality + root) and N last, as vocab.NO_CHORD
DISTILL = 256  # width of the teacher's chord-head hidden vector (data_tools.tsumugi_teacher)
# Binary outputs in head order; a head trained before "group" existed has only the first four.
BINARY = ("chord_boundary", "key_boundary", "beat", "downbeat", "group")
BAR = {"bar_beats": 16, "beats_left": 16, "bar_groups": 8}  # classes of the bar outputs
BAR_FEATURES = sum(BAR.values()) + 3  # bar facts + beat, downbeat, group
OFFSETS = ("beat_offset", "downbeat_offset")  # where the beat lies from the frame centre, in frames (optional)
UNITS = (2, 4, 8, 16)  # a meter's denominator: classes of the optional ``bar_unit`` output
MELODY_LOWEST, MELODY_KEYS = 21, 88  # the melody outputs' keys: MIDI 21 (A0) .. 108 (C8)
MELODY_SIGMOID = ("melody", "melody_on", "melody_vocal")  # the optional melody outputs read as probabilities
LINE = {"melody_pitch": 1 + MELODY_KEYS, "note_steps": 16, "steps_left": 16, "onset_pos": 4}  # softmax outputs of ``line``
CHUNK = (1500, 375)  # GRU on 60 s pieces with 15 s context either side (the heads learnt from 60-90 s crops)


class FrameHead(nn.Module):
    def __init__(
        self, states: int = 25, width: int = 1024, hidden: int = 384, binary: int = len(BINARY), bar: bool = False,
        mixes: int = 1, rnn_layers: int = 2, detail: bool = False, distill: bool = False, offset: bool = False,
        unit: bool = False, melody: bool = False, line: bool = False, extra: int = 0,
    ):  # fmt: skip
        super().__init__()
        self.binary_names = BINARY[:binary]
        self.bar = nn.Linear(hidden, sum(BAR.values())) if bar else None
        self.offset = nn.Linear(hidden, len(OFFSETS)) if offset else None
        self.unit = nn.Linear(hidden, len(UNITS)) if unit else None  # the bar's denominator
        # per key: sounding, starting; then the onset's offset and the vocal logit
        self.melody = nn.Linear(hidden, 2 * MELODY_KEYS + 2) if melody else None
        # the line's softmax outputs, then its onset, offset and vocal logits
        self.line = nn.Linear(hidden, sum(LINE.values()) + 3) if line else None
        self.extra = nn.Linear(extra, hidden) if extra else None
        if self.extra is not None:  # starts out as a head without the extra inputs
            nn.init.zeros_(self.extra.weight)
            nn.init.zeros_(self.extra.bias)
        if mixes == 1:
            self.mix = nn.Parameter(torch.zeros(states))
        else:  # mix k starts out leaning towards its own depth (low, ..., high blocks)
            depth = torch.arange(states, dtype=torch.float32)
            centres = torch.linspace(0, states - 1, mixes)
            self.mix = nn.Parameter(-(((depth[None] - centres[:, None]) / (states / mixes)) ** 2))
        self.norm = nn.LayerNorm(width)
        self.inp = nn.Linear(width * mixes, hidden)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 9, padding=4, groups=hidden // 32) for _ in range(2))
        self.rnn = nn.GRU(hidden, hidden // 2, num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.root = nn.Linear(hidden, 13)
        self.bass = nn.Linear(hidden, 13)
        self.tones = nn.Linear(hidden, 12)
        self.key = nn.Linear(hidden, 24)
        self.binary = nn.Linear(hidden, binary)
        self.detail = nn.Linear(hidden, DETAIL) if detail else None  # root x quality (tensions included)
        self.distill = nn.Linear(hidden, DISTILL) if distill else None  # the teacher's hidden vector (training only)

    def forward(self, states: list[Tensor], chunk: tuple[int, int] | None = None, extra: Tensor | None = None) -> dict[str, Tensor]:
        """Per-frame logits (``extra`` ``[B, frames, n]``: the inputs of a head with ``extra``).  With ``chunk=(core, context)`` the GRU runs on pieces of ``core`` frames with
        ``context`` frames either side, batched together: a GRU steps through its frames one by one, so a
        12-minute clip in one piece is slow, and the heads learnt from crops of 60-90 s anyway."""

        weights = torch.softmax(self.mix, -1).reshape(-1, len(states))
        mixed = [self.norm(sum(w * s.float() for w, s in zip(row, states))) for row in weights]
        x = F.gelu(self.inp(torch.cat(mixed, -1)))
        if self.extra is not None:
            x = x + self.extra(extra.to(x.dtype))
        for conv in self.convs:
            x = x + F.gelu(conv(x.transpose(1, 2)).transpose(1, 2))
        x = x.float()  # under autocast a bf16 input sends the GRU down a path ~15x slower than fp32
        x = self._rnn_chunked(x, *chunk) if chunk and x.shape[1] > sum(chunk) + chunk[1] else self.rnn(x)[0]
        out = {"root": self.root(x), "bass": self.bass(x), "tones": self.tones(x), "key": self.key(x)}
        if self.detail is not None:
            out["detail"] = self.detail(x)
        if self.distill is not None:
            out["distill"] = self.distill(x)
        binary = self.binary(x)
        out |= {name: binary[..., i] for i, name in enumerate(self.binary_names)}
        if self.bar is not None:
            out |= dict(zip(BAR, self.bar(x).split(list(BAR.values()), dim=-1)))
        if self.unit is not None:
            out["bar_unit"] = self.unit(x)
        if self.offset is not None:
            offsets = self.offset(x)
            out |= {name: offsets[..., i] for i, name in enumerate(OFFSETS)}
        if self.melody is not None:
            melody = self.melody(x)
            out |= {"melody": melody[..., :MELODY_KEYS], "melody_on": melody[..., MELODY_KEYS : 2 * MELODY_KEYS],
                    "melody_offset": melody[..., -2], "melody_vocal": melody[..., -1]}  # fmt: skip
        if self.line is not None:
            line = self.line(x)
            out |= dict(zip(LINE, line[..., :-3].split(list(LINE.values()), dim=-1)))
            out |= {"melody_on": line[..., -3], "melody_offset": line[..., -2], "melody_vocal": line[..., -1]}
        return out

    def _rnn_chunked(self, x: Tensor, core: int, context: int) -> Tensor:
        frames = x.shape[1]
        pieces = [(s, max(s - context, 0), min(s + core + context, frames)) for s in range(0, frames, core)]
        out = torch.empty(*x.shape[:2], self.rnn.hidden_size * 2, device=x.device, dtype=x.dtype)
        by_length: dict[int, list[tuple[int, int, int]]] = {}
        for piece in pieces:  # pieces of one length (all inner ones) go through the GRU together
            by_length.setdefault(piece[2] - piece[1], []).append(piece)
        for group in by_length.values():
            y, _ = self.rnn(torch.cat([x[:, lo:hi] for _, lo, hi in group]))
            y = y.to(out.dtype).split(len(x))
            for (s, lo, _), part in zip(group, y):
                end = min(s + core, frames)
                out[:, s:end] = part[:, s - lo : end - lo]
        return out


def load_head(path: Path, device="cpu") -> FrameHead:
    """A trained head (frozen, eval mode); a chords-only file leaves the other outputs untrained."""

    state = load_file(str(path), device=str(device))
    binary = state["binary.weight"].shape[0] if "binary.weight" in state else len(BINARY)
    mixes = state["mix"].shape[0] if state["mix"].dim() == 2 else 1
    layers = sum(1 for k in state if k.startswith("rnn.weight_ih_l") and not k.endswith("_reverse"))
    head = FrameHead(
        hidden=state["inp.weight"].shape[0], binary=binary, bar="bar.weight" in state, mixes=mixes, rnn_layers=layers,
        detail="detail.weight" in state, distill="distill.weight" in state, offset="offset.weight" in state,
        unit="unit.weight" in state, melody="melody.weight" in state, line="line.weight" in state,
        extra=state["extra.weight"].shape[1] if "extra.weight" in state else 0,
    ).to(device).eval()
    head.load_state_dict(state, strict=not {"key.weight"} - set(state))
    head.requires_grad_(False)
    return head


def bar_facts(out: dict[str, Tensor]) -> Tensor:
    """``[B, frames, BAR_FEATURES]``: a bar head's bar facts as probabilities (``BarFeatures``)."""

    parts = [torch.softmax(out[name].float(), -1) for name in BAR]
    parts.append(torch.stack([torch.sigmoid(out[n].float()) for n in ("beat", "downbeat", "group")], -1))
    return torch.cat(parts, -1)


class BeatInformed(nn.Module):
    """A head with ``extra`` (``BAR_FEATURES``) given a frozen beat head's bar facts on the same encoder states."""

    def __init__(self, beat: nn.Module, head: FrameHead):
        super().__init__()
        self.beat, self.head = beat, head
        self.beat.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.beat.eval()  # frozen: no dropout
        return self

    def facts(self, states: list[Tensor], chunk: tuple[int, int] | None = None) -> Tensor:
        with torch.no_grad():
            return bar_facts(self.beat(states, chunk))

    def forward(self, states: list[Tensor], chunk: tuple[int, int] | None = None) -> dict[str, Tensor]:
        return self.head(states, chunk, extra=self.facts(states, chunk))


class HeadMean(nn.Module):
    """Several heads as one: their per-frame probabilities averaged, returned as logits
    (log-probabilities of the mean for the softmax outputs, the logit of the mean for the sigmoid ones)."""

    SIGMOID = ("tones", *BINARY, *MELODY_SIGMOID)

    def __init__(self, heads: list[FrameHead]):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, states: list[Tensor], chunk: tuple[int, int] | None = None) -> dict[str, Tensor]:
        outs = [head(states, chunk) for head in self.heads]
        mean = {}
        for name in outs[0]:
            if name in OFFSETS or name == "melody_offset":
                mean[name] = sum(o[name].float() for o in outs) / len(outs)
            elif name in self.SIGMOID:
                p = sum(torch.sigmoid(o[name].float()) for o in outs) / len(outs)
                mean[name] = torch.logit(p.clamp(1e-6, 1 - 1e-6))
            else:
                mean[name] = (sum(torch.softmax(o[name].float(), -1) for o in outs) / len(outs)).clamp_min(1e-9).log()
        return mean


class BarFeatures(nn.Module):
    """What a pair of bar heads says about the bar each frame lies in, added to the decoder memory::

        memory + W [P(beats in the bar) 16, P(beats left until the next bar head) 16,
                    P(groups in the bar) 8, P(beat, bar head, group start) 3]

    ``W`` starts at zero, so a decoder starts out exactly as it was.  The heads are cross-fitted:
    head ``k`` was trained without fold ``k`` of the training songs.  A training song of fold ``k``
    gets head ``k`` (``folds``), everything else (valid and test songs, inference) the mean of both.
    """

    def __init__(self, heads: list[FrameHead], width: int = 512):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        for head in self.heads:
            head.requires_grad_(False)
            head.eval()
        self.projection = nn.Linear(BAR_FEATURES, width)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        for head in self.heads:
            head.eval()  # frozen: no dropout
        return self

    @torch.no_grad()
    def _facts(self, states: list[Tensor], k: int) -> Tensor:
        return bar_facts(self.heads[k](states, CHUNK))

    @torch.no_grad()
    def probabilities(self, states: list[Tensor], folds: list[int | None] | None = None) -> Tensor:
        """``[B, frames, BAR_FEATURES]``: head ``k`` for rows of fold ``k``, the mean of the heads otherwise."""

        pairs = [self._facts(states, k) for k in range(len(self.heads))]
        mean = sum(pairs) / len(pairs)
        if folds is None or all(f is None for f in folds):
            return mean
        rows = [pairs[f][i] if f is not None else mean[i] for i, f in enumerate(folds)]
        return torch.stack(rows)

    def forward(self, memory: Tensor, states: list[Tensor], folds=None, keep: Tensor | None = None) -> Tensor:
        """``memory`` plus the projected bar facts; rows with ``keep`` False get none."""

        features = self.probabilities(states, folds).to(self.projection.weight.dtype)
        if keep is not None:
            features = features * keep.to(features.dtype)[:, None, None]
        frames = min(memory.shape[1], features.shape[1])
        added = self.projection(features[:, :frames]).to(memory.dtype)
        return torch.cat((memory[:, :frames] + added, memory[:, frames:]), 1)

    def save_projection(self, path: Path) -> None:
        state = {f"projection.{k}": v.detach().float().cpu().contiguous() for k, v in self.projection.state_dict().items()}
        save_file(state, str(path))

    def load_projection(self, path: Path) -> None:
        with safe_open(str(path), "pt") as handle:
            state = {k.removeprefix("projection."): handle.get_tensor(k) for k in handle.keys() if k.startswith("projection.")}
        self.projection.load_state_dict(state)
