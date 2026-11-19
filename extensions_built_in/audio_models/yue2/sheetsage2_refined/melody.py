"""The melody of whole songs without the decoder: a melody head (``heads.FrameHead`` with ``melody``) on SheetSage2's
frozen encoder, then ``notes`` over its per-frame probabilities (Onsets and Frames for one line) or, for a head with
``line`` (one softmax over the rest and the keys, one onset bit, the note facts), ``line_notes`` (a Viterbi path).  A
head trained with a beat head's bar facts (``extra``) gets them from a beat head (``heads.BeatInformed``).

A model folder names its melody head in ``config.json``::

    "melody_tracker": {"head": "heads/melody-m4.safetensors",          # the melody head
                       "decoding": {"onset_weight": 2.0, "start_penalty": 0.5},  # line_notes' (or notes') options
                       "beat_head": "heads/beat-v7u.safetensors"}      # only for a head that reads bar facts

    python -m sheetsage2_refined.melody song.mp3 [more.wav ...] [--model REPO_OR_FOLDER] [--head HEAD] [--out-dir DIR]

writes ``<out-dir>/<song>.melody.json``::

    {"audio": ..., "duration": s, "notes": [{"start": s, "end": s, "pitch": midi, "track": 0 sung / 1 played}, ...]}

and, from a ``line`` head, each note's ``steps`` (its length in sixteenths) and ``pos`` (the sixteenth of its beat it
starts on).

From Python::

    from sheetsage2_refined.melody import HeadMelody

    melody = HeadMelody.load("models/sheetsage2-refined")  # SheetSage2's encoder + the model's melody head
    result = melody.transcribe("song.mp3")                  # the dict above

The head ``melody-m4`` (``models/sheetsage2-refined/heads/melody-m4.safetensors``) is a ``line`` head trained by
``training.train_melody --line --onset-pos-weight 2 --hook-sources vocal`` on SheetSage2-Refined's own melody (20k
songs) and the Hooktheory clips of sung melodies (the instrument clips' labels are doubtful).  Hooktheory test note F
0.713 (vocal clips 0.796; the decoder 0.682 / 0.770); it agrees with the decoder's melody on 0.851 of the notes of
the valid songs.  Note times are the head's: an onset frame plus its sub-frame offset, not yet placed on a beat grid.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .beats import frame_outputs, load_encoder
from .encoder import MERTEncoder
from .heads import FRAME_RATE, LINE, MELODY_LOWEST, BeatInformed, FrameHead, HeadMean, load_head
from .pipeline import DEFAULT_MODEL, model_folder
from .transcribe import load_waveform


LINE_OPTIONS = ("onset_weight", "start_penalty", "interval_weight", "octave_penalty", "end_weight")  # line_notes'
KEY_OPTIONS = ("onset", "frame")  # notes'


def melody_files(config: dict) -> list[str]:
    """The files of a model folder the melody needs: SheetSage2's weights, the melody head (``config.json``'s
    ``melody_tracker.head``) and, for a head that reads bar facts, its beat head (``melody_tracker.beat_head``)."""

    if not (config.get("melody_tracker") or {}).get("head"):
        raise ValueError("this model has no melody head (config.json has no melody_tracker.head)")
    return [config["base_weights"], config["melody_tracker"]["head"], *beat_head_files(config)]


def beat_head_files(config: dict) -> list[str]:
    return [config["melody_tracker"]["beat_head"]] if (config.get("melody_tracker") or {}).get("beat_head") else []


def notes(probs: dict[str, np.ndarray], onset: float = 0.5, frame: float = 0.5) -> list[dict]:
    """Notes from a melody head's probabilities: a note starts on each frame where the likeliest onset reaches
    ``onset`` and is a peak in time, at the key that is likeliest to start and sound there; it lasts while that key
    sounds (``frame``), at most until the next onset.  Its start is the frame centre plus the head's offset; its
    track is the head's vocal call averaged over the note."""

    on, sounding = probs["melody_on"], probs["melody"]
    strength, count = on.max(1), len(on)
    peaks = [
        t for t in range(count)
        if strength[t] >= onset and (t == 0 or strength[t] > strength[t - 1]) and (t + 1 == count or strength[t] >= strength[t + 1])
    ]  # fmt: skip
    starts = [(t + 0.5 + float(probs["melody_offset"][t])) / FRAME_RATE for t in peaks]
    out = []
    for i, t in enumerate(peaks):
        stop = peaks[i + 1] if i + 1 < len(peaks) else count
        key = int(np.argmax(on[t] + sounding[t : t + 2].mean(0)))
        end = t + 1
        while end < stop and sounding[end, key] >= frame:
            end += 1
        finish = min(end / FRAME_RATE, starts[i + 1]) if end == stop and i + 1 < len(peaks) else end / FRAME_RATE
        vocal = float(probs["melody_vocal"][t:end].mean())
        out.append({"start": round(starts[i], 4), "end": round(max(finish, starts[i] + 0.01), 4),
                    "pitch": MELODY_LOWEST + key, "track": 0 if vocal >= 0.5 else 1})  # fmt: skip
    return out


def line_notes(probs: dict[str, np.ndarray], onset_weight: float = 1.0, start_penalty: float = 0.0,
               interval_weight: float = 0.0, octave_penalty: float = 0.0, end_weight: float = 0.0) -> list[dict]:
    """Notes from a ``line`` head's probabilities: the likeliest path through the rest and the 88 keys, scored per
    frame by the log-probability of its state plus ``onset_weight`` times the log-probability of the onset bit (a
    note starts where the path enters a key, or stays on its key with an onset more likely than not), less
    ``start_penalty`` for each note started (fewer notes split by a key flickering inside one).  A move from one key
    straight to another costs ``interval_weight`` per octave of the leap and ``octave_penalty`` more for a leap of
    whole octaves (a key heard in the wrong octave); a note that ends (before a rest or another note) adds
    ``end_weight`` times the log-probability that its last frame has one sixteenth left (``steps_left``), so a note
    is not cut where the head says it goes on.  A note's start is its first frame's centre plus the head's offset;
    its track, length in sixteenths and position in its beat are the head's calls averaged over its frames."""

    pitch = np.log(np.clip(probs["melody_pitch"], 1e-9, 1.0))
    on = np.clip(probs["melody_on"], 1e-6, 1 - 1e-6)
    begin, keep = onset_weight * np.log(on) - start_penalty, onset_weight * np.log1p(-on)
    count, states = pitch.shape
    index = np.arange(states)
    ending = np.zeros(count)
    if end_weight and "steps_left" in probs:
        ending = end_weight * np.log(np.clip(probs["steps_left"][:, 0], 1e-4, 1.0))
    keys = states - 1
    leap = np.abs(index[1:, None] - index[None, 1:])
    move = -interval_weight * leap / 12 - octave_penalty * ((leap % 12 == 0) & (leap > 0))  # held key -> new key
    # States: 0 rest, 1..keys a note's first frame on a key, keys+1.. the note held.  A note lasts at least two
    # frames unless a rest follows it (two onsets on neighbouring frames are one onset smeared, not two notes).
    came = np.zeros((count, 1 + 2 * keys), dtype=np.int16)  # the state before
    rest, first, held = pitch[0, 0] + keep[0], pitch[0, 1:] + begin[0], np.full(keys, -np.inf)
    for t in range(1, count):
        end = ending[t - 1]  # the note of a key ends at t - 1
        options = np.array([rest, first.max() + end, held.max() + end])
        pick = int(options.argmax())
        came[t, 0] = (0, 1 + int(first.argmax()), 1 + keys + int(held.argmax()))[pick]
        moves = (held + end)[:, None] + move
        source = moves.argmax(0)
        struck = np.where(moves[source, index[1:] - 1] > rest, 1 + keys + source, 0)  # from a held key, or a rest
        came[t, 1 : 1 + keys] = struck
        came[t, 1 + keys :] = np.where(first >= held, 1 + index[1:] - 1, 1 + keys + index[1:] - 1)
        rest, first, held = (
            options[pick] + keep[t] + pitch[t, 0],
            np.maximum(moves[source, index[1:] - 1], rest) + begin[t] + pitch[t, 1:],
            np.maximum(first, held) + keep[t] + pitch[t, 1:],
        )
    path, starts = np.zeros(count, dtype=np.int64), np.zeros(count, dtype=bool)
    state = int(np.argmax([rest, first.max(), held.max()]))
    state = (0, 1 + int(first.argmax()), 1 + keys + int(held.argmax()))[state]
    for t in range(count - 1, -1, -1):
        path[t] = 0 if state == 0 else 1 + (state - 1) % keys
        starts[t] = 1 <= state <= keys
        state = int(came[t, state])
    onsets = [t for t in range(count) if path[t] > 0 and (starts[t] or path[t] != path[t - 1])]
    times = [(t + 0.5 + float(probs["melody_offset"][t])) / FRAME_RATE for t in onsets]
    out = []
    for i, t in enumerate(onsets):
        end = t + 1
        while end < count and path[end] == path[t] and not starts[end]:
            end += 1
        finish = times[i + 1] if i + 1 < len(onsets) and onsets[i + 1] == end else end / FRAME_RATE
        note = {"start": round(times[i], 4), "end": round(max(finish, times[i] + 0.01), 4),
                "pitch": MELODY_LOWEST + int(path[t]) - 1, "track": 0 if probs["melody_vocal"][t:end].mean() >= 0.5 else 1}  # fmt: skip
        if "note_steps" in probs:
            note["steps"] = int(probs["note_steps"][t:end].mean(0).argmax()) + 1
            note["pos"] = int(probs["onset_pos"][t:end].mean(0).argmax())
        out.append(note)
    return out


class HeadMelody:
    def __init__(self, encoder: MERTEncoder, head: FrameHead | HeadMean | BeatInformed, device: torch.device,
                 line: bool = False, **options):  # fmt: skip
        self.encoder, self.head, self.device = encoder, head, device
        self.line = line  # a ``line`` head (decoded by ``line_notes``)
        self.options = options  # ``line_notes``' (LINE_OPTIONS) or ``notes``' (KEY_OPTIONS) settings

    @classmethod
    def load(cls, model: Path | str = DEFAULT_MODEL, heads: list[Path] | None = None, device: str | None = None,
             revision: str | None = None, weights: Path | None = None, beat_head: Path | None = None,
             **options) -> "HeadMelody":  # fmt: skip
        """The encoder and the melody head of a model folder or Hugging Face repo id (at ``revision``) on ``device``
        (default: CUDA if there is one).  ``weights`` (SheetSage2's file), ``heads`` (melody heads; several: their
        mean) and ``beat_head`` (for a head that reads bar facts) replace the model's; ``options`` go to the
        decoding over the model's ``melody_tracker.decoding`` (``line_notes``: LINE_OPTIONS; ``notes``: KEY_OPTIONS)."""

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if weights is None or not heads or beat_head is None:
            files = melody_files if not heads else lambda config: [config["base_weights"], *beat_head_files(config)]
            folder = model_folder(model, revision, files)
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            weights = weights or folder / config["base_weights"]
            heads = heads or [folder / config["melody_tracker"]["head"]]
            beat_head = beat_head or next((folder / f for f in beat_head_files(config)), None)
            options = ((config.get("melody_tracker") or {}).get("decoding") or {}) | options
        loaded = [load_head(Path(p), device) for p in heads]
        if any(h.melody is None and h.line is None for h in loaded):
            raise ValueError("the melody needs a melody head (training.train_melody)")
        if len({h.line is None for h in loaded}) > 1 or len({h.extra is None for h in loaded}) > 1:
            raise ValueError("melody heads of different kinds")
        head = loaded[0] if len(loaded) == 1 else HeadMean(loaded).eval()
        if loaded[0].extra is not None:
            if len(loaded) > 1 or beat_head is None:
                raise ValueError("a head that reads bar facts needs a beat head (and goes alone)")
            head = BeatInformed(load_head(Path(beat_head), device), loaded[0]).eval()
        return cls(load_encoder(weights, device), head, device, loaded[0].line is not None, **options)

    def probabilities(self, wave: torch.Tensor) -> dict[str, np.ndarray]:
        """The head's 25 Hz melody outputs for a 24 kHz waveform as probabilities, melody_offset [frames] in frames:
        melody, melody_on [frames, 88] and melody_vocal [frames]; from a ``line`` head melody_pitch [frames, 89],
        melody_on [frames] and the note facts [frames, classes] instead."""

        out = frame_outputs(self.encoder, self.head, wave, self.device)
        probs = {k: torch.sigmoid(out[k]).numpy() for k in ("melody", "melody_on", "melody_vocal") if k in out}
        probs |= {k: torch.softmax(out[k], -1).numpy() for k in LINE if k in out}
        return probs | {"melody_offset": out["melody_offset"].numpy()}

    def notes(self, probs: dict[str, np.ndarray]) -> list[dict]:
        names = LINE_OPTIONS if self.line else KEY_OPTIONS
        options = {k: v for k, v in self.options.items() if k in names}
        return line_notes(probs, **options) if self.line else notes(probs, **options)

    def transcribe(self, audio: Path | str) -> dict:
        wave = load_waveform(Path(audio), self.encoder.config.sample_rate)
        found = self.notes(self.probabilities(wave))
        return {"audio": str(audio), "duration": len(wave) / self.encoder.config.sample_rate, "notes": found}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local model folder")
    parser.add_argument("--revision", help="the Hugging Face repo's branch, tag or commit")
    parser.add_argument("--weights", type=Path, help="SheetSage2's weights (default: the model's)")
    parser.add_argument("--head", type=Path, action="append", help="melody head(s); several: their mean (default: the model's)")
    parser.add_argument("--beat-head", type=Path, help="the beat head whose bar facts the melody head reads (default: the model's)")
    decoding = parser.add_argument_group("decoding (default: the model's melody_tracker.decoding, else the functions')")
    decoding.add_argument("--onset", type=float, help="an 88-key head: onset probability that starts a note")
    decoding.add_argument("--frame", type=float, help="an 88-key head: frame probability that keeps a note sounding")
    decoding.add_argument("--onset-weight", type=float, help="a line head: weight of the onset bit in the path")
    decoding.add_argument("--start-penalty", type=float, help="a line head: cost of starting a note")
    decoding.add_argument("--interval-weight", type=float, help="a line head: cost per octave of a leap")
    decoding.add_argument("--octave-penalty", type=float, help="a line head: extra cost of a leap of octaves")
    decoding.add_argument("--end-weight", type=float, help="a line head: weight of the note-ends-here fact")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "melody")
    parser.add_argument("--memory-fraction", type=float, default=0.6, help="cap on this process's share of GPU memory")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device).index or 0)
    given = {k: getattr(args, k) for k in (*LINE_OPTIONS, *KEY_OPTIONS) if getattr(args, k) is not None}
    melody = HeadMelody.load(args.model, args.head, args.device, args.revision, args.weights, args.beat_head, **given)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for audio in args.audio:
        started = time.perf_counter()
        result = melody.transcribe(audio)
        path = args.out_dir / f"{audio.stem}.melody.json"
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(f"{audio.name}: {len(result['notes'])} notes, {time.perf_counter() - started:.1f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
