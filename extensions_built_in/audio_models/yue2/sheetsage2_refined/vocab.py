"""SheetSage2 token vocabulary and event (de)serialisation.

A sequence is ``<sos> <prompt>... <out> (<shift>+ field-token*)* <eos>``.

Events sit on a subbeat grid: shift tokens advance the grid position, then the
event's fields follow in the fixed order timestamp, rhythm, structure, key,
chord, melody.  Token ids must match the pretrained embedding rows, so the
block order and sizes below are frozen; new blocks are only ever appended.

Added on top of SheetSage2:

* ``chord_detail``: a chord is ``bass chord``: the bass pitch class, then one
  token for root x quality over the 62 Tsumugi qualities (root and quality are
  one choice, so they cannot disagree).  The N chord is the single token
  ``chord=N``.
* ``key_mode``: 12 tonics x 9 modes (``theory.MODES``) + N.
* ``swing``: reserved for a later swing attribute; never generated yet.
"""

from __future__ import annotations

import bisect
import math

from .theory import MODES, QUALITIES, chord_to_mir_eval

PAD, SOS, EOS, OUT = 0, 1, 2, 3

# (name, mutually exclusive group).  The order is the canonical prompt order.
PROMPTS = (
    ("timestamp", "timestamp"),
    ("downbeat_meter", "rhythm"),
    ("structure", "structure"),
    ("key", "key"),
    ("chord_majmin", "chord"),
    ("chord_full", "chord"),
    ("melody_vocal", "melody"),
    ("melody_full", "melody"),
    ("chord_detail", "chord"),
    ("key_mode", "key"),
)
PROMPT_NAMES = tuple(name for name, _ in PROMPTS)
_GROUP = dict(PROMPTS)
FULL_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full")
PROMPT_CAPACITY = 256

FIELDS = ("timestamp", "rhythm", "structure", "key", "chord", "melody")
FIELD_OF_TYPE = {
    "time": "timestamp",
    "meter": "rhythm",
    "eighth_position": "rhythm",
    "structure": "structure",
    "key": "key",
    "chord_majmin": "chord",
    "chord_full": "chord",
    "chord_bass": "chord",
    "chord_detail": "chord",
    "key_mode": "key",
    "pitch": "melody",
    "duration": "melody",
}

SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
METERS = tuple((numerator, denominator) for numerator in range(1, 33) for denominator in (1, 2, 4, 8, 16, 32))
STRUCTURE_LABELS = (
    "silence",
    "intro",
    "outro",
    "verse",
    "chorus",
    "bridge",
    "pre-chorus",
    "post-chorus",
    "interlude",
    "fade-out",
    "loop",
    "rap",
    "preshot",
    "irregular",
    "instrumental",
    "intro and verse",
    "pre-chorus and chorus",
    "verse and pre-chorus",
    "solo",
    "theme",
    "development",
    "variation",
    "pre-outro",
)
MAJMIN_CHORDS = ("N", *(f"{root}:maj" for root in SHARPS), *(f"{root}:min" for root in SHARPS))
_FULL_QUALITIES = (
    "maj",
    "min",
    "dim",
    "aug",
    "maj7",
    "min7",
    "7",
    "hdim7",
    "dim7",
    "minmaj7",
    "sus2",
    "sus4",
    "sus4(b7)",
    "maj6",
    "min6",
)
_FULL_INVERSIONS = {
    "maj": ("/2", "/3", "/5"),
    "min": ("/2", "/b3", "/5"),
    "maj7": ("/3", "/5", "/7"),
    "min7": ("/b3", "/5", "/b7"),
    "7": ("/3", "/5", "/b7"),
}
FULL_CHORDS = (
    "N",
    *(
        f"{root}:{quality}{inversion}"
        for quality in _FULL_QUALITIES
        for root in SHARPS
        for inversion in (*_FULL_INVERSIONS.get(quality, ()), "")
    ),
)
# Melody durations in subbeats.
DURATION_STEPS = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)
MAX_SHIFT = 256
NO_CHORD = 12 * len(QUALITIES)  # chord_detail offset of the "N" chord
NO_KEY = len(MODES) * 12  # key_mode offset of "N"
SWING_CAPACITY = 32


class Vocab:
    def __init__(self, window_seconds: float = 300.0, time_hz: int = 100):
        self.time_hz = time_hz
        blocks = (
            ("special", 4),
            ("prompt", PROMPT_CAPACITY),
            ("shift", MAX_SHIFT + 1),
            ("time", round(window_seconds * time_hz)),
            ("meter", len(METERS)),
            ("eighth_position", 256),
            ("structure", len(STRUCTURE_LABELS)),
            ("key", 24),
            ("chord_majmin", len(MAJMIN_CHORDS)),
            ("chord_full", len(FULL_CHORDS)),
            ("pitch", 256),
            ("duration", len(DURATION_STEPS)),
            # Appended after the pretrained vocabulary (31678 tokens).
            ("chord_bass", 12),
            ("chord_detail", NO_CHORD + 1),
            ("key_mode", NO_KEY + 1),
            ("swing", SWING_CAPACITY),
        )
        self.start: dict[str, int] = {}
        self.end: dict[str, int] = {}
        offset = 0
        for name, size in blocks:
            self.start[name], self.end[name] = offset, offset + size
            offset += size
        self.size = offset
        self._starts = [self.start[name] for name, _ in blocks]
        self._names = [name for name, _ in blocks]

    # ------------------------------------------------------------------ types
    def token_type(self, token: int) -> str:
        if not 0 <= token < self.size:
            raise ValueError(f"token {token} is outside the vocabulary ({self.size})")
        name = self._names[bisect.bisect_right(self._starts, token) - 1]
        if name == "special":
            return ("pad", "sos", "eos", "out")[token]
        if name == "prompt" and token - self.start["prompt"] >= len(PROMPTS):
            raise ValueError(f"token {token} is an unused prompt slot")
        return name

    def offset(self, token: int, name: str) -> int:
        return token - self.start[name]

    # ---------------------------------------------------------------- prompts
    def prompt_prefix(self, prompts) -> list[int]:
        names = sorted(set(prompts), key=lambda name: (FIELDS.index(_GROUP[name]), PROMPT_NAMES.index(name)))
        groups = [group for name, group in PROMPTS if name in names]
        if not names or len(groups) != len(set(groups)):
            raise ValueError(f"invalid prompt set {prompts!r}")
        return [SOS, *(self.start["prompt"] + PROMPT_NAMES.index(name) for name in names), OUT]

    def shift_tokens(self, shift: int) -> list[int]:
        tokens = []
        while shift > MAX_SHIFT:
            tokens.append(self.start["shift"] + MAX_SHIFT)
            shift -= MAX_SHIFT
        return tokens + [self.start["shift"] + shift]

    def time_token(self, seconds: float) -> int:
        time_id = min(max(round(seconds * self.time_hz), 0), self.end["time"] - self.start["time"] - 1)
        return self.start["time"] + time_id

    # --------------------------------------------------------------- builders
    def meter_token(self, meter: tuple[int, int]) -> int:
        return self.start["meter"] + METERS.index(tuple(meter))

    def eighth_token(self, position: int) -> int:
        return self.start["eighth_position"] + position

    def structure_token(self, label: str) -> int:
        return self.start["structure"] + STRUCTURE_LABELS.index(label)

    def key_token(self, tonic: int, minor: bool) -> int:
        return self.start["key"] + tonic + 12 * minor

    def key_mode_token(self, tonic: int | None, mode: str | None) -> int:
        offset = NO_KEY if tonic is None else MODES.index(mode) * 12 + tonic
        return self.start["key_mode"] + offset

    def chord_detail_tokens(self, root: int | None, quality: str = "", bass: int | None = None) -> list[int]:
        """``root``/``bass`` are pitch classes (``bass=None``: root position); ``root=None`` is the N chord."""

        if root is None:
            return [self.start["chord_detail"] + NO_CHORD]
        return [
            self.start["chord_bass"] + (root if bass is None else bass),
            self.start["chord_detail"] + 12 * QUALITIES.index(quality) + root,
        ]

    @staticmethod
    def detail_chord(offset: int) -> tuple[int, str] | None:
        """``(root, quality)`` of a chord_detail offset, or None for N."""

        return None if offset == NO_CHORD else (offset % 12, QUALITIES[offset // 12])

    def note_tokens(self, pitch: int, track: int, duration_steps: int) -> list[int]:
        """A melody note; the duration is rounded to the nearest bin on a log scale."""

        bin_index = min(
            range(len(DURATION_STEPS)), key=lambda i: abs(math.log(DURATION_STEPS[i] / max(duration_steps, 1)))
        )
        return [self.start["pitch"] + 128 * track + pitch, self.start["duration"] + bin_index]

    # ----------------------------------------------------------------- values
    def field_value(self, field: str, tokens: list[int]):
        first = tokens[0]
        if field == "timestamp":
            return self.offset(first, "time") / self.time_hz
        if field == "rhythm":
            rhythm = {}
            for token in tokens:
                if self.token_type(token) == "meter":
                    rhythm["meter"] = METERS[self.offset(token, "meter")]
                else:
                    rhythm["eighth_position"] = self.offset(token, "eighth_position")
            return rhythm
        if field == "structure":
            return STRUCTURE_LABELS[self.offset(first, "structure")]
        if field == "key":
            key_id = self.offset(first, "key")
            if self.token_type(first) == "key_mode":
                key_id = self.offset(first, "key_mode")
                return "N" if key_id == NO_KEY else f"{SHARPS[key_id % 12]}:{MODES[key_id // 12]}"
            key_id = self.offset(first, "key")
            return f"{SHARPS[key_id % 12]}:{'minor' if key_id >= 12 else 'major'}"
        if field == "chord":
            kind = self.token_type(first)
            if kind == "chord_majmin":
                return MAJMIN_CHORDS[self.offset(first, "chord_majmin")]
            if kind == "chord_full":
                return FULL_CHORDS[self.offset(first, "chord_full")]
            # chord_detail; a missing bass reads as root position, a lone bass as N
            parts = {self.token_type(token): token for token in tokens}
            chord = self.detail_chord(self.offset(parts["chord_detail"], "chord_detail")) if "chord_detail" in parts else None
            if chord is None:
                return "N"
            bass = self.offset(parts["chord_bass"], "chord_bass") if "chord_bass" in parts else chord[0]
            return chord_to_mir_eval(chord[0], chord[1], bass)
        # melody: pitch tokens, each optionally followed by a duration token
        notes, index = [], 0
        while index < len(tokens):
            pitch_id = self.offset(tokens[index], "pitch")
            duration_bin = 0
            if index + 1 < len(tokens) and self.token_type(tokens[index + 1]) == "duration":
                duration_bin = self.offset(tokens[index + 1], "duration")
                index += 2
            else:
                index += 1
            notes.append(
                {
                    "pitch": pitch_id % 128,
                    "track": int(pitch_id >= 128),
                    "duration_bin": duration_bin,
                    "duration_steps": DURATION_STEPS[duration_bin],
                }
            )
        return notes

    def event_values(self, tokens: dict[str, list[int]]) -> dict:
        return {field: self.field_value(field, values) for field, values in tokens.items() if values}

    # -------------------------------------------------------------- sequences
    def decode(self, tokens: list[int]) -> list[dict]:
        """Parse a generated sequence into ``{"subbeat", "tokens", "values"}`` events.

        Events without any field are dropped, matching SheetSage2's lenient decode.
        """

        tokens = [int(token) for token in tokens]
        while tokens and tokens[-1] == PAD:
            tokens.pop()
        if not tokens or tokens[0] != SOS:
            raise ValueError("sequence must begin with <sos>")
        position = tokens.index(OUT, 1) + 1
        events, step = [], 0
        while position < len(tokens) and tokens[position] != EOS:
            if self.token_type(tokens[position]) != "shift":
                raise ValueError(f"event at token index {position} has no subbeat shift")
            while position < len(tokens) and self.token_type(tokens[position]) == "shift":
                step += self.offset(tokens[position], "shift")
                position += 1
            fields: dict[str, list[int]] = {}
            while position < len(tokens) and tokens[position] != EOS and self.token_type(tokens[position]) != "shift":
                token = tokens[position]
                fields.setdefault(FIELD_OF_TYPE[self.token_type(token)], []).append(token)
                position += 1
            if fields:
                fields = {field: fields[field] for field in FIELDS if field in fields}
                events.append({"subbeat": step, "tokens": fields, "values": self.event_values(fields)})
        return events

    def encode(self, events: list[dict], prompts, *, eos: bool = True) -> list[int]:
        """Inverse of :meth:`decode` for events sorted by ``subbeat``."""

        output, previous = self.prompt_prefix(prompts), 0
        for event in events:
            if event["subbeat"] < previous:
                raise ValueError("events must be sorted by subbeat")
            output += self.shift_tokens(event["subbeat"] - previous)
            previous = event["subbeat"]
            for field in FIELDS:
                output += event["tokens"].get(field, [])
        return output + [EOS] if eos else output
