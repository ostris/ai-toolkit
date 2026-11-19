"""The event grammar as tensor operations, so it can run inside a CUDA graph.

Within an event, fields appear in the order timestamp, rhythm, structure, key,
chord, melody; a meter must be followed by an eighth position; a pitch may be
followed by a duration; a detailed chord's bass must be followed by its
root x quality token; at most four shift tokens separate two events.  Only
the token blocks of the requested prompts can be generated.  The state of every
sequence in a batch is a handful of integer tensors, updated with
``torch.where`` so that no step needs the host.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .vocab import EOS, FULL_PROMPTS, NO_CHORD, Vocab

# Token classes.  OTHER (pad, sos, out, prompts, swing and the blocks of
# prompts not requested) is never generated.  KEY is any one-token key and
# CHORD any one-token chord (including the detailed N chord).
TYPES = (
    "eos",
    "shift",
    "time",
    "meter",
    "eighth_position",
    "structure",
    "key",
    "chord",
    "pitch",
    "duration",
    "chord_bass",
    "chord_detail",
)
OTHER = len(TYPES)
(
    EOS_T,
    SHIFT_T,
    TIME_T,
    METER_T,
    EIGHTH_T,
    STRUCTURE_T,
    KEY_T,
    CHORD_T,
    PITCH_T,
    DURATION_T,
    BASS_T,
    DETAIL_T,
) = range(len(TYPES))
# Field index (timestamp 0 ... melody 5) opened by each token class.
FIELD_OF_TYPE = (-1, -1, 0, 1, 1, 2, 3, 4, 5, 5, 4, 4, -1)
# "incomplete" states: what the last token still requires or allows
NONE, AFTER_METER, AFTER_PITCH, AFTER_BASS = range(4)
MAX_SHIFT_RUN = 4

# Token blocks (vocab block, optional sub-range, class) that each prompt enables.
PROMPT_BLOCKS = {
    "timestamp": (("time", None, TIME_T),),
    "downbeat_meter": (("meter", None, METER_T), ("eighth_position", None, EIGHTH_T)),
    "structure": (("structure", None, STRUCTURE_T),),
    "key": (("key", None, KEY_T),),
    "key_mode": (("key_mode", None, KEY_T),),
    "chord_majmin": (("chord_majmin", None, CHORD_T),),
    "chord_full": (("chord_full", None, CHORD_T),),
    "chord_detail": (
        ("chord_bass", None, BASS_T),
        ("chord_detail", (0, NO_CHORD), DETAIL_T),
        ("chord_detail", (NO_CHORD, NO_CHORD + 1), CHORD_T),
    ),
    "melody_vocal": (("pitch", (0, 128), PITCH_T), ("duration", None, DURATION_T)),
    "melody_full": (("pitch", None, PITCH_T), ("duration", None, DURATION_T)),
}

ENDED = ("running", "eos", "stop", "limit")


class TokenGrammar:
    def __init__(self, vocab: Vocab, device: torch.device | str, prompts=FULL_PROMPTS):
        self.prompts = tuple(sorted(set(prompts)))
        vocab.prompt_prefix(prompts)  # validates the prompt set
        token_class = torch.full((vocab.size,), OTHER, dtype=torch.long)
        token_class[EOS] = EOS_T
        token_class[vocab.start["shift"] : vocab.end["shift"]] = SHIFT_T
        for prompt in self.prompts:
            for block, span, kind in PROMPT_BLOCKS[prompt]:
                first, last = span or (0, vocab.end[block] - vocab.start[block])
                token_class[vocab.start[block] + first : vocab.start[block] + last] = kind
        self.token_class = token_class.to(device)
        self.field_of_class = torch.tensor(FIELD_OF_TYPE, device=device)
        # Seconds of every time token (float64, matching the reference stop test).
        seconds = torch.full((vocab.size,), float("-inf"), dtype=torch.float64)
        seconds[vocab.start["time"] : vocab.end["time"]] = (
            torch.arange(vocab.end["time"] - vocab.start["time"], dtype=torch.float64) / vocab.time_hz
        )
        self.time_seconds = seconds.to(device)
        self._cpu = None  # CPU copy for prefixes, built on first use

    @staticmethod
    def new_state(batch: int, device) -> dict[str, Tensor]:
        """State after ``<out>``: inside a shift run, no fields yet."""

        long = lambda value: torch.full((batch,), value, dtype=torch.long, device=device)  # noqa: E731
        return {
            "in_shift": torch.ones(batch, dtype=torch.bool, device=device),
            "shift_run": long(0),
            "payload": long(0),
            "last_field": long(-1),
            "incomplete": long(NONE),
            "finished": torch.zeros(batch, dtype=torch.bool, device=device),
            "ended": long(0),
        }

    def allowed(self, state: dict[str, Tensor]) -> Tensor:
        """``[B, vocab]`` mask of the tokens each sequence may emit next."""

        last, incomplete = state["last_field"], state["incomplete"]
        free = incomplete == NONE
        open_chord = incomplete == AFTER_BASS
        classes = torch.stack(
            [
                (state["payload"] > 0) & ~open_chord,  # eos
                ((state["payload"] > 0) | state["in_shift"]) & (state["shift_run"] < MAX_SHIFT_RUN) & ~open_chord,
                free & (last < 0),  # time
                free & (last < 1),  # meter
                (incomplete == AFTER_METER) | (free & (last < 1)),  # eighth position
                free & (last < 2),  # structure
                free & (last < 3),  # key
                free & (last < 4),  # chord
                (incomplete == AFTER_PITCH) | (free & (last <= 5)),  # pitch
                incomplete == AFTER_PITCH,  # duration
                free & (last < 4),  # chord bass
                open_chord,  # chord root x quality
                torch.zeros_like(free),  # other
            ],
            dim=1,
        )
        return classes[:, self.token_class]

    def update(self, state: dict[str, Tensor], tokens: Tensor, active: Tensor, stop: Tensor, at_limit: Tensor) -> None:
        """Consume ``tokens`` ([B]) for the ``active`` rows, in place.

        A row finishes on ``<eos>``, on a time token at or after its ``stop``
        seconds, or when ``at_limit`` (the decoder context is full).
        """

        kind = self.token_class[tokens]
        is_shift = kind == SHIFT_T
        is_field = (kind != SHIFT_T) & (kind != EOS_T)
        new_event = is_shift & ~state["in_shift"] & (state["payload"] > 0)

        payload = torch.where(new_event, 0, state["payload"])
        payload = torch.where(is_field, payload + 1, payload)
        last = torch.where(new_event, -1, state["last_field"])
        last = torch.where(is_field, self.field_of_class[kind], last)
        incomplete = torch.where(new_event, NONE, state["incomplete"])
        follows = (
            (kind == METER_T) * AFTER_METER
            + (kind == PITCH_T) * AFTER_PITCH
            + (kind == BASS_T) * AFTER_BASS
        )
        incomplete = torch.where(is_field, follows, incomplete)
        in_shift = torch.where(is_shift, True, torch.where(is_field, False, state["in_shift"]))
        shift_run = torch.where(is_shift, state["shift_run"] + 1, torch.where(is_field, 0, state["shift_run"]))

        eos = kind == EOS_T
        stopped = (kind == TIME_T) & (self.time_seconds[tokens] >= stop)
        ended = torch.where(eos, 1, torch.where(stopped, 2, torch.where(at_limit, 3, 0)))
        finished = eos | stopped | at_limit

        for name, value in (
            ("payload", payload),
            ("last_field", last),
            ("incomplete", incomplete),
            ("in_shift", in_shift),
            ("shift_run", shift_run),
        ):
            state[name].copy_(torch.where(active, value, state[name]))
        state["ended"].copy_(torch.where(active & finished, ended, state["ended"]))
        state["finished"].copy_(state["finished"] | (active & finished))

    def prefix_state(self, tokens: list[int], out_index: int, device) -> dict[str, Tensor]:
        """Grammar state of a single sequence after its (non-empty) prefix."""

        if self._cpu is None:
            self._cpu = TokenGrammar.__new__(TokenGrammar)
            self._cpu.prompts = self.prompts
            self._cpu.token_class, self._cpu.field_of_class, self._cpu.time_seconds = (
                self.token_class.cpu(),
                self.field_of_class.cpu(),
                self.time_seconds.cpu(),
            )
        cpu, state = self._cpu, self.new_state(1, "cpu")
        never = torch.zeros(1, dtype=torch.bool)
        for token in tokens[out_index + 1 :]:
            cpu.update(state, torch.tensor([token]), ~never, torch.tensor([float("inf")], dtype=torch.float64), never)
        return {name: value.to(device) for name, value in state.items()}
