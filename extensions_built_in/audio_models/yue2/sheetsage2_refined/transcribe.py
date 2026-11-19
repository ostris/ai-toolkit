"""Whole-song transcription: grammar-constrained greedy decoding over 300 s windows.

Songs longer than one window are decoded in windows that overlap by 200 s.
Each window accepts events up to 100 s before its end (the look-ahead), and the
next window is primed with the already accepted events of the overlap, so the
beat grid, key and chord continue across window boundaries.  A window that
fills the decoder context before its end is decoded again from a beat shortly
before where it stopped (``RESUME_BACKOFF``), primed with that one beat event.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .engine import BatchDecoder
from .heads import FrameHead, HeadMean
from .model import WINDOW_SECONDS, SheetSage
from .vocab import FIELDS, FULL_PROMPTS, Vocab

MAX_TOKENS = 5120
RESUME_BACKOFF = 5.0  # seconds re-decoded before the point where a full window stopped
MAX_RESUMES = 4  # per window


def load_waveform(path: Path, sample_rate: int = 24_000) -> torch.Tensor:
    """Read any soundfile-supported file as a mono float32 waveform."""

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.T).mean(0)
    return torchaudio.functional.resample(wave, rate, sample_rate) if rate != sample_rate else wave


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


# --------------------------------------------------------------------- windows
def window_plan(duration: float, window=WINDOW_SECONDS, overlap=200.0, lookahead=100.0) -> list[dict]:
    hop, start, accepted, plan = window - overlap, 0.0, 0.0, []
    while True:
        last = start + window >= duration - 1e-6
        accept_end = duration if last else start + window - lookahead
        plan.append(
            {
                "start": start,
                "accept_start": accepted,
                "accept_end": accept_end,
                "stop": None if last else window - lookahead,
            }
        )
        if last:
            return plan
        accepted = accept_end
        start = min(start + hop, duration - window)


def subbeat_time_map(events: list[dict], window: float):
    """Map a subbeat index to seconds in the window by interpolating timestamped events."""

    anchors = sorted({e["subbeat"]: e["values"]["timestamp"] for e in events if "timestamp" in e["values"]}.items())
    if not anchors:
        return lambda step: min(window, max(0.0, step * 0.125))
    steps = np.array([a[0] for a in anchors], dtype=np.float64)
    times = np.array([a[1] for a in anchors], dtype=np.float64)
    seconds_per_step = 0.125
    if len(anchors) >= 2:
        estimate = float(np.median(np.diff(times) / np.maximum(np.diff(steps), 1)))
        if np.isfinite(estimate) and estimate > 0:
            seconds_per_step = estimate

    def lookup(step):
        if step <= steps[0]:
            return float(np.clip(times[0] + (step - steps[0]) * seconds_per_step, 0, window))
        if step >= steps[-1]:
            return float(np.clip(times[-1] + (step - steps[-1]) * seconds_per_step, 0, window))
        return float(np.interp(step, steps, times))

    return lookup


def accept_events(events, lookup, window: dict, duration: float, subbeat_base: int) -> list[dict]:
    """Place a window's events on the song timeline and keep those it is responsible for."""

    accepted, eps = [], 1e-4
    for event in events:
        time = window["start"] + lookup(event["subbeat"])
        if time < window["accept_start"] - eps or time >= min(window["accept_end"], duration) - eps:
            continue
        event = copy.deepcopy(event)
        event["time"] = float(np.clip(time, 0.0, duration))
        event["global_subbeat"] = subbeat_base + event["subbeat"]
        if "timestamp" in event["values"]:
            event["values"]["timestamp"] = event["time"]
        for note in event["values"].get("melody", []):
            end = window["start"] + lookup(event["subbeat"] + note["duration_steps"])
            note["end_time"] = min(duration, max(event["time"] + 0.04, end))
        accepted.append(event)
    return accepted


def overlap_prefix(stitched, vocab: Vocab, prompts, window_start: float, prefix_end: float, origin=None):
    """Re-encode accepted events in ``[window_start, prefix_end)`` as the next window's prefix.

    Timestamps are relative to ``origin`` (default ``window_start``), the start of
    the window being decoded.  Returns ``(tokens, subbeat_base)`` or ``(None, 0)``
    when the overlap holds no beat.
    """

    origin = window_start if origin is None else origin

    eps = 1e-4
    source = sorted(
        (e for e in stitched if window_start - eps <= e["time"] < prefix_end - eps),
        key=lambda e: (e["global_subbeat"], e["time"]),
    )
    first = next((i for i, e in enumerate(source) if "timestamp" in e["values"] or "rhythm" in e["values"]), None)
    if first is None:
        return None, 0
    source = source[first:]
    base = source[0]["global_subbeat"]

    # State that is still active at the prefix start, carried into its first event.
    context: dict[str, list[int]] = {}
    for event in stitched:
        if event["time"] > source[0]["time"] + 1e-6:
            continue
        for field in ("structure", "key", "chord"):
            if field in event["tokens"]:
                context[field] = list(event["tokens"][field])
        meters = [t for t in event["tokens"].get("rhythm", []) if vocab.token_type(t) == "meter"]
        if meters:
            context["meter"] = meters[:1]

    prefix_events = []
    for event in source:
        tokens = {field: list(values) for field, values in event["tokens"].items()}
        if "timestamp" in tokens:
            tokens["timestamp"] = [vocab.time_token(event["time"] - origin)]
        prefix_events.append({"subbeat": event["global_subbeat"] - base, "tokens": tokens})
    head = prefix_events[0]["tokens"]
    for field in ("structure", "key", "chord"):
        if field not in head and field in context:
            head[field] = list(context[field])
    rhythm = head.get("rhythm", [])
    kinds = {vocab.token_type(t) for t in rhythm}
    if "eighth_position" in kinds and "meter" not in kinds and "meter" in context:
        head["rhythm"] = context["meter"] + rhythm
    for event in prefix_events:
        event["tokens"] = {field: event["tokens"][field] for field in FIELDS if field in event["tokens"]}
    return vocab.encode(prefix_events, prompts, eos=False), base


def resume_point(events: list[dict], window: dict) -> float | None:
    """The beat about ``RESUME_BACKOFF`` s before a full window's last accepted beat, if worth resuming."""

    beats = [e["time"] for e in events if "rhythm" in e["values"] and e["time"] >= window["accept_start"]]
    if not beats or beats[-1] >= window["accept_end"] - 1.0:
        return None
    earlier = [t for t in beats if t <= beats[-1] - RESUME_BACKOFF]
    cut = earlier[-1] if earlier else None
    return cut if cut is not None and cut > window["accept_start"] + 1.0 else None


# Frames of frame-head probabilities averaged from the current time (chords: 0.48 s, keys: 2 s,
# bar length: 0.32 s into the bar), and the half-width of the window whose largest boundary /
# downbeat probability counts.
FRAME_LOOKAHEAD = {"root": 12, "bass": 12, "detail": 12, "quality": 12, "key": 50, "meter": 8}
FRAME_OUTPUT = {"meter": "bar_beats", "quality": "detail"}  # the head output a term reads, when not its own name
FRAME_REACH = {"key_boundary": 25, "downbeat": 2}


def frame_table(out: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """One per-frame table of a frame head's output for one window: log-probabilities of root, bass
    (12 pitch classes, N left out), root x quality (744, N left out; for ``quality`` the quality given
    the root, log P(root, quality) - log P(root)), key (24) or the bar's number of beats (16) averaged over the
    next frames, so that a chord, key or bar is judged by where it is heading; or the logit of the
    largest key-change / downbeat probability near each frame."""

    if name in FRAME_LOOKAHEAD:
        span = FRAME_LOOKAHEAD[name]
        probs = torch.softmax(out[FRAME_OUTPUT.get(name, name)][0].float(), dim=-1)
        probs = probs[:, :12] if name in ("root", "bass") else probs[:, :-1] if name in ("detail", "quality") else probs
        padded = torch.cat((probs, probs[-1:].expand(span - 1, -1)))
        cumulative = torch.cat((torch.zeros_like(padded[:1]), padded.cumsum(0)))
        mean = (cumulative[span:] - cumulative[: len(probs)]) / span
        if name == "quality":  # chord_detail offsets are 12 * quality + root
            roots = mean.unflatten(-1, (-1, 12)).sum(-2)
            return mean.clamp_min(1e-6).log() - roots.clamp_min(1e-6).log().repeat(1, mean.shape[-1] // 12)
        return mean.clamp_min(1e-6).log()
    reach = FRAME_REACH[name]
    probs = torch.sigmoid(out[name][0].float())[None, None]
    peak = torch.nn.functional.max_pool1d(probs, 2 * reach + 1, stride=1, padding=reach)[0, 0]
    peak = peak.clamp(0.02, 0.98)
    return peak.log() - (1 - peak).log()


def _engine(model: SheetSage, batch: int, frames: int, prompts, mode_bias=None, frame_weight=None) -> BatchDecoder:
    """The model's decoding engine, rebuilt when the batch size, prompts, rescoring or weights change."""

    engine = getattr(model, "_batch_decoder", None)
    current = engine is not None and engine.is_current(model.decoder, batch, frames, prompts, mode_bias, frame_weight)
    if not current:
        engine = model._batch_decoder = None  # free the old caches first
        engine = model._batch_decoder = BatchDecoder(
            model.decoder,
            model.vocab,
            batch=batch,
            frames=frames,
            prompts=prompts,
            mode_bias=mode_bias,
            frame_weight=frame_weight,
        )
    return engine


@torch.inference_mode()
def transcribe_many(
    model: SheetSage,
    waveforms: list[torch.Tensor],
    prompts=FULL_PROMPTS,
    *,
    batch_size: int = 8,
    encode_batch: int = 2,
    poll_every: int = 32,
    mask_padding: bool = True,
    resume_on_limit: bool = True,
    mode_bias: list[float] | None = None,
    rescoring: dict[str, tuple[FrameHead | HeadMean, float]] | None = None,
    stats: dict | None = None,
) -> list[list[dict]]:
    """Transcribe several mono 24 kHz waveforms; returns one event list per waveform.

    Up to ``batch_size`` windows are decoded at once.  A song's windows run in
    order (each primes the next), but windows of different songs share the
    batch, and a slot is refilled as soon as its window ends.

    Windows shorter than ``WINDOW_SECONDS`` are zero-padded for the encoder
    (``model.pad_encoder``, as SheetSage2 was trained) or encoded at their own
    length (encoders tuned on unpadded audio, whose memory is then padded and
    always masked); with ``mask_padding`` the decoder does not attend to the
    padding frames.
    With ``resume_on_limit`` a window that fills the decoder context is resumed.
    ``mode_bias`` (one value per ``theory.MODES``) biases key_mode tokens by their mode.
    ``rescoring`` (``engine.FRAME_TERMS`` term -> (frame head, weight)) rescores chords, keys and
    downbeats with frame heads' predictions (``frame_table``, ``engine.BatchDecoder``).  A model with
    ``bar_features`` (``heads.BarFeatures``) adds its bar heads' bar facts to the decoder memory.
    ``stats``, if given, receives the encoder time and one diagnostic dict per window.
    """

    vocab, device = model.vocab, next(model.parameters()).device
    rate = model.sample_rate
    window_samples = round(WINDOW_SECONDS * rate)
    samples_per_frame = model.encoder.config.samples_per_frame
    rescoring = rescoring or {}
    frame_weight = {term: weight for term, (_, weight) in rescoring.items()}
    engine = _engine(model, batch_size, window_samples // samples_per_frame, prompts, mode_bias, frame_weight)
    rescore = engine.rescore
    songs = [
        {"wave": w, "duration": len(w) / rate, "next": 0, "events": [], "resume": None, "resumes": 0}
        for w in waveforms
    ]
    for song in songs:
        song["plan"] = window_plan(song["duration"])
    waiting = deque(range(len(songs)))  # songs whose next window can start
    running: dict[int, tuple] = {}
    if stats is not None:
        stats.setdefault("encode_seconds", 0.0)
        stats.setdefault("windows", [])

    while waiting or running:
        jobs = []
        for slot in engine.free_slots():
            if not waiting:
                break
            jobs.append((slot, waiting.popleft()))
        for first in range(0, len(jobs), encode_batch):
            chunk = jobs[first : first + encode_batch]
            segments, real_frames = [], []
            for _, index in chunk:
                song = songs[index]
                offset = round(song["plan"][song["next"]]["start"] * rate)
                segment = song["wave"][offset : offset + window_samples].float()
                if model.pad_encoder:
                    segment = torch.nn.functional.pad(segment, (0, window_samples - len(segment)))
                segments.append(segment)
                real_frames.append(min(-(-len(segment) // samples_per_frame), engine.frames))
            if stats is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            tables = [None] * len(segments)
            with autocast(device):
                if rescore or model.bar_features is not None:
                    memories = []
                    for i, s in enumerate(segments):
                        states = model.encoder(s[None].to(device))
                        memory = model.memory(states)
                        if model.bar_features is not None:
                            memory = model.bar_features(memory, states)
                        memories.append(memory[0])
                        if not rescore:
                            continue
                        outputs = {}  # each head runs once per window
                        for head, _ in rescoring.values():
                            if id(head) not in outputs:
                                outputs[id(head)] = head(states)
                        tables[i] = {term: frame_table(outputs[id(head)], term) for term, (head, _) in rescoring.items()}
                elif len({len(s) for s in segments}) == 1:
                    memories = list(model.encode(torch.stack(segments).to(device)))
                else:  # unpadded windows of different lengths
                    memories = [model.encode(s[None].to(device))[0] for s in segments]
            memories = [
                torch.nn.functional.pad(m[: engine.frames], (0, 0, 0, engine.frames - min(len(m), engine.frames)))
                for m in memories
            ]
            if stats is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                stats["encode_seconds"] += time.perf_counter() - started
            for (slot, index), memory, frames, table in zip(chunk, memories, real_frames, tables):
                song = songs[index]
                window = song["plan"][song["next"]]
                prefix, base = (None, 0)
                if song["resume"] is not None:
                    cut, song["resume"] = song["resume"], None
                    window = dict(window, accept_start=cut + 1e-3)  # the primed beat is already accepted
                    prefix, base = overlap_prefix(song["events"], vocab, prompts, cut, cut + 1e-3, window["start"])
                elif song["next"] > 0:
                    prefix, base = overlap_prefix(
                        song["events"], vocab, prompts, window["start"], window["accept_start"]
                    )
                    # A dense overlap (fast eighth-note beats, busy melody) can fill the context: prime
                    # the window with a shorter, later part of the overlap instead.
                    span = window["accept_start"] - window["start"]
                    while prefix is not None and len(prefix) >= MAX_TOKENS - 128 and span > 5.0:
                        span /= 2
                        prefix, base = overlap_prefix(
                            song["events"], vocab, prompts, window["accept_start"] - span, window["accept_start"], window["start"]
                        )
                if prefix is not None and len(prefix) >= MAX_TOKENS - 128:
                    raise ValueError("overlap prefix fills the decoder context")
                prefix = prefix or vocab.prompt_prefix(prompts)
                stop = (
                    window["stop"]
                    if window["stop"] is not None
                    else min(song["duration"] - window["start"], WINDOW_SECONDS)
                )
                masked = mask_padding or not model.pad_encoder
                engine.load(slot, memory[None], prefix, stop, frames if masked else None, table)
                running[slot] = (index, window, base, len(prefix))

        engine.run(poll_every)
        for slot in engine.finished():
            tokens, ended = engine.take(slot)
            index, window, base, prefix_length = running.pop(slot)
            song = songs[index]
            events = vocab.decode(tokens)
            lookup = subbeat_time_map(events, WINDOW_SECONDS)
            song["events"] += accept_events(events, lookup, window, song["duration"], base)
            cut = None
            # SheetSage2 itself tends to emit <eos> a few tokens before the context is full.
            full = ended == "limit" or len(tokens) >= MAX_TOKENS - 64
            if full and resume_on_limit and song["resumes"] < MAX_RESUMES:
                cut = resume_point(song["events"], window)
            if stats is not None:
                times = [e["values"]["timestamp"] for e in events if "timestamp" in e["values"]]
                stats["windows"].append(
                    {
                        "song": index,
                        "start": window["start"],
                        "tokens": len(tokens),
                        "generated": len(tokens) - prefix_length,
                        "ended": ended,
                        "last_time": window["start"] + max(times, default=0.0),
                    }
                )
            if cut is not None:
                # Drop the tail after the resume beat (its last events may be cut short) and redo it.
                song["events"] = [e for e in song["events"] if e["time"] <= cut + 1e-6]
                song["resume"], song["resumes"] = cut, song["resumes"] + 1
                waiting.append(index)
                continue
            song["next"] += 1
            song["resumes"] = 0
            if song["next"] < len(song["plan"]):
                waiting.append(index)

    for song in songs:
        song["events"].sort(key=lambda e: (e["time"], e["global_subbeat"]))
    return [song["events"] for song in songs]


def events_to_annotations(events: list[dict], duration: float) -> dict:
    """Extract beats, downbeats, chord intervals and key segments from events."""

    beats = [e["time"] for e in events if "rhythm" in e["values"]]
    downbeats = [e["time"] for e in events if e["values"].get("rhythm", {}).get("eighth_position") == 0]

    chords = [(e["time"], e["values"]["chord"]) for e in events if "chord" in e["values"]]
    chord_intervals = [[t, chords[i + 1][0] if i + 1 < len(chords) else duration] for i, (t, _) in enumerate(chords)]

    keys: list[tuple[float, float, str | None]] = []
    for event in events:
        if "key" not in event["values"]:
            continue
        label = None if event["values"]["key"] == "N" else event["values"]["key"]
        if keys and keys[-1][2] == label:
            continue
        if keys:
            keys[-1] = (keys[-1][0], event["time"], keys[-1][2])
        keys.append((event["time"], duration, label))

    return {
        "beats": np.asarray(beats),
        "downbeats": np.asarray(downbeats),
        "chord_intervals": np.asarray(chord_intervals, dtype=float).reshape(-1, 2),
        "chord_labels": [label for _, label in chords],
        "keys": keys,
    }
