"""Beats, bar heads and meters of whole songs without the decoder: a bar head (``heads.FrameHead`` with the bar
facts, best with ``offset``) on SheetSage2's frozen encoder, then ``bar_tracker`` over its probabilities, then the
head's sub-frame offsets on the tracked beats.

    python -m sheetsage2_refined.beats song.mp3 [more.wav ...] [--model REPO_OR_FOLDER] [--out-dir DIR]

writes ``<out-dir>/<song>.beats.json``::

    {"audio": ..., "duration": s, "beats": [s, ...], "downbeats": [s, ...],
     "bars": [{"start": s, "beats": n, "unit": d, "meter": "n/d"}, ...], "meters": [beats per bar the tracker considered]}

``unit`` and ``meter`` come with a head that predicts the denominator (``bar_unit``): ``bar_tracker.denominators``
picks them so the written tempo does not jump (the same beat keeps its denominator; 4/4 -> 7/8 is written /8, not 7/4).

From Python::

    from sheetsage2_refined.beats import HeadBeats

    tracker = HeadBeats.load()                # SheetSage2's encoder + the beat head of the Hugging Face repo
    result = tracker.track("song.mp3")        # the dict above

The head ``beat-v7u`` (``models/sheetsage2-refined/heads/beat-v7u.safetensors``) is the bar head trained with the
sub-frame offsets and the denominator, on the beat-only songs and synthetic meter songs v7 (exp/synth-v7,
``training.train_refiner --arm A --synth-dir datasets/synth_meter_v7 --offset --unit``).  On the valid songs: beat F
0.929 / downbeat F 0.883 (decoder about 0.90 / 0.85), beats a median 2.2 ms from the labels; the denominator is right
on 0.884 of the real dev and valid bars, and choosing it for a steady written tempo cuts its tempo jumps 2-4x at the
same accuracy.  ``beat-h3`` (offsets only, no denominator) was its predecessor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from . import bar_tracker
from .encoder import EncoderConfig, MERTEncoder
from .heads import CHUNK, OFFSETS, UNITS, FrameHead, HeadMean, load_head
from .model import WINDOW_SECONDS
from .pipeline import DEFAULT_MODEL, model_folder
from .transcribe import autocast, load_waveform


def beat_files(config: dict) -> list[str]:
    """The files of a model folder the beat tracker needs: SheetSage2's weights (for the encoder) and the beat head
    (``config.json``'s ``beat_tracker.head``)."""

    if not (config.get("beat_tracker") or {}).get("head"):
        raise ValueError("this model has no beat head (config.json has no beat_tracker.head)")
    return [config["base_weights"], config["beat_tracker"]["head"]]


def load_encoder(path: Path, device: str | torch.device = "cpu") -> MERTEncoder:
    """SheetSage2's encoder alone (the ``encoder.*`` weights of its file)."""

    encoder = MERTEncoder(EncoderConfig())
    state = load_file(str(path), device=str(device))
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")})
    encoder.requires_grad_(False)
    return encoder.to(device).eval()


@torch.no_grad()
def encode_windows(encoder: MERTEncoder, wave: torch.Tensor, device: torch.device):
    """The encoder states of a 24 kHz waveform in windows of ``WINDOW_SECONDS`` at their own length (as the heads were
    trained), each with how many of its frames to keep (``None``: all).  A last window too short for the STFT (at most
    ``n_fft / 2`` samples, the song a few ms longer than a multiple of the window) is padded with silence to 1 s and
    keeps the frames its own samples cover."""

    config = encoder.config
    window = round(WINDOW_SECONDS * config.sample_rate)
    for start in range(0, len(wave), window):
        piece, keep = wave[start : start + window], None
        if len(piece) <= config.n_fft // 2:
            keep = len(piece) // config.samples_per_frame
            piece = torch.nn.functional.pad(piece, (0, config.sample_rate - len(piece)))
        with autocast(device):
            states = encoder(piece[None].to(device))
        yield states, keep


def frame_outputs(encoder: MERTEncoder, head: FrameHead | HeadMean, wave: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    """A head's 25 Hz outputs (logits, ``[frames, ...]`` on the CPU in float32) for a 24 kHz waveform, encoded in
    windows (``encode_windows``)."""

    outs = []
    for states, keep in encode_windows(encoder, wave, device):
        outs.append({k: v[:, :keep] for k, v in head(states, CHUNK).items()})  # the head runs in float32 (its GRU)
    return {k: torch.cat([o[k] for o in outs], 1)[0].float().cpu() for k in outs[0]}


class HeadBeats:
    def __init__(self, encoder: MERTEncoder, head: FrameHead | HeadMean, device: torch.device, tempo_weight: float = 4.0,
                 refold: bool = True, **options):  # fmt: skip
        self.encoder, self.head, self.device = encoder, head, device
        self.tempo_weight = tempo_weight  # bar_tracker.denominators: how firmly the written tempo holds
        self.refold = refold  # bar_tracker.refold: half-length bars read again at the surrounding tempo
        self.options = options  # bar_tracker.track's settings

    @classmethod
    def load(cls, model: Path | str = DEFAULT_MODEL, heads: list[Path] | None = None, device: str | None = None,
             revision: str | None = None, weights: Path | None = None, tempo_weight: float = 4.0,
             **options) -> "HeadBeats":  # fmt: skip
        """The encoder and the beat head of a model folder or Hugging Face repo id (at ``revision``; from the Hub only
        those two files are downloaded) on ``device`` (default: CUDA if there is one).  ``weights`` (SheetSage2's
        file) and ``heads`` (bar heads; several: their mean) replace the model's; ``tempo_weight`` goes to
        ``bar_tracker.denominators``, ``options`` (``refold``, then ``bar_tracker.track``'s settings) to ``__init__``."""

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if weights is None or not heads:
            folder = model_folder(model, revision, beat_files)
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            weights = weights or folder / config["base_weights"]
            heads = heads or [folder / config["beat_tracker"]["head"]]
        paths = heads
        loaded = [load_head(Path(p), device) for p in paths]
        if any(h.bar is None for h in loaded):
            raise ValueError("beat tracking needs a bar head (trained with --bar)")
        if len({h.unit is None for h in loaded}) > 1:
            raise ValueError("either all heads predict the denominator or none")
        head = loaded[0] if len(loaded) == 1 else HeadMean(loaded).eval()
        return cls(load_encoder(weights, device), head, device, tempo_weight, **options)

    def probabilities(self, wave: torch.Tensor) -> dict[str, np.ndarray]:
        """The head's 25 Hz outputs for a 24 kHz waveform (``frame_outputs``): beat, downbeat [frames]; bar_beats,
        beats_left [frames, 16]; and the offsets (frames) and bar_unit [frames, 4] if the head has them."""

        out = frame_outputs(self.encoder, self.head, wave, self.device)
        probs = {k: torch.sigmoid(out[k]).cpu().numpy() for k in ("beat", "downbeat")}
        probs |= {k: torch.softmax(out[k], -1).cpu().numpy() for k in ("bar_beats", "beats_left", "bar_unit") if k in out}
        probs |= {k: out[k].cpu().numpy() for k in OFFSETS if k in out}
        return probs

    def track_probabilities(self, probs: dict[str, np.ndarray]) -> dict:
        beats, downbeats, metres = bar_tracker.track_head(probs, **self.options)
        if "beat_offset" in probs:
            beats, downbeats = bar_tracker.apply_offsets(beats, downbeats, probs["beat"], probs["beat_offset"])
        if self.refold:
            beats, downbeats = bar_tracker.refold(beats, downbeats)
        bars = []
        for i, d in enumerate(downbeats):
            end = downbeats[i + 1] if i + 1 < len(downbeats) else np.inf
            bars.append({"start": round(float(d), 4), "beats": int(np.sum((beats >= d - 1e-6) & (beats < end - 1e-6)))})
        if "bar_unit" in probs:
            duration = len(probs["beat"]) / bar_tracker.HEAD_RATE
            units = bar_tracker.denominators(beats, downbeats, probs["bar_unit"], duration, UNITS, self.tempo_weight)
            for bar, unit in zip(bars, units):
                bar |= {"unit": unit, "meter": f"{bar['beats']}/{unit}"}
        return {"beats": [round(float(b), 4) for b in beats], "downbeats": [round(float(d), 4) for d in downbeats],
                "bars": bars, "meters": [int(m) for m in metres]}  # fmt: skip

    def track(self, audio: Path | str) -> dict:
        wave = load_waveform(Path(audio), self.encoder.config.sample_rate)
        result = self.track_probabilities(self.probabilities(wave))
        return {"audio": str(audio), "duration": len(wave) / self.encoder.config.sample_rate} | result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local model folder")
    parser.add_argument("--revision", help="the Hugging Face repo's branch, tag or commit")
    parser.add_argument("--weights", type=Path, help="SheetSage2's weights (default: the model's)")
    parser.add_argument("--head", type=Path, action="append", help="bar head(s); several: their mean (default: the model's)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "beats")
    parser.add_argument("--memory-fraction", type=float, default=0.6, help="cap on this process's share of GPU memory")
    parser.add_argument("--threads", type=int, help="CPU threads for torch (default: torch's own)")
    parser.add_argument("--hold", type=float, default=0.0, help="probability of entering a fermata hold (0: off)")
    parser.add_argument("--no-refold", action="store_true",
                        help="keep bars at double tempo inside a meter (bar_tracker.refold off)")
    parser.add_argument("--tempo-weight", type=float, default=4.0,
                        help="denominators: cost of a jump in the written tempo (0: the head's own guess per bar)")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device).index or 0)
    if args.threads:
        torch.set_num_threads(args.threads)
    tracker = HeadBeats.load(args.model, args.head, args.device, args.revision, args.weights,
                             tempo_weight=args.tempo_weight, refold=not args.no_refold, p_hold=args.hold)  # fmt: skip
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for audio in args.audio:
        started = time.perf_counter()
        result = tracker.track(audio)
        path = args.out_dir / f"{audio.stem}.beats.json"
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        metres = sorted({b.get("meter", b["beats"]) for b in result["bars"]}, key=str)
        print(f"{audio.name}: {len(result['beats'])} beats, {len(result['bars'])} bars (meters {metres}), "
              f"{time.perf_counter() - started:.1f} s -> {path}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
