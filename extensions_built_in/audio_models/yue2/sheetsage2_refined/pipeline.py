"""A trained SheetSage2-Refined model: a model folder with ``config.json`` and the weights it names.

    from sheetsage2_refined import SheetSage2Refined

    model = SheetSage2Refined.load()                     # the Hugging Face repo DEFAULT_MODEL, or a local folder
    events = model.transcribe(["song.mp3"])[0]          # one event list per song
    beats = events_to_annotations(events, duration)      # beats, downbeats, chords, keys

``config.json`` (paths relative to the folder)::

    {
      "base_weights": "../sheetsage2/model.safetensors",  # SheetSage2: encoder, layer mix, projection
      "decoder": "decoder.safetensors",                    # the fine-tuned decoder (null: SheetSage2's own)
      "bar_features": {                                    # optional, if the decoder learnt with them
        "projection": "bar_features.safetensors",
        "heads": ["heads/bar-v5-f0.safetensors", "heads/bar-v5-f1.safetensors"]
      },
      "rescoring": {                                       # optional: term -> frame head and weight
        "downbeat": {"head": ["heads/bar-v5-f0.safetensors", "heads/bar-v5-f1.safetensors"], "weight": 1.0},
        "root": {"head": "heads/chord-v1.safetensors", "weight": 2.0}, ...  # a list of heads: their mean
      },
      "beat_tracker": {"head": "heads/beat-v7u.safetensors"},  # optional: beats without the decoder (beats.py)
      "melody_tracker": {"head": "heads/melody-m4.safetensors",  # optional: the melody without the decoder (melody.py)
                         "decoding": {"onset_weight": 2.0, "start_penalty": 0.5}},
      "section_tracker": {"head": "heads/section-s3.safetensors",  # optional: sections without the decoder (sections.py)
                          "decoding": {"boundary_weight": 8.0, "length_weight": 1.0, "transition_weight": 0.0}},
      "key_tracker": {"head": "heads/key-k10.safetensors",  # optional: keys without the decoder (keys.py)
                      "decoding": {"local_weight": 2.0, "boundary_weight": 4.0, "mode_tau": 0.5, "min_bars": 4}},
      "prompts": ["timestamp", "downbeat_meter", "structure", "key", "chord_detail", "melody_full"],
      "mode_tau": 0.2,                                     # with the key_mode prompt (see ``mode_bias``)
      "mode_counts": [...]                                 # one per theory.MODES
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .engine import FRAME_TERMS
from .heads import BarFeatures, FrameHead, HeadMean, load_head
from .model import SheetSage
from .transcribe import load_waveform, transcribe_many

DEFAULT_MODEL = "anime-song/SheetSage2-Refined"


def weight_files(config: dict) -> list[str]:
    """Every file a model folder's ``config.json`` names."""

    bar = config.get("bar_features") or {}
    names = [config["base_weights"], config.get("decoder"), bar.get("projection"), *bar.get("heads", [])]
    names.append((config.get("beat_tracker") or {}).get("head"))
    melody = config.get("melody_tracker") or {}
    names.extend([melody.get("head"), melody.get("beat_head")])
    names.append((config.get("section_tracker") or {}).get("head"))
    names.append((config.get("key_tracker") or {}).get("head"))
    for spec in config.get("rescoring", {}).values():
        names.extend(spec["head"] if isinstance(spec["head"], list) else [spec["head"]])
    return sorted({name for name in names if name})


def model_folder(model: Path | str, revision: str | None = None, files=weight_files) -> Path:
    """A local model folder as it is, or a Hugging Face model repo downloaded into the Hub cache
    (only ``config.json`` and the files ``files(config)`` names: by default every file it names)."""

    if Path(model).is_dir():
        return Path(model)
    if isinstance(model, Path) or model.count("/") != 1:
        raise FileNotFoundError(f"no model folder {model} (a Hugging Face repo id looks like owner/name)")
    from huggingface_hub import hf_hub_download, snapshot_download

    config = json.loads(Path(hf_hub_download(str(model), "config.json", revision=revision)).read_text(encoding="utf-8"))
    return Path(snapshot_download(str(model), revision=revision, allow_patterns=["config.json", *files(config)]))


def mode_bias(tau: float, counts: list[float] | None) -> list[float] | None:
    """``-tau * log P(mode)``, P from the key-mode counts of the training clips (None for tau 0).

    The decoder ranks modes well but almost never picks a rare one; this bias trades a little
    major / minor accuracy for the modal keys.  It acts only on the ``key_mode`` prompt.
    """

    if not tau or not counts:
        return None
    counts = np.asarray(counts, dtype=np.float64)
    return [float(-tau * np.log(c / counts.sum())) for c in counts]


class SheetSage2Refined:
    def __init__(self, model: SheetSage, config: dict, rescoring: dict[str, tuple[FrameHead, float]]):
        self.model, self.config, self.rescoring = model, config, rescoring

    @classmethod
    def load(
        cls,
        folder: Path | str = DEFAULT_MODEL,
        device: str | None = None,
        rescoring: list[str] | None = None,
        revision: str | None = None,
    ) -> "SheetSage2Refined":
        """Load a model folder or a Hugging Face repo id (at ``revision``) on ``device`` (default: CUDA if there is
        one); ``rescoring`` keeps only these terms of its rescoring (``[]``: none)."""

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        folder = model_folder(folder, revision)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        decoder = folder / config["decoder"] if config.get("decoder") else None
        model = SheetSage.load(folder / config["base_weights"], device, decoder=decoder)
        heads: dict = {}

        def head(name: str | list[str]) -> FrameHead | HeadMean:
            if isinstance(name, list):  # several heads: their probabilities averaged
                key = tuple(sorted((folder / n).resolve() for n in name))
                if key not in heads:
                    heads[key] = HeadMean([head(n) for n in name]).eval()
                return heads[key]
            path = (folder / name).resolve()
            if path not in heads:
                heads[path] = load_head(path, device)
            return heads[path]

        if config.get("bar_features"):
            spec = config["bar_features"]
            model.bar_features = BarFeatures([head(p) for p in spec["heads"]]).to(device).eval()
            model.bar_features.load_projection(folder / spec["projection"])
        terms = config.get("rescoring", {})
        if rescoring is not None:
            unknown = set(rescoring) - set(terms)
            if unknown:
                raise ValueError(f"{folder} has no rescoring for {sorted(unknown)}; it has {list(terms)}")
            terms = {term: terms[term] for term in rescoring}
        for term in terms:
            if term not in FRAME_TERMS:
                raise ValueError(f"unknown rescoring term {term!r}; known: {list(FRAME_TERMS)}")
        return cls(model, config, {term: (head(spec["head"]), float(spec["weight"])) for term, spec in terms.items()})

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(self.config["prompts"])

    def transcribe(
        self, audio: list, prompts=None, batch_size: int = 8, mode_tau: float | None = None, **options
    ) -> list[list[dict]]:
        """Events of each song (an audio path or a mono 24 kHz waveform).  ``prompts`` defaults to the
        config's; use ``key_mode`` instead of ``key`` for keys with modes (dorian, mixolydian, ...).
        ``mode_tau`` overrides the config's; ``options`` go to ``transcribe_many``."""

        waves = [a if isinstance(a, torch.Tensor) else load_waveform(Path(a), self.model.sample_rate) for a in audio]
        tau = self.config.get("mode_tau", 0.0) if mode_tau is None else mode_tau
        bias = mode_bias(tau, self.config.get("mode_counts"))
        return transcribe_many(
            self.model,
            waves,
            tuple(prompts or self.prompts),
            batch_size=batch_size,
            mode_bias=bias,
            rescoring=self.rescoring,
            **options,
        )
