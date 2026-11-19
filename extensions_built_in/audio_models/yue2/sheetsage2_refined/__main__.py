"""Transcribe audio files and write one JSON of events per file.

    python -m sheetsage2_refined song.mp3 [more.wav ...] [--out-dir outputs/transcriptions] [--key-mode]

The model (``--model``) is a local model folder or a Hugging Face repo id, by default the published
SheetSage2-Refined.  ``python -m sheetsage2_refined.visualize`` draws a written JSON as an HTML page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .pipeline import DEFAULT_MODEL, SheetSage2Refined
from .transcribe import load_waveform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model folder with config.json, or a Hugging Face repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hugging Face model repo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "transcriptions")
    parser.add_argument("--key-mode", action="store_true", help="keys with modes (key_mode prompt) instead of major / minor")
    parser.add_argument("--memory-fraction", type=float, default=0.85, help="cap on this process's share of GPU memory")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device).index or 0)
    model = SheetSage2Refined.load(args.model, args.device, revision=args.revision)
    prompts = [("key_mode" if p == "key" else p) for p in model.prompts] if args.key_mode else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.audio:
        wave = load_waveform(path)
        events = model.transcribe([wave], prompts)[0]
        record = {"audio": str(path), "duration": len(wave) / model.model.sample_rate, "events": events}
        out = args.out_dir / f"{path.stem}.json"
        out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        print(out, flush=True)


if __name__ == "__main__":
    main()
