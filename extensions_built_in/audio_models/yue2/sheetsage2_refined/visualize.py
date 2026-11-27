"""Draw a transcription as an HTML page: meters, beats, downbeats, chords, keys and the melody, with the audio.

    python -m sheetsage2_refined.visualize outputs/transcriptions/song.json [--embed-audio]

The page (``outputs/visualizations/<name>.html`` by default) plays the audio with a scrolling playhead and can
click the beats and play the melody along.  The audio is linked by a relative path unless ``--embed-audio`` puts
it inside the page.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

from .transcribe import events_to_annotations

TEMPLATE = Path(__file__).with_name("visualize.html")


def page_data(record: dict) -> dict:
    """The drawable parts of a transcription record."""

    events, duration = record["events"], record["duration"]
    annotations = events_to_annotations(events, duration)
    beats, bars, meter = [], [], None
    for event in events:
        rhythm = event["values"].get("rhythm")
        if rhythm is None:
            continue
        meter = rhythm.get("meter", meter)
        down = rhythm.get("eighth_position") == 0
        beats.append([round(event["time"], 3), int(down)])
        if down or not bars:
            bars.append({"start": event["time"], "beats": 0, "meter": meter, "meter_token": "meter" in rhythm})
        bars[-1]["beats"] += 1
    for bar, following in zip(bars, bars[1:] + [None]):
        bar["end"] = following["start"] if following else duration
    notes = [
        [round(event["time"], 3), round(note["end_time"], 3), note["pitch"], note["track"]]
        for event in events
        for note in event["values"].get("melody", [])
    ]
    sections, label = [], None
    for event in events:
        value = event["values"].get("structure")
        if value is not None and value != label:
            sections.append([event["time"], value])
            label = value
    return {
        "duration": duration,
        "beats": beats,
        "bars": bars,
        "chords": [[s, e, c] for (s, e), c in zip(annotations["chord_intervals"].tolist(), annotations["chord_labels"])],
        "keys": [[s, e, k or "N"] for s, e, k in annotations["keys"]],
        "sections": sections,
        "notes": notes,
    }


def write_page(
    record: dict, audio: Path, out: Path, title: str, embed_audio: bool = False, reference: dict | None = None
) -> None:
    """Write the page of a transcription record.  ``reference`` adds reference beats and bars to compare with:
    ``{"beats": [seconds, ...], "bars": [{"start": s, "end": s or None, "meter": [num, den]}, ...]}``."""

    out.parent.mkdir(parents=True, exist_ok=True)
    data = page_data(record)
    data["title"] = title
    data["reference"] = reference
    if embed_audio:
        kind = mimetypes.guess_type(audio.name)[0] or "audio/mpeg"
        data["audio"] = f"data:{kind};base64," + base64.b64encode(audio.read_bytes()).decode("ascii")
    else:
        try:
            data["audio"] = Path(os.path.relpath(audio.resolve(), out.parent.resolve())).as_posix()
        except ValueError:  # another drive
            data["audio"] = audio.resolve().as_uri()

    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out.write_text(html, encoding="utf-8")


def arguments(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcription", type=Path, help="JSON written by python -m sheetsage2_refined")
    parser.add_argument("--audio", type=Path, help="audio file (default: the one recorded in the transcription)")
    parser.add_argument("--out", type=Path, help="HTML file (default: outputs/visualizations/<name>.html)")
    parser.add_argument("--embed-audio", action="store_true", help="put the audio inside the page instead of linking it")
    return parser


def run(args: argparse.Namespace, reference: dict | None = None) -> Path:
    record = json.loads(args.transcription.read_text(encoding="utf-8"))
    out = args.out or Path("outputs") / "visualizations" / f"{args.transcription.stem}.html"
    write_page(record, args.audio or Path(record["audio"]), out, args.transcription.stem, args.embed_audio, reference)
    return out


def main() -> None:
    print(run(arguments().parse_args()))


if __name__ == "__main__":
    main()
