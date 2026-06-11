"""Generate captions for <DATASET_NAME> video chunks from filenames.

# CUSTOMIZE: Replace this docstring with the dataset-specific rationale.
# Use this template when you already know what each clip depicts and want
# deterministic captions instead of letting Gemini caption every clip.
# Common reason: the source videos are pre-labeled by content (one source
# video per concept), and split_videos_to_chunks.py has produced
# <descriptor>_chunk_NNN.mp4 files. All chunks of the same source share
# the same caption.

Each .mp4 in the dataset folder has a filename pattern like:
    <descriptor>_chunk_<NNN>.mp4

The descriptor encodes what the clip depicts. This script parses the
filename, looks up the caption template, and writes <basename>.txt next
to each .mp4 with the caption ending in the literal trigger word.

Captions describe SUBJECT + MOTION only. All style attributes are
intentionally omitted so they bind to the trigger.

Caption pattern: "<subject> <motion>, <TRIGGER>"

Usage:

    # default: dataset path baked in, trigger baked in
    python scripts/caption_<NAME>_videos_from_filename.py

    # overwrite existing .txt files
    python scripts/caption_<NAME>_videos_from_filename.py --overwrite

    # custom dataset or trigger
    python scripts/caption_<NAME>_videos_from_filename.py \\
        --dataset /custom/path --trigger MY_TRIGGER

    # dry run — print what would be written without writing files
    python scripts/caption_<NAME>_videos_from_filename.py --dry-run

Add new filename → caption mappings to CAPTION_MAP if the dataset grows.
The script reports any video whose descriptor isn't in the map so you
can add the mapping and re-run with --overwrite.

No external dependencies — pure stdlib.
"""

import argparse
import re
import sys
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}


# CUSTOMIZE: Map the filename descriptor (the part before "_chunk_NNN") to
# the caption text that goes before the trigger. Each caption describes
# subject + motion only — no style, no setting, no texture, no color.
#
# All chunks of the same source video share the same caption since they
# all depict the same subject+motion (just different time slices of it).
CAPTION_MAP = {
    "<descriptor_1>": "<caption-prefix-for-descriptor-1>",
    "<descriptor_2>": "<caption-prefix-for-descriptor-2>",
    # ... add one entry per source video
}

# Matches "<descriptor>_chunk_<digits>" — the trailing chunk index is dropped
CHUNK_PATTERN = re.compile(r"^(.+)_chunk_\d+$")


def parse_descriptor(stem: str):
    m = CHUNK_PATTERN.match(stem)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser()
    # CUSTOMIZE: bake in the dataset path
    parser.add_argument("--dataset", default="<DEFAULT_DATASET_PATH>",
                        help="Path to video chunk folder")
    # CUSTOMIZE: bake in the trigger
    parser.add_argument("--trigger", default="<TRIGGER>",
                        help="Literal trigger word appended to every caption. "
                             "Must match trigger_word in your YAML config.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-write existing .txt files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without writing files")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print(f"Dataset path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    # skip dotfiles (e.g. macOS ._* AppleDouble resource forks, .DS_Store)
    videos = sorted(
        p for p in root.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        print(f"No videos found in {root}", file=sys.stderr)
        sys.exit(1)

    written = 0
    skipped = 0
    unmapped = []

    for video in videos:
        descriptor = parse_descriptor(video.stem)
        if descriptor is None or descriptor not in CAPTION_MAP:
            unmapped.append(video.name)
            continue

        txt = video.with_suffix(".txt")
        if txt.exists() and not args.overwrite:
            skipped += 1
            continue

        caption = f"{CAPTION_MAP[descriptor]}, {args.trigger}"

        if args.dry_run:
            print(f"  [dry-run] {video.name} → {caption}")
        else:
            txt.write_text(caption + "\n", encoding="utf-8")
            written += 1

    if args.dry_run:
        print(f"\nDry-run complete. {len(videos) - len(unmapped)} would be written, "
              f"{len(unmapped)} unmapped.")
    else:
        print(f"Wrote {written} captions, skipped {skipped} existing.")

    if unmapped:
        print(f"\n{len(unmapped)} files had no descriptor mapping (caption NOT written):")
        for name in unmapped[:15]:
            print(f"  {name}")
        if len(unmapped) > 15:
            print(f"  ... and {len(unmapped) - 15} more")
        print("\nAdd their descriptor → caption mapping to CAPTION_MAP and re-run.")


if __name__ == "__main__":
    main()
