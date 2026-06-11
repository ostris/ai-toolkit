"""Caption video clips for a MOTION LoRA via Gemini (first-frame strategy).

# CUSTOMIZE: Replace this docstring with the dataset-specific rationale.
# Explain what motion the clips share, why first-frame captioning is the
# right strategy (we want motion to bind to the trigger, so caption only
# the static "before" state and never describe what happens later).

Strategy: extract the first frame of each clip and caption that as a
static image. The first frame shows the "before" state of the morph —
the subject we want to vary at inference, not the motion we want to
bind to the trigger. Captioning later frames or the whole video would
risk Gemini describing the action that should remain hidden.

Avoid-list is heavy on motion verbs (specific to this dataset's motion),
all time-evolution descriptors, anticipatory language, and video
meta-language.

Usage:

    export GEMINI_API_KEY="..."
    python scripts/caption_<NAME>_motion_first_frame_gemini.py \\
        --dataset /path/to/clips/

    # use a different Gemini model if quota-limited
    python scripts/caption_<NAME>_motion_first_frame_gemini.py \\
        --dataset /path/to/clips --model gemini-2.5-pro

    # re-caption everything (overwrites existing .txt)
    python scripts/caption_<NAME>_motion_first_frame_gemini.py \\
        --dataset /path/to/clips --overwrite

Captions are written as <clip>.txt next to each video file.

Install: pip install google-genai pillow tqdm opencv-python
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


# CUSTOMIZE: Motion-specific avoid list. Group by motion family. Add ALL
# time-evolution / anticipatory language and ALL video meta-language. The
# generic groups at the bottom (motion / time / vibe / video) are universal
# and should stay. The top groups change per dataset's motion.
# See references/avoid-words-cookbook.md for the full discipline.
AVOID_WORDS = (
    # CUSTOMIZE: motion family A (e.g. melt / slump / drip)
    "<motion-family-A-verbs>, "
    # CUSTOMIZE: motion family B (e.g. spread / merge / fuse)
    "<motion-family-B-verbs>, "
    # CUSTOMIZE: motion family C (e.g. transform / morph / change)
    "<motion-family-C-verbs>, "
    # universal — keep these for all motion datasets
    # motion / time descriptors
    "motion, moving, moves, moved, movement, animate, animated, animation, "
    "time-lapse, timelapse, time lapse, gradual, gradually, slowly, slow, "
    "eventually, after, before, during, over time, throughout, "
    "begin, beginning, begins, started, starts, end, ending, ends, finished, "
    "progress, progression, progressing, sequence, "
    # general "process happening" cues
    "process, action, dynamic, fluid, "
    # anticipatory language
    "about to, ready to, on the verge of, "
    # video-frame meta-language we don't want either
    "video, clip, footage, scene, shot, take, frame, still, "
    # vibe / hedging that creeps in
    "appears to be, seems like, looks like, as if, "
    "beautiful, striking, abstract, interesting, atmospheric, ethereal, mood"
)


SYSTEM_PROMPT_TEMPLATE = """You are captioning the FIRST FRAME of video clips for a motion LoRA training dataset. Every clip in this dataset shows the same <MOTION_DESCRIPTION> applied to <SUBJECT_DESCRIPTION>. The MOTION must remain invisible in your captions so it binds to the trigger word and fires automatically at inference.

You are looking at a STILL FRAME — the initial state before any motion has begun. Describe ONLY what is visible as a static composition. Do not describe what happens later in the clip. Do not describe motion, change, transformation, or any time-evolution. The clip's motion is what we want the model to learn implicitly via omission.

## Your one core rule

DESCRIBE the literal subject as a still image — <list the static variables: subject, count, shape, arrangement, position, surface texture, background>. Treat this as if you were captioning a frozen photograph that has no implied "next moment."

DO NOT describe motion in any form: not <list the dataset's motion verbs>, or any verb that implies change. Even anticipatory language like "about to <X>" or "ready to <Y>" is forbidden.

## ALWAYS describe (these become content variables a user can prompt at inference)

- Subject: <list valid subject vocabulary for this dataset>
- Count: a single / two / three / several / multiple / a cluster of
- Arrangement and position: <list valid spatial vocabulary>
- Shape qualities: <list valid shape vocabulary>
- Static surface texture: <list valid texture words — only if visible AS A SURFACE, never describing it as moving>
- Color of the subject and background
- Composition / framing: full frame, wide, close-up, centered, asymmetric

## NEVER describe (these must bind to the trigger via OMISSION)

Motion vocabulary, transformation language, time-evolution descriptors. Do not use any of these or close synonyms:

    {avoid_words}

Also avoid:
- Hedging: "appears to be", "seems like", "looks like" — state what is visible
- Quality / aesthetic: "beautiful", "striking", "interesting", "abstract artwork"
- Mood / vibe: "moody", "atmospheric", "ethereal"
- Anticipatory language: "about to", "ready to", "on the verge of"
- Video meta-language: "video", "clip", "footage", "scene", "frame" — describe the SUBJECT, not the artifact

## Format rules

- Natural-language prose. Phrases separated by commas is fine. No comma-tag-soup.
- One sentence is usually enough; two if there are multiple distinct elements. 15-35 words.
- Lowercase. No quotes around the caption.
- State only what is visible in this single frame. Do not invent or infer motion.

## VARY THE CAPTION across the dataset

You will caption many similar-looking clips. Vary your phrasing across captions even when subjects look alike — different ways to describe count and arrangement, different shape vocabulary. The LoRA needs to see the subject described as a *variable* (multiple wordings) so it doesn't memorize one canonical phrase.

## HARD REQUIREMENT: trigger suffix

Every caption MUST end with exactly this phrase, verbatim:

    {trigger}

End the caption with a comma and a space, then this exact phrase, then nothing.

## Examples (trigger used in examples is "{trigger}")

GOOD: <example 1 — describes static first-frame subject in detail>, {trigger}

GOOD: <example 2 — different subject configuration>, {trigger}

GOOD: <example 3 — different layout>, {trigger}

GOOD: <example 4 — variation in count or shape>, {trigger}

GOOD: <example 5 — another configuration>, {trigger}

BAD: <example using "about to <motion>"> , {trigger}    (anticipatory motion)

BAD: <example describing the dataset's actual motion verb>, {trigger}    (uses motion vocabulary)

BAD: <example with vibe / atmosphere descriptors>, {trigger}

BAD: a still from a video clip showing <subject>, {trigger}    (uses video meta-language)

## Your output

Respond with ONLY the caption text. No preamble, no explanation, no quotes, no formatting. One caption, ready to be saved as a .txt file."""


def build_system_prompt(trigger: str, avoid_words: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(trigger=trigger, avoid_words=avoid_words)


def find_videos(root: Path):
    # skip dotfiles (e.g. macOS ._* AppleDouble resource forks, .DS_Store)
    return sorted(
        p for p in root.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in VIDEO_EXTS
    )


def extract_first_frame(video_path: Path) -> Image.Image:
    """Extract first frame as a PIL RGB Image."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    # cv2 returns BGR, PIL expects RGB
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def caption_one(client: genai.Client, model: str, video_path: Path,
                system_prompt: str, trigger: str) -> str:
    image = extract_first_frame(video_path)
    response = client.models.generate_content(
        model=model,
        contents=[image, "Caption this frame."],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
        ),
    )
    text = (response.text or "").strip().strip('"').strip("'")
    if not text:
        raise RuntimeError("empty response")
    # belt-and-suspenders: enforce trigger suffix in case Gemini drifts
    if not text.rstrip(".").rstrip().endswith(trigger):
        text = text.rstrip(".").rstrip().rstrip(",").rstrip() + ", " + trigger
    return text


def caption_with_retry(client, model, video_path, system_prompt, trigger, max_attempts=4):
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return caption_one(client, model, video_path, system_prompt, trigger)
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(delay)
            delay *= 2


def main():
    parser = argparse.ArgumentParser()
    # CUSTOMIZE: bake in this dataset's path so the user can run with no flags
    parser.add_argument("--dataset", default="<DEFAULT_DATASET_PATH>",
                        help="Path to folder of video clips (.mp4 / .mov / etc.)")
    # CUSTOMIZE: bake in the trigger word (must match YAML trigger_word)
    parser.add_argument("--trigger", default="<TRIGGER>",
                        help="Literal trigger word appended to every caption. "
                             "Must match trigger_word in your YAML config.")
    parser.add_argument("--avoid-words", default=AVOID_WORDS,
                        help="Comma-separated motion / time words the captioner must NOT use")
    parser.add_argument("--model", default="gemini-3.1-pro-preview",
                        help="Gemini model id. Fallbacks if quota-limited: "
                             "gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-caption existing .txt files")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    root = Path(args.dataset)
    if not root.exists():
        print(f"Dataset path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    videos = find_videos(root)
    if not videos:
        print(f"No videos found in {root}", file=sys.stderr)
        sys.exit(1)

    jobs = []
    for video in videos:
        txt = video.with_suffix(".txt")
        if txt.exists() and not args.overwrite:
            continue
        jobs.append((video, txt))

    print(f"Found {len(videos)} videos, {len(jobs)} need captions.")
    if not jobs:
        return

    system_prompt = build_system_prompt(args.trigger, args.avoid_words)

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(caption_with_retry, client, args.model, v, system_prompt, args.trigger): (v, t)
            for v, t in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Captioning"):
            video, txt = futures[fut]
            try:
                caption = fut.result()
                txt.write_text(caption + "\n", encoding="utf-8")
            except Exception as e:
                errors.append((video, str(e)))

    if errors:
        print(f"\n{len(errors)} videos failed:")
        for video, msg in errors[:10]:
            print(f"  {video.name}: {msg}")


if __name__ == "__main__":
    main()
