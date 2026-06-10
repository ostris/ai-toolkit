"""Caption a STYLE LoRA dataset with Gemini.

# CUSTOMIZE: Replace this docstring with the dataset-specific rationale.
# Explain what the LoRA is learning, what's the SUBJECT (content / variable)
# vs. the STYLE (binds to trigger via omission), and why each major group of
# words appears in the avoid list. Future-you reads this when training a
# successor LoRA.

Designed for local use. Writes <image>.txt next to each image with a caption
that describes the *content* of the image and OMITS anything that describes
the visual style itself. Every caption ends with the literal style suffix
you pass via --suffix, which is the trigger word the LoRA will bind the
aesthetic to at inference.

Usage:

    export GEMINI_API_KEY="..."
    python scripts/caption_<DATASET_NAME>_dataset_gemini.py \\
        --dataset "/path/to/images" \\
        --suffix "<TRIGGER>"

Run it twice if needed — it skips images that already have a .txt file.
Pass --overwrite to re-caption.

Install: pip install google-genai pillow tqdm
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# CUSTOMIZE: Style-specific avoid list. Anything here is a style attribute
# we want the LoRA to learn via *omission* — describing it in captions
# teaches the model these words are promptable, weakening the trigger.
# Group by category with comment headers so future edits are easy.
# See references/avoid-words-cookbook.md for the full discipline.
AVOID_WORDS = (
    # medium / process vocabulary
    "<medium-words>, "
    # texture / surface effects
    "<texture-words>, "
    # color / tone words (the dataset's palette — must bind to trigger)
    "<color-words>, "
    # mood / vibe / aesthetic descriptors
    "<mood-words>, "
    # generic style meta-language
    "stylized, aesthetic, vibe, mood, atmospheric, cinematic"
)


SYSTEM_PROMPT_TEMPLATE = """You are captioning images for a <STYLE_NAME> style LoRA. Every training image is <ONE-SENTENCE DESCRIPTION OF WHAT THE DATASET SHOWS>. The <STYLE_NAME> aesthetic — <list the specific visual traits: palette, technique, edges, surface, mood> — is the STYLE that must remain invisible in your captions so it binds to the trigger word.

The <SUBJECT_DESCRIPTION> is the SUBJECT — describe it thoroughly so it does NOT bake into the trigger.

This separation is the entire point: <SUBJECT> = content (promptable), <STYLE> = automatic. If you confuse the two, the user will not be able to prompt "<example prompt>, <trigger>" and get a <STYLE_NAME>-style result.

## Your one core rule

DESCRIBE the literal subject and its arrangement in the frame. <List the specific variables: count, shape, type, position, layout, etc.>

DO NOT describe color, tone, texture, medium, edges, lighting quality, mood, or process.

## ALWAYS describe (these become user-promptable variables)

- <Variable 1: e.g. count of subjects>
- <Variable 2: e.g. subject type / species / kind>
- <Variable 3: e.g. layout / framing / composition>
- <Variable 4: e.g. size relationships>
- <Variable 5: e.g. orientation>
- <Variable 6: e.g. completeness — whole vs fragment>
- <Variable 7: e.g. non-subject elements visible>

## NEVER describe (these must bind to the trigger via OMISSION)

Style, medium, palette, technique, surface texture. Do not use any of these or close synonyms:

    {avoid_words}

Also avoid:
- Hedging: "appears to be", "seems like", "looks like" — state what is visible
- Quality / aesthetic descriptors: "beautiful", "striking", "delicate", "ethereal"
- Background / substrate descriptions beyond minimal — do NOT describe color, texture, or medium of the background

## Format rules

- Natural-language prose. Phrases separated by commas is fine. No comma-tag-soup.
- One sentence is usually enough; two is fine if there is a lot of detail. 15-40 words.
- State only what is visible. Do not invent details you cannot see.
- Lowercase. No quotes around the caption.

## HARD REQUIREMENT: style suffix

Every caption MUST end with exactly this phrase, verbatim:

    {suffix}

End the caption with a comma and a space, then this exact phrase, then nothing.

## Examples (suffix used in examples is "{suffix}")

GOOD: <example 1 — describes subject in detail, no style words>, {suffix}

GOOD: <example 2 — different subject configuration>, {suffix}

GOOD: <example 3 — different layout>, {suffix}

GOOD: <example 4 — edge case showing variation>, {suffix}

GOOD: <example 5 — another configuration>, {suffix}

BAD: <example using forbidden style words>, {suffix}    (uses "<word1>", "<word2>" — those are the style)

BAD: <example using vibe descriptors>, {suffix}    (uses "beautiful", "ethereal" — vibe)

BAD: <minimal/thin example>, {suffix}    (no detail — the LoRA cannot learn the subject is a *variable* if every caption is this thin)

## Your output

Respond with ONLY the caption text. No preamble, no explanation, no quotes, no formatting. One caption, ready to be saved as a .txt file."""


def build_system_prompt(suffix: str, avoid_words: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(suffix=suffix, avoid_words=avoid_words)


def find_images(root: Path):
    # skip dotfiles (e.g. macOS ._* AppleDouble resource forks, .DS_Store)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTS
    )


def caption_one(client: genai.Client, model: str, image_path: Path, system_prompt: str, suffix: str) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        response = client.models.generate_content(
            model=model,
            contents=[im, "Caption this image."],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
            ),
        )
    text = (response.text or "").strip().strip('"').strip("'")
    if not text:
        raise RuntimeError("empty response")
    if not text.rstrip(".").rstrip().endswith(suffix):
        text = text.rstrip(".").rstrip().rstrip(",").rstrip() + ", " + suffix
    return text


def caption_with_retry(client, model, image_path, system_prompt, suffix, max_attempts=4):
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return caption_one(client, model, image_path, system_prompt, suffix)
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(delay)
            delay *= 2


def main():
    parser = argparse.ArgumentParser()
    # CUSTOMIZE: bake in this dataset's path so the user can run with no flags
    parser.add_argument("--dataset", default="<DEFAULT_DATASET_PATH>",
                        help="Path to image folder")
    # CUSTOMIZE: bake in the trigger. Use real-word + leetspeak pairing if
    # the style benefits from a semantic prior (e.g. "chemigram print, 1ll6m3ns")
    parser.add_argument("--suffix", default="<TRIGGER>",
                        help="Literal trigger appended to every caption. "
                             "Must match trigger_word in your YAML config.")
    parser.add_argument("--avoid-words", default=AVOID_WORDS,
                        help="Comma-separated list of style/process words the captioner must NOT use")
    parser.add_argument("--model", default="gemini-3.1-pro-preview",
                        help="Gemini model id. Fallbacks if quota-limited: "
                             "gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Re-caption existing .txt files")
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
    images = find_images(root)
    if not images:
        print(f"No images found in {root}", file=sys.stderr)
        sys.exit(1)

    jobs = []
    for img in images:
        txt = img.with_suffix(".txt")
        if txt.exists() and not args.overwrite:
            continue
        jobs.append((img, txt))

    print(f"Found {len(images)} images, {len(jobs)} need captions.")
    if not jobs:
        return

    system_prompt = build_system_prompt(args.suffix, args.avoid_words)

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(caption_with_retry, client, args.model, img, system_prompt, args.suffix): (img, txt)
            for img, txt in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Captioning"):
            img, txt = futures[fut]
            try:
                caption = fut.result()
                txt.write_text(caption + "\n", encoding="utf-8")
            except Exception as e:
                errors.append((img, str(e)))

    if errors:
        print(f"\n{len(errors)} images failed:")
        for img, msg in errors[:10]:
            print(f"  {img.name}: {msg}")


if __name__ == "__main__":
    main()
