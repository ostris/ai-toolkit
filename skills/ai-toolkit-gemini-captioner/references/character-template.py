"""Caption a CHARACTER LoRA dataset with Gemini.

# CUSTOMIZE: Replace this docstring with the dataset-specific rationale.
# Explain who the character is at a high level (no identifying details
# that should bind to the trigger), how many images, what variables the
# captioner is exposing (clothing, pose, setting, style/medium).

Writes <image>.txt next to each image with a caption that describes
variable elements (clothing, pose, setting, style) and omits fixed
character identity (face, eye/hair color, build, age, gender, ethnicity).

The trigger word is the *first word* of every caption. Gemini writes the
literal token "TRIGGER" and the script substitutes the user's trigger.

Usage in Colab:

    !pip install -q google-genai pillow tqdm
    from google.colab import drive, userdata
    drive.mount('/content/drive')

    import os
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")

    !python caption_<NAME>_character_gemini.py \\
        --dataset "/content/drive/MyDrive/path/to/images" \\
        --trigger "<TRIGGER>"

Local usage:

    export GEMINI_API_KEY="..."
    python scripts/caption_<NAME>_character_gemini.py \\
        --dataset "/path/to/images" \\
        --trigger "<TRIGGER>"

Run it twice if needed — it skips images that already have a .txt file.

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

# CUSTOMIZE: Adjust the ALWAYS / NEVER describe lists if this character has
# unusual identity traits (e.g. always wears a specific costume that should
# bind to the trigger, has a distinctive accessory you want baked in, etc.)
SYSTEM_PROMPT = """You are captioning images for a character LoRA training dataset. The character in every image is the same person. Your captions will teach a diffusion model to generate this character across many styles, so what you caption vs. what you omit directly controls what becomes variable vs. baked-in identity.

## Your one core rule

DESCRIBE everything that varies between images. OMIT everything that defines the character's fixed identity.

## ALWAYS describe (these should be variable at inference)

- Clothing, hats, glasses, jewelry, accessories
- Pose, gesture, expression, action
- Background, setting, location, props
- Lighting direction, quality, color temperature
- Camera framing (close-up, medium shot, full body, low angle, etc.)
- Style / medium / format: "photograph", "oil painting", "3d render", "pencil sketch", "watercolor illustration", "anime drawing", "black and white photo", "film photograph", "digital illustration"

## NEVER describe (these must bind to the trigger word)

- Face shape, jawline, cheekbones
- Eye color, eye shape
- Hair color, hair type (only mention hairstyle if it clearly changes between images — e.g., "hair in a bun" vs "hair loose")
- Skin tone, complexion, freckles, moles, scars
- Body type, height, build
- Age descriptors ("young", "middle-aged")
- Gender descriptors ("woman", "man", "girl", "boy") — use the trigger word instead
- Ethnicity

## Format rules

- Start every caption with the literal word: TRIGGER
- Write natural-language prose, not comma-separated tags. Full phrases separated by commas is fine.
- One sentence or two short sentences. Aim for 20–40 words.
- Do not invent details you cannot see. If you aren't sure, leave it out.
- Do not use hedging language ("appears to be", "seems like"). State what's visible.
- Do not include quality descriptors ("high quality", "detailed", "masterpiece", "4k").
- Always end with the style/medium descriptor.

## Examples

Image: character in a cafe holding a coffee cup, shot on film
GOOD: TRIGGER wearing a cream wool sweater, holding a ceramic mug with both hands, sitting at a round wooden cafe table, soft natural window light from the left, shallow depth of field, 35mm film photograph

Image: anime-style illustration of the character running
GOOD: TRIGGER running with arms extended behind, wearing a red jacket and dark pants, speed lines in the background, dynamic low angle, anime illustration

Image: studio headshot against gray backdrop
GOOD: TRIGGER wearing a black turtleneck, looking directly at the camera with a neutral expression, plain gray studio backdrop, soft frontal key light, close-up portrait, studio photograph

Image: oil painting of the character outdoors
GOOD: TRIGGER wearing a blue coat, standing beside a tall oak tree, rolling green hills in the background, overcast diffuse daylight, three quarter view, oil painting

## Your output

Respond with ONLY the caption text. No preamble, no explanation, no quotes, no formatting. One caption, ready to be saved as a .txt file."""


def find_images(root: Path):
    # skip dotfiles (e.g. macOS ._* AppleDouble resource forks, .DS_Store)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTS
    )


def caption_one(client: genai.Client, model: str, image_path: Path, trigger: str) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        response = client.models.generate_content(
            model=model,
            contents=[im, "Caption this image."],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )
    text = (response.text or "").strip().strip('"').strip("'")
    if not text:
        raise RuntimeError("empty response")
    return text.replace("TRIGGER", trigger)


def caption_with_retry(client, model, image_path, trigger, max_attempts=4):
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return caption_one(client, model, image_path, trigger)
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(delay)
            delay *= 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to image folder")
    parser.add_argument("--trigger", required=True,
                        help="Trigger word, e.g. p3r5on. Must match trigger_word in your YAML.")
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

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(caption_with_retry, client, args.model, img, args.trigger): (img, txt)
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
