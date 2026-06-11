"""Wide factual pass over an AI Toolkit LoRA run's sample images, using Gemini.

This is the cheap, high-coverage half of the sample-reviewer workflow. It looks
at EVERY sample (every checkpoint x every prompt) and emits one structured
record per image into a single `sample_review.json`. Opus then reads that JSON
(text, cheap) to build the trajectory and pick finalists, and only directly
eyeballs the 2-3 finalist checkpoints itself.

Division of labour (this is the whole point):
  - Gemini  = wide + shallow: extract concrete facts from every image.
  - Opus    = narrow + deep: synthesize the trajectory + render the FINAL
              aesthetic verdict on a handful of finalists.

Gemini is NOT asked to pick a winner. It describes one image at a time against
the dataset ground-truth spec you pass in. The "is this the peak vs. its
neighbour / did the artist's mark-making survive" judgment stays on Opus, by
design -- a cheap VLM's "looks great" has masked mode-collapse before.

Usage:

    export GEMINI_API_KEY="..."
    python scripts/review_samples_gemini.py \\
        --config output/my_run/my_run.yaml \\
        --ground-truth /tmp/my_run_ground_truth.txt \\
        --goal style \\
        --mode quality          # or --mode fast

`--mode` picks the underlying Gemini model:
  - `quality` (default) = gemini-3.1-pro-preview. ~5 min/checkpoint. Best for
    abstract or multi-material styles where subtle material reads matter.
  - `fast` = gemini-3.1-flash-lite. ~10x cheaper and faster. Fine for clean
    photo styles or character LoRAs. Use --mode fast when you already know
    what to look for and just want to locate the candidate band.

`--ground-truth` is a plain-text file you (Opus) write first, after looking at
5-8 dataset images: the medium, palette, texture, mark-making, what varies, and
any artist-intent notes from the YAML comments. Every per-image call is anchored
to it. The script refuses to run without it -- a fidelity judgment with no
ground truth is a guess.

Re-run safe: skips images already recorded in the output JSON. Pass --overwrite
to redo them.

Install: pip install google-genai pillow pyyaml tqdm
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# <gen_time_ms>__<step_zero_padded_9>_<count>.<ext>  (see references/output-layout.md)
SAMPLE_PATTERN = re.compile(r"^\d+__(\d{9})_(\d+)\.\w+$")

# Qualitative scale the reviewer rubric uses everywhere. No numbers (deliberate).
SCALE = {"strong", "adequate", "weak", "broken", "n/a"}
CONTROL_SCALE = {"clean", "slight_bleed", "strong_bleed", "n/a"}


SYSTEM_PROMPT_TEMPLATE = """You are extracting structured facts from a single AI Toolkit LoRA training \
sample image so a downstream reviewer can pick the best checkpoint. You are doing FACTUAL EXTRACTION, \
not judging which checkpoint wins. Describe THIS ONE image against the dataset ground truth. Report \
concrete, observable facts -- never vibes, praise, or "kit density".

This LoRA's goal type is: {goal}
The trigger word is: {trigger}

## DATASET GROUND TRUTH (what a faithful result should look like)

{ground_truth}

## How triggered vs. control prompts work

- A TRIGGERED prompt contains the trigger word: the trained style/identity SHOULD appear.
- A CONTROL prompt does NOT contain the trigger: it SHOULD look like the plain base model. If a control \
image picks up the dataset's palette, texture, composition habits, or specific subjects, that is BLEED \
(over-baking). Bleed is the single most reliable over-training signal -- report it precisely.

## Output: a single JSON object with EXACTLY these keys

- "fidelity": one of ["strong","adequate","weak","broken","n/a"] -- for a TRIGGERED image, how strongly \
the trained style/identity from the ground truth appears. Use "n/a" for control prompts.
- "subject_match": one of ["strong","adequate","weak","broken"] -- does the image actually render the \
SUBJECT the prompt asked for? (A bicycle prompt must still yield a bicycle.) This catches generalization \
failures and memorization substitutions.
- "texture_fidelity": one of ["strong","adequate","weak","broken","n/a"] -- does the dataset's \
texture/grain/brushwork/material surface actually appear? "n/a" for controls or styles without strong \
texture. (Late checkpoints usually get this; early ones miss it.)
- "palette_match": one of ["strong","adequate","weak","n/a"] -- does the colour palette match the ground \
truth? "n/a" for controls.
- "control_clean": one of ["clean","slight_bleed","strong_bleed","n/a"] -- for a CONTROL image only: \
does it look like the untouched base model ("clean") or has it absorbed dataset traits? Use "n/a" for \
triggered prompts.
- "bleed_signs": array of short strings naming concrete absorbed traits (e.g. "warm dataset palette", \
"halftone grain", "recurring bust object", "centered-figure habit"). Empty array if none. Fill this for \
controls; for triggered images use it only to note clearly unprompted dataset traits.
- "gibberish_text": boolean -- are there hallucinated or garbled letterforms / fake text in the image?
- "dataset_subject_leak": boolean -- does a SPECIFIC recurring object from the dataset appear even though \
this prompt did not ask for it? (e.g. the dataset's signature bust showing up in a "red bicycle" prompt.)
- "composition": a short factual layout descriptor, <= 12 words, no adjectives of quality. Examples: \
"centered single figure, frontal, plain ground", "wide landscape, low horizon, object lower-left". The \
reviewer uses these to detect compositional repetition across different prompts (memorization).
- "artifacts": a short string naming concrete defects ("melted hands", "duplicated limbs", "blurred"), \
or "" if none.
- "notes": one short line (<= 25 words) of any other concrete observation. No praise, no hedging.

Respond with ONLY the JSON object. No preamble, no markdown fences, no explanation."""


def load_config(path: Path) -> dict:
    """Load a YAML/JSON ai-toolkit config. (yaml.safe_load also parses JSON.)"""
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        # tolerate JSONC-style // comments, then fall back to JSON
        stripped = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("//")
        )
        return json.loads(stripped)


def get_process(config: dict) -> dict:
    return config["config"]["process"][0]


def extract_prompts(process: dict) -> list:
    sample_cfg = process.get("sample") or {}
    # Older configs: sample.prompts (list[str]). Klein 9B / modern configs:
    # sample.samples (list[dict] with "prompt" + optional "ctrl_img").
    raw = sample_cfg.get("prompts")
    if raw is None:
        raw = sample_cfg.get("samples")
    if raw is None:
        raw = process.get("samples")  # legacy top-level fallback
    if raw is None:
        raw = []
    prompts = []
    for entry in raw:
        if isinstance(entry, dict):
            prompts.append(str(entry.get("prompt", "")))
        else:
            prompts.append(str(entry))
    return prompts


def render_prompt(prompt: str, trigger: str) -> str:
    return prompt.replace("[trigger]", trigger) if trigger else prompt


def is_control(prompt: str, trigger: str) -> bool:
    if "[trigger]" in prompt:
        return False
    if trigger and trigger in prompt:
        return False
    return True


def find_samples(samples_dir: Path):
    """Return {step: {prompt_index: path}} for all parseable, non-dotfile samples."""
    by_step = {}
    for p in sorted(samples_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = SAMPLE_PATTERN.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        idx = int(m.group(2))
        by_step.setdefault(step, {})[idx] = p
    return by_step


def normalize_record(raw: dict) -> dict:
    """Coerce Gemini's object into the fixed schema, defaulting anything missing."""
    def pick(key, allowed, default):
        v = str(raw.get(key, default)).strip().lower()
        return v if v in allowed else default

    signs = raw.get("bleed_signs", [])
    if not isinstance(signs, list):
        signs = [str(signs)] if signs else []

    return {
        "fidelity": pick("fidelity", SCALE, "n/a"),
        "subject_match": pick("subject_match", SCALE - {"n/a"}, "weak"),
        "texture_fidelity": pick("texture_fidelity", SCALE, "n/a"),
        "palette_match": pick("palette_match", {"strong", "adequate", "weak", "n/a"}, "n/a"),
        "control_clean": pick("control_clean", CONTROL_SCALE, "n/a"),
        "bleed_signs": [str(s) for s in signs][:8],
        "gibberish_text": bool(raw.get("gibberish_text", False)),
        "dataset_subject_leak": bool(raw.get("dataset_subject_leak", False)),
        "composition": str(raw.get("composition", ""))[:120],
        "artifacts": str(raw.get("artifacts", ""))[:120],
        "notes": str(raw.get("notes", ""))[:200],
    }


def extract_json(text: str) -> dict:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def analyze_one(client, model, image_path, system_prompt, rendered_prompt,
                control, trigger, max_side):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        role = "CONTROL prompt (no trigger -- should look like base model)" if control \
            else "TRIGGERED prompt (trained style/identity should appear)"
        user_text = (
            f"This image was generated from a {role}.\n"
            f"Rendered prompt sent to the model: {rendered_prompt!r}\n"
            f"Extract the JSON facts now."
        )
        response = client.models.generate_content(
            model=model,
            contents=[im, user_text],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
    return normalize_record(extract_json(response.text))


def analyze_with_retry(client, model, image_path, system_prompt, rendered_prompt,
                       control, trigger, max_side, max_attempts=4):
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return analyze_one(client, model, image_path, system_prompt,
                               rendered_prompt, control, trigger, max_side)
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(delay)
            delay *= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="Path to the run's config YAML (e.g. output/<run>/<run>.yaml). "
                             "Used for trigger word, prompts, run folder.")
    parser.add_argument("--ground-truth", required=True,
                        help="Path to a plain-text dataset ground-truth spec you wrote after "
                             "looking at the dataset. Anchors every fidelity judgment.")
    parser.add_argument("--goal", required=True, choices=["style", "character", "combined"],
                        help="LoRA goal type. Selects how 'fidelity' is framed.")
    parser.add_argument("--samples-dir", default=None,
                        help="Override the samples folder (default: <training_folder>/<name>/samples).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <run folder>/sample_review.json).")
    parser.add_argument("--mode", choices=["quality", "fast"], default="quality",
                        help="Speed-vs-quality preset for the Gemini wide pass. "
                             "'quality' (default) = gemini-3.1-pro-preview — strongest material/texture "
                             "reads on abstract or multi-mode styles; ~5 min per checkpoint × prompt grid. "
                             "'fast' = gemini-3.1-flash-lite — ~10× cheaper and faster, fine for the "
                             "factual extraction on clean photo-style or character LoRAs, may miss "
                             "subtle material variants on harder styles. Overridden if --model is "
                             "passed explicitly. Per [Gemini captioner fallback model] memory: use "
                             "flash-lite (NOT flash-preview, which 404s on v1beta).")
    parser.add_argument("--model", default=None,
                        help="Override the Gemini model id explicitly. Bypasses --mode. Valid: "
                             "gemini-3.1-pro-preview, gemini-3.1-flash-lite, gemini-2.5-pro, "
                             "gemini-2.5-flash. Use --mode unless you need a non-preset model.")
    parser.add_argument("--max-side", type=int, default=1024,
                        help="Downscale longest image side to this before sending (cuts Gemini cost; "
                             "most samples are already <=1024). 0 disables.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the number of images analyzed this run (0 = no cap). "
                             "Handy for a cheap dry-run; combine with re-runs to fill in the rest.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-analyze images already present in the output JSON.")
    args = parser.parse_args()

    # Resolve --mode preset to a concrete model id. --model wins if passed.
    MODE_MODELS = {
        "quality": "gemini-3.1-pro-preview",
        "fast": "gemini-3.1-flash-lite",
    }
    if args.model is None:
        args.model = MODE_MODELS[args.mode]

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running.", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"Ground-truth file not found: {gt_path}\n"
              "Write a short dataset spec (medium, palette, texture, what varies, "
              "artist intent) and pass it with --ground-truth.", file=sys.stderr)
        sys.exit(1)
    ground_truth = gt_path.read_text(encoding="utf-8").strip()
    if not ground_truth:
        print("Ground-truth file is empty.", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    process = get_process(config)
    trigger = process.get("trigger_word") or ""
    prompts = extract_prompts(process)
    name = config["config"]["name"]
    training_folder = process.get("training_folder", "output")

    run_folder = Path(training_folder) / name
    samples_dir = Path(args.samples_dir) if args.samples_dir else run_folder / "samples"
    if not samples_dir.exists():
        print(f"Samples folder not found: {samples_dir}", file=sys.stderr)
        sys.exit(1)
    output_path = Path(args.output) if args.output else run_folder / "sample_review.json"

    control_indices = [i for i, p in enumerate(prompts) if is_control(p, trigger)]

    by_step = find_samples(samples_dir)
    if not by_step:
        print(f"No parseable samples in {samples_dir}", file=sys.stderr)
        sys.exit(1)

    # Resume: load any prior results so reruns are cheap.
    existing = {}
    if output_path.exists() and not args.overwrite:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8")).get("checkpoints", {})
        except Exception:
            existing = {}

    jobs = []  # (step, idx, path, rendered_prompt, control)
    for step in sorted(by_step):
        for idx in sorted(by_step[step]):
            if str(idx) in existing.get(str(step), {}) and not args.overwrite:
                continue
            prompt = prompts[idx] if idx < len(prompts) else ""
            jobs.append((step, idx, by_step[step][idx],
                         render_prompt(prompt, trigger), is_control(prompt, trigger)))

    if args.limit and len(jobs) > args.limit:
        jobs = jobs[:args.limit]

    total_imgs = sum(len(v) for v in by_step.values())
    print(f"Run: {name}  |  goal: {args.goal}  |  trigger: {trigger or '(none)'}")
    print(f"Checkpoints: {len(by_step)}  |  prompts: {len(prompts)}  |  "
          f"control indices: {control_indices}")
    print(f"Sample images: {total_imgs}  |  to analyze: {len(jobs)}"
          f"{' (capped by --limit)' if args.limit else ''}  |  mode: {args.mode}  |  model: {args.model}")
    if not jobs:
        print("Nothing to do. (Use --overwrite to redo.)")
        return

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        goal=args.goal, trigger=trigger or "(none)", ground_truth=ground_truth)

    checkpoints = {str(s): dict(existing.get(str(s), {})) for s in by_step}
    meta = {
        "config": str(config_path),
        "run_folder": str(run_folder),
        "samples_dir": str(samples_dir),
        "model": args.model,
        "goal": args.goal,
        "trigger_word": trigger,
        "control_prompt_indices": control_indices,
        "prompts": {str(i): render_prompt(p, trigger) for i, p in enumerate(prompts)},
    }

    client = genai.Client(api_key=api_key)
    lock = threading.Lock()

    def flush():
        with lock:
            # Only persist steps that actually have records; an absent step means
            # "not analyzed yet" (every analyzed image always yields a record), so
            # empty keys would falsely read as "analyzed, found nothing".
            populated = {s: v for s, v in checkpoints.items() if v}
            output_path.write_text(
                json.dumps({"meta": meta, "checkpoints": populated}, indent=2),
                encoding="utf-8")

    errors = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(analyze_with_retry, client, args.model,
                        path, system_prompt, rendered, control, trigger, args.max_side):
            (step, idx, path)
            for (step, idx, path, rendered, control) in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
            step, idx, path = futures[fut]
            try:
                record = fut.result()
                with lock:
                    checkpoints[str(step)][str(idx)] = record
                done += 1
                if done % 20 == 0:
                    flush()
            except Exception as e:
                errors.append((path, str(e)))

    flush()
    print(f"\nWrote {output_path}")
    if errors:
        print(f"{len(errors)} images failed:")
        for path, msg in errors[:10]:
            print(f"  {path.name}: {msg}")


if __name__ == "__main__":
    main()
