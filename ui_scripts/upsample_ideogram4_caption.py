"""Upsample a short user idea into a full Ideogram4 structured-JSON caption.

Runs the Ideogram4 generation ("magic prompt") system prompt through
Qwen/Qwen3-VL-8B-Instruct as a text-only request and returns the resulting JSON.
Nothing is written to disk -- the upsampled JSON object is printed to stdout
(progress/logs go to stderr so stdout stays clean for the caller to parse).
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Make the repo importable (e.g. `toolkit.util.quantize`) regardless of cwd.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The generation prompt lives here. It's a `name = """<content>"""` file, but the
# content intentionally contains literal `\uNNNN` and `\n` sequences that are not
# valid Python escapes, so it cannot be imported -- we read the triple-quoted
# content verbatim as text instead.
_PROMPT_PATH = os.path.join(
    REPO_ROOT,
    "extensions_built_in",
    "captioner",
    "prompts",
    "ideogram4_upsample_prompt.py",
)

# Swapped into the prompt's {{mode_directive}} slot. Both keep the FIDELITY rules;
# they only differ on how much the model may expand beyond the literal prompt.
FAITHFUL_DIRECTIVE = (
    "- **Fill in ONLY what the structure needs.** Add a concrete background shell, "
    "bounding boxes, and the required elements/text -- nothing else. Do NOT add new "
    "subjects, props, narrative, mood, or a setting the user did not specify. If the "
    "prompt names no location, keep the background minimal. If the prompt is sparse, "
    "the scene stays sparse."
)

CREATIVE_DIRECTIVE = (
    "- **Expand the scene while keeping the user's idea intact.** Place the subject in "
    "a specific, believable setting and build a real background environment with fitting "
    "secondary details (props, depth layers, atmosphere) that serve the idea -- never a "
    "blank or 'plain' background when a setting can be implied. Everything you add must "
    "support, never replace or contradict, what the user asked for, and you must not "
    "introduce a different main subject. The FIDELITY rules above still hold: triggers "
    "verbatim, no invented appearance for a named person, no elaboration of a named style."
)

DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_generation_prompt() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    # Extract the triple-quoted body verbatim (see note on _PROMPT_PATH).
    start = src.find('"""')
    end = src.rfind('"""')
    if start == -1 or end <= start:
        raise RuntimeError(f"Could not parse prompt body from {_PROMPT_PATH}")
    return src[start + 3 : end]


def build_prompt(
    template: str,
    aspect_ratio: str,
    original_prompt: str,
    creative: bool = False,
    instructions: str = "",
) -> str:
    directive = CREATIVE_DIRECTIVE if creative else FAITHFUL_DIRECTIVE
    prompt = template.replace("{{mode_directive}}", directive)
    prompt = prompt.replace("{{user_instructions}}", instructions.strip() or "None.")
    prompt = prompt.replace("{{aspect_ratio}}", aspect_ratio)
    prompt = prompt.replace("{{original_prompt}}", original_prompt)
    return prompt


def extract_json(raw: str):
    """Pull the JSON object out of the model output, tolerating code fences and
    stray preamble. Returns the parsed dict or None."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def sanitize_bbox(bbox):
    """The generation prompt already emits normalized 0-1000 [y1,x1,y2,x2]. Clamp
    to range, sort each axis pair, coerce to ints (keeps y/x order). Returns the
    cleaned box or None to drop it."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        y1, x1, y2, x2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    y1, y2 = sorted((max(0, min(1000, round(y1))), max(0, min(1000, round(y2)))))
    x1, x2 = sorted((max(0, min(1000, round(x1))), max(0, min(1000, round(x2)))))
    if y2 <= y1 or x2 <= x1:
        return None
    return [y1, x1, y2, x2]


HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def sanitize_palette(palette, max_len):
    """Keep unique, valid hex colors in order, capped to max_len. Returns the
    cleaned list, or None if nothing valid remains (drop the key)."""
    if not isinstance(palette, (list, tuple)):
        return None
    seen = set()
    out = []
    for c in palette:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if not HEX_COLOR_RE.match(c):
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_len:
            break
    return out or None


def sanitize_caption(data: dict) -> dict:
    """Light cleanup: drop any aspect_ratio key (input-only context, not output),
    clean each bbox, and cap color palettes (16 per image, 5 per element)."""
    data.pop("aspect_ratio", None)
    style = data.get("style_description")
    if isinstance(style, dict) and "color_palette" in style:
        pal = sanitize_palette(style["color_palette"], 16)
        if pal is None:
            style.pop("color_palette", None)
        else:
            style["color_palette"] = pal
    decon = data.get("compositional_deconstruction", {})
    elements = decon.get("elements", [])
    if isinstance(elements, list):
        for el in elements:
            if not isinstance(el, dict):
                continue
            if "bbox" in el:
                cleaned = sanitize_bbox(el["bbox"])
                if cleaned is None:
                    el.pop("bbox", None)
                else:
                    el["bbox"] = cleaned
            if "color_palette" in el:
                pal = sanitize_palette(el["color_palette"], 5)
                if pal is None:
                    el.pop("color_palette", None)
                else:
                    el["color_palette"] = pal
    return data


def upsample_one(
    model,
    processor,
    device,
    template,
    idea,
    aspect_ratio,
    gen_kwargs,
    creative=False,
    instructions="",
) -> Optional[dict]:
    """Run one idea through the generation prompt. Returns the cleaned caption
    dict, or None if the model output couldn't be parsed."""
    full_prompt = build_prompt(
        template, aspect_ratio, idea.strip(), creative, instructions
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    data = extract_json(output_text)
    if data is None:
        log("Failed to parse JSON from model output. Raw output follows:")
        log(output_text)
        return None
    return sanitize_caption(data)


def normalize_item(item, default_aspect_ratio):
    """Accept either a bare prompt string or {'prompt': ..., 'aspect_ratio': ...}.
    Returns (idea, aspect_ratio) or None if the item is malformed/empty."""
    if isinstance(item, str):
        idea, aspect_ratio = item, default_aspect_ratio
    elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
        idea = item["prompt"]
        aspect_ratio = item.get("aspect_ratio") or default_aspect_ratio
    else:
        return None
    if not idea.strip():
        return None
    return idea, aspect_ratio


def load_model(
    model_name_or_path: str,
    dtype: torch.dtype,
    device: torch.device,
    quantize: bool,
    qtype: str,
):
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        AutoProcessor,
    )

    ModelClass = (
        Qwen3VLMoeForConditionalGeneration
        if "B-A" in model_name_or_path
        else Qwen3VLForConditionalGeneration
    )
    log(f"Loading {model_name_or_path}")
    model = ModelClass.from_pretrained(
        model_name_or_path, dtype=dtype, device_map="cpu"
    )
    if quantize:
        # Lazy import so the common (non-quantized) path needs no toolkit deps.
        from optimum.quanto import freeze
        from toolkit.util.quantize import quantize as quantize_model, get_qtype

        log(f"Quantizing model ({qtype})")
        quantize_model(model, weights=get_qtype(qtype))
        freeze(model)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    return model, processor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upsample a short idea into an Ideogram4 structured-JSON caption."
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="A single user idea to upsample (prints one JSON object).",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help=(
            "JSON list to upsample in one model load (prints a JSON list, same order). "
            'Each item is a prompt string or {"prompt": "...", "aspect_ratio": "W:H"}. '
            "Failed/empty items come back as null."
        ),
    )
    parser.add_argument(
        "--aspect_ratio",
        default="auto",
        help="Default aspect ratio as 'W:H', or 'auto'. Per-item values override it.",
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--qtype", default="float8")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. <= 0 uses greedy decoding.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--creative",
        action="store_true",
        help="Expand the prompt into a populated scene (default: faithful/minimal).",
    )
    parser.add_argument(
        "--instructions",
        default="",
        help="Extra user instructions injected into the system prompt for every item.",
    )
    parser.add_argument("--pretty", action="store_true", help="Indent the output JSON.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Emit one compact JSON line per prompt as it completes "
            '({"index": i, "caption": {...}|null}) instead of a single final list.'
        ),
    )
    args = parser.parse_args()

    if bool(args.prompt) == bool(args.prompts):
        print(
            "Provide exactly one of --prompt or --prompts.", file=sys.stderr, flush=True
        )
        return 2

    # Resolve the work list up front so we can fail fast on bad input.
    if args.prompts is not None:
        try:
            raw_items = json.loads(args.prompts)
        except json.JSONDecodeError as e:
            print(f"Failed to parse --prompts JSON: {e}", file=sys.stderr, flush=True)
            return 2
        if not isinstance(raw_items, list) or len(raw_items) == 0:
            print(
                "--prompts must be a non-empty JSON list.", file=sys.stderr, flush=True
            )
            return 2
        batch = True
    else:
        if not args.prompt.strip():
            print("--prompt must not be empty.", file=sys.stderr, flush=True)
            return 2
        raw_items = [args.prompt]
        batch = False

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    indent = 2 if args.pretty else None

    template = load_generation_prompt()

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature and args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        model, processor = load_model(
            args.model_name_or_path, dtype, device, args.quantize, args.qtype
        )

        results = []
        for idx, item in enumerate(raw_items):
            norm = normalize_item(item, args.aspect_ratio)
            if norm is None:
                log(f"[{idx + 1}/{len(raw_items)}] invalid/empty item, skipping")
                result = None
            else:
                idea, aspect_ratio = norm
                log(
                    f"[{idx + 1}/{len(raw_items)}] Generating (aspect_ratio={aspect_ratio})..."
                )
                result = upsample_one(
                    model,
                    processor,
                    device,
                    template,
                    idea,
                    aspect_ratio,
                    gen_kwargs,
                    args.creative,
                    args.instructions,
                )
            results.append(result)
            # Stream each result on its own compact line so callers can update live.
            if args.stream:
                print(
                    json.dumps({"index": idx, "caption": result}, ensure_ascii=False),
                    flush=True,
                )

    if args.stream:
        return 0 if any(r is not None for r in results) else 1

    if batch:
        print(json.dumps(results, ensure_ascii=False, indent=indent), flush=True)
        # Non-zero only if nothing succeeded.
        return 0 if any(r is not None for r in results) else 1

    if results[0] is None:
        return 1
    print(json.dumps(results[0], ensure_ascii=False, indent=indent), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
