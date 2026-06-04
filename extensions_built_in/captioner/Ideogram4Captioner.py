import json
import re
from math import gcd
from collections import OrderedDict
from typing import Optional

from PIL import Image

from .Qwen3VLCaptioner import Qwen3VLCaptioner
from .prompts.ideogram4_caption_prompt import ideogram4_caption_prompt
import transformers
import logging
import warnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# The deconstruction JSON is long. 128 tokens (base default) truncates it badly,
# so enforce a sane floor for this captioner unless the user asked for more.
MIN_NEW_TOKENS = 3072

# Largest denominator allowed when snapping a real image's aspect ratio to a
# clean W:H. Keeps captions in the same small-denominator ratio distribution the
# generator was trained on, instead of ugly fractions like 1023:768.
MAX_AR_DENOMINATOR = 16


class Ideogram4Captioner(Qwen3VLCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Ideogram4Captioner, self).__init__(process_id, job, config, **kwargs)
        if self.caption_config.max_new_tokens < MIN_NEW_TOKENS:
            print(
                f"[Ideogram4Captioner] Raising max_new_tokens "
                f"{self.caption_config.max_new_tokens} -> {MIN_NEW_TOKENS} "
                f"(the deconstruction JSON is long)."
            )
            self.caption_config.max_new_tokens = MIN_NEW_TOKENS

    def compute_aspect_ratio(self, width: int, height: int) -> str:
        """Return a clean 'W:H' string for the image, snapped to a small
        denominator so it matches the generator's ratio distribution."""
        if width <= 0 or height <= 0:
            return "1:1"
        g = gcd(width, height)
        rw, rh = width // g, height // g
        # Already clean enough.
        if rw <= MAX_AR_DENOMINATOR and rh <= MAX_AR_DENOMINATOR:
            return f"{rw}:{rh}"
        # Otherwise find the closest p:q (q <= MAX_AR_DENOMINATOR) to the true ratio.
        target = width / height
        best = None
        for q in range(1, MAX_AR_DENOMINATOR + 1):
            p = max(1, round(target * q))
            err = abs(p / q - target)
            if best is None or err < best[0]:
                best = (err, p, q)
        return f"{best[1]}:{best[2]}"

    def build_prompt(self, aspect_ratio: str) -> str:
        # caption_prompt is the user-editable ADDITIONAL INSTRUCTIONS block,
        # injected into the fixed system prompt (not the whole prompt).
        user_instructions = (self.caption_config.caption_prompt or "").strip()
        if not user_instructions:
            user_instructions = "None."
        prompt = ideogram4_caption_prompt.replace("{{aspect_ratio}}", aspect_ratio)
        prompt = prompt.replace("{{user_instructions}}", user_instructions)
        return prompt

    def _extract_json(self, raw: str) -> Optional[dict]:
        """Pull the JSON object out of the model output, tolerating fences and
        stray preamble. Returns the parsed dict or None."""
        text = raw.strip()
        # Strip ```json ... ``` fences if present.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # Fall back to the outermost {...} span.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _convert_bbox(self, bbox):
        """Qwen3-VL emits NORMALIZED 0-1000 boxes in [x1,y1,x2,y2] order (verified
        empirically: coords are stable across input resolution). Our stored
        format is also 0-1000 but in [y1,x1,y2,x2] order, so this only reorders
        and clamps -- no pixel scaling. Returns the box or None to drop it."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        x1, x2 = sorted((max(0, min(1000, round(x1))), max(0, min(1000, round(x2)))))
        y1, y2 = sorted((max(0, min(1000, round(y1))), max(0, min(1000, round(y2)))))
        if y2 <= y1 or x2 <= x1:
            return None
        # stored order is [y1, x1, y2, x2]
        return [y1, x1, y2, x2]

    def _normalize_caption(self, data: dict, aspect_ratio: str) -> dict:
        """Validate/cleanup the parsed caption before storage, reordering each
        bbox from model-native [x1,y1,x2,y2] to stored [y1,x1,y2,x2]."""
        # Force the aspect ratio we computed; the model is told to echo it but
        # we know the true value.
        data["aspect_ratio"] = aspect_ratio
        decon = data.get("compositional_deconstruction", {})
        elements = decon.get("elements", [])
        if isinstance(elements, list):
            for el in elements:
                if isinstance(el, dict) and "bbox" in el:
                    cleaned = self._convert_bbox(el["bbox"])
                    if cleaned is None:
                        el.pop("bbox", None)
                    else:
                        el["bbox"] = cleaned
        return data

    def get_caption_for_file(self, file_path: str) -> Optional[str]:
        try:
            # Read true dimensions before any resize so the aspect ratio is exact.
            with Image.open(file_path) as probe:
                width, height = probe.size
            aspect_ratio = self.compute_aspect_ratio(width, height)

            img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
            prompt = self.build_prompt(aspect_ratio)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device_torch)

            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.caption_config.max_new_tokens
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            data = self._extract_json(output_text)
            if data is None:
                print(
                    f"[IdeogramCaptioner] Could not parse JSON for {file_path}; "
                    f"saving raw output."
                )
                return output_text

            data = self._normalize_caption(data, aspect_ratio)
            # Store pretty JSON for QC/editing; the dataloader minifies at load.
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
