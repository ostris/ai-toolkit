"""Shared helpers for Ideogram-4 structured JSON captions.

This is the single source of truth for the caption schema so the captioner, the
prompt upsampler, the dataloader, and the model encoder all agree. It encodes the
official Ideogram-4 rules and, crucially, MIGRATES the old caption format we used
before those rules were published into the new one ("digest" old, emit new).

Official schema (summary):
- three top-level keys: high_level_description (optional), style_description
  (optional), compositional_deconstruction (required).
- style_description holds EXACTLY ONE of `photo` (photographs) or `art_style`
  (illustration/painting/3D/graphic design), never both. Key order is strict and
  branch-dependent:
    photo branch:     aesthetics, lighting, photo, medium, color_palette
    non-photo branch: aesthetics, lighting, medium, art_style, color_palette
- medium is one of: photograph, illustration, 3d_render, painting, graphic_design
- color_palette: UPPERCASE #RRGGBB only, up to 16 per image / 5 per element.
- elements, strict key order:
    obj:  type, bbox, desc, color_palette
    text: type, bbox, text, desc, color_palette
  bbox is optional, normalized 0-1000, [y_min, x_min, y_max, x_max], top-left.
- serialize compact: separators=(",", ":"), ensure_ascii=False (no \\uXXXX).

The OLD format we previously emitted differed by: always using `photo` (even for
non-photo media), putting `color_palette` before `desc`/`text`, title-cased medium
with a trailing period ("Illustration."), and lowercase / 3-digit hex. Every
function here accepts the old shape and returns the new one.
"""

import json
import re
from collections import OrderedDict

MAX_IMAGE_PALETTE = 16  # style_description.color_palette
MAX_ELEMENT_PALETTE = 5  # per-element color_palette

# Canonical medium tokens (official set).
MEDIUM_OPTIONS = [
    "photograph",
    "illustration",
    "3d_render",
    "painting",
    "graphic_design",
]

# Map common variants (including our old "Title." style) to the canonical token.
# Anything not listed is treated as a custom medium and preserved verbatim.
_MEDIUM_ALIASES = {
    "photograph": "photograph",
    "photo": "photograph",
    "illustration": "illustration",
    "3d render": "3d_render",
    "3d_render": "3d_render",
    "3d-render": "3d_render",
    "3drender": "3d_render",
    "render": "3d_render",
    "3d": "3d_render",
    "painting": "painting",
    "graphic design": "graphic_design",
    "graphic_design": "graphic_design",
    "graphic-design": "graphic_design",
    "graphic": "graphic_design",
}

_HEX6_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_HEX3_RE = re.compile(r"^#[0-9a-fA-F]{3}$")


def canon_medium(medium):
    """Canonicalize a medium string to an official token when recognized,
    otherwise return it stripped (custom mediums are allowed, preserved as-is)."""
    if not isinstance(medium, str):
        return medium
    key = medium.strip().rstrip(".").strip().lower()
    if key in _MEDIUM_ALIASES:
        return _MEDIUM_ALIASES[key]
    return medium.strip()


def is_photo_medium(medium):
    """True for the photograph branch (uses `photo`), False for the art_style branch."""
    return canon_medium(medium) == "photograph"


def normalize_hex(color):
    """Return an UPPERCASE #RRGGBB string, expanding #RGB -> #RRGGBB. None if invalid."""
    if not isinstance(color, str):
        return None
    s = color.strip()
    if _HEX6_RE.match(s):
        return "#" + s[1:].upper()
    if _HEX3_RE.match(s):
        return "#" + "".join(ch * 2 for ch in s[1:]).upper()
    return None


def sanitize_palette(palette, max_len):
    """Keep unique, valid, UPPERCASE hex colors in order, capped to max_len.
    Returns the cleaned list, or None if nothing valid remains (drop the key)."""
    if not isinstance(palette, (list, tuple)):
        return None
    seen = set()
    out = []
    for c in palette:
        h = normalize_hex(c)
        if h is None or h in seen:
            continue
        seen.add(h)
        out.append(h)
        if len(out) >= max_len:
            break
    return out or None


def normalize_style(style):
    """Reorder/clean style_description into the correct branch (photo vs art_style)
    with the strict key order, canonical medium, and uppercase palette. Accepts the
    old shape (always `photo`) and migrates it based on the medium."""
    if not isinstance(style, dict):
        return style

    raw_medium = style.get("medium")
    medium = canon_medium(raw_medium) if raw_medium is not None else None
    has_photo = bool(style.get("photo"))
    has_art = bool(style.get("art_style"))

    # Decide the branch. A recognized medium is authoritative; otherwise infer from
    # whichever render key the (old) data already had, defaulting to photo.
    if medium in MEDIUM_OPTIONS:
        photo_branch = medium == "photograph"
    elif has_art and not has_photo:
        photo_branch = False
    else:
        photo_branch = True

    photo_val = style.get("photo") if has_photo else None
    art_val = style.get("art_style") if has_art else None

    out = OrderedDict()
    if "aesthetics" in style:
        out["aesthetics"] = style["aesthetics"]
    if "lighting" in style:
        out["lighting"] = style["lighting"]

    if photo_branch:
        # aesthetics, lighting, photo, medium, color_palette
        val = photo_val if photo_val is not None else art_val
        if val is not None:
            out["photo"] = val
        if medium is not None:
            out["medium"] = medium
    else:
        # aesthetics, lighting, medium, art_style, color_palette
        if medium is not None:
            out["medium"] = medium
        val = art_val if art_val is not None else photo_val
        if val is not None:
            out["art_style"] = val

    pal = sanitize_palette(style.get("color_palette"), MAX_IMAGE_PALETTE)
    if pal is not None:
        out["color_palette"] = pal

    # Preserve any unexpected extra keys at the end rather than dropping them.
    for k, v in style.items():
        if k not in (
            "aesthetics",
            "lighting",
            "photo",
            "art_style",
            "medium",
            "color_palette",
        ):
            out[k] = v
    return out


def normalize_element(el):
    """Reorder an element's keys to the strict schema order and uppercase its
    palette. obj: type, bbox, desc, color_palette. text: type, bbox, text, desc,
    color_palette. bbox is kept verbatim (already [y1,x1,y2,x2] in stored form)."""
    if not isinstance(el, dict):
        return el
    etype = el.get("type", "obj")
    out = OrderedDict()
    out["type"] = etype
    if el.get("bbox") is not None:
        out["bbox"] = el["bbox"]
    if etype == "text":
        if "text" in el:
            out["text"] = el["text"]
        if "desc" in el:
            out["desc"] = el["desc"]
    else:
        if "desc" in el:
            out["desc"] = el["desc"]
    pal = sanitize_palette(el.get("color_palette"), MAX_ELEMENT_PALETTE)
    if pal is not None:
        out["color_palette"] = pal
    # Preserve any extras (e.g. future keys) at the end.
    for k, v in el.items():
        if k not in out and k != "color_palette":
            out[k] = v
    return out


def normalize_caption_dict(data):
    """Normalize a parsed caption dict in place-ish: drop input-only aspect_ratio,
    enforce top-level key order, normalize style (photo/art_style branch) and every
    element. Returns a new OrderedDict. Accepts old-format captions and emits new."""
    if not isinstance(data, dict):
        return data
    data.pop("aspect_ratio", None)  # input-only context, never part of output

    out = OrderedDict()
    if "high_level_description" in data:
        out["high_level_description"] = data["high_level_description"]
    if "style_description" in data:
        out["style_description"] = normalize_style(data["style_description"])

    decon = data.get("compositional_deconstruction")
    if isinstance(decon, dict):
        nd = OrderedDict()
        if "background" in decon:
            nd["background"] = decon["background"]
        els = decon.get("elements")
        if isinstance(els, list):
            nd["elements"] = [normalize_element(e) for e in els]
        for k, v in decon.items():
            if k not in ("background", "elements"):
                nd[k] = v
        out["compositional_deconstruction"] = nd
    elif decon is not None:
        out["compositional_deconstruction"] = decon

    for k, v in data.items():
        if k not in (
            "high_level_description",
            "style_description",
            "compositional_deconstruction",
            "aspect_ratio",
        ):
            out[k] = v
    return out


# --- bbox coordinate adaptation that does NOT require valid JSON -------------
# Captioners emit boxes as [x1,y1,x2,y2] but we store [y1,x1,y2,x2]. The
# structured normalizer can only swap per-element when the JSON parses; if the
# model returns malformed JSON, that path is skipped and the boxes stay in the
# wrong order. This regex rewrites every `"bbox":[...]` array in the raw text
# directly, so the swap still happens on un-parseable output.
_BBOX_TEXT_RE = re.compile(
    r'"bbox"\s*:\s*\[\s*'
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\]"
)


def _clamp_1000(v):
    return max(0, min(1000, round(float(v))))


def swap_bbox_xy_in_text(text):
    """Swap every [x1,y1,x2,y2] bbox to the stored [y1,x1,y2,x2] order directly in
    the raw model output -- clamping each value to 0-1000 and ordering each axis
    pair. It never parses the surrounding JSON, so it works even when the output is
    malformed. Only `"bbox":[n,n,n,n]` arrays are touched; everything else is left
    byte-for-byte. Returns the rewritten text."""
    if not isinstance(text, str):
        return text

    def _repl(m):
        x1, y1, x2, y2 = m.groups()
        cx1, cx2 = sorted((_clamp_1000(x1), _clamp_1000(x2)))
        cy1, cy2 = sorted((_clamp_1000(y1), _clamp_1000(y2)))
        return f'"bbox":[{cy1},{cx1},{cy2},{cx2}]'

    return _BBOX_TEXT_RE.sub(_repl, text)


def is_ideogram_caption_str(text):
    """True if text parses as a JSON object with a compositional_deconstruction block."""
    t = (text or "").strip()
    if not t.startswith("{"):
        return False
    try:
        d = json.loads(t)
    except Exception:
        return False
    return isinstance(d, dict) and isinstance(
        d.get("compositional_deconstruction"), dict
    )


def to_model_string(data):
    """Serialize a caption dict to the compact, model-ready string the renderer wants."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def digest_caption_string(text):
    """Parse, normalize (migrating old format), and return the compact model-ready
    string. Returns the input unchanged if it is not an Ideogram structured caption
    (plain-text captions pass straight through)."""
    t = (text or "").strip()
    if not t.startswith("{"):
        return text
    try:
        data = json.loads(t, object_pairs_hook=OrderedDict)
    except Exception:
        return text
    if not (
        isinstance(data, dict)
        and isinstance(data.get("compositional_deconstruction"), dict)
    ):
        return text
    return to_model_string(normalize_caption_dict(data))
