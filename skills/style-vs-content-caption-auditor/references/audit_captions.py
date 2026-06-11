"""Audit AI Toolkit caption files for leakage of trigger-bound concepts.

A caption file is "leaked" if it describes the trait the LoRA was supposed
to learn implicitly. Style words in style-LoRA captions, identity words in
character-LoRA captions, motion verbs in motion-LoRA captions — all of
these turn the concept into a *promptable variable* instead of binding it
to the trigger. Result: weak LoRA at inference.

This script scans caption files for known-bad words per LoRA mode, plus
structural issues (empty captions, missing trigger, BOM, length outliers).
Reports per-file findings, aggregate leak rates, and a recommendation.

No external dependencies — pure stdlib.

Usage:

    python audit_captions.py --dataset /path/to/captions --mode style \\
        --trigger 1ll6m3ns

    # with anchored trigger (real word + leetspeak)
    python audit_captions.py --dataset /path --mode style \\
        --trigger 1ll6m3ns --anchor "chemigram print"

    # add dataset-specific leaked words (from prior runs that bled)
    python audit_captions.py --dataset /path --mode style \\
        --trigger 1ll6m3ns --extra-avoid "vein,branching,translucent"

    # show top N problematic files (default 10)
    python audit_captions.py --dataset /path --mode style \\
        --trigger 1ll6m3ns --top 20

    # structural-only check (skip leak vocabulary scan — fast pre-flight)
    python audit_captions.py --dataset /path --mode style \\
        --trigger 1ll6m3ns --structural-only

Modes:
    style     — flag medium/palette/mood/vibe/quality/hedging
    character — flag identity (face/eyes/hair/skin/build/age/gender/ethnicity)
    motion    — flag motion verbs / time-evolution / anticipatory / video meta
    combined  — both style and character flags (interpret per training intent)

Exit codes:
    0   — no major issues (under threshold for spot-fix)
    1   — spot-fix recommended (low/moderate leakage)
    2   — recaption recommended (high leakage)
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ── Avoid-vocabulary per mode ────────────────────────────────────────────

STYLE_AVOID: Dict[str, List[str]] = {
    "medium": [
        "photograph", "photo", "photographic", "image", "picture", "scan",
        "scanned", "painting", "illustration", "render", "rendering",
        "digital", "artwork", "art piece", "sculpture", "drawing",
    ],
    "palette": [
        "purple", "blue", "green", "teal", "magenta", "lavender", "violet",
        "indigo", "brown", "tan", "amber", "ochre", "red", "yellow", "orange",
        "pink", "warm", "cool", "muted", "desaturated", "monochrome",
        "duotone", "sepia", "iridescent", "pearlescent", "opalescent",
        "holographic", "metallic", "shimmery", "shiny", "glossy",
        "prismatic", "rainbow", "saturated",
    ],
    "vibe": [
        "vintage", "antique", "aged", "old", "weathered", "retro", "period",
        "historical", "dreamy", "surreal", "ethereal", "otherworldly",
        "atmospheric", "moody", "evocative", "haunting", "nostalgic",
        "melancholic", "timeless", "mysterious", "stylized", "aesthetic",
        "vibe", "mood", "cinematic", "whimsical",
    ],
    "quality": [
        "high quality", "detailed", "masterpiece", "4k", "8k", "hyperdetailed",
        "intricate", "ultra realistic", "photorealistic", "professional",
        "beautiful", "striking", "delicate", "fragile", "elegant", "dramatic",
        "stunning", "gorgeous",
    ],
    "hedging": [
        "appears to be", "seems like", "looks like", "as if", "kind of",
        "sort of", "appears", "seems",
    ],
}

CHARACTER_AVOID: Dict[str, List[str]] = {
    "face": [
        "face", "facial", "jawline", "cheekbones", "cheeks", "chin", "nose",
        "lips", "mouth", "forehead", "eyebrows", "eyelashes", "expression line",
    ],
    "eyes": [
        "eye color", "blue eyes", "brown eyes", "green eyes", "hazel eyes",
        "gray eyes", "dark eyes", "almond shaped", "round eyes", "narrow eyes",
    ],
    "hair": [
        "blonde", "brunette", "redhead", "black hair", "brown hair",
        "blonde hair", "red hair", "gray hair", "white hair", "long hair",
        "short hair", "curly hair", "straight hair", "wavy hair",
    ],
    "skin": [
        "skin tone", "complexion", "fair skin", "pale skin", "dark skin",
        "tan skin", "olive skin", "freckles", "freckled", "moles", "scars",
    ],
    "body": [
        "tall", "short", "slim", "slender", "petite", "muscular", "athletic",
        "stocky", "thin", "thick", "heavyset", "build",
    ],
    "demographic": [
        "young", "youthful", "middle-aged", "elderly", "old", "teenager",
        "child", "adult", "woman", "man", "girl", "boy", "lady", "gentleman",
        "female", "male",
    ],
    "ethnicity": [
        "asian", "european", "african", "latino", "latina", "hispanic",
        "caucasian", "middle eastern", "south asian", "east asian",
    ],
    "hedging": [
        "appears to be", "seems like", "looks like", "as if",
    ],
    "quality": [
        "high quality", "detailed", "masterpiece", "4k", "8k",
    ],
}

MOTION_AVOID: Dict[str, List[str]] = {
    "motion_verbs": [
        # melt / drip family
        "melt", "melting", "melted", "slump", "slumping", "slumped",
        "droop", "drooping", "drooped", "sag", "sagging", "sagged",
        "deflate", "deflating", "deflated", "collapse", "collapsing",
        "collapsed", "drip", "dripping", "dripped", "flow", "flowing",
        "flowed", "ooze", "oozing", "oozed", "pour", "pouring", "fall",
        "falling",
        # spread / merge family
        "spread", "spreading", "fuse", "fusing", "fused", "merge", "merging",
        "merged", "blend", "blending", "blended", "combine", "combining",
        "combined", "join", "joining", "joined", "unite", "uniting",
        # transform / morph family
        "transform", "transforming", "transformed", "transformation",
        "morph", "morphing", "morphed", "morphological",
        "change", "changing", "changed", "evolve", "evolving", "evolved",
        "shift", "shifting", "shifted", "transition", "transitioning",
        "become", "becoming", "became", "turn into", "turning into",
        "develop", "developing", "developed",
        # general motion
        "motion", "moving", "moves", "moved", "movement", "animate",
        "animated", "animation",
    ],
    "time_evolution": [
        "time-lapse", "timelapse", "time lapse", "gradual", "gradually",
        "slowly", "slow", "eventually", "after", "before", "during",
        "over time", "throughout", "begin", "beginning", "begins", "started",
        "starts", "end", "ending", "ends", "finished", "progress",
        "progression", "progressing", "sequence",
    ],
    "anticipatory": [
        "about to", "ready to", "on the verge of", "set to", "poised to",
        "preparing to",
    ],
    "video_meta": [
        "video", "clip", "footage", "scene", "shot", "take", "frame", "still",
    ],
    "vibe": [
        "atmospheric", "ethereal", "dreamy", "moody", "abstract artwork",
        "beautiful", "striking", "interesting",
    ],
    "hedging": [
        "appears to be", "seems like", "looks like", "as if",
    ],
}


def get_avoid_for_mode(mode: str) -> Dict[str, List[str]]:
    if mode == "style":
        return STYLE_AVOID
    if mode == "character":
        return CHARACTER_AVOID
    if mode == "motion":
        return MOTION_AVOID
    if mode == "combined":
        # combined = style + character — user interprets per intent
        merged = dict(STYLE_AVOID)
        for k, v in CHARACTER_AVOID.items():
            merged[f"character_{k}"] = v
        return merged
    raise ValueError(f"unknown mode: {mode}")


# ── Caption-file scanning ────────────────────────────────────────────────

def find_caption_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*.txt")
        if p.is_file() and not p.name.startswith(".")
    )


def has_bom(text: str) -> bool:
    return text.startswith("﻿") or text.startswith("￾")


def scan_caption(text: str, avoid: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Return per-category list of words found in caption."""
    found: Dict[str, List[str]] = {}
    text_lower = text.lower()
    for category, words in avoid.items():
        hits = []
        for word in words:
            # whole-word match, case-insensitive
            pattern = r"\b" + re.escape(word.lower()) + r"\b"
            if re.search(pattern, text_lower):
                hits.append(word)
        if hits:
            found[category] = hits
    return found


def check_structure(text: str, mode: str, trigger: str, anchor: str = None) -> List[str]:
    """Return list of structural issues in this caption."""
    issues = []

    if not text.strip():
        issues.append("EMPTY")
        return issues

    if has_bom(text):
        issues.append("BOM character at start")

    text_lower = text.strip().lower()
    trigger_lower = trigger.lower()
    anchor_lower = anchor.lower() if anchor else None

    if mode == "character":
        # character mode: trigger should be the FIRST word
        first_token = text.strip().split()[0] if text.strip() else ""
        if first_token.lower().rstrip(",") != trigger_lower:
            issues.append(f"trigger '{trigger}' not at start (got '{first_token}')")
    else:
        # style / motion / combined: trigger appears at end
        # tolerate trailing punctuation
        text_clean = re.sub(r"[.,;:\s]+$", "", text_lower)
        if not text_clean.endswith(trigger_lower):
            issues.append(f"trigger '{trigger}' not at end")
        if anchor_lower:
            # check anchor + trigger pair is at end
            expected = anchor_lower + ", " + trigger_lower
            if not text_clean.endswith(expected):
                issues.append(f"anchor+trigger pair '{anchor}, {trigger}' not at end")

    word_count = len(text.split())
    if word_count < 5:
        issues.append(f"too short ({word_count} words)")
    elif word_count > 100:
        issues.append(f"too long ({word_count} words)")

    return issues


# ── Reporting ────────────────────────────────────────────────────────────

def aggregate_leak_rate(per_file: List[Tuple[Path, Dict[str, List[str]]]]) -> Dict[str, float]:
    """Compute % of files that have at least one hit in each category."""
    total = len(per_file)
    if total == 0:
        return {}
    counts: Dict[str, int] = defaultdict(int)
    for _, cat_hits in per_file:
        for cat in cat_hits:
            counts[cat] += 1
    return {cat: 100.0 * count / total for cat, count in counts.items()}


def severity_score(structural: List[str], leak_cats: Dict[str, List[str]]) -> int:
    """Higher = more problematic. For sorting top-N."""
    score = 0
    score += 5 * len(structural)
    score += sum(len(words) for words in leak_cats.values())
    return score


def recommendation(
    leak_rate: Dict[str, float],
    structural_count: int,
    total_files: int,
) -> Tuple[int, str]:
    """Return (exit_code, recommendation_text)."""
    if total_files == 0:
        return (2, "No caption files found — verify --dataset path.")

    structural_pct = 100.0 * structural_count / total_files
    max_leak = max(leak_rate.values()) if leak_rate else 0.0
    high_leak_cats = [c for c, r in leak_rate.items() if r >= 5.0]

    if max_leak >= 50.0:
        return (
            2,
            f"RECAPTION ALL. {max_leak:.0f}% leak rate in at least one category — captioner discipline failed; manual fix won't scale.",
        )
    if max_leak >= 15.0:
        return (
            2,
            f"RECAPTION ALL. {max_leak:.0f}% leak rate in at least one category. Route to ai-toolkit-gemini-captioner with the leaked words added to its avoid-list.",
        )
    if len(high_leak_cats) >= 2:
        return (
            2,
            f"RECAPTION ALL. Multiple categories with >5% leakage ({', '.join(high_leak_cats)}) — manual spot-fix is too tedious.",
        )
    if max_leak >= 5.0:
        return (
            1,
            f"SPOT-FIX. {max_leak:.0f}% leakage in '{max(leak_rate, key=leak_rate.get)}'. Manually edit the flagged files (see top-N list).",
        )
    if structural_pct >= 5.0:
        return (
            1,
            f"SPOT-FIX STRUCTURAL. {structural_count}/{total_files} files have structural issues. Fix manually — usually quick (missing triggers, empty files, etc.).",
        )
    return (0, "Production-ready. Train.")


def format_report(
    total_files: int,
    structural_issues: List[Tuple[Path, List[str]]],
    leak_per_file: List[Tuple[Path, Dict[str, List[str]]]],
    leak_rate: Dict[str, float],
    top_n: int,
    rec_code: int,
    rec_text: str,
) -> str:
    out = []
    out.append("=" * 72)
    out.append(f"Caption audit — {total_files} files scanned")
    out.append("=" * 72)
    out.append("")

    # Section 1: Structural
    out.append("── Structural checks ──")
    if not structural_issues:
        out.append("  ✓ All captions have no structural issues")
    else:
        out.append(f"  ⚠ {len(structural_issues)} files have structural issues:")
        for path, issues in structural_issues[:15]:
            out.append(f"    {path.name}: {'; '.join(issues)}")
        if len(structural_issues) > 15:
            out.append(f"    ... and {len(structural_issues) - 15} more")
    out.append("")

    # Section 2: Leakage
    out.append("── Leakage scan (% of files with at least one hit per category) ──")
    if not leak_rate:
        out.append("  ✓ No leakage detected in any category")
    else:
        for cat, rate in sorted(leak_rate.items(), key=lambda kv: -kv[1]):
            bar = "█" * int(rate / 5)  # 1 char per 5%
            severity = "❌" if rate >= 15 else ("⚠" if rate >= 5 else "·")
            out.append(f"  {severity} {cat:<24} {rate:5.1f}%  {bar}")
    out.append("")

    # Section 3: Top problematic files
    if leak_per_file:
        ranked = sorted(
            leak_per_file,
            key=lambda pf: -severity_score(
                next((s for p, s in structural_issues if p == pf[0]), []),
                pf[1],
            ),
        )
        problematic = [pf for pf in ranked if pf[1]][:top_n]
        if problematic:
            out.append(f"── Top {len(problematic)} most problematic files ──")
            for path, cat_hits in problematic:
                out.append(f"  {path.name}")
                for cat, words in sorted(cat_hits.items()):
                    out.append(f"    {cat}: {', '.join(words[:8])}")
                # show the actual caption (first 200 chars)
                try:
                    text = path.read_text(encoding="utf-8", errors="replace").strip()
                    snippet = text[:200] + ("..." if len(text) > 200 else "")
                    out.append(f"    > {snippet}")
                except Exception:
                    pass
                out.append("")
            out.append("")

    # Section 4: Recommendation
    out.append("── Recommendation ──")
    icon = {0: "✓", 1: "⚠", 2: "❌"}[rec_code]
    out.append(f"  {icon} {rec_text}")
    out.append("")
    out.append("=" * 72)
    return "\n".join(out)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audit AI Toolkit caption files for trigger-bound leakage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="Folder of caption .txt files (recursive)")
    parser.add_argument("--mode", required=True,
                        choices=["style", "character", "motion", "combined"],
                        help="LoRA mode — determines which avoid-list applies")
    parser.add_argument("--trigger", required=True,
                        help="Trigger word (leetspeak token)")
    parser.add_argument("--anchor", default=None,
                        help="Anchored phrase (style only), e.g. 'chemigram print'")
    parser.add_argument("--extra-avoid", default="",
                        help="Comma-separated extra words to flag (dataset-specific leaks from prior runs)")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N most problematic files (default: 10)")
    parser.add_argument("--structural-only", action="store_true",
                        help="Skip leak vocabulary scan — fast structural pre-flight")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print(f"Dataset path does not exist: {root}", file=sys.stderr)
        sys.exit(2)
    if not root.is_dir():
        print(f"Dataset path is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    files = find_caption_files(root)
    if not files:
        print(f"No .txt caption files found in {root}", file=sys.stderr)
        sys.exit(2)

    avoid = get_avoid_for_mode(args.mode)
    extra_words = [w.strip() for w in args.extra_avoid.split(",") if w.strip()]
    if extra_words:
        avoid = dict(avoid)
        avoid["dataset_specific_leaks"] = extra_words

    structural_issues: List[Tuple[Path, List[str]]] = []
    leak_per_file: List[Tuple[Path, Dict[str, List[str]]]] = []

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            structural_issues.append((f, [f"read error: {e}"]))
            leak_per_file.append((f, {}))
            continue

        issues = check_structure(text, args.mode, args.trigger, args.anchor)
        if issues:
            structural_issues.append((f, issues))

        if args.structural_only:
            leak_per_file.append((f, {}))
        else:
            cat_hits = scan_caption(text, avoid)
            leak_per_file.append((f, cat_hits))

    leak_rate = aggregate_leak_rate(leak_per_file) if not args.structural_only else {}
    rec_code, rec_text = recommendation(leak_rate, len(structural_issues), len(files))

    report = format_report(
        total_files=len(files),
        structural_issues=structural_issues,
        leak_per_file=leak_per_file,
        leak_rate=leak_rate,
        top_n=args.top,
        rec_code=rec_code,
        rec_text=rec_text,
    )
    print(report)
    sys.exit(rec_code)


if __name__ == "__main__":
    main()
