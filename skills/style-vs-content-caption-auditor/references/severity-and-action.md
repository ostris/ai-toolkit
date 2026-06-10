# Severity rubric and action plan

How to interpret audit output and decide what to do.

## The three actions

Every audit ends in one of three actions:

1. **Train as-is.** Captions are clean enough.
2. **Spot-fix.** A handful of files have issues; manual edits are
   tractable.
3. **Recaption all.** Discipline failed at the captioner level; manual
   fixes won't scale.

The audit script's exit code corresponds:
- `0` = train as-is
- `1` = spot-fix
- `2` = recaption

## The thresholds

| Aggregate leak rate | Action |
|---------------------|--------|
| **Any category < 5%** AND no widespread structural issues | Train as-is |
| **One category 5–15%** | Spot-fix that category's flagged files |
| **Multiple categories 5–15%** OR **any one ≥ 15%** | Recaption all |
| **Any one ≥ 50%** | Recaption all (captioner discipline catastrophically failed) |
| **>5% structural issues** (missing trigger, empty files) but no leakage | Spot-fix structurally |

These thresholds are empirical — derived from observing which leak rates
actually produced visible bleed in trained LoRAs. They're not magic:

- **<5%** is below the noise floor of training. The LoRA sees the
  leaked word in 1-2 captions out of 30+, the gradient is small,
  trigger still wins.
- **5–15%** is where spot-fix is faster than recaption. 3-5 files of
  manual editing beats waiting 20 minutes for Gemini to recaption 50
  files.
- **>15%** means systematic — the captioner consistently used the
  forbidden word. Manual fix would require editing 8+ files; faster
  to fix the captioner's avoid-list and regen.
- **≥50%** means the captioner's avoid-list was missing this entire
  category. The output is unsalvageable.

## When to break the threshold rule

### Train as-is even at >5% leakage

- The leaked words are **acceptable to be promptable**. Example: a
  style LoRA where you DO want "iridescent" to remain user-promptable
  (so users can ask for "iridescent dog, [trigger]"). The audit
  doesn't know your training intent — interpret category-by-category.
- You're **iterating fast** and accept a weaker v1 to test the
  pipeline. Recaption later when v1 confirms the basic approach
  works.

### Recaption even at <5% leakage

- The leakage is in a **catastrophic category**. For character LoRAs,
  even 3% of captions saying "blonde" can lock the trigger to a
  hair-color-specific identity. Some categories are more sensitive
  than others.
- A previous LoRA already failed and you suspect captions; recaption
  even cheap to validate.

## Spot-fix mechanics

The audit prints the top N most-problematic files with offending
words highlighted. To spot-fix:

```bash
# Audit
python scripts/audit_captions.py --dataset /path --mode style \
    --trigger 1ll6m3ns --top 20

# Edit the flagged files manually — open each in your editor,
# remove the leaked words, save
$EDITOR /path/to/dataset/<flagged_file_1>.txt
$EDITOR /path/to/dataset/<flagged_file_2>.txt
# ...

# Re-audit to confirm leak rate dropped
python scripts/audit_captions.py --dataset /path --mode style \
    --trigger 1ll6m3ns
```

If you're editing more than ~5 files, recaption is probably faster.

## Recaption mechanics

Route to **`ai-toolkit-gemini-captioner`** with the right mode. Tell
the captioner explicitly:

> "Add `<leaked_word_1>, <leaked_word_2>, ...` to the AVOID_WORDS
> tuple. The previous run leaked these into N% of captions."

The captioner skill expects this signal — its template has comment
hooks for "leaked in v1" entries. Document the leakage so v3 inherits
the lesson.

After recaption, **delete the latent / text embedding cache** (per
`ai-toolkit-dataset-diagnostics` cache-and-state) before retraining.
Stale embeddings will serve old captions even after `.txt` files are
updated.

## Structural issues — separate from leakage

Structural issues (missing trigger, empty files, BOM characters, length
outliers) don't relate to leak vocabulary. They're orthogonal failures
and need their own fixes.

| Issue | Fix |
|-------|-----|
| Empty caption file | Delete or recaption that single file |
| Trigger not at end (style/motion/combined) | Append `, <trigger>` to caption |
| Trigger not at start (character) | Replace caption's first word with trigger |
| BOM character | `sed -i '' '1s/^\xEF\xBB\xBF//' file.txt` |
| Caption too short (<5 words) | Recaption that file with more detail |
| Caption too long (>100 words) | Trim — over-detailed captions distract the LoRA |

For structural issues, spot-fix is almost always the right call
unless the rate is >50% (then the captioner skipped half the files
or had a bug).

## When the audit confuses you

- **"My category leak rate is high but the LoRA seems fine"** — your
  intent for that category was different from the audit's assumption.
  Example: a Wan22 style+motion LoRA where you DO want the visual
  signature bound to the trigger. The audit's `motion` mode assumes
  static-frame discipline; that's wrong for combined.
- **"The audit says train but my LoRA bleeds at inference"** — leakage
  isn't the only cause. Check `caption_dropout_rate` (should be 0 for
  most production runs) and lr (small datasets need lower lr to
  prevent bleed). See `flux2-klein-lora-config` failure-mode
  diagnosis §2 for the anti-bleed package.
- **"The audit says recaption but my LoRA worked great last time"** —
  some thresholds are conservative. If your inference output is
  clean, train as-is. The audit is a forecast, not a verdict.

## Pre-training audit habit

Run the audit before EVERY training launch:

```bash
# Save as ~/preflight_captions.sh
#!/usr/bin/env bash
DATASET="${1:?usage: preflight_captions.sh <dataset> <mode> <trigger>}"
MODE="${2:?need mode: style|character|motion|combined}"
TRIGGER="${3:?need trigger word}"

python scripts/audit_captions.py \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --trigger "$TRIGGER"
```

Then before every `python run.py config/...`:

```bash
bash ~/preflight_captions.sh /path/to/dataset style 1ll6m3ns
```

5 minutes of audit prevents 4 hours of failed training.
