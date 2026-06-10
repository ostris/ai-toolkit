---
name: style-vs-content-caption-auditor
description: >
  Audit existing AI Toolkit caption files for leakage — words that
  describe what should be the trigger, breaking the style/character/
  motion-vs-content discipline. Use when: (1) user asks to "audit my
  captions", "review captions for leakage", "check captions before
  training", (2) a previous LoRA run had bleed / weak trigger / style
  becoming promptable instead of automatic, and the user wants to know
  whether captions are at fault, (3) user is about to train and wants
  to validate caption discipline before committing to a multi-hour run,
  (4) user inherited a captioned dataset and isn't sure if it's
  training-ready. Reports per-file leakage, aggregate leak rate by
  category, and a spot-fix-vs-recaption recommendation. Routes to
  ai-toolkit-gemini-captioner if recaption is needed.
---

# Caption auditor — style vs. content discipline

A focused diagnostic skill. The user has a captioned dataset and wants
to know: are these captions training-ready? Or are they leaking the
trigger-bound concept (style / identity / motion) and going to produce
a weak LoRA?

This skill runs `references/audit_captions.py` — a generic mode-aware
script — interprets the output, and recommends an action.

## When to invoke

- **Pre-training audit:** user is about to launch training and wants
  to validate the captions first. 5 minutes here saves a 4-hour failed
  run.
- **Post-failure forensics:** previous LoRA had bleed / weak trigger /
  the style is promptable not automatic. Captions are the most common
  cause; this audit confirms.
- **Inherited dataset:** user got captions from a collaborator, an
  earlier run, or a different captioner, and wants to know if they
  meet this project's caption discipline.

If captions don't exist yet → use `ai-toolkit-gemini-captioner` to
generate them with the right discipline from the start.

## Workflow

### 1. Gather inputs

Ask the user (or infer from context):

| Input | Notes |
|-------|-------|
| **Dataset path** | Folder of `.txt` caption files. Same path as the YAML's `folder_path`. |
| **Mode** | style / character / motion / combined. Determines which avoid-list applies. |
| **Trigger word** | Leetspeak token. Audit verifies it appears at the right position in every caption. |
| **Anchored phrase** (style only) | Optional, e.g. `chemigram print`. Audit verifies anchor + trigger pair is at end. |
| **Known leakage** (optional) | Words from prior runs that bled. Add to the audit's avoid-list for this run. |

If mode is unclear, ask. The avoid-list differs significantly across
modes — running the wrong mode produces useless results.

### 2. Run the audit

Copy `references/audit_captions.py` to the user's `scripts/` folder
(or run it in place if the skill is already accessible). Then:

```bash
python scripts/audit_captions.py \
    --dataset /path/to/dataset \
    --mode style \
    --trigger <TOKEN> \
    [--anchor "anchor phrase"] \
    [--extra-avoid-words "leaked,word,list"]
```

The script outputs three sections:

1. **Structural checks** — empty captions, missing trigger, BOM
   characters, length outliers, files without `.txt`.
2. **Leakage scan** — per-category leak rate (e.g., "color words: 23%
   of captions", "motion verbs: 12%").
3. **Top N most problematic files** — sorted by issue count, with
   line content and offending words highlighted.

### 3. Interpret the results

Read `references/severity-and-action.md` for the full rubric. Quick
reference:

| Aggregate leak rate | Recommended action |
|---------------------|-------------------|
| <5% in any category | Production-ready. Train. Spot-fix the worst 1-3 files if any. |
| 5-15% in one category | Spot-fix recommended. Manually edit flagged files (audit prints them). |
| 5-15% across multiple categories | Recaption all. Manual spot-fix is too tedious; faster to regen. |
| >15% in any category | **Recaption all.** Captioner discipline failed; manual fix won't work. |
| >50% in any category | Recaption with the captioner skill. Don't even try to spot-fix. |

### 4. Take action

**If spot-fix:** edit the top N flagged files in place. Re-run audit
to confirm leak rate dropped.

**If recaption all:** route to **ai-toolkit-gemini-captioner**. Tell
the captioner skill which leaked words appeared in v1 — the captioner
should add them to its avoid-list explicitly.

**If structural issues only (no leaks):** spot-fix the structure
problems (missing triggers, empty files, BOM) — usually a quick
manual edit per file.

## Mode-specific guidance

### Style mode

**What to flag:**
- Medium / process words: photograph, painting, illustration, render, photogram, etc.
- Palette words: color names + warm/cool/muted/desaturated/iridescent family
- Surface / texture words specific to the style (chemistry, pooling, spray, etc.)
- Mood / vibe descriptors: ethereal, dreamy, beautiful, atmospheric
- Hedging: appears to be, seems like, looks like
- Quality descriptors: high quality, detailed, masterpiece, 4k

**What's OK:**
- Subject descriptions (count, shape, position, layout, framing)
- Things that genuinely vary across the dataset

### Character mode

**What to flag:**
- Identity words: face, jawline, cheekbones, eye color, hair color, hair type
- Body descriptors: skin tone, complexion, freckles, build, height
- Demographic descriptors: young, middle-aged, woman, man, girl, boy, ethnicity
- Hedging
- Quality descriptors

**What's OK:**
- Variable elements: clothing, accessories, pose, setting, lighting, style/medium
- The trigger word at the start of caption (character mode uses TRIGGER substitution)

### Motion mode

**What to flag:**
- Motion verbs in any tense: melt/melting/melted, transform/transforming/transformed, etc.
- Time-evolution: gradually, slowly, eventually, over time, throughout, during, after, before
- Anticipatory: about to, ready to, on the verge of, set to, poised to
- Video meta-language: video, clip, footage, scene, shot, frame, still, take
- Mood / vibe descriptors

**What's OK:**
- Static subject descriptions (it's a frozen-frame caption discipline)
- Subject count, arrangement, position, surface texture, color

### Combined mode (style + character)

Hybrid — varies by what's intentionally bound to trigger vs. what
should be a variable. Audit reports leakage in both axes; user
interprets based on their training intent.

## What the audit can NOT detect

The audit is mechanical — it greps for known-bad words. It can't
detect:

- **Semantic leakage** — captions describing the style indirectly
  ("photographic-looking", "painted-feel") even without the literal
  forbidden word.
- **Style-as-subject overlap** — for chemigram-like LoRAs where leaf
  veins are style-bound but leaves are content-bound, the audit can't
  know which "vein" mentions are leaks vs. legitimate.
- **Composition leakage** — if every caption describes "centered on
  white", that's leakage of composition into the trigger, but
  individual captions look fine.

For these, manual spot-checking of 5-10 random captions complements
the audit. The skill's output prints sample captions — eyeball them.

## When NOT to use this skill

- Captions don't exist yet → use `ai-toolkit-gemini-captioner` to
  generate from scratch.
- User wants to recaption regardless → skip audit, go straight to
  captioner.
- Generic Python text-grepping → just use grep directly.

## Reference files

- `references/audit_captions.py` — the audit script. Mode-aware,
  structural + vocabulary checks, severity-ranked output. Copy this
  into the user's `scripts/` folder for repeated use.
- `references/leak-vocabulary.md` — per-mode avoid-list with synonym
  families. Source of truth for what each mode considers "leaked."
- `references/severity-and-action.md` — rubric for interpreting audit
  output and choosing spot-fix vs. recaption.
