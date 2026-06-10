---
name: ai-toolkit-gemini-captioner
description: >
  Generate a dataset-specific Gemini captioning script for AI Toolkit LoRA
  training. Use when: (1) user wants a new captioner for a style, character,
  or motion LoRA dataset, (2) user says "write a captioner for X", "Gemini
  caption script", "caption these images/videos", "captioner for [dataset]",
  (3) user is starting a new LoRA dataset and needs captions written, (4) user
  asks for a first-frame video captioner for motion LoRAs. Produces a
  standalone Python script in the user's `scripts/` folder that mirrors the
  conventions of the existing `caption_*_dataset_gemini.py` scripts in the
  ai-toolkit repo. Encodes the style-vs-content discipline that makes a LoRA
  trigger fire correctly: traits described in captions become user-promptable
  variables; traits omitted bind to the trigger word.
---

# AI Toolkit Gemini Captioner Generator

Generate a `scripts/caption_<name>_<mode>_gemini.py` script tailored to one
LoRA dataset. The user's repo already contains 13+ captioners following the
same canonical scaffold — this skill produces another one.

The captioner is the leverage point of LoRA training. **What you describe in
captions becomes promptable at inference. What you omit binds to the trigger
word.** Get this wrong and the LoRA either fails to fire or refuses to
generalize. Most of this skill is about getting the omission list right.

## Workflow

### 1. Gather inputs

Ask the user (or infer from context):

| Input | Notes |
|-------|-------|
| **Mode** | style / character / motion-first-frame / filename-template |
| **Dataset path** | Absolute path. Often `/Volumes/...` external drive |
| **Trigger word** | Leetspeak token, e.g. `1ll6m3ns`, `vcr1bb`, `p3r5on`, `gr4r0cks`. Must match `trigger_word` in the YAML config |
| **Anchored suffix** (style only) | Optional real-word + trigger pair, e.g. `chemigram print, 1ll6m3ns`. Real word gives semantic prior; leetspeak is the lock-in |
| **Style/character/motion description** | What the dataset shows. Used to derive the avoid list |
| **Known leakage words** | Optional. Words from prior runs that leaked into Gemini output and hurt the LoRA |

If the user only says "write a captioner for these images," ask the mode and
trigger before writing anything. The avoid list cannot be inferred without
knowing what the LoRA is trying to learn.

### 2. Pick the mode

| Mode | Use when | Reference |
|------|----------|-----------|
| **style** | Every image shares a distinctive aesthetic (chemigram, VCRIBB, dgz_80s, attachestvo). Captions describe content, omit style. Trigger is a comma-appended suffix. | `references/style-template.py` |
| **character** | Every image is the same person/creature in different clothes/settings/styles. Captions describe variables, omit identity. Trigger is the literal first word of the caption. | `references/character-template.py` |
| **motion-first-frame** | Dataset is a folder of video clips for a motion LoRA. Caption only the first frame (extracted via cv2) as a still image. Avoid list is motion verbs + time-evolution words + anticipatory language. | `references/motion-first-frame-template.py` |
| **filename-template** | The user already knows what each clip depicts and wants deterministic captions from a `CAPTION_MAP: dict[str, str]` keyed by filename descriptor. No Gemini call. | `references/filename-template.py` |

If the user is mid-conversation about a "combined" LoRA (style + character),
default to **style** mode and have them describe the character traits in the
ALWAYS-describe list.

### 3. Build the avoid list

The avoid list is the hardest part. Read `references/avoid-words-cookbook.md`
for the full discipline. Quick rules:

- **Style mode:** include every word that describes the medium, palette,
  process, edge quality, mood, vibe, and any visual trait shared across the
  whole dataset. Include synonym families (`iridescent / pearlescent /
  opalescent / holographic`). Be aggressive — leakage compounds.
- **Character mode:** the avoid list is short and structural — face, eye
  color, hair color, skin, build, age, gender, ethnicity. The trigger word
  carries identity.
- **Motion mode:** every motion verb family the dataset exhibits (melt /
  slump / drip / flow / merge / morph), all time-evolution words (gradually,
  eventually, over time, throughout), and ALL anticipatory phrases (about to,
  ready to, on the verge of). Plus video meta-language (clip, footage, frame).

If the user mentions "the last LoRA leaked X" — add X and its synonyms
explicitly to the avoid list. Document the reason in a comment (`# leaked in
v1, baked color into trigger`).

### 4. Decide on suffix vs. TRIGGER token

| Pattern | Use when |
|---------|----------|
| `--suffix "trigger"` appended after a comma | Style mode. Captions end with `..., trigger`. The script enforces this if Gemini drifts. |
| `--suffix "anchor word, trigger"` | Style mode where you want a real-word semantic prior. E.g. `chemigram print, 1ll6m3ns` — `chemigram print` gives the LoRA something to attach to; `1ll6m3ns` is the unique lock-in. Inference must include both. |
| `TRIGGER` token replacement | Character mode. Caption starts with the literal trigger word (e.g. `p3r5on wearing a sweater...`). Gemini writes `TRIGGER`; the script substitutes. |
| `--trigger "trigger"` appended | Motion mode. Same as style suffix, but called `--trigger` since there's no anchored-word concept. |

### 5. Render the script

Copy the appropriate template from `references/` into a new file at
`scripts/caption_<name>_<mode>_gemini.py`. Then customize:

1. **Module docstring** — explain the dataset, what's being captured as
   variable vs. trigger-bound, and the rationale for the avoid list. Future-
   you reads this when training a successor LoRA.
2. **`*_AVOID_WORDS` tuple** — paste the user's curated list, grouped by
   category with `# comment` headers (e.g. `# darkroom process vocabulary`,
   `# color / tone words`). Aggressive groupings make it easy to amend.
3. **`SYSTEM_PROMPT_TEMPLATE`** — fill the `ALWAYS describe` list with the
   variables the user wants promptable, and the `NEVER describe` list with
   what should bind to the trigger. Keep the GOOD/BAD examples — they
   anchor Gemini's output more than the rules do.
4. **Defaults** — set `--dataset`, `--suffix`/`--trigger`, and `--model` to
   the values for this run so the user can invoke with no flags.

### 6. Canonical invariants (do NOT change without reason)

These are repeated across all 13 existing captioners and shouldn't drift:

| Invariant | Value |
|-----------|-------|
| Default model | `gemini-3.1-pro-preview` |
| Fallback chain (mention in `--model` help) | `gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash` |
| Temperature | `0.4` |
| Max retries | `4` with exponential backoff starting at `2.0s` |
| Default workers | `4` |
| Image extensions | `{".jpg", ".jpeg", ".png", ".webp"}` |
| Video extensions | `{".mp4", ".mov", ".mkv", ".webm", ".m4v"}` |
| Dotfile filtering | **Mandatory.** Skip `name.startswith(".")` to drop macOS `._*` AppleDouble resource forks and `.DS_Store`. External Mac drives WILL pollute datasets if you don't. |
| Skip existing `.txt` | Default behavior. `--overwrite` to re-caption. |
| Output | `<image>.txt` (or `<video>.txt`) sidecar with caption + trailing newline |
| API key env vars | `GEMINI_API_KEY` or `GOOGLE_API_KEY` (check both) |
| Required deps | `pip install google-genai pillow tqdm` (motion mode adds `opencv-python`) |

### 7. Output to the user

After writing the script, output:

1. The exact path: `scripts/caption_<name>_<mode>_gemini.py`
2. The invocation command with the user's actual args:
   ```bash
   export GEMINI_API_KEY="..."
   python scripts/caption_<name>_<mode>_gemini.py \
       --dataset "<path>" \
       --suffix "<trigger>"
   ```
3. A reminder that `--suffix` / `--trigger` MUST match the
   `trigger_word` in their YAML config.
4. If the user mentioned a prior leakage, point out where in the script
   you addressed it.

## Common mistakes to avoid

- **Don't be vague in the avoid list.** "Avoid color words" is useless.
  List `purple, blue, green, teal, magenta, lavender, violet, indigo` etc.
- **Don't put style traits in the ALWAYS describe list** — that's the bug
  the whole captioner exists to prevent.
- **Don't strip the GOOD/BAD examples block** — Gemini follows examples
  more reliably than rules. Keep at least 5 GOOD and 4 BAD.
- **Don't skip the dotfile filter.** Mac users will hit `._*` AppleDouble
  files showing up as `.jpg` and corrupting Gemini calls.
- **Don't use `gemini-3.0-pro` or `gemini-flash`** — the default is
  `gemini-3.1-pro-preview`. The user has bulk-edited all 13 existing
  scripts to that model. Stay in sync.
- **Don't add `**kwargs` or other flexibility** the user didn't ask for.
  These scripts are deliberately simple, single-purpose, copy-paste-edit.

## When NOT to use this skill

- The user already has captions and just wants to audit them — that's a
  different skill (style-vs-content auditor, not yet built).
- The user wants captions for inference prompting, not training — use the
  model-specific prompter skills (flux2-klein-prompter, etc.).
- The dataset is already captioned and the user wants to retrain — only
  generate a new captioner if they explicitly want to recaption.

## Reference files

- `references/style-template.py` — style LoRA captioner (image input)
- `references/character-template.py` — character LoRA captioner (TRIGGER
  token replacement)
- `references/motion-first-frame-template.py` — motion LoRA captioner
  (cv2 first-frame extraction from videos)
- `references/filename-template.py` — deterministic captioner driven by a
  `CAPTION_MAP` dict, no Gemini call
- `references/avoid-words-cookbook.md` — how to build a complete avoid
  list for a new style; common categories and the synonym families that
  matter
