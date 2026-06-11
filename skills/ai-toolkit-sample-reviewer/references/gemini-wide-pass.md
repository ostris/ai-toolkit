# Gemini Wide Pass (`scripts/review_samples_gemini.py`)

The token-cheap half of the reviewer. Gemini looks at **every** sample (every
checkpoint × every prompt) and emits one structured record per image into
`sample_review.json`. You then read that JSON (text, cheap) to locate the
candidate region, and spend Opus vision only on directly eyeballing the 2-3
finalists.

This exists because the reviewer's cost is almost entirely Opus vision tokens
from reading sample JPEGs — a run is easily a few hundred images. Moving the
wide, shallow pass to Gemini cuts Opus vision ~90% **and** gives full
per-checkpoint coverage (no montages, no skipped steps) **and** avoids fanning
out image-analysis subagents (which crashes the laptop).

## Division of labour — do not blur this

| | Gemini wide pass | Opus |
|---|---|---|
| Scope | every image, shallow | 2-3 finalists, deep |
| Job | extract concrete facts | synthesize trajectory + **final aesthetic verdict** |
| Output | `sample_review.json` | the recommendation |

Gemini is a coarse pass. Its `"strong"`/`"adequate"` scores have the same blind
spot a vision subagent's "densest composite" did — they can rate a
mode-collapsed checkpoint highly. **Never pick a winner from the JSON alone.**

## Prerequisites

1. **Ground-truth spec** — a plain-text file you write in Step 3 after looking
   at 5-8 dataset images: medium, palette, texture/mark-making, what varies,
   and any artist-intent notes from the YAML comments. The script refuses to
   run without it (`--ground-truth`). A fidelity judgment with no ground truth
   is a guess — for Gemini as much as for you.
2. **Goal type** — `style`, `character`, or `combined`. Selects how `fidelity`
   is framed. Same classification you made in Step 1.
3. **`GEMINI_API_KEY`** (or `GOOGLE_API_KEY`) in the environment, and the
   `google-genai` package — present in the `.venv-captioning` venv used by the
   captioner scripts.

## Invocation

```bash
source .venv-captioning/bin/activate
export GEMINI_API_KEY="..."
python scripts/review_samples_gemini.py \
    --config output/<run>/<run>.yaml \
    --ground-truth output/<run>/ground_truth.txt \
    --goal style \
    --mode quality        # or --mode fast
```

The script reads the config to find the run folder, samples dir, trigger word,
the prompt list, and which prompt indices are controls (no trigger). It parses
sample filenames into `(step, prompt_index)`, substitutes `[trigger]` to get the
rendered prompt each image actually saw, and labels each call triggered vs.
control. It is **re-run safe** — already-recorded images are skipped; pass
`--overwrite` to redo. Output defaults to `output/<run>/sample_review.json`.

Useful flags:

- `--mode` — speed-vs-quality preset for the underlying Gemini model. Ask the
  user before picking; don't default silently.
  - `quality` (default) → `gemini-3.1-pro-preview`. ~5 min/checkpoint on a
    14-ckpt × 16-prompt grid (≈50–70 min total). Use for abstract or
    multi-material styles where subtle material differences between samples
    matter (e.g. v4-style "does the trigger fire variety vs. mode-collapse to
    chrome").
  - `fast` → `gemini-3.1-flash-lite`. ~10× faster (≈5–10 min total). Use for
    clean single-mode styles, character LoRAs (identity is binary-ish), or
    iteration runs where you already know what to look for.
  - Either mode produces the same JSON schema; only the per-image reads get
    richer at `quality`. The aesthetic verdict still happens on Opus in
    Pass 2, so `fast` doesn't gate final quality — just early signal on hard
    styles. (Per the [Gemini captioner fallback model] memory: flash-lite
    NOT flash-preview — the latter 404s on v1beta.)
- `--model` — explicit override that bypasses `--mode`. Valid:
  `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-2.5-pro`,
  `gemini-2.5-flash`. Use only when you need a non-preset model (e.g. quota
  fallback). For the normal speed-vs-quality choice, prefer `--mode`.
- `--max-side` — downscales the longest image side before sending (default
  1024; most samples are already ≤1024 so it's usually a no-op). Lower it to
  cut Gemini cost further on huge sample sets.
- `--samples-dir`, `--output` — overrides if the layout is non-standard.

## `sample_review.json` structure

```jsonc
{
  "meta": {
    "run_folder": "output/my_run",
    "model": "gemini-3.1-pro-preview",
    "goal": "style",
    "trigger_word": "1ll6m3ns",
    "control_prompt_indices": [3, 7],          // prompts with no trigger
    "prompts": { "0": "<rendered prompt 0>", "1": "..." }
  },
  "checkpoints": {
    "000001500": {                              // step (zero-padded, as in filenames)
      "0": { /* per-image record, see below */ },
      "1": { ... }
    },
    "000002000": { ... }
  }
}
```

### Per-image record fields

| Field | Type | Meaning / how to use it |
|---|---|---|
| `fidelity` | strong / adequate / weak / broken / n/a | **Triggered only.** How strongly the trained style/identity appears vs. the ground truth. `n/a` on controls. → the **floor** (first `strong`/`adequate`) and **peak**. |
| `subject_match` | strong / adequate / weak / broken | Did it render the subject the prompt asked for? Weak/broken on a generalization prompt = generalization failure; weak on many = memorization substitution. |
| `texture_fidelity` | strong / adequate / weak / broken / n/a | Did the dataset's texture/grain/brushwork/material appear? Usually weak early, strong late — a key artist-intent signal. |
| `palette_match` | strong / adequate / weak / n/a | Colour palette vs. ground truth. |
| `control_clean` | clean / slight_bleed / strong_bleed / n/a | **Controls only.** `clean` = looks like base model; anything else = **bleed**. The ceiling. Watch where this first leaves `clean`. |
| `bleed_signs` | string[] | Concrete absorbed traits ("warm dataset palette", "halftone grain", "recurring bust"). Read these to *characterize* the bleed, not just flag it. |
| `gibberish_text` | bool | Hallucinated/garbled letterforms — flag in Concerns even if the image looks good otherwise. |
| `dataset_subject_leak` | bool | A specific recurring dataset object appears unprompted → content learned as style (overfitting). |
| `composition` | string | Short factual layout descriptor. **Compare across different prompts at the same step**: identical layout for different prompts = compositional memorization. |
| `artifacts` | string | Concrete defects ("melted hands", "duplicated limbs"), or "". |
| `notes` | string | One short line of extra concrete observation. |

### Reading it into a trajectory

1. Sort steps ascending.
2. **Floor** = first step where triggered `fidelity` reaches `adequate`/`strong`.
3. **Peak** = step(s) where `fidelity` + `texture_fidelity` + `palette_match`
   are jointly best across the triggered prompts.
4. **Bleed onset** = first step where any control's `control_clean` leaves
   `clean`. (If `train.diff_output_preservation: true`, this should stay
   `clean` throughout; if it doesn't, DOP isn't working — note it.)
5. **Overfitting onset** = first step with rising `dataset_subject_leak`,
   repeated `composition` across distinct prompts, or `subject_match` decay on
   generalization prompts.
6. The candidate band is roughly [just before peak → just before bleed/overfit
   onset]. Hand that band to Pass 2 and **look at it directly.**

If the JSON disagrees with your eyes on a finalist, your eyes win — the JSON is
the map, not the territory.
