---
name: ai-toolkit-sample-reviewer
description: Review AI Toolkit LoRA training samples against the original dataset and config to find the best checkpoint. Use this skill whenever the user has finished (or is partway through) a LoRA training run and wants to pick which checkpoint to keep, asks "did this training work?", asks to "review my samples", "pick a checkpoint", "compare samples to the dataset", "which save should I use", "is this overfitting", or any time they reference an ai-toolkit output folder full of step-numbered sample images. Also use proactively when the user just finished a training run and is staring at the output. The skill compares samples to dataset ground truth across all saved checkpoints, evaluates style/identity fidelity, overfitting, generalization, and control-prompt bleed, and recommends a winner plus an optional merge.
---

# AI Toolkit Sample Reviewer

Pick the best checkpoint from an ai-toolkit LoRA training run by comparing its sample images against the original dataset and the training config's intent.

## When you've been triggered

The user has a finished or in-progress LoRA run. They want one of:

1. **A winning checkpoint** to use at inference.
2. **A diagnosis** if no checkpoint looks good (and what to change in v2).
3. **A merge recommendation** if two checkpoints have complementary strengths (early style + late detail, etc.).

You'll deliver all three.

## Inputs you need

Confirm with the user (most can be inferred):

1. **Config YAML path** — usually under `config/examples/...`. Tells you trigger word, prompts, model, dataset path.
2. **Output run folder** — defaults to `output/<config.name>/`. Contains `samples/` subfolder and `.safetensors` checkpoints.
3. **Dataset folder** — pulled from `config.datasets[0].folder_path`. Use as visual ground truth.

If anything is missing, ask once. Don't proceed on guesses — the wrong config means the wrong rubric.

## Workflow

### Step 1 — Parse the config

Read the YAML and extract:

- `trigger_word`
- `sample.prompts` (ordered — the index matters; see Step 2)
- `network.linear` (rank — sets your overfitting expectations: higher rank overfits faster)
- `train.steps` and `save.save_every` (tells you how many checkpoints to expect)
- `train.diff_output_preservation` (DOP on or off changes how strict to be about control-prompt bleed)
- `model.arch` and the dataset folder
- Comments in the YAML — Derrick often documents intent there ("texture is essential", "weird artifacts welcome"). Honor those.

Classify the training goal: **style LoRA**, **character LoRA**, or **combined**. The rubric differs — see `references/evaluation-criteria.md`.

### Step 2 — Locate samples and map them to prompts

See `references/output-layout.md` for the file layout. Short version:

- Samples live in `output/<run>/samples/`.
- Filename pattern is `<gen_time_ms>__<zero_padded_9_digit_step>_<count>.jpg` — e.g. `1730812345678__000001500_001.jpg` is step 1500, prompt index 1.
- The `<count>` index matches the position in `sample.prompts` from the YAML.

Build a map: `{step: {prompt_index: image_path, ...}}`. Sort steps ascending. Identify which prompt indices have the trigger vs. which are controls (no trigger).

### Step 3 — Anchor the ground truth (look at the dataset)

Sample 5-8 dataset images using the Read tool. You want to internalize:

- **For style LoRAs**: the medium, palette, texture, mark-making, compositional language, any deliberate "imperfections" the artist values.
- **For character LoRAs**: identity features (face, build, hair), what varies across the set (clothing, pose, lighting).
- **For combined**: both, plus how the two style descriptors in captions differ.

Write a "ground truth" spec **to a text file** (e.g. `output/<run>/ground_truth.txt`) before looking at any samples. Two reasons: it anchors *you* so you don't drift toward whatever the latest checkpoint happens to do — small datasets are easy to overfit to in your own analysis — and it is the exact file the Gemini wide pass (Step 4) consumes via `--ground-truth`. Make it concrete: medium, palette, texture/mark-making, what varies across the set, and any artist-intent notes lifted from the YAML comments. A fidelity judgment with no ground truth is a guess, for you and for Gemini alike.

### Step 4 — Evaluate checkpoints

**THE NON-NEGOTIABLE RULE: every checkpoint gets looked at before you name a winner.** A winner picked from a sparse sample is not a finding — it's a guess. The real peak is routinely the checkpoint you skipped, and bleed/degradation cliffs are often one save wide. Coarse-to-fine is the method; coarse-alone is the failure mode this skill exists to prevent. With the Gemini wide pass, "looked at" splits cleanly: **every checkpoint appears in the wide-pass JSON (that satisfies coverage), and every finalist is eyeballed directly by you (that satisfies the verdict).** The JSON is never the verdict.

Total image count is too high to pour into context image-by-image, so the **wide pass runs through Gemini** (cheap VLM tokens) and you spend Opus vision only on finalists. Work in two passes — but pass 2 is mandatory, not optional, and it must leave no gaps in the candidate region.

**Pass 1 — Gemini wide factual pass, then read the JSON (locate the region).**

Run the wide pass first. It looks at EVERY sample (every checkpoint × every prompt) and writes one structured record per image — fidelity, subject match, texture/palette match, control-bleed, gibberish-text, a `composition` descriptor, and `dataset_subject_leak` — into `sample_review.json`. See `references/gemini-wide-pass.md` for the field schema and how to read it.

```bash
# from the captioning venv (it has google-genai): source .venv-captioning/bin/activate
export GEMINI_API_KEY="..."
python scripts/review_samples_gemini.py \
    --config output/<run>/<run>.yaml \
    --ground-truth output/<run>/ground_truth.txt \
    --goal style          # or character / combined
```

Then **read `sample_review.json`** (it's text — cheap) and build the trajectory from the structured facts: where the style/identity first appears (floor), where fidelity and texture peak (candidate peak), where overfitting and control-prompt bleed begin (ceiling). Cross-reference the `composition` descriptors across *different* prompts at the same step to spot memorization (different prompts → identical layout). The JSON gives full per-checkpoint coverage as text — exactly what the non-negotiable rule demands — without spending Opus vision on it. Pass 1 produces a *hypothesis about the best region* — nothing more. **Do not name a winner from the JSON:** Gemini is the coarse pass, never the verdict — its "looks strong" has masked mode-collapse before (see the hard constraints below and `references/troubleshooting.md`).

**Pass 2 — Eyeball the candidate region with your own eyes (mandatory, no gaps).**
The JSON already covers every checkpoint, so coverage is settled — *verification* is the job now. With the Read tool, directly look at every consecutive checkpoint from just-before the candidate peak to just-after the bleed/degradation onset — no skipped steps — plus the full prompt set for the genuine finalists. You are doing three things the JSON can't:
1. **Pin the exact peak.** Gemini's scores cluster; your eye separates the true peak from its near-identical neighbour.
2. **Find the cliff precisely.** Bleed and quality-collapse often turn on within a single 200-step save. You must see the last-clean step and the first-bled step adjacently.
3. **Catch what the schema flattened** — a subtle aesthetic miss that scored "adequate" but is spiritually wrong, or a per-prompt failure the per-image facts didn't surface.

**Hard constraints on the winner pick:**
- Never name a winner or runner-up you have not directly inspected across the full prompt set.
- Never name a winner that has an **uninspected adjacent checkpoint** on either side (unless it's a trajectory endpoint). If step N is your pick, you must have looked at N-1 and N+1 — the peak could be the neighbor.
- Never extrapolate a checkpoint's quality from 2-3 of its samples. If it's a finalist, look at all of its prompts.
- If you used a subagent summary for a checkpoint, that does not count as inspecting it for the final pick — verify finalists directly (subagent framing like "densest composite" has masked collapse before; see `references/troubleshooting.md` patterns).

**Coverage comes from the Gemini wide pass, not from Claude subagents.** Do NOT fan out image-analysis subagents over the samples — a dozen parallel vision subagents will crash the laptop, and a subagent's prose summary is no more trustworthy than Gemini's JSON for the final pick. The wide pass already made "look at every checkpoint" tractable as one cheap text artifact; spend your own vision budget only on directly eyeballing the 2-3 finalists across their full prompt set. (If the wide pass itself needs more horsepower on a hard/abstract style, bump `--model` to `gemini-3.1-pro-preview` — which is already the default — rather than reaching for subagents.)

### Step 5 — Score against the rubric

See `references/evaluation-criteria.md` for the full rubric per LoRA type. Score categories:

- **Fidelity** — does the trained style/identity actually appear in triggered prompts?
- **Overfitting** — repetition, identical compositions across different prompts, dataset memorization.
- **Generalization** — does the LoRA work for subjects/scenes the dataset never showed?
- **Bleed** — do *non*-triggered control prompts still look like the base model? If they pick up the dataset style, the LoRA is over-baked.
- **Honors artist intent** — if the config or YAML comments specify texture/grain/imperfection, did it survive? (this is the most-missed criterion)

### Step 6 — Recommend

Output structure (always use this template):

```
## Winner

**Checkpoint: step <N>**  (`output/<run>/<run>_<N>.safetensors`)

Why: [2-3 sentences — what this checkpoint nails that others don't]

What it sacrifices vs. other candidates: [1 sentence — honest about tradeoffs]

## Runner-up

**Checkpoint: step <M>** — [one sentence on when you'd prefer this one instead]

## Trajectory summary

- Style/identity first appears: step <X>
- Peak fidelity: step <Y>
- Overfitting starts: step <Z>
- Control-prompt bleed starts: step <W> (or "not observed")

## Merge recommendation (optional)

[Only include if two checkpoints have genuinely complementary strengths. See references/output-layout.md for the ai-toolkit merge_loras.py recipe.]

## Concerns

[Anything that should worry the user about ALL checkpoints — e.g. "no checkpoint generalized cleanly, suggests caption bleed in dataset" or "control prompts pick up palette at all steps, DOP class needs revisiting in v2"]
```

If no checkpoint is acceptable, say so — recommend a v2 config change rather than picking the least-bad option. The user values an honest "this didn't work, here's why" over a forced winner.

### Step 7 — Optional: merge recommendation

Suggest a merge only when:

1. Two checkpoints capture genuinely different things you want both of: early-step has the right *style/texture* but lacks detail; late-step has the *detail/structure* but lost the texture.
2. OR: one checkpoint is great on triggered prompts but bleeds on controls, while an earlier one is weaker on triggered but clean on controls — merging at 0.7/0.3 can dial back the bleed.

Don't suggest merges just to combine "good and slightly different" checkpoints. The cost (doubled rank, slight quality loss) isn't worth it unless there's a real complementary story.

When you do suggest one, provide the exact `scripts/merge_loras.py` invocation. See `references/output-layout.md`.

## Things to watch for (common patterns)

- **Non-monotonic trajectories** — small datasets + high rank often oscillate. Don't assume "later = better"; sometimes step 1500 beats step 2750.
- **Hallucinated text in outputs** — if dataset has any text, late checkpoints often generate gibberish letterforms. Note this in concerns even if subjectively the image looks good.
- **Same composition for different prompts** — memorization. Look at the prompts for two visually-similar samples; if the prompts are different, the LoRA collapsed.
- **Control prompts that subtly shift** — bleed is often gradual. Compare control samples at step 250 vs. final step.
- **"Identity" creeping into style LoRAs** — if specific dataset subjects (a particular bust, a particular fruit) keep recurring across triggered prompts regardless of what the prompt asks for, the LoRA picked up content as if it were style. Note in concerns.

## What NOT to do

- **Don't stop at the coarse sweep.** Large step-jumps locate the region; they never decide the winner. If you crown a checkpoint without having inspected its immediate neighbors, you have guessed, not reviewed. Fill in every checkpoint in the candidate band first (Step 4, Pass 2).
- **Don't extrapolate a checkpoint's quality from 2-3 samples.** A finalist gets its full prompt set looked at. "Step N's espresso prompt looked great" is not "step N is the winner" — the same step may fail on the horse or guitar prompt, and the next step may be strictly better. Both have happened and produced wrong picks.
- **Don't treat the Gemini wide-pass JSON (or any summary) as the verdict.** It is the coarse pass — full coverage as text, nothing more. Its scores ("strong", "adequate") have the same blind spot a subagent's "densest composite" did: they can rate a mode-collapsed checkpoint highly. Use the JSON to locate the candidate band; eyeball the finalists yourself before committing. And do NOT fan out image-analysis subagents over the samples — that crashes the laptop and buys nothing the wide pass didn't already give you.
- Don't pick the final checkpoint by default. The user could already do that — they invoked this skill specifically to find a non-default answer.
- Don't write hedging recommendations ("step 1500 might work, but step 2000 could also be good"). Commit — but commit to a checkpoint you actually inspected with inspected neighbors, not to a guess dressed as a decision.
- Don't enumerate every image you looked at. The output is a recommendation, not a review log.
- Don't conflate "different from base model" with "good". A LoRA that has *baked in* the dataset is also "different from base" — that's overfitting, not success.

## Reference files

Read these as needed:

- `references/output-layout.md` — sample filename parsing, checkpoint naming, dataset paths, merge_loras.py invocation
- `references/evaluation-criteria.md` — detailed rubric per LoRA type (style / character / combined)
- `references/gemini-wide-pass.md` — the `scripts/review_samples_gemini.py` wide pass: when to run it, the `sample_review.json` field schema, and how to read it into a trajectory
