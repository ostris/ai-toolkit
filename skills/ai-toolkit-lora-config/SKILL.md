---
name: ai-toolkit-lora-config
description: Generate a complete AI Toolkit LoRA training config (YAML) from reference images and a training goal. Use PROACTIVELY whenever the user wants to train a LoRA, a character LoRA, a style LoRA, a subject model, or a "fine-tune" on a specific person/aesthetic/dataset — even if they don't name the toolkit explicitly. Triggers on phrases like "train a LoRA on X", "make a LoRA", "train a character model", "train a style model", "finetune on these images", "set up training for", or when the user shares reference images alongside a training intent. Produces a ready-to-run YAML config plus captioning strategy and sample prompt design, tailored to the user's dataset, hardware, licensing needs, and the model that best fits their goal.
---

# AI Toolkit LoRA Config Generator

## When this runs

Someone wants to train a LoRA using [AI Toolkit](https://github.com/ostris/ai-toolkit). They have (or will have) a dataset and a goal. Your job is to produce a complete, ready-to-run training config — plus the captioning strategy and sample prompts that go with it — after asking a short, targeted set of questions.

The most useful LoRA configs are not generic templates. Each decision — model choice, rank, caption strategy, DOP class, sample prompt design — compounds. Your job is to walk through these decisions with the user, not to hand them a one-size-fits-all file.

## The flow

0. **Check for a model brief.** If `briefs/<project>-brief.md` exists (or
   `ai-toolkit-model-brief` just produced one), read it first — it answers
   most of the question flow below, so skip everything it answers. Its
   MUST capabilities are hard filters on model selection (see the
   capability matrix in `references/models.md`), its dial position sets
   rank/steps/DOP posture, its "what stays promptable" intent feeds the
   captioning strategy, and its dream prompts become the sample prompts.
   No brief and the user seems unsure what the model is even for? Route to
   `ai-toolkit-model-brief` before generating anything.
1. **Look at the images.** If the user sent reference images, read them and describe what you see to anchor the rest of the conversation. For a style LoRA you're noting medium, palette, composition, texture. For a character LoRA you're noting identity features, framing, variability.
2. **Identify the goal type.** Ask whichever of these isn't already clear:
   - Character LoRA — capture a specific person/subject, flexible across styles
   - Style LoRA — capture an aesthetic, flexible across subjects
   - Combined — a specific character rendered in a specific style
3. **Gather context.** Ask only the questions that matter for the config you're about to produce. Don't run a generic checklist. See "Question flow" below.
4. **Produce the config.** Emit the YAML, explain the choices that aren't obvious, include the captioning strategy, and give sample prompts that actually test the right thing.
5. **Flag the pitfalls.** Warn about the traps that derail this kind of training (e.g., assisted prompts in sample prompts, mixing styles in one caption set, turbo models needing assistant LoRAs).

## Question flow

Ask just the questions you need answered. Skip anything you already know from the conversation. Don't interview by checklist — cluster questions and explain *why* you're asking.

**Always need:**
- Goal type (character / style / combined) — usually inferrable from images + goal, confirm if ambiguous
- Dataset size (number of images)
- Model preference or compute budget (VRAM available)

**For character LoRAs:**
- Is the character always the same gender/presentation? (informs DOP class)
- Do you have a trigger word in mind, or should I suggest one?
- Do you want style flexibility, or a specific target style?

**For style LoRAs:**
- Describe the content type — are the training images subjects-on-white-backgrounds, scenes, portraits, mixed? (informs DOP class and sample prompts)
- What's the exact phrase someone should type at inference to activate the style? (this becomes the style descriptor appended to every caption)

**For combined:**
- Are the character dataset and style dataset in separate folders? (two-dataset config)
- How many images in each?

**Compute/licensing:**
- What's your VRAM budget? (24GB / 40GB / 80GB+ / Colab)
- Does commercial licensing matter? (rules out FLUX.1-dev and FLUX.2-dev)

Don't over-ask. If the user says "95GB, commercial is fine, go wild," take that and run.

## Model selection

Pick the model based on the user's goal, licensing needs, and VRAM. See `references/models.md` for full details on each. When a model brief exists, apply its MUSTs through the capability matrix there (lane first, hard filters second, trade-offs last) before this shorthand table.

Decision shorthand:

| Situation | Model |
|---|---|
| Highest-quality character, non-commercial OK | `Flux.2-dev` (arch: `flux2`) |
| Highest-quality character, commercial | `Flux.2-Klein-9B` (arch: `flux2_klein_9b`) |
| Style LoRA, fast iteration, commercial | `Z-Image Turbo` (arch: `zimage`, needs assistant adapter) |
| Open-source community, commercial | `Chroma` (arch: `chroma`) |

If the user names a model, use it. If they don't care, pick based on the table and explain why.

## Config structure

Use the appropriate reference YAML as a skeleton. Don't write configs from scratch — start from the closest existing example in `config/examples/` and modify:

- Character: `config/examples/train_lora_flux2_character_24gb.yaml`
- Style: `config/examples/train_lora_zimage_turbo_style.yaml`
- Combined (two datasets): `config/examples/train_lora_flux2_character_24gb.yaml` (the two-dataset structure)

Every config needs:

```yaml
job: extension
config:
  name: "<descriptive-name>"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      trigger_word: "<unique-token>"
      network:
        type: "lora"
        linear: <16 for character | 32 for style>
        linear_alpha: <same as linear>
      save: ...
      datasets: [ ... ]
      train: ...
      logging:                # ALWAYS include — see below
        log_every: 1
        use_ui_logger: true
      model: ...
      sample: ...
```

**Always include the `logging:` block.** Without it, ai-toolkit defaults to `EmptyLogger` and saves no loss/LR data to disk — you lose the ability to diagnose trajectories post-hoc or graph the run in the web UI. The block above writes `output/<run-name>/loss_log.db` (SQLite) with per-step learning_rate + loss values. Pair with `use_wandb: true` + `project_name` / `run_name` instead only when the user explicitly wants cloud visibility AND has `WANDB_API_KEY` set.

See `references/parameters.md` for a full parameter reference with notes on what to tune and why.

## Captioning strategy

This is where most LoRAs succeed or fail. See `references/captioning.md` for the full playbook, but the one-liner:

**The LoRA learns what you don't caption.** Whatever is omitted from captions becomes part of the baked-in identity/style bound to the trigger word. Whatever is described becomes a variable the user can control at inference.

- **Character LoRA**: describe clothing, pose, setting, lighting, framing, medium. Omit face, hair, eye color, build, age, gender.
- **Style LoRA**: describe content only. Every caption ends with the exact style descriptor phrase (e.g. `"oil pastel illustration"`).
- **Combined**: use the style-descriptor-as-suffix trick. Identity set ends in its descriptor (e.g. `"modern digital photograph"`), style set ends in the target descriptor (e.g. `"2010s smartphone photograph"`). This makes style promptable at inference.

Always recommend a VLM captioning script. See `references/captioning.md` for the Gemini system prompt and Colab cells to copy.

## Sample prompts — the most-missed pitfall

Sample prompts during training evaluation should be **minimal**. Do NOT include the style description in the sample prompt — that's the LoRA's job.

**Wrong:**
```
zh3ng a red bicycle rendered as an oil pastel crayon illustration on white paper with heavy waxy pigment...
```

The base model will render oil pastel from this prompt whether or not the LoRA is working, making it impossible to tell if training is actually taking hold.

**Right:**
```
zh3ng a red bicycle
```

Plus a control prompt without the trigger to see what the base model does on its own:
```
a red bicycle
```

For character LoRAs, include prompts that test multiple style descriptors (the ones you used in captions) plus generalization prompts (styles the LoRA never saw). See `references/sample-prompts.md` for full templates.

When a model brief exists, **derive the sample prompts from its dream prompts**: strip any style description (same minimal-prompt rule as above), add the trigger, and make sure every MUST capability has at least one prompt that tests it (a text MUST gets a prompt with lettering; an edit MUST gets a ctrl_img prompt). The brief's requirements are the test suite — the run should generate evidence for each one every sample cycle.

## DOP (Diff Output Preservation)

Recommended for character LoRAs and for style LoRAs where the style is subtle (e.g. both training sets are photographs with different aesthetics). Skip for very distinct style LoRAs where the base-model prior is far from the target.

- Character class: `"person"`, `"woman"`, or `"man"`
- Style class: pick a broad content category — `"illustration"`, `"photograph"`, `"portrait"` — NOT the specific style itself
- Default multiplier: `1.0`. Drop to `0.5` if learning is too weak; raise to `1.5` if non-triggered outputs still show dataset style bleed.

DOP nearly doubles memory usage during training — always pair with `gradient_checkpointing: true` unless the user has significant headroom (60GB+ free after the base model loads).

If the user reports "no difference between baseline and trained samples," DOP is usually the first suspect. See `references/troubleshooting.md`.

## VRAM presets

Adjust these based on the user's compute. See `references/parameters.md` for why each lever matters.

**24GB (or Colab free/Pro):**
```yaml
model:
  quantize: true
  quantize_te: true
train:
  batch_size: 1
  gradient_checkpointing: true
datasets:
  - resolution: [ 512, 768, 1024 ]
```

**40-48GB:**
```yaml
model:
  quantize: true
  quantize_te: false
train:
  batch_size: 2
  gradient_checkpointing: true
```

**80-95GB (H100, Colab Pro+):**
```yaml
model:
  quantize: false
  quantize_te: false
train:
  batch_size: 2-4    # smaller for small datasets (<40 images)
  gradient_checkpointing: true   # STILL needed if DOP is on
datasets:
  - resolution: [ 512, 768, 1024, 1280 ]
```

Note on batch size: counterintuitively, larger batches are worse for small datasets (under ~40 images) because they cycle through the dataset too fast and accelerate overfitting. Stay at batch 1-2 for small sets even with abundant VRAM.

## Step count heuristics

Rough starting points. Always encourage user to check 250-step interval saves and pick the sweet spot manually.

- Character LoRA, 20-40 images: 1500-2500 steps
- Style LoRA, 30-50 images: 2500-3000 steps
- Combined (character + style), 100+ effective images: 2500-3500 steps

For very small datasets (under 20 images), 1000-1500 may be enough. For large datasets (100+ images), 3000-5000.

## Producing the output

After the questions are answered:

1. **Write the YAML config to disk** at `config/examples/<descriptive-name>.yaml` using the Write tool, AND show it as a code block in the response. Both. The chat-only code block disappears when the conversation ends; the file persists. Always state the saved path explicitly so the user can run `python run.py config/examples/<name>.yaml` immediately. If you also produce a captioning script (which you should, see step 3), write that to disk too — keep config and script symmetric in where they live, since users assume "if one is on disk, both are."
2. **Explain the non-obvious decisions** — why this model, why this rank, why DOP is on/off, why this step count. Two-three sentences per decision max.
3. **Write the captioning script to disk** at `scripts/caption_<name>_dataset_gemini.py`, modeled after the closest existing script in `scripts/` (chemigram for content/style inversion, character_dataset for standard character, style_dataset for standard style). Include the project-specific avoid-word list and the inverted-rule system prompt directly in the script — don't make the user assemble it from references.
4. **Give the sample prompts** that go in the YAML — minimal content-only, control prompts, generalization tests. (Already in the YAML; just summarize what each prompt tests.)
5. **Flag 2-3 things to watch during training.** The most common failure modes for this specific config.

**Symmetry rule:** if any artifact is written to disk, all artifacts (config + caption script) should be written to disk. The asymmetry of "script saved, config in chat only" is invisible to the user — they assume both exist as files and discover the gap later when they go to run training.

Output length target: the explanation + saved-path summary + sample prompt list + watchlist should fit on one screen. Don't dump the full YAML again in chat after writing it — reference the file path instead, with a short summary of the key knobs.

## Common pitfalls to warn about

Based on the mistakes that consistently derail this work:

1. **Assisted prompts in sample prompts** — hides whether the LoRA is actually learning
2. **Conv LoRA layers on transformer models** — `conv` and `conv_alpha` don't apply to Flux/Z-Image/Chroma/etc. Only `linear`/`linear_alpha`.
3. **Captioning identity features in character LoRAs** — causes poor style flexibility
4. **Captioning style in style LoRAs** — prevents the trigger word from binding to the style
5. **Same style descriptor for visually-different datasets in combined training** — the model can't distinguish them
6. **Turbo/distilled models without assistant adapter** — Z-Image Turbo and Flux Schnell need `assistant_lora_path`
7. **DOP without gradient_checkpointing on small VRAM** — OOM
8. **Large batch on small datasets** — accelerates overfitting

## Reference files

Load these as needed:

- `references/models.md` — per-model config details (HF paths, arch names, required flags, licensing)
- `references/captioning.md` — full captioning strategy plus the Gemini VLM system prompt and Colab cells
- `references/parameters.md` — what every `train:` and `model:` parameter does, what to tune, when
- `references/sample-prompts.md` — templates for sample prompts by LoRA type
- `references/troubleshooting.md` — "my LoRA isn't learning" / "style isn't showing up" / OOM / etc.
