---
name: dop-class-advisor
description: This skill should be used when selecting or validating `train.diff_output_preservation_class` (DOP class) for AI Toolkit LoRA training. Use it when a user asks for the best class word, DOP tuning guidance, or YAML-ready DOP settings.
---

# DOP Class Advisor

## Purpose

Select the best `train.diff_output_preservation_class` for AI Toolkit training.
Recommend one primary class plus backups, and output YAML-ready settings.

## When To Use

Use this skill when:
- Choosing a DOP class for a new LoRA dataset.
- Improving class separation (for example, subject vs generic class).
- Fixing weak/overfit behavior caused by poor DOP class selection.
- Reviewing whether an existing DOP class is too broad or too specific.

## Required Inputs

Collect these inputs before deciding:
- Model/training context: image vs video, architecture if known.
- Trigger token/word (for example `grarocks`).
- Target concept summary in one sentence.
- 5 to 20 representative captions (or a summary of their recurring nouns).

If some inputs are missing, continue with best effort and state assumptions.

## Decision Workflow

1. Extract the concept head noun.
- Reduce the concept to a generic category phrase.
- Keep class terms short (1 to 3 words), singular when natural.

2. Remove non-class details.
- Remove names, trigger tokens, camera/lens terms, styling adjectives, locations, and event phrasing unless they define the class itself.

3. Generate candidates at three levels.
- Near class: most specific generic category (`young dancer`).
- Mid class: broader category (`dancer`).
- Broad class: fallback (`person`).

4. Score candidates and pick one.
- Prefer candidates that are:
- Generic enough to preserve base priors.
- Specific enough to match dataset semantics.
- Common natural language tokens for the text encoder.
- Free of trigger token overlap.

5. Return recommendation and YAML.
- Provide primary + 2 alternatives and short tradeoffs.
- Include a ready-to-paste YAML block.

## Hard Rules

- Never use the trigger token itself as DOP class.
- Never use full sentence prompts as DOP class.
- Avoid overly broad classes unless dataset is truly broad.
- Prefer nouns over style phrases.
- Keep DOP class stable across runs to make comparisons valid.

## Output Format

Return exactly:

1. `Recommended class:` one phrase.
2. `Alternatives:` two options with one-line tradeoff each.
3. `Why:` brief reasoning tied to dataset semantics.
4. `YAML:` copy-paste block:

```yaml
train:
  diff_output_preservation: true
  diff_output_preservation_class: "<recommended>"
  diff_output_preservation_multiplier: 1.0
```

5. `Tuning note:` one sentence for when to move multiplier to `0.5` or `1.5`.

## Example

Input concept: cinematic photos of young girls in a dance competition  
Trigger: `grarocks`

Expected recommendation style:
- Recommended class: `young dancer`
- Alternatives: `dancer`, `person`
