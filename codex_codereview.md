# FIBO Branch Code Review

## Scope
- Compared: `kfir/add_fibo` vs `origin/main` (equivalent to `kfirgoldberg/ai-toolkit:main` in this workspace).
- Commits reviewed:
  - `7810af7` Add FIBO model support
  - `08bded7` working with low_vram
  - `65b97a1` removing custom scheduler logic
  - `bb60dd3` undoing unnecessary changes to custom_flowmatch_sampler.py
- Net changed files:
  - `extensions_built_in/diffusion_models/__init__.py`
  - `extensions_built_in/diffusion_models/fibo/__init__.py`
  - `extensions_built_in/diffusion_models/fibo/fibo_model.py`
  - `jobs/process/BaseSDTrainProcess.py`
  - `toolkit/models/base_model.py`
  - `toolkit/prompt_utils.py`
  - `toolkit/train_tools.py`

## Executive Summary
### 1) Non-FIBO isolation (most important)
- Mostly good: new architecture is isolated under `extensions_built_in/diffusion_models/fibo/` and registry wiring.
- Shared-code touchpoints exist in `PromptEmbeds`/concat helpers and `BaseModel.set_device_state_preset`; these need hardening to guarantee no collateral issues in mixed cache states.

### 2) FIBO training/inference viability
- Core wiring is present (load/generate/noise prediction/LoRA conversion).
- **Blocking issue found** for `cache_text_embeddings`: layer order is corrupted on load due lexicographic sort.

### 3) Common-code reuse vs model-specific logic
- Good reuse overall (`BaseModel`, scheduler patterns, shared prompt embed utilities).
- Some FIBO-specific monkey-patching is unavoidable, but a few guardrails are missing in shared helper code.

### 4) Code style consistency (Flux/SD3 reference)
- Structure is mostly aligned with Flux-family model classes.
- Minor style cleanup needed (unused imports, stale comments, brittle magic constants).

## Findings (Ordered by Severity)

### P0 - Cached FIBO text layers are loaded in the wrong order
- Files:
  - `toolkit/prompt_utils.py:163`
  - `toolkit/prompt_utils.py:175`
  - `toolkit/prompt_utils.py:188`
- What happens:
  - `PromptEmbeds.load()` iterates `sorted(state_dict.keys())`.
  - Keys like `text_encoder_layer_0..text_encoder_layer_11` are sorted lexicographically (`0,1,10,11,2,...`) rather than numerically.
  - This scrambles hidden-layer ordering for DimFusion conditioning.
- Impact:
  - FIBO with `cache_text_embeddings` can train/infer on semantically wrong layer stacks.
  - Directly violates goal #2 (FIBO must work for training/inference).
- Suggested change:
  - Parse integer suffix and sort by numeric index for `text_encoder_layer_*` keys.
  - Apply the same numeric-sort pattern to any indexed key families that can exceed 9.

### P1 - `concat_prompt_embeds` crashes on mixed presence of `text_encoder_layers`
- Files:
  - `toolkit/prompt_utils.py:328`
  - `toolkit/prompt_utils.py:332`
  - `toolkit/prompt_utils.py:335`
- What happens:
  - Logic checks only the first element for `text_encoder_layers`, then assumes all embeds have it.
  - If list is mixed (e.g. partial stale cache migration), it raises `AttributeError` instead of a clear compatibility error.
- Impact:
  - Fragile behavior in practical migration states (partially regenerated text-embedding cache).
  - Can fail before the intended FIBO-specific missing-layer error path.
- Suggested change:
  - Validate all-or-none presence before concatenation.
  - If mixed, raise a clear `ValueError` instructing full cache regeneration.
  - Also validate per-item layer count consistency before padding/concat.

### P1 - `concat_prompt_embeddings` has the same mixed-attribute assumption
- Files:
  - `toolkit/train_tools.py:162`
  - `toolkit/train_tools.py:165`
- What happens:
  - Guard checks only `unconditional` for `text_encoder_layers`, then accesses `conditional.text_encoder_layers` unconditionally.
  - `zip()` silently truncates if layer list lengths differ.
- Impact:
  - Silent partial conditioning or runtime failure in paths that use this helper with FIBO-like embeds.
- Suggested change:
  - Require both inputs to either have layers or both not have layers.
  - Assert equal layer counts and fail loudly with actionable error text.

### P2 - Shared device preset behavior changed for all `BaseModel` extensions
- File:
  - `toolkit/models/base_model.py:1518`
- What changed:
  - Added `cache_text_encoder` preset activation of `text_encoder` in `BaseModel` (previously present in legacy `StableDiffusion` class, but now shared for extension models too).
- Impact:
  - Not obviously wrong, but this is a cross-model behavior change (VRAM profile and caching behavior) for all extension-model classes.
  - Needs explicit smoke validation on at least one non-FIBO extension model (e.g. Flux2/Kontext).
- Suggested change:
  - Keep this change (it is useful), but add regression checks in pre-PR validation to satisfy goal #1.

### P2 - Hardcoded empty-prompt token handling is brittle
- File:
  - `extensions_built_in/diffusion_models/fibo/fibo_model.py:399`
  - `extensions_built_in/diffusion_models/fibo/fibo_model.py:421`
- What happens:
  - Uses hardcoded `bot_token_id = 128000` and row-wide overwrite for mixed empty/non-empty prompts.
- Impact:
  - Tightly coupled to one tokenizer behavior; may break or drift with tokenizer variants/checkpoints.
- Suggested change:
  - Derive empty/special token behavior from tokenizer config or pipeline utility function.
  - Add a guard/fallback if expected token id is unavailable.

### P3 - Style/cleanup nits
- Files:
  - `extensions_built_in/diffusion_models/fibo/fibo_model.py:1`
  - `extensions_built_in/diffusion_models/fibo/fibo_model.py:2`
  - `extensions_built_in/diffusion_models/fibo/fibo_model.py:75`
  - `extensions_built_in/diffusion_models/fibo/fibo_model.py:456`
  - `extensions_built_in/diffusion_models/fibo/fibo_model.py:511`
- Notes:
  - Unused imports (`os`, `Optional`, `AutoTokenizer`).
  - VAE docstrings/comments still mention "Wan VAE"; misleading for FIBO.
- Suggested change:
  - Remove unused imports and align comments/docstrings with FIBO terminology.

## Non-Issue Notes (Verified)
- `jobs/process/BaseSDTrainProcess.py:1167` refactor is behaviorally equivalent to previous logic.
- Model registration wiring in `extensions_built_in/diffusion_models/__init__.py` is straightforward and isolated.
- LoRA key conversion patterns in FIBO follow Flux-style conventions.

## Suggested Change List (Patch Plan)
1. Fix numeric ordering in `PromptEmbeds.load()` for `text_encoder_layer_*` keys.
2. Add strict validation to `concat_prompt_embeds()`:
   - all-or-none `text_encoder_layers`
   - equal layer counts across items
   - clear migration error message on mismatch.
3. Add corresponding validation to `train_tools.concat_prompt_embeddings()`.
4. Replace hardcoded empty-prompt token logic with tokenizer-driven logic (or guarded fallback).
5. Clean up FIBO file imports/comments/docstrings.
6. Run cross-model smoke checks for at least one non-FIBO extension model to confirm no regressions.

## Pre-PR Validation Matrix
### Must pass
- FIBO training step with `cache_text_embeddings=false`.
- FIBO training step with `cache_text_embeddings=true` and freshly regenerated cache.
- FIBO sample generation with:
  - prompt strings path (no cached layers)
  - cached-embeds path (monkey-patched `encode_prompt`).
- Non-FIBO extension smoke run (e.g. Flux2/Kontext) to verify shared helpers/presets did not regress behavior.

### Migration test
- Start with old cached prompt-embed files (without `text_encoder_layers`), then partially regenerate cache.
- Expected: deterministic, clear error instructing full cache regeneration.
- Must not crash with raw `AttributeError`.

## Confidence / Limitations
- High confidence in static/codepath findings and reproducible ordering bug in `PromptEmbeds.load()`.
- End-to-end runtime validation was limited in this environment due missing local dependencies and model assets; this review focuses on branch delta correctness and failure modes observable from code.
