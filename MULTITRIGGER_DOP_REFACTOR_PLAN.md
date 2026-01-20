# MultiTrigger DOP Refactor: Move to Dataloader

## Overview

Refactor the MultiTrigger DOP (Differential Output Preservation) implementation to follow Diffusers conventions by moving all `diff_output_preservation_embeds` handling into the dataloader classes. SDTrainer should not manually handle DOP embeddings - they should be treated the same as regular `prompt_embeds`.

## Current Architecture Issues

**Problem:** SDTrainer.py manually manages DOP embeddings throughout the training loop:
- Lines 2580-2646: Manual per-file DOP embedding loading and concatenation
- Line 426-470: Trigger→class text replacement logic in trainer
- Lines 860-883: DOP precompute mixed with trainer setup
- Manual fallback logic when cache missing

**Goal:** Make DOP embeddings first-class citizens in the dataloader, just like `prompt_embeds`, `control_tensor`, and `clip_image_embeds`.

## Design Approach

### Principle: Dataloader Owns All Embedding Loading

Following existing patterns (`TextEmbeddingCachingMixin`, `CLIPCachingMixin`):

1. **Dataset Configuration** provides DOP parameters (triggers, classes)
2. **Dataloader** loads precomputed DOP embeddings from cache (read-only)
3. **Batch DTO** collates per-file embeddings into batch
4. **Trainer** consumes `batch.dop_prompt_embeds` directly (no manual handling)

### Text Transformation Strategy

**Key insight:** The trigger→class text mapping must happen BEFORE encoding, but the dataloader doesn't encode (only loads from cache). Solution:

1. **Precompute phase** (trainer-driven, once at setup):
   - Trainer calls dataloader method: `dataset.precompute_dop_embeddings(triggers, classes, replacements_digest)`
   - Dataloader iterates files, applies text transformations, requests encoding from trainer's `sd.encode_prompt()`
   - Saves to disk with stable cache keys

2. **Training phase** (dataloader-driven, every batch):
   - FileItemDTO loads precomputed DOP embeddings via `load_dop_prompt_embedding()`
   - Batch DTO collates into `batch.dop_prompt_embeds`
   - Trainer uses directly (no knowledge of triggers/classes)

## Implementation Plan

### Phase 1: Keep DOP Configuration at Train Level

**File:** `toolkit/config_modules.py` (TrainConfig class)

**No changes needed** - DOP config stays at train level:
```python
class TrainConfig:
    # Existing fields...
    diff_output_preservation: bool = False
    diff_output_preservation_class: Optional[str] = None  # CSV: "Woman, Gun"
    # trigger comes from parent config, not train-specific
```

**Rationale:** Simpler config, sufficient for standard use cases. All datasets in a job share same DOP triggers/classes.

---

### Phase 2: Create DOP Text Transformation Utility

**File:** `toolkit/prompt_utils.py` (already has CSV parsing)

Add new function:
```python
def build_dop_replacement_pairs(
    triggers_csv: str,
    classes_csv: str,
    case_insensitive: bool = False
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Returns:
        - pairs: [(trigger, class), ...] sorted by trigger length DESC
        - digest: MD5 hash of pairs for cache invalidation
    """
```

Move existing `_map_triggers_to_classes_in_text()` from SDTrainer to standalone utility:
```python
def apply_dop_replacements(
    caption: str,
    replacement_pairs: List[Tuple[str, str]],
    case_insensitive: bool = False,
    debug: bool = False
) -> str:
    """Apply trigger→class replacements following MultiTrigger.md algorithm"""
```

**Rationale:** Shared utility prevents code duplication, testable in isolation.

---

### Phase 3: Add DOP Precompute to Dataloader Mixin

**File:** `toolkit/dataloader_mixins.py` (TextEmbeddingCachingMixin)

Add new method after `cache_text_embeddings()`:

```python
def precompute_dop_embeddings(
    self,
    triggers_csv: str,
    classes_csv: str,
    encode_fn: Callable[[str], PromptEmbeds],  # Trainer's sd.encode_prompt
    case_insensitive: bool = False,
    debug: bool = False
):
    """
    Precompute DOP embeddings for all file items in dataset.

    Args:
        triggers_csv: "Jinx, Zapper"
        classes_csv: "Woman, Gun"
        encode_fn: Function to encode text → PromptEmbeds
        case_insensitive: Match triggers case-insensitively
        debug: Log replacements

    Process:
        1. Build replacement pairs and digest
        2. For each file item:
            a. Apply trigger→class replacements to caption
            b. Call encode_fn(transformed_caption) → PromptEmbeds
            c. Save to _dop_text_embedding_path
            d. Store digest in file_item for cache key stability
    """
```

**Key details:**
- Uses `build_dop_replacement_pairs()` from Phase 2
- Computes `replacements_digest` once, stores in each FileItemDTO
- Uses atomic writes (existing pattern)
- Respects `main_process_first()` for distributed training
- Reports cache hit rate before encoding

---

### Phase 4: Update FileItemDTO DOP Loading

**File:** `toolkit/dataloader_mixins.py` (TextEmbeddingFileItemDTOMixin)

**Current state:** `load_dop_prompt_embedding()` already exists (lines 2623-2661)

**Changes needed:**
1. Store `_dop_replacements_digest` persistently (set during precompute)
2. Ensure digest is included in cache path calculation (already done)
3. Add fallback detection: if DOP enabled but embedding missing, log warning

**No major changes needed** - existing implementation already supports this pattern.

---

### Phase 5: Update Batch DTO DOP Collation

**File:** `toolkit/data_transfer_object/data_loader.py` (DataLoaderBatchDTO)

**Current state:** DOP collation already exists (lines 241-254)

**Changes needed:**
1. Add defensive check: if DOP enabled but some items missing embeddings, log warning once
2. Otherwise **no changes** - existing implementation already correct

---

### Phase 6: Always Load DOP Embeddings (No Scheduling in Dataloader)

**File:** `toolkit/data_loader.py`

**Approach:** Dataloader always loads DOP embeddings if they exist (cached, fast I/O).

**No scheduling logic needed in dataloader:**
- Trainer handles scheduling via `_is_dop_scheduled()`
- Batch always has `dop_prompt_embeds` available
- Trainer decides whether to use them based on step count

**Rationale:** Simpler separation of concerns. Dataloader provides data, trainer controls usage. Matches how `prompt_embeds` works (always loaded, always available).

---

### Phase 7: Refactor SDTrainer to Use Dataloader-Provided Embeds

**File:** `extensions_built_in/sd_trainer/SDTrainer.py`

#### 7a. Setup Phase Changes (lines 755-877)

**Remove:**
- Manual DOP embedding precompute loop (lines 860-883)
- `_map_triggers_to_classes_in_text()` method (move to prompt_utils.py)

**Keep:**
- `_dop_replacements` pairs building (needed for fallback)

**Add:**
```python
# After regular text embedding cache (around line 800)
if self.train_config.diff_output_preservation:
    triggers_csv = self.trigger_word  # May be CSV from config
    classes_csv = self.train_config.diff_output_preservation_class

    # Build replacement pairs once (used for fallback encoding)
    from toolkit.prompt_utils import build_dop_replacement_pairs
    self._dop_replacement_pairs, self._dop_digest = build_dop_replacement_pairs(
        triggers_csv=triggers_csv,
        classes_csv=classes_csv,
        case_insensitive=False
    )

    # Delegate precompute to dataloader (all datasets use same triggers/classes)
    for dataset in self.data_loader.datasets:
        dataset.precompute_dop_embeddings(
            triggers_csv=triggers_csv,
            classes_csv=classes_csv,
            encode_fn=lambda caption: self.sd.encode_prompt(caption, **encode_kwargs),
            case_insensitive=False,
            debug=getattr(self.train_config, 'diff_output_preservation_debug', False)
        )
```

#### 7b. Training Loop Changes (lines 2580-2646)

**Remove entire block:**
```python
# Lines 2580-2646: Manual DOP embedding loading and fallback
```

**Replace with:**
```python
# Simply retrieve from batch (already collated by dataloader)
self.diff_output_preservation_embeds = batch.dop_prompt_embeds

# Fallback only if scheduled but missing
if self._is_dop_scheduled(for_encoding=True):
    if self.diff_output_preservation_embeds is None:
        # Fallback: encode on-the-fly if cache missing
        logger.warning("DOP cache missing for batch - encoding on-the-fly (slower)")
        from toolkit.prompt_utils import apply_dop_replacements

        dop_captions = [
            apply_dop_replacements(cap, self._dop_replacement_pairs, debug=False)
            for cap in batch.captions
        ]
        self.diff_output_preservation_embeds = self.sd.encode_prompt(dop_captions, ...)
```

#### 7c. Remove Obsolete Methods

**Delete:**
- `_map_triggers_to_classes_in_text()` (moved to prompt_utils.py)
- CSV parsing logic (lines 183-244 initialization - replaced with build_dop_replacement_pairs call)

**Keep:**
- `_dop_replacement_pairs` storage (needed for on-the-fly fallback encoding)
- `_is_dop_scheduled()` - still needed for loss calculation timing
- `_diff_output_preservation_exec_count` - metrics tracking
- Preservation loss calculation (lines 3425-3517)

---

### Phase 8: Update Dataset __getitem__ Method

**File:** `toolkit/data_loader.py` (AiToolkitDataset class)

**Current:** `__getitem__()` calls `load_prompt_embedding()` for regular embeds

**Add:** Always load DOP embeddings if precomputed:
```python
def __getitem__(self, idx):
    file_item = self.get_file_item(idx)

    # Regular embedding (existing)
    file_item.load_prompt_embedding()

    # DOP embedding (new) - always load if available
    if hasattr(self, '_dop_enabled') and self._dop_enabled:
        # Load DOP embedding (already cached during precompute)
        # Uses cache keys stored during precompute phase
        file_item.load_dop_prompt_embedding(
            dop_class=self._dop_classes_csv,
            trigger_word=self._dop_triggers_csv,
            dop_replacements_digest=self._dop_digest
        )
        # If missing, file_item.dop_prompt_embeds stays None → trainer fallback

    return file_item
```

**Rationale:** Simple, always-on loading. Fast since cached. Trainer decides usage based on scheduling.

---

### Phase 9: Handle Edge Cases

#### 9a. Multi-Trigger with Single Cached Embed

**Current behavior:** SDTrainer caches single DOP embed if only one class (line 758)

**New behavior:**
- If `len(classes) == 1`, precompute still generates per-file embeddings
- No special casing needed - dataloader doesn't distinguish single vs multi

#### 9b. On-the-Fly Fallback (Keep for Backward Compatibility)

**Current:** Trainer falls back to on-the-fly encoding if cache missing

**New:** Keep minimal fallback in trainer (as shown in Phase 7b)

**Rationale:** Maintains full backward compatibility. Allows training to continue if precompute was skipped or cache invalidated.

#### 9c. Distributed Training

**Existing:** `main_process_first()` ensures only rank 0 writes cache

**Ensure:** `precompute_dop_embeddings()` wraps loop in `main_process_first()` (existing pattern)

---

## Summary of Key Decisions

Based on user preferences:

1. **Loading Strategy:** Always load DOP embeddings (simple, fast, matches prompt_embeds pattern)
2. **Fallback Behavior:** Keep on-the-fly encoding if cache missing (backward compatible, log warning)
3. **Config Level:** Train-level only (simpler, no per-dataset override needed)

This approach prioritizes:
- **Simplicity:** Minimal changes to existing config structure
- **Backward compatibility:** Existing configs work unchanged, fallback prevents training failures
- **Clean architecture:** Dataloader owns embedding I/O, trainer owns training logic
- **Performance:** Precompute when possible, fallback when needed
