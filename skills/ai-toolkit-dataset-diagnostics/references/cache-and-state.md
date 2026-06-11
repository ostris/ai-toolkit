# Cache and state issues

AI Toolkit caches several things to make repeated runs fast. When caches
go stale, training appears to work but produces wrong results — the
cache returns yesterday's data while you're trying to train tomorrow's
model.

This doc covers what's cached, where, when each cache invalidates
automatically, and how to nuke caches cleanly when they don't.

## What's cached

| Cache | Location | Set by | Cache key |
|-------|----------|--------|-----------|
| **Latent cache** (image VAE encodings) | Inside dataset folder, typically `.dataset_cache/` or similar | `cache_latents_to_disk: true` (default true for image datasets) | File path |
| **Text embedding cache** | Same area | `cache_text_embeddings: true` (typically true for video) | Caption string content |
| **Bucket assignments** | In-memory, computed at startup from `resolution: [...]` | Always | File aspect ratio |
| **Checkpoint** | Output folder, `output/<job_name>/<step>.safetensors` | `save:` block | Step number |

## Latent cache

### When it auto-invalidates

- New file at a new path → recomputes
- File deleted → entry orphaned, harmless

### When it does NOT auto-invalidate (and bites you)

- **File content changes but path doesn't.** You replace `image.jpg`
  with a different image at the same path. Cache returns the old latent.
- **Resolution buckets change.** You bumped `resolution: [512, 768]` to
  `[768, 1024, 1280]`. Cache contains 512-bucket latents which don't
  match the new bucket sizes.
- **VAE itself changes.** You point at a different model `name_or_path`
  with a different VAE. Cache contains latents from the previous VAE.

### How to invalidate cleanly

```bash
# Find any cache directories near the dataset
find "$DATASET" -type d \( -name '*cache*' -o -name '.dataset_cache*' \)

# Delete them
find "$DATASET" -type d -name '*cache*' -exec rm -rf {} +
```

Then re-run. Latent cache rebuilds on first epoch.

### Symptom of stale latent cache

- Training "works" but the LoRA learns nothing or learns the wrong thing
- After bumping resolution, training crashes at first batch with a shape
  mismatch
- After replacing dataset images with same-named replacements, the LoRA
  doesn't learn the new content

## Text embedding cache

### When it auto-invalidates

- Caption text changes (different string → different cache key) →
  recomputes
- New caption file → recomputes

### When it does NOT auto-invalidate (and bites you)

- **Text encoder model changes.** Same caption → same key, but the
  embedding it produces is different if the TE has been swapped.
- **Caption file character encoding changes** (e.g., adding a BOM, or
  changing CRLF to LF). The string-as-key matches but the cache content
  may be stale or hash-collide weirdly.
- **`shuffle_tokens: true` and tokens were reordered between runs.** The
  cache holds the old order's embedding.

### How to invalidate cleanly

Same as latent cache — delete cache directories. The text embedding
cache lives in the same general area.

```bash
find "$DATASET" -type d -name '*embedding*' -exec rm -rf {} +
find "$DATASET" -type d -name '*cache*' -exec rm -rf {} +
```

### Symptom of stale embedding cache

- Recaptioned but trigger never fires (caches still serve old
  embeddings — but this is rare because content changes invalidate keys)
- Changed `caption_dropout_rate` and behavior didn't change (because
  cached embeddings include their pre-dropout state)
- Switched models / TE quantization but embedding shapes mismatch at
  first batch

## Caption dropout interaction with caching

```python
# toolkit/dataloader_mixins.py:386
if self.dataset_config.caption_dropout_rate > 0
    and not short_caption
    and not self.dataset_config.cache_text_embeddings:
    # dropout path
```

**The dropout path is disabled when text embeddings are cached.** This is
intentional — dropout would invalidate cache hits. But it also means:

- If you set `caption_dropout_rate: 0.05` AND `cache_text_embeddings: true`,
  dropout silently does NOTHING.
- The training behavior matches `caption_dropout_rate: 0.0` even though
  the YAML says 0.05.

If you intend dropout to fire, disable embedding caching. If you want
embedding caching (typical for video), set `caption_dropout_rate: 0.0`
explicitly so the YAML matches actual behavior.

## Checkpoint resume

### What gets saved per checkpoint

- LoRA weights (`<step>.safetensors`)
- Optimizer state (sometimes)
- Training step count
- Random sampler state

### What does NOT get saved

- The dataset configuration the run was using
- The captions the run was using
- The latent / text embedding caches (those live in the dataset folder)

### Implications

- If you change `network.linear` (rank) or `network.linear_alpha`
  between the original run and the resume, weights load partially or
  fail silently. Resume produces drift.
- If you change the dataset between runs, the checkpoint resumes its
  step count but trains on different data. The step counter will be
  high but the "fresh" training is from step 0 effectively.
- If you change the model (`name_or_path` or `arch`), the LoRA weights
  are now loaded into a different base. The output is unpredictable.

### How to resume safely

1. Use the SAME YAML you trained the original with.
2. Add a `resume:` directive (check the actual config syntax — varies
   by version) pointing at the checkpoint.
3. Don't edit anything else.

If you want to "fork" a checkpoint to a new config, you generally can't
— retrain from scratch with the new config.

## Output folder hygiene

Every job creates `output/<job_name>/`. Inside:

```
output/<job_name>/
├── 250.safetensors           # checkpoint at step 250
├── 500.safetensors
├── ...
├── samples/                  # sampled images during training
│   ├── 250_sample_001.png
│   └── ...
└── config.yaml               # the YAML the job was launched with
```

The saved YAML inside the output folder is a snapshot of what the job
was actually configured with. **Source of truth** when you want to
remember what trained a particular checkpoint.

## When to nuke EVERYTHING and start over

- Recaptioning a dataset → delete latent cache + text embedding cache,
  then retrain from step 0 (don't resume)
- Changing model / arch → delete latent cache (different VAE), retrain
- Changing rank / alpha → delete checkpoint, retrain (weights are
  topology-incompatible)
- Changing resolution buckets → delete latent cache, retrain

## When NOT to nuke caches

- Just re-running with the same config → caches are correct, save time
- Bumped `steps` / `lr` / `save_every` / `sample` block → caches are
  fine, resume or restart works

## Detecting stale state

If output looks wrong but logs look fine, suspect cache:

```bash
# Did anything in the dataset / config change since last run?
find "$DATASET" -newer /tmp/last_run_marker -type f | head

# When was each cache directory last modified?
find "$DATASET" -type d -name '*cache*' -exec stat -f "%Sm %N" {} +

# When was the YAML last modified?
stat -f "%Sm %N" /path/to/your_config.yaml
```

If the YAML is newer than the cache directories, the cache may not
reflect the current config. Investigate which config knob changed and
whether it should have invalidated the cache.

## Pragmatic rules

1. **When in doubt, delete the cache.** Cache rebuild is minutes; chasing
   a phantom bug is hours.
2. **Don't resume across config changes.** Resume is for crashes and
   manual stops, not for "let me try this with different rank."
3. **Inspect `output/<job_name>/config.yaml` to know what trained a
   checkpoint.** Don't assume the active YAML matches.
4. **Keep one job per output folder.** If you re-launch with the same
   `name:`, you're overwriting.
5. **If `cache_text_embeddings: true`, set `caption_dropout_rate: 0.0`**
   explicitly. The combination silently disables dropout, and the explicit
   0.0 makes the intent clear in the YAML.
