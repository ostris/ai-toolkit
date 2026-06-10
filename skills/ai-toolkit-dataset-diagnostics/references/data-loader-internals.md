# AI Toolkit data loader internals

How the loader actually works. Read this when a symptom doesn't match a
known error message and you need to reason from first principles.

## File locations

| File | Role |
|------|------|
| `toolkit/data_loader.py` | Top-level dataset class: scanning, file matching, bucket assignment |
| `toolkit/dataloader_mixins.py` | Per-item logic: caption loading, image/video reading, augments, latent caching |
| `toolkit/config_modules.py` | `DatasetConfig` dataclass — defines all valid YAML fields |

## The video-mode trigger

```python
# toolkit/data_loader.py:392
self.is_video = dataset_config.num_frames > 1
```

This single line decides whether the loader looks for videos or images.
There's no other config flag for "this is a video dataset." If `num_frames`
is unset, it defaults to 1 → image mode.

**Implication:** if your dataset is videos and you forget `num_frames`, the
loader scans for `.jpg/.png/.webp` and finds 0. The error is "no images
found" even though your folder is full of `.mp4` files.

## File scanning

### Image mode

Scans for files with these extensions (case-sensitive):
```python
{".jpg", ".jpeg", ".png", ".webp"}
```

Filters:
- Skips dotfiles (`name.startswith('.')`) — AppleDouble + .DS_Store
- Recursive scan? Depends on dataset class — most use `rglob("*")` so
  yes, but verify for your specific class

### Video mode

Scans for files with these extensions (case-sensitive):
```python
{".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
```

Same dotfile filter. Same recursion behavior.

**Case sensitivity matters.** `.JPG` and `.MOV` won't match. Lowercase
your extensions before training.

## Caption matching

For each image/video at `<base>.<ext>`, the loader looks for
`<base>.<caption_ext>` in the same directory.

```yaml
datasets:
  - folder_path: "/path"
    caption_ext: "txt"   # default — looks for <name>.txt
```

If the caption file doesn't exist, the clip is treated as **unconditional**
(empty caption). No error is raised. This is silent and will manifest as
"trigger never fires at inference."

To verify captions are being matched:
```bash
# Counts: should match
ls /your/dataset/path/*.mp4 | wc -l
ls /your/dataset/path/*.txt | wc -l
```

If they don't match, find the gaps:
```bash
for video in /your/dataset/path/*.mp4; do
  txt="${video%.mp4}.txt"
  [ ! -f "$txt" ] && echo "MISSING CAPTION: $video"
done
```

## Caption dropout

```python
# toolkit/dataloader_mixins.py:386
if self.dataset_config.caption_dropout_rate > 0
    and not short_caption
    and not self.dataset_config.cache_text_embeddings:
    if rand < self.dataset_config.caption_dropout_rate:
        # caption is replaced with empty string
```

Caption dropout fires *per training step*, randomly replacing the caption
with empty string. This is intended as regularization but in practice on
small datasets or strong-style LoRAs causes "style ↔ empty prompt" learning,
which manifests as bleed at inference.

**For most LoRAs, set `caption_dropout_rate: 0.0`.** The default of 0.05
is a holdover from older training recipes and rarely improves results
on focused-style or character LoRAs.

**Note:** dropout is disabled when `cache_text_embeddings: true` because
embeddings are cached at first encounter. Don't rely on this to "save you"
— if you trained once with `cache_text_embeddings: false` and dropout=0.05,
then re-ran with `cache_text_embeddings: true`, the cache holds your
original mix of dropped/undropped embeddings.

## Bucket assignment

The loader bins each file into the closest bucket from the
`resolution: [...]` array. Buckets are discrete sizes (square or
height/width pairs) the model is trained at.

```yaml
resolution: [512, 768, 1024, 1280]
# Assigns each file to a bucket whose largest dimension is closest
```

For 16:9 video like Wan22:
```yaml
resolution: [512, 720]
# Pairs interpreted as [width, height] for video buckets
```

If multiple datasets are in one job, ALL of them must define `resolution`
or NONE of them can. Mixing causes the
`buckets not found on dataset <path>` error.

## EXIF orientation traps

PIL respects EXIF orientation by default; cv2 does NOT. If a file's actual
pixel orientation is different from its EXIF-implied orientation, the
loader's bucket assignment (PIL-based) won't match the file's pixel data
(cv2-based).

This produces the `unexpected values: w=H h=W scale_to_width=...` error
where w and h are swapped relative to what the bucket expects.

**Fix at dataset-prep time:** strip EXIF orientation:
```bash
exiftool -Orientation= /your/dataset/path/*.jpg
```

Or losslessly rotate so orientation is "1" (default):
```bash
jhead -autorot /your/dataset/path/*.jpg
```

## Latent cache

When `cache_latents_to_disk: true`:

```yaml
datasets:
  - folder_path: "/path"
    cache_latents_to_disk: true   # default for image datasets
```

The VAE-encoded latent for each image is computed once and stored to
disk inside the dataset folder. On subsequent runs, the cached latent is
loaded directly. **The cache key is the image path**, not the image
content — if you replace an image at the same path, the cache will return
stale latents until you delete it.

Cache invalidates when:
- File path doesn't match a cache entry → recomputes
- `--regenerate_dataset_cache` flag is passed (or relevant config knob)

Cache does NOT invalidate when:
- File content changes but path doesn't
- Resolution buckets change
- num_frames changes (for video latent caches)

**To force re-cache:** delete the cache directory:
```bash
find /your/dataset/path -type d -name '*cache*' -exec rm -rf {} +
# or specifically:
find /your/dataset/path -type d -name '.dataset_cache' -exec rm -rf {} +
```

## Text embedding cache

`cache_text_embeddings: true` (typically required for video LoRAs to save
VRAM during the loop) caches the text encoder output for each caption.

Cache key is the caption *string content*. If you change a caption file,
the cache entry for that caption automatically becomes stale (different
content → different key → recompute). But if you change captions and the
new caption happens to match a previously-cached different caption (rare),
you'd hit the wrong cache entry.

**Pragmatic rule:** if you change captions significantly (recaptioning a
dataset), nuke the embedding cache:
```bash
find /your/dataset/path -type d -name '*embedding*' -exec rm -rf {} +
```

## Augments are image-only

```python
# toolkit/dataloader_mixins.py:461
if self.is_video:
    raise Exception('Augments not supported for videos')
```

`flip_x`, color jitter, etc. don't run on video datasets. Temporal
consistency would break (flipping frame 0 differently from frame 5).

For images, `flip_x: true` doubles your effective dataset size at no cost.

## What the loader does NOT do

These are common assumptions that are wrong:

- **Doesn't deep-recurse by default in some classes** — verify for your
  specific dataset config. If using nested folders, check the relevant
  class in `data_loader.py`.
- **Doesn't auto-detect video vs. image** — purely driven by `num_frames`.
- **Doesn't rescan mid-run** — files added to the dataset folder during
  training are not picked up. Restart required.
- **Doesn't validate caption content** — empty caption files, BOM-prefixed
  files, files with weird encoding all "load" but produce garbage at
  training time.
- **Doesn't normalize file extensions** — `.JPG` ≠ `.jpg`.
- **Doesn't fail loudly on missing captions** — silently treats clip as
  unconditional.
- **Doesn't deduplicate files within a dataset** — if a file is symlinked
  twice, it's two training items.
