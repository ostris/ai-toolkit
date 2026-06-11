---
name: ai-toolkit-dataset-diagnostics
description: >
  Diagnose AI Toolkit dataset-loading and pre-training failures fast. Use
  when: (1) error message says "no images found in", "no videos found in",
  "buckets not found on dataset", "Failed to open video file", "unexpected
  values: w=...", or any "config file error. Missing config.dataset.X" —
  these are AI Toolkit data loader errors, (2) training fails before step 0
  / dies at first batch / hangs at "Found 0 images", (3) captioner stalls
  at "0/N" with no progress, (4) training appears to run but the trigger
  never fires (likely stale latent cache), (5) user says "dataset isn't
  loading", "training crashed at startup", "diagnose my dataset".
  Maps each observable symptom to the actual data_loader.py / dataloader_mixins.py
  code path, then to the fix. Encodes the recurring traps: num_frames > 1
  toggles video mode, AppleDouble pollution on Mac drives, caption_ext
  mismatches, latent cache invalidation rules, and the (n-1) % 4 == 0
  Wan22 stride rule.
---

# AI Toolkit dataset diagnostics

A focused diagnostic skill. The user describes a symptom — error message,
crash signature, or a "feels off" observation — and this skill maps it to
the cause and fix.

Most AI Toolkit pre-training failures fall into 5 categories:

1. **Wrong dataset type** — file extensions don't match what `num_frames`
   expects.
2. **Path / pollution issues** — wrong path, AppleDouble files, missing
   captions.
3. **Config-shape errors** — bucket misconfiguration, caption_ext
   mismatch, model arch mismatch.
4. **Stale state** — latent cache, text embedding cache, or checkpoint
   resume.
5. **Pipeline stalls** — captioner hangs, training stuck in dataloader.

Read the symptom-to-cause table first, then drill into the relevant
reference doc.

## Quick lookup: error message → cause

| Error / symptom | Cause | Section |
|----------------|-------|---------|
| `no images found in <path>` | num_frames=1 but path has videos OR path has dotfiles only OR wrong extensions | [§1](#1-no-images-found-in-path) |
| `no videos found in <path>` | num_frames>1 but path has images OR path empty after dotfile filter | [§2](#2-no-videos-found-in-path) |
| `Found 0 images` (then crashes) | Same as §1 | [§1](#1-no-images-found-in-path) |
| `Failed to open video file: <path>` | Corrupt video, unsupported codec, or zero-byte file | [§3](#3-failed-to-open-video-file) |
| `Failed to read frame N from video` | Codec issue or video shorter than N frames | [§3](#3-failed-to-open-video-file) |
| `buckets not found on dataset <path>, you either need all buckets or none` | Mixed bucket config across datasets in the same job | [§4](#4-buckets-not-found-on-dataset) |
| `unexpected values: w=H h=W scale_to_width=...` | Bucket aspect ratio mismatch — image is W×H but bucket expects H×W | [§5](#5-unexpected-values-aspect-mismatch) |
| `Buckets required for video processing` | Trying to train video without the bucket config | [§6](#6-buckets-required-for-video-processing) |
| `Augments not supported for videos` | flip_x/augments enabled in video dataset config | [§7](#7-augments-not-supported-for-videos) |
| `config file error. Missing "config.dataset.X" key` | Required key missing from YAML — usually `folder_path` or `caption_ext` | [§8](#8-config-file-error-missing-key) |
| Training fails at first batch with shape mismatch | (n-1) % 4 == 0 violation on Wan22, OR `num_frames` mismatch with cache | [§9](#9-shape-mismatch-at-first-batch) |
| OOM at training startup, video LoRA | `gradient_checkpointing: false` (it's required, not optional) | [§10](#10-oom-at-startup-video) |
| Captioner stalls at "0/N" with no progress | Slow IO, corrupt files, or Gemini network hang | [§11](#11-captioner-stalls-at-0n) |
| Training runs but trigger never fires at inference | Captions weren't loaded (caption_ext mismatch) OR stale latent cache OR captions describe what should be the trigger | [§12](#12-trigger-never-fires-at-inference) |
| Resume from checkpoint produces wrong style | LoRA arch mismatch with the resumed weights | [§13](#13-checkpoint-resume-wrong-style) |

## Diagnostic preflight

If you're about to start training and want to catch issues *before* the
real run, run the preflight sequence in
`references/preflight-checklist.md`. Five commands, ~2 minutes, catches
~80% of dataset-loading failures.

---

## §1 — `no images found in <path>`

**Code path**: `toolkit/data_loader.py:541` (or `:119` for the legacy
SDDataset path). Fires when `is_video=False` (i.e. `num_frames <= 1`)
and `file_list` is empty after scanning.

**Possible causes:**

### Cause A: dataset is videos, but `num_frames` is unset (defaults to 1)

The data loader switches to video mode when `dataset_config.num_frames > 1`.
If your dataset folder contains `.mp4`/`.mov` files but `num_frames` is
unset, the loader looks for *images* and finds 0.

**Diagnose:**
```bash
# What's in the path?
ls /your/dataset/path | head
file /your/dataset/path/* | head -5
```

If the files are videos, fix the YAML:
```yaml
datasets:
  - folder_path: "/your/dataset/path"
    num_frames: 81   # or other (n-1)%4==0 value
```

### Cause B: AppleDouble pollution

External Mac drives create `._<filename>` resource forks. The data loader
filters dotfiles (`startswith('.')`), so the 30-image dataset shows up as
30 dotfiles + 30 actual images, but the loader skips both because the
dotfiles share the `.jpg`/`.png` extension and look real to some scanners.
Then if all "real" images turned out to be dotfiles (e.g., the actual
images live one folder deeper), you get 0.

**Diagnose:**
```bash
ls -la /your/dataset/path/ | grep '^-' | head
ls -la /your/dataset/path/ | grep '\._' | wc -l
```

If you see `._*` files, they may be the only `.jpg`-extensioned files
(if the real images are in a subfolder). Clean them up:
```bash
find /your/dataset/path/ -name '._*' -delete
find /your/dataset/path/ -name '.DS_Store' -delete
```

If your real images are in a subfolder, point `folder_path` at the
subfolder, NOT the parent.

### Cause C: wrong extension

The data loader's `IMAGE_EXTS` is `{".jpg", ".jpeg", ".png", ".webp"}`.
Files with `.JPG` (uppercase), `.tiff`, `.heic`, `.bmp` are silently
skipped.

**Diagnose:**
```bash
ls /your/dataset/path | sed -E 's/.*\.//' | sort -u
```

If you see uppercase or unsupported extensions, rename:
```bash
# Lowercase all
for f in /your/dataset/path/*.JPG; do mv "$f" "${f%.JPG}.jpg"; done

# Convert HEIC/TIFF to JPG (one-time, requires imagemagick)
for f in /your/dataset/path/*.heic; do
  magick "$f" "${f%.heic}.jpg"
done
```

### Cause D: path doesn't exist or is wrong

**Diagnose:**
```bash
test -d /your/dataset/path && echo "exists" || echo "MISSING"
```

Typo or unmounted external drive. Re-mount or fix the path.

---

## §2 — `no videos found in <path>`

**Code path**: `toolkit/data_loader.py:538`. Fires when `is_video=True`
(`num_frames > 1`) but no videos found.

### Cause A: dataset is images, but `num_frames > 1`

The inverse of §1. If your dataset is JPEGs but you've set `num_frames: 81`,
the loader looks for videos and finds 0.

**Fix:** either remove `num_frames` (defaults to 1, image mode) or move
to a video dataset.

### Cause B: AppleDouble pollution + only one or two real videos

Same as §1's Cause B but for videos.

### Cause C: video extension mismatch

The data loader's `VIDEO_EXTS` is `{".mp4", ".mov", ".mkv", ".webm",
".avi", ".m4v"}`. `.MOV` (uppercase), `.flv`, `.gif` won't match.

```bash
ls /your/dataset/path | sed -E 's/.*\.//' | sort -u
```

Rename to lowercase if needed.

---

## §3 — `Failed to open video file` / `Failed to read frame N`

**Code path**: `toolkit/dataloader_mixins.py:476` and `:564`. Fires when
cv2 (or the toolkit's video reader) can't decode a specific clip.

**Cause:** zero-byte file, corrupt file, codec the toolkit doesn't support,
or video shorter than the requested frame index.

**Diagnose:**
```bash
python scripts/diagnose_video_read.py --dataset /your/dataset/path/
```

Reports each video's open + first-frame-read status. Failed files are
listed at the end with codes (CANNOT OPEN / NO FRAME / EXCEPTION).

**Fix:** delete or re-encode the failed files:
```bash
# Re-encode to known-good format
ffmpeg -i broken.mp4 -c:v libx264 -crf 18 -c:a aac -y fixed.mp4
mv fixed.mp4 broken.mp4
```

If many fail, check the source — may need re-export from the original NLE.

---

## §4 — `buckets not found on dataset <path>, you either need all buckets or none`

**Code path**: `toolkit/data_loader.py:680`. Fires when a job has multiple
datasets and some have bucket config and others don't.

**Cause:** bucket config (the `resolution: [...]` array) must be either
defined on all datasets in the job or none of them.

**Fix:** either add `resolution: [...]` to the dataset that's missing it,
or remove it from the others. Be consistent across all datasets in the
same `process[*].datasets[]` list.

```yaml
datasets:
  - folder_path: "/path/a"
    resolution: [512, 768, 1024]  # has buckets
  - folder_path: "/path/b"
    resolution: [512, 768, 1024]  # must also have buckets
```

---

## §5 — `unexpected values: w=H h=W scale_to_width=...` (aspect mismatch)

**Code path**: `toolkit/dataloader_mixins.py:881`. Fires when a file's
actual aspect ratio doesn't match what its bucket assignment expects.

**Cause:** the data loader assigns each file to a bucket based on
aspect ratio. If the file was rotated by EXIF metadata that PIL respects
but cv2 doesn't (or vice versa), the actual w/h after loading doesn't
match the metadata-based assignment.

**Diagnose:** the error message includes the file path. Inspect the file:
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,side_data_list \
  -of default=nw=1 "/path/from/error.jpg"
exiftool "/path/from/error.jpg" | grep -i orient
```

If EXIF orientation is non-1, the file is being rotated on load.

**Fix:** strip EXIF orientation OR rotate the file in place:
```bash
# Strip EXIF orientation (use the actual pixel orientation)
exiftool -Orientation= "/path/from/error.jpg"

# Or losslessly rotate to match EXIF
jpegtran -copy none -rotate 90 -outfile rotated.jpg "/path/from/error.jpg"
```

For a whole dataset:
```bash
exiftool -Orientation= /your/dataset/path/*.jpg
```

---

## §6 — `Buckets required for video processing`

**Code path**: `toolkit/dataloader_mixins.py:467`. Fires when training
video without `resolution: [...]` set.

**Fix:** add a `resolution` array. For Wan22, `[512, 720]` is the default
16:9 widescreen pair.

```yaml
datasets:
  - folder_path: "/path"
    num_frames: 81
    resolution: [512, 720]
```

---

## §7 — `Augments not supported for videos`

**Code path**: `toolkit/dataloader_mixins.py:461` / `:464`. Fires when
flip_x or color augments are enabled on a video dataset.

**Fix:** remove augment config from video datasets:
```yaml
datasets:
  - folder_path: "/path"
    num_frames: 81
    flip_x: false   # or just omit
```

Video LoRAs don't get augments — temporal consistency would break.

---

## §8 — `config file error. Missing "config.dataset.X" key`

**Code path**: `toolkit/data_loader.py:131` and `:286`. Fires when a
required key is missing from the dataset config.

**Fix:** add the missing key. Required keys vary slightly by dataset
type but typically include `folder_path` and `caption_ext`. If the error
mentions a different key, add it with the appropriate default — see the
existing `config/examples/` configs for canonical structure.

---

## §9 — Shape mismatch at first batch

Training starts (loader finishes), then crashes at first batch with a
torch shape mismatch.

### Cause A: Wan22 (n-1) % 4 == 0 violation

`num_frames` must satisfy `(num_frames - 1) % 4 == 0` for Wan 2.2.

**Fix:** snap to nearest valid value:
- 80 → 81 (most common)
- 60 → 61 or 57
- 100 → 101 or 97

### Cause B: latent cache mismatch

If you previously trained at `num_frames: 60` and cached latents to disk,
then changed to `num_frames: 81`, the cached latents have the wrong
temporal shape.

**Fix:** delete the latent cache:
```bash
# Look in the dataset folder for a `.latent_cache` or `.dataset_cache` dir
find /your/dataset/path -name '*.latent*' -o -name '*cache*' 2>/dev/null

# Or in the training output folder
ls /your/repo/output/<job_name>/
```

Delete cache files, restart training. The loader will re-cache.

### Cause C: changed resolution buckets after caching

Same as Cause B but for image dimensions. Resolution change → cached
latents have wrong spatial shape. Same fix.

---

## §10 — OOM at startup, video LoRA

Training video LoRA on 80GB+ VRAM and getting OOM on the first batch.

### Cause: `gradient_checkpointing: false`

For Wan22 video at 81 frames, gradient checkpointing is **required**, not
optional. Even at 95GB. Activation memory exceeds available VRAM without
it.

**Fix:**
```yaml
train:
  gradient_checkpointing: true
```

### Other contributors

- `cache_text_embeddings: false` keeps the text encoder in VRAM during
  the loop. Set true for video.
- Quantize the TE: `quantize_te: true`, `qtype_te: "qfloat8"`.
- On <80GB, also `quantize: true` for the transformer.

---

## §11 — Captioner stalls at "0/N"

The Gemini captioner shows "Found N videos, N need captions" then no
tqdm progress.

**Diagnose:**
```bash
# Step 1: is video read fast?
python scripts/diagnose_video_read.py --dataset /your/dataset/path

# Step 2: is the full pipeline fast on ONE video?
python scripts/diagnose_caption_pipeline.py --dataset /your/dataset/path
```

The first script flags slow IO (>2s/file) and corrupt files. The second
runs ONE complete cycle (cv2 read → PIL convert → Gemini text call →
Gemini vision call) with per-step timing.

**Common causes:**

| Diagnostic result | Cause | Fix |
|------------------|-------|-----|
| `diagnose_video_read` shows many slow reads | External-drive IO bottleneck | Copy dataset to local SSD |
| `diagnose_video_read` shows CANNOT OPEN files | Corrupt videos hanging cv2 | Delete/re-encode failed files |
| `diagnose_caption_pipeline` is fast end-to-end | Threading concurrency in real captioner | Drop `--workers` from 4 to 2 or 1 |
| `diagnose_caption_pipeline` hangs at "Gemini text-only call" | API auth or network | Check `GEMINI_API_KEY`; try a different model |
| `diagnose_caption_pipeline` text call OK, vision call hangs | Image too large or content policy | Try a smaller/cleaner test image |

---

## §12 — Trigger never fires at inference

Training appeared to complete normally, but at inference the trigger word
produces base-model output. The LoRA seems to have learned nothing.

### Cause A: caption files not loaded (caption_ext mismatch)

The data loader looks for `<image>.<caption_ext>` next to each image.
Default is `txt`. If your captions are `.captions` or `.json`, the loader
silently treats clips as unconditional and the trigger never appears in
training.

**Diagnose:**
```bash
# What's the caption_ext in YAML?
grep caption_ext /path/to/your_config.yaml

# What's actually on disk?
ls /your/dataset/path | sed -E 's/.*\.//' | sort | uniq -c
```

If they don't match, fix the YAML or rename the files.

### Cause B: stale latent cache from a different LoRA / different run

You changed the trigger word but the latent cache contains pre-encoded
latents from a previous run. The loader uses the cache, so the new
trigger never makes it to the model.

**Note:** latent cache is for *image* latents; text embeddings are cached
separately. Caption changes invalidate the text embedding cache, not the
latent cache. But if you changed `caption_dropout_rate` or the captions
themselves between runs, and `cache_text_embeddings: true`, you may need
to nuke that cache.

**Fix:**
```bash
# Find caches in the dataset folder
find /your/dataset/path -name '*cache*' -type d

# Find caches in the output folder
ls /your/repo/output/<job_name>/

# Delete and retrain
rm -rf /path/to/cache_dirs
```

### Cause C: captions described what should be the trigger

If captions describe the style/character/motion the LoRA was supposed to
learn, those concepts are *promptable* now, not trigger-bound. The LoRA
trained correctly but the trigger has no unique role.

**Diagnose:**
```bash
# For style LoRAs — grep for the style traits
grep -iE 'sepia|iridescent|chemigram|<your-style-words>' \
  /your/dataset/path/*.txt | wc -l

# For character LoRAs — grep for identity descriptors
grep -iE 'woman|man|young|brown hair|blue eyes' \
  /your/dataset/path/*.txt | wc -l

# For motion LoRAs — grep for motion verbs
grep -iE 'morph|melt|transform|gradually|over time' \
  /your/dataset/path/*.txt | wc -l
```

If hits are non-zero, those concepts leaked into captions. **Fix:**
recaption with the appropriate `ai-toolkit-gemini-captioner` mode (whose
avoid lists exist precisely to prevent this). Then retrain.

---

## §13 — Checkpoint resume produces wrong style

User resumes from a checkpoint and the training continues but produces
nothing like what the original LoRA was learning.

### Cause: LoRA architecture mismatch

If the YAML's `network.linear` (rank), `network.linear_alpha` (alpha),
or model `arch` differs from what the checkpoint was trained with, the
weights load partially (or fail silently) and training drifts.

**Diagnose:**
- Was the YAML edited between the original run and the resume?
- What's `network.linear` in the YAML? Does it match the checkpoint
  filename / metadata?

**Fix:** match the YAML to the original checkpoint config exactly. If
you intended to resume into a different config, you need to retrain
from scratch with the new config.

---

## How to use the existing diagnostic scripts

Two scripts in the repo cover most diagnostic work:

| Script | What it tests | When to run |
|--------|---------------|-------------|
| `scripts/diagnose_video_read.py` | cv2 open + first-frame read on every file in dataset | Before any video captioning run; or when "no videos found" / corrupt-file errors |
| `scripts/diagnose_caption_pipeline.py` | One complete caption cycle (cv2 → PIL → Gemini) with per-step timing | When captioner is stalling at 0/N |

See `references/data-loader-internals.md` for the data loader's actual
behavior (when video mode toggles, what's cached where, what gets
filtered).

See `references/preflight-checklist.md` for the 5-command sanity sweep
to run before every training start.

## Common mistakes

- **Assuming the loader does deep recursion**: many dataset configs only
  scan immediate children, not nested folders. If your data is nested,
  flatten it or check the specific dataset class.
- **Editing YAML mid-run thinking it'll re-load**: AI Toolkit reads the
  YAML once at startup. Edits only apply on next run.
- **Adding files to a dataset folder mid-run**: not picked up. Stop and
  restart.
- **Trusting `caption_ext: "txt"` matches `.TXT`**: it's case-sensitive.
- **Skipping diagnose_video_read.py before captioning**: 30 minutes of
  captioner stalling later, you'll wish you'd run it.
- **Modifying captions but not nuking the text embedding cache**: model
  trains on stale embeddings.
- **Resuming from a checkpoint trained with different rank**: weights
  load partially or fail; training drifts.

## Reference files

- `references/data-loader-internals.md` — how the AI Toolkit data loader
  actually works: when video mode triggers, what gets filtered, where
  the bucket logic lives, how captions are matched
- `references/preflight-checklist.md` — 5-command sanity sequence to
  run before every training start
- `references/cache-and-state.md` — latent cache, text embedding cache,
  resume invariants, and how to invalidate each cleanly
