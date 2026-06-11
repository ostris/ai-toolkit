# Preflight checklist — run before every training start

5 commands, ~2 minutes. Catches ~80% of dataset-loading failures before
they cost you a training-job startup cycle.

If anything in this checklist fails or surfaces unexpected output, stop
and fix it before launching training.

## Variables

```bash
DATASET=/path/to/your/dataset
CAPTION_EXT=txt   # match your YAML's caption_ext
```

## 1. Path exists and contains the right type of files

```bash
test -d "$DATASET" && echo "✅ exists" || echo "❌ MISSING"

# What's actually there?
ls "$DATASET" | sed -E 's/.*\.//' | sort | uniq -c
```

Expected: extensions match what you intend to train on.
- Image LoRA → `.jpg / .jpeg / .png / .webp`
- Video LoRA → `.mp4 / .mov / .mkv / .webm / .m4v`

If you see `.JPG` (uppercase), `.heic`, `.tiff`, or `.MOV` — fix before
proceeding. The loader is case-sensitive.

## 2. No AppleDouble pollution

```bash
APPLE_DOUBLE=$(find "$DATASET" -name '._*' | wc -l)
if [ "$APPLE_DOUBLE" -gt 0 ]; then
    echo "⚠️  $APPLE_DOUBLE AppleDouble files found"
    echo "   Delete with: find $DATASET -name '._*' -delete"
fi
```

External Mac drives create `._*` resource forks. The loader skips dotfiles
correctly, but they pollute the file scan and can confuse other tools
(captioners, ffprobe sweeps).

## 3. Every clip has a caption file

```bash
# Count files
N_CLIPS=$(ls "$DATASET" | grep -iE '\.(mp4|mov|mkv|webm|m4v|jpg|jpeg|png|webp)$' | wc -l)
N_CAPS=$(ls "$DATASET" | grep -i "\.${CAPTION_EXT}$" | wc -l)

echo "Clips: $N_CLIPS, Captions: $N_CAPS"

if [ "$N_CLIPS" != "$N_CAPS" ]; then
    echo "⚠️  MISMATCH"
    # List clips missing captions
    for clip in "$DATASET"/*.{mp4,mov,jpg,png,webp,jpeg,mkv,webm,m4v} 2>/dev/null; do
        [ -f "$clip" ] || continue
        cap="${clip%.*}.$CAPTION_EXT"
        [ -f "$cap" ] || echo "  MISSING: $cap"
    done
fi
```

A missing caption file silently makes the clip unconditional at training
time. Trigger never fires. This is the silent killer.

## 4. Caption content sanity

```bash
# Are any captions empty?
EMPTY=$(find "$DATASET" -name "*.${CAPTION_EXT}" -size 0 | wc -l)
echo "Empty captions: $EMPTY"

# Sample a few captions to eyeball
echo "--- Random caption samples ---"
ls "$DATASET"/*."$CAPTION_EXT" 2>/dev/null | shuf -n 3 | while read f; do
    echo "── $f"
    cat "$f"
    echo
done
```

Look for:
- Empty caption files (size 0)
- Captions ending with the wrong trigger word (mismatch with YAML)
- Captions describing what should be the trigger (style words for style
  LoRAs, identity for character LoRAs, motion verbs for motion LoRAs)
- BOM characters or unexpected encoding (caption starts with `﻿`)

## 5. Video dataset readability (video LoRA only)

```bash
python scripts/diagnose_video_read.py --dataset "$DATASET"
```

Reports each video's open + first-frame-read status with timing. Flags:
- Slow reads (>2s) → external-drive IO bottleneck → copy to local SSD
- CANNOT OPEN / NO FRAME → corrupt files → re-encode or delete

Exit code 2 if any files failed. **Don't start captioning until this
returns exit 0.**

## Bonus: caption pipeline end-to-end test

For video LoRAs, also run:

```bash
python scripts/diagnose_caption_pipeline.py --dataset "$DATASET"
```

One full caption cycle on the first video, with per-step timing. If it
completes in <30 seconds, the captioner shouldn't stall at "0/N" when you
run it for real. If it hangs, the diagnostic tells you which step (cv2
read, Gemini text, Gemini vision).

## Bonus: trigger-leak grep

If this is a recaption (you've trained before and weren't happy):

```bash
# Style LoRA — what shouldn't appear in captions?
grep -iE 'iridescent|sepia|chemigram|<your-style-words>' \
  "$DATASET"/*.txt | head

# Character LoRA
grep -iE 'woman|man|young|brown hair|blue eyes|<character-traits>' \
  "$DATASET"/*.txt | head

# Motion LoRA
grep -iE 'morph|melt|transform|gradually|over time|about to' \
  "$DATASET"/*.txt | head
```

Any hits = those concepts are leaking into captions and won't bind to
the trigger. Recaption with the appropriate
`ai-toolkit-gemini-captioner` mode before training.

## YAML sanity

Read your YAML once before launching:

```bash
grep -E 'folder_path|caption_ext|num_frames|trigger_word|resolution|caption_dropout_rate|batch_size|steps|lr' \
  /path/to/your_config.yaml
```

Check:
- `folder_path` matches the path you've been auditing
- `caption_ext` matches the captions on disk
- `num_frames` is set to a value satisfying `(n-1)%4==0` for Wan22
- `trigger_word` matches the trigger word at the end of caption files
- `resolution` is appropriate for source dimensions
- `caption_dropout_rate: 0.0` for focused style/character/combined LoRAs
  (don't trust the default 0.05)
- `batch_size` is right for dataset size (≤2 for <60 images)

## All-in-one sweep

Save this as `~/preflight.sh`:

```bash
#!/usr/bin/env bash
set -e
DATASET="${1:?usage: preflight.sh <dataset-path> [caption-ext]}"
CAPTION_EXT="${2:-txt}"

echo "=== Preflight: $DATASET ==="

test -d "$DATASET" || { echo "❌ MISSING PATH"; exit 1; }

echo
echo "─── Extensions in dataset ───"
ls "$DATASET" | sed -E 's/.*\.//' | sort | uniq -c

echo
APPLE_DOUBLE=$(find "$DATASET" -name '._*' | wc -l | tr -d ' ')
echo "AppleDouble files: $APPLE_DOUBLE"
[ "$APPLE_DOUBLE" -gt 0 ] && echo "  → find $DATASET -name '._*' -delete"

echo
N_CLIPS=$(ls "$DATASET" | grep -iE '\.(mp4|mov|mkv|webm|m4v|jpg|jpeg|png|webp)$' | wc -l | tr -d ' ')
N_CAPS=$(ls "$DATASET" | grep -i "\.${CAPTION_EXT}$" | wc -l | tr -d ' ')
echo "Clips: $N_CLIPS  Captions ($CAPTION_EXT): $N_CAPS"
[ "$N_CLIPS" != "$N_CAPS" ] && echo "  ⚠️  MISMATCH"

echo
EMPTY=$(find "$DATASET" -name "*.${CAPTION_EXT}" -size 0 | wc -l | tr -d ' ')
echo "Empty captions: $EMPTY"

echo
echo "─── Sample captions ───"
ls "$DATASET"/*."$CAPTION_EXT" 2>/dev/null | shuf -n 3 | while read f; do
    echo "── $(basename $f)"
    cat "$f"
    echo
done

echo "=== Preflight complete ==="
```

Then before every training run:
```bash
bash ~/preflight.sh /your/dataset/path
```
