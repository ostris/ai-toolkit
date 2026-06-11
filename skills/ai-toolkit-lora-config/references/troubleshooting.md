# Troubleshooting

Common failure modes and what causes them.

## "My LoRA isn't learning" / "Samples at step N look the same as baseline"

Ranked by likelihood:

### 1. Assisted prompts in sample prompts
By far the most common cause. If your sample prompt already describes the style ("rendered as an oil pastel illustration on white paper..."), the base model is doing that work from the prompt alone. Baseline and trained samples look identical because neither needs the LoRA.

**Fix**: Strip sample prompts down to bare content + trigger word. Let the LoRA carry the style.

### 2. DOP multiplier too high
DOP at `1.0` on a character LoRA with class `"person"` pulls the LoRA output back toward base-model "person" rendering. The LoRA can be learning fine but its effect gets neutralized.

**Fix**: Drop `diff_output_preservation_multiplier` to `0.3-0.5`, or disable DOP entirely to confirm it's the cause.

### 3. Trigger word not in captions
If captions don't contain the trigger word, the LoRA has no anchor point — the trigger word in sample prompts does nothing.

**Fix**: Open a random `.txt` caption file. Confirm it starts with (or contains) the trigger word. If not, re-run captioning.

### 4. Training loss is flat
Check the terminal output — if loss isn't descending from step 0, training itself is broken (bad data, bad LR, OOM-skipped steps).

**Fix**: Check that dataset images are valid, captions exist, and training loss is actually going down.

### 5. Looking at wrong checkpoint
The toolkit saves at intervals. Make sure you're looking at the samples from the latest interval, not cached samples from an earlier run.

## "Style isn't showing up in triggered prompts"

- **Captions included the style phrase but prompts don't** — check the exact phrase in captions matches the inference prompt exactly. Small differences ("oil pastel illustration" vs "oil pastel drawing") prevent activation.
- **Rank too low** — style LoRAs benefit from rank 32. If on 16, bump it and retrain.
- **Too few steps** — style takes longer than character. 2500+ steps for style LoRAs.
- **Style descriptor is too specific** — "mid-2010s Tumblr aesthetic smartphone self-portrait" may not have strong priors in the base model. Try something simpler like "2010s smartphone photograph".

## "Character has dataset style baked in, doesn't generalize"

- Captions describe too much identity, not enough variability — the LoRA learned "always wearing this specific outfit" instead of a generic person. Redo captions with more of what varies, less of what's fixed.
- Dataset not diverse enough. Need images across different lighting, settings, poses, angles.
- DOP is off — add `diff_output_preservation: true` with class `"person"`.
- Training went too long. Try an earlier checkpoint from the interval saves.

## OOM during training

Peak memory hits during forward+backward with DOP on.

**Fix, in order of least-to-most intrusive:**

1. `gradient_checkpointing: true` (zero quality cost, ~30% slower per step)
2. Enable `quantize: true` and `quantize_te: true`
3. Add `unload_text_encoder: true` in `train:` (unloads TE after encoding captions)
4. Drop `batch_size` to 1
5. Drop largest resolution from bucket list
6. As a last resort, disable DOP

Gradient checkpointing alone usually drops peak memory by 60-70% — try it first before anything else.

## "OOM during training step, skipping batch X/3"

The toolkit retries each batch up to 3 times before giving up. Intermittent OOM means you're at the edge of memory — usually gradient checkpointing fixes this. Continuous OOM means you need more aggressive memory savings (quantize, etc.).

## Training super slow

- **Too many resolutions** — each resolution requires its own bucket. `[512, 768, 1024]` is standard. Dropping to fewer buckets speeds things up.
- **Text encoder on GPU and not quantized** — Qwen3-8B or Mistral-24B are huge. Set `quantize_te: true` or `unload_text_encoder: true`.
- **Gradient checkpointing on with plenty of VRAM** — if you have 60GB+ free and DOP is off, turning off grad checkpointing will noticeably speed up training.

## Samples are corrupt / weird colors / noise

- **Wrong `sample_steps` for model type** — turbo models need 8 steps, not 20-25.
- **Wrong `guidance_scale`** — turbo models use 1, normal models use 4.
- **Wrong `sampler`** — must match `noise_scheduler`. `"flowmatch"` for all modern models.
- **`dtype: bf16`** for models — not fp16 or fp32 unless you have a specific reason.

## "Unload text encoder caused errors"

`unload_text_encoder: true` moves the TE off GPU between steps. Not all models handle this gracefully. If training errors after adding it, remove it and use quantization instead.

## Colab-specific issues

- **"Disconnected" mid-training** — Colab sessions time out. For long runs (2500+ steps), use Colab Pro+ and keep a browser tab active on the notebook.
- **HF_TOKEN errors** — for gated models (Flux.1-dev, Flux.2-dev, Flux Kontext), you must accept the license on HF first, then set `HF_TOKEN` via Colab secrets.
- **Out of disk** — models are 10-30GB. Make sure `/content/drive/MyDrive` has space, or use `/content/` (ephemeral but larger).

## "All my samples look like the training images verbatim"

Overtraining. The LoRA has memorized the dataset rather than generalizing.

- Use an earlier checkpoint (500-1000 steps less).
- Reduce rank next time.
- Increase caption dropout.
- Increase DOP multiplier.
- Add more diverse images to the dataset.

## When to ask the user for the training output

If the user reports a symptom but you're not sure which of the above is the cause, ask them to share:

1. A few lines of terminal output showing training loss over the last 100 steps
2. One sample caption file (random `.txt` from the dataset)
3. One sample image from the latest checkpoint + the prompt that generated it
4. The VRAM utilization (nvidia-smi or Colab GPU RAM graph)

Those four data points diagnose 90% of training failures.
