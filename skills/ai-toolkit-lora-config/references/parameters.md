# Parameter Reference

What each parameter does and how to tune it.

## `network:` block

```yaml
network:
  type: "lora"
  linear: <rank>
  linear_alpha: <alpha, usually = rank>
```

**`linear` (rank)**: the LoRA's learning capacity. Higher rank = more parameters = more capacity to memorize, but also more risk of overfitting and larger file size.

- Character LoRA: `16` (enough for identity without memorizing dataset style)
- Style LoRA: `32` (style information is distributed across the model, needs more capacity)
- Complex subjects with distinctive visual details (specific art style + specific character): `32-64`

**`linear_alpha`**: scales the LoRA's effect. Default to match `linear`. Halving alpha (e.g., rank 32, alpha 16) reduces effective strength — useful if the LoRA is overfitting but you don't want to retrain.

**Never include `conv` or `conv_alpha`** for transformer-based models (all modern ones: Flux, Z-Image, Chroma, Qwen, HiDream). Only UNet models (SD 1.5, SDXL) have conv layers.

## `train:` block

```yaml
train:
  batch_size: <int>
  steps: <int>
  gradient_accumulation_steps: 1
  train_unet: true
  train_text_encoder: false
  gradient_checkpointing: <bool>
  noise_scheduler: "flowmatch"
  timestep_type: "weighted"   # or "sigmoid" or "shift"
  optimizer: "adamw8bit"
  lr: 1e-4
  diff_output_preservation: <bool>
  diff_output_preservation_class: "<class>"
  diff_output_preservation_multiplier: 1.0
  ema_config:
    use_ema: true
    ema_decay: 0.99
  dtype: bf16
```

**`batch_size`**: images per training step. Higher = more stable gradients per step BUT cycles dataset faster, which accelerates overfitting on small datasets.
- Dataset <40 images: 1-2
- Dataset 40-100 images: 2-4
- Dataset 100+: 4-8

**`steps`**: total training steps. See step count heuristics in SKILL.md. Always check interval saves to find the sweet spot — don't trust the final step.

**`gradient_checkpointing`**: trades compute for memory. Zero quality cost, ~25-30% slower per step. ALWAYS on for 24GB. Can disable on 80GB+ IF DOP is off.

**`timestep_type`**:
- `"weighted"` — recommended default for most modern models (Flux.2, Klein, Z-Image, Qwen)
- `"sigmoid"` — good for subject LoRAs on Flex2 or OmniGen2
- `"shift"` — HiDream and some fast-convergence scenarios

**`optimizer`**: `"adamw8bit"` is the solid default. Prodigy is also supported but rarely needed. Never tune without explicit reason.

**`lr`**: `1e-4` works for nearly all LoRA training. Higher LR (2e-4 to 5e-4) is sometimes used for quick iteration but increases instability. Lower LR (5e-5) for very fine-grained adjustments.

**`diff_output_preservation`**: regularization. See DOP section in SKILL.md. Nearly doubles memory — pair with gradient_checkpointing.

**`ema_config.use_ema: true`**: exponential moving average of LoRA weights. Smooths training, improves generalization. Small speed cost. Keep on.

**`dtype: bf16`**: required for most modern models on modern GPUs (A100+, H100). Do not change.

## `model:` block

```yaml
model:
  name_or_path: "<hf-path>"
  arch: "<model-arch>"
  quantize: <bool>
  quantize_te: <bool>
  qtype: "qfloat8"
  low_vram: <bool>
  assistant_lora_path: "<path-if-turbo>"
```

**`quantize`** / **`quantize_te`**: 8-bit quantization. Small quality cost, big memory savings. Required for 24GB training. Can be false on 80GB+ but no harm leaving on.

**`qtype`**: `"qfloat8"` is the standard quantization type. Leave as-is.

**`low_vram`**: quantizes on CPU instead of GPU. Use if the GPU is also driving your monitors.

**`assistant_lora_path`**: REQUIRED for distilled/turbo models (Z-Image Turbo, Flux Schnell). Without it, these models can't be trained with LoRA. See `models.md`.

## `datasets:` block

Each dataset folder is a separate list entry. You can have multiple.

```yaml
datasets:
  - folder_path: "/path/to/images"
    caption_ext: "txt"
    caption_dropout_rate: 0.05
    shuffle_tokens: false
    cache_latents_to_disk: true
    num_repeats: 1
    resolution: [ 512, 768, 1024 ]
```

**`caption_dropout_rate`**: probability of dropping the caption (replacing with empty string) per training step. Forces the model to learn from pixels even without text cues.
- Character: `0.05` (5%)
- Style: `0.1` (10%, style needs stronger visual-only signal)

**`cache_latents_to_disk: true`**: caches encoded latents to avoid re-encoding every epoch. Always on unless you have a reason not to.

**`num_repeats`**: how many times this dataset is repeated per epoch. Use to balance multiple datasets of different sizes.

Example: 80 identity images + 36 style images. Set `num_repeats: 2` on the style set (36*2=72 ≈ 80). Now both sets are seen with roughly equal frequency.

**`resolution`**: bucketing resolutions. Images get resized and bucketed to match. More resolutions = more learning but more memory. `[512, 768, 1024]` is the standard set.

## `sample:` block

See `sample-prompts.md` for prompt design. Key parameters:

```yaml
sample:
  sampler: "flowmatch"   # must match train.noise_scheduler
  sample_every: 250      # save interval
  guidance_scale: <int>
  sample_steps: <int>
```

**`guidance_scale`** + **`sample_steps`**:
- Normal models (Flux.2, Klein, Z-Image Base, Chroma): `guidance_scale: 4`, `sample_steps: 20-25`
- Turbo/distilled models (Z-Image Turbo, Flux Schnell): `guidance_scale: 1`, `sample_steps: 8`

`walk_seed: true` varies the seed per prompt — more useful visual variety during training. `seed: 42` is fine as the base.

## `save:` block

```yaml
save:
  dtype: float16
  save_every: 250
  max_step_saves_to_keep: 4
```

`save_every: 250` gives you multiple checkpoints to evaluate at different training points. `max_step_saves_to_keep: 4` keeps the last 4 saves to limit disk use.

## `logging:` block — ALWAYS INCLUDE

```yaml
logging:
  log_every: 1
  use_ui_logger: true
```

Default behavior (no `logging:` block) is `EmptyLogger` — nothing is saved to disk. The loss curve and learning rate trajectory are LOST and there's no way to recover them after the run. Always include the block above unless the user explicitly opts out.

**What it writes:** `output/<run-name>/loss_log.db`, a SQLite file containing per-step `learning_rate` and per-loss-key values (e.g. `loss/loss`). The ai-toolkit web UI reads this for graphs; you can also query it directly:

```bash
sqlite3 output/<run>/loss_log.db ".schema"
sqlite3 output/<run>/loss_log.db "SELECT step, key, value_real FROM metrics WHERE key LIKE 'loss/%' ORDER BY step"
```

**Why per-step diagnostics matter:** without them, the only signal during/after training is the sample grid, which is downsampled to whatever `sample_every` you set. The loss/LR log lets you see plateau points, overfitting onsets, and LR schedule behavior between sample checkpoints — critical for picking the right step in [[ai-toolkit-sample-reviewer]] reviews and for diagnosing why a v(n+1) run looked different from a v(n) run.

**`log_every: 1`** logs every step. The UI logger buffers writes and flushes asynchronously, so the I/O cost is negligible even at log-every-step granularity.

### Alternative: Weights & Biases

```yaml
logging:
  log_every: 1
  use_wandb: true
  project_name: "<wandb-project>"
  run_name: "<descriptive-run-name>"
```

Use this instead of `use_ui_logger` only when the user explicitly wants cloud visibility AND `WANDB_API_KEY` is in their environment. Don't enable both loggers in the same run — pick one. W&B also captures the full config snapshot, which is useful for comparing runs in the dashboard.

### Other knobs (rarely tuned)

- `log_every` — default 100. Setting to 1 logs every step (recommended for short runs); larger values (10, 50) reduce log volume for long runs.
- `use_wandb` / `use_ui_logger` — mutually exclusive in practice; pick one.
- `project_name` / `run_name` — only used when `use_wandb: true`. Default reasonable values.
