# Understanding Your Training Metrics

Simple guide to what the numbers mean and what you can actually control.

## Metrics You Can Read in `metrics_{jobname}.jsonl`

### Loss
- **What it is**: How wrong your model's predictions are
- **Good value**: Going down over time
- **What you can do**: Nothing directly - just wait and watch

### Gradient Stability
- **What it is**: How consistent your training updates are (0-100%)
- **Good value**:
  - **Video**: > 50%
  - **Images**: > 55%
- **Your current**: ~48% (slightly unstable)
- **What you can do**: **NOTHING** - this measures training dynamics, not a setting
- **Why it matters**: Need > 50% to move to next training phase

### Loss R² (Fit Quality)
- **What it is**: How well we can predict your loss trend (0-1 scale)
- **Good value**:
  - **Video**: > 0.01
  - **Images**: > 0.1
- **Your current**: 0.0058 (too noisy)
- **What you can do**: **NOTHING** - this is measured, not set
- **Why it matters**: Need > 0.01 to move to next phase (confirms loss is actually plateauing)

### Loss Slope
- **What it is**: How fast loss is improving (negative = good)
- **Good value**:
  - Negative (improving): -0.0001 is great
  - Near zero (plateau): Ready for phase transition
  - Positive (getting worse): Problem!
- **Your current**: -0.0001 (good, still improving)

### Learning Rates (lr_0, lr_1)
- **What it is**: How big the training updates are
- **lr_0**: High-noise expert learning rate
- **lr_1**: Low-noise expert learning rate
- **What you can do**: Set in config, automagic adjusts automatically

### Alpha Values (conv_alpha, linear_alpha)
- **What it is**: How strong your LoRA effect is
- **Current**: conv_alpha = 8 (foundation phase)
- **What you can do**: Alpha scheduler changes this automatically when phases transition

### Phase Info
- **phase**: Which training phase you're in (foundation/balance/emphasis)
- **steps_in_phase**: How long you've been in this phase
- **Current**: Foundation phase, step 404

## Phase Transition Requirements

You need **ALL** of these to move from Foundation → Balance:

| Requirement | Target | Your Value | Status |
|-------------|--------|------------|--------|
| Minimum steps | 2000 | 404 | ❌ Not yet |
| Loss plateau | < 0.005 improvement | -0.0001 slope | ✅ Good |
| Gradient stability | > 50% | 48% | ❌ Too low |
| R² confidence | > 0.01 | 0.0058 | ❌ Too noisy |

**What this means**: You're only at step 404. You need at least 2000 steps, PLUS your training needs to be more stable (>50% gradient stability) and less noisy (>0.01 R²).

## Common Questions

### "Can I make gradient stability higher?"
**No.** It measures training dynamics. It will naturally improve as training progresses.

### "Can I make R² better?"
**No.** It measures how noisy your loss is. Video training is inherently noisy. Just keep training.

### "Why is video different from images?"
Video has 10-100x more variance than images, so:
- Video R² threshold: 0.01 (vs 0.1 for images)
- Video gradient stability: 50% (vs 55% for images)
- Video loss plateau: 0.005 (vs 0.001 for images)

### "What should I actually monitor?"
1. **Loss going down**: Good
2. **Phase transitions happening**: Means training is progressing well
3. **Gradient stability trending up**: Means training is stabilizing
4. **Checkpoints being saved**: So you don't lose progress

### "What if phase transitions never happen?"
Your thresholds might be too strict for your specific data. You can:
1. Lower thresholds in your config (loss_improvement_rate_below, min_loss_r2)
2. Disable alpha scheduling and use fixed alpha
3. Keep training anyway - fixed alpha can still work

## Files

- **Metrics file**: `output/{jobname}/metrics_{jobname}.jsonl`
- **Config file**: `output/{jobname}/config.yaml`
- **Checkpoints**: `output/{jobname}/job_XXXX.safetensors`
