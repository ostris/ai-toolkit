# Training Recommendations for WAN 2.2 I2V MOTION LoRAs

## CRITICAL: Motion vs Character Training

**This document is for MOTION training (rubbing, squirting, movement).**
Character/style training research (T-LoRA, etc.) gives **OPPOSITE** recommendations.

### Character Training vs Motion Training

| Aspect | Character/Style | Motion |
|--------|----------------|--------|
| **High Noise Role** | Memorizes poses/backgrounds (BAD) | Learns coarse motion structure (CRITICAL) |
| **Low Noise Role** | Refines details (CRITICAL) | Can suppress motion if too strong |
| **LR Strategy** | Lower high noise to prevent overfitting | **HIGHER high noise to preserve motion** |
| **Training Duration** | 500-800 steps max | 1800-2200 steps |

## Problem Summary (squ1rtv15 Analysis)

Your training run showed:
1. **Motion degradation** - Early samples had crazy coarse motion, later samples became tame/no motion
2. **Low noise overpowering** - Weight growth 1.3x faster than high noise after step 2400
3. **LR ratio too small** - 1.35x ratio insufficient for motion dominance
4. **Best checkpoint still had issues** - Floaty/slow motion, weak coarse movement

## Root Causes (Weight Analysis)

### squ1rtv15 Step 2400 (Best Checkpoint) Analysis:

```
High Noise Expert:
- Loss: 0.0755 (±0.0715 std)
- Learning Rate: 0.000148
- Weight magnitude: 0.005605 (NEEDS 0.008-0.010 for strong motion)
- Training steps: ~783 high noise batches

Low Noise Expert:
- Loss: 0.0826 (±0.0415 std)
- Learning Rate: 0.000110
- Weight magnitude: 0.004710

LR Ratio: 1.35x (high/low) - INSUFFICIENT FOR MOTION
Weight Ratio: 1.19x (high/low) - TOO WEAK
```

### What Went Wrong (Steps 2400→3000):

```
High Noise: +5.4% weight growth
Low Noise:  +7.1% weight growth (1.3x FASTER!)

Result: Low noise overpowered motion, made it tame/suppressed
```

## Corrected Config for Motion Training

### Recommended: 4x LR Ratio (Motion Dominance)

```yaml
train:
  optimizer: automagic
  optimizer_params:
    # HIGH noise gets 4x MORE learning rate (motion structure is critical)
    high_noise_lr_bump: 2.0e-05    # 4x higher than low noise
    high_noise_min_lr: 2.0e-05
    high_noise_max_lr: 0.0005      # Allow growth for strong motion

    # LOW noise constrained (prevents suppressing motion)
    low_noise_lr_bump: 5.0e-06     # Same as original (worked for refinement)
    low_noise_min_lr: 5.0e-06
    low_noise_max_lr: 0.0001       # Capped to prevent overpowering

    # Shared settings
    beta2: 0.999
    weight_decay: 0.0001
    clip_threshold: 1

  steps: 2200  # Stop before low noise overpowers (was 10000)
```

### Conservative: 3x LR Ratio

If 4x seems too aggressive, try 3x:

```yaml
train:
  optimizer: automagic
  optimizer_params:
    high_noise_lr_bump: 1.5e-05    # 3x higher than low noise
    high_noise_min_lr: 1.5e-05
    high_noise_max_lr: 0.0004

    low_noise_lr_bump: 5.0e-06
    low_noise_min_lr: 5.0e-06
    low_noise_max_lr: 0.0001
```

## Training Duration Recommendations

**For Motion LoRAs (squ1rtv15 data):**
- Best checkpoint: Steps 2000-2400 (but still had issues)
- After 2400: Low noise started overpowering motion
- Total trained: 3070 steps (degraded significantly)

**Recommended for next run:**
- Target: 1800-2200 total steps
- Monitor samples every 100 steps
- Watch for motion becoming tame/suppressed (low noise overpowering)
- Stop immediately if motion quality degrades

**Warning signs to stop training:**
- Motion becomes floaty/slow
- Coarse movement weakens
- Samples lose energy/intensity
- Weight ratio (high/low) drops below 1.5x

## Phase Transition Strategy

Your original thresholds were too strict for video MoE training with gradient conflicts.

**Updated thresholds (already committed):**

```yaml
network:
  alpha_schedule:
    conv_alpha_phases:
      foundation:
        exit_criteria:
          min_gradient_stability: 0.47  # Was 0.50, you were at 0.486
          min_loss_r2: 0.005           # Advisory only
          loss_improvement_rate_below: 0.005
```

## Alternative Approaches (NOT RECOMMENDED)

### Min-SNR Loss Weighting - INCOMPATIBLE

**DO NOT USE** - WAN 2.2 uses FlowMatch scheduler which lacks `alphas_cumprod` attribute.

```
AttributeError: 'CustomFlowMatchEulerDiscreteScheduler' object has no attribute 'alphas_cumprod'
```

Min-SNR weighting only works with DDPM-based schedulers, not FlowMatch.

### Sequential Training - UNTESTED

Could train experts separately, but ai-toolkit doesn't currently support this for WAN 2.2 I2V:

```bash
# Theoretical approach (not implemented):
# Phase 1: High noise only (1000 steps)
# Phase 2: Low noise only (1500 steps)
# Phase 3: Joint fine-tuning (200 steps)
```

Easier to use differential learning rates as shown above.

## Monitoring Guidelines for Motion Training

Watch for these warning signs:

**Motion Degradation (Low Noise Overpowering):**
- Motion becomes tame/subtle compared to earlier samples
- Coarse movement weakens (less rubbing, less body movement)
- Motion feels floaty or slow-motion
- Weight ratio (high/low) decreasing over time
- **ACTION:** Stop training immediately, use earlier checkpoint

**High Noise Too Weak:**
- Weight magnitude stays below 0.008
- LR ratio under 3x
- Samples lack energy from the start
- **ACTION:** Increase high_noise_lr_bump for next run

**Low Noise Overpowering (Critical Issue):**
- Low noise weight growth FASTER than high noise
- Motion suppression after checkpoint that looked good
- Loss improving but samples getting worse
- **ACTION:** Lower low_noise_max_lr or stop training earlier

**Good Progress Indicators:**
- Weight ratio (high/low) stays above 1.5x
- Motion intensity consistent across checkpoints
- Coarse movement strong, details refining gradually
- LR ratio staying at 3-4x throughout training

## Next Steps for squ1rtv17

1. **Create new config** with 4x LR ratio (high_noise: 2e-5, low_noise: 5e-6)
2. **Set max steps to 2200** (not 10000)
3. **Monitor samples every 100 steps** - watch for motion degradation
4. **Stop immediately if**:
   - Motion becomes tame/weak
   - Weight ratio drops below 1.5x
   - Samples worse than earlier checkpoint
5. **Best checkpoint likely around step 1800-2000**

## Key Learnings from squ1rtv15

**What Worked:**
- Dataset quality good (motion present in early samples)
- WAN 2.2 I2V architecture correct
- Alpha scheduling (foundation phase at alpha=8)
- Save frequency (every 100 steps allowed finding best checkpoint)

**What Failed:**
- LR ratio too small (1.35x insufficient for motion)
- Trained too long (3070 steps, should stop ~2000)
- Low noise overpowered motion after step 2400
- High noise weights too weak (0.0056 vs needed 0.008-0.010)

**Critical Insight:**
Motion LoRAs need HIGH noise expert to dominate. Character LoRAs are opposite.

## Research Context

**WARNING:** Most LoRA research focuses on character/style training, which is backwards for motion.

**Relevant Concepts:**
- **WAN 2.2 I2V Architecture**: Dual transformer MoE (boundary_ratio=0.9)
  - transformer_1: High noise (900-1000 timesteps, 10% of denoising)
  - transformer_2: Low noise (0-900 timesteps, 90% of denoising)

- **Gradient Conflicts**: Different timestep experts can interfere (why MoE helps)

- **Weight Magnitude**: Indicates training strength (~0.008-0.010 for strong motion)

**Character Training Research (T-LoRA, etc.) - NOT APPLICABLE:**
- Recommends LOWER high noise LR (opposite of what motion needs)
- Warns about overfitting at high timesteps (not an issue for motion)
- Targets 500-800 steps (too short for motion learning)

## Diagnostic Checklist

If next training run still has issues:

**Dataset Quality:**
- [ ] All videos show clear rubbing motion
- [ ] Squirting visible in source videos
- [ ] Captions describe motion ("rubbing", "squirting")
- [ ] No corrupted frames

**Model Setup:**
- [ ] Using ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16
- [ ] Quantization: uint4 (for model), qfloat8 (for text encoder)
- [ ] arch: wan22_14b_i2v
- [ ] boundary_ratio: 0.9 (I2V default)

**Training Params:**
- [ ] LR ratio 3-5x (high/low)
- [ ] Max steps 1800-2200
- [ ] Batch size 1, gradient accumulation 1
- [ ] FlowMatch scheduler (NOT DDPM)
- [ ] No min_snr_gamma (incompatible)

**Monitoring:**
- [ ] Save every 100 steps
- [ ] Check samples at each checkpoint
- [ ] Watch weight ratios in metrics
- [ ] Stop if motion degrades
