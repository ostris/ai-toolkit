# Training Recommendations Based on Research

## Problem Summary

Your training run showed classic signs of:
1. **High noise expert overfitting** (rapid improvement then plateau with high variance)
2. **Low noise expert degradation** (performance got worse: 0.0883 → 0.0969)
3. **Gradient instability** preventing phase transitions (0.486 vs 0.50 required)

## Root Causes (Research-Backed)

### 1. High Noise Timesteps Overfit Rapidly

**Source**: T-LoRA paper (arxiv.org/html/2507.05964v1)

> "Fine-tuning at higher timesteps t∈[800;1000] leads to rapid overfitting, causing memorization of poses and backgrounds, which limits image diversity."

**Your data confirms this:**
- High noise: Loss improved 27% (0.1016 → 0.0739)
- But variance remained extremely high (±0.066)
- Trained for 1566 steps (research recommends 500-800 max)

### 2. Gradient Conflicts Between Timesteps

**Source**: Decouple-Then-Merge paper, Min-SNR Weighting Strategy

> "Optimizing a denoising function for a specific noise level can harm other timesteps" and "gradients computed at different timesteps may conflict."

**Your data confirms this:**
- Low noise loss WORSENED by 10%
- High noise's aggressive updates created conflicting gradients
- Overall gradient stability stuck at 48.6%

### 3. Your Config Amplified the Problem

```yaml
# CURRENT (WRONG)
high_noise_lr_bump: 1.0e-05  # 2x higher - encourages overfitting
low_noise_lr_bump: 5.0e-06   # 2x lower - handicaps the expert that needs help

# RESEARCH SAYS:
- T-LoRA: REDUCE training signal at high noise (fewer params, lower LR)
- TimeStep Master: Use UNIFORM learning rate (1e-4) across all experts
- Min-SNR: Use loss weighting to balance timesteps, not different LRs
```

## Recommended Config Changes

### Option 1: Equal Learning Rates (Recommended)

```yaml
train:
  optimizer: automagic
  optimizer_params:
    # Same LR for both experts (TimeStep Master approach)
    high_noise_lr_bump: 8.0e-06
    high_noise_min_lr: 8.0e-06
    high_noise_max_lr: 0.0002

    low_noise_lr_bump: 8.0e-06
    low_noise_min_lr: 8.0e-06
    low_noise_max_lr: 0.0002

    # Shared settings
    beta2: 0.999
    weight_decay: 0.0001
    clip_threshold: 1
```

### Option 2: Inverted LRs (Conservative High Noise)

```yaml
train:
  optimizer: automagic
  optimizer_params:
    # LOWER LR for high noise to prevent overfitting
    high_noise_lr_bump: 5.0e-06
    high_noise_min_lr: 5.0e-06
    high_noise_max_lr: 0.0001  # Half of low noise

    # HIGHER LR for low noise to help it learn
    low_noise_lr_bump: 1.0e-05
    low_noise_min_lr: 8.0e-06
    low_noise_max_lr: 0.0002
```

### Option 3: Reduce High Noise Rank (T-LoRA Strategy)

If the toolkit supports dynamic rank adjustment:
- High noise: Use rank 32 (half of full rank 64)
- Low noise: Use rank 64 (full capacity)

This reduces high noise's memorization capacity while maintaining low noise's detail learning.

## Training Duration Recommendations

**From T-LoRA paper:**
- 500-800 training steps per expert with orthogonal initialization
- Stop high noise early if loss plateaus with high variance

**Your training:**
- High noise: 1566 steps (2x too long - likely overfitted by step 800)
- Low noise: 1504 steps (also too long given the degradation)

**Recommendation:**
- Target 600-800 steps per expert maximum
- Monitor samples frequently (every 100 steps)
- Stop if high noise shows memorization (identical poses, backgrounds)
- Stop if low noise degrades (loss increases)

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

## Alternative Approaches

### 1. Sequential Training (Decouple-Then-Merge)

Train experts separately then merge:

```bash
# Phase 1: Train high noise ONLY
python run.py --config high_noise_only.yaml  # 500 steps

# Phase 2: Train low noise ONLY (starting from phase 1 checkpoint)
python run.py --config low_noise_only.yaml   # 800 steps

# Phase 3: Joint fine-tuning (short, both experts)
python run.py --config both_experts.yaml     # 200 steps
```

### 2. Min-SNR Loss Weighting

If supported, use SNR-based loss weighting instead of per-expert LRs:

```yaml
train:
  loss_weighting: min_snr
  min_snr_gamma: 5  # Standard value
```

### 3. Early Stopping Per Expert

Implement checkpointing:
- Save every 100 steps
- Test samples at each checkpoint
- Identify when high noise overfits (usually ~500-800 steps)
- Identify when low noise degrades
- Resume from best checkpoint

## Monitoring Guidelines

Watch for these warning signs:

**High Noise Overfitting:**
- Loss plateaus but variance stays high (±0.05+)
- Samples show memorized poses/backgrounds
- Gradient stability decreases

**Low Noise Degradation:**
- Loss INCREASES instead of decreasing
- Samples lose fine details
- Becomes worse than early checkpoints

**Gradient Conflicts:**
- Overall gradient stability stuck below 0.50
- Loss oscillates heavily between expert switches
- Phase transitions never trigger

## Next Steps

1. **Stop current training** if still running
2. **Review samples** from steps 500, 800, 1000, 1500
3. **Identify best checkpoint** before overfitting started
4. **Restart training** with equal LRs or inverted LRs
5. **Target 600-800 steps per expert** maximum
6. **Test frequently** and stop early if issues appear

## Research References

1. **T-LoRA**: Single Image Diffusion Model Customization Without Overfitting
   - arxiv.org/html/2507.05964v1
   - Key insight: High noise timesteps overfit rapidly

2. **TimeStep Master**: Asymmetrical Mixture of Timestep LoRA Experts
   - arxiv.org/html/2503.07416
   - Key insight: Use uniform LR, separate LoRAs per timestep range

3. **Min-SNR Weighting Strategy**: Efficient Diffusion Training via Min-SNR
   - openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf
   - Key insight: Gradient conflicts between timesteps

4. **Decouple-Then-Merge**: Towards Better Training for Diffusion Models
   - openreview.net/forum?id=Y0P6cOZzNm
   - Key insight: Train timestep ranges separately to avoid interference

## Questions?

If loss behavior doesn't match these patterns, or if you see unexpected results:
- Check dataset quality (corrupted frames, bad captions)
- Verify model architecture (correct WAN 2.2 I2V 14B variant)
- Review batch size / gradient accumulation
- Check for NaN/Inf in loss logs
