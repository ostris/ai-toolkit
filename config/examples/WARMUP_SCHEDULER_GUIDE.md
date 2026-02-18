# Learning Rate Scheduler Warmup Guide

## Overview

This guide explains how to use the warmup functionality for learning rate schedulers in the AI Toolkit. Warmup gradually increases the learning rate from near-zero to the target learning rate over a specified number of steps, which can help stabilize training in the early stages.

## Supported Schedulers

Warmup is supported for the following schedulers:
- `cosine` - Cosine annealing scheduler
- `cosine_with_restarts` - Cosine annealing with warm restarts (SGDR)

## How It Works

When you specify `warmup_steps > 0`, the scheduler automatically creates a composite scheduler using PyTorch's `SequentialLR`:

1. **Warmup Phase** (steps 0 to `warmup_steps`):
   - Uses `LinearLR` to gradually increase learning rate from ~0 to the target LR
   - Learning rate increases linearly: `lr = target_lr * (current_step / warmup_steps)`

2. **Main Phase** (steps `warmup_steps` to `total_steps`):
   - Uses the specified scheduler (cosine or cosine_with_restarts)
   - `total_iters` specifies the TOTAL number of training iterations (including warmup)
   - `T_0`/`T_max` specify iterations for the MAIN scheduler phase (after warmup)
   - If T_0/T_max is specified, it takes priority over calculated value from total_iters
   - Main scheduler iterations = total_iters - warmup_steps (or T_0/T_max if specified)

### Learning Rate Progression

```
LR |
   |    _/--\      /\        /\
   |   /     \    /  \      /  \
   |  /       \  /    \    /    \
   | /         \/      \  /      \
   |/                   \/        \
   +----------------------------> steps
   |<-warmup->|<-- cosine restarts -->
```

## Parameter Semantics

### Key concepts

- **`total_iters`**: TOTAL training iterations (including warmup)
- **`T_0`/`T_max`**: Main scheduler iterations (after warmup), overrides calculation from total_iters

### Example: Training with 1000 total steps

**Config 1: Using total_iters (automatic calculation)**

```yaml
train:
  steps: 1000  # Will be used as total_iters by BaseSDTrainProcess
  lr_scheduler: "cosine_with_restarts"
  lr_scheduler_params:
    warmup_steps: 100
    # total_iters will be auto-set to 1000 by BaseSDTrainProcess
    T_mult: 2
```

**Result:**
- Steps 0-100: Linear warmup
- Steps 100-1000: Cosine with restarts (900 iterations = 1000 - 100)
- Total: 1000 steps ✓

**Config 2: Using T_0 explicitly (overrides calculation)**

```yaml
train:
  steps: 1000
  lr_scheduler: "cosine_with_restarts"
  lr_scheduler_params:
    warmup_steps: 100
    total_iters: 1000
    T_0: 500  # Overrides default calculation (1000 - 100)
    T_mult: 2
```

**Result:**
- Steps 0-100: Linear warmup
- Steps 100-600: Cosine with restarts (500 iterations from T_0)
- Total: 600 steps (less than total_iters!)

**Priority:** If both `T_0` and `total_iters` are specified, `T_0` takes priority and determines main scheduler length.

## Configuration Examples

### Example 1: Cosine with Restarts + Warmup

```yaml
train:
  steps: 2000
  lr: 1e-4
  lr_scheduler: "cosine_with_restarts"
  lr_scheduler_params:
    warmup_steps: 100      # Warmup for first 100 steps
    T_mult: 2              # Double restart period each cycle
    eta_min: 1e-7          # Minimum learning rate
```

### Example 2: Cosine + Warmup

```yaml
train:
  steps: 2000
  lr: 1e-4
  lr_scheduler: "cosine"
  lr_scheduler_params:
    warmup_steps: 100      # Warmup for first 100 steps
    eta_min: 1e-7          # Minimum learning rate
```

### Example 3: Without Warmup (Backward Compatible)

```yaml
train:
  steps: 2000
  lr: 1e-4
  lr_scheduler: "cosine_with_restarts"
  lr_scheduler_params:
    T_mult: 2
    eta_min: 1e-7
    # No warmup_steps specified = no warmup
```

### Example 4: Explicitly Disable Warmup

```yaml
train:
  steps: 2000
  lr: 1e-4
  lr_scheduler: "cosine_with_restarts"
  lr_scheduler_params:
    warmup_steps: 0        # Explicitly disable warmup
    T_mult: 2
    eta_min: 1e-7
```

## Parameters

### Common Parameters

- **`warmup_steps`** (optional, default: 0)
  - Number of steps for the warmup phase
  - Set to 0 or omit to disable warmup
  - Typical values: 50-500 depending on total training steps
  - Rule of thumb: 5-10% of total steps

### Cosine Scheduler Parameters

- **`eta_min`** (optional, default: 0)
  - Minimum learning rate

### Cosine with Restarts Parameters

- **`T_mult`** (optional, default: 1)
  - Factor to increase the restart period after each restart
  - `T_mult=1`: equal restart periods
  - `T_mult=2`: double the period each time
  
- **`eta_min`** (optional, default: 0)
  - Minimum learning rate

## Choosing Warmup Steps

### General Guidelines

1. **Small datasets (< 1000 images)**
   - Warmup steps: 50-100
   - Helps prevent overfitting to early batches

2. **Medium datasets (1000-10000 images)**
   - Warmup steps: 100-250
   - Balances stability and training time

3. **Large datasets (> 10000 images)**
   - Warmup steps: 250-500
   - More warmup helps with stability

4. **Percentage-based approach**
   - Use 5-10% of total training steps
   - Example: 2000 steps → 100-200 warmup steps

### When to Use Warmup

✅ **Use warmup when:**
- Training from scratch or with random initialization
- Using high learning rates (> 1e-4)
- Experiencing unstable early training
- Training large models or with large batch sizes
- Using aggressive optimizers (Adam with high β values)

❌ **Skip warmup when:**
- Fine-tuning from a well-trained checkpoint
- Using very low learning rates (< 1e-5)
- Training is already stable without warmup
- Total training steps are very small (< 500)

## Implementation Details

### Under the Hood

The implementation uses PyTorch's built-in schedulers:

```python
# Warmup phase
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1e-10,  # Start almost at 0
    end_factor=1.0,      # End at full LR
    total_iters=warmup_steps
)

# Main phase (example for cosine_with_restarts)
main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=total_iters - warmup_steps,  # Calculated from total iterations
    # Or explicitly specify main scheduler iterations:
    # T_0=900,  # Direct specification (ignores total_iters calculation)
    T_mult=2,
    eta_min=1e-7
)

# Combined scheduler
combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[warmup_steps]
)
```

### Backward Compatibility

- All existing configurations continue to work without changes
- Warmup is only activated when `warmup_steps > 0` is explicitly specified
- Default behavior (no warmup) is preserved when `warmup_steps` is not specified

## Testing

A test script is provided to verify the warmup functionality:

```bash
# Activate your virtual environment
.\venv\Scripts\activate.ps1  # Windows PowerShell
# or
source venv/bin/activate      # Linux/Mac

# Run tests
python test_scheduler_warmup.py
```

The test script verifies:
1. Backward compatibility (schedulers work without warmup)
2. Warmup functionality (SequentialLR is created when warmup_steps > 0)
3. Learning rate progression (LR increases during warmup, then follows main scheduler)

## Full Configuration Example

See `config/examples/train_lora_flux_with_warmup.yaml` for a complete working example.

## Troubleshooting

### Issue: Learning rate doesn't increase during warmup

**Solution:** Make sure `warmup_steps` is specified in `lr_scheduler_params`, not at the top level of `train`.

```yaml
# ❌ Wrong
train:
  warmup_steps: 100
  lr_scheduler_params:
    T_mult: 2

# ✅ Correct
train:
  lr_scheduler_params:
    warmup_steps: 100
    T_mult: 2
```

### Issue: Training is unstable even with warmup

**Possible solutions:**
1. Increase `warmup_steps` (try 10-15% of total steps)
2. Reduce base learning rate
3. Use gradient clipping (`max_grad_norm`)
4. Reduce batch size

### Issue: Warmup takes too long

**Solution:** Reduce `warmup_steps`. Remember, warmup is just the initial phase. If warmup is more than 10-15% of total training, it might be too long.

## References

- [PyTorch CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
- [PyTorch CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)
- [PyTorch SequentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html)
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
