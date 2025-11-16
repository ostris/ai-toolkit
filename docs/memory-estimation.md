# Memory Estimation and Batch Size Calculation

This document explains how the AI Toolkit calculates optimal batch sizes and estimates memory usage for different hardware configurations.

## Overview

Memory management is critical for training stability. The toolkit uses different strategies for:
1. **Discrete GPU systems** (NVIDIA, AMD) - Dedicated VRAM separate from system RAM
2. **Unified memory systems** (Apple Silicon, NVIDIA Grace Hopper) - Shared memory pool for CPU and GPU

## Discrete GPU Systems

For systems with dedicated VRAM, the calculation is straightforward:

```typescript
// VRAM is dedicated, RAM is separate
const vramGB = profile.gpu.vramGB;

// Resolution-based scaling factors
if (resolution <= 512) {
  initialBatch = Math.min(16, Math.max(4, Math.floor(vramGB / 3)));
} else if (resolution <= 768) {
  initialBatch = Math.min(8, Math.max(2, Math.floor(vramGB / 4)));
} else if (resolution <= 1024) {
  initialBatch = Math.min(4, Math.max(1, Math.floor(vramGB / 6)));
} else if (resolution <= 1536) {
  initialBatch = Math.min(2, Math.max(1, Math.floor(vramGB / 10)));
} else {
  initialBatch = Math.min(2, Math.max(1, Math.floor(vramGB / 12)));
}
```

**Key assumptions:**
- Model weights loaded into VRAM
- System RAM handles DataLoader workers separately
- No memory contention between CPU and GPU operations

## Unified Memory Systems

Unified memory systems require a much more conservative approach because **everything competes for the same memory pool**.

### Memory Components

The total memory usage is calculated as:

```
Total Memory = Model Weights + Optimizer States + EMA + Workers + OS Reserve + Batch Memory
```

#### 1. Model Weights (varies by architecture)

| Model Architecture | Base Memory (GB) | Notes |
|-------------------|------------------|-------|
| SD 1.x            | 8                | Includes VAE, UNet, text encoder |
| SD 2.x            | 10               | |
| SDXL              | 14               | Two text encoders |
| SD3               | 16               | |
| Flux              | 24               | Large transformer model |
| Flux Kontext      | 24               | |
| Flex 1/2          | 20               | |
| Chroma            | 20               | |
| Lumina 2          | 22               | |
| HiDream           | 24               | |
| OmniGen 2         | 24               | |
| Qwen Image        | 28               | Includes large text encoder |

#### 2. Optimizer States (~4GB for LoRA)

AdamW optimizer stores:
- First moment (momentum): Same size as trainable parameters
- Second moment (variance): Same size as trainable parameters

For a typical LoRA (rank 32), this adds ~2-4GB.

#### 3. Regularization Features

##### EMA (Exponential Moving Average) - ~1GB for LoRA

Exponential Moving Average maintains a **shadow copy of all trainable weights** for smoother, more stable results.

**Memory cost for LoRA training:** ~1GB (only LoRA weights are copied)
**Memory cost for full fine-tuning:** Same as model weights (entire model duplicated)

```yaml
train:
  ema_config:
    use_ema: true  # +1GB for LoRA, +14-28GB for full model
    ema_decay: 0.99
```

##### Differential Output Preservation - ~3GB

Preserves model's original capabilities while learning new concepts. Requires storing original model outputs during forward pass.

**Memory cost:** ~3GB average (depends on resolution and batch size)

```yaml
train:
  diff_output_preservation: true  # +3GB
  diff_output_preservation_multiplier: 1.0
  diff_output_preservation_class: "person"
```

##### Gradient Checkpointing - SAVES ~30% memory

Trades compute time for memory by recomputing activations during backward pass instead of storing them.

**Memory savings:** ~30% of activation memory

```yaml
train:
  gradient_checkpointing: true  # Saves ~4-8GB depending on model
```

#### 4. DataLoader Workers (~8GB each)

**This is often the biggest hidden memory consumer!**

Each PyTorch DataLoader worker:
- Forks the entire Python process
- Copies the dataset object and its state
- Maintains its own memory space
- Includes any cached tensors and model references

With 8 workers: **64GB of memory just for data loading!**

Note: Worker memory usage scales with model size. Larger models (Qwen Image, Flux) have larger dataset objects, resulting in ~8GB per worker instead of the 4GB seen with smaller models.

#### 5. OS and System Reserve (8GB minimum)

Reserved for:
- Operating system
- Desktop environment
- Background processes
- System services

#### 6. Batch Memory (Resolution-dependent)

Memory per sample scales quadratically with resolution:

```typescript
const memoryPerSample = (resolution * resolution) / (1024 * 1024) * 4; // GB
```

| Resolution | Memory per Sample |
|------------|-------------------|
| 512x512    | 1.0 GB            |
| 768x768    | 2.25 GB           |
| 1024x1024  | 4.0 GB            |
| 1536x1536  | 9.0 GB            |
| 2048x2048  | 16.0 GB           |

This includes:
- Input tensors
- Activation memory during forward pass
- Gradient storage during backward pass
- Intermediate buffers

### Calculation Example

**System:** 120GB unified memory, Qwen Image model, 2048px resolution, 2 workers, EMA enabled

```
Model weights:      28 GB (Qwen Image)
Optimizer states:   56 GB (AdamW: 2x model weights)
EMA model:           1 GB (LoRA weights only)
DataLoader workers: 16 GB (2 workers × 8GB)
OS reserve:          8 GB
--------------------------
Total reserved:    109 GB

Available for batches: (120 - 109) × 0.9 = 9.9 GB (10% safety margin)

Memory per sample at 2048px: 16.0 GB
Maximum batch size: floor(9.9 / 16.0) = 0 (not enough memory!)

Final batch size: 1 (minimum)
Max batch size: 1
```

**Why is 109GB reserved?**
This explains the "double memory" phenomenon where Python shows ~9GB but total system memory is at 119GB. The optimizer states (2x model weights for AdamW) are the largest hidden consumer, followed by workers at 8GB each.

**Optimizing for this system:**
1. **Disable EMA**: Saves 1GB (minor impact)
2. **Use 0 workers**: Saves 16GB (2 workers × 8GB)
3. **Enable gradient checkpointing**: Saves ~8GB of activation memory
4. **Enable quantization**: Reduces base model from 28GB to ~14GB (8-bit) or ~7GB (4-bit)

With quantization enabled:
```
Model weights:       14 GB (8-bit quantized)
Optimizer states:    28 GB (2x quantized weights)
EMA model:            1 GB
DataLoader workers:   0 GB (0 workers - main process loads)
OS reserve:           8 GB
--------------------------
Total reserved:      51 GB

Available for batches: (120 - 51) × 0.9 = 62.1 GB
Memory per sample at 2048px: 16.0 GB
Maximum batch size: floor(62.1 / 16.0) = 3
```

### Safety Caps by Resolution

Even with available memory, batch sizes are capped to prevent memory spikes:

| Resolution | Max Initial Batch | Max Batch Size |
|------------|-------------------|----------------|
| ≤512       | 16                | 32             |
| ≤768       | 8                 | 16             |
| ≤1024      | 4                 | 8              |
| ≤1536      | 4                 | 8              |
| >1536      | 2                 | 4              |

Note: Batch sizes of 1-2 lead to noisy gradients and unstable training. The caps are set to allow reasonable batch sizes while staying within memory limits.

## Recommendations for Unified Memory Systems

### Reduce DataLoader Workers (Automatic)

Workers are the biggest hidden memory cost. The toolkit now automatically limits workers for unified memory systems:

| Total Memory | Recommended Workers | Memory Saved vs 8 Workers |
|--------------|---------------------|---------------------------|
| ≥128 GB      | 4                   | 32 GB                     |
| 64-127 GB    | 2                   | 48 GB                     |
| <64 GB       | 0                   | 64 GB                     |

**Why prioritize batch size over workers?**

1. **Batch size directly impacts training quality**: Batch size 1-2 produces very noisy gradients and unstable training. Batch size 4+ provides much better gradient estimates.

2. **Workers are mostly idle with cached latents**: When latents are cached in memory, workers just load pre-computed tensors. This is fast even with 2 workers.

3. **Memory efficiency**: 64GB for 8 workers vs. using that memory for 4+ additional training samples.

Example configuration:

```yaml
datasets:
  - num_workers: 0  # Recommended for large models on unified memory
    persistent_workers: false
```

With 0 workers instead of 8:
- Saves: 64GB of memory
- Allows: 4 additional samples in batch at 2048px resolution
- Trade-off: Slightly slower data loading, but main process handles it fine with cached latents

### Enable Gradient Checkpointing

Trades compute for memory:

```yaml
train:
  gradient_checkpointing: true  # Reduces activation memory by ~30%
```

### Use Disk Caching

Moves latent cache to disk:

```yaml
datasets:
  - cache_latents: false
    cache_latents_to_disk: true
```

This reduces worker memory since they don't need to hold the cache in memory.

### Lower Resolution Training

Consider training at 1024px instead of 1536px:
- Memory per sample: 4GB vs 9GB
- Can fit 2-3x more samples per batch
- Quality difference is often minimal for LoRA

## Debugging Memory Issues

### Check Worker Memory Usage

```bash
# Monitor worker processes
htop  # Look for "pt_data_worker" processes

# Or use:
ps aux | grep pt_data_worker
```

### Monitor Total Memory

```bash
# Real-time memory monitoring
watch -n 1 free -h

# Or use btop/htop for visual monitoring
btop
```

### Common OOM Causes

1. **Too many workers**: Each worker uses ~8GB
2. **Large batch size**: Especially at high resolutions
3. **No gradient checkpointing**: Full activation memory
4. **Memory leaks**: Check for growing memory usage over time

### Auto-Scaling Recovery

The toolkit includes automatic batch size tuning that:
- Detects OOM errors
- Reduces batch size by 25%
- Retries the training step
- Gradually increases batch size if stable

```yaml
train:
  auto_scale_batch_size: true
  min_batch_size: 1
  max_batch_size: 4  # Conservative for 1536px
  batch_size_warmup_steps: 100
```

## API Reference

### `calculateBatchDefaults()`

```typescript
function calculateBatchDefaults(
  profile: SystemProfile,
  resolution: number,
  intent: UserIntent,
  modelArch: string = 'flux',
  numWorkers: number = 4
): BatchConfig
```

Returns optimal batch configuration based on hardware and training parameters.

### `estimateVRAMUsage()`

```typescript
function estimateVRAMUsage(
  modelArch: string,
  resolution: number,
  batchSize: number,
  quantization: string | null,
  gradientCheckpointing: boolean,
  isUnifiedMemory: boolean = false,
  numWorkers: number = 4,
  useEma: boolean = false,
  useDiffOutputPreservation: boolean = false
): number
```

Returns estimated memory usage in GB.

### `calculateRegularizationMemoryCost()`

```typescript
interface RegularizationMemoryCost {
  ema: number;  // GB for EMA shadow weights
  diffOutputPreservation: number;  // GB for preservation loss computation
  gradientCheckpointingSavings: number;  // GB saved (negative = savings)
}

function calculateRegularizationMemoryCost(
  modelArch: string,
  useEma: boolean,
  useDiffOutputPreservation: boolean,
  gradientCheckpointing: boolean
): RegularizationMemoryCost
```

Returns memory costs for each regularization option. Use this to show users the impact of their choices.

## Future Improvements

1. **Runtime profiling**: Measure actual memory usage during first few steps
2. **Model-specific activation memory**: More accurate per-architecture estimates
3. **Dynamic worker adjustment**: Reduce workers when memory is tight
4. **Memory pressure monitoring**: Real-time adjustment based on system state
