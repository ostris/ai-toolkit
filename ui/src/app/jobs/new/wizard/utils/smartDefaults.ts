/**
 * Smart Defaults Engine
 *
 * Calculates optimal configuration values based on system profile,
 * user intent, and dataset characteristics.
 */

import {
  SystemProfile,
  UserIntent,
  DatasetInfo,
  BatchConfig,
  CacheConfig,
  WorkerConfig,
  AdvisorMessage
} from './types';

/**
 * Calculate optimal batch size configuration (TODO #9 Integration)
 *
 * Two separate formulas:
 * 1. Discrete GPU: Only VRAM matters, RAM is separate
 * 2. Unified Memory: Must account for model weights, optimizer states, workers, and OS overhead
 */
export function calculateBatchDefaults(
  profile: SystemProfile,
  resolution: number,
  intent: UserIntent,
  modelArch: string,
  numWorkers: number
): BatchConfig {
  let initialBatch: number;
  let maxBatch: number;

  if (profile.gpu.isUnifiedMemory) {
    // UNIFIED MEMORY SYSTEMS (Apple Silicon, etc.)
    // Everything shares the same memory pool - must be conservative

    const totalMemoryGB = profile.memory.unifiedMemory || profile.memory.totalRAM;

    // Reserve memory for:
    // - OS and system processes: 8GB minimum
    // - Model weights (base model in memory): varies by arch
    // - Optimizer states (2x LoRA weights for AdamW): ~2GB for typical LoRA
    // - EMA model (if enabled): ~1GB for LoRA
    // - DataLoader workers: ~8GB each (they fork the Python process + dataset objects)
    // - Safety margin: 10%

    const modelMemoryGB: Record<string, number> = {
      sd1: 8,
      sd15: 8,
      sd2: 10,
      sdxl: 14,
      sd3: 16,
      flux: 24,
      flux_kontext: 24,
      flex1: 20,
      flex2: 20,
      chroma: 20,
      lumina2: 22,
      hidream: 24,
      omnigen2: 24,
      qwen_image: 28  // 14GB base + optimizer states + text encoder
    };

    const baseModelMemory = modelMemoryGB[modelArch] || 24;
    const workerMemory = numWorkers * 8; // ~8GB per worker process (forks entire Python process + dataset objects)
    const osReserve = 8;
    const optimizerAndEMA = 4; // Optimizer states + EMA for LoRA

    const reservedMemory = baseModelMemory + workerMemory + osReserve + optimizerAndEMA;
    const availableForBatches = Math.max(0, totalMemoryGB - reservedMemory) * 0.9; // 10% safety margin

    // Memory per sample at different resolutions (GB) - includes activations and gradients
    const memoryPerSample = (resolution * resolution) / (1024 * 1024) * 4; // ~4GB at 1024px

    if (availableForBatches <= 0) {
      // Not enough memory - use minimum settings
      initialBatch = 1;
      maxBatch = 1;
    } else {
      initialBatch = Math.max(1, Math.floor(availableForBatches / memoryPerSample));
      // Cap based on resolution - but allow reasonable batch sizes for training stability
      // A batch size of 1 leads to very noisy gradients, so we prefer at least 2-4
      if (resolution <= 512) {
        initialBatch = Math.min(16, initialBatch);
        maxBatch = Math.min(32, initialBatch * 2);
      } else if (resolution <= 768) {
        initialBatch = Math.min(8, initialBatch);
        maxBatch = Math.min(16, initialBatch * 2);
      } else if (resolution <= 1024) {
        initialBatch = Math.min(4, initialBatch);
        maxBatch = Math.min(8, initialBatch * 2);
      } else if (resolution <= 1536) {
        // Allow up to 4 for 1536px if memory permits - better gradient stability
        initialBatch = Math.min(4, initialBatch);
        maxBatch = Math.min(8, initialBatch * 2);
      } else {
        initialBatch = Math.min(2, initialBatch);
        maxBatch = Math.min(4, initialBatch * 2);
      }
    }
  } else {
    // DISCRETE GPU SYSTEMS (NVIDIA, AMD)
    // VRAM is dedicated, RAM is separate - can be more aggressive

    const vramGB = profile.gpu.vramGB;

    // Calculate based on resolution and available VRAM
    // These assume model is loaded into VRAM with some overhead
    if (resolution <= 512) {
      initialBatch = Math.min(16, Math.max(4, Math.floor(vramGB / 3)));
      maxBatch = initialBatch * 2;
    } else if (resolution <= 768) {
      initialBatch = Math.min(8, Math.max(2, Math.floor(vramGB / 4)));
      maxBatch = Math.min(16, initialBatch * 2);
    } else if (resolution <= 1024) {
      initialBatch = Math.min(4, Math.max(1, Math.floor(vramGB / 6)));
      maxBatch = Math.min(8, initialBatch * 2);
    } else if (resolution <= 1536) {
      initialBatch = Math.min(2, Math.max(1, Math.floor(vramGB / 10)));
      maxBatch = Math.min(4, initialBatch * 2);
    } else {
      initialBatch = Math.min(2, Math.max(1, Math.floor(vramGB / 12)));
      maxBatch = Math.min(4, initialBatch * 2);
    }
  }

  // Auto-scaling is always recommended - it finds optimal batch size without affecting quality
  // Larger batches improve training stability and gradient accuracy
  const autoScale = true;

  return {
    batch_size: initialBatch,
    auto_scale_batch_size: autoScale,
    min_batch_size: 1,
    max_batch_size: maxBatch,
    batch_size_warmup_steps: 100
  };
}

/**
 * Calculate optimal caching strategy (TODO #2, #3 Integration)
 */
export function calculateCachingStrategy(
  profile: SystemProfile,
  datasetInfo: DatasetInfo
): CacheConfig {
  // Estimate cache size: ~6MB per image for latents + embeddings
  const cacheSizeGB = (datasetInfo.total_images * 6) / 1024;
  const availableRAM = profile.memory.availableRAM - 8; // Reserve for OS

  if (profile.gpu.isUnifiedMemory) {
    // Apple Silicon: Always use memory cache (no GPU transfer overhead)
    return {
      cache_latents: true,
      cache_latents_to_disk: false,
      reason: `Unified memory detected - in-memory cache optimal (no GPU transfer overhead)`
    };
  }

  if (cacheSizeGB < availableRAM * 0.7) {
    // TODO #2: Shared memory cache
    return {
      cache_latents: true,
      cache_latents_to_disk: false,
      reason: `Cache fits in RAM (${cacheSizeGB.toFixed(1)}GB < ${(availableRAM * 0.7).toFixed(1)}GB available). Using shared memory for fastest access.`
    };
  } else {
    // TODO #3: Memory-mapped disk cache
    return {
      cache_latents: false,
      cache_latents_to_disk: true,
      reason: `Cache too large for RAM (${cacheSizeGB.toFixed(1)}GB). Using disk cache with memory-mapping for minimal RAM usage.`
    };
  }
}

/**
 * Calculate optimal GPU prefetching (TODO #5 Integration)
 */
export function calculatePrefetching(
  profile: SystemProfile,
  intent: UserIntent
): number {
  if (profile.gpu.isUnifiedMemory) {
    // Less benefit with unified memory
    return 1;
  }

  const baseValue = {
    hdd: 3,
    ssd: 2,
    nvme: 1
  }[profile.storage.type];

  if (intent.priority === 'speed') {
    return baseValue + 1;
  } else if (intent.priority === 'memory_efficiency') {
    return Math.max(0, baseValue - 1);
  }

  return baseValue;
}

/**
 * Calculate optimal worker configuration (TODO #4 Integration)
 *
 * For unified memory systems, workers compete with training for memory.
 * Each worker uses ~8GB, so we prioritize batch size over worker count.
 * With cached latents, workers mostly just load pre-computed tensors,
 * so 2-4 workers is usually sufficient.
 */
export function calculateWorkers(profile: SystemProfile): WorkerConfig {
  let numWorkers: number;

  if (profile.gpu.isUnifiedMemory) {
    // Unified memory: Be conservative - workers steal memory from training
    // 2-4 workers is plenty for cached latent loading
    // This saves 32-48GB compared to 8 workers, allowing for larger batch sizes
    const totalMemoryGB = profile.memory.unifiedMemory || profile.memory.totalRAM;

    if (totalMemoryGB >= 128) {
      numWorkers = 4; // Large unified memory can afford 4 workers
    } else if (totalMemoryGB >= 64) {
      numWorkers = 2; // Medium unified memory - prioritize batch size
    } else {
      numWorkers = 0; // Small unified memory - use main process only
    }
  } else {
    // Discrete GPU: Workers use system RAM, not VRAM
    // Can be more generous here
    const baseWorkers = Math.floor(profile.memory.totalRAM / 8);
    numWorkers = Math.min(baseWorkers, profile.cpu.cores, 8);
  }

  return {
    num_workers: Math.max(0, numWorkers),
    persistent_workers: numWorkers > 0 // Keep workers alive to avoid respawn overhead
  };
}

/**
 * Calculate optimal learning rate based on training type
 */
export function calculateLearningRate(
  intent: UserIntent,
  targetType: 'lora' | 'lokr' | 'full'
): number {
  if (targetType === 'full') {
    return 1e-5;
  }

  // LoRA/LoKr learning rates by training type
  const lrMap: Record<string, number> = {
    person: 1e-4,
    style: 1e-4,
    object: 1e-4,
    concept: 5e-4,
    other: 1e-4
  };

  return lrMap[intent.trainingType] || 1e-4;
}

/**
 * Calculate recommended training steps
 */
export function calculateSteps(
  datasetInfo: DatasetInfo,
  intent: UserIntent
): number {
  const baseMultiplier = intent.priority === 'quality' ? 15 : intent.priority === 'speed' ? 8 : 10;
  const steps = datasetInfo.total_images * baseMultiplier;

  // Clamp to reasonable range
  return Math.max(500, Math.min(10000, steps));
}

/**
 * Calculate recommended LoRA rank
 */
export function calculateLoraRank(
  intent: UserIntent,
  datasetInfo: DatasetInfo
): { rank: number; alpha: number } {
  // Quality: higher rank, Speed: lower rank, Memory efficiency: lower rank
  let rank: number;

  if (intent.priority === 'quality') {
    rank = datasetInfo.total_images < 50 ? 16 : datasetInfo.total_images < 200 ? 32 : 64;
  } else if (intent.priority === 'memory_efficiency') {
    rank = datasetInfo.total_images < 50 ? 8 : 16;
  } else {
    rank = datasetInfo.total_images < 100 ? 16 : 32;
  }

  return {
    rank,
    alpha: rank // Alpha typically matches rank
  };
}

/**
 * Generate advisor messages for unified memory systems
 */
export function handleUnifiedMemory(profile: SystemProfile): AdvisorMessage[] {
  if (!profile.gpu.isUnifiedMemory) return [];

  const totalMemoryGB = profile.memory.unifiedMemory || profile.memory.totalRAM;
  const workerConfig = calculateWorkers(profile);

  const messages: AdvisorMessage[] = [
    {
      type: 'info',
      title: 'Unified Memory Detected',
      message: `Your system uses unified memory (${totalMemoryGB}GB shared between CPU and GPU). VRAM and RAM share the same pool, so memory management is critical.`
    },
    {
      type: 'tip',
      title: 'CPU Offloading Disabled',
      message: 'low_vram mode is disabled for unified memory systems. Moving tensors to "CPU" doesn\'t save memory since CPU and GPU share the same physical RAM - it only adds overhead.'
    },
    {
      type: 'tip',
      title: 'Worker Count Optimized',
      message: `Using ${workerConfig.num_workers} DataLoader workers instead of 8. Each worker uses ~8GB of memory. With ${workerConfig.num_workers} workers, you save ${(8 - workerConfig.num_workers) * 8}GB for larger batch sizes, which improves training stability.`
    },
    {
      type: 'info',
      title: 'Batch Size vs Workers Trade-off',
      message: 'Larger batch sizes provide better gradient estimates and more stable training. Workers are mostly idle when latents are cached, so we prioritize batch size over worker count.'
    },
    {
      type: 'tip',
      title: 'Unified Memory Optimization',
      message: 'GPU prefetching provides less benefit on unified memory since there\'s no discrete GPU transfer. In-memory caching is always optimal for your setup.'
    }
  ];

  // Add warning if memory is tight
  if (totalMemoryGB < 64) {
    messages.push({
      type: 'warning',
      title: 'Limited Memory',
      message: `With ${totalMemoryGB}GB unified memory, you may experience OOM errors at higher resolutions. Consider using 768px or lower resolution, or enabling gradient checkpointing.`
    });
  }

  return messages;
}

/**
 * Calculate low_vram setting
 *
 * low_vram offloads model components to CPU when not in use.
 * This is ONLY useful for discrete GPU systems with very limited VRAM.
 * For unified memory systems, it's counterproductive (CPU and GPU share same RAM).
 */
export function calculateLowVRAM(profile: SystemProfile): boolean {
  if (profile.gpu.isUnifiedMemory) {
    // Unified memory: NEVER use low_vram - it's pointless and adds overhead
    // CPU and GPU share the same physical memory
    return false;
  }

  // Discrete GPU: Only enable for very low VRAM situations
  // This is a last resort as it significantly slows training
  const vramGB = profile.gpu.vramGB;
  if (vramGB < 8) {
    return true; // <8GB VRAM - need aggressive memory saving
  }

  return false;
}

/**
 * Calculate optimal learning rate scheduler
 *
 * Different schedulers work better for different scenarios:
 * - constant: Simple, good for fine-tuning
 * - cosine: Smooth decay, great for quality
 * - linear: Gradual decay, balanced
 * - cosine_with_restarts: Helps escape local minima
 */
export interface LRSchedulerConfig {
  lr_scheduler: string;
  lr_scheduler_params: Record<string, unknown>;
}

export function calculateLRScheduler(
  intent: UserIntent,
  steps: number
): LRSchedulerConfig {
  if (intent.priority === 'speed') {
    // Fast training: constant LR is simplest
    return {
      lr_scheduler: 'constant',
      lr_scheduler_params: {}
    };
  }

  if (intent.priority === 'quality') {
    // Quality: cosine annealing with warmup
    const warmupSteps = Math.min(100, Math.floor(steps * 0.1));
    return {
      lr_scheduler: 'cosine_with_restarts',
      lr_scheduler_params: {
        num_cycles: 3,
        warmup_steps: warmupSteps,
        min_lr_ratio: 0.1
      }
    };
  }

  // Balanced: linear decay with warmup
  const warmupSteps = Math.min(50, Math.floor(steps * 0.05));
  return {
    lr_scheduler: 'linear',
    lr_scheduler_params: {
      warmup_steps: warmupSteps
    }
  };
}

/**
 * Calculate optimal noise scheduler
 *
 * - ddpm: Default, stable training
 * - euler: Faster convergence, good for fine-tuning
 * - lms: Linear multi-step, memory efficient
 */
export function calculateNoiseScheduler(
  intent: UserIntent,
  modelArch: string
): string {
  // Modern models (Flux, SD3) work best with specific schedulers
  if (modelArch === 'flux' || modelArch === 'flux_kontext') {
    return 'euler'; // Flux uses flow matching
  }
  if (modelArch === 'sd3') {
    return 'euler';
  }

  // For quality priority, use DDPM (most stable)
  if (intent.priority === 'quality') {
    return 'ddpm';
  }

  // For speed, euler converges faster
  if (intent.priority === 'speed') {
    return 'euler';
  }

  return 'ddpm'; // Default stable choice
}

/**
 * Calculate optimal loss target
 *
 * - noise: Predict the noise (default, stable)
 * - source: Predict the clean image (alternative approach)
 * - differential_noise: Predict difference (advanced)
 */
export function calculateLossTarget(
  intent: UserIntent,
  modelArch: string
): string {
  // Some architectures have preferred targets
  if (modelArch === 'flux' || modelArch === 'sd3') {
    return 'noise'; // Flow models should predict noise
  }

  // For quality training, noise prediction is most stable
  return 'noise';
}

/**
 * Calculate CFG (Classifier-Free Guidance) training settings
 *
 * CFG training can improve quality but adds complexity
 */
export interface CFGConfig {
  do_cfg: boolean;
  cfg_scale: number;
}

export function calculateCFGTraining(
  intent: UserIntent
): CFGConfig {
  // CFG training is advanced - only enable for quality priority + advanced users
  if (intent.priority === 'quality' && intent.experienceLevel === 'advanced') {
    return {
      do_cfg: true,
      cfg_scale: 3.0 // Moderate guidance during training
    };
  }

  return {
    do_cfg: false,
    cfg_scale: 1.0
  };
}

/**
 * Calculate noise offset settings
 *
 * Noise offset helps with dark/bright images training
 */
export interface NoiseOffsetConfig {
  noise_offset: number;
  min_snr_gamma?: number;
}

export function calculateNoiseOffset(
  intent: UserIntent,
  datasetInfo: DatasetInfo
): NoiseOffsetConfig {
  // For quality, enable small noise offset to help with brightness range
  if (intent.priority === 'quality') {
    return {
      noise_offset: 0.05, // Small offset improves dynamic range
      min_snr_gamma: 5.0 // SNR weighting for stable training
    };
  }

  // For speed/efficiency, skip the overhead
  return {
    noise_offset: 0.0,
    min_snr_gamma: undefined
  };
}

/**
 * Calculate timestep range for training
 *
 * Limiting timestep range can focus training on specific noise levels
 */
export interface TimestepConfig {
  min_denoising_steps: number;
  max_denoising_steps: number;
}

export function calculateTimestepRange(
  intent: UserIntent
): TimestepConfig {
  // Full range is usually best for general training
  return {
    min_denoising_steps: 0,
    max_denoising_steps: 999
  };
  // Advanced users might want to focus on specific ranges, but defaults should be full
}

/**
 * Calculate data augmentation settings
 */
export interface AugmentationDefaults {
  flip_x: boolean;
  flip_y: boolean;
  random_crop: boolean;
  random_scale: boolean;
}

export function calculateAugmentations(
  intent: UserIntent,
  datasetInfo: DatasetInfo
): AugmentationDefaults {
  // Horizontal flip is generally safe for styles/objects
  const flipX = intent.trainingType !== 'person'; // Don't flip faces

  // Vertical flip rarely makes sense
  const flipY = false;

  // Random crop/scale can help with generalization but risks quality
  const randomCrop = intent.priority === 'quality' && datasetInfo.total_images < 100;
  const randomScale = false; // Usually not helpful for LoRA

  return {
    flip_x: flipX,
    flip_y: flipY,
    random_crop: randomCrop,
    random_scale: randomScale
  };
}

/**
 * Calculate logging defaults
 */
export interface LoggingDefaults {
  use_wandb: boolean;
  log_every: number;
  verbose: boolean;
}

export function calculateLoggingDefaults(
  intent: UserIntent
): LoggingDefaults {
  return {
    use_wandb: false, // User needs to set up W&B first
    log_every: intent.priority === 'quality' ? 50 : 100,
    verbose: intent.experienceLevel === 'advanced'
  };
}

/**
 * Calculate sample/preview defaults
 */
export interface SampleDefaults {
  sampler: string;
  sample_steps: number;
  guidance_scale: number;
  seed: number;
}

export function calculateSampleDefaults(
  modelArch: string
): SampleDefaults {
  // Different models have different optimal preview settings
  if (modelArch === 'flux' || modelArch === 'flux_kontext') {
    return {
      sampler: 'euler',
      sample_steps: 20,
      guidance_scale: 3.5, // Flux uses lower guidance
      seed: 42
    };
  }

  if (modelArch === 'sd3') {
    return {
      sampler: 'euler',
      sample_steps: 28,
      guidance_scale: 7.0,
      seed: 42
    };
  }

  // SDXL and others
  return {
    sampler: 'ddpm',
    sample_steps: 20,
    guidance_scale: 7.0,
    seed: 42
  };
}

/**
 * Generate all smart defaults for a configuration
 */
export function generateSmartDefaults(
  profile: SystemProfile,
  intent: UserIntent,
  datasetInfo: DatasetInfo,
  resolution: number,
  targetType: 'lora' | 'lokr' | 'full',
  modelArch: string
) {
  const workerConfig = calculateWorkers(profile);
  const batchConfig = calculateBatchDefaults(profile, resolution, intent, modelArch, workerConfig.num_workers);
  const cacheConfig = calculateCachingStrategy(profile, datasetInfo);
  const prefetchBatches = calculatePrefetching(profile, intent);
  const learningRate = calculateLearningRate(intent, targetType);
  const steps = calculateSteps(datasetInfo, intent);
  const loraConfig = calculateLoraRank(intent, datasetInfo);
  const lowVRAM = calculateLowVRAM(profile);

  // Advanced training options
  const lrSchedulerConfig = calculateLRScheduler(intent, steps);
  const noiseScheduler = calculateNoiseScheduler(intent, modelArch);
  const lossTarget = calculateLossTarget(intent, modelArch);
  const cfgConfig = calculateCFGTraining(intent);
  const noiseOffsetConfig = calculateNoiseOffset(intent, datasetInfo);
  const timestepConfig = calculateTimestepRange(intent);
  const augmentationConfig = calculateAugmentations(intent, datasetInfo);
  const loggingConfig = calculateLoggingDefaults(intent);
  const sampleConfig = calculateSampleDefaults(modelArch);

  return {
    train: {
      ...batchConfig,
      lr: learningRate,
      ...lrSchedulerConfig,
      steps,
      gradient_accumulation_steps: 1,
      optimizer: 'adamw8bit',
      dtype: profile.gpu.vramGB >= 16 ? 'bf16' : 'fp16',
      gradient_checkpointing: profile.gpu.vramGB < 16,
      noise_scheduler: noiseScheduler,
      loss_target: lossTarget,
      ...cfgConfig,
      ...noiseOffsetConfig,
      ...timestepConfig
    },
    model: {
      low_vram: lowVRAM
    },
    dataset: {
      ...cacheConfig,
      gpu_prefetch_batches: prefetchBatches,
      ...workerConfig,
      ...augmentationConfig,
      resolution: [resolution, resolution]
    },
    network: {
      type: targetType,
      linear: loraConfig.rank,
      linear_alpha: loraConfig.alpha
    },
    sample: sampleConfig,
    logging: loggingConfig,
    advisorMessages: [
      ...handleUnifiedMemory(profile),
      ...generateAdvancedTrainingMessages(intent, lrSchedulerConfig, noiseScheduler, cfgConfig, noiseOffsetConfig, augmentationConfig)
    ]
  };
}

/**
 * Generate advisor messages for advanced training options
 */
export function generateAdvancedTrainingMessages(
  intent: UserIntent,
  lrScheduler: LRSchedulerConfig,
  noiseScheduler: string,
  cfgConfig: CFGConfig,
  noiseOffset: NoiseOffsetConfig,
  augmentations: AugmentationDefaults
): AdvisorMessage[] {
  const messages: AdvisorMessage[] = [];

  // LR Scheduler explanation
  if (lrScheduler.lr_scheduler === 'cosine_with_restarts') {
    messages.push({
      type: 'tip',
      title: 'Cosine LR Schedule with Restarts',
      message: 'Using cosine annealing with periodic restarts. This helps the model escape local minima and often produces better quality results. The learning rate will cycle through warm restarts to explore different optima.'
    });
  } else if (lrScheduler.lr_scheduler === 'linear') {
    messages.push({
      type: 'info',
      title: 'Linear LR Decay',
      message: 'Using linear learning rate decay. The learning rate will gradually decrease throughout training, providing stable convergence with a warmup period at the start.'
    });
  } else if (lrScheduler.lr_scheduler === 'constant') {
    messages.push({
      type: 'info',
      title: 'Constant Learning Rate',
      message: 'Using constant learning rate for simplicity. This is fastest but may not produce optimal results for longer training runs.'
    });
  }

  // Noise scheduler explanation
  if (noiseScheduler === 'euler') {
    messages.push({
      type: 'tip',
      title: 'Euler Noise Scheduler',
      message: 'Using Euler scheduler which provides faster convergence compared to DDPM. Particularly effective for flow-matching models like Flux and SD3.'
    });
  }

  // CFG training
  if (cfgConfig.do_cfg) {
    messages.push({
      type: 'tip',
      title: 'Classifier-Free Guidance Training',
      message: `CFG training enabled with scale ${cfgConfig.cfg_scale}. This teaches the model to respond to guidance during inference, potentially improving quality but increasing training complexity.`
    });
  }

  // Noise offset
  if (noiseOffset.noise_offset > 0) {
    messages.push({
      type: 'tip',
      title: 'Noise Offset Enabled',
      message: `Noise offset of ${noiseOffset.noise_offset} will help the model better handle dark and bright regions in images. This improves dynamic range preservation.`
    });
  }

  // SNR gamma
  if (noiseOffset.min_snr_gamma) {
    messages.push({
      type: 'info',
      title: 'SNR Loss Weighting',
      message: `Min SNR gamma of ${noiseOffset.min_snr_gamma} provides more balanced training across different noise levels, reducing the over-emphasis on high-noise timesteps.`
    });
  }

  // Augmentations
  if (augmentations.flip_x) {
    messages.push({
      type: 'info',
      title: 'Horizontal Flip Enabled',
      message: 'Horizontal flipping will double your effective dataset size. Disabled for person training to preserve facial features.'
    });
  }

  if (augmentations.random_crop) {
    messages.push({
      type: 'tip',
      title: 'Random Crop Enabled',
      message: 'Random cropping helps the model learn from different image regions, improving generalization especially for small datasets.'
    });
  }

  // Experience-based tips
  if (intent.experienceLevel === 'beginner') {
    messages.push({
      type: 'info',
      title: 'Smart Defaults Applied',
      message: 'Advanced training parameters have been set to optimal values for your setup. You can adjust these settings as you gain experience, but the defaults should work well.'
    });
  } else if (intent.experienceLevel === 'advanced') {
    messages.push({
      type: 'tip',
      title: 'Advanced Controls Available',
      message: 'Fine-tune noise scheduling, timestep ranges, and loss functions to optimize training for your specific use case. Consider experimenting with different LR schedulers for better convergence.'
    });
  }

  return messages;
}

/**
 * Memory cost breakdown for regularization options
 */
export interface RegularizationMemoryCost {
  ema: number;  // GB for EMA shadow weights
  diffOutputPreservation: number;  // GB for preservation loss computation
  gradientCheckpointingSavings: number;  // GB saved (negative = savings)
}

/**
 * Calculate memory costs for regularization options
 */
export function calculateRegularizationMemoryCost(
  modelArch: string,
  useEma: boolean,
  useDiffOutputPreservation: boolean,
  gradientCheckpointing: boolean
): RegularizationMemoryCost {
  // EMA memory = copy of all trainable weights (model weights without optimizer states)
  // For LoRA: ~1-2GB. For full model: much larger
  const modelWeightsGB: Record<string, number> = {
    sd1: 4,
    sd15: 4,
    sd2: 5,
    sdxl: 7,
    sd3: 8,
    flux: 12,
    flex1: 10,
    flex2: 10,
    chroma: 10,
    lumina2: 11,
    hidream: 12,
    omnigen2: 12,
    qwen_image: 14
  };

  const baseModelWeights = modelWeightsGB[modelArch] || 10;

  // For LoRA training, EMA only copies the LoRA weights (small)
  // For full model training, EMA copies entire model
  const emaLoRA = 1; // ~1GB for LoRA EMA
  const emaFullModel = baseModelWeights; // Full model weights for full fine-tuning

  // Diff output preservation requires storing original model outputs
  // during forward pass (~2-4GB depending on resolution and batch size)
  const diffPreservationBase = 3; // ~3GB average

  return {
    ema: useEma ? emaLoRA : 0,
    diffOutputPreservation: useDiffOutputPreservation ? diffPreservationBase : 0,
    gradientCheckpointingSavings: gradientCheckpointing ? -baseModelWeights * 0.3 : 0 // Saves ~30% of activations
  };
}

/**
 * Calculate VRAM/Memory usage estimate
 *
 * For discrete GPUs: Returns VRAM usage
 * For unified memory: Returns total system memory usage (more comprehensive)
 */
export function estimateVRAMUsage(
  modelArch: string,
  resolution: number,
  batchSize: number,
  quantization: string | null,
  gradientCheckpointing: boolean,
  isUnifiedMemory: boolean = false,
  numWorkers: number = 4,
  useEma: boolean = false,
  useDiffOutputPreservation: boolean = false
): number {
  // Base VRAM for model (rough estimates in GB)
  const modelVRAM: Record<string, number> = {
    sd1: 4,
    sd15: 4,
    sd2: 5,
    sdxl: 7,
    sd3: 8,
    flux: 12,
    flux_kontext: 12,
    flex1: 10,
    flex2: 10,
    chroma: 10,
    lumina2: 11,
    hidream: 12,
    omnigen2: 12,
    qwen_image: 14
  };

  let baseVRAM = modelVRAM[modelArch] || 10;

  // Adjust for quantization
  if (quantization === '4bit') {
    baseVRAM *= 0.5;
  } else if (quantization === '8bit') {
    baseVRAM *= 0.75;
  }

  // Calculate regularization costs
  const regCosts = calculateRegularizationMemoryCost(
    modelArch,
    useEma,
    useDiffOutputPreservation,
    gradientCheckpointing
  );

  // Apply gradient checkpointing savings to base VRAM (reduces activation memory)
  if (gradientCheckpointing) {
    baseVRAM *= 0.7;
  }

  // Add batch size and resolution overhead
  const resolutionFactor = (resolution * resolution) / (1024 * 1024);
  const batchOverhead = batchSize * resolutionFactor * 4; // ~4GB per batch at 1024px (activations + gradients)

  let totalMemory = baseVRAM + batchOverhead;

  // Add regularization costs
  totalMemory += regCosts.ema;
  totalMemory += regCosts.diffOutputPreservation;

  if (isUnifiedMemory) {
    // For unified memory, add additional overheads that share the same memory pool
    const optimizerStates = baseVRAM * 2; // AdamW momentum + variance (2x model weights)
    const workerOverhead = numWorkers * 8; // Each worker forks Python process (~8GB each with large models)
    const osReserve = 8; // OS and system processes

    totalMemory += optimizerStates + workerOverhead + osReserve;
  }

  return totalMemory;
}

/**
 * Estimate training time
 */
export function estimateTrainingTime(
  steps: number,
  batchSize: number,
  resolution: number,
  profile: SystemProfile
): { stepTime: number; totalMinutes: number } {
  // Base step time in seconds (rough estimate)
  let stepTime = 1.0;

  // Adjust for resolution
  stepTime *= (resolution * resolution) / (1024 * 1024);

  // Adjust for batch size
  stepTime *= batchSize;

  // Adjust for GPU type
  if (profile.gpu.type === 'apple_silicon') {
    stepTime *= 1.5; // Apple Silicon generally slower for ML
  } else if (profile.gpu.vramGB >= 24) {
    stepTime *= 0.8; // High-end GPU
  }

  const totalMinutes = (steps * stepTime) / 60;

  return { stepTime, totalMinutes };
}

/**
 * Estimate disk space needed
 */
export function estimateDiskSpace(
  datasetInfo: DatasetInfo,
  steps: number,
  saveEvery: number,
  cacheToDisK: boolean
): number {
  let totalGB = 0;

  // Cache space
  if (cacheToDisK) {
    totalGB += (datasetInfo.total_images * 6) / 1024; // ~6MB per image
  }

  // Checkpoint space
  const numCheckpoints = Math.ceil(steps / saveEvery);
  totalGB += numCheckpoints * 0.5; // ~500MB per checkpoint for LoRA

  return totalGB;
}
