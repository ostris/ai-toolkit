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
 */
export function calculateBatchDefaults(
  profile: SystemProfile,
  resolution: number,
  intent: UserIntent
): BatchConfig {
  // For unified memory (Apple Silicon), use 70% of total memory
  const vramGB = profile.gpu.isUnifiedMemory
    ? (profile.memory.unifiedMemory || profile.memory.totalRAM) * 0.7
    : profile.gpu.vramGB;

  let initialBatch: number;
  let maxBatch: number;

  // Calculate based on resolution and available VRAM
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
    initialBatch = Math.min(2, Math.max(1, Math.floor(vramGB / 8)));
    maxBatch = Math.min(4, initialBatch * 2);
  } else {
    initialBatch = Math.min(2, Math.max(1, Math.floor(vramGB / 10)));
    maxBatch = Math.min(4, initialBatch * 2);
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
 */
export function calculateWorkers(profile: SystemProfile): WorkerConfig {
  const baseWorkers = Math.floor(profile.memory.totalRAM / 8);
  const numWorkers = Math.min(baseWorkers, profile.cpu.cores, 8);

  return {
    num_workers: Math.max(0, numWorkers),
    persistent_workers: numWorkers > 0 // TODO #4 - Keep workers alive
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

  return [
    {
      type: 'info',
      title: 'Unified Memory Detected',
      message: `Your system uses unified memory (${profile.memory.unifiedMemory || profile.memory.totalRAM}GB shared between CPU and GPU). VRAM and RAM share the same pool, so batch size calculations account for this.`
    },
    {
      type: 'tip',
      title: 'Unified Memory Optimization',
      message: 'GPU prefetching provides less benefit on unified memory since there\'s no discrete GPU transfer. In-memory caching is always optimal for your setup.'
    }
  ];
}

/**
 * Generate all smart defaults for a configuration
 */
export function generateSmartDefaults(
  profile: SystemProfile,
  intent: UserIntent,
  datasetInfo: DatasetInfo,
  resolution: number,
  targetType: 'lora' | 'lokr' | 'full' = 'lora'
) {
  const batchConfig = calculateBatchDefaults(profile, resolution, intent);
  const cacheConfig = calculateCachingStrategy(profile, datasetInfo);
  const prefetchBatches = calculatePrefetching(profile, intent);
  const workerConfig = calculateWorkers(profile);
  const learningRate = calculateLearningRate(intent, targetType);
  const steps = calculateSteps(datasetInfo, intent);
  const loraConfig = calculateLoraRank(intent, datasetInfo);

  return {
    train: {
      ...batchConfig,
      lr: learningRate,
      steps,
      gradient_accumulation_steps: 1,
      optimizer: 'adamw8bit',
      dtype: profile.gpu.vramGB >= 16 ? 'bf16' : 'fp16',
      gradient_checkpointing: profile.gpu.vramGB < 16
    },
    dataset: {
      ...cacheConfig,
      gpu_prefetch_batches: prefetchBatches,
      ...workerConfig,
      resolution: [resolution, resolution]
    },
    network: {
      type: targetType,
      linear: loraConfig.rank,
      linear_alpha: loraConfig.alpha
    },
    advisorMessages: handleUnifiedMemory(profile)
  };
}

/**
 * Calculate VRAM usage estimate
 */
export function estimateVRAMUsage(
  modelArch: string,
  resolution: number,
  batchSize: number,
  quantization: string | null,
  gradientCheckpointing: boolean
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

  // Adjust for gradient checkpointing
  if (gradientCheckpointing) {
    baseVRAM *= 0.7;
  }

  // Add batch size and resolution overhead
  const resolutionFactor = (resolution * resolution) / (1024 * 1024);
  const batchOverhead = batchSize * resolutionFactor * 2; // ~2GB per batch at 1024px

  return baseVRAM + batchOverhead;
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
