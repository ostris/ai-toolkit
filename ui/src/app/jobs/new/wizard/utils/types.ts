/**
 * Type definitions for the Comprehensive Guided Config Wizard
 */

// System Profile Types
export interface SystemProfile {
  gpu: {
    type: 'nvidia' | 'amd' | 'unified_memory' | 'cpu_only';
    name: string;
    vramGB: number;
    driverVersion?: string;
    isUnifiedMemory: boolean;
  };
  memory: {
    totalRAM: number;
    availableRAM: number;
    unifiedMemory?: number; // Apple Silicon, DGX Spark, NVIDIA Grace, etc.
  };
  storage: {
    type: 'hdd' | 'ssd' | 'nvme';
    availableSpaceGB: number;
  };
  cpu: {
    cores: number;
    name: string;
  };
}

// User Intent Types
export interface UserIntent {
  trainingType: 'person' | 'style' | 'object' | 'concept' | 'other';
  priority: 'quality' | 'speed' | 'memory_efficiency';
  experienceLevel: 'beginner' | 'intermediate' | 'advanced';
}

// Dataset Info Types
export interface DatasetInfo {
  total_images: number;
  most_common_resolution: [number, number];
  resolutions: Record<string, number>;
  has_captions: boolean;
  caption_ext: string;
  formats: Record<string, number>;
  average_file_size_mb?: number;
}

// Configuration Types
export interface BatchConfig {
  batch_size: number;
  auto_scale_batch_size: boolean;
  min_batch_size: number;
  max_batch_size: number;
  batch_size_warmup_steps: number;
}

export interface CacheConfig {
  cache_latents: boolean;
  cache_latents_to_disk: boolean;
  reason: string;
}

export interface WorkerConfig {
  num_workers: number;
  persistent_workers: boolean;
}

// Advisor Types
export interface AdvisorMessage {
  type: 'info' | 'tip' | 'warning' | 'error';
  title: string;
  message: string;
  field?: string;
  autoFix?: () => void;
}

export interface ValidationResult {
  errors: AdvisorMessage[];
  warnings: AdvisorMessage[];
  suggestions: AdvisorMessage[];
}

// Performance Prediction Types
export interface PerformancePrediction {
  estimatedVRAM: string;
  estimatedStepTime: string;
  totalTrainingTime: string;
  diskSpaceNeeded: string;
  memoryUsage: string;
}

// Wizard State
export interface WizardState {
  currentStep: number;
  systemProfile: SystemProfile | null;
  userIntent: UserIntent | null;
  datasetInfo: DatasetInfo | null;
  advisorMessages: AdvisorMessage[];
  validationResults: ValidationResult;
  isPreflightComplete: boolean;
}

// Config Summary for Header
export interface ConfigSummary {
  model: string;
  resolution: string;
  steps: number | null;
  estimatedVRAM: string;
  warnings: string[];
}

// Wizard Step Definition
export interface WizardStep {
  id: string;
  title: string;
  description: string;
  isOptional?: boolean;
  isAdaptive?: boolean;
}
