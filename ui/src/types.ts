/**
 * GPU API response
 */

export interface GpuUtilization {
  gpu: number;
  memory: number;
}

export interface GpuMemory {
  total: number;
  free: number;
  used: number;
}

export interface GpuPower {
  draw: number;
  limit: number;
}

export interface GpuClocks {
  graphics: number;
  memory: number;
}

export interface GpuFan {
  speed: number;
}

export interface GpuInfo {
  index: number;
  name: string;
  driverVersion: string;
  temperature: number;
  utilization: GpuUtilization;
  memory: GpuMemory;
  power: GpuPower;
  clocks: GpuClocks;
  fan: GpuFan;
}

export interface GPUApiResponse {
  hasNvidiaSmi: boolean;
  gpus: GpuInfo[];
  error?: string;
}

/**
 * Training configuration
 */

export interface NetworkConfig {
  type: string;
  linear: number;
  linear_alpha: number;
  lokr_full_rank: boolean;
  lokr_factor: number;
  network_kwargs: {
    ignore_if_contains: string[];
  }
}

export interface SaveConfig {
  dtype: string;
  save_every: number;
  max_step_saves_to_keep: number;
  save_format: string;
  push_to_hub: boolean;
}

export interface DatasetConfig {
  folder_path: string;
  mask_path: string | null;
  mask_min_value: number;
  default_caption: string;
  caption_ext: string;
  caption_dropout_rate: number;
  shuffle_tokens?: boolean;
  is_reg: boolean;
  network_weight: number;
  cache_latents_to_disk?: boolean;
  resolution: number[];
}

export interface EMAConfig {
  use_ema: boolean;
  ema_decay: number;
}

export interface TrainConfig {
  batch_size: number;
  bypass_guidance_embedding?: boolean;
  steps: number;
  gradient_accumulation: number;
  train_unet: boolean;
  train_text_encoder: boolean;
  gradient_checkpointing: boolean;
  noise_scheduler: string;
  timestep_type: string;
  content_or_style: string;
  optimizer: string;
  lr: number;
  ema_config?: EMAConfig;
  dtype: string;
  unload_text_encoder: boolean;
  optimizer_params: {
    weight_decay: number;
  };
  diff_output_preservation: boolean;
  diff_output_preservation_multiplier: number;
  diff_output_preservation_class: string;
}

export interface QuantizeKwargsConfig {
  exclude: string[];
}

export interface ModelConfig {
  name_or_path: string;
  quantize: boolean;
  quantize_te: boolean;
  quantize_kwargs?: QuantizeKwargsConfig;
  arch: string;
  low_vram: boolean;
}

export interface SampleConfig {
  sampler: string;
  sample_every: number;
  width: number;
  height: number;
  prompts: string[];
  neg: string;
  seed: number;
  walk_seed: boolean;
  guidance_scale: number;
  sample_steps: number;
  num_frames: number;
  fps: number;
}

export interface ProcessConfig {
  type: 'ui_trainer';
  sqlite_db_path?: string;
  training_folder: string;
  performance_log_every: number;
  trigger_word: string | null;
  device: string;
  network?: NetworkConfig;
  save: SaveConfig;
  datasets: DatasetConfig[];
  train: TrainConfig;
  model: ModelConfig;
  sample: SampleConfig;
}

export interface ConfigObject {
  name: string;
  process: ProcessConfig[];
}

export interface MetaConfig {
  name: string;
  version: string;
}

export interface JobConfig {
  job: string;
  config: ConfigObject;
  meta: MetaConfig;
}
