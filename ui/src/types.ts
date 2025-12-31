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

export interface CpuInfo {
  name: string;
  cores: number;
  temperature: number;
  totalMemory: number;
  freeMemory: number;
  availableMemory: number;
  currentLoad: number;
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
  conv: number;
  conv_alpha: number;
  lokr_full_rank: boolean;
  lokr_factor: number;
  network_kwargs: {
    ignore_if_contains: string[];
  };
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
  controls: string[];
  control_path?: string | null;
  num_frames: number;
  shrink_video_to_frames: boolean;
  do_i2v: boolean;
  flip_x: boolean;
  flip_y: boolean;
  control_path_1?: string | null;
  control_path_2?: string | null;
  control_path_3?: string | null;
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
  cache_text_embeddings: boolean;
  optimizer_params: {
    weight_decay: number;
  };
  skip_first_sample: boolean;
  force_first_sample: boolean;
  disable_sampling: boolean;
  diff_output_preservation: boolean;
  diff_output_preservation_multiplier: number;
  diff_output_preservation_class: string;
  blank_prompt_preservation?: boolean;
  blank_prompt_preservation_multiplier?: number;
  switch_boundary_every: number;
  loss_type: 'mse' | 'mae' | 'wavelet' | 'stepped';
  do_differential_guidance?: boolean;
  differential_guidance_scale?: number;
}

export interface QuantizeKwargsConfig {
  exclude: string[];
}

export interface ModelConfig {
  name_or_path: string;
  quantize: boolean;
  quantize_te: boolean;
  qtype: string;
  qtype_te: string;
  quantize_kwargs?: QuantizeKwargsConfig;
  arch: string;
  low_vram: boolean;
  model_kwargs: { [key: string]: any };
  layer_offloading?: boolean;
  layer_offloading_transformer_percent?: number;
  layer_offloading_text_encoder_percent?: number;
  assistant_lora_path?: string;
}

export interface SampleItem {
  prompt: string;
  width?: number;
  height?: number;
  neg?: string;
  seed?: number;
  guidance_scale?: number;
  sample_steps?: number;
  fps?: number;
  num_frames?: number;
  ctrl_img?: string | null;
  ctrl_idx?: number;
  network_multiplier?: number;
  ctrl_img_1?: string | null;
  ctrl_img_2?: string | null;
  ctrl_img_3?: string | null;
}

export interface SampleConfig {
  sampler: string;
  sample_every: number;
  width: number;
  height: number;
  prompts?: string[];
  samples: SampleItem[];
  neg: string;
  seed: number;
  walk_seed: boolean;
  guidance_scale: number;
  sample_steps: number;
  num_frames: number;
  fps: number;
}

export interface LoggingConfig {
  log_every: number;
  use_ui_logger: boolean;
}

export interface SliderConfig {
  guidance_strength?: number;
  anchor_strength?: number;
  positive_prompt?: string;
  negative_prompt?: string;
  target_class?: string;
  anchor_class?: string | null;
}

export interface ProcessConfig {
  type: string;
  sqlite_db_path?: string;
  training_folder: string;
  performance_log_every: number;
  trigger_word: string | null;
  device: string;
  network?: NetworkConfig;
  slider?: SliderConfig;
  save: SaveConfig;
  datasets: DatasetConfig[];
  train: TrainConfig;
  logging: LoggingConfig;
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

export interface ConfigDoc {
  title: string | React.ReactNode;
  description: React.ReactNode;
}

export interface SelectOption {
  readonly value: string;
  readonly label: string;
}
export interface GroupedSelectOption {
  readonly label: string;
  readonly options: SelectOption[];
}

export type JobStatus = 'queued' | 'running' | 'stopping' | 'stopped' | 'completed' | 'error';
