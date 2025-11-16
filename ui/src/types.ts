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
  alpha?: number;
  dropout?: number;
  transformer_only?: boolean;
  lokr_full_rank: boolean;
  lokr_factor: number;
  network_kwargs: {
    ignore_if_contains: string[];
    only_if_contains?: string[];
  };
}

export interface SaveConfig {
  dtype: string;
  save_every: number;
  max_step_saves_to_keep: number;
  save_format: string;
  push_to_hub: boolean;
  hf_repo_id?: string;
  hf_private?: boolean;
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
  cache_latents?: boolean;
  cache_latents_to_disk?: boolean;
  cache_text_embeddings?: boolean;
  cache_clip_vision_to_disk?: boolean;
  resolution: number[];
  controls: string[];
  control_path?: string | null;
  num_frames: number;
  shrink_video_to_frames: boolean;
  do_i2v: boolean;
  flip_x: boolean;
  flip_y: boolean;
  random_crop?: boolean;
  random_scale?: boolean;
  scale?: number;
  square_crop?: boolean;
  num_repeats?: number;
  keep_tokens?: number;
  token_dropout_rate?: number;
  augmentations?: AugmentationConfig[];
  control_path_1?: string | null;
  control_path_2?: string | null;
  control_path_3?: string | null;
  // Advanced dataset options
  buckets?: boolean;
  bucket_tolerance?: number;
  enable_ar_bucket?: boolean;
  inpaint_path?: string | null;
  unconditional_path?: string | null;
  clip_image_path?: string | null;
  clip_image_from_same_folder?: boolean;
  alpha_mask?: boolean;
  invert_mask?: boolean;
  prior_reg?: boolean;
  loss_multiplier?: number;
  prefetch_factor?: number;
  persistent_workers?: boolean;
  num_workers?: number;
  gpu_prefetch_batches?: number;
  standardize_images?: boolean;
}

export interface AugmentationConfig {
  name: string;
  params?: Record<string, unknown>;
}

export interface EMAConfig {
  use_ema: boolean;
  ema_decay: number;
}

export interface TrainConfig {
  batch_size: number;
  auto_scale_batch_size?: boolean;
  min_batch_size?: number;
  max_batch_size?: number;
  batch_size_warmup_steps?: number;
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
  unet_lr?: number;
  text_encoder_lr?: number;
  lr_scheduler?: string;
  lr_scheduler_params?: Record<string, unknown>;
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
  loss_target?: string;
  do_differential_guidance?: boolean;
  differential_guidance_scale?: number;
  do_cfg?: boolean;
  cfg_scale?: number;
  noise_offset?: number;
  min_snr_gamma?: number;
  snr_gamma?: number;
  min_denoising_steps?: number;
  max_denoising_steps?: number;
  max_grad_norm?: number;
  prompt_dropout_prob?: number;
  // Advanced preservation
  blank_prompt_preservation?: boolean;
  blank_prompt_preservation_multiplier?: number;
  inverted_mask_prior?: boolean;
  inverted_mask_prior_multiplier?: number;
  do_prior_divergence?: boolean;
  // Differential guidance
  do_differential_guidance?: boolean;
  differential_guidance_scale?: number;
  // Training control
  train_refiner?: boolean;
  skip_first_sample?: boolean;
  force_first_sample?: boolean;
  disable_sampling?: boolean;
  // Advanced optimization
  weight_jitter?: number;
  train_turbo?: boolean;
  learnable_snr_gos?: boolean;
  correct_pred_norm?: boolean;
  correct_pred_norm_multiplier?: number;
  // Batch control
  batch_size_warmup_steps?: number;
  single_item_batching?: boolean;
  // Additional settings
  negative_prompt?: string;
  unconditional_prompt?: string;
  // Feature extractors
  latent_feature_extractor_path?: string;
  latent_feature_loss_weight?: number;
  diffusion_feature_extractor_path?: string;
  diffusion_feature_extractor_weight?: number;
  // Advanced noise options
  optimal_noise_pairing_samples?: number;
  force_consistent_noise?: boolean;
  blended_blur_noise?: boolean;
  show_turbo_outputs?: boolean;
  free_u?: boolean;
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
  compile?: boolean;
  vae_path?: string;
  assistant_lora_path?: string;
  inference_lora_path?: string;
  text_encoder_bits?: number;
  // Advanced model options
  dtype?: string;
  refiner_name_or_path?: string;
  lora_path?: string;
  attn_masking?: boolean;
  split_model_over_gpus?: boolean;
  split_model_other_module_param_count_scale?: number;
  use_text_encoder_1?: boolean;
  use_text_encoder_2?: boolean;
  experimental_xl?: boolean;
  use_flux_cfg?: boolean;
  // Device-specific options
  vae_device?: string;
  vae_dtype?: string;
  te_device?: string;
  te_dtype?: string;
  unet_path?: string;
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
  guidance_rescale?: number;
  sample_steps: number;
  num_frames: number;
  fps: number;
  format?: string;
  network_multiplier?: number;
  refiner_start_at?: number;
}

export interface SliderConfig {
  guidance_strength?: number;
  anchor_strength?: number;
  positive_prompt?: string;
  negative_prompt?: string;
  target_class?: string;
  anchor_class?: string | null;
}

export interface MonitoringConfig {
  enabled?: boolean;
  sample_interval_seconds?: number;
  track_per_process?: boolean;
  analyze_on_complete?: boolean;
  memory_warning_threshold?: number;
  memory_critical_threshold?: number;
}

export interface LoggingConfig {
  use_wandb?: boolean;
  project_name?: string;
  run_name?: string;
  log_every?: number;
  verbose?: boolean;
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
  model: ModelConfig;
  sample: SampleConfig;
  monitoring?: MonitoringConfig;
  logging?: LoggingConfig;
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
