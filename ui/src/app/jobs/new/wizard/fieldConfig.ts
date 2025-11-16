/**
 * Field Configuration System for Data-Driven Wizard Rendering
 *
 * This module provides a declarative way to define wizard fields with:
 * - Model-specific visibility (fields only shown for applicable models)
 * - Conditional visibility (fields shown based on other field values)
 * - Automatic section grouping and ordering
 * - Type-safe field rendering
 *
 * HOW TO ADD NEW FIELDS:
 * 1. Add field definition to the `fields` array with appropriate:
 *    - id: The config path (e.g., 'config.process[0].model.some_option')
 *    - label: Human-readable label
 *    - description: Tooltip/help text
 *    - type: 'boolean' | 'number' | 'string' | 'select' | 'slider'
 *    - step: Which wizard step this belongs to
 *    - section: Which section within the step
 *    - applicableModels: (optional) Array of model names that support this field
 *    - showWhen: (optional) Conditional visibility based on another field's value
 *
 * 2. If needed, add a new section to the `sections` array with:
 *    - id: Unique section identifier
 *    - title: Section header text
 *    - step: Which wizard step
 *    - order: Display order within the step
 *    - applicableModels: (optional) Only show section for these models
 *
 * EXAMPLE:
 * ```typescript
 * // Adding a new field
 * {
 *   id: 'config.process[0].model.my_new_option',
 *   label: 'My New Option',
 *   description: 'Enables a cool feature',
 *   type: 'boolean',
 *   defaultValue: false,
 *   step: 'quantization',
 *   section: 'model_quantization',
 *   applicableModels: ['flux', 'sdxl'], // Only show for Flux and SDXL
 *   showWhen: {
 *     field: 'config.process[0].model.quantize',
 *     value: true, // Only show when quantization is enabled
 *   },
 * }
 * ```
 */

export type WizardStepId =
  | 'model'
  | 'quantization'
  | 'target'
  | 'dataset'
  | 'resolution'
  | 'memory'
  | 'optimizer'
  | 'advanced'
  | 'regularization'
  | 'training'
  | 'sampling'
  | 'save'
  | 'logging'
  | 'monitoring'
  | 'review';

export type FieldType = 'boolean' | 'number' | 'string' | 'select' | 'slider';

export interface FieldConfig {
  id: string; // config path: 'config.process[0].model.use_flux_cfg'
  label: string; // display label
  description?: string; // help text/tooltip
  type: FieldType;
  defaultValue?: any;
  step: WizardStepId;
  section?: string; // groups fields under a header
  applicableModels?: string[]; // whitelist; undefined = all models

  // Type-specific
  options?: { value: any; label: string }[]; // select
  min?: number; // number/slider
  max?: number;
  numberStep?: number; // increment for number inputs
  placeholder?: string; // string

  // Conditional visibility
  showWhen?: {
    field: string;
    value: any;
  };
}

export interface SectionConfig {
  id: string;
  title: string;
  description?: string;
  step: WizardStepId;
  order: number; // display order within step
  applicableModels?: string[]; // whitelist; undefined = all models
}

// Helper function to get nested value from object using dot notation path
// Supports array notation like 'config.process[0].model.arch'
export function getNestedValue(obj: any, path: string): any {
  if (!obj || !path) return undefined;

  const parts = path.split(/[.[\]]/).filter(Boolean);
  let current = obj;

  for (const part of parts) {
    if (current === null || current === undefined) {
      return undefined;
    }
    current = current[part];
  }

  return current;
}

// Helper function to set nested value in object using dot notation path
// Returns a new object with the value set (immutable)
export function setNestedValue<T>(obj: T, path: string, value: any): T {
  if (!path) return obj;

  const parts = path.split(/[.[\]]/).filter(Boolean);
  const result = JSON.parse(JSON.stringify(obj)) as T;
  let current: any = result;

  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (current[part] === undefined) {
      // Determine if next part is array index
      const nextPart = parts[i + 1];
      current[part] = /^\d+$/.test(nextPart) ? [] : {};
    }
    current = current[part];
  }

  current[parts[parts.length - 1]] = value;
  return result;
}

// Section definitions organized by wizard step
export const sections: SectionConfig[] = [
  // Quantization step sections
  {
    id: 'model_quantization',
    title: 'Model Quantization',
    description: 'Reduce VRAM usage by quantizing model weights',
    step: 'quantization',
    order: 1,
  },
  {
    id: 'memory_optimization',
    title: 'Memory Optimization',
    description: 'Advanced memory management options',
    step: 'quantization',
    order: 2,
  },
  {
    id: 'model_compilation',
    title: 'Model Compilation',
    description: 'JIT compilation and optimization options',
    step: 'quantization',
    order: 3,
  },
  {
    id: 'model_paths',
    title: 'Model Paths',
    description: 'Custom model and component paths',
    step: 'quantization',
    order: 4,
  },
  {
    id: 'flux_options',
    title: 'Flux-Specific Options',
    step: 'quantization',
    order: 10,
    applicableModels: ['flux', 'flux_kontext', 'flex1', 'flex2', 'chroma'],
  },
  {
    id: 'sdxl_options',
    title: 'SDXL-Specific Options',
    step: 'quantization',
    order: 11,
    applicableModels: ['sdxl'],
  },
  {
    id: 'video_model_options',
    title: 'Video Model Options',
    step: 'quantization',
    order: 12,
    applicableModels: [
      'wan21:1b',
      'wan21:14b',
      'wan21_i2v:14b',
      'wan21_i2v:14b480p',
      'wan22_14b:t2v',
      'wan22_14b_i2v',
      'wan22_5b',
    ],
  },
  {
    id: 'qwen_options',
    title: 'Qwen Model Options',
    step: 'quantization',
    order: 13,
    applicableModels: ['qwen_image', 'qwen_image_edit', 'qwen_image_edit_plus'],
  },
  {
    id: 'hidream_options',
    title: 'HiDream Options',
    step: 'quantization',
    order: 14,
    applicableModels: ['hidream', 'hidream_e1'],
  },

  // Target step sections
  {
    id: 'target_type',
    title: 'Training Target',
    description: 'Select the type of adapter to train',
    step: 'target',
    order: 1,
  },
  {
    id: 'lora_settings',
    title: 'LoRA Configuration',
    description: 'Configure rank and alpha values for LoRA layers',
    step: 'target',
    order: 2,
  },
  {
    id: 'advanced_lora',
    title: 'Advanced LoRA Settings',
    description: 'Fine-tune LoRA behavior',
    step: 'target',
    order: 3,
  },

  // Dataset step sections
  {
    id: 'dataset_path',
    title: 'Dataset Location',
    description: 'Path to your training images',
    step: 'dataset',
    order: 1,
  },
  {
    id: 'captioning',
    title: 'Caption Settings',
    description: 'Configure how captions are used',
    step: 'dataset',
    order: 2,
  },
  {
    id: 'augmentation',
    title: 'Data Augmentation',
    description: 'Augment training data for better generalization',
    step: 'dataset',
    order: 3,
  },
  {
    id: 'video_dataset',
    title: 'Video Dataset Settings',
    step: 'dataset',
    order: 4,
    applicableModels: [
      'wan21:1b',
      'wan21:14b',
      'wan21_i2v:14b',
      'wan21_i2v:14b480p',
      'wan22_14b:t2v',
      'wan22_14b_i2v',
      'wan22_5b',
    ],
  },

  // Training step sections
  {
    id: 'training_duration',
    title: 'Training Duration',
    description: 'Configure how long to train',
    step: 'training',
    order: 1,
  },
  {
    id: 'timestep_config',
    title: 'Timestep Configuration',
    description: 'Control the noise schedule sampling',
    step: 'training',
    order: 2,
  },

  // Sampling step sections
  {
    id: 'sample_generation',
    title: 'Sample Generation',
    description: 'Configure preview image generation during training',
    step: 'sampling',
    order: 1,
  },
  {
    id: 'sample_prompts',
    title: 'Sample Prompts',
    description: 'Prompts to use for generating samples',
    step: 'sampling',
    order: 2,
  },

  // Optimizer step sections
  {
    id: 'learning_rate',
    title: 'Learning Rate',
    description: 'Configure the learning rate and warmup',
    step: 'optimizer',
    order: 1,
  },
  {
    id: 'optimizer_type',
    title: 'Optimizer Selection',
    description: 'Choose the optimization algorithm',
    step: 'optimizer',
    order: 2,
  },
  {
    id: 'lr_scheduler',
    title: 'Learning Rate Scheduler',
    description: 'How the learning rate changes over time',
    step: 'optimizer',
    order: 3,
  },
  {
    id: 'optimizer_advanced',
    title: 'Advanced Optimizer Settings',
    description: 'Fine-tune optimizer behavior',
    step: 'optimizer',
    order: 4,
  },

  // Memory step sections
  {
    id: 'batch_settings',
    title: 'Batch Configuration',
    description: 'Control batch size and gradient accumulation',
    step: 'memory',
    order: 1,
  },
  {
    id: 'caching',
    title: 'Latent Caching',
    description: 'Cache encoded images for faster training',
    step: 'memory',
    order: 2,
  },
  {
    id: 'dataloader',
    title: 'DataLoader Settings',
    description: 'Configure data loading workers',
    step: 'memory',
    order: 3,
  },

  // Advanced step sections
  {
    id: 'noise_scheduler',
    title: 'Noise Scheduler',
    description: 'Configure the diffusion noise schedule',
    step: 'advanced',
    order: 1,
  },
  {
    id: 'loss_function',
    title: 'Loss Function',
    description: 'Configure training loss computation',
    step: 'advanced',
    order: 2,
  },
  {
    id: 'guidance_settings',
    title: 'Guidance Settings',
    description: 'Classifier-free guidance configuration',
    step: 'advanced',
    order: 3,
  },
  {
    id: 'regularization_techniques',
    title: 'Regularization',
    description: 'Prevent overfitting and improve generalization',
    step: 'advanced',
    order: 4,
  },

  // Resolution step sections
  {
    id: 'image_resolution',
    title: 'Training Resolution',
    description: 'Set the resolution for training images',
    step: 'resolution',
    order: 1,
  },
  {
    id: 'bucket_settings',
    title: 'Aspect Ratio Bucketing',
    description: 'Group images by aspect ratio for efficient training',
    step: 'resolution',
    order: 2,
  },

  // Save step sections
  {
    id: 'checkpoint_settings',
    title: 'Checkpoint Saving',
    description: 'Configure when and how to save model checkpoints',
    step: 'save',
    order: 1,
  },
  {
    id: 'save_format',
    title: 'Save Format',
    description: 'Output format and compression settings',
    step: 'save',
    order: 2,
  },

  // Logging step sections
  {
    id: 'wandb_settings',
    title: 'Weights & Biases',
    description: 'Configure experiment tracking with W&B',
    step: 'logging',
    order: 1,
  },
  {
    id: 'local_logging',
    title: 'Local Logging',
    description: 'Configure local log output',
    step: 'logging',
    order: 2,
  },
];

// Field definitions
export const fields: FieldConfig[] = [
  // === Quantization Step Fields ===

  // Universal quantization fields
  {
    id: 'config.process[0].model.quantize',
    label: 'Quantize Transformer',
    description:
      'Quantize transformer weights to reduce VRAM usage. Essential for large models like Flux on consumer GPUs.',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'model_quantization',
  },
  {
    id: 'config.process[0].model.qtype',
    label: 'Quantization Type',
    description:
      'Precision level for quantized weights. Lower bits = less VRAM but potentially lower quality. float8 is recommended for most cases.',
    type: 'select',
    defaultValue: 'qfloat8',
    step: 'quantization',
    section: 'model_quantization',
    options: [
      { value: 'qfloat8', label: 'float8 (default)' },
      { value: 'uint7', label: '7 bit' },
      { value: 'uint6', label: '6 bit' },
      { value: 'uint5', label: '5 bit' },
      { value: 'uint4', label: '4 bit' },
      { value: 'uint3', label: '3 bit' },
      { value: 'uint2', label: '2 bit' },
    ],
    showWhen: {
      field: 'config.process[0].model.quantize',
      value: true,
    },
  },
  {
    id: 'config.process[0].model.quantize_te',
    label: 'Quantize Text Encoder',
    description: 'Quantize text encoder weights to save additional VRAM.',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'model_quantization',
  },
  {
    id: 'config.process[0].model.dtype',
    label: 'Model Data Type',
    description: 'Precision for model computation. BFloat16 recommended for modern GPUs, Float16 for wider compatibility.',
    type: 'select',
    defaultValue: 'bf16',
    step: 'quantization',
    section: 'model_quantization',
    options: [
      { value: 'bf16', label: 'BFloat16 (recommended)' },
      { value: 'float16', label: 'Float16' },
      { value: 'float32', label: 'Float32 (full precision)' },
    ],
  },

  // Memory optimization fields
  {
    id: 'config.process[0].model.layer_offloading',
    label: 'Enable Layer Offloading',
    description:
      'Offload model layers to CPU to save VRAM. Moves layers between CPU and GPU as needed. Enables training larger models with limited VRAM but slower.',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'memory_optimization',
  },
  {
    id: 'config.process[0].model.layer_offloading_transformer_percent',
    label: 'Transformer Offload %',
    description: 'Percentage of transformer layers to offload to CPU (100% = all layers can be offloaded)',
    type: 'slider',
    defaultValue: 1.0,
    step: 'quantization',
    section: 'memory_optimization',
    min: 0,
    max: 1,
    numberStep: 0.1,
    showWhen: {
      field: 'config.process[0].model.layer_offloading',
      value: true,
    },
  },
  {
    id: 'config.process[0].model.layer_offloading_text_encoder_percent',
    label: 'Text Encoder Offload %',
    description: 'Percentage of text encoder layers to offload to CPU',
    type: 'slider',
    defaultValue: 1.0,
    step: 'quantization',
    section: 'memory_optimization',
    min: 0,
    max: 1,
    numberStep: 0.1,
    showWhen: {
      field: 'config.process[0].model.layer_offloading',
      value: true,
    },
  },
  {
    id: 'config.process[0].model.split_model_over_gpus',
    label: 'Split Model Across GPUs',
    description: 'Distribute model layers across multiple GPUs for larger models (requires 2+ GPUs)',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'memory_optimization',
  },
  {
    id: 'config.process[0].model.attn_masking',
    label: 'Enable Attention Masking',
    description: 'Enable attention masking to save memory (Flux models only)',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'memory_optimization',
    applicableModels: ['flux', 'flux_kontext', 'flex1', 'flex2'],
  },

  // Model compilation
  {
    id: 'config.process[0].model.compile',
    label: 'Enable torch.compile()',
    description:
      'JIT compiles the model for faster execution. Increases startup time but faster per-step training. Requires PyTorch 2.0+',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'model_compilation',
  },

  // Model paths
  {
    id: 'config.process[0].model.assistant_lora_path',
    label: 'Assistant LoRA Path',
    description:
      'Path to assistant LoRA for training (e.g., for Schnell training adapter). Optional: Path to LoRA that assists training (useful for distilled models like Schnell).',
    type: 'string',
    defaultValue: '',
    step: 'quantization',
    section: 'model_paths',
    placeholder: 'e.g., ostris/FLUX.1-schnell-training-adapter',
  },
  {
    id: 'config.process[0].model.vae_path',
    label: 'Custom VAE Path',
    description: 'Path to custom VAE model (leave empty for default)',
    type: 'string',
    defaultValue: '',
    step: 'quantization',
    section: 'model_paths',
    placeholder: 'Optional: path/to/custom/vae',
  },
  {
    id: 'config.process[0].model.lora_path',
    label: 'Base LoRA Path',
    description: 'Continue training from an existing LoRA instead of starting fresh',
    type: 'string',
    defaultValue: '',
    step: 'quantization',
    section: 'model_paths',
    placeholder: 'Optional: path/to/existing/lora.safetensors',
  },
  {
    id: 'config.process[0].model.unet_path',
    label: 'Custom UNet Path',
    description: 'Path to custom UNet model (for modified architectures)',
    type: 'string',
    defaultValue: '',
    step: 'quantization',
    section: 'model_paths',
    placeholder: 'Optional: path/to/custom/unet',
  },

  // Flux-specific fields
  {
    id: 'config.process[0].model.use_flux_cfg',
    label: 'Enable Flux CFG Mode',
    description:
      'Enable classifier-free guidance for distillation and certain training workflows. Model learns both conditional and unconditional generation. Cost: ~30-50% more memory and training time due to dual path computation.',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'flux_options',
    applicableModels: ['flux', 'flux_kontext', 'flex1', 'flex2', 'chroma'],
  },

  // SDXL-specific fields
  {
    id: 'config.process[0].model.use_text_encoder_1',
    label: 'Use CLIP-L Text Encoder',
    description:
      'SDXL uses two text encoders. CLIP-L (~400MB) handles basic concepts. Disabling saves memory but reduces quality.',
    type: 'boolean',
    defaultValue: true,
    step: 'quantization',
    section: 'sdxl_options',
    applicableModels: ['sdxl'],
  },
  {
    id: 'config.process[0].model.use_text_encoder_2',
    label: 'Use OpenCLIP-G Text Encoder',
    description:
      'OpenCLIP-G (~1.4GB) provides deeper understanding. Disabling one saves memory but reduces prompt understanding.',
    type: 'boolean',
    defaultValue: true,
    step: 'quantization',
    section: 'sdxl_options',
    applicableModels: ['sdxl'],
  },
  {
    id: 'config.process[0].model.refiner_name_or_path',
    label: 'Refiner Model Path',
    description: 'Optional path to SDXL refiner model for two-stage generation.',
    type: 'string',
    defaultValue: '',
    step: 'quantization',
    section: 'sdxl_options',
    applicableModels: ['sdxl'],
    placeholder: 'stabilityai/stable-diffusion-xl-refiner-1.0',
  },
  {
    id: 'config.process[0].model.experimental_xl',
    label: 'Enable Experimental SDXL Features',
    description:
      'Enable bleeding-edge optimizations that are unstable and may not work with all configurations. Only enable if you understand the risks.',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'sdxl_options',
    applicableModels: ['sdxl'],
  },

  // Video model fields
  {
    id: 'config.process[0].model.low_vram',
    label: 'Low VRAM Mode',
    description: 'Enable aggressive memory optimizations for limited VRAM. May slow down training.',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'video_model_options',
    applicableModels: [
      'wan21:1b',
      'wan21:14b',
      'wan21_i2v:14b',
      'wan21_i2v:14b480p',
      'wan22_14b:t2v',
      'wan22_14b_i2v',
      'wan22_5b',
    ],
  },

  // Qwen model fields
  {
    id: 'config.process[0].model.model_kwargs.match_target_res',
    label: 'Match Target Resolution',
    description: 'Match output resolution to target image resolution.',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'qwen_options',
    applicableModels: ['qwen_image_edit_plus'],
  },

  // HiDream fields
  {
    id: 'config.process[0].network.network_kwargs.ignore_if_contains',
    label: 'Ignore MoE Layers',
    description: 'Skip training on Mixture of Experts layers to save VRAM.',
    type: 'boolean',
    defaultValue: true,
    step: 'quantization',
    section: 'hidream_options',
    applicableModels: ['hidream', 'hidream_e1'],
  },

  // === Target Step Fields ===
  {
    id: 'config.process[0].network.type',
    label: 'Training Target Type',
    description: 'LoRA is recommended for most cases. LoKr uses Kronecker products for potentially better quality.',
    type: 'select',
    defaultValue: 'lora',
    step: 'target',
    section: 'target_type',
    options: [
      { value: 'lora', label: 'LoRA (Low-Rank Adaptation)' },
      { value: 'lokr', label: 'LoKr (Kronecker Product)' },
      { value: 'lorm', label: 'LoRM (Mixture of Ranks)' },
    ],
  },
  {
    id: 'config.process[0].network.linear',
    label: 'Linear Rank',
    description:
      'Rank for linear layers. Higher = more learning capacity but larger file size. 16-32 is typical for character training, 8-16 for styles.',
    type: 'number',
    defaultValue: 16,
    step: 'target',
    section: 'lora_settings',
    min: 1,
    max: 256,
    numberStep: 1,
  },
  {
    id: 'config.process[0].network.linear_alpha',
    label: 'Linear Alpha',
    description: 'Alpha scaling for linear layers. Usually set equal to rank. Controls the strength of the trained weights.',
    type: 'number',
    defaultValue: 16,
    step: 'target',
    section: 'lora_settings',
    min: 1,
    max: 256,
    numberStep: 1,
  },
  {
    id: 'config.process[0].network.conv',
    label: 'Conv Rank',
    description: 'Rank for convolution layers. Set to 0 to skip conv training. Only useful for SD1.5/SDXL.',
    type: 'number',
    defaultValue: 0,
    step: 'target',
    section: 'lora_settings',
    min: 0,
    max: 256,
    numberStep: 1,
    applicableModels: ['sdxl', 'sd15'],
  },
  {
    id: 'config.process[0].network.conv_alpha',
    label: 'Conv Alpha',
    description: 'Alpha scaling for conv layers. Usually set equal to conv rank.',
    type: 'number',
    defaultValue: 0,
    step: 'target',
    section: 'lora_settings',
    min: 0,
    max: 256,
    numberStep: 1,
    applicableModels: ['sdxl', 'sd15'],
  },
  {
    id: 'config.process[0].network.dropout',
    label: 'Dropout Rate',
    description: 'Randomly drop LoRA outputs during training to prevent overfitting. 0.0 = no dropout, 0.1 = 10% dropout.',
    type: 'number',
    defaultValue: 0.0,
    step: 'target',
    section: 'advanced_lora',
    min: 0,
    max: 1,
    numberStep: 0.05,
  },
  {
    id: 'config.process[0].train.train_unet',
    label: 'Train UNet/Transformer',
    description: 'Train the main model (UNet for SD, Transformer for Flux). Usually enabled.',
    type: 'boolean',
    defaultValue: true,
    step: 'target',
    section: 'advanced_lora',
  },
  {
    id: 'config.process[0].train.train_text_encoder',
    label: 'Train Text Encoder',
    description: 'Train the text encoder along with the model. Can improve prompt following but increases VRAM usage.',
    type: 'boolean',
    defaultValue: false,
    step: 'target',
    section: 'advanced_lora',
  },

  // === Dataset Step Fields ===
  {
    id: 'config.process[0].datasets[0].folder_path',
    label: 'Dataset Folder Path',
    description: 'Path to the folder containing your training images. Images should have matching .txt caption files.',
    type: 'string',
    defaultValue: '',
    step: 'dataset',
    section: 'dataset_path',
    placeholder: '/path/to/your/dataset',
  },
  {
    id: 'config.process[0].trigger_word',
    label: 'Trigger Word',
    description:
      'A unique word that activates your LoRA. Will be prepended to all captions. Use something uncommon like "ohwx" or "sks".',
    type: 'string',
    defaultValue: '',
    step: 'dataset',
    section: 'captioning',
    placeholder: 'e.g., ohwx, sks, p3rs0n',
  },
  {
    id: 'config.process[0].datasets[0].caption_ext',
    label: 'Caption File Extension',
    description: 'File extension for caption files. Common options are .txt or .caption.',
    type: 'string',
    defaultValue: '.txt',
    step: 'dataset',
    section: 'captioning',
    placeholder: '.txt',
  },
  {
    id: 'config.process[0].datasets[0].default_caption',
    label: 'Default Caption',
    description: 'Caption to use when no caption file exists for an image.',
    type: 'string',
    defaultValue: '',
    step: 'dataset',
    section: 'captioning',
    placeholder: 'a photo',
  },
  {
    id: 'config.process[0].datasets[0].shuffle_captions',
    label: 'Shuffle Captions',
    description: 'Randomly shuffle the order of tags/words in captions each epoch. Helps prevent overfitting to caption structure.',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'captioning',
  },
  {
    id: 'config.process[0].datasets[0].random_crop',
    label: 'Random Crop',
    description: 'Randomly crop images instead of center cropping. Adds variety but may cut important details.',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'augmentation',
  },
  {
    id: 'config.process[0].datasets[0].flip_p',
    label: 'Horizontal Flip Probability',
    description: 'Probability of flipping images horizontally. 0.0 = never flip, 0.5 = flip half the time.',
    type: 'number',
    defaultValue: 0.0,
    step: 'dataset',
    section: 'augmentation',
    min: 0,
    max: 1,
    numberStep: 0.1,
  },
  {
    id: 'config.process[0].datasets[0].num_frames',
    label: 'Number of Frames',
    description: 'Number of frames to extract from each video. Must be odd number for some models.',
    type: 'number',
    defaultValue: 41,
    step: 'dataset',
    section: 'video_dataset',
    min: 1,
    max: 241,
    numberStep: 8,
    applicableModels: [
      'wan21:1b',
      'wan21:14b',
      'wan21_i2v:14b',
      'wan21_i2v:14b480p',
      'wan22_14b:t2v',
      'wan22_14b_i2v',
      'wan22_5b',
    ],
  },
  {
    id: 'config.process[0].datasets[0].fps',
    label: 'Frames Per Second',
    description: 'Target FPS for video training. Higher FPS = smoother motion but more computation.',
    type: 'number',
    defaultValue: 16,
    step: 'dataset',
    section: 'video_dataset',
    min: 1,
    max: 60,
    numberStep: 1,
    applicableModels: [
      'wan21:1b',
      'wan21:14b',
      'wan21_i2v:14b',
      'wan21_i2v:14b480p',
      'wan22_14b:t2v',
      'wan22_14b_i2v',
      'wan22_5b',
    ],
  },

  // === Training Step Fields ===
  {
    id: 'config.process[0].train.steps',
    label: 'Total Training Steps',
    description:
      'Total number of training steps. More steps = longer training. Typically 500-2000 for character LoRA, 100-500 for styles.',
    type: 'number',
    defaultValue: 1000,
    step: 'training',
    section: 'training_duration',
    min: 1,
    max: 100000,
    numberStep: 100,
  },
  {
    id: 'config.process[0].train.max_step_times_epochs',
    label: 'Max Epochs',
    description: 'Maximum number of times to repeat the dataset. Training stops when either steps or epochs is reached.',
    type: 'number',
    defaultValue: 10000,
    step: 'training',
    section: 'training_duration',
    min: 1,
    max: 100000,
    numberStep: 10,
  },
  {
    id: 'config.process[0].train.timestep_type',
    label: 'Timestep Sampling',
    description:
      'How to sample noise timesteps during training. "sigmoid" is standard, "shift" emphasizes certain noise levels, "weighted" balances high/low noise.',
    type: 'select',
    defaultValue: 'sigmoid',
    step: 'training',
    section: 'timestep_config',
    options: [
      { value: 'sigmoid', label: 'Sigmoid (Standard)' },
      { value: 'shift', label: 'Shift (Emphasize Middle)' },
      { value: 'weighted', label: 'Weighted (Balance High/Low)' },
      { value: 'linear', label: 'Linear (Uniform)' },
    ],
  },
  {
    id: 'config.process[0].train.timestep_bias_portion',
    label: 'Timestep Bias Portion',
    description: 'Portion of training to apply timestep bias. 1.0 = always apply bias.',
    type: 'number',
    defaultValue: 0.25,
    step: 'training',
    section: 'timestep_config',
    min: 0,
    max: 1,
    numberStep: 0.05,
  },

  // === Sampling Step Fields ===
  {
    id: 'config.process[0].sample.sample_every',
    label: 'Sample Every N Steps',
    description: 'Generate preview images every N training steps. Lower = more frequent samples but slower training.',
    type: 'number',
    defaultValue: 250,
    step: 'sampling',
    section: 'sample_generation',
    min: 1,
    max: 10000,
    numberStep: 50,
  },
  {
    id: 'config.process[0].sample.width',
    label: 'Sample Width',
    description: 'Width of generated sample images in pixels.',
    type: 'number',
    defaultValue: 1024,
    step: 'sampling',
    section: 'sample_generation',
    min: 256,
    max: 4096,
    numberStep: 64,
  },
  {
    id: 'config.process[0].sample.height',
    label: 'Sample Height',
    description: 'Height of generated sample images in pixels.',
    type: 'number',
    defaultValue: 1024,
    step: 'sampling',
    section: 'sample_generation',
    min: 256,
    max: 4096,
    numberStep: 64,
  },
  {
    id: 'config.process[0].sample.steps',
    label: 'Sampling Steps',
    description: 'Number of denoising steps for sample generation. Higher = better quality but slower.',
    type: 'number',
    defaultValue: 28,
    step: 'sampling',
    section: 'sample_generation',
    min: 1,
    max: 150,
    numberStep: 1,
  },
  {
    id: 'config.process[0].sample.guidance_scale',
    label: 'Guidance Scale (CFG)',
    description: 'How closely to follow the prompt. Higher = more prompt adherence but less creativity. 3.5-7 typical.',
    type: 'number',
    defaultValue: 4,
    step: 'sampling',
    section: 'sample_generation',
    min: 1,
    max: 30,
    numberStep: 0.5,
  },
  {
    id: 'config.process[0].sample.seed',
    label: 'Random Seed',
    description: 'Seed for reproducible samples. Use -1 for random seed each time.',
    type: 'number',
    defaultValue: 42,
    step: 'sampling',
    section: 'sample_generation',
    min: -1,
    max: 2147483647,
    numberStep: 1,
  },
  {
    id: 'config.process[0].sample.num_frames',
    label: 'Video Sample Frames',
    description: 'Number of frames to generate for video samples.',
    type: 'number',
    defaultValue: 41,
    step: 'sampling',
    section: 'sample_generation',
    min: 1,
    max: 241,
    numberStep: 8,
    applicableModels: [
      'wan21:1b',
      'wan21:14b',
      'wan21_i2v:14b',
      'wan21_i2v:14b480p',
      'wan22_14b:t2v',
      'wan22_14b_i2v',
      'wan22_5b',
    ],
  },
  {
    id: 'config.process[0].sample.fps',
    label: 'Video Sample FPS',
    description: 'Frames per second for video sample output.',
    type: 'number',
    defaultValue: 16,
    step: 'sampling',
    section: 'sample_generation',
    min: 1,
    max: 60,
    numberStep: 1,
    applicableModels: [
      'wan21:1b',
      'wan21:14b',
      'wan21_i2v:14b',
      'wan21_i2v:14b480p',
      'wan22_14b:t2v',
      'wan22_14b_i2v',
      'wan22_5b',
    ],
  },

  // === Optimizer Step Fields ===
  {
    id: 'config.process[0].train.lr',
    label: 'Learning Rate',
    description:
      'Base learning rate. Higher = faster learning but risk of instability. Typical: 1e-4 to 5e-4 for LoRA. Lower for styles, higher for characters.',
    type: 'number',
    defaultValue: 0.0001,
    step: 'optimizer',
    section: 'learning_rate',
    min: 0.0000001,
    max: 0.01,
    numberStep: 0.00001,
  },
  {
    id: 'config.process[0].train.unet_lr',
    label: 'UNet/Transformer Learning Rate',
    description: 'Specific learning rate for UNet/Transformer. Leave empty to use base LR.',
    type: 'number',
    defaultValue: null,
    step: 'optimizer',
    section: 'learning_rate',
    min: 0.0000001,
    max: 0.01,
    numberStep: 0.00001,
  },
  {
    id: 'config.process[0].train.text_encoder_lr',
    label: 'Text Encoder Learning Rate',
    description: 'Specific learning rate for text encoder. Usually lower than UNet LR (e.g., 5e-5).',
    type: 'number',
    defaultValue: null,
    step: 'optimizer',
    section: 'learning_rate',
    min: 0.0000001,
    max: 0.01,
    numberStep: 0.00001,
  },
  {
    id: 'config.process[0].train.warmup_steps',
    label: 'Warmup Steps',
    description: 'Number of steps to gradually increase LR from 0. Helps stability at start. 0 = no warmup.',
    type: 'number',
    defaultValue: 0,
    step: 'optimizer',
    section: 'learning_rate',
    min: 0,
    max: 10000,
    numberStep: 10,
  },
  {
    id: 'config.process[0].train.optimizer',
    label: 'Optimizer',
    description:
      'Optimization algorithm. AdamW is standard. Prodigy auto-tunes LR. AdaFactor saves memory. 8-bit variants reduce VRAM.',
    type: 'select',
    defaultValue: 'adamw',
    step: 'optimizer',
    section: 'optimizer_type',
    options: [
      { value: 'adamw', label: 'AdamW (Standard)' },
      { value: 'adamw8bit', label: 'AdamW 8-bit (Lower VRAM)' },
      { value: 'prodigy', label: 'Prodigy (Auto LR)' },
      { value: 'adafactor', label: 'Adafactor (Memory Efficient)' },
      { value: 'sgd', label: 'SGD (Simple)' },
      { value: 'lion', label: 'Lion (Experimental)' },
    ],
  },
  {
    id: 'config.process[0].train.weight_decay',
    label: 'Weight Decay',
    description: 'L2 regularization strength. Helps prevent overfitting. 0.01 is typical for AdamW.',
    type: 'number',
    defaultValue: 0.01,
    step: 'optimizer',
    section: 'optimizer_advanced',
    min: 0,
    max: 1,
    numberStep: 0.001,
  },
  {
    id: 'config.process[0].train.max_grad_norm',
    label: 'Max Gradient Norm',
    description: 'Clip gradients to this max norm. Prevents exploding gradients. 1.0 is typical.',
    type: 'number',
    defaultValue: 1.0,
    step: 'optimizer',
    section: 'optimizer_advanced',
    min: 0,
    max: 10,
    numberStep: 0.1,
  },
  {
    id: 'config.process[0].train.lr_scheduler',
    label: 'LR Scheduler',
    description:
      'How to adjust learning rate over training. Cosine gradually decreases, constant stays fixed, linear decreases linearly.',
    type: 'select',
    defaultValue: 'cosine',
    step: 'optimizer',
    section: 'lr_scheduler',
    options: [
      { value: 'constant', label: 'Constant' },
      { value: 'cosine', label: 'Cosine Annealing' },
      { value: 'cosine_with_restarts', label: 'Cosine with Restarts' },
      { value: 'linear', label: 'Linear Decay' },
      { value: 'polynomial', label: 'Polynomial Decay' },
    ],
  },
  {
    id: 'config.process[0].train.min_lr',
    label: 'Minimum Learning Rate',
    description: 'Minimum LR for schedulers that decay. Usually 0 or very small (e.g., 1e-6).',
    type: 'number',
    defaultValue: 0,
    step: 'optimizer',
    section: 'lr_scheduler',
    min: 0,
    max: 0.001,
    numberStep: 0.000001,
  },

  // === Memory Step Fields ===
  {
    id: 'config.process[0].train.batch_size',
    label: 'Batch Size',
    description: 'Number of images per training step. Higher = more stable but more VRAM. 1-4 typical for LoRA.',
    type: 'number',
    defaultValue: 1,
    step: 'memory',
    section: 'batch_settings',
    min: 1,
    max: 32,
    numberStep: 1,
  },
  {
    id: 'config.process[0].train.gradient_accumulation_steps',
    label: 'Gradient Accumulation Steps',
    description:
      'Accumulate gradients over N steps before updating. Simulates larger batch size without VRAM cost. Effective batch = batch_size * accumulation.',
    type: 'number',
    defaultValue: 1,
    step: 'memory',
    section: 'batch_settings',
    min: 1,
    max: 64,
    numberStep: 1,
  },
  {
    id: 'config.process[0].train.gradient_checkpointing',
    label: 'Gradient Checkpointing',
    description: 'Trade compute for memory. Recomputes activations during backward pass. Saves VRAM but slower.',
    type: 'boolean',
    defaultValue: true,
    step: 'memory',
    section: 'batch_settings',
  },
  {
    id: 'config.process[0].datasets[0].cache_latents',
    label: 'Cache Latents to Disk',
    description: 'Pre-encode images to latent space and cache. Faster training but uses disk space.',
    type: 'boolean',
    defaultValue: false,
    step: 'memory',
    section: 'caching',
  },
  {
    id: 'config.process[0].datasets[0].cache_latents_to_disk',
    label: 'Save Cache to Disk',
    description: 'Save cached latents to disk instead of RAM. Required for large datasets.',
    type: 'boolean',
    defaultValue: false,
    step: 'memory',
    section: 'caching',
    showWhen: {
      field: 'config.process[0].datasets[0].cache_latents',
      value: true,
    },
  },
  {
    id: 'config.process[0].train.num_workers',
    label: 'DataLoader Workers',
    description: 'Number of CPU workers for data loading. More = faster but more RAM. 0 = main process only.',
    type: 'number',
    defaultValue: 4,
    step: 'memory',
    section: 'dataloader',
    min: 0,
    max: 16,
    numberStep: 1,
  },
  {
    id: 'config.process[0].train.pin_memory',
    label: 'Pin Memory',
    description: 'Pin data to GPU memory for faster transfer. Uses more RAM but faster data loading.',
    type: 'boolean',
    defaultValue: true,
    step: 'memory',
    section: 'dataloader',
  },

  // === Advanced Step Fields ===
  {
    id: 'config.process[0].train.noise_scheduler',
    label: 'Noise Scheduler',
    description:
      'How noise is scheduled during training. flowmatch for Flux/modern models, ddpm for SD1.5/SDXL.',
    type: 'select',
    defaultValue: 'flowmatch',
    step: 'advanced',
    section: 'noise_scheduler',
    options: [
      { value: 'flowmatch', label: 'Flow Matching (Flux/Modern)' },
      { value: 'ddpm', label: 'DDPM (SD1.5/SDXL)' },
      { value: 'ddim', label: 'DDIM' },
    ],
  },
  {
    id: 'config.process[0].train.loss_type',
    label: 'Loss Type',
    description: 'How to compute training loss. MSE is standard, Huber is more robust to outliers.',
    type: 'select',
    defaultValue: 'mse',
    step: 'advanced',
    section: 'loss_function',
    options: [
      { value: 'mse', label: 'MSE (Mean Squared Error)' },
      { value: 'huber', label: 'Huber (Robust)' },
      { value: 'l1', label: 'L1 (Mean Absolute Error)' },
    ],
  },
  {
    id: 'config.process[0].train.snr_gamma',
    label: 'SNR Gamma',
    description:
      'Signal-to-noise ratio weighting. Min-SNR weighting with gamma=5 often improves quality. 0 = disabled.',
    type: 'number',
    defaultValue: 0,
    step: 'advanced',
    section: 'loss_function',
    min: 0,
    max: 20,
    numberStep: 1,
  },
  {
    id: 'config.process[0].train.bypass_guidance_embedding',
    label: 'Bypass Guidance Embedding',
    description: 'Skip guidance embedding for distilled models. Required for some Flux variants.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'guidance_settings',
    applicableModels: ['flex1', 'flex2'],
  },
  {
    id: 'config.process[0].train.ema_decay',
    label: 'EMA Decay',
    description:
      'Exponential moving average decay rate. 0 = disabled, 0.9999 typical. Smooths weights over training.',
    type: 'number',
    defaultValue: 0,
    step: 'advanced',
    section: 'regularization_techniques',
    min: 0,
    max: 1,
    numberStep: 0.0001,
  },
  {
    id: 'config.process[0].train.noise_offset',
    label: 'Noise Offset',
    description:
      'Add offset to noise schedule. Can help with very dark/bright images. 0.05-0.1 typical. 0 = disabled.',
    type: 'number',
    defaultValue: 0,
    step: 'advanced',
    section: 'regularization_techniques',
    min: 0,
    max: 1,
    numberStep: 0.01,
  },

  // === Resolution Step Fields ===
  {
    id: 'config.process[0].datasets[0].resolution[0]',
    label: 'Training Width',
    description: 'Width of training images in pixels. Must be divisible by 64 for most models.',
    type: 'number',
    defaultValue: 1024,
    step: 'resolution',
    section: 'image_resolution',
    min: 256,
    max: 4096,
    numberStep: 64,
  },
  {
    id: 'config.process[0].datasets[0].resolution[1]',
    label: 'Training Height',
    description: 'Height of training images in pixels. Must be divisible by 64 for most models.',
    type: 'number',
    defaultValue: 1024,
    step: 'resolution',
    section: 'image_resolution',
    min: 256,
    max: 4096,
    numberStep: 64,
  },
  {
    id: 'config.process[0].datasets[0].bucket_tolerance',
    label: 'Bucket Tolerance',
    description:
      'How much to allow images to differ from bucket center. Higher = more efficient but less precise crops.',
    type: 'number',
    defaultValue: 64,
    step: 'resolution',
    section: 'bucket_settings',
    min: 0,
    max: 256,
    numberStep: 8,
  },
  {
    id: 'config.process[0].datasets[0].keep_aspect_ratio',
    label: 'Keep Aspect Ratio',
    description: 'Preserve original aspect ratio when resizing. Uses bucketing to group similar ratios.',
    type: 'boolean',
    defaultValue: true,
    step: 'resolution',
    section: 'bucket_settings',
  },

  // === Save Step Fields ===
  {
    id: 'config.process[0].save.save_every',
    label: 'Save Every N Steps',
    description: 'Save a checkpoint every N training steps. Lower = more frequent saves.',
    type: 'number',
    defaultValue: 250,
    step: 'save',
    section: 'checkpoint_settings',
    min: 1,
    max: 10000,
    numberStep: 50,
  },
  {
    id: 'config.process[0].save.max_step_saves_to_keep',
    label: 'Max Checkpoints to Keep',
    description: 'Maximum number of step checkpoints to keep. Older ones are deleted. 0 = keep all.',
    type: 'number',
    defaultValue: 4,
    step: 'save',
    section: 'checkpoint_settings',
    min: 0,
    max: 100,
    numberStep: 1,
  },
  {
    id: 'config.process[0].save.save_format',
    label: 'Save Format',
    description: 'Format for saved LoRA weights. safetensors is recommended (safe, fast, compressed).',
    type: 'select',
    defaultValue: 'safetensors',
    step: 'save',
    section: 'save_format',
    options: [
      { value: 'safetensors', label: 'SafeTensors (Recommended)' },
      { value: 'pt', label: 'PyTorch (.pt)' },
      { value: 'ckpt', label: 'Checkpoint (.ckpt)' },
    ],
  },
  {
    id: 'config.process[0].save.push_to_hub',
    label: 'Push to HuggingFace Hub',
    description: 'Automatically upload checkpoints to HuggingFace Hub.',
    type: 'boolean',
    defaultValue: false,
    step: 'save',
    section: 'checkpoint_settings',
  },

  // === Logging Step Fields ===
  {
    id: 'config.process[0].logging.use_wandb',
    label: 'Enable Weights & Biases',
    description: 'Log training metrics to Weights & Biases for experiment tracking.',
    type: 'boolean',
    defaultValue: false,
    step: 'logging',
    section: 'wandb_settings',
  },
  {
    id: 'config.process[0].logging.wandb_project',
    label: 'W&B Project Name',
    description: 'Name of the W&B project to log to.',
    type: 'string',
    defaultValue: '',
    step: 'logging',
    section: 'wandb_settings',
    placeholder: 'my-lora-training',
    showWhen: {
      field: 'config.process[0].logging.use_wandb',
      value: true,
    },
  },
  {
    id: 'config.process[0].logging.wandb_run_name',
    label: 'W&B Run Name',
    description: 'Name for this specific training run.',
    type: 'string',
    defaultValue: '',
    step: 'logging',
    section: 'wandb_settings',
    placeholder: 'character-lora-v1',
    showWhen: {
      field: 'config.process[0].logging.use_wandb',
      value: true,
    },
  },
  {
    id: 'config.process[0].logging.log_every',
    label: 'Log Every N Steps',
    description: 'Log training metrics every N steps.',
    type: 'number',
    defaultValue: 10,
    step: 'logging',
    section: 'local_logging',
    min: 1,
    max: 1000,
    numberStep: 1,
  },
  {
    id: 'config.process[0].logging.verbose',
    label: 'Verbose Logging',
    description: 'Enable detailed logging output for debugging.',
    type: 'boolean',
    defaultValue: false,
    step: 'logging',
    section: 'local_logging',
  },
];

// Helper function to check if a field is applicable to a model
export function isFieldApplicable(field: FieldConfig, modelArch: string): boolean {
  if (!field.applicableModels) return true;
  return field.applicableModels.includes(modelArch);
}

// Helper function to check if a section is applicable to a model
export function isSectionApplicable(section: SectionConfig, modelArch: string): boolean {
  if (!section.applicableModels) return true;
  return section.applicableModels.includes(modelArch);
}

// Helper function to get all visible fields for a step and model
export function getVisibleFields(stepId: WizardStepId, modelArch: string, jobConfig: any): FieldConfig[] {
  return fields.filter(field => {
    if (field.step !== stepId) return false;
    if (!isFieldApplicable(field, modelArch)) return false;
    if (field.showWhen) {
      const conditionValue = getNestedValue(jobConfig, field.showWhen.field);
      if (conditionValue !== field.showWhen.value) return false;
    }
    return true;
  });
}

// Helper function to get all visible sections for a step and model
export function getVisibleSections(stepId: WizardStepId, modelArch: string): SectionConfig[] {
  return sections.filter(section => section.step === stepId && isSectionApplicable(section, modelArch)).sort((a, b) => a.order - b.order);
}
