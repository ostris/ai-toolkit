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

export type FieldType = 'boolean' | 'number' | 'string' | 'select' | 'slider' | 'compound' | 'sample_prompts_array';

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

  // Compound field support - single UI control sets multiple config values
  compoundOptions?: {
    value: any; // UI selection value
    label: string;
    sets: { path: string; value: any }[]; // config paths and values to set
  }[];
  compoundGetter?: (jobConfig: any) => any; // Function to derive UI value from config

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
  // Model step sections
  {
    id: 'model_selection',
    title: 'Model Selection',
    description: 'Choose your base model and architecture',
    step: 'model',
    order: 1,
  },
  {
    id: 'model_naming',
    title: 'Model Naming',
    description: 'Configure output model name and identifier',
    step: 'model',
    order: 2,
  },

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
    id: 'advanced_quantization',
    title: 'Advanced Quantization',
    description: 'Fine-grained quantization control for expert users',
    step: 'quantization',
    order: 4,
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

  // Regularization step sections
  {
    id: 'lora_regularization',
    title: 'LoRA Regularization',
    description: 'Techniques to prevent overfitting during LoRA training',
    step: 'regularization',
    order: 1,
  },
  {
    id: 'training_regularization',
    title: 'Training Regularization',
    description: 'General regularization techniques for stable training',
    step: 'regularization',
    order: 2,
  },

  // Monitoring step sections
  {
    id: 'monitoring_config',
    title: 'Resource Monitoring',
    description: 'Monitor GPU and system resources during training',
    step: 'monitoring',
    order: 1,
  },
  {
    id: 'monitoring_thresholds',
    title: 'Memory Thresholds',
    description: 'Set warning and critical memory usage thresholds',
    step: 'monitoring',
    order: 2,
  },
];

// Field definitions
export const fields: FieldConfig[] = [
  // === Model Step Fields ===

  // Model selection fields
  {
    id: 'config.process[0].model.name_or_path',
    label: 'Base Model',
    description:
      'Path or HuggingFace model ID of the base model to fine-tune. This determines the architecture and starting weights.',
    type: 'string',
    placeholder: 'e.g., black-forest-labs/FLUX.1-dev',
    step: 'model',
    section: 'model_selection',
  },
  {
    id: 'config.process[0].model.arch',
    label: 'Model Architecture',
    description: 'The architecture type of the model. Auto-detected based on model selection.',
    type: 'string',
    placeholder: 'flux, sdxl, sd15, etc.',
    step: 'model',
    section: 'model_selection',
  },

  // Model naming fields
  {
    id: 'config.name',
    label: 'Training Job Name',
    description: 'Name for this training job. Used for logging and saving checkpoints.',
    type: 'string',
    placeholder: 'my_lora_v1',
    step: 'model',
    section: 'model_naming',
  },
  {
    id: 'config.process[0].network.linear',
    label: 'LoRA Rank',
    description: 'How much your LoRA can learn. Higher = learns more details but bigger file size. 8-16 for styles, 16-32 for characters, 32+ for complex concepts.',
    type: 'number',
    defaultValue: 16,
    min: 1,
    max: 256,
    numberStep: 1,
    step: 'model',
    section: 'model_naming',
  },
  {
    id: 'config.process[0].network.linear_alpha',
    label: 'LoRA Alpha',
    description: 'Controls how strongly the LoRA affects the image. Usually set equal to rank. Higher alpha = stronger effect at same strength setting.',
    type: 'number',
    defaultValue: 16,
    min: 1,
    max: 256,
    numberStep: 1,
    step: 'model',
    section: 'model_naming',
  },

  // === Quantization Step Fields ===

  // Compound quantization controls (combines multiple config paths into single UI)
  {
    id: 'compound_quantization_level',
    label: 'Model Quantization',
    description:
      'Reduce model precision to save VRAM. 8-bit recommended for most cases. 4-bit for maximum savings.',
    type: 'compound',
    defaultValue: 'none',
    step: 'quantization',
    section: 'model_quantization',
    compoundOptions: [
      {
        value: 'none',
        label: 'No Quantization (Full Precision)',
        sets: [{ path: 'config.process[0].model.quantize', value: false }],
      },
      {
        value: '8bit',
        label: '8-bit (Recommended for most cases)',
        sets: [
          { path: 'config.process[0].model.quantize', value: true },
          { path: 'config.process[0].model.qtype', value: 'qfloat8' },
        ],
      },
      {
        value: '4bit',
        label: '4-bit (Maximum VRAM savings)',
        sets: [
          { path: 'config.process[0].model.quantize', value: true },
          { path: 'config.process[0].model.qtype', value: 'uint4' },
        ],
      },
    ],
  },
  {
    id: 'compound_te_quantization',
    label: 'Text Encoder Quantization',
    description: 'Also quantize the text encoder to save more VRAM.',
    type: 'compound',
    defaultValue: 'no',
    step: 'quantization',
    section: 'model_quantization',
    compoundOptions: [
      {
        value: 'no',
        label: 'No (Full Precision Text Encoder)',
        sets: [{ path: 'config.process[0].model.quantize_te', value: false }],
      },
      {
        value: 'yes',
        label: 'Yes (8-bit Text Encoder)',
        sets: [
          { path: 'config.process[0].model.quantize_te', value: true },
          { path: 'config.process[0].model.qtype_te', value: 'qfloat8' },
        ],
      },
    ],
    showWhen: {
      field: 'config.process[0].model.quantize',
      value: true,
    },
  },

  // Individual quantization fields (for fine-grained control, not shown by default)
  {
    id: 'config.process[0].model.quantize',
    label: 'Quantize Transformer',
    description:
      'Quantize transformer weights to reduce VRAM usage. Essential for large models like Flux on consumer GPUs.',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'advanced_quantization',
  },
  {
    id: 'config.process[0].model.qtype',
    label: 'Quantization Type',
    description:
      'Precision level for quantized weights. Lower bits = less VRAM but potentially lower quality. float8 is recommended for most cases.',
    type: 'select',
    defaultValue: 'qfloat8',
    step: 'quantization',
    section: 'advanced_quantization',
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
    section: 'advanced_quantization',
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
    description: 'Limits how much the model can change in one step. Prevents training from going crazy. 1.0 works well for most cases.',
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
      'Simulates training on more images at once without using extra VRAM. Set to 4 with batch size 1 to simulate batch size 4. Higher = more stable but slower.',
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
    description: 'Saves VRAM by recalculating some data instead of storing it. Makes training ~20-30% slower but uses much less memory. Recommended for most users.',
    type: 'boolean',
    defaultValue: true,
    step: 'memory',
    section: 'batch_settings',
  },
  {
    id: 'config.process[0].datasets[0].cache_latents',
    label: 'Cache Latents to Disk',
    description: 'Pre-processes your images once and saves the results. Makes training faster because it skips image processing each step. Highly recommended!',
    type: 'boolean',
    defaultValue: false,
    step: 'memory',
    section: 'caching',
  },
  {
    id: 'config.process[0].datasets[0].cache_latents_to_disk',
    label: 'Save Cache to Disk',
    description: 'Saves the processed images to disk instead of RAM. Turn on if you have more than 50-100 images. Uses ~4-8MB per image.',
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
    description: 'Speeds up moving data to your GPU. Uses a bit more RAM but makes training faster. Leave on unless you are running out of system RAM.',
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
      'Balances how much the model learns from noisy vs clean versions of images. Set to 5 for better quality on most models. 0 = disabled.',
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
      'Creates a smoothed version of your LoRA by averaging weights over time. Reduces sudden quality spikes. 0.9999 typical, 0 = disabled.',
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

  // === Regularization Step Fields ===
  {
    id: 'config.process[0].network.dropout',
    label: 'LoRA Dropout',
    description: 'Dropout rate for LoRA layers. Helps prevent overfitting. 0.0 = disabled, 0.1 typical.',
    type: 'number',
    defaultValue: 0.0,
    step: 'regularization',
    section: 'lora_regularization',
    min: 0,
    max: 1,
    numberStep: 0.05,
  },
  {
    id: 'config.process[0].train.weight_decay',
    label: 'Weight Decay',
    description: 'L2 regularization factor. Helps prevent large weights. 0.01 typical. 0 = disabled.',
    type: 'number',
    defaultValue: 0.01,
    step: 'regularization',
    section: 'training_regularization',
    min: 0,
    max: 1,
    numberStep: 0.001,
  },
  {
    id: 'config.process[0].train.gradient_clipping',
    label: 'Gradient Clipping',
    description: 'Maximum gradient norm. Prevents exploding gradients. 1.0 typical. 0 = disabled.',
    type: 'number',
    defaultValue: 1.0,
    step: 'regularization',
    section: 'training_regularization',
    min: 0,
    max: 10,
    numberStep: 0.1,
  },
  {
    id: 'config.process[0].train.min_snr_gamma',
    label: 'Min SNR Gamma',
    description:
      'Min-SNR weighting gamma. Balances loss across noise levels. 5.0 recommended. 0 = disabled.',
    type: 'number',
    defaultValue: 5.0,
    step: 'regularization',
    section: 'training_regularization',
    min: 0,
    max: 20,
    numberStep: 0.5,
  },
  {
    id: 'config.process[0].train.ema_config.use_ema',
    label: 'Use EMA',
    description:
      'Enable Exponential Moving Average of weights. Provides smoother, more stable results.',
    type: 'boolean',
    defaultValue: false,
    step: 'regularization',
    section: 'training_regularization',
  },
  {
    id: 'config.process[0].train.ema_config.ema_decay',
    label: 'EMA Decay',
    description: 'Decay rate for EMA. Higher = more smoothing. 0.9999 typical.',
    type: 'number',
    defaultValue: 0.9999,
    step: 'regularization',
    section: 'training_regularization',
    min: 0.9,
    max: 1.0,
    numberStep: 0.0001,
    showWhen: {
      field: 'config.process[0].train.ema_config.use_ema',
      value: true,
    },
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

  // === Monitoring Step Fields ===
  {
    id: 'config.process[0].monitoring.enabled',
    label: 'Enable Monitoring',
    description: 'Enable resource monitoring during training to track GPU memory and system utilization.',
    type: 'boolean',
    defaultValue: false,
    step: 'monitoring',
    section: 'monitoring_config',
  },
  {
    id: 'config.process[0].monitoring.sample_interval_seconds',
    label: 'Sample Interval (seconds)',
    description: 'How often to sample resource usage. Lower = more detailed but more overhead.',
    type: 'number',
    defaultValue: 5,
    step: 'monitoring',
    section: 'monitoring_config',
    min: 1,
    max: 60,
    numberStep: 1,
    showWhen: {
      field: 'config.process[0].monitoring.enabled',
      value: true,
    },
  },
  {
    id: 'config.process[0].monitoring.track_per_process',
    label: 'Track Per Process',
    description: 'Track resource usage per training process (useful for multi-GPU).',
    type: 'boolean',
    defaultValue: false,
    step: 'monitoring',
    section: 'monitoring_config',
    showWhen: {
      field: 'config.process[0].monitoring.enabled',
      value: true,
    },
  },
  {
    id: 'config.process[0].monitoring.analyze_on_complete',
    label: 'Analyze On Complete',
    description: 'Generate resource usage analysis report when training completes.',
    type: 'boolean',
    defaultValue: true,
    step: 'monitoring',
    section: 'monitoring_config',
    showWhen: {
      field: 'config.process[0].monitoring.enabled',
      value: true,
    },
  },
  {
    id: 'config.process[0].monitoring.memory_warning_threshold',
    label: 'Memory Warning Threshold',
    description: 'Warn when GPU memory usage exceeds this percentage.',
    type: 'slider',
    defaultValue: 0.85,
    step: 'monitoring',
    section: 'monitoring_thresholds',
    min: 0,
    max: 1,
    numberStep: 0.05,
    showWhen: {
      field: 'config.process[0].monitoring.enabled',
      value: true,
    },
  },
  {
    id: 'config.process[0].monitoring.memory_critical_threshold',
    label: 'Memory Critical Threshold',
    description: 'Alert when GPU memory usage exceeds this critical percentage.',
    type: 'slider',
    defaultValue: 0.95,
    step: 'monitoring',
    section: 'monitoring_thresholds',
    min: 0,
    max: 1,
    numberStep: 0.05,
    showWhen: {
      field: 'config.process[0].monitoring.enabled',
      value: true,
    },
  },

  // === Additional Quantization Fields ===
  {
    id: 'config.process[0].model.qtype_te',
    label: 'Text Encoder Quantization Type',
    description: 'Precision for quantized text encoder weights.',
    type: 'select',
    defaultValue: 'qfloat8',
    step: 'quantization',
    section: 'model_quantization',
    options: [
      { value: 'qfloat8', label: 'float8 (default)' },
      { value: 'uint4', label: '4 bit' },
    ],
    showWhen: {
      field: 'config.process[0].model.quantize_te',
      value: true,
    },
  },
  {
    id: 'config.process[0].model.text_encoder_bits',
    label: 'Text Encoder Bits',
    description: 'Precision bits for text encoder. 16 = full precision, lower = less VRAM.',
    type: 'select',
    defaultValue: 16,
    step: 'quantization',
    section: 'model_quantization',
    options: [
      { value: 16, label: '16-bit (Full)' },
      { value: 8, label: '8-bit' },
      { value: 4, label: '4-bit' },
    ],
  },

  // === Additional Target/Network Fields ===
  {
    id: 'config.process[0].network.alpha',
    label: 'Global Alpha',
    description: 'Global alpha scaling for all LoRA layers. Overrides individual linear/conv alpha.',
    type: 'number',
    defaultValue: null,
    step: 'target',
    section: 'advanced_lora',
    min: 1,
    max: 256,
    numberStep: 1,
  },
  {
    id: 'config.process[0].network.transformer_only',
    label: 'Transformer Only',
    description: 'Only train transformer layers, skip other components like attention blocks.',
    type: 'boolean',
    defaultValue: false,
    step: 'target',
    section: 'advanced_lora',
  },
  {
    id: 'config.process[0].network.lokr_full_rank',
    label: 'LoKr Full Rank',
    description: 'Sets how detailed the LoKr can be. Higher = better quality but larger file. Leave empty for automatic.',
    type: 'number',
    defaultValue: null,
    step: 'target',
    section: 'advanced_lora',
    min: 1,
    max: 256,
    numberStep: 1,
  },
  {
    id: 'config.process[0].network.lokr_factor',
    label: 'LoKr Factor',
    description: 'How the LoKr breaks down the model weights. -1 = automatic. Higher = more efficient but may lose detail.',
    type: 'number',
    defaultValue: null,
    step: 'target',
    section: 'advanced_lora',
    min: -1,
    max: 64,
    numberStep: 1,
  },

  // === Additional Dataset Fields ===
  {
    id: 'config.process[0].datasets[0].num_repeats',
    label: 'Dataset Repeats',
    description: 'Number of times to repeat the dataset per epoch. Higher = more training on same data.',
    type: 'number',
    defaultValue: 1,
    step: 'dataset',
    section: 'dataset_path',
    min: 1,
    max: 1000,
    numberStep: 1,
  },
  {
    id: 'config.process[0].datasets[0].keep_tokens',
    label: 'Keep Tokens',
    description: 'Number of tokens to keep at start of caption (not shuffled). Preserves trigger word position.',
    type: 'number',
    defaultValue: 0,
    step: 'dataset',
    section: 'captioning',
    min: 0,
    max: 77,
    numberStep: 1,
  },
  {
    id: 'config.process[0].datasets[0].token_dropout_rate',
    label: 'Token Dropout Rate',
    description: 'Randomly drop tokens from captions. Helps model generalize. 0 = no dropout.',
    type: 'number',
    defaultValue: 0.0,
    step: 'dataset',
    section: 'captioning',
    min: 0,
    max: 1,
    numberStep: 0.05,
  },
  {
    id: 'config.process[0].datasets[0].shuffle_tokens',
    label: 'Shuffle Tokens',
    description: 'Randomly shuffle token order in captions each step.',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'captioning',
  },
  {
    id: 'config.process[0].datasets[0].caption_dropout_rate',
    label: 'Caption Dropout Rate',
    description: 'Probability of dropping entire caption (trains unconditional generation). 0.1 typical.',
    type: 'number',
    defaultValue: 0.0,
    step: 'dataset',
    section: 'captioning',
    min: 0,
    max: 1,
    numberStep: 0.05,
  },
  {
    id: 'config.process[0].datasets[0].scale',
    label: 'Image Scale',
    description: 'Scale factor for images. 1.0 = original size, <1 = downscale, >1 = upscale.',
    type: 'number',
    defaultValue: 1.0,
    step: 'resolution',
    section: 'image_resolution',
    min: 0.1,
    max: 4.0,
    numberStep: 0.1,
  },
  {
    id: 'config.process[0].datasets[0].square_crop',
    label: 'Square Crop',
    description: 'Force square crops for all images.',
    type: 'boolean',
    defaultValue: false,
    step: 'resolution',
    section: 'bucket_settings',
  },
  {
    id: 'config.process[0].datasets[0].enable_ar_bucket',
    label: 'Enable Aspect Ratio Bucketing',
    description: 'Group images by aspect ratio for efficient training without stretching.',
    type: 'boolean',
    defaultValue: true,
    step: 'resolution',
    section: 'bucket_settings',
  },
  {
    id: 'config.process[0].datasets[0].buckets',
    label: 'Use Fixed Buckets',
    description: 'Use predefined aspect ratio buckets instead of dynamic bucketing.',
    type: 'boolean',
    defaultValue: false,
    step: 'resolution',
    section: 'bucket_settings',
  },

  // === Additional Memory Fields ===
  {
    id: 'config.process[0].train.gradient_accumulation',
    label: 'Gradient Accumulation',
    description: 'Alias for gradient accumulation steps. Simulates larger batch size.',
    type: 'number',
    defaultValue: 1,
    step: 'memory',
    section: 'batch_settings',
    min: 1,
    max: 64,
    numberStep: 1,
  },
  {
    id: 'config.process[0].train.auto_scale_batch_size',
    label: 'Auto Scale Batch Size',
    description: 'Automatically adjust batch size based on available VRAM.',
    type: 'boolean',
    defaultValue: false,
    step: 'memory',
    section: 'batch_settings',
  },
  {
    id: 'config.process[0].train.min_batch_size',
    label: 'Minimum Batch Size',
    description: 'Minimum batch size for auto-scaling.',
    type: 'number',
    defaultValue: 1,
    step: 'memory',
    section: 'batch_settings',
    min: 1,
    max: 32,
    numberStep: 1,
    showWhen: {
      field: 'config.process[0].train.auto_scale_batch_size',
      value: true,
    },
  },
  {
    id: 'config.process[0].train.max_batch_size',
    label: 'Maximum Batch Size',
    description: 'Maximum batch size for auto-scaling.',
    type: 'number',
    defaultValue: 8,
    step: 'memory',
    section: 'batch_settings',
    min: 1,
    max: 64,
    numberStep: 1,
    showWhen: {
      field: 'config.process[0].train.auto_scale_batch_size',
      value: true,
    },
  },
  {
    id: 'config.process[0].datasets[0].cache_text_embeddings',
    label: 'Cache Text Embeddings',
    description: 'Pre-compute and cache text encoder outputs for faster training. Note: Not compatible with "Train Text Encoder" - will be ignored if text encoder training is enabled.',
    type: 'boolean',
    defaultValue: false,
    step: 'memory',
    section: 'caching',
  },
  {
    id: 'config.process[0].datasets[0].cache_clip_vision_to_disk',
    label: 'Cache CLIP Vision to Disk',
    description: 'Cache CLIP vision encoder outputs to disk. Only works with IP-Adapter or similar adapter models - leave disabled for standard LoRA training.',
    type: 'boolean',
    defaultValue: false,
    step: 'memory',
    section: 'caching',
  },
  {
    id: 'config.process[0].datasets[0].gpu_prefetch_batches',
    label: 'GPU Prefetch Batches',
    description: 'Number of batches to prefetch to GPU. Higher = faster but more VRAM.',
    type: 'number',
    defaultValue: 2,
    step: 'memory',
    section: 'dataloader',
    min: 0,
    max: 16,
    numberStep: 1,
  },
  {
    id: 'config.process[0].datasets[0].prefetch_factor',
    label: 'Prefetch Factor',
    description: 'Number of batches to prefetch per worker. Requires num_workers > 0.',
    type: 'number',
    defaultValue: 2,
    step: 'memory',
    section: 'dataloader',
    min: 0,
    max: 16,
    numberStep: 1,
  },
  {
    id: 'config.process[0].datasets[0].persistent_workers',
    label: 'Persistent Workers',
    description: 'Keep data loader workers alive between epochs. Faster but uses more RAM.',
    type: 'boolean',
    defaultValue: false,
    step: 'memory',
    section: 'dataloader',
  },

  // === Additional Training Fields ===
  {
    id: 'config.process[0].train.prompt_dropout_prob',
    label: 'Prompt Dropout Probability',
    description: 'Probability of dropping prompt conditioning. Trains model to work without prompts.',
    type: 'number',
    defaultValue: 0.0,
    step: 'training',
    section: 'training_duration',
    min: 0,
    max: 1,
    numberStep: 0.05,
  },
  {
    id: 'config.process[0].train.loss_target',
    label: 'Loss Target',
    description: 'What the model predicts during training. "noise" predicts added noise, "source" predicts clean image.',
    type: 'select',
    defaultValue: 'noise',
    step: 'advanced',
    section: 'loss_function',
    options: [
      { value: 'noise', label: 'Noise Prediction' },
      { value: 'source', label: 'Source Prediction' },
      { value: 'v_prediction', label: 'V Prediction' },
    ],
  },
  {
    id: 'config.process[0].train.do_cfg',
    label: 'Enable CFG Training',
    description: 'Train with classifier-free guidance. Model learns both conditional and unconditional generation.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'guidance_settings',
  },
  {
    id: 'config.process[0].train.cfg_scale',
    label: 'CFG Scale',
    description: 'Classifier-free guidance scale during training.',
    type: 'number',
    defaultValue: 1.0,
    step: 'advanced',
    section: 'guidance_settings',
    min: 1,
    max: 30,
    numberStep: 0.5,
    showWhen: {
      field: 'config.process[0].train.do_cfg',
      value: true,
    },
  },
  {
    id: 'config.process[0].train.cache_text_embeddings',
    label: 'Cache Text Embeddings (Train)',
    description: 'Cache text encoder outputs during training to save computation. Note: Not compatible with "Train Text Encoder" - will be ignored if text encoder training is enabled.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'regularization_techniques',
  },
  {
    id: 'config.process[0].train.unload_text_encoder',
    label: 'Unload Text Encoder',
    description: 'Unload text encoder after encoding to save VRAM. Requires cached embeddings.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'regularization_techniques',
  },

  // === Additional Sampling Fields ===
  {
    id: 'config.process[0].sample.sampler',
    label: 'Sampler',
    description: 'Sampling algorithm for generating preview images.',
    type: 'select',
    defaultValue: 'euler',
    step: 'sampling',
    section: 'sample_generation',
    options: [
      { value: 'euler', label: 'Euler' },
      { value: 'euler_a', label: 'Euler Ancestral' },
      { value: 'dpm++_2m', label: 'DPM++ 2M' },
      { value: 'dpm++_sde', label: 'DPM++ SDE' },
      { value: 'heun', label: 'Heun' },
    ],
  },
  {
    id: 'config.process[0].sample.sample_steps',
    label: 'Sample Denoising Steps',
    description: 'Alias for sampling steps. Number of denoising iterations.',
    type: 'number',
    defaultValue: 28,
    step: 'sampling',
    section: 'sample_generation',
    min: 1,
    max: 150,
    numberStep: 1,
  },
  {
    id: 'config.process[0].sample.walk_seed',
    label: 'Walk Seed',
    description: 'Enable seed walking for sample variety. Generates slightly different seeds each sample.',
    type: 'boolean',
    defaultValue: false,
    step: 'sampling',
    section: 'sample_generation',
  },
  {
    id: 'config.process[0].sample.samples',
    label: 'Sample Prompts',
    description: 'Prompts to generate sample images during training. Each prompt can have custom resolution and seed.',
    type: 'sample_prompts_array',
    defaultValue: [{ prompt: '' }],
    step: 'sampling',
    section: 'sample_prompts',
  },
  {
    id: 'config.process[0].sample.neg',
    label: 'Negative Prompt',
    description: 'Negative prompt for sample generation. What to avoid in generated images.',
    type: 'string',
    defaultValue: '',
    step: 'sampling',
    section: 'sample_prompts',
    placeholder: 'blurry, low quality, distorted',
  },
  {
    id: 'config.process[0].sample.network_multiplier',
    label: 'Network Multiplier',
    description: 'Strength of LoRA during sampling. 1.0 = full strength, lower = weaker effect.',
    type: 'number',
    defaultValue: 1.0,
    step: 'sampling',
    section: 'sample_generation',
    min: 0,
    max: 2,
    numberStep: 0.1,
  },
  {
    id: 'config.process[0].sample.guidance_rescale',
    label: 'Guidance Rescale',
    description: 'Rescale guidance to reduce artifacts at high CFG. 0 = disabled.',
    type: 'number',
    defaultValue: 0,
    step: 'sampling',
    section: 'sample_generation',
    min: 0,
    max: 1,
    numberStep: 0.05,
  },
  {
    id: 'config.process[0].sample.format',
    label: 'Sample Format',
    description: 'Output format for generated samples.',
    type: 'select',
    defaultValue: 'png',
    step: 'sampling',
    section: 'sample_generation',
    options: [
      { value: 'png', label: 'PNG' },
      { value: 'jpg', label: 'JPEG' },
      { value: 'webp', label: 'WebP' },
    ],
  },
  {
    id: 'config.process[0].sample.refiner_start_at',
    label: 'Refiner Start At',
    description: 'Step to switch to refiner model (SDXL only). 0.8 = last 20% of steps.',
    type: 'number',
    defaultValue: 0.8,
    step: 'sampling',
    section: 'sample_generation',
    min: 0,
    max: 1,
    numberStep: 0.05,
    applicableModels: ['sdxl'],
  },
  {
    id: 'config.process[0].train.disable_sampling',
    label: 'Disable Sampling',
    description: 'Turn off sample generation during training to speed up training.',
    type: 'boolean',
    defaultValue: false,
    step: 'sampling',
    section: 'sample_generation',
  },
  {
    id: 'config.process[0].train.skip_first_sample',
    label: 'Skip First Sample',
    description: 'Skip generating a sample at the start of training.',
    type: 'boolean',
    defaultValue: false,
    step: 'sampling',
    section: 'sample_generation',
  },
  {
    id: 'config.process[0].train.force_first_sample',
    label: 'Force First Sample',
    description: 'Always generate a sample at the very first step.',
    type: 'boolean',
    defaultValue: false,
    step: 'sampling',
    section: 'sample_generation',
  },

  // === Additional Save Fields ===
  {
    id: 'config.process[0].save.dtype',
    label: 'Save Data Type',
    description: 'Precision for saved weights. float16 for compatibility, bfloat16 for newer hardware.',
    type: 'select',
    defaultValue: 'float16',
    step: 'save',
    section: 'save_format',
    options: [
      { value: 'float16', label: 'Float16' },
      { value: 'bfloat16', label: 'BFloat16' },
      { value: 'float32', label: 'Float32' },
    ],
  },
  {
    id: 'config.process[0].save.hf_repo_id',
    label: 'HuggingFace Repository ID',
    description: 'Repository ID for HuggingFace Hub upload (username/repo-name).',
    type: 'string',
    defaultValue: '',
    step: 'save',
    section: 'checkpoint_settings',
    placeholder: 'username/my-lora',
    showWhen: {
      field: 'config.process[0].save.push_to_hub',
      value: true,
    },
  },
  {
    id: 'config.process[0].save.hf_private',
    label: 'Private Repository',
    description: 'Make the HuggingFace repository private.',
    type: 'boolean',
    defaultValue: true,
    step: 'save',
    section: 'checkpoint_settings',
    showWhen: {
      field: 'config.process[0].save.push_to_hub',
      value: true,
    },
  },

  // === MISSING FIELD DEFINITIONS (Regression Fixes) ===

  // Dataset fields - additional augmentation and configuration
  {
    id: 'config.process[0].datasets[0].alpha_mask',
    label: 'Alpha Mask',
    description: 'For images with transparent backgrounds: focuses learning on the visible parts only. Ignores the transparent areas. Great for object/character training.',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'augmentation',
  },
  {
    id: 'config.process[0].datasets[0].control_path',
    label: 'Control Image Path',
    description: 'Path to control images for ControlNet training.',
    type: 'string',
    defaultValue: '',
    step: 'dataset',
    section: 'dataset_path',
    placeholder: '/path/to/control/images',
  },
  {
    id: 'config.process[0].datasets[0].flip_x',
    label: 'Random Horizontal Flip',
    description: 'Randomly flip images horizontally for data augmentation.',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'augmentation',
  },
  {
    id: 'config.process[0].datasets[0].flip_y',
    label: 'Random Vertical Flip',
    description: 'Randomly flip images vertically for data augmentation.',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'augmentation',
  },
  {
    id: 'config.process[0].datasets[0].inpaint_path',
    label: 'Inpainting Mask Path',
    description: 'Path to inpainting masks for inpaint model training.',
    type: 'string',
    defaultValue: '',
    step: 'dataset',
    section: 'dataset_path',
    placeholder: '/path/to/inpaint/masks',
  },
  {
    id: 'config.process[0].datasets[0].is_reg',
    label: 'Regularization Dataset',
    description: 'LEAVE THIS OFF for your primary training images. Only enable for a SECOND dataset containing generic class images (like "a person" for character training) to prevent overfitting. If enabled on your only dataset, training will fail with "NoneType has no attribute get_caption_list".',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'augmentation',
  },
  {
    id: 'config.process[0].datasets[0].loss_multiplier',
    label: 'Loss Multiplier',
    description: 'Multiply loss for this dataset. Higher values increase its importance.',
    type: 'number',
    defaultValue: 1.0,
    step: 'dataset',
    section: 'augmentation',
    min: 0.1,
    max: 10,
    numberStep: 0.1,
  },
  {
    id: 'config.process[0].datasets[0].num_workers',
    label: 'Dataset Workers',
    description: 'Number of parallel workers for loading this dataset.',
    type: 'number',
    defaultValue: 2,
    step: 'dataset',
    section: 'dataset_path',
    min: 0,
    max: 16,
    numberStep: 1,
  },
  {
    id: 'config.process[0].datasets[0].prior_reg',
    label: 'Prior Regularization',
    description: 'Compare training results to the original model to preserve what it already knows. Helps avoid "forgetting" how to draw other things.',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'augmentation',
  },
  {
    id: 'config.process[0].datasets[0].random_scale',
    label: 'Random Scale',
    description: 'Randomly scale images during training (e.g., 0.9-1.1).',
    type: 'boolean',
    defaultValue: false,
    step: 'dataset',
    section: 'augmentation',
  },
  {
    id: 'config.process[0].datasets[0].standardize_images',
    label: 'Standardize Images',
    description: 'Adjusts all image colors/brightness to a consistent range. Recommended to leave on. Only disable if your images are already perfectly calibrated.',
    type: 'boolean',
    defaultValue: true,
    step: 'dataset',
    section: 'augmentation',
  },

  // Logging fields
  {
    id: 'config.process[0].logging.project_name',
    label: 'Project Name',
    description: 'W&B project name for organizing experiments.',
    type: 'string',
    defaultValue: '',
    step: 'logging',
    section: 'wandb_settings',
    placeholder: 'my-lora-experiments',
  },
  {
    id: 'config.process[0].logging.run_name',
    label: 'Run Name',
    description: 'W&B run name for this specific training session.',
    type: 'string',
    defaultValue: '',
    step: 'logging',
    section: 'wandb_settings',
    placeholder: 'flux-lora-v1',
  },

  // Model fields - device and dtype configuration
  {
    id: 'config.process[0].model.split_model_other_module_param_count_scale',
    label: 'Split Model Parameter Scale',
    description: 'When splitting model across GPUs, this controls how much memory each GPU gets. Higher = more even distribution. Only change if you have unbalanced GPUs.',
    type: 'number',
    defaultValue: 1.0,
    step: 'quantization',
    section: 'memory_optimization',
    min: 0.1,
    max: 10,
    numberStep: 0.1,
  },
  {
    id: 'config.process[0].model.te_device',
    label: 'Text Encoder Device',
    description: 'Where to run the text encoder. GPU is faster but uses more VRAM. CPU saves VRAM but is slower. Use CUDA:0/1 if you have multiple GPUs.',
    type: 'select',
    defaultValue: 'cuda',
    step: 'quantization',
    section: 'memory_optimization',
    options: [
      { value: 'cuda', label: 'CUDA (GPU)' },
      { value: 'cpu', label: 'CPU' },
      { value: 'cuda:0', label: 'CUDA:0' },
      { value: 'cuda:1', label: 'CUDA:1' },
    ],
  },
  {
    id: 'config.process[0].model.te_dtype',
    label: 'Text Encoder Data Type',
    description: 'Number format for the text encoder. Float16 saves memory and works on most GPUs. BFloat16 is better on newer GPUs (RTX 30/40 series). Float32 uses more memory but is most accurate.',
    type: 'select',
    defaultValue: 'float16',
    step: 'quantization',
    section: 'memory_optimization',
    options: [
      { value: 'float32', label: 'Float32 (Full Precision)' },
      { value: 'float16', label: 'Float16 (Half Precision)' },
      { value: 'bfloat16', label: 'BFloat16' },
    ],
  },
  {
    id: 'config.process[0].model.vae_device',
    label: 'VAE Device',
    description: 'Where to run the image encoder/decoder. GPU is much faster. CPU saves ~1-2GB VRAM but slows down caching. Only move to CPU if desperate for VRAM.',
    type: 'select',
    defaultValue: 'cuda',
    step: 'quantization',
    section: 'memory_optimization',
    options: [
      { value: 'cuda', label: 'CUDA (GPU)' },
      { value: 'cpu', label: 'CPU' },
      { value: 'cuda:0', label: 'CUDA:0' },
      { value: 'cuda:1', label: 'CUDA:1' },
    ],
  },
  {
    id: 'config.process[0].model.vae_dtype',
    label: 'VAE Data Type',
    description: 'Number format for the image encoder/decoder. BFloat16 recommended for most cases. Float32 uses more memory but may produce slightly better colors.',
    type: 'select',
    defaultValue: 'bfloat16',
    step: 'quantization',
    section: 'memory_optimization',
    options: [
      { value: 'float32', label: 'Float32 (Full Precision)' },
      { value: 'float16', label: 'Float16 (Half Precision)' },
      { value: 'bfloat16', label: 'BFloat16' },
    ],
  },

  // Training fields - advanced training options
  {
    id: 'config.process[0].train.blank_prompt_preservation',
    label: 'Blank Prompt Preservation',
    description: 'Keeps the model working well even without prompts. Set to 0.1-0.3 to prevent your LoRA from breaking promptless generation. 0 = disabled.',
    type: 'number',
    defaultValue: 0,
    step: 'advanced',
    section: 'regularization_techniques',
    min: 0,
    max: 1,
    numberStep: 0.01,
  },
  {
    id: 'config.process[0].train.blended_blur_noise',
    label: 'Blended Blur Noise',
    description: 'Softens the noise added during training. Can help produce smoother, less grainy results. Experimental feature.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'noise_scheduler',
  },
  {
    id: 'config.process[0].train.diff_output_preservation',
    label: 'Diff Output Preservation',
    description: 'Helps maintain the original model style while learning new concepts. Higher values (0.1-0.5) reduce style drift. 0 = disabled.',
    type: 'number',
    defaultValue: 0,
    step: 'advanced',
    section: 'regularization_techniques',
    min: 0,
    max: 1,
    numberStep: 0.01,
  },
  {
    id: 'config.process[0].train.diffusion_feature_extractor_path',
    label: 'Diffusion Feature Extractor Path',
    description: 'Advanced: Path to a model that helps compare image features during training. Leave empty unless you know what this is.',
    type: 'string',
    defaultValue: '',
    step: 'advanced',
    section: 'loss_function',
    placeholder: '/path/to/feature/extractor',
  },
  {
    id: 'config.process[0].train.do_prior_divergence',
    label: 'Prior Divergence Loss',
    description: 'Compares your trained model to the original to prevent forgetting existing knowledge. Helps avoid overfitting on small datasets.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'regularization_techniques',
  },
  {
    id: 'config.process[0].train.force_consistent_noise',
    label: 'Force Consistent Noise',
    description: 'Uses the same random noise pattern throughout training. Can make training more stable but less diverse. Experimental.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'noise_scheduler',
  },
  {
    id: 'config.process[0].train.free_u',
    label: 'FreeU Enhancement',
    description: 'A technique that can make generated images sharper and more detailed. May not work well with all models. Experimental.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'guidance_settings',
  },
  {
    id: 'config.process[0].train.inverted_mask_prior',
    label: 'Inverted Mask Prior',
    description: 'For inpainting training: focuses learning on the area outside the mask instead of inside. Specialized use case.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'regularization_techniques',
  },
  {
    id: 'config.process[0].train.latent_feature_extractor_path',
    label: 'Latent Feature Extractor Path',
    description: 'Advanced: Path to a model that compares compressed image representations. Leave empty unless you have a specific model.',
    type: 'string',
    defaultValue: '',
    step: 'advanced',
    section: 'loss_function',
    placeholder: '/path/to/latent/extractor',
  },
  {
    id: 'config.process[0].train.max_denoising_steps',
    label: 'Max Denoising Steps',
    description: 'Training samples noise levels from 0 to this max. Default 1000 covers full range. Lower values focus on specific noise levels.',
    type: 'number',
    defaultValue: 1000,
    step: 'training',
    section: 'timestep_config',
    min: 1,
    max: 1000,
    numberStep: 1,
  },
  {
    id: 'config.process[0].train.min_denoising_steps',
    label: 'Min Denoising Steps',
    description: 'Start sampling from this noise level instead of 0. Useful to skip very noisy steps. 0 = start from beginning.',
    type: 'number',
    defaultValue: 0,
    step: 'training',
    section: 'timestep_config',
    min: 0,
    max: 1000,
    numberStep: 1,
  },
  {
    id: 'config.process[0].train.optimal_noise_pairing_samples',
    label: 'Optimal Noise Pairing Samples',
    description: 'Tries multiple noise patterns and picks the best one for each image. More samples = better quality but slower. 0 = disabled.',
    type: 'number',
    defaultValue: 0,
    step: 'advanced',
    section: 'noise_scheduler',
    min: 0,
    max: 100,
    numberStep: 1,
  },
  {
    id: 'config.process[0].train.optimizer_params.weight_decay',
    label: 'Weight Decay',
    description: 'L2 regularization to prevent overfitting. Higher values = stronger regularization.',
    type: 'number',
    defaultValue: 0.01,
    step: 'optimizer',
    section: 'optimizer_advanced',
    min: 0,
    max: 1,
    numberStep: 0.001,
  },
  {
    id: 'config.process[0].train.show_turbo_outputs',
    label: 'Show Turbo Outputs',
    description: 'Shows what the fast/distilled version of the model produces. Only useful if training turbo models. Leave off for normal training.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'guidance_settings',
  },
  {
    id: 'config.process[0].train.train_turbo',
    label: 'Train Turbo Model',
    description: 'Trains a faster version of the model that generates images in fewer steps. Advanced feature - only enable if you understand distillation.',
    type: 'boolean',
    defaultValue: false,
    step: 'advanced',
    section: 'guidance_settings',
  },
  {
    id: 'config.process[0].train.weight_jitter',
    label: 'Weight Jitter',
    description: 'Adds tiny random changes to LoRA weights during training. Can help prevent overfitting. Try 0.001-0.01. 0 = disabled.',
    type: 'number',
    defaultValue: 0,
    step: 'regularization',
    section: 'lora_regularization',
    min: 0,
    max: 0.1,
    numberStep: 0.001,
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
