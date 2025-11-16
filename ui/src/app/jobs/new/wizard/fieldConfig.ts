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
