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
