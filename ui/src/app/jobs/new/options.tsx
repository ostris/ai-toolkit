import React from 'react';
import { GroupedSelectOption, SelectOption, JobConfig, ConfigDoc } from '@/types';
import { defaultSliderConfig } from './jobConfig';

type Control = 'depth' | 'line' | 'pose' | 'inpaint';

type DisableableSections =
  | 'model.quantize'
  | 'model.quantize_te'
  | 'train.timestep_type'
  | 'network.conv'
  | 'trigger_word'
  | 'train.diff_output_preservation'
  | 'train.blank_prompt_preservation'
  | 'train.unload_text_encoder'
  | 'slider';

type AdditionalSections =
  | 'datasets.control_path'
  | 'datasets.multi_control_paths'
  | 'datasets.do_i2v'
  | 'datasets.do_audio'
  | 'datasets.audio_normalize'
  | 'datasets.audio_preserve_pitch'
  | 'datasets.auto_frame_count'
  | 'sample.ctrl_img'
  | 'sample.multi_ctrl_imgs'
  | 'sample.duration'
  | 'train.audio_loss_multiplier'
  | 'datasets.num_frames'
  | 'model.multistage'
  | 'model.layer_offloading'
  | 'model.low_vram'
  | 'model.qie.match_target_res'
  | 'model.assistant_lora_path'
  | 'model.unconditional_lora_path'
  | 'model.model_kwargs.kv_cache'
  | 'model.model_kwargs.instruction'
  | 'ideogram_4_prompt';

type ModelGroup = 'image' | 'instruction' | 'video' | 'experimental' | 'audio' | 'llm';

export interface CustomModelSelectOption {
  type?: 'select';
  label: string;
  options: SelectOption[];
  getValue: (config: JobConfig) => string | undefined;
  onChange: (value: string, config: JobConfig, setJobConfig: (value: any, key: string) => void) => void;
  doc?: ConfigDoc;
}

export interface CustomModelCheckboxOption {
  type: 'checkbox';
  label: string;
  getValue: (config: JobConfig) => boolean;
  onChange: (value: boolean, config: JobConfig, setJobConfig: (value: any, key: string) => void) => void;
  doc?: ConfigDoc;
}

export type CustomModelOption = CustomModelSelectOption | CustomModelCheckboxOption;

export type SampleTag = {
  title: string;
  type: 'text' | 'multiline' | 'number';
  full?: boolean;
};

export interface SampleTags {
  [key: string]: SampleTag;
}

export type GenerateModality = 'image' | 'video' | 'audio';

// Per-arch overrides for the Generate page. Everything it needs is derived
// from the training entry (name_or_path / quantize defaults, video/audio
// group, ctrl_img section); set these only where the derivation is wrong.
export interface GenerateOptions {
  modality?: GenerateModality;
  model?: { [key: string]: any }; // extra ModelConfig kwargs
  sample?: { [key: string]: any }; // GenerateImageConfig kwargs
  needsControlImage?: boolean;
  sizeLocked?: boolean;
}

export interface GenerateDefaults {
  arch: string;
  label: string;
  group: ModelGroup;
  modality: GenerateModality;
  model: { [key: string]: any };
  sample: { [key: string]: any };
  needsControlImage: boolean;
  sizeLocked: boolean;
  /** structured prompt fields (audio models): the prompt is their tagged form */
  sampleTags?: SampleTags;
}

export interface ModelArch {
  name: string;
  label: string;
  /** label shown by the Generate page instead of `label` (training-specific
   * wording like "w/ Training Adapter" does not apply to inference) */
  generateNameOverride?: string;
  group: ModelGroup;
  generate?: GenerateOptions;
  controls?: Control[];
  isVideoModel?: boolean;
  hasMultiLinePrompts?: boolean;
  defaults?: { [key: string]: any };
  disableSections?: DisableableSections[];
  additionalSections?: AdditionalSections[];
  accuracyRecoveryAdapters?: { [key: string]: string };
  sampleTags?: SampleTags;
  gateUrl?: string;
  modelNotes?: React.ReactNode;
  customModelSelectOptions?: CustomModelOption[];
}

/** Grouped select options for an arch list (see useModelArchs). */
export const groupModelOptions = (archs: ModelArch[]): GroupedSelectOption[] =>
  archs.reduce((acc, arch) => {
    const group = acc.find(g => g.label === arch.group);
    if (group) {
      group.options.push({ value: arch.name, label: arch.label });
    } else {
      acc.push({
        label: arch.group,
        options: [{ value: arch.name, label: arch.label }],
      });
    }
    return acc;
  }, [] as GroupedSelectOption[]);

export const quantizationOptions: SelectOption[] = [
  { value: '', label: '- NONE -' },
  { value: 'qfloat8', label: 'qfloat8 (default)' },
  { value: 'float8', label: 'float8' },
  { value: 'convrot8', label: '8bit convrot' },
  { value: 'convrot4', label: '4bit convrot (nvfp4)' },
  { value: 'nvfp4', label: 'nvfp4 (4bit weight only)' },
  { value: 'convrotint7', label: '7bit convrot' },
  { value: 'convrotint6', label: '6bit convrot' },
  { value: 'convrotint5', label: '5bit convrot' },
  { value: 'convrotint4', label: '4bit convrot' },
  { value: 'convrotint3', label: '3bit convrot' },
  { value: 'convrotint2', label: '2bit convrot' },
  { value: 'convrotbitnet', label: '1.58bit convrot (bitnet)' },
  { value: 'uint7', label: '7 bit' },
  { value: 'uint6', label: '6 bit' },
  { value: 'uint5', label: '5 bit' },
  { value: 'uint4', label: '4 bit' },
  { value: 'uint3', label: '3 bit' },
  { value: 'uint2', label: '2 bit' },
];

export const defaultQtype = 'qfloat8';

interface JobTypeOption extends SelectOption {
  disableSections?: DisableableSections[];
  processSections?: string[];
  onActivate?: (config: JobConfig) => JobConfig;
  onDeactivate?: (config: JobConfig) => JobConfig;
}

export const jobTypeOptions: JobTypeOption[] = [
  {
    value: 'diffusion_trainer',
    label: 'LoRA Trainer',
    disableSections: ['slider'],
  },
  {
    value: 'concept_slider',
    label: 'Concept Slider',
    disableSections: ['trigger_word', 'train.diff_output_preservation'],
    onActivate: (config: JobConfig) => {
      // add default slider config
      config.config.process[0].slider = { ...defaultSliderConfig };
      return config;
    },
    onDeactivate: (config: JobConfig) => {
      // remove slider config
      delete config.config.process[0].slider;
      return config;
    },
  },
];

const MODEL_PREFIX = 'config.process[0].model.';
const SAMPLE_PREFIX = 'config.process[0].sample';
// training-only model settings: the training adapter (e.g. Z-Image Turbo's
// de-distill LoRA) and the unconditional LoRA must not load for inference
const TRAINING_ONLY_MODEL_KEYS = new Set(['assistant_lora_path', 'unconditional_lora_path', 'inference_lora_path']);

/** What the Generate page sends the inference engine for an arch: ModelConfig
 * kwargs + GenerateImageConfig kwargs, derived from the training defaults. */
export const getGenerateDefaults = (arch: ModelArch): GenerateDefaults => {
  const defaults = arch.defaults || {};
  const model: { [key: string]: any } = {};
  const sample: { [key: string]: any } = { width: 1024, height: 1024, num_inference_steps: 25, guidance_scale: 4 };
  for (const [key, pair] of Object.entries(defaults)) {
    const value = Array.isArray(pair) ? pair[0] : pair;
    if (key.startsWith(MODEL_PREFIX)) {
      const field = key.slice(MODEL_PREFIX.length);
      if (value === '' || value === undefined || value === null) continue;
      if (field.includes('.')) continue; // nested model_kwargs etc.
      if (TRAINING_ONLY_MODEL_KEYS.has(field)) continue;
      model[field] = value;
    } else if (key === SAMPLE_PREFIX && value && typeof value === 'object') {
      // whole SampleConfig object (audio models)
      const sc = value as any;
      if (sc.width) sample.width = sc.width;
      if (sc.height) sample.height = sc.height;
      if (sc.sample_steps) sample.num_inference_steps = sc.sample_steps;
      if (sc.guidance_scale !== undefined) sample.guidance_scale = sc.guidance_scale;
      if (sc.num_frames) sample.num_frames = sc.num_frames;
      if (sc.fps) sample.fps = sc.fps;
      if (sc.duration) sample.duration = sc.duration;
      if (sc.neg) sample.negative_prompt = sc.neg;
    } else if (key.startsWith(SAMPLE_PREFIX + '.')) {
      const field = key.slice(SAMPLE_PREFIX.length + 1);
      if (value === undefined || value === null || value === '') continue;
      if (field === 'sample_steps') sample.num_inference_steps = value;
      else if (field === 'neg') sample.negative_prompt = value;
      else if (['width', 'height', 'guidance_scale', 'num_frames', 'fps', 'duration'].includes(field))
        sample[field] = value;
    }
  }
  // the engine defaults to convrot8; the training default qtype is the
  // quanto one, which is the slower inference choice
  if (model.quantize && (!model.qtype || model.qtype === 'qfloat8')) model.qtype = 'convrot8';
  if (model.quantize_te && (!model.qtype_te || model.qtype_te === 'qfloat8')) model.qtype_te = 'convrot8';
  const sections = arch.additionalSections || [];
  const gen = arch.generate || {};
  const modality: GenerateModality =
    gen.modality || (arch.group === 'audio' ? 'audio' : arch.isVideoModel ? 'video' : 'image');
  if (modality !== 'video') {
    delete sample.num_frames;
    delete sample.fps;
  }
  return {
    arch: arch.name,
    label: arch.generateNameOverride || arch.label,
    group: arch.group,
    modality,
    model: { ...model, ...(gen.model || {}) },
    sample: { ...sample, ...(gen.sample || {}) },
    needsControlImage:
      gen.needsControlImage ?? (sections.includes('sample.ctrl_img') || sections.includes('sample.multi_ctrl_imgs')),
    sizeLocked: gen.sizeLocked ?? false,
    sampleTags: arch.sampleTags,
  };
};
