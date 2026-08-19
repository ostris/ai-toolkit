import React from 'react';
import Link from 'next/link';
import { GroupedSelectOption, SelectOption, JobConfig, ConfigDoc } from '@/types';
import { defaultSliderConfig } from './jobConfig';
import { defaultAudioSampleConfig, defaultSampleConfig, defaultIdeogramSamplesConfig } from '@/helpers/defaultSamples';

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
  | 'train.audio_loss_multiplier'
  | 'datasets.num_frames'
  | 'model.multistage'
  | 'model.layer_offloading'
  | 'model.low_vram'
  | 'model.qie.match_target_res'
  | 'model.assistant_lora_path'
  | 'model.unconditional_lora_path'
  | 'model.model_kwargs.kv_cache'
  | 'ideogram_4_prompt';

type ModelGroup = 'image' | 'instruction' | 'video' | 'experimental' | 'audio';

export interface CustomModelSelectOption {
  label: string;
  options: SelectOption[];
  getValue: (config: JobConfig) => string | undefined;
  onChange: (value: string, config: JobConfig, setJobConfig: (value: any, key: string) => void) => void;
  doc?: ConfigDoc;
}

export type SampleTag = {
  title: string;
  type: 'text' | 'multiline' | 'number';
  full?: boolean;
};

export interface SampleTags {
  [key: string]: SampleTag;
}

export interface ModelArch {
  name: string;
  label: string;
  group: ModelGroup;
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
  customModelSelectOptions?: CustomModelSelectOption[];
}

const defaultNameOrPath = '';
const defaultLinearRank = 32;

export const modelArchs: ModelArch[] = [
  {
    name: 'anima',
    label: 'Anima',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['circlestone-labs/Anima-Base-v1.0-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].model.qtype': ['', 'qfloat8'],
      'config.process[0].model.qtype_te': ['', 'qfloat8'],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].sample.neg': [
        'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, signature, artist name',
        '',
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'flux',
    label: 'FLUX.1',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.1-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
    gateUrl: 'https://huggingface.co/black-forest-labs/FLUX.1-dev',
  },
  {
    name: 'flux_kontext',
    label: 'FLUX.1-Kontext-dev',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.1-Kontext-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img'],
    gateUrl: 'https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev',
  },
  {
    name: 'flex1',
    label: 'Flex.1',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ostris/Flex.1-alpha', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.bypass_guidance_embedding': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'flex2',
    label: 'Flex.2',
    group: 'image',
    controls: ['depth', 'line', 'pose', 'inpaint'],
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ostris/Flex.2-preview', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.model_kwargs': [
        {
          invert_inpaint_mask_chance: 0.2,
          inpaint_dropout: 0.5,
          control_dropout: 0.5,
          inpaint_random_chance: 0.2,
          do_random_inpainting: true,
          random_blur_mask: true,
          random_dialate_mask: true,
        },
        {},
      ],
      'config.process[0].train.bypass_guidance_embedding': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'chroma',
    label: 'Chroma',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['lodestones/Chroma1-Base', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'zeta_chroma',
    label: 'Zeta Chroma',
    group: 'experimental',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': [
        'lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors',
        defaultNameOrPath,
      ],
      'config.process[0].model.extras_name_or_path': ['Tongyi-MAI/Z-Image-Turbo', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'wan21:1b',
    label: 'Wan 2.1 (1.3B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-T2V-1.3B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].datasets[x].fps': [16, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram', 'datasets.auto_frame_count'],
  },
  {
    name: 'wan21_i2v:14b480p',
    label: 'Wan 2.1 I2V (14B-480P)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].fps': [16, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram', 'datasets.auto_frame_count'],
  },
  {
    name: 'wan21_i2v:14b',
    label: 'Wan 2.1 I2V (14B-720P)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-I2V-14B-720P-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].fps': [16, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram', 'datasets.auto_frame_count'],
  },
  {
    name: 'wan21:14b',
    label: 'Wan 2.1 (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-T2V-14B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].datasets[x].fps': [16, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram', 'datasets.auto_frame_count'],
  },
  {
    name: 'wan22_14b:t2v',
    label: 'Wan 2.2 (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].datasets[x].fps': [16, undefined],
      'config.process[0].model.model_kwargs': [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.num_frames',
      'model.low_vram',
      'model.multistage',
      'model.layer_offloading',
      'datasets.auto_frame_count',
    ],
    accuracyRecoveryAdapters: {
      // '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint3.safetensors',
      '4 bit with ARA': 'uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors',
    },
  },
  {
    name: 'wan22_14b_i2v',
    label: 'Wan 2.2 I2V (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].datasets[x].fps': [16, undefined],
      'config.process[0].model.model_kwargs': [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.low_vram',
      'model.multistage',
      'model.layer_offloading',
      'datasets.auto_frame_count',
    ],
    accuracyRecoveryAdapters: {
      '4 bit with ARA': 'uint4|ostris/accuracy_recovery_adapters/wan22_14b_i2v_torchao_uint4.safetensors',
    },
  },
  {
    name: 'wan22_5b',
    label: 'Wan 2.2 TI2V (5B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.2-TI2V-5B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.low_vram',
      'datasets.do_i2v',
      'datasets.auto_frame_count',
    ],
  },
  {
    name: 'lumina2',
    label: 'Lumina2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Alpha-VLLM/Lumina-Image-2.0', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'qwen_image',
    label: 'Qwen-Image',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image:2512',
    label: 'Qwen-Image-2512',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image-2512', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
    // Training an ARA now, the other one will not work
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint3.safetensors',
      '4 bit with ARA': 'uint4|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint4.safetensors',
    },
  },
  {
    name: 'qwen_image_edit',
    label: 'Qwen-Image-Edit',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image-Edit', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img', 'model.low_vram', 'model.layer_offloading'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image_edit_plus',
    label: 'Qwen-Image-Edit-2509',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image-Edit-2509', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2509_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image_edit_plus:2511',
    label: 'Qwen-Image-Edit-2511',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Qwen/Qwen-Image-Edit-2511', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors',
    },
  },
  {
    name: 'hidream',
    label: 'HiDream',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['HiDream-ai/HiDream-I1-Full', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.lr': [0.0002, 0.0001],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.experts', 'ff_i.gate'], []],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/hidream_i1_full_torchao_uint3.safetensors',
    },
  },
  {
    name: 'hidream_e1',
    label: 'HiDream E1',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['HiDream-ai/HiDream-E1-1', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.lr': [0.0001, 0.0001],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.experts', 'ff_i.gate'], []],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img', 'model.low_vram'],
  },
  {
    name: 'sdxl',
    label: 'SDXL',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['stabilityai/stable-diffusion-xl-base-1.0', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].sample.sampler': ['ddpm', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['ddpm', 'flowmatch'],
      'config.process[0].sample.guidance_scale': [6, 4],
    },
    disableSections: ['model.quantize', 'train.timestep_type'],
  },
  {
    name: 'sd15',
    label: 'SD 1.5',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['stable-diffusion-v1-5/stable-diffusion-v1-5', defaultNameOrPath],
      'config.process[0].sample.sampler': ['ddpm', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['ddpm', 'flowmatch'],
      'config.process[0].sample.width': [512, 1024],
      'config.process[0].sample.height': [512, 1024],
      'config.process[0].sample.guidance_scale': [6, 4],
    },
    disableSections: ['model.quantize', 'train.timestep_type'],
  },
  {
    name: 'omnigen2',
    label: 'OmniGen2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['OmniGen2/OmniGen2', defaultNameOrPath],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img'],
  },
  {
    name: 'flux2',
    label: 'FLUX.2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.2-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    gateUrl: 'https://huggingface.co/black-forest-labs/FLUX.2-dev',
  },
  {
    name: 'zimage:turbo',
    label: 'Z-Image Turbo (w/ Training Adapter)',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Tongyi-MAI/Z-Image-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.assistant_lora_path': [
        'ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [9, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading', 'model.assistant_lora_path'],
  },
  {
    name: 'zimage',
    label: 'Z-Image',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Tongyi-MAI/Z-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample.sample_steps': [30, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'zimage:deturbo',
    label: 'Z-Image De-Turbo (De-Distilled)',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['ostris/Z-Image-De-Turbo', defaultNameOrPath],
      'config.process[0].model.extras_name_or_path': ['Tongyi-MAI/Z-Image-Turbo', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample.guidance_scale': [3, 4],
      'config.process[0].sample.sample_steps': [25, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'minimax_h3',
    label: 'MiniMax-H3',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Comfy-Org/MiniMax-H3', defaultNameOrPath],
      // the Comfy-Org weights are pre-quantized (int8 convrot DiT, nvfp4 TE); these
      // qtypes match the checkpoints exactly, so the load is unchanged. Picking a
      // different qtype re-quantizes layer by layer into that format.
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.qtype': ['convrot8', 'qfloat8'],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.qtype_te': ['nvfp4', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.cache_text_embeddings': [true, false],
      'config.process[0].train.do_guidance_loss': [true, undefined],
      'config.process[0].train.guidance_loss_target': [3.5, undefined],
      'config.process[0].model.assistant_lora_path': [
        'ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors',
        undefined,
      ],
      'config.process[0].network.linear': [16, defaultLinearRank],
      'config.process[0].network.linear_alpha': [16, defaultLinearRank],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['adaln_proj'], []],
      'config.process[0].sample.num_frames': [107, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [28, 25],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].num_frames': [39, undefined],
      'config.process[0].datasets[x].auto_frame_count': [true, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
      'model.assistant_lora_path',
    ],
    customModelSelectOptions: [
      {
        label: 'Distillation Handling Method',
        options: [
          { value: 'cg', label: 'Contrastive Guidance' },
          { value: 'ta', label: 'Training Adapter' },
          { value: 'both', label: 'Contrastive Guidance + Training Adapter (default)' },
        ],
        getValue: (config: JobConfig) => {
          const assistantLoraPath = config?.config?.process?.[0]?.model?.assistant_lora_path;
          const hasAssistantLoraPath = assistantLoraPath && assistantLoraPath.trim() !== '';
          const hasContrastiveGuidance = config?.config?.process?.[0]?.train?.do_guidance_loss;
          if (hasAssistantLoraPath && hasContrastiveGuidance) {
            return 'both';
          }
          if (hasAssistantLoraPath) {
            return 'ta';
          }
          return 'cg';
        },
        onChange: (value: string, config: JobConfig, setJobConfig: (value: any, key: string) => void) => {
          if (value === 'cg') {
            setJobConfig(true, 'config.process[0].train.do_guidance_loss');
            setJobConfig(undefined, 'config.process[0].model.assistant_lora_path');
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, 'config.process[0].train.guidance_loss_target');
            }
          } else if (value === 'ta') {
            setJobConfig(undefined, 'config.process[0].train.do_guidance_loss');
            setJobConfig(undefined, 'config.process[0].train.guidance_loss_target');
            setJobConfig(
              'ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors',
              'config.process[0].model.assistant_lora_path',
            );
          } else if (value === 'both') {
            setJobConfig(true, 'config.process[0].train.do_guidance_loss');
            setJobConfig(
              'ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors',
              'config.process[0].model.assistant_lora_path',
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, 'config.process[0].train.guidance_loss_target');
            }
          }
        },
        doc: {
          title: 'MiniMax-H3 Distillation Handling',
          description: (
            <div>
              MiniMax H3 is a guidance distilled model, so training on it directly will make the guidance distillation
              break down. There are two different ways to train on this model without breaking the guidance
              distillation: Contrastive Guidance and Training Adapter. Both have their pros and cons. The adapter is
              faster, but will still break down over a long run. Contrastive Guidance is slower, but is less likely to
              break down.
            </div>
          ),
        },
      },
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          Weights load from the{' '}
          <Link href="/settings" className="text-blue-400 hover:underline">
            Models Folder Path
          </Link>{' '}
          set in settings. Anything missing is downloaded there from <code>Comfy-Org/MiniMax-H3</code> on first load
          (~43GB total). Files used:
        </p>
        <pre className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs overflow-x-auto">
          <code>{`<MODELS_PATH>/
├── diffusion_models/
│   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors`}</code>
        </pre>
        <p>
          The checkpoints are pre-quantized and load directly: int8 ConvRot DiT (~21GB) and nvfp4 Qwen3-VL text encoder
          (~16GB). The default qtypes (<code>convrot8</code> / <code>nvfp4</code>) match the files exactly, so nothing
          is re-quantized on load. Picking a different quantization re-quantizes the pre-quantized layers into that
          format, one layer at a time.
        </p>
        <p>
          Supports t2v and first-frame i2v (ctrl img / i2v datasets) with joint audio. The model is guidance-distilled —
          keep guidance scale at 1. Video is fixed 24 fps and frame counts snap down to the 17n+5 grid (5, 22, 39, 56,
          ..., 107, 124 ≈ 5s). Image datasets (num_frames 1) train as single frames, and a sample with num_frames 1
          renders a single image.
        </p>
      </div>
    ),
  },
  {
    name: 'minimax_h3_ref2va',
    label: 'MiniMax-H3 Ref2V',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Comfy-Org/MiniMax-H3', defaultNameOrPath],
      // pre-quantized weights: matching qtypes keep the load unchanged
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.qtype': ['convrot8', 'qfloat8'],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.qtype_te': ['nvfp4', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.cache_text_embeddings': [true, false],
      'config.process[0].train.do_guidance_loss': [true, undefined],
      'config.process[0].train.guidance_loss_target': [3.5, undefined],
      'config.process[0].model.assistant_lora_path': [
        'ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v1.safetensors',
        undefined,
      ],
      'config.process[0].network.linear': [16, defaultLinearRank],
      'config.process[0].network.linear_alpha': [16, defaultLinearRank],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['adaln_proj'], []],
      'config.process[0].sample.num_frames': [107, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [28, 25],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].num_frames': [39, undefined],
      'config.process[0].datasets[x].auto_frame_count': [true, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.multi_ctrl_imgs',
      'datasets.multi_control_paths',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
      'model.assistant_lora_path',
    ],
    customModelSelectOptions: [
      {
        label: 'Distillation Handling Method',
        options: [
          { value: 'cg', label: 'Contrastive Guidance' },
          { value: 'ta', label: 'Training Adapter' },
          { value: 'both', label: 'Contrastive Guidance + Training Adapter (default)' },
        ],
        getValue: (config: JobConfig) => {
          const assistantLoraPath = config?.config?.process?.[0]?.model?.assistant_lora_path;
          const hasAssistantLoraPath = assistantLoraPath && assistantLoraPath.trim() !== '';
          const hasContrastiveGuidance = config?.config?.process?.[0]?.train?.do_guidance_loss;
          if (hasAssistantLoraPath && hasContrastiveGuidance) {
            return 'both';
          }
          if (hasAssistantLoraPath) {
            return 'ta';
          }
          return 'cg';
        },
        onChange: (value: string, config: JobConfig, setJobConfig: (value: any, key: string) => void) => {
          if (value === 'cg') {
            setJobConfig(true, 'config.process[0].train.do_guidance_loss');
            setJobConfig(undefined, 'config.process[0].model.assistant_lora_path');
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, 'config.process[0].train.guidance_loss_target');
            }
          } else if (value === 'ta') {
            setJobConfig(undefined, 'config.process[0].train.do_guidance_loss');
            setJobConfig(undefined, 'config.process[0].train.guidance_loss_target');
            setJobConfig(
              'ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v1.safetensors',
              'config.process[0].model.assistant_lora_path',
            );
          } else if (value === 'both') {
            setJobConfig(true, 'config.process[0].train.do_guidance_loss');
            setJobConfig(
              'ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v1.safetensors',
              'config.process[0].model.assistant_lora_path',
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, 'config.process[0].train.guidance_loss_target');
            }
          }
        },
        doc: {
          title: 'MiniMax-H3 Distillation Handling',
          description: (
            <div>
              MiniMax H3 is a guidance distilled model, so training on it directly will make the guidance distillation
              break down. There are two different ways to train on this model without breaking the guidance
              distillation: Contrastive Guidance and Training Adapter. Both have their pros and cons. The adapter is
              faster, but will still break down over a long run. Contrastive Guidance is slower, but is less likely to
              break down.
            </div>
          ),
        },
      },
      {
        label: 'Image Reference Presentation',
        options: [
          { value: 'picture', label: 'Picture (default)' },
          { value: 'video', label: 'Static video clip' },
        ],
        getValue: (config: JobConfig) => {
          return config?.config?.process?.[0]?.model?.model_kwargs?.image_refs_as_video ? 'video' : 'picture';
        },
        onChange: (value: string, config: JobConfig, setJobConfig: (value: any, key: string) => void) => {
          const kwargs = { ...(config?.config?.process?.[0]?.model?.model_kwargs ?? {}) };
          if (value === 'video') {
            kwargs.image_refs_as_video = true;
          } else {
            delete kwargs.image_refs_as_video;
            delete kwargs.image_ref_video_frames;
          }
          setJobConfig(kwargs, 'config.process[0].model.model_kwargs');
        },
        doc: {
          title: 'MiniMax-H3 Image Reference Presentation',
          description: (
            <div className="space-y-2">
              <p>
                How still-image references (dataset control images and sample ctrl images) are shown to the model.
                Video references always use the video path.
              </p>
              <p>
                <strong>Picture</strong>: the native ref2va recipe — a single-frame reference block, shown to Qwen3-VL as
                a <code>&lt;Picture i&gt;</code> block, scaled down only.
              </p>
              <p>
                <strong>Static video clip</strong>: the image is held for 5 frames (2 latent frames) and routed through
                the exact path a reference VIDEO takes — video sizing (matched to the target's pixel area), multi-frame
                reference block, <code>&lt;Video k&gt;</code> timestamped presentation. Use this when training on image
                references but sampling with video references, so the LoRA learns the pathway it will be used through.
                Adds a handful of rows per reference. Frame count is adjustable with{' '}
                <code>model_kwargs.image_ref_video_frames</code> (17n+5). Changing this re-caches text embeddings.
              </p>
            </div>
          ),
        },
      },
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          Reference-to-video: control images and videos condition the output as subject/style references (never as a
          first frame). References keep their own aspect and are matched to the target's pixel area (images scale down
          only, never up; a same-aspect video reference is exactly the target size). Each rides into the packed sequence
          as a reference block, and is also shown to the Qwen3-VL conditioner as a <code>&lt;Picture i&gt;</code> (image)
          or timestamped <code>&lt;Video k&gt;</code> (video) vision block. Training references come from the dataset
          control path(s); sampling uses the sample ctrl images — always as references. The Image Reference Presentation
          option can route still images through the video-reference path as short static clips.
        </p>
        <p>
          Weights load like MiniMax-H3 (see that arch's notes) from the{' '}
          <Link href="/settings" className="text-blue-400 hover:underline">
            Models Folder Path
          </Link>
          , using <code>diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors</code> — the ref2va partition
          of the same release; text encoder and VAEs are shared with the fl2va arch. Everything else (pre-quantized
          load, 24 fps, 17n+5 frame grid, guidance scale 1, single-image mode) matches MiniMax-H3.
        </p>
      </div>
    ),
  },
  {
    name: 'ltx2',
    label: 'LTX-2',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Lightricks/LTX-2', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].auto_frame_count': [false, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
    ],
  },
  {
    name: 'ltx2.3',
    label: 'LTX-2.3',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].auto_frame_count': [false, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
    ],
  },
  {
    name: 'ltx2.5',
    label: 'LTX-2.5',
    gateUrl: 'https://huggingface.co/Lightricks/LTX-2.5',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      // comfy-style split files resolve from/download to the models folder;
      // the int8 ConvRot dev transformer is the default
      'config.process[0].model.name_or_path': ['Lightricks/LTX-2.5', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.qtype': ['convrot8', 'qfloat8'],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.qtype_te': ['convrot8', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].auto_frame_count': [false, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
    ],
  },
  {
    name: 'flux2_klein_4b',
    label: 'FLUX.2-klein-base-4B',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.2-klein-base-4B', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
  {
    name: 'ernie_image',
    label: 'ERNIE-Image',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['baidu/ERNIE-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'flux2_klein_9b',
    label: 'FLUX.2-klein-base-9B',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.2-klein-base-9B', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    gateUrl: 'https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B',
  },
  {
    name: 'ace_step_15_xl',
    label: 'ACE-Step 1.5 XL',
    group: 'audio',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': [
        'ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_xl_base_aio.safetensors',
        defaultNameOrPath,
      ],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample': [defaultAudioSampleConfig, defaultSampleConfig],
    },
    sampleTags: {
      CAPTION: {
        title: 'Audio Prompt',
        type: 'text',
        full: true,
      },
      LYRICS: {
        title: 'Lyrics',
        type: 'multiline',
        full: true,
      },
      BPM: {
        title: 'BPM',
        type: 'number',
      },
      KEYSCALE: {
        title: 'Key Scale',
        type: 'text',
      },
      TIMESIGNATURE: {
        title: 'Time Signature',
        type: 'text',
      },
      DURATION: {
        title: 'Duration (sec)',
        type: 'number',
      },
      LANGUAGE: {
        title: 'Language',
        type: 'text',
      },
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.multi_ctrl_imgs', 'model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'ace_step_15',
    label: 'ACE-Step 1.5',
    group: 'audio',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': [
        'ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors',
        defaultNameOrPath,
      ],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample': [defaultAudioSampleConfig, defaultSampleConfig],
    },
    sampleTags: {
      CAPTION: {
        title: 'Audio Prompt',
        type: 'text',
        full: true,
      },
      LYRICS: {
        title: 'Lyrics',
        type: 'multiline',
        full: true,
      },
      BPM: {
        title: 'BPM',
        type: 'number',
      },
      KEYSCALE: {
        title: 'Key Scale',
        type: 'text',
      },
      TIMESIGNATURE: {
        title: 'Time Signature',
        type: 'text',
      },
      DURATION: {
        title: 'Duration (sec)',
        type: 'number',
      },
      LANGUAGE: {
        title: 'Language',
        type: 'text',
      },
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.multi_ctrl_imgs', 'model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'nucleus_image',
    label: 'Nucleus-Image',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['NucleusAI/Nucleus-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['img_mlp.experts', 'img_mlp.gate'], []],
      'config.process[0].network.linear': [128, defaultLinearRank],
      'config.process[0].network.linear_alpha': [128, defaultLinearRank],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram'],
  },
  {
    name: 'hidream_o1',
    label: 'HiDream-O1',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['HiDream-ai/HiDream-O1-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].train.max_loss': [1.0, undefined],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['lm_head', 'patch_embed', 'visual'], []],
      'config.process[0].network.transformer_only': [false, undefined],
      'config.process[0].sample.width': [2048, 1024],
      'config.process[0].sample.height': [2048, 1024],
      'config.process[0].model.model_kwargs': [
        {
          noise_scale_inference: 8.0,
          noise_scale: 8.0,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'model.quantize_te', 'train.unload_text_encoder'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'zimage_l2p',
    label: 'Z-Image L2P (pixel space)',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['zhen-nan/L2P/model-1k-merge.safetensors', defaultNameOrPath],
      'config.process[0].model.extras_name_or_path': ['Tongyi-MAI/Z-Image-Turbo', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'ideogram4',
    label: 'Ideogram4',
    group: 'experimental',
    defaults: {
      'config.process[0].model.name_or_path': ['ideogram-ai/ideogram-4-fp8', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample': [defaultIdeogramSamplesConfig, defaultSampleConfig],
      'config.process[0].model.unconditional_lora_path': [
        'ostris/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors',
        undefined,
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'model.low_vram',
      'model.layer_offloading',
      'ideogram_4_prompt',
      'model.unconditional_lora_path',
    ],
    hasMultiLinePrompts: true,
    gateUrl: 'https://huggingface.co/ideogram-ai/ideogram-4-fp8',
  },
  {
    name: 'prx_pixel',
    label: 'PRXPixel (pixel space)',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['Photoroom/prxpixel-t2i', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'krea2',
    label: 'Krea 2 (raw)',
    group: 'image',
    gateUrl: 'https://huggingface.co/krea/Krea-2-Raw',
    defaults: {
      'config.process[0].model.name_or_path': ['krea/Krea-2-Raw', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'krea2:turbo',
    label: 'Krea 2 Turbo (w/ Training Adapter)',
    group: 'image',
    gateUrl: 'https://huggingface.co/krea/Krea-2-Turbo',
    defaults: {
      'config.process[0].model.name_or_path': ['krea/Krea-2-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].model.assistant_lora_path': [
        'ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [9, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading', 'model.assistant_lora_path'],
  },
  {
    name: 'krea2:o_edit',
    label: 'Krea 2 (raw) [Edit Training]',
    gateUrl: 'https://huggingface.co/krea/Krea-2-Raw',
    group: 'experimental',
    defaults: {
      'config.process[0].model.name_or_path': ['krea/Krea-2-Raw', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].model.model_kwargs': [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
      'model.model_kwargs.kv_cache',
    ],
  },
  {
    name: 'krea2:o_edit_turbo',
    label: 'Krea 2 Turbo (w/ Training Adapter) [Edit Training]',
    gateUrl: 'https://huggingface.co/krea/Krea-2-Turbo',
    group: 'experimental',
    defaults: {
      'config.process[0].model.name_or_path': ['krea/Krea-2-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].model.assistant_lora_path': [
        'ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [8, 25],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].model.model_kwargs': [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.assistant_lora_path',
      'model.qie.match_target_res',
      'model.model_kwargs.kv_cache',
    ],
  },
  {
    name: 'mageflow',
    label: 'Mage-Flow',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['microsoft/Mage-Flow-Base', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.guidance_scale': [4, 4],
      'config.process[0].sample.sample_steps': [25, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'mageflow_edit',
    label: 'Mage-Flow Edit',
    group: 'instruction',
    defaults: {
      'config.process[0].model.name_or_path': ['microsoft/Mage-Flow-Edit-Base', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.guidance_scale': [4, 4],
      'config.process[0].sample.sample_steps': [25, 25],
      'config.process[0].train.unload_text_encoder': [false, false],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
    ],
  },
  {
    name: 'boogu_image',
    label: 'Boogu Image',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['Boogu/Boogu-Image-0.1-Base', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'boogu_image_edit',
    label: 'Boogu Image Edit',
    group: 'instruction',
    defaults: {
      'config.process[0].model.name_or_path': ['Boogu/Boogu-Image-0.1-Edit', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
].sort((a, b) => {
  // Sort by label, case-insensitive
  return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' });
}) as any;

export const groupedModelOptions: GroupedSelectOption[] = modelArchs.reduce((acc, arch) => {
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
