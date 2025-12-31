import { GroupedSelectOption, SelectOption, JobConfig } from '@/types';
import { defaultSliderConfig } from './jobConfig';

type Control = 'depth' | 'line' | 'pose' | 'inpaint';

type DisableableSections =
  | 'model.quantize'
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
  | 'sample.ctrl_img'
  | 'sample.multi_ctrl_imgs'
  | 'datasets.num_frames'
  | 'model.multistage'
  | 'model.layer_offloading'
  | 'model.low_vram'
  | 'model.qie.match_target_res'
  | 'model.assistant_lora_path';
type ModelGroup = 'image' | 'instruction' | 'video';

export interface ModelArch {
  name: string;
  label: string;
  group: ModelGroup;
  controls?: Control[];
  isVideoModel?: boolean;
  defaults?: { [key: string]: any };
  disableSections?: DisableableSections[];
  additionalSections?: AdditionalSections[];
  accuracyRecoveryAdapters?: { [key: string]: string };
}

const defaultNameOrPath = '';

export const modelArchs: ModelArch[] = [
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
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram'],
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
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram'],
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
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram'],
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
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram'],
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
      'config.process[0].model.model_kwargs': [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram', 'model.multistage', 'model.layer_offloading'],
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
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram', 'datasets.do_i2v'],
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
      'config.process[0].sample.sample_steps': [8, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading', 'model.assistant_lora_path'],
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
  { value: 'qfloat8', label: 'float8 (default)' },
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
