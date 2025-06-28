type Control = 'depth' | 'line' | 'pose' | 'inpaint';

type DisableableSections = 'model.quantize' | 'train.timestep_type' | 'network.conv';
type AdditionalSections = 'datasets.control_path' | 'sample.ctrl_img'

export interface ModelArch {
  name: string;
  label: string;
  controls?: Control[];
  isVideoModel?: boolean;
  defaults?: { [key: string]: any };
  disableSections?: DisableableSections[];
  additionalSections?: AdditionalSections[];
}

const defaultNameOrPath = '';


export const modelArchs: ModelArch[] = [
  {
    name: 'flux',
    label: 'FLUX.1',
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
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['lodestones/Chroma', defaultNameOrPath],
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
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-T2V-1.3B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [40, 1],
      'config.process[0].sample.fps': [15, 1],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'wan21:14b',
    label: 'Wan 2.1 (14B)',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-T2V-14B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [40, 1],
      'config.process[0].sample.fps': [15, 1],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'lumina2',
    label: 'Lumina2',
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
    name: 'hidream',
    label: 'HiDream',
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
  },
  {
    name: 'sdxl',
    label: 'SDXL',
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
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['OmniGen2/OmniGen2', defaultNameOrPath],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
    },
    disableSections: ['network.conv'],
  },
].sort((a, b) => {
  // Sort by label, case-insensitive
  return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
}) as any;
