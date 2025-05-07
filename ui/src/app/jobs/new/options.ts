export interface Model {
  name_or_path: string;
  arch: string;
  dev_only?: boolean;
  defaults?: { [key: string]: any };
}

export interface Option {
  model: Model[];
}

export interface ModelArch {
  name: string;
  label: string;
  defaults?: { [key: string]: [any, any] };
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
  },
  // { name: 'flex2', label: 'Flex.2' },
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
  },
  {
    name: 'wan21:1b',
    label: 'Wan 2.1 (1.3B)',
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
  },
  {
    name: 'wan21:14b',
    label: 'Wan 2.1 (14B)',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Wan-AI/Wan2.1-T2V-14B-Diffuserss', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [40, 1],
      'config.process[0].sample.fps': [15, 1],
    },
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
  },
];

export const isVideoModelFromArch = (arch: string) => {
  const videoArches = ['wan21'];
  return videoArches.includes(arch);
};

const defaultModelArch = 'flux';

export const options: Option = {
  model: [
    {
      name_or_path: 'ostris/Flex.1-alpha',
      arch: 'flex1',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].train.bypass_guidance_embedding': [true, false],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      },
    },
    {
      name_or_path: 'black-forest-labs/FLUX.1-dev',
      arch: 'flux',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      },
    },
    {
      name_or_path: 'lodestones/Chroma',
      arch: 'chroma',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      },
    },
    {
      name_or_path: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
      arch: 'wan21',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [false, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
        'config.process[0].sample.num_frames': [40, 1],
        'config.process[0].sample.fps': [15, 1],
      },
    },
    {
      name_or_path: 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
      arch: 'wan21',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
        'config.process[0].sample.num_frames': [40, 1],
        'config.process[0].sample.fps': [15, 1],
      },
    },
    {
      name_or_path: 'Alpha-VLLM/Lumina-Image-2.0',
      arch: 'lumina2',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [false, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      },
    },
    {
      name_or_path: 'HiDream-ai/HiDream-I1-Full',
      arch: 'hidream',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.lr': [0.0002, 0.0001],
        'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
        'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.experts', 'ff_i.gate'], []],
      },
    },
    {
      name_or_path: 'ostris/objective-reality',
      arch: 'sd1',
      dev_only: true,
      defaults: {
        'config.process[0].sample.sampler': ['ddpm', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['ddpm', 'flowmatch'],
      },
    },
  ],
} as Option;
