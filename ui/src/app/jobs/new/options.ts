export interface Model {
  name_or_path: string;
  dev_only?: boolean;
  defaults?: { [key: string]: any };
}

export interface Option {
  model: Model[];
}

export const modelArchs = [
  { name: 'flux', label: 'Flux.1' },
  { name: 'wan21', label: 'Wan 2.1' },
  { name: 'lumina2', label: 'Lumina2' },
  { name: 'hidream', label: 'HiDream' },
];

export const isVideoModelFromArch = (arch: string) => {
  const videoArches = ['wan21'];
  return videoArches.includes(arch);
};

const defaultModelArch = 'flux';

export const options = {
  model: [
    {
      name_or_path: 'ostris/Flex.1-alpha',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.arch': ['flux', defaultModelArch],
        'config.process[0].train.bypass_guidance_embedding': [true, false],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      },
    },
    {
      name_or_path: 'black-forest-labs/FLUX.1-dev',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.arch': ['flux', defaultModelArch],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      },
    },
    {
      name_or_path: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [false, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.arch': ['wan21', defaultModelArch],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
        'config.process[0].sample.num_frames': [40, 1],
        'config.process[0].sample.fps': [15, 1],
      },
    },
    {
      name_or_path: 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.arch': ['wan21', defaultModelArch],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
        'config.process[0].sample.num_frames': [40, 1],
        'config.process[0].sample.fps': [15, 1],
      },
    },
    {
      name_or_path: 'Alpha-VLLM/Lumina-Image-2.0',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [false, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.arch': ['lumina2', defaultModelArch],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      },
    },
    {
      name_or_path: 'HiDream-ai/HiDream-I1-Full',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.arch': ['hidream', defaultModelArch],
        'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
        'config.process[0].train.lr': [0.0002, 0.0001],
        'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
        'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.experts', 'ff_i.gate'], []],
      },
    },
    {
      name_or_path: 'ostris/objective-reality',
      dev_only: true,
      defaults: {
        'config.process[0].sample.sampler': ['ddpm', 'flowmatch'],
        'config.process[0].train.noise_scheduler': ['ddpm', 'flowmatch'],
        'config.process[0].model.arch': ['sd1', defaultModelArch],
      },
    },
  ],
} as Option;
