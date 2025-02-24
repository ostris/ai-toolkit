export interface Model {
  name_or_path: string;
  dev_only?: boolean;
  defaults?: { [key: string]: any };
}

export interface Option {
  model: Model[];
}

export const options = {
  model: [
    {
      name_or_path: 'ostris/Flex.1-alpha',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.is_flux': [true, false],
        'config.process[0].train.bypass_guidance_embedding': [true, false],
      },
    },
    {
      name_or_path: 'black-forest-labs/FLUX.1-dev',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [true, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.is_flux': [true, false],
      },
    },
    {
      name_or_path: 'Alpha-VLLM/Lumina-Image-2.0',
      defaults: {
        // default updates when [selected, unselected] in the UI
        'config.process[0].model.quantize': [false, false],
        'config.process[0].model.quantize_te': [true, false],
        'config.process[0].model.is_lumina2': [true, false],
      },
    },
    {
      name_or_path: 'ostris/objective-reality',
      dev_only: true,
      defaults: {
      },
    },
  ],
} as Option;
