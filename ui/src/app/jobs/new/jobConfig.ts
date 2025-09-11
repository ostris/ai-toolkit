import { JobConfig, DatasetConfig } from '@/types';

export const defaultDatasetConfig: DatasetConfig = {
  folder_path: '/path/to/images/folder',
  control_path: null,
  mask_path: null,
  mask_min_value: 0.1,
  default_caption: '',
  caption_ext: 'txt',
  caption_dropout_rate: 0.05,
  cache_latents_to_disk: false,
  is_reg: false,
  network_weight: 1,
  resolution: [512, 768, 1024],
  controls: [],
  shrink_video_to_frames: true,
  num_frames: 1,
  do_i2v: true,
  flip_x: false,
  flip_y: false,
};

export const defaultJobConfig: JobConfig = {
  job: 'extension',
  config: {
    name: 'my_first_lora_v1',
    process: [
      {
        type: 'ui_trainer',
        training_folder: 'output',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        trigger_word: null,
        performance_log_every: 10,
        network: {
          type: 'lora',
          linear: 32,
          linear_alpha: 32,
          conv: 16,
          conv_alpha: 16,
          lokr_full_rank: true,
          lokr_factor: -1,
          network_kwargs: {
            ignore_if_contains: [],
          },
        },
        save: {
          dtype: 'bf16',
          save_every: 250,
          max_step_saves_to_keep: 4,
          save_format: 'diffusers',
          push_to_hub: false,
        },
        datasets: [defaultDatasetConfig],
        train: {
          batch_size: 1,
          bypass_guidance_embedding: true,
          steps: 3000,
          gradient_accumulation: 1,
          train_unet: true,
          train_text_encoder: false,
          gradient_checkpointing: true,
          noise_scheduler: 'flowmatch',
          optimizer: 'adamw8bit',
          timestep_type: 'sigmoid',
          content_or_style: 'balanced',
          optimizer_params: {
            weight_decay: 1e-4,
          },
          unload_text_encoder: false,
          cache_text_embeddings: false,
          lr: 0.0001,
          ema_config: {
            use_ema: false,
            ema_decay: 0.99,
          },
          skip_first_sample: false,
          force_first_sample: false,
          disable_sampling: false,
          dtype: 'bf16',
          diff_output_preservation: false,
          diff_output_preservation_multiplier: 1.0,
          diff_output_preservation_class: 'person',
          switch_boundary_every: 1,
        },
        model: {
          name_or_path: 'ostris/Flex.1-alpha',
          quantize: true,
          qtype: 'qfloat8',
          quantize_te: true,
          qtype_te: 'qfloat8',
          arch: 'flex1',
          low_vram: false,
          model_kwargs: {},
        },
        sample: {
          sampler: 'flowmatch',
          sample_every: 250,
          width: 1024,
          height: 1024,
          samples: [
            {
              prompt: 'woman with red hair, playing chess at the park, bomb going off in the background'
            },
            {
              prompt: 'a woman holding a coffee cup, in a beanie, sitting at a cafe',
            },
            {
              prompt: 'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
            },
            {
              prompt: 'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
            },
            {
              prompt: 'a bear building a log cabin in the snow covered mountains',
            },
            {
              prompt: 'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
            },
            {
              prompt: 'hipster man with a beard, building a chair, in a wood shop',
            },
            {
              prompt: 'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
            },
            {
              prompt: "a man holding a sign that says, 'this is a sign'",
            },
            {
              prompt: 'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle',
            },
          ],
          neg: '',
          seed: 42,
          walk_seed: true,
          guidance_scale: 4,
          sample_steps: 25,
          num_frames: 1,
          fps: 1,
        },
      },
    ],
  },
  meta: {
    name: '[name]',
    version: '1.0',
  },
};

export const migrateJobConfig = (jobConfig: JobConfig): JobConfig => {
  // upgrade prompt strings to samples
  if (
    jobConfig?.config?.process &&
    jobConfig.config.process[0]?.sample &&
    Array.isArray(jobConfig.config.process[0].sample.prompts) &&
    jobConfig.config.process[0].sample.prompts.length > 0
  ) {
    let newSamples = [];
    for (const prompt of jobConfig.config.process[0].sample.prompts) {
      newSamples.push({
        prompt: prompt,
      });
    }
    jobConfig.config.process[0].sample.samples = newSamples;
    delete jobConfig.config.process[0].sample.prompts;
  }
  return jobConfig;
};
