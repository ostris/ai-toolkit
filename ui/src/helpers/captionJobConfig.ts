import { apiClient } from '@/utils/api';
import { CaptionJobConfig } from '@/types';
import { captionerTypes } from './captionOptions';


export const fallbackCaptionJobConfig: CaptionJobConfig = {
  job: 'extension',
  config: {
    name: 'Caption Directory',
    process: [
      {
        type: 'AceStepCaptioner',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        caption: {
          model_name_or_path: "ACE-Step/acestep-transcriber",
          model_name_or_path2: "ACE-Step/acestep-captioner",
          dtype: 'bf16',
          quantize: true,
          qtype: 'float8',
          low_vram: true,
          extensions: ['mp3', 'wav', 'flac', 'ogg'],
          path_to_caption: '',
          recaption: false,
          caption_extension: 'txt',
        },
      },
    ],
  },
};

function repairDefaults(defaults: { [key: string]: any }) {
  const newDefaults: { [key: string]: any } = {};
  // if the key doesnt start with config.process[0]., then add it
  for (const key in defaults) {
    if (!key.startsWith('config.process[0].')) {
      newDefaults[`config.process[0].${key}`] = defaults[key];
    } else {
      newDefaults[key] = defaults[key];
    }
  }
  return newDefaults;
}

const buildCaptionJobConfigFromType = (typeName: string): CaptionJobConfig => {
  const captionerType = captionerTypes.find(option => option.name === typeName);
  const repairedDefaults = repairDefaults(captionerType?.defaults || {});
  const modelNameOrPath = repairedDefaults['config.process[0].caption.model_name_or_path']?.[0] || '';
  const modelNameOrPath2 = repairedDefaults['config.process[0].caption.model_name_or_path2']?.[0];
  const extensions = repairedDefaults['config.process[0].caption.extensions']?.[0] || [];
  const captionPrompt = repairedDefaults['config.process[0].caption.caption_prompt']?.[0];
  const maxRes = repairedDefaults['config.process[0].caption.max_res']?.[0];
  const maxNewTokens = repairedDefaults['config.process[0].caption.max_new_tokens']?.[0];

  return {
    job: 'extension',
    config: {
      name: 'Caption Directory',
      process: [
        {
          type: typeName,
          sqlite_db_path: './aitk_db.db',
          device: 'cuda',
          caption: {
            model_name_or_path: modelNameOrPath,
            ...(modelNameOrPath2 ? { model_name_or_path2: modelNameOrPath2 } : {}),
            dtype: 'bf16',
            quantize: true,
            qtype: 'float8',
            low_vram: true,
            extensions,
            path_to_caption: '',
            recaption: false,
            caption_extension: 'txt',
            ...(captionPrompt ? { caption_prompt: captionPrompt } : {}),
            ...(typeof maxRes === 'number' ? { max_res: maxRes } : {}),
            ...(typeof maxNewTokens === 'number' ? { max_new_tokens: maxNewTokens } : {}),
          },
        },
      ],
    },
  };
};

const imageDefaultCaptionJobConfig = buildCaptionJobConfigFromType('Qwen3VLCaptioner');

export const defaultCaptionJobConfig = async (datasetPath: string): Promise<CaptionJobConfig> => {
  try {
    const { imageFileCount, audioFileCount } = await apiClient
      .post('/api/datasets/fileCounts', { datasetPath })
      .then(res => res.data);

    if (audioFileCount > imageFileCount) {
      return fallbackCaptionJobConfig;
    }

    return imageDefaultCaptionJobConfig;
  } catch (error) {
    console.error('Error loading dataset file counts for caption defaults:', error);
    return fallbackCaptionJobConfig;
  }
};



export const handleCaptionerTypeChange = (
  currentTypeName: string,
  newTypeName: string,
  jobConfig: CaptionJobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const currentType = captionerTypes.find(a => a.name === currentTypeName);
  if (!currentType || currentType.name === newTypeName) {
    return;
  }

  // update the defaults when a model is selected
  const newType = captionerTypes.find(model => model.name === newTypeName);

  let currentDefaults = repairDefaults(currentType.defaults || {});
  let newDefaults = repairDefaults(newType?.defaults || {});

  // set new model
  setJobConfig(newTypeName, 'config.process[0].type');

  // revert defaults from previous model
  for (const key in currentDefaults) {
    setJobConfig(currentDefaults[key][1], key);
  }

  for (const key in newDefaults) {
    setJobConfig(newDefaults[key][0], key);
  }
};
