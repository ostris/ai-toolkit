import { CaptionJobConfig } from "@/types";
import { captionerTypes } from "./captionOptions";


export const defaultCaptionJobConfig: CaptionJobConfig = {
  job: 'extension',
  config: {
    name: 'Caption Directory',
    process: [
      {
        type: 'Qwen3OmniCaptioner',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        caption: {
          model_name_or_path: 'ai-toolkit/Qwen3-Omni-30B-A3B-Thinking',
          dtype: 'bf16',
          quantize: true,
          qtype: 'convrot8',
          low_vram: false,
          extensions: ['mp4', 'mov', 'webm', 'mkv', 'avi', 'jpg', 'jpeg', 'png', 'bmp', 'webp'],
          path_to_caption: '',
          recaption: false,
          compile: true,
          caption_extension: 'txt',
          caption_prompt:
            "Caption this video as if you were going to try to generate it with a video generator. Describe the visual content, how it moves and changes over time, and the camera work. Also describe the audio, including any speech, music, or sound effects, and transcribe spoken dialogue verbatim in quotes. Be decisive by stating things as they are. Do not say things like \"It appears that\" Or \"possibly\". No preamble. Just get to the point.",
          max_res: 512,
          max_new_tokens: 512,
          batch_size: 1,
        },
      },
    ],
  },
};


const repairDefaults = (defaults: { [key: string]: any }) => {
  let newDefaults: { [key: string]: any } = {};
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
