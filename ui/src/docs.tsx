import React from 'react';
import { ConfigDoc } from '@/types';

const docs: { [key: string]: ConfigDoc } = {
  'config.name': {
    title: 'Training Name',
    description: (
      <>
        The name of the training job. This name will be used to identify the job in the system and will the the filename
        of the final model. It must be unique and can only contain alphanumeric characters, underscores, and dashes. No
        spaces or special characters are allowed.
      </>
    ),
  },
  'gpuids': {
    title: 'GPU ID',
    description: (
      <>
        This is the GPU that will be used for training. Only one GPU can be used per job at a time via the UI currently. 
				However, you can start multiple jobs in parallel, each using a different GPU.
      </>
    ),
  },
  'config.process[0].trigger_word': {
    title: 'Trigger Word',
    description: (
      <>
        Optional: This will be the word or token used to trigger your concept or character. 
				<br />
				<br />
				When using a trigger word, 
				If your captions do not contain the trigger word, it will be added automatically the beginning of the caption. If you do not have
				captions, the caption will become just the trigger word. If you want to have variable trigger words in your captions to put it in different spots,
				you can use the <code>{'[trigger]'}</code> placeholder in your captions. This will be automatically replaced with your trigger word.
				<br />
				<br />
				Trigger words will not automatically be added to your test prompts, so you will need to either add your trigger word manually or use the 
				<code>{'[trigger]'}</code> placeholder in your test prompts as well.
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    title: 'Name or Path',
    description: (
      <>
        The name of a diffusers repo on Huggingface or the local path to the base model you want to train from. The folder needs to be in 
				diffusers format for most models. For some models, such as SDXL and SD1, you can put the path to an all in one safetensors checkpoint here.
      </>
    ),
  },
  'datasets.control_path': {
    title: 'Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be matching file pairs. 
        These images are fed as control/input images during training. 
      </>
    ),
  },
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
