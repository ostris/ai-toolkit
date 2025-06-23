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
  'multigpu': {
    title: 'Multi-GPU Training',
    description: (
      <>
        Enable distributed training across multiple GPUs using Hugging Face Accelerate. This allows you to train larger models
        or use larger batch sizes by distributing the workload across multiple GPUs.
        <br /><br />
        <strong>Requirements:</strong>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Multiple GPUs with sufficient VRAM</li>
          <li>Proper Accelerate configuration</li>
          <li>Compatible model architecture</li>
        </ul>
        <br />
        <strong>Benefits:</strong>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Faster training with larger batch sizes</li>
          <li>Ability to train larger models</li>
          <li>Better memory utilization</li>
        </ul>
      </>
    ),
  },
  'accelerate_config': {
    title: 'Accelerate Configuration',
    description: (
      <>
        Configuration for Hugging Face Accelerate distributed training. This includes settings for:
        <br /><br />
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>Distributed Type:</strong> Multi-GPU, Multi-CPU, or No Distribution</li>
          <li><strong>Mixed Precision:</strong> FP16, BF16, or No Mixed Precision</li>
          <li><strong>GPU IDs:</strong> Comma-separated list of GPU indices to use</li>
          <li><strong>Number of Processes:</strong> Number of parallel processes</li>
        </ul>
        <br />
        The configuration is automatically generated based on your GPU setup and can be customized for advanced use cases.
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
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
