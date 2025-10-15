import React from 'react';
import { ConfigDoc } from '@/types';
import { IoFlaskSharp } from 'react-icons/io5';

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
  gpuids: {
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
        When using a trigger word, If your captions do not contain the trigger word, it will be added automatically the
        beginning of the caption. If you do not have captions, the caption will become just the trigger word. If you
        want to have variable trigger words in your captions to put it in different spots, you can use the{' '}
        <code>{'[trigger]'}</code> placeholder in your captions. This will be automatically replaced with your trigger
        word.
        <br />
        <br />
        Trigger words will not automatically be added to your test prompts, so you will need to either add your trigger
        word manually or use the
        <code>{'[trigger]'}</code> placeholder in your test prompts as well.
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    title: 'Name or Path',
    description: (
      <>
        The name of a diffusers repo on Huggingface or the local path to the base model you want to train from. The
        folder needs to be in diffusers format for most models. For some models, such as SDXL and SD1, you can put the
        path to an all in one safetensors checkpoint here.
      </>
    ),
  },
  'datasets.control_path': {
    title: 'Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training. The control images will be
        resized to match the training images.
      </>
    ),
  },
  'datasets.multi_control_paths': {
    title: 'Multi Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training.
        <br />
        <br />
        For multi control datasets, the controls will all be applied in the order they are listed. If the model does not
        require the images to be the same aspect ratios, such as with Qwen/Qwen-Image-Edit-2509, then the control images
        do not need to match the aspect size or aspect ratio of the target image and they will be automatically resized
        to the ideal resolutions for the model / target images.
      </>
    ),
  },
  'datasets.num_frames': {
    title: 'Number of Frames',
    description: (
      <>
        This sets the number of frames to shrink videos to for a video dataset. If this dataset is images, set this to 1
        for one frame. If your dataset is only videos, frames will be extracted evenly spaced from the videos in the
        dataset.
        <br />
        <br />
        It is best to trim your videos to the proper length before training. Wan is 16 frames a second. Doing 81 frames
        will result in a 5 second video. So you would want all of your videos trimmed to around 5 seconds for best
        results.
        <br />
        <br />
        Example: Setting this to 81 and having 2 videos in your dataset, one is 2 seconds and one is 90 seconds long,
        will result in 81 evenly spaced frames for each video making the 2 second video appear slow and the 90second
        video appear very fast.
      </>
    ),
  },
  'datasets.do_i2v': {
    title: 'Do I2V',
    description: (
      <>
        For video models that can handle both I2V (Image to Video) and T2V (Text to Video), this option sets this
        dataset to be trained as an I2V dataset. This means that the first frame will be extracted from the video and
        used as the start image for the video. If this option is not set, the dataset will be treated as a T2V dataset.
      </>
    ),
  },
  'datasets.flip': {
    title: 'Flip X and Flip Y',
    description: (
      <>
        You can augment your dataset on the fly by flipping the x (horizontal) and/or y (vertical) axis. Flipping a
        single axis will effectively double your dataset. It will result it training on normal images, and the flipped
        versions of the images. This can be very helpful, but keep in mind it can also be destructive. There is no
        reason to train people upside down, and flipping a face can confuse the model as a person's right side does not
        look identical to their left side. For text, obviously flipping text is not a good idea.
        <br />
        <br />
        Control images for a dataset will also be flipped to match the images, so they will always match on the pixel
        level.
      </>
    ),
  },
  'train.unload_text_encoder': {
    title: 'Unload Text Encoder',
    description: (
      <>
        Unloading text encoder will cache the trigger word and the sample prompts and unload the text encoder from the
        GPU. Captions in for the dataset will be ignored
      </>
    ),
  },
  'train.cache_text_embeddings': {
    title: 'Cache Text Embeddings',
    description: (
      <>
        <small>(experimental)</small>
        <br />
        Caching text embeddings will process and cache all the text embeddings from the text encoder to the disk. The
        text encoder will be unloaded from the GPU. This does not work with things that dynamically change the prompt
        such as trigger words, caption dropout, etc.
      </>
    ),
  },
  'model.multistage': {
    title: 'Stages to Train',
    description: (
      <>
        Some models have multi stage networks that are trained and used separately in the denoising process. Most
        common, is to have 2 stages. One for high noise and one for low noise. You can choose to train both stages at
        once or train them separately. If trained at the same time, The trainer will alternate between training each
        model every so many steps and will output 2 different LoRAs. If you choose to train only one stage, the trainer
        will only train that stage and output a single LoRA.
      </>
    ),
  },
  'train.switch_boundary_every': {
    title: 'Switch Boundary Every',
    description: (
      <>
        When training a model with multiple stages, this setting controls how often the trainer will switch between
        training each stage.
        <br />
        <br />
        For low vram settings, the model not being trained will be unloaded from the gpu to save memory. This takes some
        time to do, so it is recommended to alternate less often when using low vram. A setting like 10 or 20 is
        recommended for low vram settings.
        <br />
        <br />
        The swap happens at the batch level, meaning it will swap between a gradient accumulation steps. To train both
        stages in a single step, set them to switch every 1 step and set gradient accumulation to 2.
      </>
    ),
  },
  'train.force_first_sample': {
    title: 'Force First Sample',
    description: (
      <>
        This option will force the trainer to generate samples when it starts. The trainer will normally only generate a
        first sample when nothing has been trained yet, but will not do a first sample when resuming from an existing
        checkpoint. This option forces a first sample every time the trainer is started. This can be useful if you have
        changed sample prompts and want to see the new prompts right away.
      </>
    ),
  },
  'model.layer_offloading': {
    title: (
      <>
        Layer Offloading{' '}
        <span className="text-yellow-500">
          ( <IoFlaskSharp className="inline text-yellow-500" name="Experimental" /> Experimental)
        </span>
      </>
    ),
    description: (
      <>
        This is an experimental feature based on{' '}
        <a className="text-blue-500" href="https://github.com/lodestone-rock/RamTorch" target="_blank">
          RamTorch
        </a>
        . This feature is early and will have many updates and changes, so be aware it may not work consistently from
        one update to the next. It will also only work with certain models.
        <br />
        <br />
        Layer Offloading uses the CPU RAM instead of the GPU ram to hold most of the model weights. This allows training
        a much larger model on a smaller GPU, assuming you have enough CPU RAM. This is slower than training on pure GPU
        RAM, but CPU RAM is cheaper and upgradeable. You will still need GPU RAM to hold the optimizer states and LoRA
        weights, so a larger card is usually still needed.
        <br />
        <br />
        You can also select the percentage of the layers to offload. It is generally best to offload as few as possible
        (close to 0%) for best performance, but you can offload more if you need the memory.
      </>
    ),
  },
  'model.qie.match_target_res': {
    title: 'Match Target Res',
    description: (
      <>
        This setting will make the control images match the resolution of the target image. The official inference
        example for Qwen-Image-Edit-2509 feeds the control image is at 1MP resolution, no matter what size you are
        generating. Doing this makes training at lower res difficult because 1MP control images are fed in despite how
        large your target image is. Match Target Res will match the resolution of your target to feed in the control
        images allowing you to use less VRAM when training with smaller resolutions. You can still use different aspect
        ratios, the image will just be resizes to match the amount of pixels in the target image.
      </>
    ),
  },

  'model.reserve_ram_for_spillover': {
    title: 'Reserve RAM for spillover',
    description: (
      <>
        This option will reserve the specified amount of RAM (in gigabytes) at the beginning of the training run and
        release it just before the model is loaded to video memory.
        <br />
        <br />
        This can prevent CUDA Out of Memory errors during the training initialization in case when both RAM and VRAM are
        at full capacity and the system cannot allocate RAM for video memory spillover. Especially useful in combination
        with the "Layer Offloading" option as it uses a lot of RAM.
        <br />
        <br />
        On low-end systems with both low RAM and low VRAM, system memory will be paged to disk and video memory will
        fall back to RAM. This makes training slow but it will complete successfully even on low-end systems.
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
