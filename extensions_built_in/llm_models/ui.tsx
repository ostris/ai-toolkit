// UI entries (training form + Generate page) for the models this package
// registers in AI_TOOLKIT_MODELS. Loaded at runtime by the UI, not bundled:
// see ui/src/extensions/README.md for the convention and the allowed imports.
import type { ModelArch } from "@/app/jobs/new/options";
import {
  defaultSampleConfig,
  defaultQwen25OmniSampleConfig,
} from "@/helpers/defaultSamples";

const defaultNameOrPath = "";

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  {
    name: "qwen25_omni",
    label: "Qwen2.5-Omni",
    group: "llm",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ai-toolkit/Qwen2.5-Omni-7B/qwen2_5_omni_7b_convrot8.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].model.low_vram": [false, false],
      // the single-file thinker ships convrot8 layers; requesting convrot8 keeps them as-is
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.batch_size": [1, 1],
      "config.process[0].sample": [
        defaultQwen25OmniSampleConfig,
        defaultSampleConfig,
      ],
      // media is encoded on the GPU per step; the cache is optional and large (~54 MB per 300 s of audio)
      "config.process[0].datasets[x].cache_latents_to_disk": [false, true],
      "config.process[0].datasets[x].resolution": [[512], [512, 768, 1024]],
      // the caption is the training target; a blank one trains nothing
      "config.process[0].datasets[x].caption_dropout_rate": [0, 0.05],
      "config.process[0].model.model_kwargs": [
        { instruction: "Describe this in detail." },
        {},
      ],
    },
    disableSections: [
      "network.conv",
      "trigger_word",
      "train.diff_output_preservation",
      "train.blank_prompt_preservation",
      "train.unload_text_encoder",
      "slider",
    ],
    additionalSections: [
      "model.model_kwargs.instruction",
      "sample.ctrl_img",
      "datasets.num_frames",
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          Qwen2.5-Omni 7B thinker as a text-generating model: audio, image or
          video in, text out. Each dataset item is a media file (mp3, wav, flac,
          ogg, jpg, png, webp, mp4, ...) with a caption file next to it; the
          caption is the text the model learns to produce for that media. One
          dataset can mix all three kinds. Video files are used when Num Frames
          is above 1 and are seen as frames only (no audio track).
        </p>
        <p>
          The instruction in <code>model_kwargs.instruction</code> is the user
          turn for every training item; use the same wording when captioning
          with the trained LoRA. Samples take a media file per prompt and write
          the generated text as a .txt file.
        </p>
        <p>
          Media is encoded by the frozen audio and vision towers on the GPU each
          step, so Cache Latents to Disk can stay off (the cache is about 54 MB
          per 300 seconds of audio). Batch size is 1. The LoRA trains the text
          stack only. Watch <code>loss/ce</code>: next-token cross-entropy on
          the caption.
        </p>
      </div>
    ),
  },
];
