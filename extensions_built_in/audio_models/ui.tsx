// UI entries (training form + Generate page) for the models this package
// registers in AI_TOOLKIT_MODELS. Loaded at runtime by the UI, not bundled:
// see ui/src/extensions/README.md for the convention and the allowed imports.
import type { ModelArch } from "@/app/jobs/new/options";
import {
  defaultSampleConfig,
  defaultAudioSampleConfig,
  defaultYue2SampleConfig,
} from "@/helpers/defaultSamples";

const defaultNameOrPath = "";

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  {
    name: "ace_step_15_xl",
    label: "ACE-Step 1.5 XL",
    group: "audio",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_xl_base_aio.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].sample": [
        defaultAudioSampleConfig,
        defaultSampleConfig,
      ],
    },
    sampleTags: {
      CAPTION: {
        title: "Audio Prompt",
        type: "text",
        full: true,
      },
      LYRICS: {
        title: "Lyrics",
        type: "multiline",
        full: true,
      },
      BPM: {
        title: "BPM",
        type: "number",
      },
      KEYSCALE: {
        title: "Key Scale",
        type: "text",
      },
      TIMESIGNATURE: {
        title: "Time Signature",
        type: "text",
      },
      DURATION: {
        title: "Duration (sec)",
        type: "number",
      },
      LANGUAGE: {
        title: "Language",
        type: "text",
      },
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
    ],
  },
  {
    name: "yue2",
    label: "YuE2",
    group: "audio",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Comfy-Org/YuE2/checkpoints/yue2_3b_int8_convrot.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].model.low_vram": [false, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["sigmoid", "sigmoid"],
      // the int8 repack ships convrot8 layers; requesting convrot8 keeps them as-is (no requantization)
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].sample": [
        defaultYue2SampleConfig,
        defaultSampleConfig,
      ],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, true],
      // audio has no resolution; every bucket would duplicate the whole dataset
      "config.process[0].datasets[x].resolution": [[512], [512, 768, 1024]],
      // blank captions break lyric following; the AR must always see the prefix
      "config.process[0].datasets[x].caption_dropout_rate": [0, 0.05],
      "config.process[0].model.model_kwargs": [
        {
          cot: "full",
          abc_dropout: 0.5,
          sample_ar_repetition_penalty: 1.2,
          ar_kl_weight: 0.2,
        },
        {},
      ],
    },
    // native YuE2 prompt: style text, a [Lyrics] line, the lyrics
    hasMultiLinePrompts: true,
    modelNotes: (
      <div className="space-y-2">
        <p className="font-semibold text-amber-400">
          Experimental. The AR (composition) model memorizes quickly and does
          not work well on small datasets. A handful of songs is enough for it
          to learn the exact token sequence of each song; after that it stops
          generalizing and free-running samples drift away from the training
          material. Expect to need a large, varied dataset for the AR side to
          learn a style rather than the songs themselves.
        </p>
        <p>
          YuE2 is two experts on one backbone. The <b>AR expert</b> reads the
          style line and lyrics and writes the song as a sequence of semantic
          codec tokens (25 per second). The <b>NAR expert</b> then renders those
          tokens into audio latents with flow matching. Training a LoRA here
          trains both: next-token loss on the AR over the whole song from its
          start, flow loss on the NAR over a random window.
        </p>
        <p>
          The AR loss is the one to watch (<code>loss/ar_ce</code>). It starts
          near 5 and, on a small dataset, falls toward 0 within a few hundred
          steps, which is memorization. <code>ar_kl_weight</code> in model
          kwargs anchors the AR to the base model so it cannot collapse onto the
          training songs; <code>ar_lr_multiplier</code> and a smaller AR rank
          slow it further. Style comes mostly from the NAR, lyric following from
          the AR. Do not use caption dropout: a blank prompt breaks lyric
          following.
        </p>
        <p>
          The official audio-to-token encoder is unreleased. Training uses the
          community tokenizer by Kytra (
          <a
            href="https://x.com/sin_ceriously"
            target="_blank"
            rel="noreferrer"
            className="text-blue-400 hover:underline"
          >
            @sin_ceriously
          </a>
          ), a MERT-v2-FullSong head that maps real audio to YuE2 codec tokens:{" "}
          <a
            href="https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
            target="_blank"
            rel="noreferrer"
            className="text-blue-400 hover:underline"
          >
            Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4
          </a>
          . It is downloaded on first use. Its tokens are an approximation of
          the model's own, so rendered samples will not be bit-faithful to the
          training audio even when the AR replays a song exactly. Though it is
          extremely close.
        </p>
        <p>
          <b>ABC generation.</b> Generation is two stages: the AR first writes a
          lead sheet of the whole song in ABC notation (sections, chords, vocal
          and instrumental melody), then writes the codec tokens conditioned on
          that sheet. It can also run without a sheet ("off" mode), which uses a
          different instruction line. Samples here do the two stages with{" "}
          <code>cot: full</code> (chords) or <code>cot: melody</code> (melody
          only), or the single stage with <code>cot: off</code>.
        </p>
        <p>
          <b>Training for both.</b> Every song is transcribed to an ABC sheet
          with SheetSage2 and the AR is trained on lyrics to sheet and sheet to
          tokens. <code>abc_dropout</code> (default 0.5) is the fraction of
          training items fed without the sheet instead, as an off-mode prompt,
          so one LoRA works in both modes. Set it to 0 to train the sheet path
          only, or 1 to train off mode only.
        </p>
        <p>
          <b>Caching is required.</b> The sheet is produced at latent-cache time
          and stored with the latents and codec tokens, so Cache Latents to Disk
          must stay on: without it, SheetSage2 (about 12 s per song), the MERT
          tokenizer and the VAE would run again on every training step. The
          cache records which mode built it; changing <code>cot</code> means
          deleting the dataset's <code>_latent_cache</code> folder so the sheets
          are rebuilt. Audio has no resolution, so keep a single resolution
          bucket per dataset or every bucket duplicates the songs.
        </p>
        <p>
          Prompt format: a style line, then <code>[Lyrics]</code>, then the
          lyrics with bracketed section headers such as <code>[Verse 1]</code>{" "}
          and <code>[Chorus]</code>. The Qwen3-Omni captioner has a YuE2 preset
          that writes captions in this layout. Sample length is set by the
          Duration field in the sample section.
        </p>
      </div>
    ),
    // no separate text encoder: the prompt side is the AR expert, covered by the transformer quantization
    disableSections: ["network.conv", "model.quantize_te"],
    additionalSections: ["model.low_vram", "sample.duration"],
  },
  {
    name: "ace_step_15",
    label: "ACE-Step 1.5",
    group: "audio",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].sample": [
        defaultAudioSampleConfig,
        defaultSampleConfig,
      ],
    },
    sampleTags: {
      CAPTION: {
        title: "Audio Prompt",
        type: "text",
        full: true,
      },
      LYRICS: {
        title: "Lyrics",
        type: "multiline",
        full: true,
      },
      BPM: {
        title: "BPM",
        type: "number",
      },
      KEYSCALE: {
        title: "Key Scale",
        type: "text",
      },
      TIMESIGNATURE: {
        title: "Time Signature",
        type: "text",
      },
      DURATION: {
        title: "Duration (sec)",
        type: "number",
      },
      LANGUAGE: {
        title: "Language",
        type: "text",
      },
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
    ],
  },
];
