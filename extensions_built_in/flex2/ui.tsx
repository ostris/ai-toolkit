// UI entries (training form + Generate page) for the models this package
// registers in AI_TOOLKIT_MODELS. Loaded at runtime by the UI, not bundled:
// see ui/src/extensions/README.md for the convention and the allowed imports.
import type { ModelArch } from "@/app/jobs/new/options";

const defaultNameOrPath = "";

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  {
    name: "flex2",
    label: "Flex.2",
    group: "image",
    controls: ["depth", "line", "pose", "inpaint"],
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ostris/Flex.2-preview",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.model_kwargs": [
        {
          invert_inpaint_mask_chance: 0.2,
          inpaint_dropout: 0.5,
          control_dropout: 0.5,
          inpaint_random_chance: 0.2,
          do_random_inpainting: true,
          random_blur_mask: true,
          random_dialate_mask: true,
        },
        {},
      ],
      "config.process[0].train.bypass_guidance_embedding": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
];
