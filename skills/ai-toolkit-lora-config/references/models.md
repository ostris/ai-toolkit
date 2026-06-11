# Model Reference

All models use `job: extension` with `type: "sd_trainer"`. The `arch` field in `model:` determines which model class loads.

For all transformer-based models below, **only `linear`/`linear_alpha` apply** in the network config. Do NOT include `conv` or `conv_alpha` — those only apply to UNet-based models (SD 1.5, SDXL, SD 2.x).

## Flux.2-dev

- HF path: `black-forest-labs/FLUX.2-dev`
- `arch: "flux2"`
- License: BFL non-commercial (LoRAs inherit)
- Text encoder: Mistral-Small-3.1-24B (large, quantize when tight on VRAM)
- Min VRAM with quantization: ~24GB
- Use when: best quality character LoRAs and user doesn't need commercial rights

```yaml
model:
  name_or_path: "black-forest-labs/FLUX.2-dev"
  arch: "flux2"
  quantize: true       # for ≤24GB; can be false on 80GB+
  quantize_te: true    # Mistral TE is ~14GB on its own
```

## Flux.2-Klein-4B / Klein-9B

- HF paths: `black-forest-labs/FLUX.2-klein-base-4B`, `black-forest-labs/FLUX.2-klein-base-9B`
- `arch: "flux2_klein_4b"` or `"flux2_klein_9b"`
- License: Apache 2.0 (commercial-friendly)
- Text encoder: Qwen3-8B (smaller than Mistral)
- Use when: need commercial rights, want Flux-family quality
- Requires: `timestep_type: "weighted"` in train, `match_target_res: false` in model_kwargs

```yaml
model:
  name_or_path: "black-forest-labs/FLUX.2-klein-base-9B"
  arch: "flux2_klein_9b"
  quantize: true
  quantize_te: true
  qtype: "qfloat8"
  model_kwargs:
    match_target_res: false
train:
  timestep_type: "weighted"
```

## Z-Image / Z-Image Turbo / Z-Image De-Turbo

- Base: `Tongyi-MAI/Z-Image` (arch: `zimage`)
- Turbo: `Tongyi-MAI/Z-Image-Turbo` (arch: `zimage`, needs assistant adapter)
- De-Turbo: `ostris/Z-Image-De-Turbo` (arch: `zimage`, needs `extras_name_or_path: "Tongyi-MAI/Z-Image-Turbo"` for TE/VAE/tokenizer)
- License: Commercial-friendly
- Text encoder: Qwen3-8B (~14GB unquantized)
- Bucket divisibility: 32 (16*2)

**Turbo requires the assistant adapter** — it's distilled (like Flux Schnell) and can't learn LoRA training without it:

```yaml
model:
  name_or_path: "Tongyi-MAI/Z-Image-Turbo"
  arch: "zimage"
  qtype: "qfloat8"
  assistant_lora_path: "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors"
train:
  timestep_type: "weighted"
sample:
  guidance_scale: 1   # turbo needs low guidance
  sample_steps: 8     # turbo needs few steps
```

De-Turbo is the distillation removed — use when you want the turbo-trained LoRA to work at normal guidance/steps at inference.

## Chroma

- HF path: `lodestones/Chroma` (auto-latest) or pin like `lodestones/Chroma/v28`
- `arch: "chroma"`
- License: Apache 2.0 (commercial-friendly)
- Based on modified Flux architecture
- Use when: want community-driven open model, commercial rights, active development

```yaml
model:
  name_or_path: "lodestones/Chroma"
  arch: "chroma"
  quantize: true
```

## Flex.2-preview

- HF path: `ostris/Flex.2-preview`
- `arch: "flex2"`
- Use when: want built-in control conditioning (depth, line, pose) + inpainting during training
- Requires: `bypass_guidance_embedding: true` in train
- Complex — skip unless user specifically asks for control/inpaint features

## FLUX.1-dev (legacy but still common)

- HF path: `black-forest-labs/FLUX.1-dev`
- `is_flux: true` (not an `arch`)
- License: BFL non-commercial
- Older but still widely supported; prefer Flux.2-dev unless user specifically requests it

## FLUX.1-Kontext-dev

- HF path: `black-forest-labs/FLUX.1-Kontext-dev`
- `arch: "flux_kontext"`
- Image editing model — requires paired dataset with `control_path`
- Only use for image-editing LoRAs, not character/style

## HiDream, OmniGen2, Qwen-Image, Wan (video)

Available but niche. See `README.md` in ai-toolkit for details. Don't recommend unless user specifically names them or has clear use case.

## Choosing between them

Two axes matter: licensing/quality, and **texture fidelity**. Don't pick on speed alone.

### Texture fidelity tiers

Distilled/turbo models compress the sampling trajectory to 4-8 steps. High-frequency detail (paper grain, halftone dots, fine woodblock linework, fabric weave, etching crosshatch, brushwork particulate, fine pattern repeats) is what gets sacrificed in that compression — the model produces a "vibe of" the texture instead of the texture itself.

- **Tier 1 (best texture fidelity, slowest):** Flux.2-dev, Flux.2-Klein-9B base, Chroma — full sampling trajectory, intricate high-frequency detail survives
- **Tier 2 (decent texture, faster):** Flux.2-Klein-4B base, Z-Image Base — base-trained, smaller capacity than tier 1 but no distillation
- **Tier 3 (low texture fidelity, fastest):** Z-Image Turbo, Flux Schnell, Z-Image De-Turbo — distilled, 4-8 step sampling. Fine for graphic/flat styles, wrong choice for texture-heavy styles

### Model selection table

| User says / dataset has... | Recommend |
|---|---|
| "best quality for my character" + non-commercial OK | Flux.2-dev |
| "best quality for my character" + commercial | Flux.2-Klein-9B base |
| Style LoRA, **texture-heavy** (paper grain, halftone, woodblock, fabric, etching, brushwork, fine repeating pattern) | Klein 9B base or Chroma — NOT turbo |
| Style LoRA, **graphic/flat** (vector-feeling, bold color blocks, clean illustration, modern UI-style) | Z-Image Turbo OK, fast iteration |
| "fast iteration" with no other constraints | Z-Image Turbo (with caveat above) |
| "open source" or "community model" | Chroma |
| "I want to use my LoRA commercially" | Klein 9B or Chroma or Z-Image (any variant) |
| "I have a 3090" (24GB) | any with `quantize: true` |
| "I have an H100" or "Colab Pro+" | any with `quantize: false` |

### Capability matrix (for mapping a model brief)

When a model brief exists (`briefs/<project>-brief.md`, produced by
`ai-toolkit-model-brief`), its MUST capabilities are hard filters over this
matrix, applied **before** the selection table above. Ratings are working
tiers, not benchmarks — when a MUST rides on one, verify with a quick
sample from the base model before committing the run.

| Model | Text rendering | Edit support | Motion | Prompt adherence | Faces | License | Hosted availability |
|---|---|---|---|---|---|---|---|
| Flux.2-dev | strong | — | — | strong | strong | non-commercial | wide |
| Flux.2-Klein-9B | good | native via ctrl_img | — | good | good | Apache 2.0 | wide |
| Flux.2-Klein-4B | fair | native via ctrl_img | — | fair | fair | Apache 2.0 | wide |
| Qwen-Image / Qwen-Image-Edit (2511) | strong (best of set) | **purpose-built (Edit)** | — | strong | good | commercial-friendly | wide (note: hosted runtimes often need LoRA strength >1.0) |
| Z-Image Base | fair | — | — | good | fair | commercial-friendly | moderate |
| Z-Image Turbo | weak (distilled) | — | — | fair | fair | commercial-friendly | moderate |
| Chroma | fair | — | — | good | fair | Apache 2.0 | narrow |
| FLUX.1-Kontext-dev | good | purpose-built (paired) | — | good | good | non-commercial | moderate |
| Wan2.2 (video) | — | — | **yes** | good | fair | commercial-friendly | moderate |
| SDXL | weak | — | — | weak | fair | commercial-friendly | universal |

How to apply a brief:

1. **Lane first** (brief axes 3–4): edit intent → Qwen-Image-Edit /
   Kontext / Klein ctrl_img only. Motion → Wan. Otherwise text-to-image.
2. **Hard filters** (brief MUSTs + constraints): text rendering MUST drops
   the "weak" column entries; commercial use drops non-commercial bases;
   the inference destination must actually host the arch (an unhosted arch
   runs only locally — if the brief says "don't know yet", prefer
   wide-availability archs).
3. **Trade-offs last**: volume/speed (brief axis 8) vs texture fidelity
   (axis 9) is the turbo-vs-base tier choice — never resolve it silently;
   the texture-fidelity tiers above are the vocabulary for that
   conversation.
4. **The dial** (brief axis 1) doesn't pick the model — it sets rank,
   steps, captioning aggressiveness, and DOP once the model is chosen.

### Validation-run pattern

When committing to a long run on a tier-1 model (Klein 9B base, Flux.2-dev), it's worth doing a tier-2 sanity check first to validate captioning before burning hours on the canonical run:

1. Use **Z-Image Base** (not Turbo) on the same dataset, same trigger word, same captions
2. Train 1000-1500 steps
3. Sample with the same prompts you'll use on the long run
4. If outputs are recognizable-but-soft versions of the target style: captioning is working, the long Klein run will sharpen what's there
5. If outputs ignore the trigger or show heavy content/style leakage: re-caption before the long run

This catches caption-strategy failures (the most common reason a long run fails) at ~10% of the compute cost, and the diagnostic signal is reliable because tier-2 models still respond to caption discipline — they just render the result with less fidelity.

### Existing repo evidence

The chemigram style LoRA in this repo (`config/examples/train_lora_flux2_klein_9b_chemigram_style.yaml`) is on Klein 9B base, not Turbo, despite being a style LoRA. The dataset features vein patterning, spray/atomized chemistry, halftone, and pooling — all high-frequency detail. The choice is implicit — texture-heavy style → tier-1 base model — and the table above makes it explicit.
