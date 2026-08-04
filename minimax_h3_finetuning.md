# Fine-tuning MiniMax H3 (Hailuo 03)

MiniMax H3 is a 33B joint video+audio DiT conditioned on a Qwen3-VL-32B text
encoder. The released checkpoints are **CFG-distilled**: classifier-free
guidance is baked into the weights, the pipeline runs exactly one transformer
forward per step, and the intended guidance scale is 1. That property makes it
fast to serve — and hostile to naive fine-tuning. This doc explains why, what
this fork changes to make fine-tuning work, and how to train a
**de-distillation training adapter**, which is the proper long-term fix.

## Why naive fine-tuning produces mush

A non-distilled flow model gets its sharpness at inference from CFG: two
forwards per step (conditional + unconditional) extrapolated by a guidance
scale of ~3–7. Distillation trains the released model so that a *single*
forward at guidance 1 reproduces that guided output — the guidance lives in
the weights.

The standard training objective (`target = noise - clean`, which is what
`get_loss_target` in `extensions_built_in/diffusion_models/minimax_h3/`
returns) points at the **un-guided** data distribution. Every optimizer step
that minimizes it therefore pulls the weights away from their distilled
equilibrium — the model progressively *un-distills*. Sampling still happens at
guidance 1 with no CFG to compensate, so outputs drift toward what a normal
model looks like without guidance: soft, low-contrast, desaturated, weak
motion — "mush." The damage scales with steps, learning rate, and LoRA rank.
This is not an optimizer problem; the objective itself is mismatched, so the
fixes below change the loss or the sampler, not the optimizer.

The same failure hit FLUX.1-schnell and Z-Image-Turbo; both were solved with a
training adapter (see below).

## Fixes in this fork

### 1. Preservation (anchor) loss — slows the drift

`blank_prompt_preservation: true` in the `train:` block anchors every step to
the **frozen base model**: the trainer computes the base prediction with the
LoRA deactivated and adds an MSE term pulling the LoRA'd model's blank-prompt
output back toward it (`SDTrainer` preservation path). Your LoRA still learns
the data; the anchor resists the systematic un-distillation.

```yaml
train:
  blank_prompt_preservation: true
  blank_prompt_preservation_multiplier: 1.0   # raise to preserve more, learn slower
```

Caveats:
- Costs extra forwards per step (roughly 2x step time).
- The anchor covers **video only**. H3's audio prediction rides a side channel
  (`batch.audio_pred` / `batch.audio_target`) with freshly drawn noise per
  forward, so the audio branch is unanchored. A future fork change can pin the
  audio noise on the batch to extend the anchor to audio.
- It is a dial, not a cure: multiplier too low and the model still drifts, too
  high and it learns nothing. Start at 1.0.

### 2. True CFG sampling — recovers drifted checkpoints, and diagnoses them

The H3 pipeline previously accepted `guidance_scale` and ignored it. It now
implements **real two-pass classifier-free guidance** when
`sample.guidance_scale > 1`: conditional and unconditional prompts are packed
into their own sequences (text length shifts the rotary media clock, so each
pass needs its own layout) over shared latent state, and both video and audio
velocities are extrapolated by the guidance scale.

Uses:
- **Diagnostic**: if a mushy checkpoint sharpens dramatically at
  `guidance_scale: 2.5`, the mush is distillation drift, not bad data.
- **Recovery**: a drifted-but-learned LoRA can be *served* with real CFG at
  2x inference cost. Sweet spot is usually 2–3.5.

Note the base model was never trained with an unconditional branch, so real
CFG is off-recipe: on an undrifted model it over-sharpens, and the
unconditional pass is slightly out-of-distribution. `guidance_scale: 1`
remains the default and is bit-identical to the released recipe.

### 3. Audio fixes

- **`audio_preserve_pitch: true`** (dataset config): H3 snaps clip lengths
  down to the 17n+5 frame grid at a fixed 24 fps, which time-stretches nearly
  every training clip. The default stretch is linear interpolation over time —
  a pitch shift. With this flag the stretch preserves pitch.
- **Silent clips**: items without a soundtrack become pure-noise audio rows
  with *no* loss (fine), but in mixed batches with cached audio latents,
  missing items get zero-filled *normalized* latents — which decode to the
  per-channel mean, not silence. Either train silent datasets with
  `do_audio: false`, or don't mix silent and sounded clips in one dataset.
- `audio_loss_multiplier` is applied to an unweighted global mean added after
  the video loss; if audio learns faster/slower than video, tune it (LTX-2
  experience suggests ~0.5 when audio dominates).

### 4. Containment hyperparameters (no code)

If training without preservation or an adapter: LR ≤ 5e-5, rank ≤ 16,
≤ 1000 steps, and sample every ≤ 250 steps — the best checkpoint is usually
well before the last. Style transfers tolerate drift better than
subject/identity learning.

## The training adapter — the proper fix

### What it is

A de-distillation adapter is a LoRA whose only job is to *undo the
distillation in a controlled way*. Mechanics at train time (the pattern
`z_image.py` and the FLUX schnell adapter use):

1. The adapter is **merged into the frozen base weights** (`+1.0`), so the
   model your LoRA trains against behaves like a normal, CFG-requiring base
   model — the standard objective is now well-posed and training does not
   fight the distillation.
2. The adapter module itself is held **inactive during training** with
   `multiplier = -1.0`.
3. At sampling, the adapter activates and **subtracts itself back out** —
   distillation restored, your LoRA applies cleanly at guidance 1.

MiniMaxAI released no undistilled teacher (both partitions on
`MiniMaxAI/MiniMax-H3` are CFG-distilled), so the adapter must be *made* by
deliberate un-distillation. Ostris is training an official one; the recipe
below produces your own.

### Dataset

The adapter must un-distill **without teaching the model anything new** — the
dataset's job is to be a neutral, broad sample of "video in general":

- **Size**: 5k–10k clips. (The FLUX de-distill used ~150k images / 6k steps;
  a LoRA adapter needs less, and video steps are expensive.)
- **Diversity over quality-ceiling**: people, nature, urban, indoor, close-up,
  wide, day/night, fast and slow motion. Any skew (all cinematic drone shots,
  all faces) becomes a style the adapter imprints — and everything the adapter
  learns leaks into every fine-tune that uses it.
- **Native audio on most clips** — H3 trains audio jointly, and the adapter
  should un-distill the audio pathway too. This rules out most silent
  video-caption sets.
- **Length/format**: 3–10 s, ≥ 768px short side, natural sound (speech,
  ambience, music), 24+ fps sources.
- **Captions**: dense, factual, no trigger words. Auto-captioning with a
  strong VLM is fine; caption what is *seen and heard*.

Concrete starting points:
- **OpenVid-1M (HQ subset)** — open, dense captions, good visual diversity;
  its weakness is audio coverage, so filter for clips with audio tracks.
- **Pexels / Mixkit CC0 pulls with original audio** — high visual quality,
  permissive licensing, decent ambient audio; needs captioning.
- **VGGSound-style AV sets** — strong audio-visual correspondence for the
  audio head (mind the research-only licensing for anything shipped).

A practical blend: ~70% OpenVid-HQ (visual breadth) + ~30% audio-rich clips,
re-captioned uniformly, deduplicated, shuffled.

### Training recipe

```yaml
network:
  type: "lora"
  linear: 64            # adapter wants more capacity than a subject LoRA
  linear_alpha: 64
train:
  lr: 1e-5              # low and slow — controlled breakdown, not learning
  timestep_type: "shift"
  batch_size: 1
  steps: 3000-6000
  blank_prompt_preservation: false   # the drift IS the goal here
sample:
  guidance_scale: 1     # watch these go soft — that is the adapter working
  # also render a second set at guidance_scale: 3.5 (true CFG)
```

Progress signals — this is the counterintuitive part:
- **Guidance-1 previews getting mushy is success.** The adapter is supposed to
  make the merged model require CFG again.
- **Guidance-3.5 (true CFG) previews should stay sharp** and look like the
  base model. When g1 is clearly soft and g3.5-CFG is clean and prompt-adherent
  across diverse prompts, stop — past that you're teaching the dataset's look.
- Sanity check after training: base model alone at g1 must be untouched
  (bit-identical — the adapter is separate weights).

### Integration work still needed in this fork

Using any H3 adapter (ours or Ostris's) requires loader code that does not
exist yet — `toolkit/assistant_lora.py` is FLUX-only and `BaseModel` never
populates `self.assistant_lora`:

1. A `load_training_adapter()` on `MinimaxH3Model` mirroring
   `z_image.py:84-165`: build the LoRA network over the transformer, load the
   adapter state dict (mapping ComfyUI `diffusion_model.` keys to
   `transformer.`), `merge_in(1.0)`, then hold it inactive with
   `multiplier = -1.0` and set `invert_assistant_lora = True`.
2. **The quantization ordering problem**: the shipped DiT arrives
   already-quantized (int8 ConvRot), and `merge_in` needs mergeable weights.
   Either dequantize→merge→requantize, or load the bf16 DiT variant
   (`minimax_h3_fl2va_bf16.safetensors`, 66 GB) via
   `model_kwargs.dit_fl2va_path`, merge, then let `quantize` re-quantize
   in-process. The bf16 route is simpler and only costs load time.
3. Note the `transformer_only` LoRA targeting matches `token_refiner.blocks.*`
   as well as the 50 DiT blocks (substring match on "blocks") — the adapter
   and user LoRAs will share that scope, which is consistent as long as both
   are trained in this fork.

## What is still out of scope

- **ref2va training** (reference-to-video: audio→video, video→video): the
  ref2va DiT checkpoint can be loaded (`model_kwargs.partition: ref2va`), but
  the trainer only builds fl2va-shaped packed sequences — no reference
  segment, no dataset plumbing for reference video/audio. Implementing it
  means extending `packing.build_packed_sequence` with a reference segment
  (rotary coords, clean-pinned row timesteps, prediction slicing) and adding
  reference media to the dataloader, mirroring the layout in MiniMax's
  reference implementation. Sized as its own 1–2 week project.
- **Audio anchoring** in the preservation loss (see caveat above).
- W&B logging of video+audio previews is stubbed out
  (`blank_log_image_function`); previews land on disk only.
