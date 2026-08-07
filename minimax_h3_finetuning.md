# Fine-tuning MiniMax H3 (Hailuo 03)

MiniMax H3 is a 33B joint video+audio DiT conditioned on a Qwen3-VL-32B text
encoder. The released checkpoints are **CFG-distilled**: classifier-free
guidance is baked into the weights, the pipeline runs exactly one transformer
forward per step, and the intended guidance scale is 1. That property makes it
fast to serve — and hostile to naive fine-tuning. This doc explains why, what
this fork changes to make fine-tuning work, and how to train a
**de-distillation training adapter**, which is the proper long-term fix.

## Why naive fine-tuning produces mush

**The intuition first.** A normal diffusion model is like a camera that shoots
flat, neutral footage: to get the crisp, vivid, prompt-faithful look, the
*sampler* applies a boost at playback time — classifier-free guidance (CFG)
runs the model twice per step, once with your prompt and once without, and
exaggerates the difference between the two by a factor of ~3–7. "More of what
the prompt changed, less of what it didn't."

H3 ships with that boost **baked into the weights**. During distillation,
MiniMax trained the model so a *single* forward pass at guidance 1 reproduces
what the boosted two-pass output used to look like. That's why it's fast — and
why it's fragile: the model's outputs are permanently "pre-amplified."

Now fine-tune it the standard way. The standard objective
(`target = noise - clean`, which is what `get_loss_target` in
`extensions_built_in/diffusion_models/minimax_h3/` returns) says: *your output
should match the training clip*. But training clips are natural footage — they
look like **un-boosted** output. So every optimizer step teaches the model
"be a little less amplified," and the baked-in guidance erodes. Sampling still
runs single-pass at guidance 1 with nothing to re-apply the boost, so outputs
drift toward what any diffusion model looks like *without* CFG: soft,
low-contrast, desaturated, weak motion — "mush." The model isn't failing to
learn your data; it's learning your data while simultaneously unlearning its
own sharpening.

The damage is cumulative — it scales with steps, learning rate, and LoRA
rank — and it is not an optimizer problem: the objective itself points at the
wrong distribution, so the fixes below change the *loss* or the *sampler*,
never the optimizer. The same failure hit FLUX.1-schnell and Z-Image-Turbo;
both were solved with a training adapter (see below).

## Fixes in this fork

### 0. Contrastive guidance loss — the objective-level fix (upstream)

Upstream ships `do_guidance_loss` ("contrastive guidance loss") for H3. It
was the default training method until the official training adapter (below)
replaced it — the adapter produces better results and is faster (no extra
blank-prompt forward per step). Contrastive guidance remains the fallback
when no adapter is set.

**Why it works.** The mush problem is that the plain objective says "your
output should equal the training clip" — an un-boosted target. The contrastive
objective instead says: *"your output should stand guidance-scale× beyond your
own no-prompt baseline, in the direction of the training clip."* Concretely,
each step the trainer runs one extra (no-grad) forward with a **blank prompt**
to get the model's own unconditional prediction, then rebuilds the target as
the CFG extrapolation `uncond + g * (raw_target - uncond)` with
`g = guidance_loss_target` (4.0). The model learns your data *as if it were a
guided output* — the "conditional is an amplified version of unconditional"
relationship that distillation encoded is exactly what the loss now preserves,
while the content still moves toward your data. The audio target is
extrapolated the same way, so the joint audio stream trains contrastively too.

**Why the sigma schedule.** The boost only matters early in denoising, when
the model is deciding composition and motion. Near the end of denoising the
`(target - uncond)` direction is dominated by the fresh random noise term that
nothing can predict — extrapolating it would just amplify noise. So the
effective scale decays with the noise level: `g_eff = 1 + (g - 1) * sigma`.
Full contrastive shaping at high noise, plain flow matching at low noise.

```yaml
train:
  do_guidance_loss: true
  guidance_loss_target: 4.0   # the boost strength the target assumes
```

Cost: one extra no-grad forward per step. This is the primary defense; the
preservation loss below is now an optional extra rather than the default.
(`guidance_loss_target` is effectively "how strong we believe the baked-in
boost is" — raising it trains toward a punchier look, lowering it toward a
flatter one.)

### 1. Preservation (anchor) loss — slows the drift

`blank_prompt_preservation: true` in the `train:` block anchors every step to
the **frozen base model**: the trainer computes the base prediction with the
LoRA deactivated and adds an MSE term pulling the LoRA'd model's blank-prompt
output back toward it (`SDTrainer` preservation path). Your LoRA still learns
the data; the anchor resists the systematic un-distillation.

**Why it works.** Think of it as a leash to the factory weights: "whatever you
learn from the prompts, when given *no* instructions you must still behave
exactly like you did from the factory." The un-distillation drift shows up in
the unconditional behavior first — this term measures that drift directly
(base vs. LoRA'd prediction on a blank prompt) and charges the optimizer for
it. Unlike contrastive guidance, it doesn't reshape *what* is learned, it just
penalizes movement — which is why it trades off against learning speed.

```yaml
train:
  blank_prompt_preservation: true
  blank_prompt_preservation_multiplier: 1.0   # raise to preserve more, learn slower
```

Caveats:
- Costs extra forwards per step (roughly 2x step time), on top of the
  contrastive guidance forward when both are enabled.
- Audio is now anchored too: upstream fixed the audio side-channel handling
  for prior/guidance/preservation passes (primary-prediction tracking on the
  batch), so the earlier video-only limitation no longer applies.
- It is a dial, not a cure: multiplier too low and the model still drifts, too
  high and it learns nothing. Start at 1.0.

### 2. True CFG sampling — recovers drifted checkpoints, and diagnoses them

The H3 pipeline previously accepted `guidance_scale` and ignored it. It now
implements **real two-pass classifier-free guidance** when
`sample.guidance_scale > 1`: conditional and unconditional prompts are packed
into their own sequences (text length shifts the rotary media clock, so each
pass needs its own layout) over shared latent state, and both video and audio
velocities are extrapolated by the guidance scale.

**Why it works.** If training eroded the baked-in boost, the model has drifted
back toward being a *normal* diffusion model — and normal diffusion models are
exactly what CFG was invented for. Real CFG re-applies at playback the
amplification the weights no longer carry: two takes per step (with prompt,
without prompt), exaggerate the difference. The further a checkpoint has
drifted, the more real CFG gives back.

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

**The "underwater" audio problem.** H3 snaps clip lengths down to the 17n+5
frame grid at 24 fps, and by default (`shrink_video_to_frames`) the *whole*
clip is retimed to the snapped length — so nearly every training clip's audio
gets time-stretched. Both stretch paths audibly degrade it: the
pitch-preserving path is a phase vocoder without phase locking (the classic
"muffled / reverberant / underwater" phasiness artifact), and the plain path
is linear interpolation (pitch shift + high-frequency rolloff + aliasing).
The audio head adapts fast, so a fine-tune learns the artifact within a few
hundred steps while video still looks fine.

Fixes in this fork:
- **Stretch avoidance**: when the audio length is within 1% of target, the
  trainer now trims/pads instead of stretching. Clips whose durations sit
  exactly on the frame grid never engage the stretch at all — **the best fix
  is to pre-trim clips to 17n+5/24-second durations at 24 fps with 32 kHz
  audio, the model's native rate** — the audio VAE runs at 32 kHz and the
  trainer resamples anything else down to it, so delivering 32 kHz directly
  skips a resample hop (the Oxen converter does this automatically with
  ffmpeg).
- **Vocoder overlap**: the STFT hop is now capped at `n_fft/4` (4x overlap).
  Previously 44.1 kHz sources ran at ~2x overlap, far below what a phase
  vocoder needs, making them sound dramatically worse than 48 kHz sources.

Remaining knobs:
- **`audio_preserve_pitch: true`** (dataset config): when a stretch does
  happen, preserve pitch instead of tape-speed shifting.
- **Silent clips**: items without a soundtrack become pure-noise audio rows
  with *no* loss (fine), but in mixed batches with cached audio latents,
  missing items get zero-filled *normalized* latents — which decode to the
  per-channel mean, not silence. Either train silent datasets with
  `do_audio: false`, or don't mix silent and sounded clips in one dataset.
- `audio_loss_multiplier` is applied to an unweighted global mean added after
  the video loss; if audio learns faster/slower than video, tune it (LTX-2
  experience suggests ~0.5 when audio dominates).

### 4. Pruned-checkpoint stability, and the unpruned escape hatch (upstream)

The default training checkpoint is the **pruned** DiT: its 13B parameters of
timestep conditioning (a 2688-dim time-embedding MLP plus full-rank
`[96768 x 2688]` AdaLN projections per block) are distilled into a 1025-entry
**8-dimensional** lookup table with tiny factored fp16 projections. That is
calibrated for inference on the released sampling grid — training stresses it
harder: timesteps are sampled continuously and every gradient backprops
through the factored modulation.

Upstream fixes now in this fork:
- **fp16 AdaLN overflow**: the pruned checkpoints' factored AdaLN layers ran
  in fp16 and could overflow during training; they are now upcast to fp32.
- **Finite-loss guard**: the trainer checks the loss is finite (not just
  non-NaN) before backpropagating, so an infinity from a bad step no longer
  corrupts the optimizer state.
- **Layer offloading**: the AdaLN projection layers move correctly when layer
  offloading is enabled.
- **Unpruned variants are now loadable**: `model_kwargs.partition` accepts
  `fl2va` (unpruned int8, 34GB, full timestep MLP and full-rank AdaLN) and
  `ref2va`, alongside the pruned defaults (`fl2va_pruned`, `ref2va_pruned`).

Fork fix on top of those: the unpruned checkpoints' quantized AdaLN
projections ship a bias (RMS ≈ 0.16, with ≈ −0.3 baseline offsets on the
modulation components), and the loader was silently dropping it — the model
was built on the wrong assumption that only pruned AdaLN layers carry one.
Every block's shift/scale/gate came out shifted, which compounded across the
50 blocks into garbage output from the very first sample. The bias now loads,
and the quantized-checkpoint importer refuses to silently drop a nonzero bias
anywhere else.

If quality plateaus below the base model — especially on audio, whose
modulation shares the same 8-dim bottleneck — the unpruned variant is the
experiment to run: `model_kwargs: { partition: fl2va }` costs ~13GB more VRAM
and removes the timestep-conditioning approximation entirely. Match the
serving variant to the training variant when it matters.

### 5. Containment hyperparameters (no code)

If training without preservation or an adapter: LR ≤ 5e-5, rank ≤ 16,
≤ 1000 steps, and sample every ≤ 250 steps — the best checkpoint is usually
well before the last. Style transfers tolerate drift better than
subject/identity learning.

**Why.** The drift is *cumulative erosion*, not a threshold: every step
removes a sliver of the baked-in boost, and learning rate × steps × rank is
roughly how much total movement you allow. Cooler and shorter means less
erosion for the same amount of subject learning, and early checkpoints capture
the point where "learned your data" and "still sharp" overlap best.

## What the other knobs actually do

- **`timestep_type: shift` (video shift 12).** During training, each step
  practices denoising at a randomly chosen noise level. The `shift` schedule
  concentrates that practice on the same heavily-noise-shifted grid the H3
  sampler actually visits at inference (video models spend most of their
  sampling budget at high noise, where motion and composition are decided).
  Training on any other distribution practices noise levels the model will
  rarely be asked to handle, and under-trains the ones it will.
- **`lora_rank` / `lora_alpha`.** The LoRA is a low-rank "overlay" on the
  frozen weights; rank is the overlay's capacity (how much new behavior it can
  express) and alpha is its volume knob (how strongly it's applied). More
  capacity learns faster — and erodes the distillation faster, which is why H3
  wants modest rank.
- **`audio_preserve_pitch`.** Clips must land on the 17n+5 frame grid at a
  fixed 24 fps, so nearly every clip gets time-stretched. Naive stretching is
  the tape-speed effect: slow it down and the audio drops in pitch. With
  preserve-pitch on, the stretch is done phase-aware so duration changes but
  pitch doesn't — otherwise the audio head trains on subtly detuned sound.
- **`audio_loss_multiplier`.** Video and audio denoise jointly in one
  sequence and their losses are simply added; this is the mixing fader between
  them. If preview audio learns faster than video (or vice versa), rebalance.
- **`cache_latents_to_disk` / `cache_text_embeddings`.** The VAE encode of a
  clip and the Qwen3-VL encode of a caption are deterministic — cache them
  once instead of recomputing every epoch. Pure speed; no quality effect.
- **`auto_frame_count` + `batch_size: 1`.** Every clip keeps its own on-grid
  length instead of being cropped to one duration — but variable lengths can't
  be stacked into one batch, hence batch size 1 (use gradient accumulation for
  effective batch).
- **`guidance_loss_target: 4.0`.** The boost strength the contrastive target
  assumes the weights carry. It's a taste parameter more than a correctness
  one: higher trains toward a punchier, more saturated look; lower toward
  flatter output.

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
deliberate un-distillation. Ostris released an official alpha
(`ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_alpha.safetensors`),
now the default training method upstream and in this fork: it produces better
results than contrastive guidance loss and is faster (no extra blank-prompt
forward per step). The recipe below produces your own.

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

### How the adapter runs in this fork

`model.assistant_lora_path` accepts a local file, a filename under
`MODELS_PATH/loras`, or a `user/repo/file.safetensors` HuggingFace path
(downloaded on first use). The adapter loads as a frozen LoRA network over
the transformer — never merged into the quantized base weights — and stays
active during training, so your LoRA trains against the un-distilled
behavior. The sampler deactivates it for previews, so samples reflect the
distilled model your LoRA will actually serve against.

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
