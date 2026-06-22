# Example Model — a template for adding a new architecture to ai-toolkit

This folder is a complete, heavily commented template for wiring a brand-new
diffusion model into ai-toolkit. It assumes the worst (and most common) case:
**diffusers does not have your model**, so you vendor the network and a minimal
sampling pipeline yourself.

It is intentionally **not registered** — it never appears as a trainable arch.
It exists purely as a guide for people (and agents) adding image, editing,
video, or i2v models.

## File map

```
example/
├── README.md          <- you are here
├── __init__.py        <- exports ExampleModel (registration notes inside)
├── example_model.py   <- the BaseModel subclass: every override documented
│                         with exact inputs/outputs
└── src/               <- everything diffusers does NOT provide
    ├── model.py       <- a minimal DiT with the gradient-checkpointing pattern
    └── pipeline.py    <- a minimal embeds-only flow-matching sampler
```

## How a model gets registered

1. `toolkit/util/get_model.py:get_all_models()` scans every package directly
   under `extensions/` and `extensions_built_in/` for a module-level
   `AI_TOOLKIT_MODELS` list.
2. For models in this folder, that list lives in
   `extensions_built_in/diffusion_models/__init__.py` — import your class
   there and append it to `AI_TOOLKIT_MODELS`.
   (Alternatively, give your model its own folder under `extensions/` with its
   own `AI_TOOLKIT_MODELS` list — see `extensions/z_image_pixel/`.)
3. The class attribute `arch` (e.g. `"example"`) is matched against
   `model.arch` in the training config YAML to pick your class.
4. To expose it in the web UI, add an entry to
   `ui/src/app/jobs/new/options.ts` (search for an existing arch like
   `ideogram4` to copy the shape).

Minimal config YAML to train it:

```yaml
model:
  arch: "example"
  name_or_path: "/path/to/weights"   # folder with transformer/, text_encoder/,
                                     # tokenizer/, vae/
  quantize: true        # optional: qfloat8 the transformer
  quantize_te: true     # optional: qfloat8 the text encoder
train:
  gradient_checkpointing: true
```

## Lifecycle — who calls what, in order

1. **Load** — `load_model()` builds the transformer, text encoder(s),
   tokenizer(s), VAE and scheduler and stores them on `self`. Everything else
   reads `self.model` / `self.vae` / `self.text_encoder`.
2. **Caching (optional)** — before training, the trainer may call
   `encode_images()` per dataset image (latent cache) and
   `get_prompt_embeds()` per caption (text-embed cache, saved via
   `AdvancedPromptEmbeds.save`, one file per caption).
3. **Train step** (every step, see `extensions_built_in/sd_trainer/SDTrainer.py`):
   1. clean latents come from the cache or `encode_images()`
   2. noise + timestep are sampled; `add_noise()` (BaseModel) mixes them
   3. `condition_noisy_latents(noisy_latents, batch)` — your hook to inject
      control/reference conditioning
   4. `get_noise_prediction(latent_model_input, timestep, text_embeddings)` —
      the forward pass, under autograd
   5. loss = MSE(prediction, `get_loss_target(noise=..., batch=...)`)
4. **Sampling previews** — `generate_images()` (BaseModel) encodes each sample
   prompt with `get_prompt_embeds()`, then calls your
   `get_generation_pipeline()` once and `generate_single_image(...)` per
   prompt. Your pipeline only ever receives **embeds, never text**.
5. **Saving** — full fine-tunes go through `save_model()`. LoRA files are
   written by the network code, with your
   `convert_lora_weights_before_save/load()` mapping keys to the public
   convention (usually the `diffusion_model.` prefix).

## Conventions to keep straight

- **Pixels** are `(B, 3, H, W)` in `[-1, 1]` (control tensors arrive in
  `[0, 1]` — multiply by 2 and subtract 1 before encoding).
- **Latents** are `(B, C, h, w)`; video latents are `(B, C, frames, h, w)`.
- **Timesteps** cross the BaseModel API on a `0..1000` scale where 1000 is
  pure noise. Convert to your model's native convention inside
  `get_noise_prediction` — and watch for models whose native time runs the
  other way (t=1 = clean); flip and/or negate there (ideogram4 does both).
- **Flow-matching target** in this codebase is `noise - clean`
  (`get_loss_target`), i.e. the velocity pointing from data to noise.
- `self.model` / `self.transformer` / `self.unet` are aliases for the same
  thing on BaseModel.
- **`use_old_lokr_format = False`** — set this class attribute on every NEW
  model. `BaseModel` defaults it to `True` purely for backwards-compatibility
  with LoKr checkpoints trained before the format change; all new architectures
  should use the new LoKr format. (Plain LoRA training is unaffected — this only
  matters for `network.type: "lokr"`.)

## AdvancedPromptEmbeds

`toolkit/advanced_prompt_embeds.py`. The flexible container for text
conditioning, preferred for all new models over the older `PromptEmbeds`:

- Every key holds a **list of tensors, one per batch item**
  (`AdvancedPromptEmbeds(text_embeds=[t0, t1, ...])`). Store each item at its
  natural length and pad to the batch max only at the model call
  (`src/pipeline.py:pad_prompt_embeds`) — caches stay small and any prompts
  can share a batch.
- **Keep each per-item tensor 2D `(L, D)`.** This is a hard requirement, not a
  convention: `BaseModel.predict_noise` infers the text batch size from the
  embed list, and it only counts the list as one-per-item when each tensor is
  2D (`len(text_embeds[0].shape) == 2`). A 3D per-item tensor is read as an
  already-batched `(B, L, D)` and its *first axis* is taken as the batch size —
  so a single 3D prompt of length `L` looks like a batch of `L`, and training
  dies with *"Batch size of latents must be the same or half the batch size of
  text embeddings."* If your conditioning has an extra axis (e.g. a stack of N
  encoder layers, giving `(L, N, D)`), **flatten it into the feature axis**
  (`(L, N*D)`) in `get_prompt_embeds` and **restore it** (`reshape(B, Lt, N, D)`)
  in `get_noise_prediction` / the pipeline, right before the model call.
- Add as many keys as your model needs (`pooled_embeds`, image features, …).
- Keys that must not be dtype-cast (token ids, masks) go in
  `embeds.frozen_dtype_keys`.
- CFG concat (`concat_prompt_embeds`), batch expansion, `.to()`, `.save()` /
  `.load()` for the disk cache are all handled for you.

If you ever change what `get_prompt_embeds` produces, bump the
`text_embedding_space_version` property so stale on-disk caches invalidate.

## Gradient checkpointing

With `train.gradient_checkpointing: true`, `BaseSDTrainProcess` calls
`model.enable_gradient_checkpointing()` if it exists, else sets
`model.gradient_checkpointing = True`. Your network re-runs each block under
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` when the flag is
set **and** `torch.is_grad_enabled()` is true — never gate on `self.training`.
See `src/model.py` for the full pattern and rationale.

## Quantization

With `quantize: true`, `quantize_model` swaps every `nn.Linear` for an
`optimum.quanto` quantized one. Their matmul kernel **only accepts 2D or 3D
activations** (`assert activations.ndim in (2, 3)`) — a `Linear` you feed a 4D
tensor works fine in bf16 but throws once quantized. If your network applies a
`Linear` over a 4D tensor (e.g. projecting a `(B, L, D, N)` layer axis),
reshape to 3D for the call and back afterwards.

Also watch out for **slow bf16 kernels on vendored components**: `Conv3d` has no
fast cuDNN bf16 path (it falls back to a slow one). If a frozen sub-model carries
a `Conv3d` you don't actually run — e.g. a vision tower's patch embed on a VL
text encoder — drop it (`text_encoder.model.visual = None`) to skip loading it;
if you must run one, consider running that component in fp16/fp32.

## Attention backends (don't force flash-attn)

Reference repos very often hard-code an attention kernel — `flash_attn`,
xformers, sage — and import it at module top level. **Do not carry that
requirement over.** ai-toolkit has to import and load your model on machines
where that package isn't installed (CPU boxes, headless CI, plain installs), so
a top-level `from flash_attn import ...` turns "load the model" into an
`ImportError`.

The rule:

- **Default to torch's built-in `F.scaled_dot_product_attention`** (the
  "native" backend). It needs no extra dependency, runs on CPU and CUDA, and
  already dispatches to a fused/flash kernel on supported hardware. `src/model.py`
  does exactly this.
- **Make any other kernel OPTIONAL**, selected at runtime — never required at
  import. The clean pattern:
  1. Guard the import so a missing package is a flag, not a crash:
     ```python
     try:
         from flash_attn import flash_attn_varlen_func
         _FLASH_ATTN_AVAILABLE = True
     except ImportError:
         flash_attn_varlen_func = None
         _FLASH_ATTN_AVAILABLE = False
     ```
  2. Give each attention module an `attention_backend` flag (default
     `"native"`) and **branch inside its forward** — `"flash"` runs the flash
     kernel, anything else runs SDPA.
  3. Expose a `set_attention_backend("native"|"flash")` on the parent model
     that validates the name, raises a clear error if `"flash"` is requested
     while `_FLASH_ATTN_AVAILABLE` is `False`, and propagates the flag to every
     attention module.
  4. Wire it to a config knob so it stays opt-in, e.g.
     `model_kwargs.attention_backend: "flash"` read in `load_model`.

Branch on a per-module **flag**, don't swap the processor/module instance:
attention modules that own trained q/k/v weights (joint/dual-stream blocks)
would lose those weights if you replaced them with a different instance.

Worked implementations to copy: `../ideogram4/src/transformer.py`
(`set_attention_backend`, native+flash in one `Attention.forward`) and
`../boogu_image/src/attention_processor.py` (guarded import, per-processor
`attention_backend` flag, flash varlen branch alongside SDPA).

## Adapting this template

### Editing / instruct model (image in, image out)
- In `condition_noisy_latents`, encode `batch.control_tensor`
  (`(B, 3, H, W)` in `[0, 1]`) with the VAE and attach it to the noisy
  latents — extra channels (`torch.cat(..., dim=1)`) or extra sequence tokens.
  Slice the prediction back down in `get_noise_prediction` before returning.
  Reference: `../flux_kontext/flux_kontext.py`.
- If the text encoder must *see* the control image (VL encoders), set
  `self.encode_control_in_text_embeddings = True`; `get_prompt_embeds` then
  receives `control_images`. Reference: `../qwen_image/qwen_image_edit.py`.
- Multiple reference images: `self.has_multiple_control_images = True`
  (`batch.control_tensor_list`). Reference:
  `../qwen_image/qwen_image_edit_plus.py`.
- In `generate_single_image`, load `gen_config.ctrl_img` (a file path) and run
  the same conditioning for previews.

### Video model (t2v)
- Batches arrive as `(B, frames, 3, H, W)`; latents as
  `(B, C, frames_latent, h, w)`. Override `encode_images`/`decode_latents`
  for your video VAE (temporal compression means
  `frames_latent = (frames - 1) // 4 + 1` for most VAEs).
- `gen_config.num_frames` drives previews; return a **list of PIL frames**
  from `generate_single_image` and the harness saves a video.
- Reference: `../wan22/wan22_5b_model.py` and `../ltx2/`.

### Image-to-video (i2v)
- Same as video, plus first-frame conditioning: in `get_noise_prediction`
  take frame 0 from `batch.tensor` (declare `batch` in your signature to
  receive it), encode it, and merge it into the latent input. For previews do
  the same with `gen_config.ctrl_img`.
- Reference: `../wan22/wan22_14b_i2v_model.py` and
  `toolkit/models/wan21/wan_utils.py:add_first_frame_conditioning`.

### Other useful hooks (all on `toolkit/models/base_model.py:BaseModel`)
| Override | When you need it |
|---|---|
| `get_model_to_train()` | LoRA should attach to something other than `self.model` |
| `text_embedding_space_version` / `latent_space_version` | invalidate users' caches after a breaking change |
| `te_padding_side` | LLM text encoders that need left padding |
| `is_multistage`, `multistage_boundaries` | multi-expert models split by timestep range (`../wan22/wan22_14b_model.py`) |
| `load_training_adapter()` pattern | assistant LoRAs (de-distillation adapters), see `../z_image/z_image.py` |
| `get_latent_noise_from_latents()` | custom noise (default: `randn_like`) |
| `encode_audio()` | audio-conditioned models (`../ltx2/`) |
