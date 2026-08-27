# v2 Model Module Restructure — Planning

## Goal

Every component the toolkit loads (DiTs/transformers, unets, text encoders, vision
encoders, VAEs, audio VAEs) becomes a class extending one base module in
`toolkit/models/v2`. `BaseModel` (`toolkit/models/base_model.py`) stays as the
multimodal holder that each arch in `extensions_built_in/diffusion_models` extends —
that layer is good. The layer below it is what gets unified: one loading entry point,
one quantization path, one save path, shared component definitions instead of
per-model-folder copies.

End state this enables:

- **Live server with model hot-swap**: a resident process where, when a generation or
  training run requests a different model, the unused components are dropped and the
  new ones loaded. Shared component classes (same TE/VAE reused across archs) make
  component-level reuse possible instead of full teardown/reload.
- **Model loading test suite**: a test that loads each registered arch one at a time
  and runs an inference pass. Every model type gets added to this suite as it is
  migrated (see Testing below).
- **Comfy-aligned weights**: weights live in the ComfyUI folder layout under
  `MODELS_PATH` (shareable with a ComfyUI install), download there when missing, and
  saves are comfy-format. Eventually defaults move to comfy / our own prequantized
  releases for everything.

## Current state (survey 2026-08-27)

Three generations of loading conventions coexist:

1. Legacy monolith `toolkit/stable_diffusion_model.py` (`is_flux` / `is_v3` branches);
   still the silent fallback in `toolkit/util/get_model.py` when an arch string
   doesn't match.
2. `BaseModel` subclass per arch (35 registered classes), each with a hand-written
   `load_model()` / `save_model()`.
3. `toolkit/models/v2/_mixin.py` (`OstrisModelMixin`) — the intended fix, currently
   used by one model (`v2/z_image.py` → z_image extension).

### Duplication highlights

- BFL KL autoencoder: full copies in `flux2/src/autoencoder.py` and
  `ideogram4/src/vae.py` (header says "Flux2 KL autoencoder").
- Qwen3-VL text encoder loaded independently in qwen_image, nucleus_image, krea2,
  ideogram4, mageflow, minimax_h3 — from several different repo sources; krea2
  hand-patches the vision tower locally.
- `Qwen3ForCausalLM` TE load: 3 verbatim line-for-line copies
  (`z_image/z_image.py:257`, `z_image/z_image_l2p_model.py:471`,
  `zeta_chroma/zeta_chroma_model.py:146`).
- Flux1 VAE + T5 + CLIP trio loaded 4x from 2 different repos (chroma ×2,
  flux_kontext, legacy SD path).
- Comfy-file resolver copy-pasted: `minimax_h3/minimax_h3.py:248`
  (`_resolve_comfy_file`) → `ltx2/ltx2.py:1254` ("mirrors MinimaxH3Model").
- `AutoencoderKLQwenImage` latents mean/std handling triplicated (qwen_image,
  nucleus_image, krea2).
- `transformer.` ↔ `diffusion_model.` LoRA key rename copy-pasted ~15x in
  `convert_lora_weights_before_save/load` overrides.
- Fake CLIP/TE/config stubs redefined in ~5 places
  (canonical: `toolkit/models/FakeVAE.py`, `toolkit/unloader.py`).

### Inconsistency highlights

- **Quantization, 5 paths**: `quantize_model()` (block-streaming, ARA-aware — ~21
  users), raw `quantize()` (~10 users, no block streaming/excludes, but the only path
  honoring `quantize_kwargs`), hidream's hand-rolled block loop, the v2 mixin's own
  `quantize_`, and bare TE quantization everywhere.
- **Known bugs**: ~9 sites quantize the TE with `qtype` instead of `qtype_te`
  (chroma ×2, flux2, flux_kontext, cogview4, wan21, legacy SD, ...);
  `toolkit/models/loaders/umt5.py` accepts a `comfy_files` param it never uses, so
  wan21's comfy-TE path is a silent no-op.
- **Saving, 4 incompatible styles**: diffusers `save_pretrained` folders, flat
  safetensors, safetensors-inside-diffusers-folder hybrids, and z_image's
  loaded-format-dependent branch. Dequant-on-save done 3 ways; the
  `isinstance(v, QTensor)` variant (chroma, flux2, boogu_image, ideogram4) misses
  torchao and Ostris weights entirely. Every `save_pretrained` override ignores its
  `save_dtype` argument.
- Registry: linear scan in `toolkit/util/get_model.py`, silent SD1 fallback on a
  typo'd arch, eager import of every model file at startup. Second unsynchronized
  registry in `ui/src/app/jobs/new/options.tsx`.

## Decisions (locked in)

1. **Save format = ComfyUI format.** Single-file safetensors in comfy key layout.
   Must support saving quantized — primarily convrot8 and nvfp4 (comfy_quant marker
   format, see `toolkit/util/comfy_quant_import.py`) — and plain bf16, all in comfy
   format. Loading stays backwards compatible: diffusers dirs, transformers repos,
   and single files all still digest through `load_model`; only saving standardizes
   on comfy.
2. **v2 folder layout mirrors the comfy save path structure**:

   ```
   toolkit/models/v2/
     _mixin.py            # base module (OstrisModelMixin, evolving)
     resolver.py          # comfy-layout weight resolution (lift from minimax_h3)
     diffusion_models/    # one file per DiT/unet family
     text_encoders/       # qwen3_vl.py, qwen3.py, t5.py, clip.py, gemma.py, ...
     vae/                 # flux_kl.py, qwen_image.py, wan.py, audio VAEs, ...
     vision_encoders/
   ```

3. **Method names win from `BaseModel`**: `get_transformer_block_names` and
   `get_quantization_exclude_modules`. The mixin's `get_quantization_block_names`
   gets renamed to match; resolve the classmethod-vs-instance-method mismatch while
   doing so.
4. **Loading policy**:
   - Per-model special handling is allowed via the hook methods.
   - If `name_or_path` is a diffusers/transformers source, load it with
     diffusers/transformers for now. **Step 1 is migrating every model to the v2
     module format and loader without breaking anything** — same weights, same
     sources, same results.
   - Each model declares a `comfy_weight_names` dict keyed per standard
     `name_or_path`. If the user points at a local folder or a non-standard repo,
     load it as-is. If `name_or_path` is the standard repo and we have matching
     comfy weight names, load those instead when any of them exist (locally under
     `MODELS_PATH` in comfy layout, or downloadable to there).
   - Eventually the default flips to comfy weights / our own prequantized releases
     for everything.

## Base module: what `OstrisModelMixin` still needs

The mixin already handles: diffusers dir / hub repo / local single file /
`org/repo/file.safetensors`, key-conversion hooks on load and save, overridable
backend hooks for transformers-lib models, block-wise quantize.

To add:

- [x] **Comfy weight spec + resolver.** `aitk_comfy_repo` / `aitk_comfy_weight_names`
      class attrs + `find_comfy_weights` (local-only until Phase 2); resolution chain
      generalized from `minimax_h3._resolve_comfy_file` into `v2/resolver.py`:
      explicit override → `MODELS_PATH` at the repo-relative comfy path → flat at
      root → recursive walk of the category folder → hub download **to the
      repo-relative path** (folder stays shareable with ComfyUI, no duplicate
      downloads).
- [x] **Automatic prequantized import.** Single-file path sniffs `comfy_quant`
      markers and routes through `import_comfy_quantized_layers` before
      `load_state_dict`, including the OstrisLinear missing-key whitelist that
      minimax_h3 and ltx2 each hand-rolled.
- [x] **One save path.** `save_model(path, dtype)`: dequantize via
      `dequantize_if_quantized` (honors dtype), run `convert_state_dict_on_save`,
      write single-file comfy-layout safetensors. (Quantized-storage saves —
      convrot8 / nvfp4 with comfy_quant markers — land with Phase 2; diffusers-folder
      save as an explicit flag still to add.)
- [x] **Tokenizer/processor declaration** for text encoders
      (`aitk_tokenizer_repo`/`aitk_processor_repo` + `load_tokenizer`/`load_processor`).
- [x] Rename quantization hooks to the `BaseModel` spellings (decision 3):
      `get_transformer_block_names` (classmethod on the module).

## Migration steps

Track progress here; check items off as they land.

### Phase 0 — foundation (done 2026-08-27)
- [x] Evolve `_mixin.py` per the list above (comfy spec local-only until Phase 2)
- [x] Create `v2/diffusion_models/`, `v2/text_encoders/`, `v2/vae/`,
      `v2/vision_encoders/`; move `v2/z_image.py` → `v2/diffusion_models/z_image.py`
- [x] Lift the comfy resolver out of minimax_h3 into `v2/resolver.py`; point
      minimax_h3 and ltx2 at it (delete their copies)
- [x] `BaseModel` default `convert_lora_weights_before_save/load` doing the
      `transformer.` ↔ `diffusion_model.` rename, gated on the class attr
      `lora_keys_use_comfy_prefix` (default False, so passthrough models keep
      their behavior); the ~18 identical overrides replaced with the flag.
      Custom conversions (anima, hidream_o1, ltx2, wan21) keep their overrides;
      ltx2's now composes with the flag via super().

### Phase 1 — migrate all models to v2 modules, no behavior change
Every arch's components become v2 classes; if `name_or_path` is diffusers, it still
loads via diffusers. Nothing about sources or outputs changes yet. Suggested order
(worst duplication first), each including its loading test (see Testing):

- [ ] Shared TEs: `text_encoders/qwen3_vl.py`, `text_encoders/qwen3.py`,
      `text_encoders/t5.py`, `text_encoders/clip.py`
- [ ] Shared VAEs: `vae/flux_kl.py` (kills flux2 + ideogram4 copies and the 4
      scattered `AutoencoderKL` loads), `vae/qwen_image.py` (mean/std handling
      built in)
- [ ] z_image / z_image_l2p (already half-migrated)
- [ ] qwen_image family (qwen_image, qwen_image_edit, qwen_image_edit_plus)
- [ ] nucleus_image, krea2, ideogram4, mageflow
- [ ] chroma, chroma_radiance, zeta_chroma
- [ ] flux2, flux_kontext
- [ ] minimax_h3 (+ ref2va), ltx2 family
- [ ] wan21 / wan22 family
- [ ] hidream family, omnigen2
- [ ] anima, boogu_image, ernie_image, f_light, prx_pixel_t2i
- [ ] audio_models (ace_step)
- [ ] Per-model fixes folded in as each migrates: `qtype_te` bug, dequant-on-save
      (`dequantize_if_quantized` everywhere), raw-`quantize()` → `quantize_model()`

### Phase 2 — comfy weights become the preferred source
- [ ] Wire `comfy_weight_names` per model; standard-repo `name_or_path` + existing
      comfy weights → load comfy
- [ ] Comfy-format save (bf16 + convrot8/nvfp4 quantized) as the default
      full-weight save
- [ ] Publish/verify comfy repacks per model as they flip

### Phase 3 — live server
- [ ] Component-level identity (which TE/VAE instances are shared between archs) so
      a model switch drops only what the next run doesn't need
- [ ] Resident process: request comes in → diff requested components vs loaded →
      unload/load the difference
- [ ] Legacy `stable_diffusion_model.py` archs: grandfather or port last

## Testing

- [ ] `testing/` (or `tests/`) harness: for each migrated arch, load the model via
      its v2 modules and run one small inference pass (single low-step sample; video
      models at minimum frame count). One arch at a time, full unload between archs.
- [ ] Weights resolved through the normal resolver against `MODELS_PATH`
      (GPU + local-weights test, not CI-portable at first; skip archs whose weights
      are absent rather than failing).
- [ ] Round-trip test per model: load → save comfy format → reload from the save →
      outputs match (bf16) / load cleanly (quantized saves).
- [ ] Each newly migrated model adds its test in the same PR as its migration.

## TODO / look at later

- [ ] Quantize-path consolidation quirks: `quantize_kwargs` is honored only by the
      raw `quantize()` call sites and silently dropped by `quantize_model()`; the
      ARA path inside `quantize_model` hardcodes `uint8`. Decide the unified
      behavior when consolidating.
- [ ] `toolkit/models/loaders/umt5.py` dead `comfy_files` param (wan21 comfy-TE
      no-op) — fix when wan migrates.
- [ ] Registry hardening: error (don't fall back to SD1) on unknown arch; lazy
      per-arch imports; single source of truth shared with the UI's
      `options.tsx` model list.
- [ ] Fake/stub components: consolidate on `toolkit/models/FakeVAE.py` /
      `toolkit/unloader.py`, delete local copies.
- [ ] Vendored upstream code (hidream/src, omnigen2/src, ltx2 converter's private
      comfy-quant parser): dedupe against toolkit utils where practical.
