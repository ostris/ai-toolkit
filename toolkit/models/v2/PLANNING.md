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

- [x] `text_encoders/qwen3.py` — Qwen3TextEncoder + `OstrisTransformersMixin`
      backend + `BaseModel.prepare_text_encoder` policy helper; the 3 verbatim
      TE stanzas (z_image, z_image_l2p, zeta_chroma) replaced. Verified with
      real Z-Image weights (load + encode on GPU).
- [x] `text_encoders/qwen3_vl.py` — Qwen3VLTextEncoder with
      `drop_vision_tower` / `patch_vision_patch_embed`; the 4 identical
      `patch_qwen_vl_patch_embed` copies (krea2, mageflow, boogu_image,
      Qwen3VLCaptioner) consolidated; TE loads migrated in krea2, mageflow,
      nucleus_image. Still on their own paths: ideogram4 (loads via AutoModel),
      minimax_h3 (custom truncated/prequantized comfy load — port later),
      qwen_image (Qwen2.5-VL, needs its own class)
- [x] `text_encoders/t5.py`, `text_encoders/clip.py` — T5TextEncoder,
      CLIPTextEncoder, CLIPTextEncoderWithProjection; migrated chroma ×2,
      flux_kontext, f_light (T5 stanzas → `prepare_text_encoder`, fixing their
      `qtype` → `qtype_te` bug) and hidream (CLIP ×2 + T5 with subfolder
      overrides; slow-tokenizer classes preserved via `use_fast=False`)
- [x] `vae/qwen_image.py` — QwenImageVAE + QwenImageVAEHolderMixin (frame-dim +
      latents mean/std handling built in, tiling opt-in via
      `vae_decode_tiled_on_low_vram`); the triplicated encode/decode deleted
      from qwen_image, nucleus_image, krea2 and all three VAE loads routed
      through the v2 loader
- [x] `vae/autoencoder_kl.py` — KLVAE (diffusers AutoencoderKL through the
      universal loader); migrated the scattered loads in chroma, flux_kontext,
      f_light, hidream, z_image
- [x] `vae/flux2_kl.py` — the BFL-style Flux2 KL autoencoder unified from the
      flux2 + ideogram4 copies (both files deleted; flux2's
      encode/decode/small-decoder superset + ideogram4's diffusers key
      converter). Verified bit-identical to both originals (weights, encode/
      decode outputs, converter mapping) and round-tripped real ae.safetensors
      weights on GPU. Packing/normalization stays per-model — flux2 packs
      `(c pi pj)` with BatchNorm running stats, ideogram4 packs `(ph pw c)`
      with its latent_norm tables; the conventions are incompatible.
- [x] z_image — transformer, TE (qwen3), and VAE (KLVAE) all on v2 modules.
      z_image_l2p still has its local progressive-transformer subclass
      (rebasing it onto the v2 class deferred; its TE is migrated)
- [x] qwen_image family — `v2/diffusion_models/qwen_image.py` (single-file
      loads stay on diffusers' from_single_file until the comfy flip) +
      `v2/text_encoders/qwen25_vl.py` (slow tokenizer preserved); edit
      variants inherit
- [x] nucleus_image — `v2/diffusion_models/nucleus_image.py`, TE stanza
      collapsed to prepare_text_encoder
- [x] krea2, ideogram4, mageflow — DiTs on the mixin via the `config=`
      passthrough (holder builds config from model_kwargs/holder state, mixin
      does build/markers/whitelist/casting; plain nn.Module classes now work
      with the default builder). krea2 + ideogram4 verified by harness;
      mageflow untestable while its repo 404s
- [x] chroma, chroma_radiance — both vendored Chroma classes now carry
      `OstrisModelMixin` with the block-count sniff moved into a new
      `aitk_config_from_state_dict` hook (mixin now supports checkpoint-derived
      configs + `load_from_state_dict` for non-safetensors sources, used by
      radiance's .pth path). zeta_chroma transformer left as-is: its config
      depends on holder state (patch_size), not the checkpoint
- [x] flux_kontext — `v2/diffusion_models/flux.py` (FluxTransformer2DModel);
      whole model now loads through v2 (transformer, T5, CLIP, KLVAE)
- [x] flux2 + zeta_chroma DiTs on the mixin via `config=` passthrough
      (flux2_klein_4b verified by harness). Every arch's DiT now loads
      through the mixin except: ltx2 family (one-file→two-modules split,
      helper delegated), anima (diffusers modular pipeline), ace_step
      (bundled single-file loader), z_image_l2p's local subclass, and the
      grandfathered legacy stable_diffusion_model archs
- [x] minimax_h3 (+ ref2va) transformer ported to the mixin: config sniffed
      from the checkpoint via aitk_config_from_state_dict (adaln_t_table),
      marker attach + stored-precision load via the new
      `aitk_cast_on_load = False` knob. Verified on the real pruned convrot
      file: 200 ConvRot linears, pruned table detected, fp32/fp16/bf16 mix
      preserved, no meta leftovers. Its TE stays custom (50-layer truncation
      + key_map). ltx2.5's `_load_quantized_module` now delegates to the
      mixin's whitelist/meta helper (~30 lines deleted); its full port is
      blocked on the one-comfy-file → transformer+connectors split, which
      doesn't fit the per-class single-file shape — revisit with the live
      server's component model
- [x] wan21 / wan22 family — `v2/diffusion_models/wan.py`
      (WanTransformer3DModel, both wan22 dual loads included) +
      `v2/text_encoders/umt5.py` (UMT5TextEncoder + PatchedT5Tokenizer;
      `loaders/umt5.py` is now a thin compat shim, `comfy_files` still
      reserved for Phase 2 — no local comfy umt5 file to verify the key
      conversion against). wan21's TE `qtype` → `qtype_te` bug fixed via
      prepare_text_encoder
- [x] hidream family — vendored transformer carries the mixin;
      `v2/diffusion_models/hidream.py` wraps the diffusers class for
      hidream_e1; both load via the switchable `hidream_transformer_class`
      through `load_model`
- [x] omnigen2 — vendored transformer carries the mixin, load migrated
- [x] boogu_image, ernie_image, prx_pixel_t2i — their vendored diffusers-style
      DiT classes now carry OstrisModelMixin (subfolder + block names on the
      class) and the holders load via `load_model`
- [x] f_light — DiT class carries the mixin (`aitk_subfolder="dit_model"`),
      load migrated
- [ ] anima — loads through diffusers modular pipelines (AnimaModularPipeline);
      not a mixin fit, revisit at Phase 2
- [ ] flux2 DiT — holder-config params classes (Flux2/Klein variants), defer
      like krea2/mageflow
- [ ] ace_step — one bundled safetensors holds model+TE+VAE+tokenizer via its
      own load_models; decomposing into v2 components is its own task
- [ ] Per-model fixes folded in as each migrates: `qtype_te` bug, dequant-on-save
      (`dequantize_if_quantized` everywhere), raw-`quantize()` → `quantize_model()`

### Phase 2 — comfy weights become the preferred source

Decisions:
- Comfy weights come from the Comfy-Org hub repos (per-model repos, comfy
  layout nested under `split_files/` — stripped when placing files into
  MODELS_PATH). Repos ship several precision variants of each component.
- **Selection preference: convrot8 > float8 mixed > float8 > bf16 > fp16**
  (`resolver.comfy_precision_rank`; nvfp4/unmarked rank last and are only
  used when explicitly listed). **Local-first**: the best-ranked LOCAL
  candidate wins; only when no candidate is local is the best-ranked one
  downloaded.
- Per-model candidate lists (`aitk_comfy_weight_names`) hold only the
  variants the class can actually digest. Constraint discovered: comfy
  convrot/quantized files carry markers on the ORIGINAL module layout (e.g.
  z_image's fused attention.qkv) — diffusers-layout classes with split
  modules can't attach them until they grow fused-layout support; until then
  those models list bf16/fp8 variants only. (Vendored comfy-layout classes —
  minimax/ltx pattern — take convrot directly.)
- The standard repo still supplies the config; local dirs and unregistered
  repos load as-is; `model_kwargs.use_comfy_weights: false` opts out.

- [x] Mechanism: `resolver.comfy_precision_rank` / `comfy_local_rel` /
      `resolve_comfy_candidates` + `OstrisModelMixin.resolve_comfy_weights`,
      integrated into `load_model` (comfy file preferred for registered
      standard repos, downloaded into the shared comfy layout)
- [x] First wired model: z_image (Comfy-Org/z_image_turbo, bf16 candidate).
      Verified end-to-end against the real shared ComfyUI folder
      (MODELS_PATH=/mnt/Models/comfy_models): standard-repo name_or_path
      resolved to the locally-present comfy file and produced the identical
      generation to the diffusers-shards load
- [x] qwen_image wired: its comfy files use the diffusers key layout
      directly — `fp8mixed` (float8-mixed, rank 1) attaches its 839
      `float8_e4m3fn` markers straight onto the class; candidates fp8mixed →
      fp8_e4m3fn (raw cast) → bf16. Verified with real weights: loaded the
      shared folder's local fp8 file (local-first, no download) and generated
      correctly. Holder skips re-quantization for prequantized checkpoints.
- [x] New `float8_e4m3fn` Ostris backend (toolkit/util/float8_quant.py):
      ComfyUI's fp8 + per-tensor-scale storage with dequantized matmul, in
      get_ostris_quantizer + comfy import/export. Round-trip verified.
- [x] wan family wired: comfy wan files (original key layout) convert via
      diffusers' own `convert_wan_transformer_to_diffusers` (rename-only, so
      quantized weight/scale keys ride along with their modules). Candidate
      keys support `(repo, subfolder)` tuples for wan2.2 A14B's dual DiTs
      (transformer = high noise, transformer_2 = low noise) and per-entry
      comfy-repo overrides ({"repo": ..., "files": [...]}) since wan2.1 and
      2.2 files live in different Comfy-Org repos. Wired: 2.2 TI2V-5B,
      T2V/I2V-A14B (fp8_scaled), 2.1 T2V 1.3B/14B, I2V 480P/720P. Verified
      with real weights: wan21 1.3B (downloaded comfy bf16) and wan22 5B
      (local comfy fp16) both load through the converter and generate video.
      Fix along the way: the mixin's meta build now uses accelerate
      init_empty_weights (params meta, buffers real) so init-computed
      non-persistent buffers like wan's rope tables materialize.
- [x] Legacy ComfyUI scaled-fp8 support (`scaled_fp8` marker + per-layer
      fp8 weight / scalar scale_weight, e.g. every wan *_fp8_scaled file):
      imports onto the float8 backend; scale_input (activation quant) is
      dropped, matmuls run dequantized.
- [x] wan comfy-format saves: `convert_state_dict_on_save` inverts diffusers'
      rename table (base/t2v/i2v; vace/animate excluded — their reverse
      mappings collide). Round-trip verified on both real comfy files
      (exact; the 2.1 file's legacy model.diffusion_model. prefix drops per
      the modern convention) and with real weights (load → save → 825-key
      original-layout file → reload bit-equal). wan21 + wan22_5b save one
      comfy file; wan22_14b saves the comfy-standard _high_noise/_low_noise
      pair instead of two diffusers folders.
- [ ] Wire remaining archs' candidate lists (chroma/others as their key
      conversions are verified per file)
- [x] Fused-layout quantized attach for diffusers-split classes:
      `split_fused_quantized_keys` / `fuse_split_quantized_keys`
      (comfy_quant_import) do exact out-dim row surgery on quantized comfy
      entries for all three formats (int8 rows+scales slice; fp8 scalar and
      nvfp4 per-tensor scales shared; nvfp4 block scales
      unswizzle→split→reswizzle). z_image's load/save converters use them, so
      its convrot8 candidate is live and top-ranked. Unit-verified exact both
      directions.
- [x] Comfy-format save: `save_model` auto-keeps quantized storage
      (comfy_quant markers) for convrot8 / nvfp4 / convrotcomfyw4a4 layers via
      `toolkit/util/comfy_quant_export.py` (inverse of comfy_quant_import;
      nvfp4 nibbles re-swapped + scales re-swizzled to the cuBLAS tile
      layout), plain layers save at bf16; partially-exportable models fall
      back to dequantized. Round-trip verified: save → mixin reload → outputs
      match for convrot8, nvfp4, and plain layers.
- [x] Save unification started: z_image and qwen_image holders now save
      comfy-format single files via the mixin regardless of how they loaded
      (z_image's dual-style branch deleted). Real round trip verified: the
      published z_image int8_convrot file loads (270 quantized linears,
      split-attach), resaves to the IDENTICAL 857-key comfy layout with
      bit-exact fused qkv weights/scales/markers, and the reload's quantized
      forward is bit-identical — toolkit saves are byte-compatible with
      ComfyUI.
- [x] chroma + chroma_radiance saves flipped to the mixin (their class keys
      ARE the original layout) — also fixes their quanto-only dequant bug
      (torchao/Ostris weights now dequantize on save). Tiny-model round trip
      verified. Save flips so far: z_image, qwen_image, wan21, wan22_5b,
      wan22_14b (dual files), chroma ×2.
- [ ] flux_kontext comfy wiring deferred: its Comfy-Org repo ships a single
      legacy-fp8 file in fused BFL layout — needs the flux fused-split
      conversion (split_fused_quantized_keys pattern + BFL↔diffusers maps)
- [ ] Flip the remaining per-arch `save_model` overrides as each arch's
      save-side key conversion is in place
- [ ] Publish/verify comfy repacks per model as they flip

### Phase 3 — live server
- [ ] Component-level identity (which TE/VAE instances are shared between archs) so
      a model switch drops only what the next run doesn't need
- [ ] Resident process: request comes in → diff requested components vs loaded →
      unload/load the difference
- [ ] Legacy `stable_diffusion_model.py` archs: grandfather or port last

## 100% mixin coverage (2026-08-28)

Every component of every non-legacy arch now loads through OstrisModelMixin —
DiTs, text encoders, VAEs, vision encoders, connectors/vocoders, and the
custom cases that previously bypassed it:

- New wrapper classes: llama, gemma3 + gemma4, mistral3 (×2), qwen3 base,
  qwen3-vl base + text-only, wan VAE, diffusers flux2 KL VAE, CLIP vision
  (first vision encoder), cosmos DiT, anima text conditioner, the full ltx2
  set (transformer, video/audio VAEs, connectors, vocoders).
- Existing-class swaps: omnigen2 (mllm + VAE), boogu (TE + VAE), klein TE,
  hidream llama, ideogram4 TE, prx TE.
- Custom restructures: minimax video/audio VAEs and TE, MageVAE
  (deferred-load ctor), the shared flux2_kl AutoEncoder (small-decoder sniff
  as a class hook; flux2 + ideogram4 route through it), ace_step's bundle
  (per-component class loaders), anima (modular-pipeline load replaced with
  component-wise v2 loads + update_components), hidream_o1's Qwen3VL DiT,
  z_image_l2p rebased onto the v2 class, ltx2's converter builds v2 classes.
- Mixin: kwargs flow through the single-file chain (component ctor args like
  MageVAE's sample_posterior).

Verified with real weights: ltx2.3 + ltx2.5 (full v2 family), anima,
ace_step_15 (new harness entry), minimax VAEs, plus the standing harness
coverage. The legacy monolith archs joined the system per the inference-engine
goal ("send a job with any base_model, reload/unload components on the fly"):
six new wrapper classes complete the component vocabulary (UNet2DCondition,
SD3/PixArt ×2/AuraFlow/Lumina2 transformers, Gemma2), and
`adopt_component` (in-place class swap, OstrisLinear-style) rebinds
pipeline-loaded components onto their wrappers at the monolith's single
post-load funnel — covering every legacy arch without touching its fifteen
load branches, with all pipeline references staying valid. Verified: sd1
(UNet/KLVAE/CLIPTextEncoder all mixin instances + generation) and sdxl
(dual-CLIP adoption + 1024² generation); harness gained sd1/sdxl entries,
a legacy scheduler fallback, and the sampler-name pass-through. Full
monolith decomposition (per-arch v2 loading with comfy candidates) remains
future work, but every resident component is now poolable by the engine.

## Unified load API (2026-08-28)

Loading policy moved out of the holders into the mixin. The surface:

- `ModelClassName.load(name_or_path, qtype=..., offload=..., dtype=...,
  device=..., ...)` — sourcing (`load_model`) + `aitk_post_load` in one call.
- `module.aitk_post_load(**kwargs)` — the post-load half alone, for holders
  whose checkpoint sourcing is custom (state-dict surgery, combined files).
  Handles: qtype (incl. `"qtype|ara_path"` accuracy recovery adapters, with
  prequantized-skip via `aitk_is_quantized`), block-streamed quantization
  (`quantize_module` in toolkit/util/quantize.py), MemoryManager layer
  offloading with per-class `get_offload_ignore_modules()` hooks, and device
  placement (low_vram parks on cpu).
- `BaseModel.component_load_kwargs(role)` derives the kwargs from
  model_config for roles "transformer" / "te" / "vae" (qtype/qtype_te, ARA
  recombination, offload percents, low_vram, te/vae devices).

All holders converted (transformer + TE quantize/offload/placement blocks
deleted): krea2, z_image, qwen_image, chroma ×2, flux_kontext, f_light,
nucleus, zeta_chroma, z_image_l2p, wan21 (+ subclasses), wan22_14b (dual:
per-transformer load, dual-ARA branch kept), boogu, ernie, ideogram4,
mageflow, prx_pixel, omnigen2, flux2 + klein, hidream + hidream_o1, ltx2 +
ltx2.5 (prequantized ConvRot flagged via aitk_is_quantized), minimax_h3,
anima (conditioner rides the transformer quantize flag at qtype_te),
ace_step_15, example_model (template now teaches the new API).
New `get_offload_ignore_modules` hooks: wan (scale_shift tables — also fixes
wan21 offload which previously offloaded them), ltx2 (all four per-block
tables), gemma3 (embed_tokens), gemma4 (embed_tokens + layer_scalar),
ideogram4 (rotary inv_freq + input/cond projections), ernie (x_embedder),
zeta ZImageDCT (pad tokens). New block-name hooks where only the holder had
them: Flux2, MageFlow, ZImageDCT, Ideogram4Transformer2DModel (+ its exclude
list on MageFlow). `prepare_text_encoder` remains only as the legacy-monolith
path.

## Load/quantize optimization round 1 (2026-08-28)

Baseline: testing/.model_test_outputs/report_baseline.md (+ per-arch
metrics_baseline.json). Profiling findings and fixes:

- The "slow quantization" in the baseline was mostly NOT quantize math
  (convrot8 on chroma's 57 blocks: 0.5s of GPU kernels). It was data
  movement: mmap-backed safetensors faulting pages in per-tensor cudaMemcpy
  order (cold cache on the 92%-full NVMe reads at ~0.32GB/s; the drive's
  sequential ceiling is ~435MB/s — cold loads are storage-bound), plus the
  old stream-quantize choreography crossing the bus three times (up, back
  into a fresh host copy, up again at final placement).
- `quantize_module(keep_on_device=True)`: when the final home IS the
  quantize gpu (no offload/low_vram — aitk_post_load detects this), blocks
  stay on the gpu after quantizing and the extras pass runs there too.
  Weights cross the bus once; the model-sized host-RAM spike of the
  intermediate copy is gone (chroma RSS peak 25.3 -> 17.3GB, the rest is
  reclaimable page cache; flux2's 93GB spike class eliminated). Warm-cache
  chroma blocks: 46s -> 3.9s; wan22_5b quantize 10.6s -> 3.8s.
- Offload/low_vram path keeps cpu residency but quantizes extras layer-by-
  layer via quantize_device instead of the all-cores cpu burst (was 350-650%
  cpu avg with 1600% spikes).
- `quantize()` skips the quantize_device round-trip for same-qtype
  OstrisLinears (guaranteed no-op) and, for ostris backends, for non-Linear
  leaves (norms/embeddings were being ferried across the bus for nothing).
- posix_fadvise(WILLNEED) readahead on single-file loads (mixin._readahead)
  — ~10-15% on cold reads, free when warm, no-op off POSIX.

Round 2 (same day):
- ARA path (`attach_ara_and_quantize`) now quantizes each hijacked linear on
  the gpu (was: wherever it lived, historically the cpu) with the same
  keep_on_device policy; the uint8 extras pass follows suit. Verified with
  hidream + its uint3 ARA end to end.
- `quantize_model` delegates its non-ARA branch to `quantize_module`
  (keep_on_device derived from model_config); dead `_quantize_model_blocks`
  removed; `patch_dequantization_on_save` made idempotent (double-patching
  would have recursed on save).
- Local-first hub loads: `_local_first` wraps from_pretrained / load_config /
  AutoTokenizer / AutoProcessor — fully-cached repos load with
  local_files_only (13x on VAE metadata, 0.53s -> 0.04s; a failed local
  attempt is instant, so uncached repos fall through to the online path at
  zero cost). Tokenizer-load slowness was DIAGNOSED as these per-process hub
  etag HEADs, not the use_fast=False sites.

Round 3 — vram spike elimination (2026-08-28):
- keep_on_device's extras pass had moved the whole unquantized remainder to
  the gpu in bf16 before quantizing (+7-10GB transient over final residency
  on big-extras models). Now `quantize(keep_on_quantize_device=True)`
  quantizes each extra layer on the gpu and leaves it — transient is one
  layer; non-quantized leftovers move only at final placement. Same for the
  ARA uint8 pass.
- convrot8 `quantize_` chunked: takes the weight at stored dtype
  (wants_fp32_weight=False; ConvRotIntN keeps True) and rotates+quantizes in
  256MB row-chunks — a 152k-vocab TE projection's transient fell 7.6GB ->
  1.4GB. Sub-threshold layers keep the one-shot path, verified bit-identical.
- qwen_image measured: standard load peak 35.3 -> 27.5GB (below the 28.0
  baseline; process peak now set by generation); low_vram load quantizes the
  whole 20B stack never holding >2.4GB gpu.

Ideas not yet done: pinned-buffer staged h2d (~+60% warm-upload bandwidth,
9.5 vs 5.8GB/s measured), overlapping h2d with quantize kernels (small — the
kernels are ~2% of the pass), wan22 low_vram generate choreography (165s of
unload/reload per sample), holders' legacy "just to make sure" .to(device)
lines that move quantized TEs to gpu even under low_vram (qwen's Preparing
Model stage parks 7.6GB — policy question, not a quantize spike).

## Offload hardening + final certification (2026-08-28)

The harness gained a per-arch 100%-offload probe (attach MemoryManager at
offload_percent=1.0 to the DiT + TEs + connectors/vision towers, then a
2-step sample — 2 not 1: diffusers' shift_terminal stretch divides by zero
at a single step). It flushed out and fixed, in the manager:
- fully-offloaded modules now REPORT the compute device (class swap with
  stringification-stable __module__/__qualname__ — transformers keys its
  hidden-states capture registry by str(class)); bouncing ops also pull cpu
  activations to the staged device
- weights tied to embeddings (lm_head <-> embed_tokens) are never pinned out
  from under the unmanaged embedding
- ignore_modules are moved to the compute device at attach
- the nested attach walk no longer double-lists managed modules as unmanaged
  (a managed Embedding was getting hauled back to gpu by pipeline.to)
- NEW: bouncing embeddings — vocab tables >64MB stay cpu-resident with the
  row gather done ON cpu (only looked-up rows cross the bus); tied lm_heads
  fall through to normal bouncing
- wan22's dual wrapper skips its whole-14B .to() swaps when memory-managed,
  and offload attaches per sub-transformer (pipelines hold those directly)

Measured offload-phase peaks: wan22_14b 14.7 -> 1.5GB, ltx2.5 14.0 ->
11.8GB (remainder: video activations + resident VAEs/vocoder by design),
most image archs 1-4GB vs 10-53GB resident. flux2's Mistral warnings
silenced behavior-preservingly (tie_word_embeddings=False, explicit
fix_mistral_regex=False, processor_kwargs — tokenization verified
bit-identical). ConvRot skip notices gated to layers >=1M weights.

FINAL CERTIFICATION: full sweep, 33/34 PASS, 0 FAIL, 1 SKIP (mageflow
upstream 404) — load + convrot8 quantize + generate + 100%-offload forward
for every arch on the complete optimized stack.

## Testing

- [x] `testing/test_model_loading.py`: per-arch load + one small sample through
      the normal training-style flow (get_model_class → load_model →
      generate_images). `--arch X` runs one in-process; `--all` runs every
      registered arch in its own subprocess (full unload between archs).
      15 archs registered so far — add each model type as it migrates.
- [x] Missing weights skip rather than fail: default is HF_HUB_OFFLINE=1 and
      hub/file errors classify as SKIP; `--allow-download` opts into fetching.
      (GPU + local-weights test, not CI-portable.)
- [x] Final certification sweep (2026-08-27, post-polish): 14/14 runnable
      archs PASS — comfy-source loads (zimage convrot8, qwen fp8, wan ×2 +
      fp8 umt5 TE), all ported holder-config DiTs, and the migrated
      quantize_model paths (chroma, flux_kontext, f_light block-streamed) in
      one run; mageflow remains the upstream 404 skip. One regression caught
      and fixed: qwen's _load_single_file override needed the new config
      kwarg.
- [x] Full sweep run 2026-08-27: 14/15 PASS (zimage, qwen_image, krea2,
      boogu_image, ernie_image, ideogram4, hidream_o1, anima, wan21, wan22_5b,
      chroma, flux_kontext, flux2_klein_4b, ltx2.3 — the quantized 22B ltx
      stack doesn't fit 32GB, needs the 96GB card). mageflow blocked
      upstream: microsoft/Mage-Flow-Base 404s on the hub (cached locally, so
      it runs offline — recheck whether the repo moved/went private).
- [x] Registry carries realistic per-arch sample settings (native res, steps,
      CFG) so sweep outputs are visually verifiable, not just "a file
      exists". Verified: all 14 produce proper generations. Findings from
      the quality pass: boogu emits a black frame below native res at
      low-step/high-CFG (settings regime, present pre-restructure, not a
      migration bug); chroma's FakeCLIP hardcoded device 'cuda' broke any
      non-cuda:0 run (pre-existing, fixed — FakeCLIP now takes the real
      device); ideogram4's fp8 release renders its own "blocked by safety
      filter" card for a plain cat prompt (model behavior, not a bug —
      investigate its trigger).
- [x] Round-trip verified for the first comfy-save arch: z_image convrot
      load → comfy save → identical key set + bit-exact quantized entries vs
      the published file → reload → bit-identical quantized forward. Extend
      per arch as saves flip.
- [x] Full UI-default coverage (2026-08-27): the registry now holds all 31
      UI-facing archs, and 30/31 load + generate through the new stack with
      their real UI defaults (mageflow blocked upstream by its hub 404).
      This round verified the previously-untested tail with real weights:
      wan22_14b + i2v (comfy fp8 pairs via the legacy importer, candidate
      keys added for the ai-toolkit bf16 default repos), wan21_i2v, hidream,
      hidream_e1 (native 768² editing), nucleus, omnigen2, ltx2.5, flux2,
      flux2_klein_9b, prx_pixel, zeta_chroma, zimage_l2p, both qwen edit
      archs, f-lite. Fixes found by the run: ltx2.5's fp32 scale_shift
      tables promoted hidden states into bf16 linears under the diffusers
      class — fixed ComfyUI-style (toolkit/util/mixed_precision.py):
      per-op input casting hooks on every weighted module + the stored-fp32
      tensors pinned against parent .to(dtype) casts (device moves pass
      through), so the tables stay genuinely fp32 at sample time. The
      mechanism is general — any mixed-precision comfy checkpoint on a
      diffusers-class arch can use it; harness configs
      for e1 resolution and zeta/l2p extras_name_or_path corrected to match
      the UI defaults.
- [ ] Each newly migrated model adds its test in the same PR as its migration.

## TODO / look at later

- [x] Quantize consolidation: `quantize_model` now honors `quantize_kwargs`
      (blocks + extras) and tolerates missing block names; the chroma ×2,
      flux_kontext, f_light, omnigen2 raw-quantize sites migrated onto it
      (gaining block streaming, excludes, ARA, dequant-on-save patching) with
      holder block names added. Remaining raw sites are legacy/extension
      (flex2, cogview4, stable_diffusion_model). The ARA uint8 hardcode
      stands — revisit if a non-uint8 ARA base is ever wanted.
- [x] Last known `qtype_te` bugs fixed (flux2's Mistral TE, anima's
      text_conditioner) — 9/9 sites from the survey now correct outside the
      grandfathered legacy monolith (cogview4/legacy SD remain as-is).
- [x] wan comfy-TE resolved for real: UMT5TextEncoder carries comfy
      candidates (fp8_e4m3fn_scaled via the legacy importer, fp16), files
      already in transformers key layout (spiece blob dropped, tied
      embed_tokens materialized). Verified: wan21 samples with the local
      comfy fp8 TE. The loaders/umt5.py `comfy_files` param stays as a
      no-op shim for old callers.
- [x] Registry hardening: unknown archs now raise with the known-arch list
      (legacy monolith archs whitelisted via LEGACY_ARCHS). Still open: lazy
      per-arch imports; single source of truth shared with the UI's
      `options.tsx` model list.
- [x] Stub dedup where identical: chroma_radiance imports FakeCLIP/FakeConfig
      from chroma_model. The other Fake* copies (hidream_o1, flux2, zeta)
      carry model-specific values — left in place.
- [ ] Vendored upstream code (hidream/src, omnigen2/src, ltx2 converter's private
      comfy-quant parser): dedupe against toolkit utils where practical.
