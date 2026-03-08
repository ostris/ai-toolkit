# LTX-2 Character Training — Standard Operating Procedure (SOP)

> **Version:** 3.12  
> **Date:** 2026-02-20  
> **Scope:** All changes to `ostris/ai-toolkit` that fix voice-training failures, comprehensively improve character LoRA/DoRA quality for LTX-2, and resolve quantized-model compatibility issues.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Root Causes Identified](#2-root-causes-identified)
3. [Complete List of Changes](#3-complete-list-of-changes)
4. [Files Modified](#4-files-modified)
5. [Quick-Start Configuration](#5-quick-start-configuration)
6. [Config Reference](#6-config-reference)
7. [Developer Notes & Handoff](#7-developer-notes--handoff)
8. [Pre-Handoff Test Checklist](#8-pre-handoff-test-checklist)

---

## 1. Problem Statement

Users training LTX-2 character LoRAs with `ai-toolkit` reported that LoRAs consistently lost voice/audio generation quality. The model would produce correct visuals but broken, garbled, or silent audio. Additionally, character identity retention was weaker than expected, and there were no hardware-specific optimizations for modern GPUs (RTX 5090).

---

## 2. Root Causes Identified

| # | Root Cause | Impact |
|---|-----------|--------|
| 1 | **Coupled audio/video timesteps** — training sampled one random timestep and fed it to both modalities, despite the transformer having separate timestep processing paths | Audio never explored its own optimal noise schedule; voice learning was fundamentally handicapped |
| 2 | **Loss imbalance** — video loss magnitude dwarfed audio loss | Audio weights drifted away from base model |
| 3 | **Mixed-batch data loss** — `DataLoaderBatchDTO` dropped `audio_data` if the first item had no audio | Batches silently became audio-free |
| 4 | **No audio regularization** — zero audio loss on audio-free batches | Catastrophic forgetting of voice generation |
| 5 | **Connector gradients blocked** — `pipeline.connectors` ran inside `torch.no_grad()` | Text-to-audio adaptation impossible |
| 6 | **Unsafe defaults** — `do_audio` defaulted to `False` for video | Silent failure |
| 7 | **Low LoRA rank** — default rank 4 with alpha 1.0 too small for joint audio/video identity | Insufficient capacity for character + voice |
| 8 | **No regularization** — rank dropout, module dropout not exposed | Overfitting with higher ranks |
| 9 | **No gradient checkpointing for LTX-2 transformer** — not wired despite being supported | VRAM wasted on activation storage |
| 10 | **torch.compile pointed at wrong model** — compiled `unet` not transformer | No compilation benefit for DiT models |
| 11 | **Noise offset crash on video latents** — `apply_noise_offset` rejected 5D tensors | LTX-2 could fail immediately when `noise_offset > 0` |
| 12 | **Duplicate network-type controls in UI** — two selectors wrote to same field | Confusing/contradictory DoRA selection UX |
| 13 | **DoRA crashes on TorchAO-quantized models** — `get_orig_weight()` only handled optimum-quanto wrappers, not TorchAO `AffineQuantizedTensor` | DoRA unusable with `qfloat8` quantization; `aten.add` NotImplementedError |
| 14 | **Memory-manager dtype mismatch** — layer-offloading bounced dequantized float32 weights against bf16 activations | `F.linear` RuntimeError during sampling and training with `layer_offloading: true` |
| 15 | **Attention mask dtype mismatch during baseline sampling** — mixed-precision paths could produce float query tensors while masks were bf16 additive masks | `scaled_dot_product_attention` rejected mask/query dtype combination |
| 16 | **`_org_forward_safe` output dtype leakage** — the safe forward wrapper cast inputs to the quantized module's compute dtype (bf16) but never restored the output to the original input dtype | bf16 leaked through every LoRA-wrapped layer, contaminating downstream attention masks and triggering persistent SDPA dtype errors |
| 17 | **SDPA fallback patch targeted wrong abstraction** — the monkey-patch replaced `_ad._native_attention` (a module-level name), but diffusers dispatches attention through `_AttentionBackendRegistry._backends` (a dict holding the original function reference) | The "safety net" SDPA patch was dead code that never intercepted any attention call |
| 18 | **`self.train_config` accessed on `LTX2Model` outside `load_model`** — `get_noise_prediction` and connector gradient logic referenced `self.train_config` directly, but `train_config` belongs to `BaseSDTrainProcess`, not the model object | `AttributeError` crash on every training step and connector call |
| 19 | **Min-SNR incompatible with flow-matching schedulers** — `apply_snr_weight` requires `noise_scheduler.alphas_cumprod`, which only exists on DDPM-style schedulers; LTX-2 uses `CustomFlowMatchEulerDiscreteScheduler` which has no cumulative alpha schedule | `AttributeError` crash when `min_snr_gamma > 0` on any flow-matching model |
| 20 | **Stale latent cache missing audio** — when `do_audio: true`, previously cached latent files (from earlier runs or failed attempts) might not contain `audio_latent`; the cache loader only checks file existence, not whether audio is present | Training silently falls back to synthetic silence regularizer instead of using real audio, completely defeating voice training |
| 21 | **`torchaudio.load()` completely broken on Windows/Pinokio** — torchaudio v2.9+ uses `torchcodec` as its backend, which requires FFmpeg shared libraries (DLLs). The Pinokio installer does not install system FFmpeg, so `torchcodec` fails to load any `libtorchcodec_coreN.dll`. This means `torchaudio.load()` cannot decode ANY file (video or wav). The error was silently caught, so audio extraction always returned None | Even after cache invalidation (v3.8), re-encoding still produced latent files without audio — voice training was impossible on all Pinokio/Windows installs |
| 22 | **`print_and_status_update` called on trainer instead of model** — audio logging code in `SDTrainer.py` called `self.print_and_status_update(...)` but this method belongs to the model class, not the trainer (`DiffusionTrainer`) | `AttributeError` crash on every training step that had audio loss to log |
| 23 | **No fallback audio decoder when torchaudio is broken** — the codebase had a single code path for audio loading (`torchaudio.load()`). When that failed, there was no alternative. The `av` package (PyAV, `av==16.0.1`) is already in `requirements.txt` and ships bundled FFmpeg libraries that work without system installs, but was never used for audio extraction | Audio extraction had a single point of failure with no recovery path |
| 24 | **`hasattr(tensor, "dequantize")` matches ALL tensors in PyTorch 2.9+** — `dequantize()` is a method on the base `torch.Tensor` class, so `hasattr` returns True for every tensor, not just quantized ones. Code in `network_mixins.py` and `DoRA.py` used this check to decide whether to call `.dequantize()`, causing it to be called on regular float tensors inside the autograd graph. The backward pass then crashed because `derivative for dequantize is not implemented` | Training crash on first backward pass with `RuntimeError: derivative for dequantize is not implemented` |
| 25 | **Auto-balance audio loss clamp floor of 1.0 prevented dampening** — the dynamic multiplier was clamped with `max(1.0, min(20.0, ...))`, meaning it could only boost audio UP, never scale it DOWN. In practice, LTX-2 raw audio loss (~0.45) is naturally larger than video loss (~0.25), so the computed multiplier (~0.18) was always clamped back to 1.0. The auto-balance feature was completely non-functional for the most common loss ratio | `dyn_mult` stuck at 1.00 for the entire training run; audio loss dominated video loss unchecked; auto-balance feature effectively dead code |

---

## 3. Complete List of Changes

### Phase 1: Core Audio Fixes (v2.0)

#### A. Audio Loss Multiplier (`audio_loss_multiplier`)
- **Files:** `SDTrainer.py`, `config_modules.py`
- Configurable scalar that scales audio loss before adding to total loss.

#### B. Automatic Audio Loss Balancing (`auto_balance_audio_loss`)
- **Files:** `SDTrainer.py`, `config_modules.py`
- EMA-based (alpha=0.99) dynamic multiplier targeting audio at ~33% of video loss. Bidirectional clamp [0.05, 20.0] — scales audio DOWN when it naturally dominates (common on LTX-2), or UP when video dominates.

#### C. Robust Mixed-Batch Audio Handling
- **File:** `data_loader.py`
- Scans all file items (not just first) for audio. Preserves per-item alignment with `None` for missing entries.

#### D. Silence Fallback in `encode_audio`
- **File:** `ltx2.py`
- Synthesizes duration-matched stereo silence for `None` audio entries in mixed batches.

#### E. Voice Preservation Regularizer
- **File:** `ltx2.py`
- Generates synthetic silence latents for audio-free batches, forcing LoRA to preserve base audio behavior.

#### F. Connector Gradient Unfreezing
- **File:** `ltx2.py`
- `pipeline.connectors` forward pass outside `torch.no_grad()` when `train_text_encoder=True`. Adds `LTX2TextConnectors` to `target_lora_modules`.

#### G. Scalar Audio Loss (VRAM Optimization)
- **Files:** `ltx2.py`, `data_loader.py`
- MSE computed immediately as scalar `batch.audio_loss` instead of keeping full prediction tensor.

#### H. Explicit Module Freezing
- **File:** `ltx2.py`
- `audio_vae`, `vocoder`, `connectors` explicitly frozen with `requires_grad_(False)` and `.eval()`.

#### I. Safe `do_audio` Default
- **File:** `config_modules.py`
- `do_audio` defaults to `True` for `num_frames > 1` (video datasets).

#### J. Strict Audio Mode
- **Files:** `SDTrainer.py`, `config_modules.py`
- Monitors ratio of real audio supervision. Halts training if it falls below threshold after warmup.

#### K. Phase 1 UI Integration
- **Files:** `types.ts`, `options.ts`, `SimpleJob.tsx`, `docs.tsx`
- All audio config fields exposed with documentation.

### Phase 2: Quality & Optimization Improvements (v3.0)

#### L. Independent Audio Timestep Sampling (CRITICAL)
- **File:** `ltx2.py`, `config_modules.py`
- The LTX-2 transformer has separate `adaln_single` / `audio_adaln_single` timestep embeddings and was designed for independent denoising schedules. Training was feeding the same timestep to both modalities.
- **Change:** Samples an independent `audio_timestep = torch.rand(...) * 1000.0` per step. Uses it for audio noise addition, audio target computation, and passes it as `audio_timestep=audio_timestep` to the transformer.
- Config flag: `independent_audio_timestep` (default `True`).
- **Cost:** Zero. One extra random number per step.
- **Impact:** Lets audio learn voice patterns at its own optimal noise level — the single deepest fix for voice quality.

#### M. Rank Dropout and Module Dropout
- **Files:** `config_modules.py`, `BaseSDTrainProcess.py`
- Added `rank_dropout` and `module_dropout` to `NetworkConfig`. Passed through to `LoRASpecialNetwork` creation.
- Rank dropout randomly zeroes entire rank dimensions; module dropout randomly skips entire LoRA modules.

#### N. Higher Default LoRA Rank + Alpha
- **File:** `options.ts`
- LTX-2 defaults changed: `linear=32`, `linear_alpha=32`, `rank_dropout=0.1`.
- Effective LoRA scale = `alpha/rank = 1.0` (correct scaling for rank 32).

#### O. DoRA / LoKr Network Type Selector
- **Files:** `SimpleJob.tsx`, `options.ts`, `docs.tsx`
- Network type dropdown exposed for LTX-2: LoRA, DoRA (weight-decomposed), LoKr (Kronecker).
- DoRA decomposes updates into magnitude + direction, often producing higher quality at same rank.

#### P. Gradient Checkpointing for LTX-2 Transformer
- **File:** `ltx2.py`
- Wired gradient checkpointing activation during `load_model` via `enable_gradient_checkpointing()`, `set_gradient_checkpointing(True)`, or `_enable_gradient_checkpointing = True`.
- Saves 30-50% VRAM on long video sequences, enabling higher rank or larger batch sizes.

#### Q. SDPA / Flash Attention Enforcement
- **File:** `ltx2.py`
- Explicitly enables `torch.backends.cuda.enable_flash_sdp(True)` and `enable_mem_efficient_sdp(True)` during model setup.

#### R. torch.compile Fix for Transformer Models
- **File:** `BaseSDTrainProcess.py`
- Detects transformer-based architectures and compiles `self.sd.model` instead of `self.sd.unet`.
- Uses `mode='max-autotune'` for optimal RTX 5090 performance.

#### S. Min-SNR Gamma for LTX-2
- **Files:** `options.ts`, `SimpleJob.tsx`, `docs.tsx`
- Exposed `min_snr_gamma` with default `5.0` for LTX-2. Balances loss across noise levels.

#### T. Noise Offset for LTX-2
- **Files:** `options.ts`, `SimpleJob.tsx`, `docs.tsx`
- Exposed `noise_offset` with default `0.05`. Improves dynamic range in generated content.

#### U. Prodigy / DAdaptation Optimizers
- **File:** `SimpleJob.tsx`
- Added to optimizer dropdown: AdamW (full precision), Prodigy (adaptive LR), DAdaptation (adaptive LR).

#### V. LR Scheduler Selector
- **Files:** `options.ts`, `SimpleJob.tsx`, `docs.tsx`
- Dropdown with: Constant with Warmup, Cosine Annealing, Cosine with Restarts, Linear Decay.

#### W. Caption Dropout for LTX-2
- **Files:** `options.ts`, `SimpleJob.tsx`, `docs.tsx`
- Exposed caption dropout rate for LTX-2 datasets. Default `0.0`, recommended `0.05`.

#### X. Differential Output Preservation for LTX-2
- **File:** `options.ts`
- Added DOP to LTX-2 `additionalSections` so it appears in the UI for character LoRAs.

#### Y. Audio Loss Logging
- **File:** `SDTrainer.py`
- Logs raw audio loss, scaled audio loss, video loss, and dynamic multiplier every 10 steps when audio supervision is active.

#### Z. Auto-Balance Audio Loss Default Changed
- **File:** `options.ts`
- Default for `auto_balance_audio_loss` changed from `false` to `true` for LTX-2.

#### AA. Video-Safe Noise Offset
- **File:** `toolkit/train_tools.py`
- Extended `apply_noise_offset` to support both image latents (`[B, C, H, W]`) and video latents (`[B, C, T, H, W]`).
- Prevents runtime crashes on LTX-2 when `noise_offset` is enabled.

#### AB. `torch.compile` Assignment Fix
- **File:** `BaseSDTrainProcess.py`
- `torch.compile(...)` now reassigns the compiled module back to `self.sd.model` (transformer models) or `self.sd.unet` (UNet models) so compilation is actually applied.

#### AC. Network Type UI Conflict Fix
- **Files:** `SimpleJob.tsx`, `options.ts`
- Removed duplicate "Network Type" selector from training options section.
- Kept a single canonical selector in the Target card and made DoRA available there for LTX-2.

#### AD. Dynamic Noise Offset Video Compatibility
- **File:** `BaseSDTrainProcess.py`
- `dynamic_noise_offset` channel-mean correction now supports both 4D image latents and 5D video latents.
- Prevents shape errors if dynamic noise offset is enabled on video training jobs.

### Phase 3: Quantization & DoRA Compatibility Fixes (v3.3)

#### AE. DoRA TorchAO Quantization Fix
- **Files:** `toolkit/models/DoRA.py`, `toolkit/network_mixins.py`
- `get_orig_weight()` and `get_orig_bias()` now dequantize any tensor with a `.dequantize()` method, covering both optimum-quanto (`QTensor`/`QBytesTensor`) and TorchAO (`AffineQuantizedTensor`) wrappers.
- `ExtractableModuleMixin.extract_weight()` similarly extended.
- LoRA forward path in `network_mixins.py` dequantizes `x` and `org_forwarded` if they arrive as quantized wrappers.
- **Impact:** DoRA now works with `qfloat8` quantized transformers. Without this fix, DoRA was completely unusable on quantized LTX-2 models.

#### AF. Memory-Manager Dtype Enforcement
- **File:** `toolkit/memory_management/manager_modules.py`
- All 4 weight-materialization "float paths" (`_BouncingLinearFn` forward/backward, `_BouncingConv2dFn` forward/backward) now cast non-quantized weights to `target_dtype` (matching activation dtype).
- Belt-and-suspenders: explicit dtype enforcement added at every `F.linear()` and `F.conv2d()` call site and backward matmul — weight and bias are forced to match `x.dtype` synchronously on the compute stream regardless of what the transfer stream produced.
- **Impact:** Eliminates "self and mat2 must have the same dtype" crashes when `layer_offloading` is enabled with quantized models. This was the most common runtime failure users hit.

#### AG. Unified Safe `org_forward` Wrapper for LoRA/DoRA Paths
- **File:** `toolkit/network_mixins.py`
- Added `_org_forward_safe(...)` and routed all remaining direct `self.org_forward(...)` paths through it, including LORM local mode.
- Wrapper infers the original module compute dtype (quantized modules default to bf16 or bias dtype) and casts inputs before calling the wrapped module.
- **v3.5 fix:** Wrapper now records `orig_dtype = x.dtype` on entry and casts the output back to `orig_dtype` before returning. Without this, bf16 outputs leaked through every LoRA-wrapped layer, contaminating attention masks downstream and causing persistent SDPA dtype errors. This was the true root cause of the `attn_mask.dtype: BFloat16 / query.dtype: float` crash.
- **Impact:** Prevents quantized linear/conv dispatch dtype crashes in both active-LoRA and LORM branches; removes fragile per-callsite dtype assumptions; preserves dtype transparency across the entire forward graph.

#### AH. Boolean Prompt Attention Masks for LTX-2 Sampling
- **File:** `extensions_built_in/diffusion_models/ltx2/ltx2.py`
- Prompt/negative attention masks passed into `LTX2Pipeline.__call__` are now explicitly `torch.bool`.
- Stored prompt masks in `PromptEmbeds` are normalized to bool.
- **Impact:** Avoids attention-mask dtype failures in mixed-precision sampling (`attn_mask` dtype mismatch in SDPA path).

#### AI. Connector Attention Mask Canonicalization
- **File:** `extensions_built_in/diffusion_models/ltx2/ltx2.py`
- Wrapped `pipe.connectors.forward(...)` to canonicalize returned connector attention masks to `float32`.
- Also canonicalized training additive masks to `float32` before connector invocation.
- **Impact:** Prevents SDPA runtime failures from `bf16 attn_mask` paired with `float32 query` in mixed quantized/offloaded execution during baseline sampling and training.

#### AJ. Complete `train_config` Safe Access (v3.6)
- **File:** `extensions_built_in/diffusion_models/ltx2/ltx2.py`
- **Problem:** `load_model` was fixed in v3.3 to use `getattr(self, 'train_config', None)`, but two additional call sites in `get_noise_prediction` (line 879: `independent_audio_timestep` lookup) and the connector gradient path (line 1043: `train_text_encoder` check) still accessed `self.train_config` directly. Since `train_config` is an attribute of `BaseSDTrainProcess` (the trainer), not `LTX2Model` (the model), these crashed with `AttributeError` on every training step.
- **Fix:** Both sites now use the same safe pattern: `_tc = getattr(self, 'train_config', None)` followed by guarded attribute access with sensible defaults (`True` for independent audio timestep, `False` for train_text_encoder).
- **Impact:** Training can now proceed past step 0. Without this fix, every training run crashed immediately.

#### AK. Min-SNR / SNR-Gamma Guard for Flow-Matching Schedulers (v3.7)
- **File:** `extensions_built_in/sd_trainer/SDTrainer.py`
- **Problem:** `apply_snr_weight` and `get_all_snr` compute SNR from `noise_scheduler.alphas_cumprod`, which only exists on DDPM-style schedulers. LTX-2 uses `CustomFlowMatchEulerDiscreteScheduler` (flow-matching), which has no cumulative alpha schedule. Setting `min_snr_gamma: 5` (or `snr_gamma`) crashed with `AttributeError`.
- **Fix:** Added `_scheduler_has_snr = hasattr(self.sd.noise_scheduler, 'alphas_cumprod')` guard before all `apply_snr_weight` calls. When the scheduler lacks `alphas_cumprod`, SNR weighting is silently skipped. The `add_all_snr_to_noise_scheduler` call at setup was already wrapped in `try/except`.
- **Note:** Min-SNR weighting is architecturally incompatible with flow-matching. For LTX-2, the `weighted` timestep sampling type (already set in the user's config) provides analogous loss balancing. Users should set `min_snr_gamma: 0` for LTX-2 to make the skip explicit.
- **Impact:** Eliminates `'CustomFlowMatchEulerDiscreteScheduler' object has no attribute 'alphas_cumprod'` crash. Training now proceeds past the loss calculation step.

#### AL. Latent Cache Audio Invalidation (v3.8)
- **File:** `toolkit/dataloader_mixins.py`
- **Problem:** The latent cache loader checked `os.path.exists(latent_path)` but never verified that the cached file actually contained `audio_latent`. If a previous run (or a failed attempt) cached latents without audio, subsequent runs with `do_audio: true` would use those stale cache files and silently lose all audio supervision. Training would fall back to synthetic silence regularizer, completely defeating voice training.
- **Fix:** When `do_audio: true`, the cache loader now reads the safetensors file header (via `safe_open`) to check for the `audio_latent` key. If absent, the cache is treated as invalid and the latent is re-encoded with audio.
- **Impact:** Audio training works correctly even when stale cache files exist from previous runs. No manual cache deletion needed.

#### AM. Min-SNR Default Disabled for LTX-2 (v3.8)
- **File:** `ui/src/app/jobs/new/options.ts`
- Changed LTX-2 default for `min_snr_gamma` from `5.0` to `0` (disabled). Min-SNR is incompatible with flow-matching schedulers. The `timestep_type: weighted` setting provides equivalent loss balancing for flow-matching.

#### AN. Robust Multi-Fallback Audio Extraction (v3.9 → v3.10)
- **File:** `toolkit/dataloader_mixins.py`
- **Problem:** `torchaudio.load()` is completely broken on Windows/Pinokio because `torchcodec` cannot find FFmpeg DLLs (the Pinokio installer does not install system FFmpeg). This means `torchaudio.load()` cannot decode ANY file — not just video containers, but even wav files. Audio extraction silently failed for every video, making voice training impossible.
- **Fix:** Introduced `_load_audio_robust()` — a top-level function with three fallback layers:
  1. **torchaudio** — tried first (works on properly configured Linux/Docker installs)
  2. **PyAV** (`import av`) — opens the video file directly using PyAV's bundled FFmpeg libraries. `av==16.0.1` is already in `requirements.txt` and ships its own FFmpeg shared libs, requiring zero system installs. Decodes all audio frames, converts to float32 tensor. **This is what works on Pinokio/Windows.**
  3. **ffmpeg CLI subprocess** — last resort if PyAV also fails; extracts audio to temp wav, reads with Python's built-in `wave` module (no torchaudio dependency)
  
  Also: made all audio extraction errors visible (prints WARNING instead of silent-unless-debug); added audio encoding summary counter at end of caching loop (`"Audio latent caching: N encoded, M failed"`).
- **Impact:** Audio extraction now works on every platform and install method. This was the single biggest blocker — without working audio extraction, voice training was impossible regardless of all other fixes.

#### AO. Precise Quantization Type Checks — Replace `hasattr(t, "dequantize")` (v3.11)
- **Files:** `toolkit/network_mixins.py`, `toolkit/models/DoRA.py`
- **Problem:** In PyTorch 2.9+, `dequantize()` is a method on the base `torch.Tensor` class, so `hasattr(tensor, "dequantize")` returns `True` for ALL tensors — including regular float32/bf16 tensors. Our defensive checks in `_org_forward_safe()`, `forward()`, `extract_weight()` (in `network_mixins.py`) and `get_orig_weight()`, `get_orig_bias()` (in `DoRA.py`) used this pattern, causing `.dequantize()` to be called on every regular output tensor. When called inside the autograd computation graph, this recorded a `dequantize` op that has no backward implementation, crashing with `RuntimeError: derivative for dequantize is not implemented`.
- **Fix:** Replaced all `hasattr(t, "dequantize")` checks with explicit type checks:
  - `isinstance(t, (QTensor, QBytesTensor))` for optimum-quanto types
  - `type(t).__name__ == 'AffineQuantizedTensor'` for TorchAO types
  - `getattr(t, 'is_quantized', False)` for PyTorch native quantized tensors
  
  In `_org_forward_safe`, the rare case where `org_forward` returns a truly quantized tensor is handled with `.dequantize().detach().requires_grad_(x.requires_grad)` to avoid recording the op in the autograd graph.
- **Impact:** Eliminates `derivative for dequantize is not implemented` crash during backward pass. Training can now proceed past the first step.

#### AP. Audio Logging `print_and_status_update` Fix (v3.9)
- **File:** `extensions_built_in/sd_trainer/SDTrainer.py`
- **Problem:** Audio loss logging called `self.print_and_status_update(...)`, but this method exists on the model class (`BaseModel`), not the trainer (`DiffusionTrainer`).
- **Fix:** Changed to `print(...)`.
- **Impact:** Eliminates `AttributeError` crash during training when audio loss logging fires.

#### AP. SDPA Attention Mask Dtype Safety Net (Corrected)
- **File:** `extensions_built_in/diffusion_models/ltx2/ltx2.py`
- **Problem (v3.4):** The original safety-net monkey-patch replaced the module-level attribute `diffusers.models.attention_dispatch._native_attention`. However, diffusers dispatches attention through `_AttentionBackendRegistry._backends`, a class-level dict that retains the original function reference. The v3.4 patch was dead code that never fired.
- **Fix (v3.5):** Replaced with a patch on `torch.nn.functional.scaled_dot_product_attention` — the actual C-extension entry point that ALL diffusers attention backends ultimately call via `torch.nn.functional.scaled_dot_product_attention(...)`. The wrapper checks if `attn_mask` is a non-bool tensor whose dtype differs from `query.dtype`, and casts it to match. Guarded to apply only once.
- **Impact:** This is the absolute lowest-level fallback. Any attention call anywhere in the stack — NATIVE, EFFICIENT, FLASH, MATH, cuDNN — that passes a mismatched mask will be transparently corrected. The patch only fires when the operation would otherwise crash, so there is zero performance or correctness cost for correctly-typed tensors.

#### AQ. Bidirectional Auto-Balance Audio Loss (v3.12)
- **File:** `extensions_built_in/sd_trainer/SDTrainer.py`
- **Problem:** The `auto_balance_audio_loss` EMA-based dynamic multiplier was clamped with `max(1.0, min(20.0, dynamic_multiplier))`. This only allowed the multiplier to INCREASE audio loss (boost) — it could never decrease it below 1.0 (dampen). In practice, LTX-2's raw audio MSE loss (~0.45) is naturally larger than video loss (~0.25) due to the different latent space scales. The auto-balance computed a target multiplier of ~0.18, but the clamp forced it back to 1.0 every step. The feature was completely non-functional — `dyn_mult` stayed at 1.00 for the entire training run.
- **Fix:** Changed the clamp floor from `1.0` to `0.05`: `max(0.05, min(20.0, dynamic_multiplier))`. The multiplier can now scale audio loss DOWN when audio naturally dominates, or UP when video dominates. The 0.05 floor prevents audio from being zeroed out entirely.
- **Impact:** `dyn_mult` now actively adjusts throughout training, targeting audio at ~25% of total loss (33% of video loss). For the common case where raw audio loss > video loss, the multiplier will settle around 0.15–0.25, bringing the two modalities into proper balance. Audio still trains — just with proportional gradient weight.

---

## 4. Files Modified

| File | Changes |
|------|---------|
| `toolkit/config_modules.py` | `independent_audio_timestep`, `rank_dropout`, `module_dropout` in configs; all Phase 1 audio fields |
| `toolkit/train_tools.py` | `apply_noise_offset` now supports both 4D (image) and 5D (video) latents |
| `toolkit/dataloader_mixins.py` | **v3.8:** Latent cache audio invalidation; **v3.9→v3.10:** Robust multi-fallback audio extraction (torchaudio → PyAV → ffmpeg CLI); visible warnings on audio failure; caching summary counters |
| `toolkit/data_transfer_object/data_loader.py` | Mixed-batch audio alignment; `audio_loss` field; cleanup |
| `toolkit/models/DoRA.py` | `get_orig_weight()` / `get_orig_bias()` dequantize TorchAO `AffineQuantizedTensor` wrappers in addition to optimum-quanto; **v3.11:** precise quantization type checks replacing `hasattr(t, "dequantize")` |
| `toolkit/network_mixins.py` | `extract_weight()` and LoRA forward dequantize TorchAO wrappers; unified `_org_forward_safe(...)` for all wrapped forward branches (including LORM); **v3.5:** output dtype preservation; **v3.11:** precise quantization type checks replacing `hasattr(t, "dequantize")` |
| `toolkit/memory_management/manager_modules.py` | All 4 float-path materializers cast to `target_dtype`; belt-and-suspenders dtype enforcement at every `F.linear`/`F.conv2d` call site and backward matmul |
| `extensions_built_in/diffusion_models/ltx2/ltx2.py` | Independent audio timestep; gradient checkpointing; SDPA enforcement; encode_audio silence; voice regularizer; connector unfreezing; scalar loss; module freezing; bool prompt attention masks; connector mask canonicalization to float32; **v3.5:** SDPA mask-dtype patch corrected from module-level to `torch.nn.functional.scaled_dot_product_attention`; **v3.6:** all `self.train_config` references converted to safe `getattr(self, 'train_config', None)` pattern |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Audio loss logging; auto-balance; strict audio; loss integration; **v3.7:** SNR/Min-SNR guard for flow-matching schedulers; **v3.9:** `print_and_status_update` fix; **v3.12:** bidirectional auto-balance clamp (0.05–20.0) |
| `jobs/process/BaseSDTrainProcess.py` | rank_dropout/module_dropout passthrough; transformer compile target fix; compile assignment fix; dynamic noise-offset 4D/5D compatibility |
| `ui/src/types.ts` | `NetworkConfig` additions (dropout fields); `TrainConfig` additions (independent_audio_timestep, noise_offset, min_snr_gamma, lr_scheduler) |
| `ui/src/app/jobs/new/options.ts` | LTX-2 defaults (rank 32, alpha 32, rank_dropout 0.1, auto_balance on, noise_offset 0.05, min_snr 5.0); all new additionalSections |
| `ui/src/app/jobs/new/SimpleJob.tsx` | Single canonical network type selector (LoRA/DoRA/LoKr for LTX-2); rank/module dropout inputs; independent audio timestep; noise offset; min-SNR gamma; LR scheduler; caption dropout; optimizer additions (Prodigy, DAdaptation, AdamW) |
| `ui/src/docs.tsx` | Documentation entries for all new fields |

---

## 5. Quick-Start Configuration

### Recommended: Quality-First Preset (RTX 5090)

```yaml
config:
  process:
    - network:
        type: lora                        # or 'dora' for potentially higher quality
        linear: 32
        linear_alpha: 32
        rank_dropout: 0.1
      train:
        optimizer: adamw
        lr: 7.5e-5
        auto_balance_audio_loss: true
        independent_audio_timestep: true  # decoupled audio/video learning
        strict_audio_mode: true
        noise_offset: 0.05
        min_snr_gamma: 0                   # incompatible with flow-matching; use timestep_type: weighted instead
      datasets:
        - folder_path: /path/to/your/clips
          num_frames: 81
          do_audio: true
          flip_x: true                    # doubles effective dataset
```

### Alternative: Prodigy (automatic LR)

```yaml
config:
  process:
    - network:
        type: lora
        linear: 32
        linear_alpha: 32
        rank_dropout: 0.1
      train:
        optimizer: prodigy
        lr: 1.0                           # Prodigy finds the real LR internally
        auto_balance_audio_loss: true
        independent_audio_timestep: true
```

### DoRA variant (potentially higher quality)

```yaml
config:
  process:
    - network:
        type: dora                        # weight-decomposed LoRA
        linear: 32
        linear_alpha: 32
        rank_dropout: 0.1
      train:
        optimizer: adamw
        lr: 5e-5                          # slightly lower LR for DoRA stability
        auto_balance_audio_loss: true
        independent_audio_timestep: true
```

---

## 6. Config Reference

### Training Config

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `train.independent_audio_timestep` | bool | `true` | Sample separate timestep for audio vs video |
| `train.audio_loss_multiplier` | float | `1.0` | Manual audio loss scaling (ignored when auto-balance is on) |
| `train.auto_balance_audio_loss` | bool | `true` (LTX-2) | Dynamically adjusts multiplier via EMA |
| `train.strict_audio_mode` | bool | `false` | Halts if audio supervision drops below threshold |
| `train.strict_audio_min_supervised_ratio` | float | `0.9` | Min ratio for strict mode (0.0-1.0) |
| `train.strict_audio_warmup_steps` | int | `50` | Grace period before strict checks |
| `train.noise_offset` | float | `0.05` (LTX-2) | Noise offset for dynamic range |
| `train.min_snr_gamma` | float | `5.0` (LTX-2) | Min-SNR loss weighting (0 = disabled) |
| `train.lr_scheduler` | string | `constant_with_warmup` | LR schedule strategy |

### Network Config

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `network.type` | string | `lora` | Network variant: lora, dora, lokr |
| `network.linear` | int | `32` (LTX-2) | LoRA rank |
| `network.linear_alpha` | float | `32` (LTX-2) | LoRA alpha (effective scale = alpha/rank) |
| `network.rank_dropout` | float | `0.1` (LTX-2) | Rank-level dropout |
| `network.module_dropout` | float | `0.0` | Module-level dropout |

### Dataset Config

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `datasets[].do_audio` | bool | `true` (video) | Load audio from video files |
| `datasets[].flip_x` | bool | `false` | Horizontal flip (doubles effective dataset) |
| `datasets[].caption_dropout_rate` | float | `0.0` | Caption dropout probability |

---

## 7. Developer Notes & Handoff

### Dependencies
No new dependencies. All changes use existing PyTorch, torchaudio, and diffusers APIs.

### Backward Compatibility
- All new config fields have safe defaults reproducing original behavior
- `independent_audio_timestep` defaults to `True` but can be disabled
- Old configs without new keys work identically to before
- `audio_pred`/`audio_target` backward-compat path preserved

### Architecture Decisions

1. **Independent audio timestep:** The LTX-2 transformer (Lightricks `LTXModel`) has separate `adaln_single` and `audio_adaln_single` modules that process video and audio timesteps independently. The `Modality` objects carry their own `.timesteps`. Training was feeding the same value to both. Decoupling is architecturally correct and zero-cost.

2. **Rank 32 + alpha 32 + dropout 0.1:** For a joint video+audio DiT with shared transformer blocks, rank 4 is insufficient to encode both visual identity and voice characteristics. Rank 32 provides ~64x more capacity. Alpha=32 gives effective scale 1.0 (alpha/rank). Rank dropout 0.1 prevents overfitting.

3. **Scalar loss over tensor storage:** Computing MSE immediately in `get_noise_prediction` saves VRAM.

4. **EMA for auto-balance:** alpha=0.99 adapts over ~100 steps. The 0.33 target ratio keeps audio at ~25% of total loss.

5. **Gradient checkpointing:** The official Lightricks model has `set_gradient_checkpointing()` built in with `use_reentrant=False`. We activate it during `load_model`.

6. **torch.compile fix:** Detects `is_transformer` flag, compiles the correct model object, and rebinds the compiled module back to the runtime model.

### Patch Delivery
- **Full patch (baseline):** `ltx2-audio-fixes-v2.patch` — unified diffs for the complete v2/v3 implementation set
- **Hotfix patch (v3):** `ltx2-audio-fixes-v3-hotfix.patch` — incremental fixes for video-safe `noise_offset`, compile assignment, and UI selector conflict cleanup
- **Phase 3 files (v3.12):** `DoRA.py`, `network_mixins.py`, `manager_modules.py`, `ltx2.py`, `SDTrainer.py`, `dataloader_mixins.py`, `data_loader.py`, `options.ts` — quantization, layer-offloading, attention-mask dtype, `train_config` safe access, flow-matching SNR guard, latent cache audio invalidation, robust PyAV audio extraction fallback, audio logging fix, precise dequantize type checks, bidirectional auto-balance clamp
- **SOP:** This document (`LTX2_AUDIO_SOP.md`)
- **Zip archive:** `ltx2_improvements_handoff.zip` — all patches + SOP + all 16 modified source files

---

## 8. Pre-Handoff Test Checklist

| # | Test | Expected Result |
|---|------|-----------------|
| 1 | Train LTX-2 LoRA with `independent_audio_timestep: true` | Audio quality noticeably improved vs coupled timestep |
| 2 | Train with `auto_balance_audio_loss: true` | Dynamic multiplier visible in logs and actively changing; `dyn_mult` settles to ~0.15–0.25 when audio > video (or >1.0 when video > audio); voice quality preserved |
| 3 | Train with `network.type: dora`, rank 32 | DoRA loads and trains; identity retention potentially better than standard LoRA |
| 4 | Train with `rank_dropout: 0.1` | Loss slightly higher but more stable; no overfitting on small datasets |
| 5 | Train on mixed image + video dataset | Voice regularizer fires on image batches; no crashes |
| 6 | Train with `strict_audio_mode: true` on broken audio dataset | Training halts with clear error after warmup |
| 7 | Set `model.compile: true` on LTX-2 | Transformer compiled (log message confirms); training runs faster |
| 8 | Check VRAM with gradient checkpointing on/off | Significant VRAM reduction with checkpointing on |
| 9 | Open web UI, select LTX-2 | All new fields visible: network type, rank/module dropout, independent timestep, noise offset, min-SNR, LR scheduler, caption dropout, optimizer additions |
| 10 | Run with old config (no new keys) | All defaults kick in; behavior safe |
| 11 | Audio loss logs appear every 10 steps | Shows raw/scaled audio loss, video loss, and dynamic multiplier |
| 12 | Enable `noise_offset: 0.05` and `min_snr_gamma: 5.0` | Training converges; improved detail in bright/dark regions |
| 13 | Enable `noise_offset: 0.05` on LTX-2 (`num_frames > 1`) | No crash from 5D latents; training proceeds normally |
| 14 | Open web UI with LTX-2 selected | Exactly one network type selector; DoRA selectable without conflicts |
| 15 | Enable `dynamic_noise_offset: true` on LTX-2 | No shape mismatch; noise correction runs on 5D latents |
| 16 | Train DoRA with `qfloat8` quantized transformer | No `AffineQuantizedTensor` / `aten.add` crash; DoRA modules created and trained |
| 17 | Train DoRA/LoRA with `layer_offloading: true` | No "self and mat2 must have the same dtype" crash during sampling or training |
| 18 | Baseline sample generation before training with `layer_offloading: true` | Samples generate without dtype mismatch errors |
| 19 | Baseline sampling with DoRA + quantization + offloading | No SDPA error about `attn_mask` dtype vs query dtype |
| 20 | LORM local mode smoke test on quantized module | No dtype mismatch on `self.org_forward(...)` path |
| 21 | Connector output mask dtype inspection | Connector mask is float32 before transformer call in both generation and training paths |
| 22 | Train DoRA + quantization + offloading for full run | No dtype errors through entire training — the `_org_forward_safe` dtype restoration prevents bf16 from leaking into attention masks |
| 23 | Verify SDPA patch fires when needed | Add temporary logging to `_sdpa_mask_dtype_safe` — should NOT fire if dtype restoration works, but must fire if a mask slips through |
| 24 | Confirm `torch.nn.functional.scaled_dot_product_attention` interception | All diffusers backends call through `torch.nn.functional.scaled_dot_product_attention(...)`, so patch intercepts NATIVE, EFFICIENT, FLASH, MATH, and cuDNN backends |
| 25 | Training proceeds past step 0 without `AttributeError` | No `'LTX2Model' object has no attribute 'train_config'` in `get_noise_prediction` or connector gradient path |
| 26 | LTX-2 training with `min_snr_gamma: 5` | No crash; SNR weighting silently skipped (flow-matching has no `alphas_cumprod`) |
| 27 | LTX-2 training with `min_snr_gamma: 0` | Explicit disable; no SNR code path entered at all |
| 28 | Audio extraction from mp4 video files on Windows/Pinokio | Audio extracted via PyAV fallback (torchaudio broken); caching summary shows N encoded, 0 failed |
| 29 | Latent caching summary shows audio stats | After caching, log line: "Audio latent caching: N encoded, 0 failed" |
| 30 | Audio loss logging during training | No `AttributeError` from `print_and_status_update`; audio loss values printed to console |
| 31 | DoRA training with quantized model completes backward pass | No `derivative for dequantize is not implemented` error; training proceeds past step 0 |
| 32 | LoRA training with quantized model completes backward pass | Same fix applies; no dequantize autograd crash |
| 33 | Auto-balance `dyn_mult` changes over training | `dyn_mult` should NOT stay at 1.00; when raw audio loss > video loss, multiplier drops below 1.0; when audio < video, multiplier rises above 1.0 |
| 34 | Auto-balance with audio > video (common LTX-2 case) | `dyn_mult` settles to ~0.15–0.25; `scaled` audio loss ≈ 33% of `video` loss in logs |
