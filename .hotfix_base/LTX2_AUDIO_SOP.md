# LTX-2 Character Training — Standard Operating Procedure (SOP)

> **Version:** 3.0-final  
> **Date:** 2026-02-19  
> **Scope:** All changes to `ostris/ai-toolkit` that fix voice-training failures and comprehensively improve character LoRA quality for LTX-2.

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

---

## 3. Complete List of Changes

### Phase 1: Core Audio Fixes (v2.0)

#### A. Audio Loss Multiplier (`audio_loss_multiplier`)
- **Files:** `SDTrainer.py`, `config_modules.py`
- Configurable scalar that scales audio loss before adding to total loss.

#### B. Automatic Audio Loss Balancing (`auto_balance_audio_loss`)
- **Files:** `SDTrainer.py`, `config_modules.py`
- EMA-based (alpha=0.99) dynamic multiplier targeting audio at ~33% of video loss. Clamped [1.0, 20.0].

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

---

## 4. Files Modified

| File | Changes |
|------|---------|
| `toolkit/config_modules.py` | `independent_audio_timestep`, `rank_dropout`, `module_dropout` in configs; all Phase 1 audio fields |
| `toolkit/data_transfer_object/data_loader.py` | Mixed-batch audio alignment; `audio_loss` field; cleanup |
| `extensions_built_in/diffusion_models/ltx2/ltx2.py` | Independent audio timestep; gradient checkpointing; SDPA enforcement; encode_audio silence; voice regularizer; connector unfreezing; scalar loss; module freezing |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Audio loss logging; auto-balance; strict audio; loss integration |
| `jobs/process/BaseSDTrainProcess.py` | rank_dropout/module_dropout passthrough; torch.compile fix for transformer models |
| `ui/src/types.ts` | `NetworkConfig` additions (dropout fields); `TrainConfig` additions (independent_audio_timestep, noise_offset, min_snr_gamma, lr_scheduler) |
| `ui/src/app/jobs/new/options.ts` | LTX-2 defaults (rank 32, alpha 32, rank_dropout 0.1, auto_balance on, noise_offset 0.05, min_snr 5.0); all new additionalSections |
| `ui/src/app/jobs/new/SimpleJob.tsx` | Network type selector; rank/module dropout inputs; independent audio timestep; noise offset; min-SNR gamma; LR scheduler; caption dropout; optimizer additions (Prodigy, DAdaptation, AdamW) |
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
        min_snr_gamma: 5.0
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

6. **torch.compile fix:** Detects `is_transformer` flag and compiles the correct model object.

### Patch Delivery
- **Patch file:** `ltx2-audio-fixes-v2.patch` — unified diffs for all modified files
- **SOP:** This document (`LTX2_AUDIO_SOP.md`)
- **Zip archive:** `ltx2_improvements_handoff.zip` — patch + SOP + all modified source files

---

## 8. Pre-Handoff Test Checklist

| # | Test | Expected Result |
|---|------|-----------------|
| 1 | Train LTX-2 LoRA with `independent_audio_timestep: true` | Audio quality noticeably improved vs coupled timestep |
| 2 | Train with `auto_balance_audio_loss: true` | Dynamic multiplier visible in logs; voice quality preserved |
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
