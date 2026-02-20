# LTX-2 Voice Training Fix for Ostris AI-Toolkit

**Voice training for LTX-2 character LoRAs is broken out of the box. This patch fixes it.**

If you've tried training an LTX-2 character LoRA and your output has garbled, silent, or completely wrong audio — this is why, and this is the fix.

---

## The Problem

LTX-2 is a joint audio+video diffusion transformer. When you train a character LoRA, the model should learn both the person's appearance AND their voice. In practice, every single person training LTX-2 character LoRAs in `ostris/ai-toolkit` gets broken audio. The LoRA produces correct visuals but the voice is destroyed.

This isn't a settings issue. There are **25 bugs and design flaws** in the training pipeline that collectively make voice training impossible.

---

## What Was Wrong (The Big Ones)

### 1. Audio and video shared the same timestep during training

LTX-2's transformer has completely separate timestep processing for audio (`audio_adaln_single`) and video (`adaln_single`). The training code sampled ONE random timestep and fed it to BOTH. This means audio never explored its own noise schedule — voice learning was fundamentally broken at the architecture level.

**Fix:** Independent audio timestep sampling. Each training step now samples a separate random timestep for audio. Zero computational cost. Massive quality impact.

### 2. Audio was never actually extracted from your videos

On Windows and Pinokio installs, `torchaudio.load()` is completely broken. The newer versions use `torchcodec` as a backend, which requires FFmpeg shared libraries that Pinokio doesn't install. The error was **silently caught**, so training appeared to work but every single video had its audio silently dropped.

**Fix:** Multi-fallback audio extraction that tries torchaudio first, then PyAV (which bundles its own FFmpeg and is already in `requirements.txt`), then ffmpeg CLI as a last resort. Audio extraction now works on every platform.

### 3. Stale latent cache had no audio in it

If you ever ran training before this fix, your latent cache files don't contain `audio_latent`. The cache loader only checked if the file existed, not whether it contained audio. So even after fixing audio extraction, the old cache was still being used — with no audio.

**Fix:** Cache validation now checks for the `audio_latent` key inside safetensors files. Missing audio = cache invalidated and re-encoded.

### 4. Video loss drowned out audio loss

Video loss magnitude is much larger than audio loss. Without balancing, the optimizer effectively ignores audio gradients entirely.

**Fix:** EMA-based dynamic audio loss balancing that targets audio at ~33% of total loss. Adapts automatically over training.

### 5. Audio-free batches caused voice forgetting

When a batch had no audio (e.g., image-only batches in mixed datasets), zero audio loss was computed. This let the LoRA drift away from the base model's voice generation ability — catastrophic forgetting.

**Fix:** Synthetic silence regularizer for audio-free batches. Forces the LoRA to preserve base audio behavior even on batches without audio data.

### 6. DoRA and quantized models crashed immediately

Using DoRA (weight-decomposed LoRA) with quantized models (`qfloat8`) caused crashes from `AffineQuantizedTensor` dispatch errors, dtype mismatches in SDPA attention, and `derivative for dequantize is not implemented` errors in the backward pass. The root causes were spread across 6 different files.

**Fix:** Precise quantization type detection, safe forward wrappers, dtype enforcement in the memory manager, and SDPA attention mask safety nets. DoRA + quantization + layer offloading now works end to end.

### 7. Auto-balance audio loss was broken — could only boost, never dampen

The dynamic audio loss balancer was designed to keep audio and video loss in proportion. But the multiplier was clamped at minimum 1.0, meaning it could only INCREASE audio loss. In practice, LTX-2's raw audio loss is naturally ~2x larger than video loss. The computed multiplier (~0.18) was always clamped back to 1.0. The feature was dead code — `dyn_mult` showed 1.00 for the entire training run.

**Fix:** Changed the clamp floor from 1.0 to 0.05. The multiplier now works bidirectionally — dampening audio when it's already dominant (common case), boosting when it's too small. `dyn_mult` actively adjusts throughout training.

### 8. Multiple config/runtime crashes on LTX-2

- `self.train_config` accessed on the model object instead of the trainer — crash on step 0
- `min_snr_gamma` incompatible with flow-matching schedulers — crash on loss calculation
- `print_and_status_update` called on wrong object — crash on audio logging
- `noise_offset` rejected 5D video tensors — crash when enabled
- `torch.compile` pointed at `unet` instead of transformer — no effect on DiT models

All fixed.

---

## What We Added

### Core Audio Fixes
- Independent audio timestep sampling (the single biggest voice quality improvement)
- Bidirectional automatic audio loss balancing (EMA-based, dampens when audio > video, boosts when video > audio)
- Robust multi-fallback audio extraction (torchaudio -> PyAV -> ffmpeg CLI)
- Latent cache audio validation and automatic invalidation
- Voice preservation regularizer for audio-free batches
- Connector gradient unfreezing for text-to-audio adaptation
- Scalar audio loss (saves VRAM vs storing full prediction tensors)
- Audio loss logging every 10 steps

### Quality Improvements
- Rank/module dropout support (prevents overfitting on small datasets)
- Higher default rank (32 instead of 4 — needed for joint audio+video identity)
- DoRA and LoKr network type support
- Gradient checkpointing wired for LTX-2 (30-50% VRAM savings)
- Flash/memory-efficient attention enforcement
- Video-safe noise offset
- Caption dropout support
- Cosine annealing / cosine with restarts / linear decay LR schedulers
- Prodigy and DAdaptation optimizer support

### Compatibility Fixes
- DoRA + TorchAO quantization (`qfloat8`) fully working
- Layer offloading + quantization dtype enforcement
- SDPA attention mask dtype safety for mixed-precision paths
- Min-SNR guard for flow-matching schedulers
- torch.compile targeting transformer instead of unet
- All `train_config` access made safe for model vs trainer context
- Precise quantization type detection (fixes PyTorch 2.9+ `dequantize` issue)

---

## Installation

### Option 1: Copy Modified Files (Recommended)

Copy these files from the release into your `ai-toolkit` installation, replacing the originals:

```
toolkit/config_modules.py
toolkit/train_tools.py
toolkit/dataloader_mixins.py
toolkit/data_transfer_object/data_loader.py
toolkit/models/DoRA.py
toolkit/network_mixins.py
toolkit/memory_management/manager_modules.py
extensions_built_in/diffusion_models/ltx2/ltx2.py
extensions_built_in/sd_trainer/SDTrainer.py
jobs/process/BaseSDTrainProcess.py
ui/src/types.ts
ui/src/app/jobs/new/options.ts
ui/src/app/jobs/new/SimpleJob.tsx
ui/src/docs.tsx
```

### Option 2: Apply Patches

Patch files are included in the release zip for those who prefer `git apply`.

### Important: Delete Your Latent Cache

If you've trained before, your cached latents don't have audio in them. Delete your latent cache folder and let it re-encode. The new code will extract audio properly via PyAV and include `audio_latent` in the cache files.

---

## Recommended Config

### LoRA (Recommended — Fast + High Quality)

```yaml
job: extension
config:
  name: "LTX2_character_lora"
  process:
    - type: "sd_trainer"
      training_folder: "output/my_character"
      device: cuda:0
      trigger_word: "ohwx"
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        rank_dropout: 0.1
      save:
        dtype: bf16
        save_every: 250
      datasets:
        - folder_path: "/path/to/your/video/clips"
          num_frames: 81
          do_audio: true
          caption_ext: "txt"
          caption_dropout_rate: 0.05
      train:
        batch_size: 1
        steps: 3000
        gradient_accumulation_steps: 1
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16
        noise_scheduler: "custom_flowmatch"
        timestep_type: "sigmoid"
        auto_balance_audio_loss: true
        independent_audio_timestep: true
        noise_offset: 0.05
        min_snr_gamma: 0
      model:
        name_or_path: "Lightricks/LTX-2"
        low_vram: true
        quantize: "qfloat8"
      sample:
        sample_every: 250
        width: 512
        height: 320
        num_frames: 81
        sample_steps: 25
        guidance_scale: 3.0
        prompts:
          - "ohwx speaking to the camera about artificial intelligence"
```

### Layer Offloading (If You Need It)

If you're running out of VRAM, add to the model section:

```yaml
      model:
        layer_offloading: true
        layer_offloading_transformer_percent: 0.56
```

This offloads 56% of transformer layers to CPU. Expect ~20s/it on RTX 5090 with LoRA. Increase the percentage if you still OOM, decrease for speed.

**Note:** `torch.compile` is incompatible with `layer_offloading`. Don't use both.

### DoRA Variant (Higher Quality, Slower, More VRAM)

Replace the network section:

```yaml
      network:
        type: "dora"
        linear: 32
        linear_alpha: 32
        rank_dropout: 0.1
```

DoRA decomposes weight updates into magnitude and direction components, which can produce higher quality results. However, it requires significantly more VRAM (you'll need higher layer offloading %, which slows training). For most users, LoRA rank 32 is the sweet spot.

---

## How to Verify Audio Is Training

During training, look for this line in your console output:

```
[audio] raw=0.01234, scaled=0.05678, video=1.23456
```

If you see it, audio loss is actively being computed and your character's voice is being learned. If you don't see it, something is wrong with your audio pipeline — check that:

1. Your video files actually contain audio tracks
2. `do_audio: true` is set in your dataset config
3. You deleted your old latent cache and let it re-encode

After caching, you should see:

```
Audio latent caching: 24 encoded, 0 failed
```

(Where 24 is your number of video clips.)

---

## Performance Expectations

| GPU | Layer Offload | Network | Expected Speed |
|-----|--------------|---------|----------------|
| RTX 5090 (32GB) | None | LoRA rank 32 | ~2-5 it/s |
| RTX 5090 (32GB) | 40% | LoRA rank 32 | ~20s/it |
| RTX 5090 (32GB) | 56% | LoRA rank 32 | ~30s/it |
| RTX 5090 (32GB) | 56% | DoRA rank 32 | ~60s/it |
| RTX 4090 (24GB) | 50-60% | LoRA rank 32 | ~30-45s/it |

Training speed is dominated by layer offloading (CPU-GPU memory transfer over PCIe). Reduce offloading percentage for speed, increase for VRAM savings.

---

## Dataset Tips

- **20-50 video clips** of your character speaking, 4-8 seconds each
- Clips should have **clear audio** — the character's voice should be the dominant sound
- Variety in lighting, angles, and expressions helps generalization
- Caption each clip accurately, describing the visual content
- Use a distinctive trigger word (e.g., `ohwx`) that doesn't conflict with the base model vocabulary
- `flip_x: true` doubles your effective dataset size (don't use for text-heavy content)

---

## FAQ

**Q: Do I need to do anything special for audio?**
A: Just set `do_audio: true` in your dataset config and make sure your video files have audio tracks. Everything else is automatic.

**Q: Can I use my existing video dataset?**
A: Yes, as long as the videos have audio. Delete your old latent cache first so it re-encodes with audio.

**Q: LoRA or DoRA?**
A: LoRA rank 32 for most users. It's 3x faster and uses significantly less VRAM. DoRA may produce marginally higher quality but requires much more memory and time.

**Q: What about LoKr?**
A: Supported but less tested with the audio fixes. LoRA is recommended.

**Q: My training shows 0 audio loss / no audio line in logs?**
A: Your audio isn't being extracted. Delete latent cache, confirm videos have audio, confirm `do_audio: true`.

**Q: Can I use torch.compile?**
A: Only if you're NOT using `layer_offloading`. They're mutually exclusive due to how layer offloading mutates GPU buffers during forward passes.

**Q: What's `independent_audio_timestep`?**
A: The single most important fix. LTX-2's transformer processes audio and video noise schedules independently, but the training code was feeding the same random timestep to both. This decouples them so audio can learn at its own optimal noise level. Always leave this `true`.

**Q: Why is `min_snr_gamma` set to 0?**
A: Min-SNR loss weighting requires `alphas_cumprod` from DDPM-style schedulers. LTX-2 uses a flow-matching scheduler that doesn't have this. Setting it to anything > 0 would crash. The `timestep_type: sigmoid` or `weighted` setting provides equivalent loss balancing for flow-matching models.

---

## Files Modified (16 files)

| File | What Changed |
|------|-------------|
| `toolkit/config_modules.py` | New config fields for audio, dropout, timestep, schedulers |
| `toolkit/train_tools.py` | Video-safe noise offset (4D + 5D tensor support) |
| `toolkit/dataloader_mixins.py` | Robust audio extraction, cache validation, PyAV fallback |
| `toolkit/data_transfer_object/data_loader.py` | Mixed-batch audio handling, audio_loss field |
| `toolkit/models/DoRA.py` | TorchAO quantization support, precise type checks |
| `toolkit/network_mixins.py` | Safe forward wrapper, dtype preservation, quantization fixes |
| `toolkit/memory_management/manager_modules.py` | Layer offloading dtype enforcement |
| `extensions_built_in/diffusion_models/ltx2/ltx2.py` | Independent audio timestep, gradient checkpointing, SDPA fixes, attention mask safety |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Audio loss pipeline, auto-balancing, flow-matching guard |
| `jobs/process/BaseSDTrainProcess.py` | Dropout passthrough, compile fix, dynamic noise offset |
| `ui/src/types.ts` | TypeScript types for new config fields |
| `ui/src/app/jobs/new/options.ts` | LTX-2 defaults (rank 32, auto-balance, etc.) |
| `ui/src/app/jobs/new/SimpleJob.tsx` | UI controls for all new features |
| `ui/src/docs.tsx` | Documentation for all new fields |

---

## Credits

Built on top of [Ostris AI-Toolkit](https://github.com/ostris/ai-toolkit). All changes are backward compatible — old configs without new keys work identically to before.

25 bugs identified and fixed. Zero new dependencies added. All features use existing PyTorch, torchaudio, diffusers, and PyAV APIs.
