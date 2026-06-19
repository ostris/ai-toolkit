/**
 * Hardware-aware preset matrix.
 *
 * Each entry is a list of paths (matching the format used by
 * `setNestedValue`) plus the value to assign. Applying a preset is a
 * loop over these paths — no deep merge needed. This matches how
 * `modelArch.defaults` already works.
 *
 * VRAM numbers are **approximate peak** at 1024-class image resolution
 * (or default-frames for video). Reality varies by dataset, sampler
 * settings, etc. Give yourself ~15% headroom.
 */

export type VramTier = 'memory' | 'balanced' | 'quality';

export interface ConfigPreset {
  /** Stable id; used as React key. */
  id: string;
  /** Short label shown on the card. */
  label: string;
  /** One-line description shown under the label. */
  description: string;
  /** Approximate peak VRAM in GB. Used to mark the "Recommended" card. */
  approxVramGB: number;
  /** Which preset family this belongs to (only one balanced per arch, etc.) */
  tier: VramTier;
  /** Path → value overrides to apply via setNestedValue. */
  overrides: Record<string, unknown>;
}

/**
 * Indexed by `modelArch.name` from `options.ts`.
 *
 * Keep presets DECLARATIVE — no logic; just config keys.
 */
export const CONFIG_PRESETS: Record<string, ConfigPreset[]> = {
  // ───────────────────────────────────────────────────────────────────
  // FLUX.1
  // ───────────────────────────────────────────────────────────────────
  flux: [
    {
      id: 'flux-memory',
      label: 'Memory',
      description: '12–16 GB. Aggressive offloading, slower. Use when VRAM is tight.',
      approxVramGB: 14,
      tier: 'memory',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': true,
        'config.process[0].model.layer_offloading': true,
        'config.process[0].model.layer_offloading_transformer_percent': 0.5,
        'config.process[0].model.layer_offloading_text_encoder_percent': 1.0,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.gradient_accumulation': 4,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768],
      },
    },
    {
      id: 'flux-balanced',
      label: 'Balanced',
      description: '16–22 GB. Recommended for most 24 GB cards.',
      approxVramGB: 20,
      tier: 'balanced',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': false,
        'config.process[0].model.layer_offloading': false,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.gradient_accumulation': 1,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768, 1024],
      },
    },
    {
      id: 'flux-quality',
      label: 'Quality',
      description: '32+ GB. No quantization, full precision base model.',
      approxVramGB: 34,
      tier: 'quality',
      overrides: {
        'config.process[0].model.quantize': false,
        'config.process[0].model.quantize_te': false,
        'config.process[0].model.low_vram': false,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.gradient_accumulation': 1,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768, 1024],
      },
    },
  ],

  // ───────────────────────────────────────────────────────────────────
  // Qwen-Image (heavy text encoder, benefits hugely from uint3 + ARA)
  // ───────────────────────────────────────────────────────────────────
  qwen_image: [
    {
      id: 'qwen-memory',
      label: 'Memory',
      description: '12–18 GB. uint3 + layer offload. Slower but fits on consumer cards.',
      approxVramGB: 15,
      tier: 'memory',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.qtype':
          'uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors',
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.qtype_te': 'qfloat8',
        'config.process[0].model.low_vram': true,
        'config.process[0].model.layer_offloading': true,
        'config.process[0].model.layer_offloading_transformer_percent': 0.5,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768],
      },
    },
    {
      id: 'qwen-balanced',
      label: 'Balanced',
      description: '22–28 GB. Matches the official 24 GB example. uint3 base, FP8 TE, text-embedding cache.',
      approxVramGB: 24,
      tier: 'balanced',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.qtype':
          'uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors',
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.qtype_te': 'qfloat8',
        'config.process[0].model.low_vram': true,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768, 1024],
      },
    },
    {
      id: 'qwen-quality',
      label: 'Quality',
      description: '40+ GB. bf16 base + FP8 TE. Best quality if VRAM allows.',
      approxVramGB: 42,
      tier: 'quality',
      overrides: {
        'config.process[0].model.quantize': false,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.qtype_te': 'qfloat8',
        'config.process[0].model.low_vram': false,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768, 1024],
      },
    },
  ],

  // ───────────────────────────────────────────────────────────────────
  // SDXL (lightweight compared to FLUX/Qwen; can fit much more comfortably)
  // ───────────────────────────────────────────────────────────────────
  sdxl: [
    {
      id: 'sdxl-memory',
      label: 'Memory',
      description: '6–8 GB. Smallest config; for 8 GB cards.',
      approxVramGB: 7,
      tier: 'memory',
      overrides: {
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.gradient_accumulation': 4,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [768, 1024],
      },
    },
    {
      id: 'sdxl-balanced',
      label: 'Balanced',
      description: '10–16 GB. Recommended default for SDXL.',
      approxVramGB: 12,
      tier: 'balanced',
      overrides: {
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.gradient_accumulation': 1,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [768, 1024],
      },
    },
    {
      id: 'sdxl-quality',
      label: 'Quality',
      description: '18+ GB. Larger batch, no checkpointing.',
      approxVramGB: 20,
      tier: 'quality',
      overrides: {
        'config.process[0].train.gradient_checkpointing': false,
        'config.process[0].train.gradient_accumulation': 1,
        'config.process[0].train.batch_size': 2,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [768, 1024],
      },
    },
  ],

  // ───────────────────────────────────────────────────────────────────
  // HiDream
  // ───────────────────────────────────────────────────────────────────
  hidream: [
    {
      id: 'hidream-memory',
      label: 'Memory',
      description: '20–28 GB. quantize + layer offload + GC.',
      approxVramGB: 24,
      tier: 'memory',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': true,
        'config.process[0].model.layer_offloading': true,
        'config.process[0].model.layer_offloading_transformer_percent': 0.5,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.gradient_accumulation': 1,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768],
      },
    },
    {
      id: 'hidream-balanced',
      label: 'Balanced',
      description: '32–40 GB. quantize on, no offload.',
      approxVramGB: 36,
      tier: 'balanced',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': false,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768, 1024],
      },
    },
    {
      id: 'hidream-quality',
      label: 'Quality',
      description: '48+ GB. No quantization. Matches train_lora_hidream_48.yaml.',
      approxVramGB: 50,
      tier: 'quality',
      overrides: {
        'config.process[0].model.quantize': false,
        'config.process[0].model.quantize_te': false,
        'config.process[0].model.low_vram': false,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].resolution': [512, 768, 1024],
      },
    },
  ],

  // ───────────────────────────────────────────────────────────────────
  // Wan 2.1 (T2V 14B). Video model; num_frames + resolution dominate VRAM.
  // ───────────────────────────────────────────────────────────────────
  wan21: [
    {
      id: 'wan21-memory',
      label: 'Memory',
      description: '16–20 GB. 24 frames, 480p. Aggressive offload.',
      approxVramGB: 18,
      tier: 'memory',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': true,
        'config.process[0].model.layer_offloading': true,
        'config.process[0].model.layer_offloading_transformer_percent': 0.5,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 24,
        'config.process[0].datasets[0].resolution': [480],
      },
    },
    {
      id: 'wan21-balanced',
      label: 'Balanced',
      description: '22–30 GB. 40 frames @ 480p. Matches the official 24 GB example.',
      approxVramGB: 24,
      tier: 'balanced',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 40,
        'config.process[0].datasets[0].resolution': [632],
      },
    },
    {
      id: 'wan21-quality',
      label: 'Quality',
      description: '40+ GB. 81 frames @ 720p. No quantization.',
      approxVramGB: 44,
      tier: 'quality',
      overrides: {
        'config.process[0].model.quantize': false,
        'config.process[0].model.quantize_te': false,
        'config.process[0].model.low_vram': false,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 81,
        'config.process[0].datasets[0].resolution': [720],
      },
    },
  ],

  // ───────────────────────────────────────────────────────────────────
  // Wan 2.2 14B (I2V). uint4 + ARA is the big VRAM win at 24 GB.
  // ───────────────────────────────────────────────────────────────────
  wan22_14b_i2v: [
    {
      id: 'wan22-14b-memory',
      label: 'Memory',
      description: '18–22 GB. uint4 ARA + layer offload, short clips.',
      approxVramGB: 20,
      tier: 'memory',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.qtype':
          'uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors',
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.qtype_te': 'qfloat8',
        'config.process[0].model.low_vram': true,
        'config.process[0].model.layer_offloading': true,
        'config.process[0].model.layer_offloading_transformer_percent': 0.5,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 1,
        'config.process[0].datasets[0].resolution': [512, 768],
      },
    },
    {
      id: 'wan22-14b-balanced',
      label: 'Balanced',
      description: '24–32 GB. uint4 ARA, FP8 TE. Matches the official 24 GB example.',
      approxVramGB: 26,
      tier: 'balanced',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.qtype':
          'uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors',
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.qtype_te': 'qfloat8',
        'config.process[0].model.low_vram': true,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 1,
        'config.process[0].datasets[0].resolution': [512, 768, 1024],
      },
    },
    {
      id: 'wan22-14b-quality',
      label: 'Quality',
      description: '40+ GB. bf16 base. Highest fidelity.',
      approxVramGB: 44,
      tier: 'quality',
      overrides: {
        'config.process[0].model.quantize': false,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.qtype_te': 'qfloat8',
        'config.process[0].model.low_vram': false,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 1,
        'config.process[0].datasets[0].resolution': [512, 768, 1024],
      },
    },
  ],

  // ───────────────────────────────────────────────────────────────────
  // Wan 2.2 5B (TI2V). Smaller — all tiers fit comfortably.
  // ───────────────────────────────────────────────────────────────────
  wan22_5b: [
    {
      id: 'wan22-5b-memory',
      label: 'Memory',
      description: '12–16 GB. Quantize + offload + short clips.',
      approxVramGB: 14,
      tier: 'memory',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': true,
        'config.process[0].model.layer_offloading': true,
        'config.process[0].model.layer_offloading_transformer_percent': 0.5,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 33,
        'config.process[0].datasets[0].resolution': [480],
      },
    },
    {
      id: 'wan22-5b-balanced',
      label: 'Balanced',
      description: '18–24 GB. Quantize on, no offload, 121 frames.',
      approxVramGB: 20,
      tier: 'balanced',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': false,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 121,
        'config.process[0].datasets[0].resolution': [480, 632],
      },
    },
    {
      id: 'wan22-5b-quality',
      label: 'Quality',
      description: '28+ GB. bf16 base, longer clips.',
      approxVramGB: 30,
      tier: 'quality',
      overrides: {
        'config.process[0].model.quantize': false,
        'config.process[0].model.quantize_te': false,
        'config.process[0].model.low_vram': false,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 121,
        'config.process[0].datasets[0].resolution': [480, 632, 720],
      },
    },
  ],

  // ───────────────────────────────────────────────────────────────────
  // LTX-2 (video)
  // ───────────────────────────────────────────────────────────────────
  ltx2: [
    {
      id: 'ltx2-memory',
      label: 'Memory',
      description: '16–22 GB. Quantize + offload, fewer frames.',
      approxVramGB: 18,
      tier: 'memory',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': true,
        'config.process[0].model.layer_offloading': true,
        'config.process[0].model.layer_offloading_transformer_percent': 0.5,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 49,
        'config.process[0].datasets[0].resolution': [512, 704],
      },
    },
    {
      id: 'ltx2-balanced',
      label: 'Balanced',
      description: '24–32 GB. Quantize on, full 121-frame clips at 768p.',
      approxVramGB: 26,
      tier: 'balanced',
      overrides: {
        'config.process[0].model.quantize': true,
        'config.process[0].model.quantize_te': true,
        'config.process[0].model.low_vram': true,
        'config.process[0].train.cache_text_embeddings': true,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 121,
        'config.process[0].datasets[0].resolution': [512, 704, 768],
      },
    },
    {
      id: 'ltx2-quality',
      label: 'Quality',
      description: '40+ GB. bf16 base.',
      approxVramGB: 42,
      tier: 'quality',
      overrides: {
        'config.process[0].model.quantize': false,
        'config.process[0].model.quantize_te': false,
        'config.process[0].model.low_vram': false,
        'config.process[0].train.gradient_checkpointing': true,
        'config.process[0].train.batch_size': 1,
        'config.process[0].datasets[0].cache_latents_to_disk': true,
        'config.process[0].datasets[0].num_frames': 121,
        'config.process[0].datasets[0].resolution': [768, 1024],
      },
    },
  ],
};

// Wan 2.1 1.3B is much lighter than 14B - fits comfortably even on small cards.
// Sized from train_lora_wan21_1b_24gb.yaml: TE quant only, no base quant.
CONFIG_PRESETS['wan21:1b'] = [
  {
    id: 'wan21-1b-memory',
    label: 'Memory',
    description: '8–12 GB. TE quantized, shorter clips. For 12 GB cards.',
    approxVramGB: 10,
    tier: 'memory',
    overrides: {
      'config.process[0].model.quantize': false,
      'config.process[0].model.quantize_te': true,
      'config.process[0].model.qtype_te': 'qfloat8',
      'config.process[0].model.low_vram': true,
      'config.process[0].train.gradient_checkpointing': true,
      'config.process[0].train.batch_size': 1,
      'config.process[0].datasets[0].cache_latents_to_disk': true,
      'config.process[0].datasets[0].num_frames': 24,
      'config.process[0].datasets[0].resolution': [480],
    },
  },
  {
    id: 'wan21-1b-balanced',
    label: 'Balanced',
    description: '14–20 GB. bf16 base + FP8 TE. Matches the official 24 GB example.',
    approxVramGB: 16,
    tier: 'balanced',
    overrides: {
      'config.process[0].model.quantize': false,
      'config.process[0].model.quantize_te': true,
      'config.process[0].model.qtype_te': 'qfloat8',
      'config.process[0].model.low_vram': false,
      'config.process[0].train.gradient_checkpointing': true,
      'config.process[0].train.batch_size': 1,
      'config.process[0].datasets[0].cache_latents_to_disk': true,
      'config.process[0].datasets[0].num_frames': 40,
      'config.process[0].datasets[0].resolution': [632],
    },
  },
  {
    id: 'wan21-1b-quality',
    label: 'Quality',
    description: '22+ GB. Full precision, longer clips, higher resolution.',
    approxVramGB: 24,
    tier: 'quality',
    overrides: {
      'config.process[0].model.quantize': false,
      'config.process[0].model.quantize_te': false,
      'config.process[0].model.low_vram': false,
      'config.process[0].train.gradient_checkpointing': true,
      'config.process[0].train.batch_size': 1,
      'config.process[0].datasets[0].cache_latents_to_disk': true,
      'config.process[0].datasets[0].num_frames': 81,
      'config.process[0].datasets[0].resolution': [720],
    },
  },
];

// ─────────────────────────────────────────────────────────────────────
// Aliases for arch names that share VRAM characteristics with an existing
// preset family. The UI registers many more arch names than the example
// YAMLs target — these aliases prevent the picker from silently vanishing
// when a closely-related arch is selected.
// ─────────────────────────────────────────────────────────────────────
CONFIG_PRESETS['wan21:14b'] = CONFIG_PRESETS.wan21;
CONFIG_PRESETS['wan21_i2v:14b'] = CONFIG_PRESETS.wan21;
CONFIG_PRESETS['wan21_i2v:14b480p'] = CONFIG_PRESETS.wan21;
CONFIG_PRESETS['wan22_14b:t2v'] = CONFIG_PRESETS.wan22_14b_i2v;
CONFIG_PRESETS['ltx2.3'] = CONFIG_PRESETS.ltx2;
// Qwen-Image:2512 is the same model family with an updated checkpoint.
CONFIG_PRESETS['qwen_image:2512'] = CONFIG_PRESETS.qwen_image;

/**
 * Given a detected VRAM amount in GB, choose which preset to mark
 * "Recommended". The `approxVramGB` numbers are "target GPU class" — i.e. the
 * VRAM tier the preset is designed for — so we recommend the highest preset
 * whose target is ≤ detected. Falls back to the memory tier when even the
 * smallest preset doesn't fit.
 */
export function recommendPreset(presets: ConfigPreset[], detectedVramGB: number): ConfigPreset | null {
  if (!presets.length) return null;
  const fitting = presets
    .filter(p => p.approxVramGB <= detectedVramGB)
    .sort((a, b) => b.approxVramGB - a.approxVramGB);
  return fitting[0] ?? presets.find(p => p.tier === 'memory') ?? presets[0];
}
