# Quick Reference: Hidden Config Options by Category

## Training Configuration (Affects Training Behavior)

| Field | Default | Impact | Hidden? |
|-------|---------|--------|---------|
| noise_scheduler | 'ddpm' | Core training algorithm | YES |
| lr_scheduler | 'constant' | Learning rate schedule | YES |
| lr_scheduler_params | {} | LR warmup/decay config | YES |
| loss_target | 'noise' | What model predicts | YES |
| do_cfg | False | CFG guidance training | YES |
| cfg_scale | 1.0 | CFG strength | YES |
| cache_text_embeddings | False | Text embedding caching | YES |
| unload_text_encoder | False | Offload text encoder | YES |
| noise_offset | 0.0 | Helps dark images | YES |
| min_denoising_steps | 0 | Min noise level | YES |
| max_denoising_steps | 999 | Max noise level | YES |
| inverted_mask_prior | False | Masking strategy | YES |
| blank_prompt_preservation | False | Keep base model knowledge | YES |
| do_prior_divergence | False | Negative loss | YES |

## Learning Rate Control (Hidden)

| Field | Use Case |
|-------|----------|
| unet_lr | Separate UNet learning rate |
| text_encoder_lr | Separate text encoder LR |
| refiner_lr | Refiner model learning rate (SDXL) |
| embedding_lr | Embedding training LR |
| adapter_lr | Adapter network LR |

## Memory & Performance (Hidden)

| Field | Default | Impact |
|-------|---------|--------|
| low_vram | False | Enable low VRAM mode |
| layer_offloading | False | Offload layers to CPU |
| layer_offloading_transformer_percent | 1.0 | % transformer to offload |
| layer_offloading_text_encoder_percent | 1.0 | % text encoder to offload |
| batch_size_warmup_steps | 100 | Batch size ramp steps |
| do_paramiter_swapping | False | Swap parameters (advanced) |
| paramiter_swapping_factor | 0.1 | % parameters active |
| attn_masking | False | Flux attention optimization |
| split_model_over_gpus | False | Multi-GPU split (Flux) |
| compile | False | torch.compile() model |

## Dataset Configuration (Hidden)

| Field | Default | Impact |
|-------|---------|--------|
| augmentations | None | Data augmentation pipeline |
| cache_latents | False | Cache in memory (5x faster) |
| cache_latents_to_disk | False | Cache to disk (10x faster) |
| random_scale | False | Random scaling augmentation |
| random_crop | False | Random crop augmentation |
| flip_x | False | Horizontal flip |
| flip_y | False | Vertical flip |
| control_path | None | ControlNet image paths |
| mask_path | None | Masking image paths |
| alpha_mask | False | Use alpha as mask |
| keep_tokens | 0 | Protect first N tokens |
| token_dropout_rate | 0.0 | Token dropout |
| num_repeats | 1 | Dataset repetition |

## Sample/Preview Configuration (Hidden)

| Field | Default | Purpose |
|-------|---------|---------|
| sampler | 'ddpm' | Preview generation sampler |
| sample_steps | 20 | Steps per preview |
| guidance_scale | 7 | CFG for previews |
| seed | 0 | Preview seed |
| width | 512 | Preview width |
| height | 512 | Preview height |

## Logging Configuration (100% Hidden)

| Field | Default | Purpose |
|-------|---------|---------|
| use_wandb | False | Weights & Biases logging |
| project_name | 'ai-toolkit' | W&B project |
| run_name | None | W&B run identifier |
| log_every | 100 | Log frequency |
| verbose | False | Verbose output |

## Model Configuration (Hidden)

| Field | Purpose |
|-------|---------|
| vae_path | Custom VAE model |
| refiner_name_or_path | SDXL refiner path |
| te_name_or_path | Custom text encoder |
| lora_path | Base LoRA to load |
| vae_device | VAE computation device |
| vae_dtype | VAE data type |
| te_device | Text encoder device |
| te_dtype | Text encoder dtype |
| use_text_encoder_1 | Enable TE1 (SDXL) |
| use_text_encoder_2 | Enable TE2 (SDXL) |
| text_encoder_bits | TE quantization (16/8/4) |
| use_flux_cfg | Enable Flux CFG mode |

## Save Configuration (Mostly Hidden)

| Field | Default | Purpose |
|-------|---------|---------|
| save_format | 'safetensors' | Save format |
| dtype | 'float16' | Weight precision |
| push_to_hub | False | Upload to HF |
| hf_repo_id | None | HF repository |
| hf_private | False | Private repo |

## Network Configuration (Hidden)

| Field | Default | Impact |
|-------|---------|--------|
| conv | None | Conv layer rank |
| alpha | 1.0 | Alpha scaling |
| conv_alpha | None | Conv alpha |
| dropout | None | LoRA dropout |
| transformer_only | True | Only train transformers |
| lokr_full_rank | False | LoKr full rank mode |
| lokr_factor | -1 | LoKr factorization |
| split_multistage_loras | True | Multi-stage support |
| lorm_config | None | LoRM configuration |

---

## Access Method Reference

### Option 1: YAML Config File
```yaml
train:
  noise_scheduler: euler
  lr_scheduler: linear
  lr_scheduler_params:
    warmup_steps: 500
  loss_target: noise
  do_cfg: true
  cfg_scale: 1.0
  cache_text_embeddings: true
  unload_text_encoder: true

dataset:
  augmentations:
    - name: RandomHorizontalFlip
      p: 0.5
  cache_latents: true
  num_repeats: 2

model:
  layer_offloading: true
  vae_path: /path/to/vae
```

### Option 2: REST API
```json
{
  "config": {
    "train": {
      "noise_scheduler": "euler",
      "lr_scheduler": "cosine"
    }
  }
}
```

### Option 3: CLI
```bash
python train.py --config /path/to/config.yaml
```

---

## Top 10 Hidden Options to Know About

1. **cache_latents_to_disk** - Can make training 5-10x faster
2. **noise_scheduler** - Fundamental to model behavior
3. **lr_scheduler** - Controls learning rate over time
4. **layer_offloading** - Save 20-30% VRAM
5. **augmentations** - Complete data pipeline control
6. **loss_target** - Change what model learns
7. **do_cfg** - Enable guidance during training
8. **unload_text_encoder** - Save memory
9. **inverted_mask_prior** - Advanced masking
10. **use_wandb** - Professional logging

