# Backend Configuration Options NOT Exposed in the Guided Wizard

## Summary
The toolkit backend (toolkit/config_modules.py) supports **200+ configuration options** across multiple categories. The guided wizard only exposes approximately **30-35 fields**, leaving ~165+ fields hidden that power advanced features, optimization, and specialized training modes.

---

## TRAINING CONFIGURATION (65+ hidden options)

### Learning & Optimization Strategy
- **noise_scheduler** (default: 'ddpm') - Algorithm for noise schedule (ddpm, euler, lms, etc.)
- **lr_scheduler** (default: 'constant') - Learning rate scheduling strategy
- **lr_scheduler_params** (default: {}) - Parameters for learning rate scheduler
- **unet_lr** - Separate learning rate for UNet component
- **text_encoder_lr** - Separate learning rate for text encoder
- **refiner_lr** - Learning rate for refiner model (SDXL)
- **embedding_lr** - Learning rate for embedding training
- **adapter_lr** - Learning rate for adapter networks

### Timestep & Denoising Control
- **timestep_type** (default: 'sigmoid') - How to sample timesteps (sigmoid, linear, lognorm_blend, next_sample, weighted, one_step)
- **next_sample_timesteps** (default: 8) - Number of timesteps for next_sample strategy
- **linear_timesteps** (default: False) - Use linear spacing for timesteps
- **linear_timesteps2** (default: False) - Alternative linear timestep method
- **min_denoising_steps** (default: 0) - Minimum noise level to train on
- **max_denoising_steps** (default: 999) - Maximum noise level to train on
- **num_train_timesteps** (default: 1000) - Total timesteps in noise schedule

### Noise & Image Multipliers
- **noise_multiplier** (default: 1.0) - Scales noise during training
- **target_noise_multiplier** (default: 1.0) - Target noise scale
- **random_noise_multiplier** (default: 0.0) - Add randomness to noise scaling
- **random_noise_shift** (default: 0.0) - Shift in noise randomness
- **img_multiplier** (default: 1.0) - Scale for image latents
- **noisy_latent_multiplier** (default: 1.0) - Scale for noisy latents
- **latent_multiplier** (default: 1.0) - General latent scaling

### Loss & Preservation Mechanisms
- **loss_target** (default: 'noise') - What to predict: noise, source, unaugmented, differential_noise
- **pred_scaler** (default: 1.0) - Scale predictions (increase for more detail)
- **min_snr_gamma** - Minimum SNR for loss weighting
- **snr_gamma** - SNR value for loss weighting
- **learnable_snr_gos** (default: False) - Learn SNR adjustment parameters
- **correct_pred_norm** (default: False) - Correct prediction norm for drift
- **correct_pred_norm_multiplier** (default: 1.0) - Strength of norm correction
- **target_norm_std** - Target standard deviation for output normalization
- **target_norm_std_value** (default: 1.0) - Value of target std

### Advanced Preservation Strategies
- **blank_prompt_preservation** (default: False) - Preserve model's blank prompt understanding
- **blank_prompt_preservation_multiplier** (default: 1.0) - Strength of blank prompt preservation
- **inverted_mask_prior** (default: False) - Use inverted mask as regularization
- **inverted_mask_prior_multiplier** (default: 0.5) - Strength of inverted mask prior
- **do_prior_divergence** (default: False) - Apply negative loss to encourage divergence from prior

### Guidance & CFG Training
- **do_cfg** (default: False) - Enable classifier-free guidance during training
- **do_random_cfg** (default: False) - Random CFG scale during training
- **cfg_scale** (default: 1.0) - Classifier-free guidance scale
- **max_cfg_scale** (default: 1.0) - Maximum CFG scale
- **cfg_rescale** - CFG rescale factor
- **do_guidance_loss** (default: False) - Enable guidance/contrastive loss
- **guidance_loss_target** (default: 3.0) - Target guidance loss value
- **do_guidance_loss_cfg_zero** (default: False) - CFG guidance loss at zero
- **unconditional_prompt** (default: '') - Unconditional prompt for guidance
- **do_differential_guidance** (default: False) - Differential guidance strategy
- **differential_guidance_scale** (default: 3.0) - Scale for differential guidance

### Gradient & Accumulation
- **gradient_accumulation** (default: 1) - Steps to accumulate gradients
- **gradient_accumulation_steps** (default: 1) - Legacy accumulation setting
- **max_grad_norm** (default: 1.0) - Gradient clipping threshold
- **weight_jitter** (default: 0.0) - Add jitter to weights during training

### Advanced Training Modes
- **train_turbo** (default: False) - Enable turbo training mode
- **show_turbo_outputs** (default: False) - Display turbo outputs during training
- **train_unet** (default: True) - Whether to train UNet
- **train_text_encoder** (default: False) - Whether to train text encoder
- **train_refiner** (default: True) - Whether to train refiner (SDXL)
- **free_u** (default: False) - FreeU training mode
- **disable_sampling** (default: False) - Disable sample generation during training
- **skip_first_sample** (default: False) - Skip sampling at first step
- **force_first_sample** (default: False) - Force sampling at first step

### Prompt & Embedding Handling
- **prompt_dropout_prob** (default: 0.0) - Dropout rate before encoding
- **prompt_saturation_chance** (default: 0.0) - Chance to repeat prompt
- **short_and_long_captions** (default: False) - Double batch with short/long captions
- **short_and_long_captions_encoder_split** (default: False) - Split captions between encoders
- **match_adapter_chance** (default: 0.0) - Chance to match adapter
- **cache_text_embeddings** (default: False) - Cache text embeddings to memory
- **unload_text_encoder** (default: False) - Unload text encoder to CPU
- **negative_prompt** - Default negative prompt
- **max_negative_prompts** (default: 1) - Number of negative prompts

### Adaptive & Dynamic Features
- **adaptive_scaling_factor** (default: False) - Adapt VAE scaling based on image norm
- **dynamic_noise_offset** (default: False) - Dynamically adjust noise offset
- **match_noise_norm** (default: False) - Match noise norm to preserve brightness
- **standardize_images** (default: False) - Standardize images to model stats
- **standardize_latents** (default: False) - Standardize latents to model stats

### Advanced Optimizer Settings
- **single_item_batching** (default: False) - Accumulate single items (special gradient accumulation)
- **noise_offset** (default: 0.0) - Offset added to noise (helps with dark images)
- **batch_size_warmup_steps** (default: 100) - Steps to warmup batch size
- **do_paramiter_swapping** (default: False) - Swap parameters during training (VRAM optimization)
- **paramiter_swapping_factor** (default: 0.1) - Fraction of parameters active at once

### Adapter Assistance
- **adapter_assist_name_or_path** - Path to adapter for training assistance
- **adapter_assist_type** (default: 't2i') - Type of assistant adapter (t2i, control_net)

### Regularization
- **reg_weight** (default: 1.0) - Multiplier for regularization loss

### Multi-Stage Training
- **switch_boundary_every** (default: 1) - How often to switch stage boundary

### Feature Extraction
- **latent_feature_extractor_path** - Path to feature extractor model
- **latent_feature_loss_weight** (default: 1.0) - Weight for feature loss
- **diffusion_feature_extractor_path** - Path to diffusion feature extractor
- **diffusion_feature_extractor_weight** (default: 1.0) - Weight for diffusion features
- **optimal_noise_pairing_samples** (default: 1) - Samples for optimal noise pairing
- **force_consistent_noise** (default: False) - Force same noise for same image at size
- **blended_blur_noise** (default: False) - Blend blur with noise

---

## MODEL CONFIGURATION (30+ hidden options)

### Memory & Performance Optimization
- **low_vram** (default: False) - Enable low VRAM mode
- **layer_offloading** (default: False) - Offload layers to CPU as needed
- **layer_offloading_transformer_percent** (default: 1.0) - % of transformer layers to offload
- **layer_offloading_text_encoder_percent** (default: 1.0) - % of text encoder to offload
- **attn_masking** (default: False) - Use attention masking (Flux only)
- **split_model_over_gpus** (default: False) - Split model across GPUs (Flux only)
- **split_model_other_module_param_count_scale** (default: 0.3) - Scale factor for GPU split

### Model Variants & Paths
- **vae_path** - Custom VAE model path
- **refiner_name_or_path** - Path to refiner model (SDXL)
- **lora_path** - Path to base LoRA weights
- **assistant_lora_path** - Path to assistant LoRA
- **inference_lora_path** - Path to inference LoRA
- **te_name_or_path** - Path to text encoder
- **extras_name_or_path** - Path to extras (VAE, text encoder, etc.)
- **accuracy_recovery_adapter** - Path to accuracy recovery adapter

### Device & Dtype Control
- **dtype** - Data type for model (float16, float32, bfloat16)
- **vae_device** - Device for VAE (cuda, cpu)
- **vae_dtype** - Data type for VAE
- **te_device** - Device for text encoder
- **te_dtype** - Data type for text encoder
- **unet_path** - Path to custom UNet
- **unet_sample_size** - Sample size for UNet
- **latent_space_version** - Version of latent space to use

### Text Encoder Configuration
- **text_encoder_bits** (default: 16) - Quantization bits for text encoder (16, 8, 4)
- **use_text_encoder_1** (default: True) - Use first text encoder (SDXL)
- **use_text_encoder_2** (default: True) - Use second text encoder (SDXL)

### Compilation & Optimization
- **compile** (default: False) - Compile model with torch.compile()
- **model_kwargs** (default: {}) - Extra kwargs for model initialization
- **quantize_kwargs** (default: {}) - Extra kwargs for quantization
- **ignore_if_contains** - Module patterns to ignore when training
- **only_if_contains** - Only train modules matching patterns

### Special Configurations
- **experimental_xl** (default: False) - Enable experimental SDXL features
- **use_flux_cfg** (default: False) - Enable Flux CFG mode

---

## NETWORK/LORA CONFIGURATION (10+ hidden options)

### Rank & Scaling
- **conv** - Convolution layer rank
- **alpha** (default: 1.0) - Alpha scaling factor
- **conv_alpha** - Alpha for conv layers
- **dropout** - Dropout in LoRA weights

### Advanced Network Types
- **transformer_only** (default: True) - Only train transformer layers
- **lokr_full_rank** (default: False) - Use full rank for LoKr
- **lokr_factor** (default: -1) - LoKr factorization (-1 = auto)
- **split_multistage_loras** (default: True) - Split LoRAs for multi-stage models
- **layer_offloading** (default: False) - Offload network layers

### Custom Network Config
- **lorm_config** - LoRM-specific configuration
- **network_kwargs** - Extra network parameters

---

## DATASET CONFIGURATION (40+ hidden options)

### Image Processing
- **random_scale** (default: False) - Random scaling of images
- **random_crop** (default: False) - Random cropping
- **scale** (default: 1.0) - Scale factor for images
- **flip_x** (default: False) - Horizontal flip augmentation
- **flip_y** (default: False) - Vertical flip augmentation
- **square_crop** (default: False) - Crop to square

### Bucketing & Resolution
- **buckets** (default: True) - Enable aspect ratio bucketing
- **bucket_tolerance** (default: 64) - Tolerance for bucketing
- **enable_ar_bucket** - Enable aspect ratio bucketing (dynamically set in wizard)

### Caption & Token Handling
- **caption_ext** - File extension for captions (.txt, .md, etc.)
- **keep_tokens** (default: 0) - First N tokens to always keep
- **token_dropout_rate** (default: 0.0) - Dropout rate for tokens
- **shuffle_tokens** (default: False) - Shuffle tokens
- **use_short_captions** (default: False) - Use 'caption_short' from JSON
- **caption_dropout_rate** (EXPOSED in wizard)
- **trigger_word** - Trigger word for dataset

### Dataset Variants
- **dataset_path** - Path to JSON/folder dataset
- **default_caption** - Default caption if missing
- **random_triggers** - List of random trigger words
- **random_triggers_max** (default: 1) - Max random triggers per image
- **type** (default: 'image') - Dataset type (image, slider, reference)

### Repetition & Sampling
- **num_repeats** (default: 1) - Repeat dataset N times
- **cache_latents** (default: False) - Cache image latents in memory
- **cache_latents_to_disk** (default: False) - Cache latents to disk
- **cache_text_embeddings** (default: False) - Cache text embeddings
- **cache_clip_vision_to_disk** (default: False) - Cache CLIP vision embeddings

### Augmentation
- **augments** - List of augmentation names
- **augmentations** - Albumentations configuration
- **shuffle_augmentations** (default: False) - Randomize augmentation order
- **clip_image_augmentations** - Augmentations for CLIP images
- **clip_image_shuffle_augmentations** (default: False) - Shuffle CLIP augmentations

### Control Images & Masks
- **control_path** - Path to control images (depth, pose, etc.)
- **control_path_1, 2, 3** - Multiple control image paths
- **control_transparent_color** (default: [0, 0, 0]) - Color for transparent regions
- **inpaint_path** - Path to inpaint masks
- **full_size_control_images** (default: True) - Use full-size control images
- **alpha_mask** (default: False) - Use alpha channel as mask
- **mask_path** - Path to focus masks
- **unconditional_path** - Path to unconditional images
- **invert_mask** (default: False) - Invert mask values
- **mask_min_value** (default: 0.0) - Minimum mask value

### Regularization
- **is_reg** (default: False) - This is a regularization dataset
- **prior_reg** (default: False) - This is a prior regularization set
- **network_weight** (default: 1.0) - Weight for network training on this dataset
- **loss_multiplier** (default: 1.0) - Multiplier for loss

### CLIP & Reference Images
- **clip_image_path** - Path to CLIP reference images
- **clip_image_from_same_folder** (default: False) - Get CLIP images from same folder
- **replacements** - Text replacements to apply
- **poi** - Point of interest for auto-crop

### Workers & Performance
- **prefetch_factor** (default: 2) - Prefetch batches in dataloader
- **persistent_workers** (default: False) - Keep workers alive between epochs
- **gpu_prefetch_batches** (EXPOSED in wizard)

### Video & Multi-Frame
- **num_frames** (default: 1) - Number of frames for video
- **shrink_video_to_frames** (default: True) - Shrink video to frames
- **fps** (default: 16) - FPS for video extraction
- **do_i2v** (default: True) - Do image-to-video on multi-modal models

### Advanced Features
- **controls** - List of automatic control types (depth, etc.)
- **fast_image_size** (default: False) - Fast (but potentially inaccurate) image size detection
- **standardize_images** (default: False) - Standardize to model mean/std
- **debug** (default: False) - Debug mode for frame selection
- **extra_values** - Extra values to track per sample

---

## SAMPLE/PREVIEW CONFIGURATION (15+ hidden options)

The wizard only exposes **sample_every**. Hidden options include:

- **sampler** (default: 'ddpm') - Sampler algorithm for generation
- **width/height** - Preview image dimensions
- **neg** (default: False) - Whether to use negative prompt
- **seed** (default: 0) - Seed for reproducibility
- **walk_seed** (default: False) - Walk through seed values
- **guidance_scale** (default: 7) - CFG guidance scale for previews
- **sample_steps** (default: 20) - Steps for preview generation
- **network_multiplier** (default: 1) - Network weight for previews
- **guidance_rescale** (default: 0.0) - Guidance rescale
- **format** (default: 'jpg') - Preview image format (jpg, png, webp)
- **adapter_conditioning_scale** (default: 1.0) - Adapter scale for previews
- **refiner_start_at** (default: 0.5) - When to start refiner (0.0-1.0)
- **do_cfg_norm** (default: False) - CFG normalization
- **num_frames/fps** - For video previews
- **extra_values** - Extra preview values

---

## SAVE CONFIGURATION (4+ hidden options)

The wizard only exposes **save_every** and **max_step_saves_to_keep**. Hidden:

- **dtype** - Data type for saved weights (float16, float32)
- **save_format** (default: 'safetensors') - Save format (safetensors, diffusers)
- **push_to_hub** (default: False) - Push to HuggingFace Hub
- **hf_repo_id** - HuggingFace repo ID
- **hf_private** (default: False) - Make Hub repo private

---

## LOGGING CONFIGURATION (5 completely hidden)

- **log_every** (default: 100) - Log metrics every N steps
- **verbose** (default: False) - Verbose logging
- **use_wandb** (default: False) - Log to Weights & Biases
- **project_name** (default: 'ai-toolkit') - Project name for logging
- **run_name** - Run name for logging session

---

## ADAPTER CONFIGURATION (30+ completely hidden)

These are entirely hidden but powerful for advanced training:

### IP-Adapter & Vision Features
- **type** - Adapter type (t2i, ip, ip+, clip, ilora, photo_maker, control_net, control_lora, i2v)
- **in_channels/channels** - Network channels
- **num_res_blocks/downscale_factor** - Architecture parameters
- **image_encoder_path** - Path to image encoder
- **num_tokens** - Number of tokens for IP-adapter
- **train_image_encoder** - Whether to train encoder
- **train_only_image_encoder** - Only train encoder
- **clip_layer** - Which CLIP layer to use

### Specialized Adapters
- **ilora_down/mid/up** - iLoRA layer configuration
- **head_dim/num_heads** - Attention dimensions
- **pixtral_max_image_size** - Max image size for Pixtral
- **quad_image** - Quad image mode

### Control Features
- **control_image_dropout** (default: 0.0) - Dropout for control images
- **num_control_images** (default: 1) - Number of control inputs
- **has_inpainting_input** - Support inpainting
- **invert_inpaint_mask_chance** - Chance to invert mask
- **lora_config** - Custom LoRA for control_lora

### Memory & Optimization
- **train_scaler** - Train output scaler
- **merge_scaler** - Merge scaler into weights
- **conv_pooling** - Use conv pooling
- **sparse_autoencoder_dim** - Sparse autoencoder dimension

---

## KEY INSIGHTS

### Most Impactful Hidden Options (for training quality):

1. **noise_scheduler** - Fundamental to training dynamics
2. **lr_scheduler & lr_scheduler_params** - Fine-grained learning control
3. **loss_target** - Changes what the model learns
4. **do_cfg** - Enables classifier-free guidance training
5. **cache_latents / cache_latents_to_disk** - Can drastically speed up training
6. **augmentations** - Data augmentation pipeline
7. **control_path** - For ControlNet training
8. **adapter** configs - For advanced adapter training

### Memory Optimization Options Hidden:
- layer_offloading system
- batch_size_warmup_steps
- do_paramiter_swapping
- quantize_kwargs for fine-grained quant control

### Quality Enhancement Hidden:
- inverted_mask_prior
- blank_prompt_preservation
- do_prior_divergence
- learnable_snr_gos
- optimal_noise_pairing

---

## RECOMMENDATIONS

**For users who need advanced configuration:**
1. Create custom YAML configs directly with these hidden options
2. Update the wizard to expose more commonly-used options like:
   - lr_scheduler / lr_scheduler_params (learning rate control)
   - noise_scheduler (training dynamics)
   - cache_latents / cache_latents_to_disk (speed)
   - augmentations (data quality)
   - loss_target (learning target)

**High-priority candidates for wizard exposure:**
- Learning rate scheduler control
- Noise scheduler selection
- Caching strategies
- Augmentation configuration
- Noise scheduling parameters (min/max denoising steps)
