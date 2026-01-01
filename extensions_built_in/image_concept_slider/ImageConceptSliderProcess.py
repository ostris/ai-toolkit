"""
Image-based Concept Slider Trainer Process

This implements training of concept sliders using image sequences rather than text prompts.
It supports variable-length image sequences with user-defined suffixes for positive and negative images.

The training approach (dual-pass like concept_slider):
1. Load image sequences (multiple pos/neg images per concept)
2. Encode all images to latents
3. Add the same noise to all latents
4. Compute direction targets (enhance/erase) using guidance scale
5. Do two forward passes: one with network multiplier +1.0, one with -1.0
6. Compute enhance/erase losses for each pass and combine
7. Optionally compute anchor loss to preserve unrelated content

Based on the visual slider concept from https://sliders.baulab.info/
"""

import copy
import random
from collections import OrderedDict
import os
from typing import Optional, Union, List, Literal
from torch.utils.data import ConcatDataset, DataLoader

from toolkit.stable_diffusion_model import StableDiffusion, PromptEmbeds
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
import gc
from toolkit import train_tools
import torch
from jobs.process import BaseSDTrainProcess
from toolkit.basic import value_map
from toolkit.print import print_acc


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def norm_like_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize the tensor to have the same mean and std as the target tensor."""
    tensor_mean = tensor.mean()
    tensor_std = tensor.std()
    target_mean = target.mean()
    target_std = target.std()
    normalized_tensor = (tensor - tensor_mean) / (
        tensor_std + 1e-8
    ) * target_std + target_mean
    return normalized_tensor


class ImageConceptSliderConfig:
    """
    Extended config for image-based concept slider training with anchor support.
    Uses datasets from the main 'datasets' section and applies suffix filtering globally.
    """
    def __init__(self, **kwargs):
        # Global suffix configuration - applied to all datasets
        pos_suffixes = kwargs.get('positive_suffixes', ['_pos'])
        neg_suffixes = kwargs.get('negative_suffixes', ['_neg'])
        
        if isinstance(pos_suffixes, str):
            self.positive_suffixes: List[str] = [s.strip() for s in pos_suffixes.split(',') if s.strip()]
        else:
            self.positive_suffixes: List[str] = pos_suffixes
            
        if isinstance(neg_suffixes, str):
            self.negative_suffixes: List[str] = [s.strip() for s in neg_suffixes.split(',') if s.strip()]
        else:
            self.negative_suffixes: List[str] = neg_suffixes
        
        # Scales for each suffix (positive values for positive direction, negative for negative)
        scales = kwargs.get('scales', None)
        if scales is None:
            self.scales: List[float] = [1.0] * len(self.positive_suffixes) + [-1.0] * len(self.negative_suffixes)
        elif isinstance(scales, str):
            self.scales: List[float] = [float(s.strip()) for s in scales.split(',') if s.strip()]
        else:
            self.scales: List[float] = scales
        
        self.weight_jitter: float = kwargs.get('weight_jitter', 0.0)
        # Additional loss types (e.g., 'prior_preservation')
        self.additional_losses: List[str] = kwargs.get('additional_losses', [])
        
        # Guidance strength for direction calculation (like concept_slider)
        self.guidance_strength: float = kwargs.get('guidance_strength', 3.0)
        
        # Anchor configuration
        # anchor_mode: "none" | "suffix" | "prompt"
        self.anchor_mode: str = kwargs.get('anchor_mode', 'none')
        self.anchor_strength: float = kwargs.get('anchor_strength', 1.0)
        
        # For suffix mode: anchor images loaded from dataset with these suffixes
        anchor_suffixes = kwargs.get('anchor_suffixes', [])
        if isinstance(anchor_suffixes, str):
            self.anchor_suffixes: List[str] = [s.strip() for s in anchor_suffixes.split(',') if s.strip()]
        else:
            self.anchor_suffixes: List[str] = anchor_suffixes if anchor_suffixes else []
        
        # For prompt mode: generate anchor latents from this prompt
        self.anchor_prompt: str = kwargs.get('anchor_prompt', '')
        
        # Anchor generation mode: "once" (cache at start) or "per_batch" (generate each batch)
        self.anchor_generation_mode: str = kwargs.get('anchor_generation_mode', 'once')


class ImageConceptSliderProcess(BaseSDTrainProcess):
    """
    Training process for image-based concept sliders.
    
    Supports multiple images per sequence (not just pairs) with configurable
    suffixes and scales for fine-grained control over the slider direction.
    
    Uses the main 'datasets' section for folder paths and resolution, and
    the 'image_slider' section for suffix/scale configuration.
    """
    sd: StableDiffusion
    data_loader: DataLoader = None

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.step_num = 0
        self.start_step = 0
        self.device = self.get_conf('device', self.job.device)
        self.device_torch = torch.device(self.device)
        
        # Store raw dataset configs before clearing self.datasets
        # We use SequenceImageDataset instead of the standard AiToolkitDataset
        self.raw_dataset_configs = self.get_conf('datasets', [])
        
        # Prevent parent from creating standard data loader
        # ImageConceptSlider uses its own SequenceImageDataset in load_datasets()
        self.datasets = None
        self.datasets_reg = None
        
        # Load slider config from 'image_slider' key
        slider_raw = self.get_conf('image_slider', None)
        if slider_raw is None:
            slider_raw = self.get_conf('slider', {})
        self.slider_config = ImageConceptSliderConfig(**slider_raw)
        
        print_acc(f"Image Concept Slider Config:")
        print_acc(f"  - Positive suffixes: {self.slider_config.positive_suffixes}")
        print_acc(f"  - Negative suffixes: {self.slider_config.negative_suffixes}")
        print_acc(f"  - Scales: {self.slider_config.scales}")
        print_acc(f"  - Weight jitter: {self.slider_config.weight_jitter}")
        print_acc(f"  - Guidance strength: {self.slider_config.guidance_strength}")
        print_acc(f"  - Anchor mode: {self.slider_config.anchor_mode}")
        if self.slider_config.anchor_mode == 'suffix':
            print_acc(f"  - Anchor suffixes: {self.slider_config.anchor_suffixes}")
        elif self.slider_config.anchor_mode == 'prompt':
            print_acc(f"  - Anchor prompt: {self.slider_config.anchor_prompt}")
            print_acc(f"  - Anchor generation mode: {self.slider_config.anchor_generation_mode}")
        print_acc(f"  - Anchor strength: {self.slider_config.anchor_strength}")
        
        # For prompt-based anchor, we'll cache the generated latents here
        self.anchor_latents_cache: Optional[torch.Tensor] = None
        self.anchor_embeds_cache: Optional[PromptEmbeds] = None

    def load_datasets(self):
        """Load datasets from main config and apply suffix filtering from image_slider config."""
        if self.data_loader is None:
            print_acc(f"Loading image sequence datasets")
            datasets = []
            
            # Use raw dataset configs stored in __init__
            raw_datasets = self.raw_dataset_configs
            
            if not raw_datasets:
                raise ValueError("No datasets configured. Add datasets in the 'datasets' section.")
            
            # Choose dataset class based on anchor mode
            use_anchor_dataset = (
                self.slider_config.anchor_mode == 'suffix' and 
                len(self.slider_config.anchor_suffixes) > 0
            )
            
            if use_anchor_dataset:
                from extensions_built_in.image_concept_slider.AnchorSequenceImageDataset import AnchorSequenceImageDataset
                DatasetClass = AnchorSequenceImageDataset
            else:
                from toolkit.data_loader import SequenceImageDataset
                DatasetClass = SequenceImageDataset
            
            for dataset_raw in raw_datasets:
                # Parse the dataset config
                if isinstance(dataset_raw, dict):
                    folder_path = dataset_raw.get('folder_path', '')
                    resolution = dataset_raw.get('resolution', 512)
                    # Support both 'resolution' and 'size' keys
                    if isinstance(resolution, list):
                        size = resolution[0] if len(resolution) > 0 else 512
                    else:
                        size = dataset_raw.get('size', resolution)
                    network_weight = dataset_raw.get('network_weight', 1.0)
                    target_class = dataset_raw.get('default_caption', '') or dataset_raw.get('target_class', '')
                else:
                    # Handle DatasetConfig objects
                    folder_path = dataset_raw.folder_path
                    size = dataset_raw.resolution[0] if hasattr(dataset_raw, 'resolution') and dataset_raw.resolution else 512
                    network_weight = dataset_raw.network_weight if hasattr(dataset_raw, 'network_weight') else 1.0
                    target_class = dataset_raw.default_caption if hasattr(dataset_raw, 'default_caption') else ''
                
                print_acc(f"  - Dataset: {folder_path}")
                print_acc(f"    Resolution: {size}")
                print_acc(f"    Target class: {target_class}")
                
                # Build config dict using global suffixes from image_slider
                config_dict = {
                    'folder_path': folder_path,
                    'positive_suffixes': self.slider_config.positive_suffixes,
                    'negative_suffixes': self.slider_config.negative_suffixes,
                    'scales': self.slider_config.scales,
                    'network_weight': network_weight,
                    'target_class': target_class,
                    'size': size,
                }
                
                # Add anchor suffixes if using anchor dataset
                if use_anchor_dataset:
                    config_dict['anchor_suffixes'] = self.slider_config.anchor_suffixes
                
                image_dataset = DatasetClass(config_dict)
                datasets.append(image_dataset)

            if len(datasets) == 0:
                raise ValueError("No datasets configured for image concept slider training")
                
            concatenated_dataset = ConcatDataset(datasets)
            
            # Custom collate function for sequence data
            def collate_fn(batch):
                # batch is list of tuples - length depends on dataset class
                # Standard: (images_tensor, scales, prompt, network_weight)
                # With anchors: (images_tensor, scales, prompt, network_weight, anchor_images)
                has_anchors = len(batch[0]) == 5
                
                images_list = [item[0] for item in batch]
                scales_list = [item[1] for item in batch]
                prompts = [item[2] for item in batch]
                network_weights = [item[3] for item in batch]
                
                # Stack if all have same shape
                try:
                    images = torch.stack(images_list, dim=0)
                    scales = torch.stack(scales_list, dim=0)
                except RuntimeError:
                    images = images_list
                    scales = scales_list
                
                if has_anchors:
                    anchor_list = [item[4] for item in batch]
                    # Anchor images may be None for some items
                    return images, scales, prompts, network_weights, anchor_list
                else:
                    return images, scales, prompts, network_weights, None
            
            self.data_loader = DataLoader(
                concatenated_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=collate_fn,
            )
            
            print_acc(f"Loaded {len(concatenated_dataset)} image sequences total")

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        """Called before the training loop starts."""
        self.sd.vae.eval()
        self.sd.vae.to(self.device_torch)
        self.load_datasets()
        
        # For prompt-based anchor mode, encode the anchor prompt
        if self.slider_config.anchor_mode == 'prompt' and self.slider_config.anchor_prompt:
            print_acc(f"Encoding anchor prompt: {self.slider_config.anchor_prompt}")
            with torch.no_grad():
                self.anchor_embeds_cache = (
                    self.sd.encode_prompt([self.slider_config.anchor_prompt])
                    .to(self.device_torch, dtype=self.sd.torch_dtype)
                    .detach()
                )
            
            # If anchor_generation_mode is 'once', generate anchor latents now
            if self.slider_config.anchor_generation_mode == 'once':
                print_acc("Generating anchor latents (once mode)...")
                self._generate_anchor_latents_once()
    
    def _generate_anchor_latents_once(self):
        """
        Generate anchor latents once at the start of training.
        These are used to compute anchor loss throughout training.
        """
        # Generate a batch of random latents using the model
        # We use a fixed seed for reproducibility
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)
            
            # Get a sample resolution from first dataset
            sample_size = 512
            if self.raw_dataset_configs:
                first_ds = self.raw_dataset_configs[0]
                if isinstance(first_ds, dict):
                    res = first_ds.get('resolution', 512)
                    sample_size = res[0] if isinstance(res, list) else res
            
            # Calculate latent dimensions
            latent_size = sample_size // 8  # VAE downsampling factor
            
            # Generate random latent
            generator = torch.Generator(device=self.device_torch).manual_seed(42)
            anchor_latent = torch.randn(
                1, 4, latent_size, latent_size,
                generator=generator,
                device=self.device_torch,
                dtype=dtype
            )
            
            self.anchor_latents_cache = anchor_latent.detach()
            print_acc(f"  Cached anchor latents with shape: {self.anchor_latents_cache.shape}")

    def hook_train_loop(self, batch):
        """
        Main training step for image-based concept slider with dual-pass training.
        
        This implements the same approach as concept_slider:
        1. Compute neutral, positive, and negative predictions (network disabled)
        2. Calculate enhance/erase targets using guidance scale
        3. Do forward pass with network multiplier = +1.0, compute loss, backward
        4. Do forward pass with network multiplier = -1.0, compute loss, backward
        5. Optionally compute anchor loss to preserve unrelated content
        """
        # Handle batch from gradient accumulation
        if isinstance(batch, list):
            batch = batch[0]
        
        # Unpack batch - may have 4 or 5 elements depending on anchor mode
        if len(batch) == 5:
            images, scales, prompts, network_weights, anchor_images_batch = batch
        else:
            images, scales, prompts, network_weights = batch
            anchor_images_batch = None
        
        with torch.no_grad():
            # Handle single batch item
            if isinstance(network_weights, list):
                network_weight = network_weights[0]
            else:
                network_weight = network_weights.item() if isinstance(network_weights, torch.Tensor) else network_weights
            
            # Apply weight jitter for regularization
            loss_jitter_multiplier = 1.0
            weight_jitter = self.slider_config.weight_jitter
            if weight_jitter > 0.0:
                jitter = random.uniform(-weight_jitter, weight_jitter)
                network_weight += jitter
                loss_jitter_multiplier = value_map(abs(jitter), 0.0, weight_jitter, 1.0, 0.0)
            
            dtype = get_torch_dtype(self.train_config.dtype)
            
            # images shape: [B, num_images, C, H, W] or list
            if isinstance(images, list):
                images = images[0]
            if images.dim() == 5:
                images = images[0]  # [num_images, C, H, W]
            
            images = images.to(self.device_torch, dtype=dtype)
            num_images = images.shape[0]
            
            # Get scales for this batch
            if isinstance(scales, list):
                scales = scales[0]
            if scales.dim() == 2:
                scales = scales[0]
            scales = scales.to(self.device_torch, dtype=dtype)
            
            # Separate positive and negative images based on scale sign
            positive_indices = [i for i in range(num_images) if scales[i].item() > 0]
            negative_indices = [i for i in range(num_images) if scales[i].item() < 0]
            
            # Encode all images to latents
            latents_list = []
            for i in range(num_images):
                img = images[i:i+1]
                latent = self.sd.encode_images(img)
                latents_list.append(latent)
            
            all_latents = torch.cat(latents_list, dim=0)
            
            # Compute mean latents for positive and negative directions
            if positive_indices:
                positive_latents = torch.stack([latents_list[i] for i in positive_indices], dim=0)
                positive_mean = positive_latents.mean(dim=0)
            else:
                positive_mean = all_latents.mean(dim=0, keepdim=True)
            
            if negative_indices:
                negative_latents = torch.stack([latents_list[i] for i in negative_indices], dim=0)
                negative_mean = negative_latents.mean(dim=0)
            else:
                negative_mean = all_latents.mean(dim=0, keepdim=True)
            
            # Neutral is the midpoint
            neutral_latent = (positive_mean + negative_mean) / 2.0
            
            # Process anchor images if present (suffix mode only)
            anchor_latent = None
            if self.slider_config.anchor_mode == 'suffix' and anchor_images_batch is not None:
                if isinstance(anchor_images_batch, list):
                    anchor_imgs = anchor_images_batch[0]
                else:
                    anchor_imgs = anchor_images_batch
                
                if anchor_imgs is not None and anchor_imgs.numel() > 0:
                    if anchor_imgs.dim() == 5:
                        anchor_imgs = anchor_imgs[0]
                    anchor_imgs = anchor_imgs.to(self.device_torch, dtype=dtype)
                    # Encode anchor images and take mean
                    anchor_latents_list = []
                    for i in range(anchor_imgs.shape[0]):
                        anchor_lat = self.sd.encode_images(anchor_imgs[i:i+1])
                        anchor_latents_list.append(anchor_lat)
                    anchor_latent = torch.cat(anchor_latents_list, dim=0).mean(dim=0, keepdim=True)
            
            # Note: For prompt mode, we don't need separate anchor latents
            # We use the same noisy_neutral with anchor embeddings in Phase 1
            
            if self.train_config.gradient_checkpointing:
                self.sd.unet.enable_gradient_checkpointing()

            noise_scheduler = self.sd.noise_scheduler
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler

            self.sd.noise_scheduler.set_timesteps(
                self.train_config.max_denoising_steps, device=self.device_torch
            )

            # Sample a random timestep
            timesteps = torch.randint(0, self.train_config.max_denoising_steps, (1,), device=self.device_torch)
            timesteps = timesteps.long()

            # Generate ONE noise tensor for all images (crucial for learning direction)
            noise = self.sd.get_latent_noise_from_latents(
                latents=neutral_latent,
                noise_offset=self.train_config.noise_offset,
            ).to(self.device_torch, dtype=dtype)
            
            # Add noise to key latents
            noisy_positive = noise_scheduler.add_noise(positive_mean, noise, timesteps)
            noisy_negative = noise_scheduler.add_noise(negative_mean, noise, timesteps)
            noisy_neutral = noise_scheduler.add_noise(neutral_latent, noise, timesteps)
            
            # For suffix mode, also add noise to anchor latents
            noisy_anchor = None
            if anchor_latent is not None:
                # Resize noise if anchor has different shape
                if anchor_latent.shape != noise.shape:
                    anchor_noise = self.sd.get_latent_noise_from_latents(
                        latents=anchor_latent,
                        noise_offset=self.train_config.noise_offset,
                    ).to(self.device_torch, dtype=dtype)
                else:
                    anchor_noise = noise
                noisy_anchor = noise_scheduler.add_noise(anchor_latent, anchor_noise, timesteps)

        # Encode prompt
        with torch.set_grad_enabled(self.train_config.train_text_encoder):
            prompt_list = []
            for prompt in prompts:
                if isinstance(prompt, tuple):
                    prompt = prompt[0]
                prompt_list.append(prompt)
            
            conditional_embeds = self.sd.encode_prompt(prompt_list).to(self.device_torch, dtype=dtype)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # ===== PHASE 1: Compute targets with network DISABLED =====
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        
        with torch.no_grad():
            # Predict noise for positive, negative, and neutral separately
            # (Batching can cause issues with some model architectures)
            positive_pred = self.sd.predict_noise(
                latents=noisy_positive,
                conditional_embeddings=conditional_embeds,
                timestep=timesteps,
            )
            
            negative_pred = self.sd.predict_noise(
                latents=noisy_negative,
                conditional_embeddings=conditional_embeds,
                timestep=timesteps,
            )
            
            neutral_pred = self.sd.predict_noise(
                latents=noisy_neutral,
                conditional_embeddings=conditional_embeds,
                timestep=timesteps,
            )
            
            # Compute anchor target if anchor mode is enabled
            # For prompt mode: use same noisy_neutral but with anchor prompt embeddings
            # For suffix mode: use noisy_anchor (from actual anchor images)
            anchor_target = None
            if self.slider_config.anchor_mode == 'prompt' and self.anchor_embeds_cache is not None:
                # Prompt mode: predict for neutral latents with anchor prompt
                # This constrains the output to match what anchor prompt would produce
                anchor_target = self.sd.predict_noise(
                    latents=noisy_neutral,
                    conditional_embeddings=self.anchor_embeds_cache.to(self.device_torch, dtype=dtype),
                    timestep=timesteps,
                )
            elif self.slider_config.anchor_mode == 'suffix' and noisy_anchor is not None:
                # Suffix mode: predict for actual anchor image latents
                anchor_target = self.sd.predict_noise(
                    latents=noisy_anchor,
                    conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                    timestep=timesteps,
                )
            
            # Calculate direction targets (like concept_slider)
            guidance_scale = self.slider_config.guidance_strength
            
            # Direction vectors
            positive_dir = (positive_pred - neutral_pred) - (negative_pred - neutral_pred)
            negative_dir = (negative_pred - neutral_pred) - (positive_pred - neutral_pred)
            
            # Enhance/erase targets
            enhance_positive_target = neutral_pred + guidance_scale * positive_dir
            enhance_negative_target = neutral_pred + guidance_scale * negative_dir
            erase_negative_target = neutral_pred - guidance_scale * negative_dir
            erase_positive_target = neutral_pred - guidance_scale * positive_dir
            
            # Normalize to neutral distribution
            enhance_positive_target = norm_like_tensor(enhance_positive_target, neutral_pred)
            enhance_negative_target = norm_like_tensor(enhance_negative_target, neutral_pred)
            erase_negative_target = norm_like_tensor(erase_negative_target, neutral_pred)
            erase_positive_target = norm_like_tensor(erase_positive_target, neutral_pred)
        
        # Restore network
        if self.network is not None:
            self.network.is_active = was_network_active
        
        # ===== PHASE 2: Training with network ENABLED =====
        
        # === Pass 1: Positive direction (multiplier = +1.0) ===
        with self.network:
            assert self.network.is_active
            self.network.set_multiplier(1.0 * network_weight)
            
            # Predict for neutral (main training target)
            class_pred = self.sd.predict_noise(
                latents=noisy_neutral.to(self.device_torch, dtype=dtype),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                timestep=timesteps,
            )
            
            # Compute losses for positive pass
            enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_positive_target)
            erase_loss = torch.nn.functional.mse_loss(class_pred, erase_negative_target)
            
            # Anchor loss constrains the network output to not drift from anchor target
            # anchor_target was computed in Phase 1 with network disabled
            if anchor_target is not None:
                anchor_loss = torch.nn.functional.mse_loss(class_pred, anchor_target)
                anchor_loss = anchor_loss * self.slider_config.anchor_strength
                total_pos_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
            else:
                total_pos_loss = (enhance_loss + erase_loss) / 2.0
            
            total_pos_loss = total_pos_loss * loss_jitter_multiplier
            total_pos_loss.backward()
            pos_loss_value = total_pos_loss.detach().item()
            
            # === Pass 2: Negative direction (multiplier = -1.0) ===
            self.network.set_multiplier(-1.0 * network_weight)
            
            # Predict for neutral
            class_pred = self.sd.predict_noise(
                latents=noisy_neutral.to(self.device_torch, dtype=dtype),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                timestep=timesteps,
            )
            
            # Compute losses for negative pass (reversed targets)
            enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_negative_target)
            erase_loss = torch.nn.functional.mse_loss(class_pred, erase_positive_target)
            
            # Anchor loss constrains the network output to not drift from anchor target
            if anchor_target is not None:
                anchor_loss = torch.nn.functional.mse_loss(class_pred, anchor_target)
                anchor_loss = anchor_loss * self.slider_config.anchor_strength
                total_neg_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
            else:
                total_neg_loss = (enhance_loss + erase_loss) / 2.0
            
            total_neg_loss = total_neg_loss * loss_jitter_multiplier
            total_neg_loss.backward()
            neg_loss_value = total_neg_loss.detach().item()
            
            # Reset multiplier
            self.network.set_multiplier(1.0)

        # Apply gradients
        optimizer.step()
        lr_scheduler.step()

        # Combined loss
        total_loss = (pos_loss_value + neg_loss_value) / 2.0

        loss_dict = OrderedDict({
            'loss': total_loss,
            'loss_pos': pos_loss_value,
            'loss_neg': neg_loss_value,
        })

        return loss_dict
