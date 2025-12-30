"""
Image-based Concept Slider Trainer Process

This implements training of concept sliders using image sequences rather than text prompts.
It supports variable-length image sequences with user-defined suffixes for positive and negative images.

The training approach:
1. Load image sequences (multiple pos/neg images per concept)
2. Encode all images to latents
3. Add the same noise to all latents
4. Set network multiplier based on the scale for each image
5. Predict noise and compute loss for each direction
6. The network learns to shift the model's output toward positive and away from negative

Based on the visual slider concept from https://sliders.baulab.info/
"""

import copy
import random
from collections import OrderedDict
import os
from typing import Optional, Union, List
from torch.utils.data import ConcatDataset, DataLoader

from toolkit.config_modules import ImageConceptSliderConfig, DatasetConfig
from toolkit.data_loader import SequenceImageDataset
from toolkit.prompt_utils import concat_prompt_embeds, split_prompt_embeds
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

    def load_datasets(self):
        """Load datasets from main config and apply suffix filtering from image_slider config."""
        if self.data_loader is None:
            print_acc(f"Loading image sequence datasets")
            datasets = []
            
            # Get datasets from the main 'datasets' config section
            raw_datasets = self.get_conf('datasets', [])
            
            if not raw_datasets:
                raise ValueError("No datasets configured. Add datasets in the 'datasets' section.")
            
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
                
                image_dataset = SequenceImageDataset(config_dict)
                datasets.append(image_dataset)

            if len(datasets) == 0:
                raise ValueError("No datasets configured for image concept slider training")
                
            concatenated_dataset = ConcatDataset(datasets)
            
            # Custom collate function for sequence data
            def collate_fn(batch):
                # batch is list of (images_tensor, scales, prompt, network_weight)
                images_list = [item[0] for item in batch]  # List of [num_images, C, H, W] tensors
                scales_list = [item[1] for item in batch]  # List of scale tensors
                prompts = [item[2] for item in batch]  # List of prompts
                network_weights = [item[3] for item in batch]  # List of network weights
                
                # For now, we process one sequence at a time (batch_size should typically be 1)
                # Stack if all have same shape, otherwise return as list
                try:
                    images = torch.stack(images_list, dim=0)  # [B, num_images, C, H, W]
                    scales = torch.stack(scales_list, dim=0)  # [B, num_scales]
                except RuntimeError:
                    # Different sequence lengths in batch - shouldn't happen if dataset is consistent
                    images = images_list
                    scales = scales_list
                
                return images, scales, prompts, network_weights
            
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

    def hook_train_loop(self, batch):
        """
        Main training step for image-based concept slider.
        
        For each image sequence:
        1. Encode all images to latents
        2. Add the same noise to all latents
        3. Predict noise with network multiplier = scale for each image
        4. Compute weighted loss
        """
        # batch is actually a batch_list from gradient accumulation
        # For simplicity, we just take the first batch (gradient_accumulation should be 1)
        if isinstance(batch, list):
            batch = batch[0]
        
        with torch.no_grad():
            images, scales, prompts, network_weights = batch
            
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
                images = images[0]  # Take first batch item
            if images.dim() == 5:
                images = images[0]  # Remove batch dimension: [num_images, C, H, W]
            
            images = images.to(self.device_torch, dtype=dtype)
            num_images = images.shape[0]
            
            # Get scales for this batch
            if isinstance(scales, list):
                scales = scales[0]
            if scales.dim() == 2:
                scales = scales[0]  # [num_scales]
            scales = scales.to(self.device_torch, dtype=dtype)
            
            # Encode all images to latents
            latents_list = []
            for i in range(num_images):
                img = images[i:i+1]  # Keep batch dimension: [1, C, H, W]
                latent = self.sd.encode_images(img)
                latents_list.append(latent)
            
            # Stack latents: [num_images, latent_channels, H, W]
            all_latents = torch.cat(latents_list, dim=0)
            
            height = images.shape[2]
            width = images.shape[3]
            
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

            # Generate ONE noise tensor that will be used for all images in the sequence
            # This is crucial for learning the direction between positive and negative
            noise = self.sd.get_latent_noise(
                pixel_height=height,
                pixel_width=width,
                batch_size=1,
                noise_offset=self.train_config.noise_offset,
            ).to(self.device_torch, dtype=dtype)
            
            # Expand noise to match number of images
            noise_expanded = noise.expand(num_images, -1, -1, -1)
            
            # Add noise to all latents with the same noise
            timesteps_expanded = timesteps.expand(num_images)
            noisy_latents = noise_scheduler.add_noise(all_latents, noise_expanded, timesteps_expanded)

        # Zero gradients
        self.optimizer.zero_grad()
        noisy_latents.requires_grad = False

        # Encode prompt
        with torch.set_grad_enabled(self.train_config.train_text_encoder):
            prompt_list = []
            for prompt in prompts:
                if isinstance(prompt, tuple):
                    prompt = prompt[0]
                prompt_list.append(prompt)
            
            # Encode once and repeat for all images
            conditional_embeds = self.sd.encode_prompt(prompt_list).to(self.device_torch, dtype=dtype)
            # Expand to match num_images
            conditional_embeds = concat_prompt_embeds([conditional_embeds] * num_images)

        losses = []
        
        # Process each image in the sequence with its corresponding scale as network multiplier
        # We process in chunks to manage memory
        with self.network:
            assert self.network.is_active
            
            for i in range(num_images):
                scale = scales[i].item() * network_weight
                
                # Set network multiplier based on scale
                # Positive scale = enhance this attribute
                # Negative scale = erase this attribute
                self.network.multiplier = scale
                
                # Get the noisy latent and target for this image
                noisy_latent_i = noisy_latents[i:i+1]
                noise_i = noise  # Same noise for all
                timestep_i = timesteps
                
                # Get corresponding conditional embedding
                cond_i = split_prompt_embeds(conditional_embeds, num_images)[i]
                
                # Predict noise
                noise_pred = self.sd.predict_noise(
                    latents=noisy_latent_i.to(self.device_torch, dtype=dtype),
                    conditional_embeddings=cond_i.to(self.device_torch, dtype=dtype),
                    timestep=timestep_i,
                )
                
                # Compute target
                if self.sd.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(noisy_latent_i, noise_i, timestep_i)
                else:
                    target = noise_i
                
                target = target.to(self.device_torch, dtype=dtype)
                
                # Compute MSE loss
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])
                
                # Apply SNR weighting if configured
                if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
                    loss = apply_snr_weight(loss, timestep_i, noise_scheduler, self.train_config.min_snr_gamma)
                
                # Weight by absolute scale (direction is handled by network multiplier)
                loss = loss * abs(scale)
                loss = loss.mean() * loss_jitter_multiplier
                
                loss_float = loss.item()
                losses.append(loss_float)
                
                # Accumulate gradients
                loss.backward()

        # Apply gradients
        optimizer.step()
        lr_scheduler.step()

        # Reset network multiplier
        self.network.multiplier = 1.0

        loss_dict = OrderedDict(
            {'loss': sum(losses) / len(losses) if len(losses) > 0 else 0.0}
        )

        return loss_dict
