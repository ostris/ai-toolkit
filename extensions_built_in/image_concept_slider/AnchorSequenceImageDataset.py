"""
Extended SequenceImageDataset with anchor image support and ZImage-compatible bucketing.

This subclass adds:
1. Anchor image support (suffix-based matching)
2. Consistent resolution across all images in a sequence
3. Dimensions divisible by 16 for ZImage compatibility
"""

import os
from typing import List, Dict, Optional, Any
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms

from toolkit.data_loader import SequenceImageDataset, get_bucket_for_image_size, RescaleTransform
from toolkit.print import print_acc


def get_zimage_compatible_bucket(width: int, height: int, resolution: int) -> Dict[str, int]:
    """
    Get a bucket resolution that is compatible with ZImage's patch requirements.
    ZImage requires latent dimensions divisible by 2, which means image dimensions
    must be divisible by 16 (8x VAE downscaling * 2 patch size).
    """
    bucket = get_bucket_for_image_size(
        width=width,
        height=height,
        resolution=resolution,
        divisibility=16  # Use 16 instead of 8 for ZImage compatibility
    )
    
    # Ensure dimensions are divisible by 16
    bucket_w = bucket['width']
    bucket_h = bucket['height']
    
    if bucket_w % 16 != 0:
        bucket_w = (bucket_w // 16) * 16
    if bucket_h % 16 != 0:
        bucket_h = (bucket_h // 16) * 16
    
    # Ensure minimum size
    bucket_w = max(bucket_w, 64)
    bucket_h = max(bucket_h, 64)
    
    return {'width': bucket_w, 'height': bucket_h}


class FixedResolutionSequenceDataset(SequenceImageDataset):
    """
    A SequenceImageDataset that ensures all images in a sequence have the same resolution.
    Uses the first image to determine bucket, then applies to all images in sequence.
    Also ensures ZImage-compatible dimensions (divisible by 16).
    """
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        images = []
        bucket_resolution = None
        
        # Load and process all images in the sequence
        for i, file_path in enumerate(sequence['files']):
            img = exif_transpose(Image.open(file_path)).convert('RGB')
            
            # Determine bucket resolution from first image only
            if bucket_resolution is None:
                bucket_resolution = get_zimage_compatible_bucket(
                    width=img.width,
                    height=img.height,
                    resolution=self.size,
                )
            
            # Scale to fit bucket
            if bucket_resolution['width'] / bucket_resolution['height'] > img.width / img.height:
                # Image is taller than bucket - scale by width
                scale_to_width = bucket_resolution["width"]
                scale_to_height = int(img.height * (bucket_resolution["width"] / img.width))
            else:
                # Image is wider than bucket - scale by height
                scale_to_height = bucket_resolution["height"]
                scale_to_width = int(img.width * (bucket_resolution["height"] / img.height))
            
            img = img.resize((scale_to_width, scale_to_height), Image.BICUBIC)
            img = transforms.CenterCrop((bucket_resolution["height"], bucket_resolution["width"]))(img)
            
            img_tensor = self.transform(img)
            images.append(img_tensor)
        
        # Stack all images into a single tensor
        images_tensor = torch.stack(images, dim=0)  # Shape: [num_images, C, H, W]
        
        prompt = self.get_prompt_item(index)
        scales = torch.tensor(self.scales, dtype=torch.float32)
        
        return images_tensor, scales, prompt, self.network_weight


class AnchorSequenceImageDataset(FixedResolutionSequenceDataset):
    """
    Extended SequenceImageDataset with anchor image support.
    
    Adds anchor_suffixes configuration for loading anchor images.
    Anchor images are images that should remain unchanged by the slider.
    
    Example dataset structure:
        image1_pos1.jpg  (positive direction)
        image1_neg1.jpg  (negative direction)  
        image1_anchor1.jpg  (anchor - should not change)
        image1.txt  (caption)
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Extract anchor suffixes before parent init
        anchor_suffixes = config.get('anchor_suffixes', [])
        if isinstance(anchor_suffixes, str):
            self.anchor_suffixes = [s.strip() for s in anchor_suffixes.split(',') if s.strip()]
        else:
            self.anchor_suffixes = anchor_suffixes if anchor_suffixes else []
        
        # Initialize parent
        super().__init__(config)
        
        # Now add anchor files to sequences
        if self.anchor_suffixes:
            self._add_anchor_files()
    
    def _add_anchor_files(self):
        """Add anchor file paths to each sequence."""
        supported_exts = ('.jpg', '.jpeg', '.png', '.webp', '.JPEG', '.JPG', '.PNG', '.WEBP')
        
        for sequence in self.sequences:
            base_name = sequence['base_name']
            anchor_files = []
            
            for suffix in self.anchor_suffixes:
                # Try to find anchor file with any supported extension
                anchor_path = None
                for ext in supported_exts:
                    candidate = os.path.join(self.folder_path, f"{base_name}{suffix}{ext}")
                    if os.path.exists(candidate):
                        anchor_path = candidate
                        break
                
                if anchor_path:
                    anchor_files.append(anchor_path)
            
            sequence['anchor_files'] = anchor_files
        
        # Count sequences with anchors
        sequences_with_anchors = sum(1 for s in self.sequences if s.get('anchor_files'))
        print_acc(f"  -  Anchor suffixes: {self.anchor_suffixes}")
        print_acc(f"  -  Sequences with anchor images: {sequences_with_anchors}/{len(self.sequences)}")
    
    def __getitem__(self, index: int):
        """
        Get a sequence with its anchor images.
        
        Returns:
            images: Tensor of shape [num_images, C, H, W]
            scales: Tensor of scales for each image
            prompt: Text prompt for the sequence
            network_weight: Network weight for this dataset
            anchor_images: Tensor of shape [num_anchors, C, H, W] or None
        """
        sequence = self.sequences[index]
        images = []
        bucket_resolution = None
        
        # Load and process all images in the sequence (same as parent)
        for i, file_path in enumerate(sequence['files']):
            img = exif_transpose(Image.open(file_path)).convert('RGB')
            
            # Determine bucket resolution from first image only
            if bucket_resolution is None:
                bucket_resolution = get_zimage_compatible_bucket(
                    width=img.width,
                    height=img.height,
                    resolution=self.size,
                )
            
            # Scale to fit bucket
            if bucket_resolution['width'] / bucket_resolution['height'] > img.width / img.height:
                scale_to_width = bucket_resolution["width"]
                scale_to_height = int(img.height * (bucket_resolution["width"] / img.width))
            else:
                scale_to_height = bucket_resolution["height"]
                scale_to_width = int(img.width * (bucket_resolution["height"] / img.height))
            
            img = img.resize((scale_to_width, scale_to_height), Image.BICUBIC)
            img = transforms.CenterCrop((bucket_resolution["height"], bucket_resolution["width"]))(img)
            
            img_tensor = self.transform(img)
            images.append(img_tensor)
        
        images_tensor = torch.stack(images, dim=0)
        prompt = self.get_prompt_item(index)
        scales = torch.tensor(self.scales, dtype=torch.float32)
        
        # Load anchor images using the SAME bucket resolution
        anchor_images = None
        anchor_files = sequence.get('anchor_files', [])
        if anchor_files and bucket_resolution is not None:
            anchor_list = []
            for anchor_path in anchor_files:
                try:
                    img = exif_transpose(Image.open(anchor_path)).convert('RGB')
                    
                    # Use same bucket resolution as main images
                    if bucket_resolution['width'] / bucket_resolution['height'] > img.width / img.height:
                        scale_to_width = bucket_resolution["width"]
                        scale_to_height = int(img.height * (bucket_resolution["width"] / img.width))
                    else:
                        scale_to_height = bucket_resolution["height"]
                        scale_to_width = int(img.width * (bucket_resolution["height"] / img.height))
                    
                    img = img.resize((scale_to_width, scale_to_height), Image.BICUBIC)
                    img = transforms.CenterCrop((bucket_resolution["height"], bucket_resolution["width"]))(img)
                    
                    img_tensor = self.transform(img)
                    anchor_list.append(img_tensor)
                except Exception as e:
                    print_acc(f"Warning: Error loading anchor image {anchor_path}: {e}")
            
            if anchor_list:
                anchor_images = torch.stack(anchor_list, dim=0)  # [num_anchors, C, H, W]
        
        return images_tensor, scales, prompt, network_weight, anchor_images
