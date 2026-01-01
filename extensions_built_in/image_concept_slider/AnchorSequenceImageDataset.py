"""
Extended SequenceImageDataset with anchor image support.

This subclass adds the ability to load anchor images from the dataset using
suffix-based matching (e.g., _anchor1, _anchor2). Anchor images are used to
constrain the slider training so it doesn't affect unrelated parts of the image.
"""

import os
from typing import List, Dict, Optional, Any
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms

from toolkit.data_loader import SequenceImageDataset, get_bucket_for_image_size
from toolkit.print import print_acc


class AnchorSequenceImageDataset(SequenceImageDataset):
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
        # Get base data from parent
        images_tensor, scales, prompt, network_weight = super().__getitem__(index)
        
        # Load anchor images if present
        sequence = self.sequences[index]
        anchor_images = None
        anchor_files = sequence.get('anchor_files', [])
        if anchor_files:
            anchor_list = []
            for anchor_path in anchor_files:
                try:
                    img = exif_transpose(Image.open(anchor_path)).convert('RGB')
                    
                    # Use same resize logic as parent
                    bucket_resolution = get_bucket_for_image_size(
                        width=img.width,
                        height=img.height,
                        resolution=self.size,
                    )
                    
                    if bucket_resolution['width'] > bucket_resolution['height']:
                        scale_to_height = bucket_resolution["height"]
                        scale_to_width = int(img.width * (bucket_resolution["height"] / img.height))
                    else:
                        scale_to_width = bucket_resolution["width"]
                        scale_to_height = int(img.height * (bucket_resolution["width"] / img.width))
                    
                    img = img.resize((scale_to_width, scale_to_height), Image.BICUBIC)
                    img = transforms.CenterCrop((bucket_resolution["height"], bucket_resolution["width"]))(img)
                    
                    img_tensor = self.transform(img)
                    anchor_list.append(img_tensor)
                except Exception as e:
                    print_acc(f"Warning: Error loading anchor image {anchor_path}: {e}")
            
            if anchor_list:
                anchor_images = torch.stack(anchor_list, dim=0)  # [num_anchors, C, H, W]
        
        return images_tensor, scales, prompt, network_weight, anchor_images
