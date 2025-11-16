#!/usr/bin/env python3
"""
Script to refactor dataloader_mixins.py into multiple focused modules.

This splits the large 2183-line file into smaller,  more maintainable modules.
"""

import re
import os

def extract_classes_from_file(filepath):
    """Parse the file and extract class definitions with their line ranges."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    classes = []
    current_class = None
    class_indent = None

    for i, line in enumerate(lines):
        # Detect class definition
        if re.match(r'^class \w+', line):
            # Save previous class if exists
            if current_class:
                current_class['end_line'] = i
                classes.append(current_class)

            # Start new class
            class_name = re.search(r'^class (\w+)', line).group(1)
            current_class = {
                'name': class_name,
                'start_line': i,
                'end_line': None
            }
            class_indent = 0

        # Detect function definitions (to track class boundaries)
        elif current_class and re.match(r'^(class |def \w+|[a-zA-Z_])', line):
            # If we hit a non-indented line that's not part of class, end the class
            if not line.startswith(' ') and not line.startswith('class') and line.strip():
                current_class['end_line'] = i
                classes.append(current_class)
                current_class = None

    # Don't forget the last class
    if current_class:
        current_class['end_line'] = len(lines)
        classes.append(current_class)

    return classes, lines

def categorize_class(class_name):
    """Categorize a class into its appropriate module."""
    if class_name in ['Augments', 'ArgBreakMixin']:
        return 'core'
    elif class_name in ['CaptionMixin', 'CaptionProcessingDTOMixin']:
        return 'caption'
    elif class_name in ['Bucket', 'BucketsMixin']:
        return 'bucket'
    elif class_name in ['ImageProcessingDTOMixin', 'AugmentationFileItemDTOMixin']:
        return 'image'
    elif class_name in ['InpaintControlFileItemDTOMixin', 'ControlFileItemDTOMixin',
                        'ClipImageFileItemDTOMixin', 'UnconditionalFileItemDTOMixin']:
        return 'control'
    elif class_name in ['MaskFileItemDTOMixin', 'PoiFileItemDTOMixin']:
        return 'mask'
    elif 'Caching' in class_name or 'LatentCaching' in class_name or 'TextEmbedding' in class_name:
        return 'caching'
    else:
        return 'misc'

def get_common_header():
    """Return the common imports header for all modules."""
    return '''"""
Dataloader mixins - refactored for better organization.
"""

import base64
import glob
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Dict, Union
import traceback

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor

from toolkit.basic import flush, value_map
from toolkit.buckets import get_bucket_for_image_size, get_resolution
from toolkit.config_modules import ControlTypes
from toolkit.control_generator import ControlGenerator
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.pixtral_vision import PixtralVisionImagePreprocessorCompatible
from toolkit.prompt_utils import inject_trigger_into_prompt
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageOps import exif_transpose
import albumentations as A
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator
from toolkit.prompt_utils import PromptEmbeds
from torchvision.transforms import functional as TF
from toolkit.train_tools import get_torch_dtype

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.stable_diffusion_model import StableDiffusion

accelerator = get_accelerator()

'''

def main():
    source_file = 'toolkit/dataloader_mixins.py'
    output_dir = 'toolkit/dataloader_mixins'

    print(f"Parsing {source_file}...")
    classes, lines = extract_classes_from_file(source_file)

    print(f"Found {len(classes)} classes")
    for cls in classes:
        print(f"  - {cls['name']}: lines {cls['start_line']}-{cls['end_line']}")

    # Group classes by category
    categorized = {}
    for cls in classes:
        category = categorize_class(cls['name'])
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(cls)

    print(f"\nGrouped into {len(categorized)} modules:")
    for category, cls_list in categorized.items():
        print(f"  - {category}.py: {[c['name'] for c in cls_list]}")

    # Extract helper functions and constants (lines before first class)
    first_class_line = min(c['start_line'] for c in classes)
    header_lines = lines[:first_class_line]

    # Find where imports end
    import_end = 0
    for i, line in enumerate(header_lines):
        if line.strip() and not line.startswith('import') and not line.startswith('from') and not line.startswith('#'):
            if 'accelerator' in line or '=' in line or 'def ' in line or 'class ' in line:
                import_end = i
                break

    # Get helper functions/constants
    helpers = ''.join(header_lines[import_end:])

    # Now create module files
    os.makedirs(output_dir, exist_ok=True)

    for category, cls_list in categorized.items():
        output_file = os.path.join(output_dir, f'{category}.py')

        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(get_common_header())

            # Write helpers if this is core module
            if category == 'core':
                f.write(helpers)

            # Write class definitions
            for cls in cls_list:
                start = cls['start_line']
                end = cls['end_line']
                f.write('\n')
                f.write(''.join(lines[start:end]))

        print(f"\nCreated {output_file}")

    print("\nRefactoring complete!")
    print("Next step: Create __init__.py to export all classes for backward compatibility")

if __name__ == '__main__':
    main()
