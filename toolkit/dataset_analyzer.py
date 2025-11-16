#!/usr/bin/env python3
"""
Simple dataset analyzer for the UI
Scans a folder of images and returns statistics
"""

import os
import sys
from pathlib import Path
from collections import Counter
from PIL import Image


def analyze_dataset(folder_path):
    """
    Analyze a dataset folder and return statistics

    Args:
        folder_path: Path to the dataset folder

    Returns:
        dict with dataset statistics or None if no images found
    """
    folder = Path(folder_path)

    if not folder.exists():
        return None

    # Image extensions to look for
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    caption_exts = {'.txt', '.caption', '.cap'}

    # Counters
    resolutions = Counter()
    formats = Counter()
    total_images = 0
    caption_files = Counter()

    # Scan directory
    for item in folder.iterdir():
        if item.is_file():
            ext = item.suffix.lower()

            # Check if it's an image
            if ext in image_exts:
                try:
                    with Image.open(item) as img:
                        width, height = img.size
                        resolutions[(width, height)] += 1
                        formats[ext] += 1
                        total_images += 1
                except Exception as e:
                    # Skip corrupted images
                    print(f"Warning: Could not read {item}: {e}", file=sys.stderr)
                    continue

            # Check for caption files
            elif ext in caption_exts:
                caption_files[ext] += 1

    if total_images == 0:
        return None

    # Find most common resolution
    most_common_res = resolutions.most_common(1)[0][0] if resolutions else (0, 0)

    # Determine if captions exist and what extension
    has_captions = len(caption_files) > 0
    caption_ext = caption_files.most_common(1)[0][0][1:] if has_captions else ""  # Remove leading dot

    return {
        'total_images': total_images,
        'most_common_resolution': list(most_common_res),
        'resolutions': dict(resolutions),
        'formats': dict(formats),
        'has_captions': has_captions,
        'caption_ext': caption_ext,
    }


if __name__ == '__main__':
    # When run as a script, expect folder path as argument
    if len(sys.argv) < 2:
        print("Usage: python dataset_analyzer.py <folder_path>")
        sys.exit(1)

    import json

    folder_path = sys.argv[1]
    result = analyze_dataset(folder_path)

    if result:
        # Convert tuple keys to strings for JSON
        result['resolutions'] = {f"{k[0]}x{k[1]}": v for k, v in result['resolutions'].items()}
        print(json.dumps(result))
    else:
        print(json.dumps({'error': 'No images found'}))
