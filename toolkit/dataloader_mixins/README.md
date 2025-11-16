# Dataloader Mixins Package

This package contains refactored dataloader mixins, split from the original 2183-line `dataloader_mixins.py` file into focused modules for better organization and maintainability.

## Package Structure

```
toolkit/dataloader_mixins/
├── __init__.py          # Exports all classes for backward compatibility
├── core.py              # Core utility mixins (75 lines)
├── caption.py           # Caption handling (266 lines)
├── bucket.py            # Bucket resolution (168 lines)
├── image.py             # Image processing and augmentation (468 lines)
├── control.py           # Control/conditional images (548 lines)
├── mask.py              # Masks and points of interest (310 lines)
├── caching.py           # Latent and embedding caching (598 lines)
└── README.md            # This file
```

## Module Descriptions

### core.py
Core utility classes and helper functions:
- `Augments`: Augmentation configuration handler
- `ArgBreakMixin`: Argument breaking utility
- Helper functions: `standardize_images()`, `clean_caption()`
- Constants: `transforms_dict`, `img_ext_list`

### caption.py
Caption loading and processing:
- `CaptionMixin`: Dataset-level caption loading
- `CaptionProcessingDTOMixin`: DTO-level caption processing

### bucket.py
Bucket resolution management for variable-sized batches:
- `Bucket`: Bucket data structure
- `BucketsMixin`: Dataset-level bucket management

### image.py
Image loading, processing, and augmentation:
- `ImageProcessingDTOMixin`: Load and process images
- `AugmentationFileItemDTOMixin`: Apply augmentations

### control.py
Control and conditional image handling:
- `InpaintControlFileItemDTOMixin`: Inpainting control
- `ControlFileItemDTOMixin`: General control images (depth, canny, etc.)
- `ClipImageFileItemDTOMixin`: CLIP vision embeddings
- `UnconditionalFileItemDTOMixin`: Unconditional image handling

### mask.py
Mask and spatial guidance:
- `MaskFileItemDTOMixin`: Focus masks
- `PoiFileItemDTOMixin`: Points of interest for cropping

### caching.py
Caching for latents, text embeddings, and CLIP:
- `LatentCachingFileItemDTOMixin`: DTO-level latent caching
- `LatentCachingMixin`: Dataset-level latent caching
- `TextEmbeddingFileItemDTOMixin`: DTO-level text embedding caching
- `TextEmbeddingCachingMixin`: Dataset-level text embedding caching
- `CLIPCachingMixin`: CLIP vision embedding caching
- `ControlCachingMixin`: Control image caching

## Usage

### Backward Compatible (Recommended)

Existing code continues to work without changes:

```python
# Import from main module (works exactly as before)
from toolkit.dataloader_mixins import (
    CaptionMixin,
    LatentCachingMixin,
    ImageProcessingDTOMixin
)
```

### Direct Module Imports (Optional)

For more explicit imports, you can import from specific modules:

```python
# Import from specific submodules
from toolkit.dataloader_mixins.caption import CaptionMixin
from toolkit.dataloader_mixins.caching import LatentCachingMixin
from toolkit.dataloader_mixins.image import ImageProcessingDTOMixin
```

Both approaches are equivalent - use whichever you prefer!

## Migration Guide

**No migration needed!** The refactoring maintains 100% backward compatibility.

- `toolkit/dataloader_mixins.py` still exists and re-exports all classes
- Existing imports work without any changes
- All functionality is preserved

## Benefits of Refactoring

1. **Better organization**: Related functionality grouped together
2. **Easier maintenance**: Smaller, focused modules (75-598 lines vs 2183 lines)
3. **Improved navigation**: Find specific mixins faster
4. **Better testing**: Test individual modules in isolation
5. **Clear separation of concerns**: Each module has a single responsibility

## Development

When adding new mixins:

1. Choose the appropriate module based on functionality
2. Add the mixin class to that module
3. Export it in `__init__.py` for backward compatibility
4. Update this README

## Original File

The original monolithic file is backed up as:
```
toolkit/dataloader_mixins.py.backup
```

## Refactoring Details

- Original file: 2183 lines, 20 classes
- Refactored: 7 modules, 20 classes (same functionality)
- Reduction: Each module is 75-598 lines (avg ~305 lines)
- Completed: [Date of refactoring]
- Related TODO: #7 - Refactor Dataloader Mixins
