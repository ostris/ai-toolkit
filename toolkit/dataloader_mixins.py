"""
Dataloader mixins - Backward compatibility shim.

This file has been refactored into a package structure for better organization.
All classes and functions are re-exported from toolkit.dataloader_mixins/* submodules.

New modular structure:
- toolkit/dataloader_mixins/core.py: Core utility mixins
- toolkit/dataloader_mixins/caption.py: Caption handling
- toolkit/dataloader_mixins/bucket.py: Bucket resolution
- toolkit/dataloader_mixins/image.py: Image processing
- toolkit/dataloader_mixins/control.py: Control/conditional images
- toolkit/dataloader_mixins/mask.py: Masks and POI
- toolkit/dataloader_mixins/caching.py: Latent and embedding caching

This file maintains backward compatibility - existing imports will continue to work:
    from toolkit.dataloader_mixins import CaptionMixin  # Still works!

For new code, you can import from the package or continue using this file.
Both approaches are equivalent:
    from toolkit.dataloader_mixins import CaptionMixin
    from toolkit.dataloader_mixins.caption import CaptionMixin  # More explicit

Original file backed up as: toolkit/dataloader_mixins.py.backup
"""

# Re-export everything from the package for backward compatibility
from toolkit.dataloader_mixins import *  # noqa: F401, F403

# Explicit re-exports for better IDE support
from toolkit.dataloader_mixins import (
    # Core
    Augments,
    ArgBreakMixin,
    transforms_dict,
    img_ext_list,
    standardize_images,
    clean_caption,
    # Caption
    CaptionMixin,
    CaptionProcessingDTOMixin,
    # Bucket
    Bucket,
    BucketsMixin,
    # Image
    ImageProcessingDTOMixin,
    AugmentationFileItemDTOMixin,
    # Control
    InpaintControlFileItemDTOMixin,
    ControlFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
    # Mask
    MaskFileItemDTOMixin,
    PoiFileItemDTOMixin,
    # Caching
    LatentCachingFileItemDTOMixin,
    LatentCachingMixin,
    TextEmbeddingFileItemDTOMixin,
    TextEmbeddingCachingMixin,
    CLIPCachingMixin,
    ControlCachingMixin,
)

__all__ = [
    'Augments',
    'ArgBreakMixin',
    'transforms_dict',
    'img_ext_list',
    'standardize_images',
    'clean_caption',
    'CaptionMixin',
    'CaptionProcessingDTOMixin',
    'Bucket',
    'BucketsMixin',
    'ImageProcessingDTOMixin',
    'AugmentationFileItemDTOMixin',
    'InpaintControlFileItemDTOMixin',
    'ControlFileItemDTOMixin',
    'ClipImageFileItemDTOMixin',
    'UnconditionalFileItemDTOMixin',
    'MaskFileItemDTOMixin',
    'PoiFileItemDTOMixin',
    'LatentCachingFileItemDTOMixin',
    'LatentCachingMixin',
    'TextEmbeddingFileItemDTOMixin',
    'TextEmbeddingCachingMixin',
    'CLIPCachingMixin',
    'ControlCachingMixin',
]
