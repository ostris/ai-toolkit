"""
Dataloader mixins package - refactored for better organization.

This package splits the original large dataloader_mixins.py file into focused modules:
- core.py: Core utility mixins (Augments, ArgBreakMixin)
- caption.py: Caption handling mixins
- bucket.py: Bucket resolution mixins
- image.py: Image processing and augmentation mixins
- control.py: Control/conditional image mixins
- mask.py: Mask and POI (Point of Interest) mixins
- caching.py: Latent, text embedding, and CLIP caching mixins

All classes are re-exported here for backward compatibility.
Existing imports like `from toolkit.dataloader_mixins import CaptionMixin` will continue to work.
"""

# Import all classes from submodules for backward compatibility
from .core import (
    Augments,
    ArgBreakMixin,
    # Helper data
    transforms_dict,
    img_ext_list,
    # Helper functions
    standardize_images,
    clean_caption,
)

from .caption import (
    CaptionMixin,
    CaptionProcessingDTOMixin,
)

from .bucket import (
    Bucket,
    BucketsMixin,
)

from .image import (
    ImageProcessingDTOMixin,
    AugmentationFileItemDTOMixin,
)

from .control import (
    InpaintControlFileItemDTOMixin,
    ControlFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
)

from .mask import (
    MaskFileItemDTOMixin,
    PoiFileItemDTOMixin,
)

from .caching import (
    LatentCachingFileItemDTOMixin,
    LatentCachingMixin,
    TextEmbeddingFileItemDTOMixin,
    TextEmbeddingCachingMixin,
    CLIPCachingMixin,
    ControlCachingMixin,
)

# Export all for "from toolkit.dataloader_mixins import *"
__all__ = [
    # Core
    'Augments',
    'ArgBreakMixin',
    'transforms_dict',
    'img_ext_list',
    'standardize_images',
    'clean_caption',
    # Caption
    'CaptionMixin',
    'CaptionProcessingDTOMixin',
    # Bucket
    'Bucket',
    'BucketsMixin',
    # Image
    'ImageProcessingDTOMixin',
    'AugmentationFileItemDTOMixin',
    # Control
    'InpaintControlFileItemDTOMixin',
    'ControlFileItemDTOMixin',
    'ClipImageFileItemDTOMixin',
    'UnconditionalFileItemDTOMixin',
    # Mask
    'MaskFileItemDTOMixin',
    'PoiFileItemDTOMixin',
    # Caching
    'LatentCachingFileItemDTOMixin',
    'LatentCachingMixin',
    'TextEmbeddingFileItemDTOMixin',
    'TextEmbeddingCachingMixin',
    'CLIPCachingMixin',
    'ControlCachingMixin',
]
