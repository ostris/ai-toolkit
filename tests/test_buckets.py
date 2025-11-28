"""
Tests for toolkit/buckets.py
"""

import pytest
from toolkit.buckets import (
    get_bucket_sizes,
    get_resolution,
    get_bucket_for_image_size,
    BucketResolution,
)


class TestBuckets:
    def test_get_bucket_size(self):
        # Test rounding bug
        #
        # For resolution = 585
        #     scaler = 585/1024 = 0.5712890625
        #
        # For bucket [896, 1088]:
        #     896 * 0.5712890625 = 511.875
        #
        #     int(511.875) = 511
        #     511 % 8 = 7
        #     Adjusted: 511 - 7 = 504
        #
        #     round(511.875) = 512
        #     512 % 8 = 0
        #     Adjusted: 512 - 0 = 512
        #
        #     1088 * 0.5712890625 = 621.5625 -> 624
        buckets = get_bucket_sizes(resolution=585, divisibility=8)
        assert({"width": 616, "height": 504} not in buckets)
        assert({"width": 624, "height": 512} in buckets)

    def test_resolution(self):
        # Test rounding bug
        resolution = get_resolution(534, 640)
        assert(resolution == 585)

    def test_get_bucket_for_image_size(self):
        # Test rounding bug
        bucket = get_bucket_for_image_size(
            896, 1088, None, 585, 8)
        assert(bucket == {"width": 512, "height": 624})


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
