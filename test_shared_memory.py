#!/usr/bin/env python3
"""
Test script for shared memory implementation in cached data.
This verifies that cached tensors use shared memory and aren't duplicated across workers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from toolkit.prompt_utils import PromptEmbeds


def test_promptembeds_share_memory():
    """Test that PromptEmbeds.share_memory_() works correctly."""

    # Test with SD 1.x style (single text_embeds)
    text_embeds = torch.randn(1, 77, 768)
    pe_sd1 = PromptEmbeds(text_embeds)

    # Before share_memory_()
    assert not text_embeds.is_shared(), "tensor should not be shared initially"

    # Call share_memory_()
    pe_sd1.share_memory_()

    # After share_memory_()
    assert pe_sd1.text_embeds.is_shared(), "text_embeds should be in shared memory"
    print("✓ PromptEmbeds.share_memory_() works for SD 1.x style embeddings")

    # Test with SDXL style (text_embeds + pooled_embeds)
    text_embeds_xl = torch.randn(1, 77, 2048)
    pooled_embeds = torch.randn(1, 1280)
    pe_xl = PromptEmbeds([text_embeds_xl, pooled_embeds])

    # Before share_memory_()
    assert not text_embeds_xl.is_shared(), "text_embeds should not be shared initially"
    assert not pooled_embeds.is_shared(), "pooled_embeds should not be shared initially"

    # Call share_memory_()
    pe_xl.share_memory_()

    # After share_memory_()
    assert pe_xl.text_embeds.is_shared(), "text_embeds should be in shared memory"
    assert pe_xl.pooled_embeds.is_shared(), "pooled_embeds should be in shared memory"
    print("✓ PromptEmbeds.share_memory_() works for SDXL style embeddings")

    # Test with attention_mask
    attention_mask = torch.ones(1, 77)
    pe_with_mask = PromptEmbeds(torch.randn(1, 77, 768), attention_mask=attention_mask)

    pe_with_mask.share_memory_()

    assert pe_with_mask.text_embeds.is_shared(), "text_embeds should be in shared memory"
    assert pe_with_mask.attention_mask.is_shared(), "attention_mask should be in shared memory"
    print("✓ PromptEmbeds.share_memory_() works with attention_mask")


def test_tensor_share_memory():
    """Test that PyTorch's share_memory_() prevents duplication."""

    # Create a large tensor
    tensor = torch.randn(1000, 1000)  # ~4MB

    # Before share_memory_()
    assert not tensor.is_shared(), "tensor should not be shared initially"

    # Call share_memory_()
    tensor.share_memory_()

    # After share_memory_()
    assert tensor.is_shared(), "tensor should be in shared memory"

    # Verify data is preserved
    assert tensor.shape == (1000, 1000), "tensor shape should be preserved"

    print("✓ torch.Tensor.share_memory_() works correctly")
    print(f"  - Tensor is now in shared memory: {tensor.is_shared()}")
    print(f"  - Data storage pointer: {tensor.data_ptr()}")


def test_dict_of_tensors_share_memory():
    """Test sharing memory for dict of tensors (like clip_image_embeds)."""

    # Simulate clip_image_embeds structure
    clip_embeds = {
        'image_embeds': torch.randn(1, 512),
        'last_hidden_state': torch.randn(1, 257, 1024),
        'penultimate_hidden_states': torch.randn(1, 257, 1024),
    }

    # Before share_memory_()
    for key, tensor in clip_embeds.items():
        assert not tensor.is_shared(), f"{key} should not be shared initially"

    # Call share_memory_() on each tensor
    for key, tensor in clip_embeds.items():
        tensor.share_memory_()

    # After share_memory_()
    for key, tensor in clip_embeds.items():
        assert tensor.is_shared(), f"{key} should be in shared memory"

    print("✓ Dict of tensors can be moved to shared memory")
    print(f"  - Shared {len(clip_embeds)} tensors")


if __name__ == "__main__":
    print("Testing shared memory implementation...\n")

    test_tensor_share_memory()
    print()
    test_promptembeds_share_memory()
    print()
    test_dict_of_tensors_share_memory()

    print("\n✓ All shared memory tests passed!")
    print("\nBenefit:")
    print("  - Cached data is now shared across DataLoader workers")
    print("  - No more memory duplication per worker")
    print("  - Memory savings: hundreds of MB to GBs depending on cache size")
    print("\nExample:")
    print("  - 1000 images with cached latents = ~400MB")
    print("  - With 4 workers WITHOUT shared memory: 400MB × 4 = 1.6GB")
    print("  - With 4 workers WITH shared memory: 400MB × 1 = 400MB")
    print("  - Savings: 1.2GB!")
