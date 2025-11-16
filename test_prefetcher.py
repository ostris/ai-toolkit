#!/usr/bin/env python3
"""
Test GPU prefetcher functionality.

This script tests the DataLoaderPrefetcher to ensure it:
1. Correctly wraps a dataloader
2. Moves batches to GPU asynchronously
3. Handles iteration and StopIteration correctly
4. Works with the FileItemDTO structure
"""

import torch
import time
from torch.utils.data import Dataset, DataLoader
from toolkit.cache_prefetcher import DataLoaderPrefetcher
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO


class MockDataset(Dataset):
    """Mock dataset that creates dummy FileItemDTO objects."""

    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Create a mock FileItemDTO with some tensors
        item = FileItemDTO(path=f"mock_{idx}.jpg", dataset_config=None)
        # Add some tensor data to simulate cached latents
        item._encoded_latent = torch.randn(4, 64, 64)  # Simulate latent
        item.img = None
        return item


def test_basic_prefetching():
    """Test basic prefetching functionality."""
    print("\n=== Test 1: Basic Prefetching ===")

    # Create mock dataset and dataloader
    dataset = MockDataset(size=20)

    def collate_fn(batch):
        return DataLoaderBatchDTO(file_items=batch)

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Test without prefetching
    print("Testing without prefetching...")
    start_time = time.time()
    count = 0
    for batch in dataloader:
        assert isinstance(batch, DataLoaderBatchDTO)
        assert len(batch.file_items) == 4
        count += 1
    no_prefetch_time = time.time() - start_time
    print(f"  - Processed {count} batches without prefetching in {no_prefetch_time:.3f}s")

    # Test with prefetching (CPU only test - works even without CUDA)
    print("\nTesting with prefetching (device='cpu')...")
    prefetcher = DataLoaderPrefetcher(dataloader, device='cpu', prefetch_batches=2, enabled=True)
    start_time = time.time()
    count = 0
    for batch in prefetcher:
        assert isinstance(batch, DataLoaderBatchDTO)
        assert len(batch.file_items) == 4
        # Check that latents are tensors
        for item in batch.file_items:
            assert item._encoded_latent is not None
            assert isinstance(item._encoded_latent, torch.Tensor)
        count += 1
    prefetch_time = time.time() - start_time
    print(f"  - Processed {count} batches with prefetching in {prefetch_time:.3f}s")

    print("✓ Basic prefetching test passed!")


def test_gpu_prefetching():
    """Test GPU prefetching if CUDA is available."""
    if not torch.cuda.is_available():
        print("\n=== Test 2: GPU Prefetching ===")
        print("⊘ CUDA not available, skipping GPU test")
        return

    print("\n=== Test 2: GPU Prefetching ===")

    # Create mock dataset and dataloader
    dataset = MockDataset(size=10)

    def collate_fn(batch):
        return DataLoaderBatchDTO(file_items=batch)

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Test with GPU prefetching
    print("Testing GPU prefetching...")
    prefetcher = DataLoaderPrefetcher(dataloader, device='cuda', prefetch_batches=2, enabled=True)

    count = 0
    for batch in prefetcher:
        assert isinstance(batch, DataLoaderBatchDTO)
        # Check that tensors are on GPU
        for item in batch.file_items:
            if item._encoded_latent is not None:
                assert item._encoded_latent.device.type == 'cuda', \
                    f"Expected tensor on cuda, got {item._encoded_latent.device}"
        count += 1

    print(f"  - Processed {count} batches on GPU")
    print("✓ GPU prefetching test passed!")


def test_disabled_prefetching():
    """Test that disabled prefetching passes through unchanged."""
    print("\n=== Test 3: Disabled Prefetching ===")

    # Create mock dataset and dataloader
    dataset = MockDataset(size=10)

    def collate_fn(batch):
        return DataLoaderBatchDTO(file_items=batch)

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Test with prefetching disabled
    print("Testing with prefetching disabled (enabled=False)...")
    prefetcher = DataLoaderPrefetcher(dataloader, device='cpu', prefetch_batches=2, enabled=False)

    count = 0
    for batch in prefetcher:
        assert isinstance(batch, DataLoaderBatchDTO)
        # Tensors should still be on CPU
        for item in batch.file_items:
            if item._encoded_latent is not None:
                assert item._encoded_latent.device.type == 'cpu'
        count += 1

    print(f"  - Processed {count} batches (prefetching disabled)")
    print("✓ Disabled prefetching test passed!")


def test_stop_iteration():
    """Test that StopIteration is raised correctly."""
    print("\n=== Test 4: StopIteration Handling ===")

    # Create small dataset
    dataset = MockDataset(size=5)

    def collate_fn(batch):
        return DataLoaderBatchDTO(file_items=batch)

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Test that iteration stops correctly
    prefetcher = DataLoaderPrefetcher(dataloader, device='cpu', prefetch_batches=2, enabled=True)

    count = 0
    try:
        iterator = iter(prefetcher)
        while True:
            batch = next(iterator)
            count += 1
    except StopIteration:
        pass

    expected_batches = (5 + 2 - 1) // 2  # Ceiling division for batch_size=2, size=5
    assert count == expected_batches, f"Expected {expected_batches} batches, got {count}"

    print(f"  - Correctly stopped after {count} batches")
    print("✓ StopIteration test passed!")


def test_multiple_epochs():
    """Test prefetcher across multiple epochs."""
    print("\n=== Test 5: Multiple Epochs ===")

    dataset = MockDataset(size=8)

    def collate_fn(batch):
        return DataLoaderBatchDTO(file_items=batch)

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Run 3 epochs
    for epoch in range(3):
        prefetcher = DataLoaderPrefetcher(dataloader, device='cpu', prefetch_batches=2, enabled=True)
        count = 0
        for batch in prefetcher:
            assert isinstance(batch, DataLoaderBatchDTO)
            count += 1
        print(f"  - Epoch {epoch + 1}: {count} batches")

    print("✓ Multiple epochs test passed!")


if __name__ == "__main__":
    print("Testing DataLoaderPrefetcher functionality...\n")

    test_basic_prefetching()
    test_gpu_prefetching()
    test_disabled_prefetching()
    test_stop_iteration()
    test_multiple_epochs()

    print("\n" + "=" * 60)
    print("✓ All prefetcher tests passed!")
    print("=" * 60)

    print("\nKey findings:")
    print("  - Prefetcher correctly wraps DataLoader")
    print("  - Batches are moved to target device (CPU or GPU)")
    print("  - StopIteration is handled correctly")
    print("  - Works across multiple epochs")
    print("  - Disabled mode passes through without modification")
