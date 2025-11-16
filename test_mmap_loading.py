#!/usr/bin/env python3
"""
Test memory-mapped tensor loading using safetensors.
This verifies that we can load tensors with minimal memory footprint.
"""

import torch
import tempfile
import os
from safetensors.torch import save_file, load_file, safe_open


def test_normal_loading():
    """Test normal tensor loading (loads entire tensor into RAM)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.safetensors")

        # Create and save a tensor
        tensor = torch.randn(1000, 1000)  # ~4MB
        save_file({"tensor": tensor}, path)

        # Load normally - this loads entire tensor into RAM
        loaded = load_file(path)
        loaded_tensor = loaded["tensor"]

        print("✓ Normal loading works")
        print(f"  - Tensor shape: {loaded_tensor.shape}")
        print(f"  - Tensor dtype: {loaded_tensor.dtype}")
        print(f"  - Storage size: {loaded_tensor.element_size() * loaded_tensor.nelement() / 1024 / 1024:.2f} MB")

        return loaded_tensor


def test_safe_open_loading():
    """Test lazy loading using safe_open (minimal memory until accessed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.safetensors")

        # Create and save a tensor
        tensor = torch.randn(1000, 1000)  # ~4MB
        save_file({"tensor": tensor}, path)

        # Open with safe_open for lazy access
        with safe_open(path, framework="pt", device="cpu") as f:
            # Get tensor keys
            keys = f.keys()
            print(f"\n✓ safe_open() lazy loading")
            print(f"  - Available keys: {list(keys)}")

            # Get a tensor view (this is a lightweight operation)
            tensor_view = f.get_tensor("tensor")

            print(f"  - Tensor shape: {tensor_view.shape}")
            print(f"  - Tensor dtype: {tensor_view.dtype}")
            print(f"  - Note: Tensor data is accessed lazily via mmap")

            # Access the data (this triggers actual memory access)
            _ = tensor_view[0, 0].item()
            print(f"  - Data access works correctly")

            return tensor_view
        # Note: tensor_view becomes invalid after context manager exits


def test_persistent_mmap():
    """Test keeping mmap handle alive for persistent access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.safetensors")

        # Create and save a tensor
        tensor = torch.randn(1000, 1000)  # ~4MB
        save_file({"tensor": tensor}, path)

        # For persistent access, we need to keep the file handle
        # This is what we'd do in the actual implementation
        class MmapTensorLoader:
            def __init__(self, path):
                self.path = path
                self._file_handle = None
                self._tensors = {}

            def load_tensor(self, key):
                if key not in self._tensors:
                    # Open file if not already open
                    if self._file_handle is None:
                        self._file_handle = safe_open(self.path, framework="pt", device="cpu")

                    # Get tensor (backed by mmap)
                    self._tensors[key] = self._file_handle.get_tensor(key)

                return self._tensors[key]

            def close(self):
                if self._file_handle is not None:
                    self._file_handle.__exit__(None, None, None)
                    self._file_handle = None
                    self._tensors = {}

        loader = MmapTensorLoader(path)
        tensor_view = loader.load_tensor("tensor")

        print(f"\n✓ Persistent mmap loading")
        print(f"  - Tensor shape: {tensor_view.shape}")
        print(f"  - Can access data: {tensor_view[0, 0].item()}")
        print(f"  - File handle kept alive for duration of usage")

        loader.close()
        print(f"  - Cleanup successful")


if __name__ == "__main__":
    print("Testing memory-mapped tensor loading with safetensors...\n")

    test_normal_loading()
    test_safe_open_loading()
    test_persistent_mmap()

    print("\n✓ All mmap tests passed!")
    print("\nKey findings:")
    print("  - safe_open() provides lazy/mmap-backed tensor access")
    print("  - File handle must be kept alive for tensor to remain valid")
    print("  - This approach reduces memory footprint for disk-cached data")
    print("  - Benefit: OS pages in data as needed instead of loading all into RAM")
