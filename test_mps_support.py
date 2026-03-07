#!/usr/bin/env python3
"""
Test script to verify MPS (Metal Performance Shaders) support in the AI Toolkit.
This script checks if MPS is available and tests basic functionality.
"""

import sys
import torch
from toolkit.device_utils import get_optimal_device, get_device_name, empty_cache, get_device_memory_info

def test_mps_support():
    """Test MPS availability and basic functionality."""
    print("🔍 Testing AI Toolkit MPS Support")
    print("=" * 50)

    # Check PyTorch version
    print(f"📦 PyTorch Version: {torch.__version__}")

    # Test device detection
    print("\n🎯 Device Detection:")
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    print(f"   CUDA Available: {'✅' if cuda_available else '❌'}")
    print(f"   MPS Available:  {'✅' if mps_available else '❌'}")

    # Test optimal device selection
    optimal_device = get_optimal_device()
    device_name = get_device_name()
    print(f"   Optimal Device: {optimal_device} ({device_name})")

    if mps_available:
        print("\n🍎 Apple Silicon MPS Support: ✅")

        # Test basic tensor operations on MPS
        try:
            device = get_optimal_device('mps')
            print(f"\n🧪 Testing MPS Operations on {device}:")

            # Test tensor creation and basic operations
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.matmul(x, y)

            print(f"   ✅ Tensor creation: {x.shape}")
            print(f"   ✅ Matrix multiplication: {z.shape}")
            print(f"   ✅ Device placement: {z.device}")

            # Test gradient computation
            x.requires_grad_(True)
            loss = (x.sum() ** 2)
            loss.backward()
            print(f"   ✅ Gradient computation: {x.grad is not None}")

            # Test memory info
            memory_info = get_device_memory_info(device)
            print(f"   📊 Memory info: {memory_info}")

            # Test cache clearing
            empty_cache(device)
            print("   ✅ Cache clearing")

        except Exception as e:
            print(f"   ❌ MPS Operation Error: {e}")
            return False

    else:
        print("\n🍎 Apple Silicon MPS Support: ❌")
        print("   Note: MPS is only available on Apple Silicon Macs (M1/M2/M3/M4)")

    # Test fallback behavior
    print("\n🔄 Testing Device Fallback:")

    # Test with invalid device
    fallback_device = get_optimal_device('cuda:999')  # Invalid CUDA device
    print(f"   Invalid device fallback: {fallback_device}")

    # Test device string normalization
    from toolkit.device_utils import normalize_device_string
    test_strings = ['cuda', 'mps', 'metal', 'apple', 'cpu', 'gpu']
    print("   Device string normalization:")
    for s in test_strings:
        normalized = normalize_device_string(s)
        print(f"     '{s}' -> '{normalized}'")

    print("\n🎉 Test Complete!")

    if mps_available:
        print("✅ Your AI Toolkit installation supports Apple Silicon MPS acceleration!")
        print("💡 You can use 'device: mps' in your configuration files.")
    elif cuda_available:
        print("✅ CUDA support detected. Use 'device: cuda' in your configuration.")
    else:
        print("⚠️  No GPU acceleration detected. CPU-only mode.")

    return True

def test_basic_training_components():
    """Test basic training components with MPS."""
    print("\n🏋️ Testing Training Components:")

    try:
        from toolkit.style import ContentLoss, StyleLoss, get_style_model_and_losses
        from toolkit.util.mask import generate_random_mask

        device = get_optimal_device()
        print(f"   Using device: {device}")

        # Test ContentLoss
        content_loss = ContentLoss(device=device)
        print("   ✅ ContentLoss initialization")

        # Test StyleLoss
        style_loss = StyleLoss(device=device)
        print("   ✅ StyleLoss initialization")

        # Test mask generation
        masks = generate_random_mask(batch_size=2, height=64, width=64, device=device)
        print(f"   ✅ Random mask generation: {masks.shape}")

        # Test basic tensor operations that would be common in training
        x = torch.randn(1, 3, 64, 64, device=device)
        x_fft = torch.fft.fft2(x.float(), dim=(-2, -1))
        print("   ✅ FFT operations")

        return True

    except Exception as e:
        print(f"   ❌ Training component error: {e}")
        return False

if __name__ == "__main__":
    success = test_mps_support()
    success = test_basic_training_components() and success

    if success:
        print("\n🎊 All tests passed! Your setup is ready for training.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check your installation.")
        sys.exit(1)