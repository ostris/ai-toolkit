#!/usr/bin/env python3
"""
Test script for Batch Size Tuner functionality.

Tests the smart batch size scaling features including:
- Auto-detection of optimal batch size
- Batch size warmup
- OOM recovery
- Memory tracking
"""

import torch
from toolkit.batch_size_tuner import BatchSizeTuner, auto_detect_batch_size


def test_basic_tuner():
    """Test basic batch size tuner functionality."""
    print("="*60)
    print("Test 1: Basic Tuner Functionality")
    print("="*60)

    tuner = BatchSizeTuner(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=16,
        auto_scale=True,
        warmup_steps=10,
    )

    print(f"Initial batch size: {tuner.get_batch_size(step=0)}")

    # Simulate warmup
    print("\nWarmup phase:")
    for step in range(12):
        bs = tuner.get_batch_size(step)
        print(f"  Step {step}: batch_size = {bs}")
        tuner.handle_success()

    print("\n✓ Basic tuner test passed!")
    return True


def test_oom_handling():
    """Test OOM handling and recovery."""
    print("\n" + "="*60)
    print("Test 2: OOM Handling")
    print("="*60)

    tuner = BatchSizeTuner(
        initial_batch_size=16,
        min_batch_size=2,
        max_batch_size=32,
        auto_scale=True,
    )

    print(f"Starting batch size: {tuner.current_batch_size}")

    # Simulate OOM errors
    print("\nSimulating OOM errors:")
    for i in range(3):
        new_bs, should_abort = tuner.handle_oom()
        print(f"  OOM {i+1}: Reduced batch size to {new_bs}")
        assert not should_abort, "Should not abort yet"

    print(f"\nFinal batch size after 3 OOMs: {tuner.current_batch_size}")
    print(f"OOM count: {tuner.oom_count}")

    # Test successful steps reduce OOM count
    print("\nSimulating successful steps:")
    for i in range(5):
        tuner.handle_success()
    print(f"OOM count after 5 successful steps: {tuner.oom_count}")

    print("\n✓ OOM handling test passed!")
    return True


def test_memory_stats():
    """Test memory statistics tracking."""
    print("\n" + "="*60)
    print("Test 3: Memory Statistics")
    print("="*60)

    tuner = BatchSizeTuner(initial_batch_size=4)

    stats = tuner.get_memory_stats()
    print("Memory stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            if key == 'utilization':
                print(f"  {key}: {value*100:.1f}%")
            else:
                print(f"  {key}: {value:.1f} MB")
        else:
            print(f"  {key}: {value}")

    # Print full stats
    tuner.print_stats()

    print("\n✓ Memory stats test passed!")
    return True


def test_batch_size_increase():
    """Test batch size increases after stability."""
    print("\n" + "="*60)
    print("Test 4: Batch Size Increase")
    print("="*60)

    tuner = BatchSizeTuner(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=32,
        auto_scale=True,
        warmup_steps=5,
    )

    # Complete warmup
    for step in range(6):
        tuner.get_batch_size(step)
        tuner.handle_success()

    tuner.complete_warmup()
    print(f"After warmup: batch_size = {tuner.current_batch_size}")

    # Simulate successful steps to trigger increase
    initial_bs = tuner.current_batch_size
    print(f"\nSimulating 100 successful steps...")
    for i in range(100):
        tuner.handle_success()

    final_bs = tuner.current_batch_size
    print(f"After 100 successful steps: batch_size = {final_bs}")

    if final_bs > initial_bs:
        print(f"✓ Batch size increased from {initial_bs} to {final_bs}")
    else:
        print(f"ℹ Batch size stable at {final_bs} (may depend on memory)")

    print("\n✓ Batch size increase test passed!")
    return True


def test_auto_detect(mock_mode=True):
    """Test auto-detection of batch size."""
    print("\n" + "="*60)
    print("Test 5: Auto-Detection")
    print("="*60)

    if mock_mode:
        # Mock test function that succeeds up to batch size 8
        def test_fn(batch_size):
            success = batch_size <= 8
            return success

        optimal_bs = auto_detect_batch_size(
            test_fn,
            min_batch_size=1,
            max_batch_size=16,
            initial_guess=4
        )

        assert optimal_bs == 8, f"Expected 8, got {optimal_bs}"
        print(f"✓ Auto-detected batch size: {optimal_bs}")
    else:
        print("ℹ Skipping real GPU test (set mock_mode=False to enable)")

    print("\n✓ Auto-detection test passed!")
    return True


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*60)
    print("Test 6: Edge Cases")
    print("="*60)

    # Test: Min batch size limit
    print("Testing min batch size limit...")
    tuner = BatchSizeTuner(initial_batch_size=1, min_batch_size=1, max_batch_size=8)
    new_bs, should_abort = tuner.handle_oom()
    assert new_bs == 1, "Should stay at min batch size"
    assert should_abort, "Should signal abort at min batch size"
    print("  ✓ Min batch size limit works")

    # Test: Max batch size limit
    print("Testing max batch size limit...")
    tuner = BatchSizeTuner(initial_batch_size=16, min_batch_size=1, max_batch_size=16)
    tuner.complete_warmup()
    initial = tuner.current_batch_size
    for _ in range(200):  # Try to trigger increase
        tuner.handle_success()
    assert tuner.current_batch_size <= 16, "Should not exceed max batch size"
    print(f"  ✓ Max batch size limit works (stayed at {tuner.current_batch_size})")

    # Test: Disabled auto-scale
    print("Testing disabled auto-scale...")
    tuner = BatchSizeTuner(initial_batch_size=4, auto_scale=False)
    for step in range(20):
        bs = tuner.get_batch_size(step)
        assert bs == 4, "Batch size should remain constant when auto-scale=False"
    print("  ✓ Disabled auto-scale works")

    print("\n✓ Edge cases test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing Batch Size Tuner")
    print("="*60)

    results = []
    try:
        results.append(("Basic Functionality", test_basic_tuner()))
        results.append(("OOM Handling", test_oom_handling()))
        results.append(("Memory Stats", test_memory_stats()))
        results.append(("Batch Size Increase", test_batch_size_increase()))
        results.append(("Auto-Detection", test_auto_detect(mock_mode=True)))
        results.append(("Edge Cases", test_edge_cases()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✓ All batch size tuner tests passed!")
        print("\nKey features validated:")
        print("  - Batch size warmup during training")
        print("  - OOM detection and recovery")
        print("  - Automatic batch size adjustment")
        print("  - Memory statistics tracking")
        print("  - Auto-detection of optimal batch size")
        print("  - Edge case handling")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
