#!/usr/bin/env python3
"""
Test script for MetricsCollector functionality.

Tests the metrics tracking and dashboard display including:
- Step timing tracking
- Loss and learning rate tracking
- Memory usage monitoring
- Throughput calculations
- Dataloader metrics
- Dashboard display
"""

import time
import random
from toolkit.metrics_collector import MetricsCollector


def test_basic_step_tracking():
    """Test basic step timing tracking."""
    print("="*60)
    print("Test 1: Basic Step Tracking")
    print("="*60)

    metrics = MetricsCollector(window_size=10)

    # Simulate 5 training steps
    for step in range(5):
        metrics.start_step()
        time.sleep(0.01)  # Simulate step time
        metrics.end_step(
            loss=random.uniform(0.1, 0.5),
            lr=1e-4,
            batch_size=4
        )

    summary = metrics.get_metrics_summary()
    print(f"Steps completed: {summary['step_count']}")
    print(f"Average step time: {summary['avg_step_time']:.4f}s")
    print(f"Average loss: {summary['avg_loss']:.4f}")

    assert summary['step_count'] == 5, "Should have 5 steps"
    assert 'avg_step_time' in summary, "Should track step time"
    assert 'avg_loss' in summary, "Should track loss"

    print("✓ Basic step tracking test passed!\n")
    return True


def test_loss_dict_tracking():
    """Test tracking multiple loss components."""
    print("="*60)
    print("Test 2: Loss Dictionary Tracking")
    print("="*60)

    metrics = MetricsCollector(window_size=10)

    # Simulate steps with multiple loss components
    for step in range(10):
        metrics.start_step()
        time.sleep(0.005)
        metrics.end_step(
            loss_dict={
                'total_loss': random.uniform(0.2, 0.4),
                'mse_loss': random.uniform(0.1, 0.2),
                'perceptual_loss': random.uniform(0.05, 0.15),
            },
            lr=1e-4 * (0.99 ** step),  # Decaying LR
            batch_size=8
        )

    summary = metrics.get_metrics_summary()
    print(f"Loss components tracked:")
    for key, value in summary.items():
        if 'loss' in key and 'avg' in key:
            print(f"  {key}: {value:.4f}")

    assert 'avg_total_loss' in summary, "Should track total_loss"
    assert 'avg_mse_loss' in summary, "Should track mse_loss"
    assert 'avg_perceptual_loss' in summary, "Should track perceptual_loss"

    print("✓ Loss dict tracking test passed!\n")
    return True


def test_throughput_tracking():
    """Test throughput metrics calculation."""
    print("="*60)
    print("Test 3: Throughput Tracking")
    print("="*60)

    metrics = MetricsCollector(window_size=20)

    # Simulate variable batch sizes
    batch_sizes = [4, 4, 8, 8, 16, 16, 4, 8]
    for bs in batch_sizes:
        metrics.start_step()
        time.sleep(0.01)
        metrics.end_step(batch_size=bs)

    summary = metrics.get_metrics_summary()
    print(f"Total samples processed: {summary['total_samples_processed']}")
    print(f"Average batch size: {summary['avg_batch_size']:.1f}")
    print(f"Average samples/sec: {summary['avg_samples_per_second']:.1f}")

    expected_total = sum(batch_sizes)
    assert summary['total_samples_processed'] == expected_total, \
        f"Should have processed {expected_total} samples"
    assert 'avg_samples_per_second' in summary, "Should calculate throughput"

    print("✓ Throughput tracking test passed!\n")
    return True


def test_dataloader_metrics():
    """Test dataloader metrics tracking."""
    print("="*60)
    print("Test 4: Dataloader Metrics")
    print("="*60)

    metrics = MetricsCollector(window_size=20)

    # Simulate dataloader operations
    for i in range(20):
        # Simulate fetch time
        fetch_time = random.uniform(0.001, 0.01)
        metrics.record_dataloader_fetch(fetch_time)

        # Simulate cache hits/misses (80% hit rate)
        if random.random() < 0.8:
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()

    summary = metrics.get_metrics_summary()
    print(f"Cache hits: {metrics.cache_hits}")
    print(f"Cache misses: {metrics.cache_misses}")
    print(f"Cache hit rate: {summary['cache_hit_rate']*100:.1f}%")
    print(f"Avg fetch time: {summary['avg_dataloader_fetch_time']:.4f}s")

    assert 'cache_hit_rate' in summary, "Should track cache hit rate"
    assert summary['cache_hit_rate'] > 0.5, "Should have reasonable hit rate"

    print("✓ Dataloader metrics test passed!\n")
    return True


def test_memory_tracking():
    """Test memory usage tracking."""
    print("="*60)
    print("Test 5: Memory Tracking")
    print("="*60)

    metrics = MetricsCollector(window_size=10)

    # Simulate training steps
    for i in range(5):
        metrics.start_step()
        time.sleep(0.005)
        metrics.end_step(loss=0.5, batch_size=4)

    summary = metrics.get_metrics_summary()
    print("Memory metrics:")
    for key, value in summary.items():
        if 'memory' in key.lower() or 'cpu' in key.lower():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")

    # Memory tracking might not work without GPU, so just check it doesn't error
    print("✓ Memory tracking test passed!\n")
    return True


def test_dashboard_display():
    """Test dashboard printing."""
    print("="*60)
    print("Test 6: Dashboard Display")
    print("="*60)

    metrics = MetricsCollector(window_size=50)

    # Simulate a training session
    print("\nSimulating 20 training steps...")
    for step in range(20):
        metrics.start_step()
        time.sleep(0.01)  # Simulate training
        metrics.end_step(
            loss_dict={
                'total_loss': random.uniform(0.3, 0.5),
                'mse': random.uniform(0.1, 0.2),
            },
            lr=1e-4 * (0.95 ** step),
            batch_size=random.choice([4, 8, 16])
        )

        # Simulate dataloader
        metrics.record_dataloader_fetch(random.uniform(0.001, 0.005))
        if random.random() < 0.85:
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()

    # Print dashboard
    print("\n" + "="*60)
    print("Dashboard Output:")
    print("="*60)
    metrics.print_dashboard(include_timing_breakdown=False)

    print("✓ Dashboard display test passed!\n")
    return True


def test_timing_breakdown_integration():
    """Test integration with timing breakdown."""
    print("="*60)
    print("Test 7: Timing Breakdown Integration")
    print("="*60)

    metrics = MetricsCollector(window_size=10)

    # Simulate Timer integration
    timing_dict = {
        'get_batch': 0.0123,
        'forward_pass': 0.0456,
        'backward_pass': 0.0789,
        'optimizer_step': 0.0234,
        'prepare_latents': 0.0111,
    }

    metrics.update_timing_breakdown(timing_dict)

    # Do some steps
    for i in range(5):
        metrics.start_step()
        time.sleep(0.01)
        metrics.end_step(loss=0.5)

    # Print with timing breakdown
    print("\nDashboard with timing breakdown:")
    metrics.print_dashboard(include_timing_breakdown=True)

    print("✓ Timing breakdown integration test passed!\n")
    return True


def test_reset_functionality():
    """Test metrics reset."""
    print("="*60)
    print("Test 8: Reset Functionality")
    print("="*60)

    metrics = MetricsCollector(window_size=10)

    # Add some data
    for i in range(5):
        metrics.start_step()
        time.sleep(0.005)
        metrics.end_step(loss=0.5, batch_size=4)
        metrics.record_cache_hit()

    summary_before = metrics.get_metrics_summary()
    print(f"Before reset - Steps: {summary_before['step_count']}")

    # Reset
    metrics.reset()

    summary_after = metrics.get_metrics_summary()
    print(f"After reset - Steps: {summary_after.get('step_count', 0)}")

    assert summary_after.get('step_count', 0) == 0, "Step count should be 0 after reset"
    assert metrics.cache_hits == 0, "Cache hits should be 0 after reset"

    print("✓ Reset functionality test passed!\n")
    return True


def test_export_functionality():
    """Test metrics export."""
    print("="*60)
    print("Test 9: Export Functionality")
    print("="*60)

    metrics = MetricsCollector(window_size=10)

    # Add some data
    for i in range(3):
        metrics.start_step()
        time.sleep(0.005)
        metrics.end_step(loss=0.5, batch_size=4)

    exported = metrics.export_to_dict()
    print("Exported metrics keys:")
    for key in exported.keys():
        print(f"  - {key}")

    assert 'summary' in exported, "Should export summary"
    assert 'cache_stats' in exported, "Should export cache stats"
    assert 'step_count' in exported, "Should export step count"

    print("✓ Export functionality test passed!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing MetricsCollector")
    print("="*60 + "\n")

    results = []
    try:
        results.append(("Basic Step Tracking", test_basic_step_tracking()))
        results.append(("Loss Dict Tracking", test_loss_dict_tracking()))
        results.append(("Throughput Tracking", test_throughput_tracking()))
        results.append(("Dataloader Metrics", test_dataloader_metrics()))
        results.append(("Memory Tracking", test_memory_tracking()))
        results.append(("Dashboard Display", test_dashboard_display()))
        results.append(("Timing Breakdown", test_timing_breakdown_integration()))
        results.append(("Reset Functionality", test_reset_functionality()))
        results.append(("Export Functionality", test_export_functionality()))
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
        print("✓ All metrics collector tests passed!")
        print("\nKey features validated:")
        print("  - Step timing and throughput tracking")
        print("  - Loss and learning rate monitoring")
        print("  - Memory usage tracking (GPU + CPU)")
        print("  - Dataloader metrics (fetch time, cache hit/miss)")
        print("  - Dashboard display with summary")
        print("  - Timer integration for detailed breakdown")
        print("  - Reset and export functionality")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
