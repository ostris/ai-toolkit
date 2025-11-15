#!/usr/bin/env python3
"""
Test script for the Interactive Config Wizard

This script tests the config wizard functionality without requiring user interaction.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add toolkit to path
sys.path.insert(0, str(Path(__file__).parent))

from toolkit.config_wizard import ConfigWizard


def test_basic_config_generation():
    """Test basic config generation with simulated inputs."""
    print("=" * 70)
    print("TEST: Basic Config Generation")
    print("=" * 70)

    wizard = ConfigWizard()

    # Simulate user inputs
    wizard.answers = {
        'gpu_vram': 24,
        'ram_gb': 64,
        'storage': 'ssd',
        'dataset_size': 1000,
        'resolution': 1024,
        'training_goal': 'LoRA (Low-Rank Adaptation)',
        'epochs': 10,
        'optimization_level': 'Balanced (recommended)',
        'output_path': tempfile.mktemp(suffix='.yaml')
    }

    try:
        # Generate config
        wizard.calculate_optimal_config()

        # Validate config structure
        assert 'job' in wizard.config
        assert 'config' in wizard.config
        assert 'process' in wizard.config['config']

        process = wizard.config['config']['process'][0]
        assert 'train' in process
        assert 'datasets' in process

        train = process['train']
        dataset = process['datasets'][0]

        print("\n✓ Config structure validated")

        # Validate train config
        assert 'batch_size' in train
        assert train['batch_size'] > 0
        assert 'lr' in train
        assert 'steps' in train
        print(f"  Batch size: {train['batch_size']}")
        print(f"  Learning rate: {train['lr']}")
        print(f"  Steps: {train['steps']}")

        # Validate dataset config
        assert 'num_workers' in dataset
        assert dataset['num_workers'] > 0
        print(f"  Workers: {dataset['num_workers']}")

        assert 'cache_latents' in dataset
        assert 'cache_latents_to_disk' in dataset
        print(f"  Cache strategy: {'memory' if dataset['cache_latents'] else 'disk'}")

        # Generate YAML
        yaml_content = wizard.generate_yaml()
        assert len(yaml_content) > 0
        assert 'job:' in yaml_content
        assert 'config:' in yaml_content
        print("\n✓ YAML generation successful")

        # Save config
        wizard.save_config()
        assert Path(wizard.answers['output_path']).exists()
        print(f"\n✓ Config saved to: {wizard.answers['output_path']}")

        # Read back and validate
        with open(wizard.answers['output_path'], 'r') as f:
            content = f.read()
            assert 'batch_size:' in content
            assert 'num_workers:' in content
            print("✓ Config file validated")

        # Cleanup
        Path(wizard.answers['output_path']).unlink()

        print("\n" + "=" * 70)
        print("TEST PASSED: Basic Config Generation")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggressive_optimization():
    """Test aggressive optimization configuration."""
    print("\n" + "=" * 70)
    print("TEST: Aggressive Optimization")
    print("=" * 70)

    wizard = ConfigWizard()

    wizard.answers = {
        'gpu_vram': 48,  # High-end GPU
        'ram_gb': 128,  # Lots of RAM
        'storage': 'nvme',
        'dataset_size': 5000,
        'resolution': 1024,
        'training_goal': 'Full Fine-tuning',
        'epochs': 20,
        'optimization_level': 'Aggressive (maximum performance)',
        'output_path': tempfile.mktemp(suffix='.yaml')
    }

    try:
        wizard.calculate_optimal_config()

        process = wizard.config['config']['process'][0]
        train = process['train']
        dataset = process['datasets'][0]

        # Aggressive should enable auto-scaling
        assert train.get('auto_scale_batch_size', False)
        print(f"✓ Auto-scaling enabled: {train['min_batch_size']}-{train['max_batch_size']}")

        # Should have GPU prefetching
        assert dataset.get('gpu_prefetch_batches', 0) > 0
        print(f"✓ GPU prefetching: {dataset['gpu_prefetch_batches']} batches")

        # Cleanup
        wizard.save_config()
        Path(wizard.answers['output_path']).unlink()

        print("\n" + "=" * 70)
        print("TEST PASSED: Aggressive Optimization")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conservative_optimization():
    """Test conservative optimization configuration."""
    print("\n" + "=" * 70)
    print("TEST: Conservative Optimization")
    print("=" * 70)

    wizard = ConfigWizard()

    wizard.answers = {
        'gpu_vram': 8,  # Low VRAM
        'ram_gb': 16,  # Limited RAM
        'storage': 'hdd',
        'dataset_size': 500,
        'resolution': 512,
        'training_goal': 'DreamBooth',
        'epochs': 5,
        'optimization_level': 'Conservative (safe defaults)',
        'output_path': tempfile.mktemp(suffix='.yaml')
    }

    try:
        wizard.calculate_optimal_config()

        process = wizard.config['config']['process'][0]
        train = process['train']
        dataset = process['datasets'][0]

        # Conservative should not enable auto-scaling
        assert not train.get('auto_scale_batch_size', False)
        print("✓ Auto-scaling disabled (conservative)")

        # Should have gradient checkpointing for low VRAM
        assert train.get('gradient_checkpointing', False)
        print("✓ Gradient checkpointing enabled for low VRAM")

        # Fewer workers for limited RAM
        assert dataset['num_workers'] <= 4
        print(f"✓ Limited workers for RAM conservation: {dataset['num_workers']}")

        # Cleanup
        wizard.save_config()
        Path(wizard.answers['output_path']).unlink()

        print("\n" + "=" * 70)
        print("TEST PASSED: Conservative Optimization")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_large_dataset_disk_cache():
    """Test that large datasets use disk cache instead of memory."""
    print("\n" + "=" * 70)
    print("TEST: Large Dataset Disk Cache")
    print("=" * 70)

    wizard = ConfigWizard()

    wizard.answers = {
        'gpu_vram': 24,
        'ram_gb': 32,  # Limited RAM for large dataset
        'storage': 'ssd',
        'dataset_size': 10000,  # Large dataset
        'resolution': 1024,
        'training_goal': 'LoRA (Low-Rank Adaptation)',
        'epochs': 10,
        'optimization_level': 'Balanced (recommended)',
        'output_path': tempfile.mktemp(suffix='.yaml')
    }

    try:
        wizard.calculate_optimal_config()

        process = wizard.config['config']['process'][0]
        dataset = process['datasets'][0]

        # Large dataset should use disk cache
        assert not dataset['cache_latents']
        assert dataset['cache_latents_to_disk']
        print("✓ Large dataset correctly uses disk cache")

        # Cleanup
        wizard.save_config()
        Path(wizard.answers['output_path']).unlink()

        print("\n" + "=" * 70)
        print("TEST PASSED: Large Dataset Disk Cache")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hardware_detection():
    """Test hardware detection functions."""
    print("\n" + "=" * 70)
    print("TEST: Hardware Detection")
    print("=" * 70)

    wizard = ConfigWizard()

    try:
        # Test GPU detection (may fail if no GPU)
        gpu_name, gpu_vram = wizard.detect_gpu_info()
        if gpu_name:
            print(f"✓ GPU detected: {gpu_name} ({gpu_vram} GB)")
        else:
            print("⚠ GPU not detected (expected if no NVIDIA GPU)")

        # Test RAM detection
        ram_gb = wizard.detect_ram()
        assert ram_gb > 0
        print(f"✓ RAM detected: {ram_gb} GB")

        # Test storage detection
        storage = wizard.detect_storage_type()
        print(f"✓ Storage type detected: {storage}")

        print("\n" + "=" * 70)
        print("TEST PASSED: Hardware Detection")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CONFIG WIZARD - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Hardware Detection", test_hardware_detection),
        ("Basic Config Generation", test_basic_config_generation),
        ("Aggressive Optimization", test_aggressive_optimization),
        ("Conservative Optimization", test_conservative_optimization),
        ("Large Dataset Disk Cache", test_large_dataset_disk_cache),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
