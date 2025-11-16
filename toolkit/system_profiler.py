#!/usr/bin/env python3
"""
System Profiler for AI Toolkit

Detects system hardware information for the Guided Config Wizard.
"""

import json
import os
import platform
import subprocess
import sys

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def detect_gpu():
    """Detect GPU information."""
    gpu_info = {
        'type': 'cpu_only',
        'name': 'No GPU detected',
        'vramGB': 0,
        'driverVersion': None,
        'isUnifiedMemory': False
    }

    # Check for Apple Silicon (unified memory)
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        gpu_info['type'] = 'unified_memory'
        gpu_info['name'] = 'Apple Silicon'
        gpu_info['isUnifiedMemory'] = True
        # VRAM will be set from unified memory
        return gpu_info

    # Try NVIDIA
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                if len(parts) >= 2:
                    gpu_info['type'] = 'nvidia'
                    gpu_info['name'] = parts[0].strip()
                    vram_mb = int(float(parts[1].strip()))
                    gpu_info['vramGB'] = vram_mb // 1024
                    if len(parts) >= 3:
                        gpu_info['driverVersion'] = parts[2].strip()
                    return gpu_info
    except Exception:
        pass

    # Try AMD ROCm
    try:
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--csv'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info['type'] = 'amd'
            gpu_info['name'] = 'AMD GPU (ROCm)'
            # Parse VRAM from rocm-smi output
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if 'Total' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        vram_bytes = int(parts[1].strip())
                        gpu_info['vramGB'] = vram_bytes // (1024 * 1024 * 1024)
            return gpu_info
    except Exception:
        pass

    return gpu_info


def detect_memory():
    """Detect system memory information."""
    memory_info = {
        'totalRAM': 0,
        'availableRAM': 0,
        'unifiedMemory': None
    }

    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        memory_info['totalRAM'] = mem.total // (1024 ** 3)
        memory_info['availableRAM'] = mem.available // (1024 ** 3)
    else:
        # Fallback: try to read from /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        memory_info['totalRAM'] = kb // (1024 * 1024)
                    elif line.startswith('MemAvailable:'):
                        kb = int(line.split()[1])
                        memory_info['availableRAM'] = kb // (1024 * 1024)
        except Exception:
            pass

    # For Apple Silicon, unified memory is total RAM
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        memory_info['unifiedMemory'] = memory_info['totalRAM']

    return memory_info


def detect_storage():
    """Detect storage type and available space."""
    storage_info = {
        'type': 'ssd',  # Default to SSD
        'availableSpaceGB': 0
    }

    # Get available space
    if HAS_PSUTIL:
        disk = psutil.disk_usage('/')
        storage_info['availableSpaceGB'] = disk.free // (1024 ** 3)
    else:
        try:
            statvfs = os.statvfs('/')
            storage_info['availableSpaceGB'] = (statvfs.f_frsize * statvfs.f_bavail) // (1024 ** 3)
        except Exception:
            pass

    # Try to detect storage type on Linux
    if sys.platform.startswith('linux'):
        try:
            result = subprocess.run(
                ['lsblk', '-d', '-o', 'name,rota'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        if parts[1] == '0':  # Non-rotational
                            if 'nvme' in parts[0]:
                                storage_info['type'] = 'nvme'
                            else:
                                storage_info['type'] = 'ssd'
                            break
                        else:
                            storage_info['type'] = 'hdd'
                            break
        except Exception:
            pass

    return storage_info


def detect_cpu():
    """Detect CPU information."""
    cpu_info = {
        'cores': os.cpu_count() or 1,
        'name': 'Unknown CPU'
    }

    # Try to get CPU name
    if platform.system() == 'Darwin':
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                cpu_info['name'] = result.stdout.strip()
        except Exception:
            pass
    elif sys.platform.startswith('linux'):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_info['name'] = line.split(':')[1].strip()
                        break
        except Exception:
            pass
    else:
        cpu_info['name'] = platform.processor() or 'Unknown CPU'

    return cpu_info


def get_system_profile():
    """Get complete system profile."""
    gpu_info = detect_gpu()
    memory_info = detect_memory()
    storage_info = detect_storage()
    cpu_info = detect_cpu()

    # For unified memory systems (Apple Silicon, DGX Spark, Grace), set GPU VRAM to unified memory
    if gpu_info['type'] == 'unified_memory' and memory_info['unifiedMemory']:
        gpu_info['vramGB'] = memory_info['unifiedMemory']

    return {
        'gpu': gpu_info,
        'memory': memory_info,
        'storage': storage_info,
        'cpu': cpu_info
    }


def main():
    """Main entry point."""
    try:
        profile = get_system_profile()
        print(json.dumps(profile, indent=2))
        return 0
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
