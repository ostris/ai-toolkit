"""
Platform-specific metric collection for different system types.
Supports: macOS (unified memory), Linux, NVIDIA discrete GPUs.
"""

import os
import platform
import subprocess
import time
from typing import Dict, Any, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_system_metrics() -> Dict[str, Any]:
    """
    Collect system memory and CPU metrics.
    Returns platform-appropriate metrics.
    """
    system = platform.system()

    metrics = {
        'total_memory_gb': 0.0,
        'used_memory_gb': 0.0,
        'available_memory_gb': 0.0,
        'swap_used_gb': 0.0,
        'cpu_utilization_percent': 0.0,
    }

    if HAS_PSUTIL:
        # Use psutil if available (most reliable)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        metrics['total_memory_gb'] = mem.total / (1024 ** 3)
        metrics['used_memory_gb'] = mem.used / (1024 ** 3)
        metrics['available_memory_gb'] = mem.available / (1024 ** 3)
        metrics['swap_used_gb'] = swap.used / (1024 ** 3)
        metrics['cpu_utilization_percent'] = psutil.cpu_percent(interval=0.1)
    elif system == 'Darwin':
        # macOS fallback using vm_stat
        metrics.update(_get_macos_memory())
    elif system == 'Linux':
        # Linux fallback using /proc
        metrics.update(_get_linux_memory())

    return metrics


def _get_macos_memory() -> Dict[str, float]:
    """Get memory info on macOS using vm_stat."""
    metrics = {}

    try:
        # Get total memory using sysctl
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            total_bytes = int(result.stdout.strip())
            metrics['total_memory_gb'] = total_bytes / (1024 ** 3)

        # Get memory usage using vm_stat
        result = subprocess.run(
            ['vm_stat'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            page_size = 16384  # Default for Apple Silicon

            # Parse page size from first line
            if 'page size' in lines[0]:
                page_size = int(lines[0].split()[-2])

            stats = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().rstrip('.')
                    if value.isdigit():
                        stats[key] = int(value)

            # Calculate used memory
            wired = stats.get('Pages wired down', 0) * page_size
            active = stats.get('Pages active', 0) * page_size
            compressed = stats.get('Pages occupied by compressor', 0) * page_size

            used_bytes = wired + active + compressed
            metrics['used_memory_gb'] = used_bytes / (1024 ** 3)
            metrics['available_memory_gb'] = metrics.get('total_memory_gb', 0) - metrics['used_memory_gb']

        # Get swap info
        result = subprocess.run(
            ['sysctl', '-n', 'vm.swapusage'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse: "total = 0.00M  used = 0.00M  free = 0.00M"
            parts = result.stdout.strip().split()
            for i, part in enumerate(parts):
                if part == 'used':
                    # Next part after '=' is the value
                    value_str = parts[i + 2].rstrip('MG')
                    if 'M' in parts[i + 2]:
                        metrics['swap_used_gb'] = float(value_str) / 1024
                    elif 'G' in parts[i + 2]:
                        metrics['swap_used_gb'] = float(value_str)
                    break
    except Exception:
        pass

    return metrics


def _get_linux_memory() -> Dict[str, float]:
    """Get memory info on Linux using /proc/meminfo."""
    metrics = {}

    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value_kb = int(parts[1])
                    meminfo[key] = value_kb

            total_kb = meminfo.get('MemTotal', 0)
            available_kb = meminfo.get('MemAvailable', 0)
            swap_total_kb = meminfo.get('SwapTotal', 0)
            swap_free_kb = meminfo.get('SwapFree', 0)

            metrics['total_memory_gb'] = total_kb / (1024 ** 2)
            metrics['available_memory_gb'] = available_kb / (1024 ** 2)
            metrics['used_memory_gb'] = (total_kb - available_kb) / (1024 ** 2)
            metrics['swap_used_gb'] = (swap_total_kb - swap_free_kb) / (1024 ** 2)
    except Exception:
        pass

    return metrics


def get_process_memory(pid: Optional[int] = None) -> Dict[str, float]:
    """
    Get memory usage for the current process and its children.
    Returns main process memory and total worker memory.
    """
    if pid is None:
        pid = os.getpid()

    metrics = {
        'main_process_memory_gb': 0.0,
        'worker_memory_gb': 0.0,
        'worker_count': 0,
    }

    if not HAS_PSUTIL:
        return metrics

    try:
        main_process = psutil.Process(pid)
        main_memory = main_process.memory_info().rss / (1024 ** 3)
        metrics['main_process_memory_gb'] = main_memory

        # Get child processes (workers)
        children = main_process.children(recursive=True)
        worker_memory = 0.0
        worker_count = 0

        for child in children:
            try:
                child_memory = child.memory_info().rss / (1024 ** 3)
                worker_memory += child_memory
                worker_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        metrics['worker_memory_gb'] = worker_memory
        metrics['worker_count'] = worker_count

    except Exception:
        pass

    return metrics


def get_nvidia_gpu_metrics() -> Dict[str, float]:
    """
    Get NVIDIA GPU metrics using nvidia-smi.
    Returns GPU memory usage and utilization.
    """
    metrics = {
        'gpu_memory_used_gb': 0.0,
        'gpu_utilization_percent': 0.0,
    }

    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            total_memory_mb = 0.0
            total_utilization = 0.0
            gpu_count = 0

            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    memory_mb = float(parts[0].strip())
                    utilization = float(parts[1].strip())
                    total_memory_mb += memory_mb
                    total_utilization += utilization
                    gpu_count += 1

            if gpu_count > 0:
                metrics['gpu_memory_used_gb'] = total_memory_mb / 1024
                metrics['gpu_utilization_percent'] = total_utilization / gpu_count
    except Exception:
        pass

    return metrics


def get_amd_gpu_metrics() -> Dict[str, float]:
    """
    Get AMD GPU metrics using rocm-smi.
    Returns GPU memory usage and utilization.
    """
    metrics = {
        'gpu_memory_used_gb': 0.0,
        'gpu_utilization_percent': 0.0,
    }

    try:
        # Get memory usage
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--csv'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if 'Used' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        used_bytes = int(parts[1].strip())
                        metrics['gpu_memory_used_gb'] = used_bytes / (1024 ** 3)
                        break

        # Get utilization
        result = subprocess.run(
            ['rocm-smi', '--showuse'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'GPU use' in line:
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            metrics['gpu_utilization_percent'] = float(part.rstrip('%'))
                            break
    except Exception:
        pass

    return metrics


def collect_all_metrics(pid: Optional[int] = None, gpu_type: str = 'auto') -> Dict[str, Any]:
    """
    Collect all available metrics for the system.

    Args:
        pid: Process ID to track (defaults to current process)
        gpu_type: 'nvidia', 'amd', 'unified', or 'auto' to detect

    Returns:
        Dictionary with all collected metrics
    """
    # Start with system metrics
    metrics = get_system_metrics()

    # Add process metrics
    process_metrics = get_process_memory(pid)
    metrics.update(process_metrics)

    # Add GPU metrics based on type
    if gpu_type == 'auto':
        # Try NVIDIA first
        gpu_metrics = get_nvidia_gpu_metrics()
        if gpu_metrics['gpu_memory_used_gb'] == 0:
            # Try AMD
            gpu_metrics = get_amd_gpu_metrics()
        metrics.update(gpu_metrics)
    elif gpu_type == 'nvidia':
        metrics.update(get_nvidia_gpu_metrics())
    elif gpu_type == 'amd':
        metrics.update(get_amd_gpu_metrics())
    # For unified memory systems, GPU memory is part of system memory
    # so we don't need separate GPU metrics

    return metrics


def is_unified_memory_system() -> bool:
    """Check if this is a unified memory system (Apple Silicon, Grace, etc.)."""
    system = platform.system()
    processor = platform.processor()

    # Apple Silicon
    if system == 'Darwin' and processor == 'arm':
        return True

    # NVIDIA Grace (aarch64 with unified memory)
    # This is a heuristic - Grace systems are aarch64 but not all aarch64 is Grace
    if platform.machine() == 'aarch64':
        # Check for NVIDIA Grace signature
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'NVIDIA' in cpuinfo or 'Grace' in cpuinfo:
                    return True
        except Exception:
            pass

    return False
