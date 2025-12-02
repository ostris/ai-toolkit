import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import path from 'path';
import fs from 'fs';

const execAsync = promisify(exec);

// Get path to bundled macmon binary
function getMacmonPath(): string | null {
  // Check bundled binary in ui/bin folder
  const bundledPath = path.join(process.cwd(), 'bin', 'macmon');
  if (fs.existsSync(bundledPath)) {
    return bundledPath;
  }
  // Fallback to system PATH
  return 'macmon';
}

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';
    const isMac = platform === 'darwin';

    // Check for Apple Silicon Mac first
    if (isMac) {
      const macGpuStats = await getMacGpuStats();
      if (macGpuStats) {
        return NextResponse.json({
          hasNvidiaSmi: true, // Reusing this flag to indicate GPU available
          hasMps: true,
          gpus: [macGpuStats],
        });
      }
    }

    // Check if nvidia-smi is available
    const hasNvidiaSmi = await checkNvidiaSmi(isWindows);

    if (!hasNvidiaSmi) {
      return NextResponse.json({
        hasNvidiaSmi: false,
        gpus: [],
        error: isMac
          ? 'No Apple Silicon GPU detected. MPS may not be available.'
          : 'nvidia-smi not found or not accessible',
      });
    }

    // Get GPU stats
    const gpuStats = await getGpuStats(isWindows);

    return NextResponse.json({
      hasNvidiaSmi: true,
      gpus: gpuStats,
    });
  } catch (error) {
    console.error('Error fetching GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

// Check if macmon is available (bundled or system)
async function hasMacmon(): Promise<string | null> {
  const macmonPath = getMacmonPath();
  if (!macmonPath) return null;

  try {
    // Check if bundled binary exists and is executable
    if (macmonPath !== 'macmon') {
      await execAsync(`"${macmonPath}" --version`);
      return macmonPath;
    }
    // Check system PATH
    await execAsync('which macmon');
    return 'macmon';
  } catch {
    return null;
  }
}

// Get a single sample from macmon
async function getMacmonStats(macmonPath: string): Promise<MacmonData | null> {
  try {
    // Run macmon pipe for a short duration and get one sample
    const { stdout } = await execAsync(`"${macmonPath}" pipe -i 100 2>&1 | head -1`, {
      timeout: 2000,
    });
    const lines = stdout.trim().split('\n');
    if (lines.length > 0) {
      return JSON.parse(lines[lines.length - 1]);
    }
    return null;
  } catch {
    return null;
  }
}

interface MacmonData {
  gpu_usage: [number, number]; // [frequency_mhz, usage_percent]
  gpu_power: number;
  temp: {
    cpu_temp_avg: number;
    gpu_temp_avg: number;
  };
  memory: {
    ram_total: number;
    ram_usage: number;
    swap_total: number;
    swap_usage: number;
  };
  all_power: number;
  sys_power: number;
}

async function getMacGpuStats() {
  try {
    // Check if this is Apple Silicon
    const { stdout: cpuInfo } = await execAsync('sysctl -n machdep.cpu.brand_string');
    const isAppleSilicon = cpuInfo.toLowerCase().includes('apple');

    if (!isAppleSilicon) {
      return null;
    }

    // Try to get real-time stats from macmon first
    const macmonPath = await hasMacmon();
    let macmonData: MacmonData | null = null;
    if (macmonPath) {
      macmonData = await getMacmonStats(macmonPath);
    }

    // Get GPU name from system_profiler
    let gpuName = 'Apple Silicon GPU';
    let gpuCores = 0;
    try {
      const { stdout: gpuInfo } = await execAsync(
        'system_profiler SPDisplaysDataType -json 2>/dev/null'
      );
      const gpuData = JSON.parse(gpuInfo);
      const displays = gpuData.SPDisplaysDataType || [];
      if (displays.length > 0 && displays[0].sppci_model) {
        gpuName = displays[0].sppci_model;
      } else {
        gpuName = cpuInfo.trim().replace('Apple ', '') + ' GPU';
      }
    } catch {
      gpuName = cpuInfo.trim().replace('Apple ', '') + ' GPU';
    }

    // Get GPU cores if available
    try {
      const { stdout: gpuCoreInfo } = await execAsync(
        'system_profiler SPDisplaysDataType | grep "Total Number of Cores"'
      );
      const coreMatch = gpuCoreInfo.match(/(\d+)/);
      if (coreMatch) {
        gpuCores = parseInt(coreMatch[1]);
      }
    } catch {
      // Ignore if we can't get GPU cores
    }

    // Calculate memory stats
    let totalMemoryMB: number;
    let usedMemoryMB: number;
    let freeMemoryMB: number;

    if (macmonData) {
      // Use macmon data for accurate memory stats
      totalMemoryMB = Math.round(macmonData.memory.ram_total / (1024 * 1024));
      usedMemoryMB = Math.round(macmonData.memory.ram_usage / (1024 * 1024));
      freeMemoryMB = totalMemoryMB - usedMemoryMB;
    } else {
      // Fallback to sysctl/vm_stat
      const { stdout: memTotal } = await execAsync('sysctl -n hw.memsize');
      const totalMemoryBytes = parseInt(memTotal.trim());
      totalMemoryMB = Math.round(totalMemoryBytes / (1024 * 1024));

      const { stdout: vmStat } = await execAsync('vm_stat');
      const pageSize = 16384;
      const freeMatch = vmStat.match(/Pages free:\s+(\d+)/);
      const inactiveMatch = vmStat.match(/Pages inactive:\s+(\d+)/);
      const freePages = freeMatch ? parseInt(freeMatch[1]) : 0;
      const inactivePages = inactiveMatch ? parseInt(inactiveMatch[1]) : 0;
      freeMemoryMB = Math.round(((freePages + inactivePages) * pageSize) / (1024 * 1024));
      usedMemoryMB = totalMemoryMB - freeMemoryMB;
    }

    // GPU utilization and temperature from macmon
    const gpuUtil = macmonData ? Math.round(macmonData.gpu_usage[1] * 100) : 0;
    const gpuTemp = macmonData ? Math.round(macmonData.temp.gpu_temp_avg) : 0;
    const gpuFreq = macmonData ? Math.round(macmonData.gpu_usage[0]) : 0;
    const gpuPower = macmonData ? Math.round(macmonData.gpu_power * 100) / 100 : 0;
    const totalPower = macmonData ? Math.round(macmonData.sys_power * 100) / 100 : 0;

    return {
      index: 0,
      name: gpuName + (gpuCores ? ` (${gpuCores} cores)` : ''),
      driverVersion: 'MPS (Metal Performance Shaders)',
      temperature: gpuTemp,
      utilization: {
        gpu: gpuUtil,
        memory: Math.round((usedMemoryMB / totalMemoryMB) * 100),
      },
      memory: {
        total: totalMemoryMB,
        free: freeMemoryMB,
        used: usedMemoryMB,
      },
      power: {
        draw: gpuPower,
        limit: totalPower, // Using system power as a reference
      },
      clocks: {
        graphics: gpuFreq,
        memory: 0, // Not available on Apple Silicon
      },
      fan: {
        speed: 0, // No fan speed reporting on Apple Silicon
      },
    };
  } catch (error) {
    console.error('Error getting Mac GPU stats:', error);
    return null;
  }
}

async function checkNvidiaSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      // Check if nvidia-smi is available on Windows
      // It's typically located in C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe
      // but we'll just try to run it directly as it may be in PATH
      await execAsync('nvidia-smi -L');
    } else {
      // Linux/macOS check
      await execAsync('which nvidia-smi');
    }
    return true;
  } catch (error) {
    return false;
  }
}

async function getGpuStats(isWindows: boolean) {
  // Command is the same for both platforms, but the path might be different
  const command =
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits';

  // Execute command
  const { stdout } = await execAsync(command, {
    env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
  });

  // Parse CSV output
  const gpus = stdout
    .trim()
    .split('\n')
    .map(line => {
      const [
        index,
        name,
        driverVersion,
        temperature,
        gpuUtil,
        memoryUtil,
        memoryTotal,
        memoryFree,
        memoryUsed,
        powerDraw,
        powerLimit,
        clockGraphics,
        clockMemory,
        fanSpeed,
      ] = line.split(', ').map(item => item.trim());

      return {
        index: parseInt(index),
        name,
        driverVersion,
        temperature: parseInt(temperature),
        utilization: {
          gpu: parseInt(gpuUtil),
          memory: parseInt(memoryUtil),
        },
        memory: {
          total: parseInt(memoryTotal),
          free: parseInt(memoryFree),
          used: parseInt(memoryUsed),
        },
        power: {
          draw: parseFloat(powerDraw),
          limit: parseFloat(powerLimit),
        },
        clocks: {
          graphics: parseInt(clockGraphics),
          memory: parseInt(clockMemory),
        },
        fan: {
          speed: parseInt(fanSpeed) || 0, // Some GPUs might not report fan speed, default to 0
        },
      };
    });

  return gpus;
}
