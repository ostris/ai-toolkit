import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';

const execAsync = promisify(exec);

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';

    // Check for NVIDIA GPUs first
    const hasNvidiaSmi = await checkNvidiaSmi(isWindows);
    if (hasNvidiaSmi) {
      const gpuStats = await getNvidiaGpuStats(isWindows);
      return NextResponse.json({
        hasNvidiaSmi: true,
        hasRocmSmi: false,
        gpus: gpuStats,
      });
    }

    // Check for ROCm/AMD GPUs
    const hasRocmSmi = await checkRocmSmi(isWindows);
    if (hasRocmSmi) {
      const gpuStats = await getRocmGpuStats(isWindows);
      return NextResponse.json({
        hasNvidiaSmi: false,
        hasRocmSmi: true,
        gpus: gpuStats,
      });
    }

    // No GPU detection available
    return NextResponse.json({
      hasNvidiaSmi: false,
      hasRocmSmi: false,
      gpus: [],
      error: 'Neither nvidia-smi nor rocm-smi found. GPU detection unavailable.',
    });
  } catch (error) {
    console.error('Error fetching GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        hasRocmSmi: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
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

async function checkRocmSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      // On Windows, try to run rocm-smi directly (may be in PATH or Program Files)
      await execAsync('rocm-smi --version');
    } else {
      // Linux/macOS check
      await execAsync('which rocm-smi');
    }
    return true;
  } catch (error) {
    return false;
  }
}

async function getNvidiaGpuStats(isWindows: boolean) {
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

async function getRocmGpuStats(isWindows: boolean) {
  // Get GPU list using CSV format
  const command =
    'rocm-smi --showid --showproductname --showtemp --showuse --showmemuse --showmeminfo vram --showpower --showclocks --csv';

  try {
    const { stdout } = await execAsync(command);
    const lines = stdout.trim().split('\n');
    
    // Skip header line
    if (lines.length < 2) {
      return [];
    }

    const gpus = lines.slice(1).map((line, idx) => {
      const fields = line.split(',');
      
      // Parse device name (card0, card1, etc.) to get index
      const deviceName = fields[0]?.trim() || '';
      const index = deviceName.replace('card', '') ? parseInt(deviceName.replace('card', '')) : idx;
      
      // Extract fields - order: device,GPU ID,Temperature,mclk clock speed,mclk clock level,sclk clock speed,sclk clock level,socclk clock speed,socclk clock level,Power,GPU use,Memory Activity,VRAM Total Memory,VRAM Total Used Memory,Card series,Card model,Card vendor,Card SKU
      const temperature = parseFloat(fields[2]?.trim() || '0');
      const gpuUtil = parseFloat(fields[10]?.trim() || '0');
      const memoryTotal = parseFloat(fields[12]?.trim() || '0');
      const memoryUsed = parseFloat(fields[13]?.trim() || '0');
      const memoryFree = memoryTotal - memoryUsed;
      const powerDraw = parseFloat(fields[9]?.trim() || '0');
      
      // Parse clock speeds (format: "(1000Mhz)" -> 1000)
      const mclkStr = fields[3]?.trim() || '(0Mhz)';
      const sclkStr = fields[5]?.trim() || '(0Mhz)';
      const clockGraphics = parseInt(mclkStr.replace(/[()Mhz]/g, '')) || 0;
      const clockMemory = parseInt(sclkStr.replace(/[()Mhz]/g, '')) || 0;
      
      // Get GPU name from Card SKU (most descriptive), then Card model, then fallback
      // CSV fields: device,GPU ID,Temperature,...,Card series,Card model,Card vendor,Card SKU
      const cardSku = fields[17]?.trim() || '';
      const cardModel = fields[15]?.trim() || '';
      const cardVendor = fields[16]?.trim() || '';
      // Use Card SKU if available and not a hex ID, otherwise use a descriptive name
      let name = '';
      if (cardSku && !cardSku.startsWith('0x')) {
        name = cardSku;
      } else if (cardVendor && cardVendor.includes('AMD')) {
        name = `AMD GPU ${index}`;
      } else {
        name = `GPU ${index}`;
      }

      return {
        index,
        name,
        driverVersion: 'ROCm', // rocm-smi doesn't provide driver version in CSV
        temperature: Math.round(temperature),
        utilization: {
          gpu: Math.round(gpuUtil),
          memory: Math.round((memoryUsed / memoryTotal) * 100) || 0,
        },
        memory: {
          total: Math.round(memoryTotal / (1024 * 1024 * 1024)), // Convert bytes to GB
          free: Math.round(memoryFree / (1024 * 1024 * 1024)),
          used: Math.round(memoryUsed / (1024 * 1024 * 1024)),
        },
        power: {
          draw: powerDraw,
          limit: 0, // rocm-smi CSV doesn't provide power limit
        },
        clocks: {
          graphics: clockGraphics,
          memory: clockMemory,
        },
        fan: {
          speed: 0, // rocm-smi CSV doesn't provide fan speed
        },
      };
    });

    return gpus;
  } catch (error) {
    console.error('Error parsing rocm-smi output:', error);
    // Fallback: try to detect number of devices
    try {
      if (isWindows) {
        // On Windows, try to query via WMI or check for AMD devices
        // For now, return empty array - Windows ROCm detection is limited
        return [];
      } else {
        // Linux: try to detect via /dev/dri/renderD*
        const { stdout } = await execAsync('ls -1 /dev/dri/renderD* 2>/dev/null | wc -l');
        const deviceCount = parseInt(stdout.trim()) || 0;
        return Array.from({ length: deviceCount }, (_, i) => ({
          index: i,
          name: `AMD GPU ${i}`,
          driverVersion: 'ROCm',
          temperature: 0,
          utilization: { gpu: 0, memory: 0 },
          memory: { total: 0, free: 0, used: 0 },
          power: { draw: 0, limit: 0 },
          clocks: { graphics: 0, memory: 0 },
          fan: { speed: 0 },
        }));
      }
    } catch (fallbackError) {
      return [];
    }
  }
}
