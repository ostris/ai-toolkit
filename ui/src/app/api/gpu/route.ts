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

    // Check if nvidia-smi is available
    const hasNvidiaSmi = await checkNvidiaSmi(isWindows);

    if (!hasNvidiaSmi) {
      return NextResponse.json({
        hasNvidiaSmi: false,
        gpus: [],
        error: 'rocm-smi not found or not accessible',
      });
    }

    // Get GPU stats
    const gpuStats = await getGpuStats(isWindows);

    return NextResponse.json({
      hasNvidiaSmi: true,
      gpus: gpuStats,
    });
  } catch (error) {
    console.error('Error fetching NVIDIA GPU stats:', error);
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

async function checkNvidiaSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      // Check if nvidia-smi is available on Windows
      // It's typically located in C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe
      // but we'll just try to run it directly as it may be in PATH
      await execAsync('nvidia-smi -L');
    } else {
      // Linux/macOS check
      await execAsync('which rocm-smi');
    }
    return true;
  } catch (error) {
    return false;
  }
}




export async function getGpuStats(isWindows: boolean) {
  try {
    const command = 'rocm-smi --showid --showdriverversion --showtemp --showuse --showmeminfo vram --showpower --showclocks --showfan --json';
    const { stdout } = await execAsync(command);
    const parsed = JSON.parse(stdout);

    const driverVersion = parsed.system?.['Driver version'] ?? 'Unknown';

    const gpus = Object.entries(parsed)
      .filter(([key]) => key.startsWith('card'))
      .map(([key, info], idx) => {
        const card = info as any;

        const totalMem = parseInt(card['VRAM Total Memory (B)'], 10) || 0;
        const usedMem = parseInt(card['VRAM Total Used Memory (B)'], 10) || 0;

        return {
          index: idx,
          name: card['Device Name'] || `AMD GPU ${idx}`,
          driverVersion,
          temperature: parseFloat(card['Temperature (Sensor junction) (C)']) || 0,
          utilization: {
            gpu: parseInt(card['GPU use (%)'], 10) || 0,
            memory: totalMem ? Math.round((usedMem / totalMem) * 100) : 0,
          },
          memory: {
            total: Math.round(totalMem / (1024 * 1024)), // convert B → MB
            used: Math.round(usedMem / (1024 * 1024)),
            free: Math.round((totalMem - usedMem) / (1024 * 1024)),
          },
          power: {
            draw: parseFloat(card['Current Socket Graphics Package Power (W)']) || 0,
            limit: 0, // Not exposed in this JSON
          },
          clocks: {
            graphics: parseInt(card['sclk clock speed:']?.replace(/\D/g, ''), 10) || 0,
            memory: parseInt(card['mclk clock speed:']?.replace(/\D/g, ''), 10) || 0,
          },
          fan: {
            speed: 0,
          },
        };
      });

    // return { hasRocmSmi: true, gpus };
    return gpus;

  } catch (err: any) {
    return { hasRocmSmi: false, gpus: [], error: err.message };
  }
}

