import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function GET() {
  try {
    // Check if nvidia-smi is available
    const hasNvidiaSmi = await checkNvidiaSmi();

    if (!hasNvidiaSmi) {
      return NextResponse.json({
        hasNvidiaSmi: false,
        gpus: [],
        error: 'nvidia-smi not found or not accessible',
      });
    }

    // Get GPU stats
    const gpuStats = await getGpuStats();

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

async function checkNvidiaSmi(): Promise<boolean> {
  try {
    await execAsync('which nvidia-smi');
    return true;
  } catch (error) {
    return false;
  }
}

async function getGpuStats() {
  // Get detailed GPU information in JSON format including fan speed
  const { stdout } = await execAsync(
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits',
  );

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
          speed: parseInt(fanSpeed), // Fan speed as percentage
        },
      };
    });

  return gpus;
}