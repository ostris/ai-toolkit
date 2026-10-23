import { NextResponse } from 'next/server';
import { exec, execFile, execSync } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { cached } from '@/server/apiCache';
import { loadMacstats } from '@/server/macstats';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

/**
 * Resolve nvidia-smi even when PATH is scrubbed (Stability Matrix / launcher
 * envs often keep only node+venv dirs and drop /usr/bin).
 */
function resolveNvidiaSmi(isWindows: boolean): string | null {
  const fromEnv = process.env.NVIDIA_SMI;
  if (fromEnv && fs.existsSync(fromEnv)) return fromEnv;

  const candidates = isWindows
    ? [
        path.join(
          process.env['ProgramFiles'] || 'C:\\Program Files',
          'NVIDIA Corporation',
          'NVSMI',
          'nvidia-smi.exe',
        ),
        path.join(
          process.env['ProgramW6432'] || 'C:\\Program Files',
          'NVIDIA Corporation',
          'NVSMI',
          'nvidia-smi.exe',
        ),
      ]
    : ['/usr/bin/nvidia-smi', '/bin/nvidia-smi', '/usr/local/bin/nvidia-smi'];

  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) return candidate;
    } catch {
      // ignore
    }
  }

  // Last resort: honor PATH (normal desktop installs).
  try {
    const whichCmd = isWindows ? 'where nvidia-smi' : 'command -v nvidia-smi';
    const stdout = execSync(whichCmd, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      env: {
        ...process.env,
        PATH: [process.env.PATH || '', isWindows ? '' : '/usr/local/bin:/usr/bin:/bin']
          .filter(Boolean)
          .join(path.delimiter),
      },
    });
    const hit = stdout.trim().split(/\r?\n/)[0];
    if (hit && fs.existsSync(hit)) return hit;
  } catch {
    // not found
  }
  return null;
}

interface MacGpuResult {
  name: string;
  memUsed: number;
  memTotal: number;
  gpuLoad: number;
  temperature: number;
  fanSpeed: number;
  powerDraw: number;
}

async function getMacGpuInfo(): Promise<MacGpuResult | null> {
  try {
    const memoryTotal = os.totalmem() / (1024 * 1024);

    // Get GPU name and core count from system_profiler
    let gpuName = 'Apple GPU';
    try {
      const { stdout: spOut } = await execAsync(
        'system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores"',
        { encoding: 'utf-8', timeout: 5000 },
      );
      const nameMatch = spOut.match(/Chipset Model:\s*(.+)/);
      const coresMatch = spOut.match(/Total Number of Cores:\s*(\d+)/);
      if (nameMatch) {
        gpuName = nameMatch[1].trim();
        if (coresMatch) {
          gpuName += ` GPU (${coresMatch[1]} cores)`;
        }
      }
    } catch {
      // fallback to generic name
    }

    let temperature = 0;
    let gpuLoad = 0;
    let fanSpeed = 0;
    let powerDraw = 0;
    let memUsed = 0;
    let memTotal = memoryTotal;

    const ms = loadMacstats();
    if (ms) {
      try {
        const gpuData = ms.getGpuDataSync();
        temperature = gpuData.temperature || 0;
        gpuLoad = gpuData.usage || 0;
      } catch {
        // ignore
      }

      try {
        const fanData = ms.getFanDataSync();
        const fanKeys = Object.keys(fanData);
        if (fanKeys.length > 0) {
          fanSpeed = fanData[fanKeys[0]].rpm || 0;
        }
      } catch {
        // ignore
      }

      try {
        const powerData = ms.getPowerDataSync();
        powerDraw = powerData.gpu || 0;
      } catch {
        // ignore
      }

      try {
        const ramData = ms.getRAMUsageSync();
        memUsed = ramData.used / (1024 * 1024);
        memTotal = ramData.total / (1024 * 1024);
      } catch {
        // ignore
      }
    }

    return { name: gpuName, memUsed, memTotal, gpuLoad, temperature, fanSpeed, powerDraw };
  } catch {
    return null;
  }
}

async function getGpuInfo() {
  // Get platform
  const platform = os.platform();
  const isWindows = platform === 'win32';
  const isMac = platform === 'darwin';

  if (isMac) {
    const macGpu = await getMacGpuInfo();
    if (macGpu) {
      return {
        hasNvidiaSmi: false,
        isMac: true,
        gpus: [
          {
            index: 0,
            name: macGpu.name,
            driverVersion: 'macOS',
            temperature: Math.round(macGpu.temperature),
            utilization: {
              gpu: macGpu.gpuLoad,
              memory: macGpu.memTotal > 0 ? Math.round((macGpu.memUsed / macGpu.memTotal) * 100) : 0,
            },
            memory: {
              total: Math.round(macGpu.memTotal),
              free: Math.round(macGpu.memTotal - macGpu.memUsed),
              used: Math.round(macGpu.memUsed),
            },
            power: { draw: macGpu.powerDraw, limit: 0 },
            clocks: { graphics: 0, memory: 0 },
            fan: { speed: macGpu.fanSpeed },
          },
        ],
      };
    }
    return {
      hasNvidiaSmi: false,
      isMac: true,
      gpus: [],
      error: 'Could not read Mac GPU stats',
    };
  }

  // Check if nvidia-smi is available
  const hasNvidiaSmi = await checkNvidiaSmi(isWindows);

  if (!hasNvidiaSmi) {
    return {
      hasNvidiaSmi: false,
      isMac: false,
      gpus: [],
      error: 'nvidia-smi not found or not accessible',
    };
  }

  // Get GPU stats
  const gpuStats = await getGpuStats(isWindows);

  return {
    hasNvidiaSmi: true,
    gpus: gpuStats,
  };
}

export async function GET() {
  try {
    const gpuInfo = await cached('gpu-info', getGpuInfo);
    return NextResponse.json(gpuInfo);
  } catch (error) {
    console.error('Error fetching NVIDIA GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        isMac: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

async function checkNvidiaSmi(isWindows: boolean): Promise<boolean> {
  const smi = resolveNvidiaSmi(isWindows);
  if (!smi) return false;
  try {
    await execFileAsync(smi, ['-L'], { timeout: 8000 });
    return true;
  } catch {
    return false;
  }
}

async function getGpuStats(isWindows: boolean) {
  const smi = resolveNvidiaSmi(isWindows);
  if (!smi) {
    throw new Error('nvidia-smi not found');
  }

  const { stdout } = await execFileAsync(
    smi,
    [
      '--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed',
      '--format=csv,noheader,nounits',
    ],
    {
      timeout: 8000,
      env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
      encoding: 'utf8',
    },
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
          speed: parseInt(fanSpeed) || 0, // Some GPUs might not report fan speed, default to 0
        },
      };
    });

  return gpus;
}
