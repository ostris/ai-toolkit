import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import { createRequire } from 'module';
import os from 'os';

const execAsync = promisify(exec);

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

    try {
      // Use createRequire to hide from webpack static analysis so it doesn't fail on non-mac platforms
      const nativeRequire = createRequire(import.meta.url);
      const ms = nativeRequire('macstats') as any;

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
    } catch (error) {
      console.warn('macstats not available:', error);
    }

    return { name: gpuName, memUsed, memTotal, gpuLoad, temperature, fanSpeed, powerDraw };
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';
    const isMac = platform === 'darwin';

    if (isMac) {
      const macGpu = await getMacGpuInfo();
      if (macGpu) {
        return NextResponse.json({
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
        });
      }
      return NextResponse.json({
        hasNvidiaSmi: false,
        isMac: true,
        gpus: [],
        error: 'Could not read Mac GPU stats',
      });
    }

    // nvidia-smi can be installed but non-functional ("couldn't communicate
    // with the NVIDIA driver") on machines with another GPU vendor, so a
    // failed query falls through to amd-smi instead of erroring out.
    if (await checkNvidiaSmi(isWindows)) {
      try {
        const gpuStats = await getGpuStats(isWindows);
        return NextResponse.json({
          hasNvidiaSmi: true,
          gpus: gpuStats,
        });
      } catch {
        // fall through to amd-smi
      }
    }

    if (await checkAmdSmi(isWindows)) {
      try {
        const gpuStats = await getAmdGpuStats();
        return NextResponse.json({
          hasNvidiaSmi: false,
          hasAmdSmi: true,
          gpus: gpuStats,
        });
      } catch (error) {
        console.error('Error fetching AMD GPU stats:', error);
      }
    }

    return NextResponse.json({
      hasNvidiaSmi: false,
      hasAmdSmi: false,
      isMac: false,
      gpus: [],
      error: 'No working GPU monitoring tool found (tried nvidia-smi and amd-smi)',
    });
  } catch (error) {
    console.error('Error fetching GPU stats:', error);
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

async function checkAmdSmi(isWindows: boolean): Promise<boolean> {
  try {
    await execAsync(isWindows ? 'where amd-smi' : 'which amd-smi');
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

// amd-smi reports missing values as the string 'N/A' and wraps most numbers
// in { value, unit } objects; normalize both to a plain number.
function amdSmiNumber(raw: any): number {
  const value = raw && typeof raw === 'object' ? raw.value : raw;
  const num = typeof value === 'number' ? value : parseFloat(String(value));
  return Number.isFinite(num) ? num : 0;
}

// Older amd-smi versions emit a bare array instead of { gpu_data: [...] }.
function parseAmdSmiJson(stdout: string): any[] {
  const parsed = JSON.parse(stdout);
  return Array.isArray(parsed) ? parsed : (parsed.gpu_data ?? []);
}

async function getAmdGpuStats() {
  // Query only the sections we need: a bare `amd-smi metric` can crash on
  // some cards when it tries to read the voltage-curve tables.
  const [{ stdout: staticOut }, { stdout: metricOut }] = await Promise.all([
    execAsync('amd-smi static --asic --limit --driver --json'),
    execAsync('amd-smi metric --usage --power --temperature --mem-usage --fan --clock --json'),
  ]);

  const staticByGpu = new Map(parseAmdSmiJson(staticOut).map(gpu => [gpu.gpu, gpu]));

  return parseAmdSmiJson(metricOut).map(metric => {
    const staticInfo = staticByGpu.get(metric.gpu) ?? {};
    const edgeTemp = amdSmiNumber(metric.temperature?.edge);

    return {
      index: metric.gpu,
      name: staticInfo.asic?.market_name ?? 'AMD GPU',
      driverVersion: staticInfo.driver?.name ?? 'amdgpu',
      temperature: Math.round(edgeTemp || amdSmiNumber(metric.temperature?.hotspot)),
      utilization: {
        gpu: Math.round(amdSmiNumber(metric.usage?.gfx_activity)),
        memory: Math.round(amdSmiNumber(metric.usage?.umc_activity)),
      },
      memory: {
        total: Math.round(amdSmiNumber(metric.mem_usage?.total_vram)),
        free: Math.round(amdSmiNumber(metric.mem_usage?.free_vram)),
        used: Math.round(amdSmiNumber(metric.mem_usage?.used_vram)),
      },
      power: {
        draw: amdSmiNumber(metric.power?.socket_power),
        limit: amdSmiNumber(staticInfo.limit?.ppt0?.socket_power_limit),
      },
      clocks: {
        graphics: Math.round(amdSmiNumber(metric.clock?.gfx_0?.clk)),
        memory: Math.round(amdSmiNumber(metric.clock?.mem_0?.clk)),
      },
      fan: {
        // The widget renders fan speed as a percentage
        speed: Math.round(amdSmiNumber(metric.fan?.usage)),
      },
    };
  });
}

