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


    // Check for GPU monitor (nvidia-smi or rocm-smi)
    const monitorType = await checkGpuMonitor(isWindows);

    if (!monitorType) {
      return NextResponse.json({
        hasGpuMonitor: false,
        gpus: [],
        error: 'No GPU monitor found (nvidia-smi or rocm-smi)',
      });
    }

    // Get GPU stats
    const gpuStats = await getGpuStats(monitorType, isWindows);

    return NextResponse.json({
      hasGpuMonitor: true,
      gpus: gpuStats,
    });
  } catch (error) {
    console.error('Error fetching GPU stats:', error);
    return NextResponse.json(
      {
        hasGpuMonitor: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

async function checkGpuMonitor(isWindows: boolean): Promise<'nvidia' | 'amd' | null> {
  try {
    // Check nvidia-smi first
    if (isWindows) {
      await execAsync('nvidia-smi -L');
      return 'nvidia';
    } else {
      try {
        await execAsync('which nvidia-smi');
        return 'nvidia';
      } catch {
        // failed
      }
    }
  } catch (error) {
    // continue to check rocm-smi
  }

  // Check rocm-smi
  try {
    if (!isWindows) {
      // rocm-smi is typically linux only or wsl
      try {
        await execAsync('which rocm-smi');
        return 'amd';
      } catch {
        // failed
      }
    }
  } catch (error) {
    return null;
  }

  return null;
}

async function getGpuStats(type: 'nvidia' | 'amd', isWindows: boolean) {
  if (type === 'nvidia') {
    return getNvidiaStats(isWindows);
  } else {
    return getAmdStats();
  }
}

async function getNvidiaStats(isWindows: boolean) {
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

async function getAmdStats() {
  // Use rocm-smi with json output
  // We request all relevant details
  // flags: --showproductname --showdriverversion --showtemp --showuse --showmeminfo --showpower --showclocks --showfan --json
  const command = 'rocm-smi --showproductname --showdriverversion --showtemp --showuse --showmeminfo vram --showpower --showclocks --showfan --json';

  const { stdout } = await execAsync(command);

  try {
    // Attempt to find the JSON object if there's extra output (e.g. warnings or exceptions)
    const jsonStart = stdout.indexOf('{');
    const jsonEnd = stdout.lastIndexOf('}');

    let jsonStr = stdout;
    if (jsonStart !== -1 && jsonEnd !== -1) {
      jsonStr = stdout.substring(jsonStart, jsonEnd + 1);
    }

    const data = JSON.parse(jsonStr);
    // rocm-smi returns object with keys like "card0", "card1"
    const gpus = Object.keys(data).map((cardKey, idx) => {
      const card = data[cardKey];

      // Parse values. Note: keys might vary slightly by version, trying to be robust
      // Typical keys: "Temperature (Sensor edge) (C)", "GPU use (%)", "Average Graphics Package Power (W)", "System Clock (MHz)"

      const getVal = (keys: string[], defaultVal: any = 0) => {
        for (const k of keys) {
          if (card[k] !== undefined) return card[k];
        }
        return defaultVal;
      };

      const name = getVal(['Card Series', 'Card Model', 'Device ID'], `AMD GPU ${idx}`);
      const driverVersion = getVal(['Driver version'], 'Unknown');

      const temp = parseFloat(getVal(['Temperature (Sensor edge) (C)', 'Temperature (Junction) (C)'], '0'));
      const gpuUtil = parseFloat(getVal(['GPU use (%)'], '0'));
      const memUtil = parseFloat(getVal(['GPU Memory use (%)', 'Memory Activity'], '0'));

      const memTotal = parseFloat(getVal(['VRAM Total Memory (B)', 'VRAM Total Memory (MB)', 'VRAM Total Memory (kB)'], '0')); // careful with units
      const memUsed = parseFloat(getVal(['VRAM Total Used Memory (B)', 'VRAM Total Used Memory (MB)', 'VRAM Total Used Memory (kB)'], '0'));

      // Normalize memory to MB
      // rocm-smi usually reports bytes if (B) is in key
      let memTotalMB = memTotal;
      let memUsedMB = memUsed;

      if (card['VRAM Total Memory (B)']) memTotalMB = memTotal / 1024 / 1024;
      if (card['VRAM Total Used Memory (B)']) memUsedMB = memUsed / 1024 / 1024;

      const powerDraw = parseFloat(getVal(['Average Graphics Package Power (W)', 'Average SOC Power (W)'], '0'));
      const powerLimit = parseFloat(getVal(['Max Graphics Package Power (W)', 'Socket Power Limit (W)'], '0'));

      const sclk = parseFloat(getVal(['System Clock (MHz)', 'sclk_0 (MHz)'], '0'));
      const mclk = parseFloat(getVal(['Memory Clock (MHz)', 'mclk_0 (MHz)'], '0'));

      const fan = parseFloat(getVal(['Fan Speed (%)'], '0'));

      return {
        index: idx,
        name,
        driverVersion,
        temperature: Math.round(temp),
        utilization: {
          gpu: Math.round(gpuUtil),
          memory: Math.round(memUtil),
        },
        memory: {
          total: Math.round(memTotalMB),
          free: Math.round(memTotalMB - memUsedMB),
          used: Math.round(memUsedMB),
        },
        power: {
          draw: powerDraw,
          limit: powerLimit,
        },
        clocks: {
          graphics: Math.round(sclk),
          memory: Math.round(mclk),
        },
        fan: {
          speed: Math.round(fan),
        },
      };
    });

    return gpus;
  } catch (e) {
    console.error('Failed to parse rocm-smi output', e);
    throw new Error('Failed to parse rocm-smi output');
  }
}
