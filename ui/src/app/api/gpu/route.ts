import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import fs from 'fs';

const execAsync = promisify(exec);
const statAsync = promisify(fs.stat);

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';


    // Check for GPU monitor (nvidia-smi or rocm-smi)
    const monitor = await checkGpuMonitor(isWindows);

    if (!monitor) {
      return NextResponse.json({
        hasGpuMonitor: false,
        gpus: [],
        error: "Unable to load 'nvidia-smi' or 'rocm-smi' - ensure the correct GPU monitoring tool is installed",
      });
    }

    // Get GPU stats
    const gpuStats = await getGpuStats(monitor, isWindows);

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

interface MonitorInfo {
  type: 'nvidia' | 'amd';
  path: string;
}

async function checkGpuMonitor(isWindows: boolean): Promise<MonitorInfo | null> {
  try {
    // Check nvidia-smi first
    if (isWindows) {
      await execAsync('nvidia-smi -L');
      return { type: 'nvidia', path: 'nvidia-smi' };
    } else {
      try {
        await execAsync('which nvidia-smi');
        return { type: 'nvidia', path: 'nvidia-smi' };
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
      // 1. Check path (which rocm-smi)
      try {
        await execAsync('which rocm-smi');
        return { type: 'amd', path: 'rocm-smi' };
      } catch {
        // failed
      }

      // 2. Check $ROCM_PATH/bin/rocm-smi
      if (process.env.ROCM_PATH) {
        try {
          const manualPath = `${process.env.ROCM_PATH}/bin/rocm-smi`;
          await statAsync(manualPath);
          return { type: 'amd', path: manualPath };
        } catch {
          // failed
        }
      }

      // 3. Check /usr/bin/rocm-smi
      try {
        const manualPath = '/usr/bin/rocm-smi';
        await statAsync(manualPath);
        return { type: 'amd', path: manualPath };
      } catch {
        // failed
      }

      // 4. Check /opt/rocm/bin/rocm-smi
      try {
        const manualPath = '/opt/rocm/bin/rocm-smi';
        await statAsync(manualPath);
        return { type: 'amd', path: manualPath };
      } catch {
        // failed
      }
    } else {
      // Windows - check for hipinfo.exe
      // 1. Check HIP_PATH
      if (process.env.HIP_PATH) {
        try {
          // HIP_PATH usually ends with trailing slash, but ensure reliable join
          const basePath = process.env.HIP_PATH.replace(/\\$/, '');
          const manualPath = `${basePath}\\bin\\hipinfo.exe`;
          await statAsync(manualPath);
          return { type: 'amd', path: manualPath };
        } catch {
          // failed
        }
      }

      // 2. Check path
      try {
        // 'where' is windows equivalent of 'which'
        await execAsync('where hipinfo.exe');
        return { type: 'amd', path: 'hipinfo.exe' };
      } catch {
        // failed
      }

      // 3. Check common default location C:\Program Files\AMD\ROCm\*\bin\hipInfo.exe
      // Since we can't easily glob, checking a likely default version or just skipping. 
      // The user provided C:\AMD\ROCm\6.2\bin in their prompt example, so maybe check C:\AMD\ROCm ?
      // Given the variability, reliance on PATH or HIP_PATH is best for now.
    }
  } catch (error) {
    return null;
  }

  return null;
}

async function getGpuStats(monitor: MonitorInfo, isWindows: boolean) {
  if (monitor.type === 'nvidia') {
    return getNvidiaStats(monitor.path, isWindows);
  } else {
    if (monitor.path.toLowerCase().includes('hipinfo')) {
      return getHipStats(monitor.path);
    }
    return getAmdStats(monitor.path);
  }
}

async function getNvidiaStats(path: string, isWindows: boolean) {
  // Command is the same for both platforms, but the path might be different
  const command =
    `${path} --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits`;

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

async function getHipStats(path: string) {
  const { stdout } = await execAsync(`"${path}"`);

  // Parse hipinfo output
  // Output format is block-based per device#
  const devices = stdout.split('device#').slice(1); // skip preamble if any

  return devices.map((deviceBlock, idx) => {
    const lines = deviceBlock.split('\n');
    const getVal = (key: string) => {
      const line = lines.find(l => l.trim().startsWith(key));
      if (!line) return null;
      // split by first colon
      const parts = line.split(':');
      if (parts.length < 2) return null;
      return parts.slice(1).join(':').trim();
    };

    const name = getVal('Name') || `AMD GPU ${idx}`;
    // clockRate: 2162 Mhz
    const clockRateStr = getVal('clockRate');
    const clockRate = clockRateStr ? parseFloat(clockRateStr.split(' ')[0]) : 0;

    const memClockStr = getVal('memoryClockRate');
    const memClock = memClockStr ? parseFloat(memClockStr.split(' ')[0]) : 0;

    // memInfo.total: 0.16 GB
    // memInfo.free: 0.02 GB (14%)
    const memTotalStr = getVal('memInfo.total');
    let memTotalMB = 0;
    if (memTotalStr) {
      if (memTotalStr.includes('GB')) {
        memTotalMB = parseFloat(memTotalStr.split(' ')[0]) * 1024;
      } else if (memTotalStr.includes('MB')) {
        memTotalMB = parseFloat(memTotalStr.split(' ')[0]);
      }
    }

    const memFreeStr = getVal('memInfo.free');
    let memFreeMB = 0;
    if (memFreeStr) {
      // extract number before unit
      const val = parseFloat(memFreeStr.split(' ')[0]);
      if (memFreeStr.includes('GB')) {
        memFreeMB = val * 1024;
      } else if (memFreeStr.includes('MB')) {
        memFreeMB = val;
      }
    }

    const memUsedMB = memTotalMB - memFreeMB;

    return {
      index: idx,
      name,
      driverVersion: 'Unknown', // hipinfo doesn't seem to show driver ver easily in this block
      temperature: 0, // Not available
      utilization: {
        gpu: 0, // Not available
        memory: memTotalMB > 0 ? Math.round((memUsedMB / memTotalMB) * 100) : 0,
      },
      memory: {
        total: Math.round(memTotalMB),
        free: Math.round(memFreeMB),
        used: Math.round(memUsedMB),
      },
      power: {
        draw: 0, // Not available
        limit: 0, // Not available
      },
      clocks: {
        graphics: Math.round(clockRate),
        memory: Math.round(memClock),
      },
      fan: {
        speed: 0, // Not available
      },
    };
  });
}

async function getAmdStats(path: string) {
  // Use rocm-smi with json output
  // We request all relevant details
  // flags: --showproductname --showdriverversion --showtemp --showuse --showmeminfo --showpower --showclocks --showfan --json
  const command = `${path} --showproductname --showdriverversion --showtemp --showuse --showmeminfo vram --showpower --showclocks --showfan --json`;

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
    // It also includes "system" which we should ignore
    const gpus = Object.keys(data)
      .filter(key => key.startsWith('card'))
      .map((cardKey, idx) => {
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
      })
      .filter(gpu => gpu.memory.total > 0); // Filter out devices with 0 VRAM (e.g. dummy devices or NPUs with no VRAM reporting)

    return gpus;
  } catch (e) {
    console.error('Failed to parse rocm-smi output', e);
    throw new Error('Failed to parse rocm-smi output');
  }
}
