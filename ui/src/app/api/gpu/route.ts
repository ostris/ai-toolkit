import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import fs from 'fs';
import path from 'path';

const execAsync = promisify(exec);
const statAsync = promisify(fs.stat);
const readFileAsync = promisify(fs.readFile);
const readdirAsync = promisify(fs.readdir);

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';


    // Check for GPU monitor (nvidia-smi, amd-smi or rocm-smi)
    const monitor = await checkGpuMonitor(isWindows);

    if (!monitor) {
      return NextResponse.json({
        hasGpuMonitor: false,
        gpus: [],
        error: "Unable to load 'nvidia-smi', 'amd-smi' or 'rocm-smi' - ensure the correct GPU monitoring tool is installed",
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
  type: 'nvidia' | 'amd-smi' | 'rocm-smi';
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
    // continue to check amd/rocm
  }

  // Check amd-smi (prioritize over rocm-smi as it is newer)
  try {
    if (!isWindows) {
      try {
        await execAsync('which amd-smi');
        return { type: 'amd-smi', path: 'amd-smi' };
      } catch {
        // failed
      }
    }
  } catch (error) {
    // continue
  }

  // Check rocm-smi
  try {
    if (!isWindows) {
      // rocm-smi is typically linux only or wsl
      // 1. Check path (which rocm-smi)
      try {
        await execAsync('which rocm-smi');
        return { type: 'rocm-smi', path: 'rocm-smi' };
      } catch {
        // failed
      }

      // 2. Check $ROCM_PATH/bin/rocm-smi
      if (process.env.ROCM_PATH) {
        try {
          const manualPath = `${process.env.ROCM_PATH}/bin/rocm-smi`;
          await statAsync(manualPath);
          return { type: 'rocm-smi', path: manualPath };
        } catch {
          // failed
        }
      }

      // 3. Check /usr/bin/rocm-smi
      try {
        const manualPath = '/usr/bin/rocm-smi';
        await statAsync(manualPath);
        return { type: 'rocm-smi', path: manualPath };
      } catch {
        // failed
      }

      // 4. Check /opt/rocm/bin/rocm-smi
      try {
        const manualPath = '/opt/rocm/bin/rocm-smi';
        await statAsync(manualPath);
        return { type: 'rocm-smi', path: manualPath };
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
          return { type: 'rocm-smi', path: manualPath }; // Treat hidden hipinfo as rocm-smi/amd generic
        } catch {
          // failed
        }
      }

      // 2. Check path
      try {
        // 'where' is windows equivalent of 'which'
        await execAsync('where hipinfo.exe');
        return { type: 'rocm-smi', path: 'hipinfo.exe' };
      } catch {
        // failed
      }
    }
  } catch (error) {
    return null;
  }

  return null;
}

async function getGpuStats(monitor: MonitorInfo, isWindows: boolean) {
  if (monitor.type === 'nvidia') {
    return getNvidiaStats(monitor.path, isWindows);
  } else if (monitor.type === 'amd-smi') {
    return getAmdSmiStats(monitor.path);
  } else {
    // rocm-smi or hipinfo
    if (monitor.path.toLowerCase().includes('hipinfo')) {
      return getHipStats(monitor.path);
    }
    return getRocmStats(monitor.path);
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

// Helper to read Sysfs values safely
async function readSysfs(path: string): Promise<string | null> {
  try {
    const data = await readFileAsync(path, 'utf8');
    return data.trim();
  } catch {
    return null;
  }
}

async function getAmdSmiStats(toolPath: string) {
  try {
    // 1. Get Static Info (Name, Driver)
    // amd-smi static --asic --driver --json
    const staticCmd = `${toolPath} static --asic --driver --json`;
    const { stdout: staticStdout } = await execAsync(staticCmd);
    const staticData = JSON.parse(staticStdout);

    // 2. Get Metrics (Usage, Power, Clock, Temp, Fan, Mem)
    // amd-smi metric --usage --mem-usage --power --fan --clock --temperature --json
    const metricCmd = `${toolPath} metric --usage --mem-usage --power --fan --clock --temperature --json`;
    const { stdout: metricStdout } = await execAsync(metricCmd);
    const metricData = JSON.parse(metricStdout);

    // Map by GPU ID
    // Assumption: Both commands return gpu_data array with corresponding "gpu" indices.
    const staticGpus = staticData.gpu_data || [];
    const metricGpus = metricData.gpu_data || [];

    // Use traditional for-loop to support await
    const gpus = await Promise.all(staticGpus.map(async (staticInfo: any) => {
      const gpuId = staticInfo.gpu;
      const metrics = metricGpus.find((m: any) => m.gpu === gpuId) || {};

      const name = staticInfo.asic?.market_name || `AMD GPU ${gpuId}`;
      const driverVersion = staticInfo.driver?.version || 'Unknown';

      const parseVal = (val: any) => {
        if (typeof val === 'number') return val;
        if (typeof val === 'string' && val !== 'N/A') return parseFloat(val);
        return 0;
      };

      const parseUnitVal = (obj: any) => {
        if (!obj || obj === 'N/A' || !obj.value || obj.value === 'N/A') return null;
        return parseFloat(obj.value);
      };

      // --- Try amd-smi values first, then fallback to sysfs ---

      const sysfsCardPath = `/sys/class/drm/card${gpuId}/device`;
      let hwmonPath: string | null = null;

      const getHwmonPath = async () => {
        if (hwmonPath) return hwmonPath;
        try {
          const files = await readdirAsync(`${sysfsCardPath}/hwmon`);
          // Usually just one hwmon folder, pick first
          if (files.length > 0) {
            hwmonPath = `${sysfsCardPath}/hwmon/${files[0]}`;
            return hwmonPath;
          }
        } catch { }
        return null;
      }


      // -- Temperature --
      let temp = parseUnitVal(metrics.temperature?.edge);
      if (temp === null) {
        // Fallback: cat $HWMON_PATH/temp1_input (millidegrees)
        const hwmon = await getHwmonPath();
        if (hwmon) {
          const t = await readSysfs(`${hwmon}/temp1_input`);
          if (t) temp = parseInt(t) / 1000;
        }
      }
      temp = temp || 0;

      // -- GPU Utilization --
      let gpuUtil = parseUnitVal(metrics.usage?.gfx_activity);
      if (gpuUtil === null) {
        // Fallback: cat $CARD_PATH/gpu_busy_percent
        const busyp = await readSysfs(`${sysfsCardPath}/gpu_busy_percent`);
        if (busyp) gpuUtil = parseInt(busyp);
      }
      gpuUtil = gpuUtil || 0;

      // -- Memory (Used/Total) --
      let memVramTotalMB = parseUnitVal(metrics.mem_usage?.total_vram);
      let memVramUsedMB = parseUnitVal(metrics.mem_usage?.used_vram);

      if (memVramUsedMB === null) {
        // Fallback: cat $CARD_PATH/mem_info_vram_used (bytes)
        const used = await readSysfs(`${sysfsCardPath}/mem_info_vram_used`);
        if (used) memVramUsedMB = parseInt(used) / 1024 / 1024;
      }

      // If total VRAM is missing, might be trickier from sysfs (mem_info_vram_total)
      if (memVramTotalMB === null) {
        const total = await readSysfs(`${sysfsCardPath}/mem_info_vram_total`);
        if (total) memVramTotalMB = parseInt(total) / 1024 / 1024;
      }

      // -- GTT Memory (Used/Total) --
      let memGttTotalMB = parseUnitVal(metrics.mem_usage?.total_gtt);
      let memGttUsedMB = parseUnitVal(metrics.mem_usage?.used_gtt);

      if (memGttUsedMB === null) {
        // Fallback: cat $CARD_PATH/mem_info_gtt_used (bytes)
        const used = await readSysfs(`${sysfsCardPath}/mem_info_gtt_used`);
        if (used) memGttUsedMB = parseInt(used) / 1024 / 1024;
      }

      if (memGttTotalMB === null) {
        const total = await readSysfs(`${sysfsCardPath}/mem_info_gtt_total`);
        if (total) memGttTotalMB = parseInt(total) / 1024 / 1024;
      }

      // Keep VRAM and GTT separate
      const memTotalMB = memVramTotalMB || 0;
      const memUsedMB = memVramUsedMB || 0;

      const gttTotalMB = memGttTotalMB || 0;
      const gttUsedMB = memGttUsedMB || 0;

      // Default free calculation based on VRAM only
      const memFreeMB = parseUnitVal(metrics.mem_usage?.free_vram) || (memTotalMB - memUsedMB);

      let memUtil = 0;
      if (memTotalMB > 0) {
        memUtil = Math.round((memUsedMB / memTotalMB) * 100);
      }

      // -- Power --
      let powerDraw = parseUnitVal(metrics.power?.socket_power);
      if (powerDraw === null) {
        // Fallback: cat $HWMON_PATH/power1_average (microwatts)
        const hwmon = await getHwmonPath();
        if (hwmon) {
          const p = await readSysfs(`${hwmon}/power1_average`);
          if (p) powerDraw = parseInt(p) / 1000000;
        }
      }
      const powerLimit = 0; // Usually not easily available in sysfs without digging

      // -- Clocks --
      let sclk = parseUnitVal(metrics.clock?.gfx_0?.clk);
      let mclk = parseUnitVal(metrics.clock?.mem_0?.clk);

      if (sclk === null) {
        // Fallback: cat $HWMON_PATH/freq1_input (Hz)
        const hwmon = await getHwmonPath();
        if (hwmon) {
          const c = await readSysfs(`${hwmon}/freq1_input`);
          if (c) sclk = parseInt(c) / 1000 / 1000;
        }
      }
      if (mclk === null) {
        // Fallback: cat $HWMON_PATH/freq2_input (Hz)
        const hwmon = await getHwmonPath();
        if (hwmon) {
          const c = await readSysfs(`${hwmon}/freq2_input`);
          if (c) mclk = parseInt(c) / 1000 / 1000;
        }
      }

      // -- Fan --
      const fanSpeed = parseUnitVal(metrics.fan?.usage) || 0;
      // iGPUs usually don't have own fans anyway, so 0 is fine. Dedicated likely has amd-smi working.

      return {
        index: gpuId,
        name,
        driverVersion,
        temperature: Math.round(temp as number),
        utilization: {
          gpu: Math.round(gpuUtil as number),
          memory: Math.round(memUtil),
        },
        memory: {
          total: Math.round(memTotalMB),
          free: Math.round(memFreeMB),
          used: Math.round(memUsedMB),
          gttTotal: Math.round(gttTotalMB),
          gttUsed: Math.round(gttUsedMB),
          gttFree: Math.round(gttTotalMB - gttUsedMB),
        },
        power: {
          draw: powerDraw || 0,
          limit: powerLimit,
        },
        clocks: {
          graphics: Math.round(sclk || 0),
          memory: Math.round(mclk || 0),
        },
        fan: {
          speed: Math.round(fanSpeed),
        },
      };
    }));

    return gpus;

  } catch (e) {
    console.error('Failed to run/parse amd-smi', e);
    // Fallback? Or just throw.
    throw e;
  }
}

async function getRocmStats(path: string) {
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
