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
    const hasAmdSmi = await checkAMDSmi(isWindows);

    if (!hasNvidiaSmi && !hasAmdSmi) {
      return NextResponse.json({
        hasNvidiaSmi: false,
        gpus: [],
        error: 'nvidia-smi not found or not accessible',
      });
    }

    // Get GPU stats
    if (hasNvidiaSmi) {
      const gpuStats = await getGpuStats(isWindows);
      return NextResponse.json({
        hasNvidiaSmi: true,
        gpus: gpuStats,
      });
    } else {
      const gpuStats = await getAMDGpuStats(isWindows);
      return NextResponse.json({
        hasNvidiaSmi: true,
        gpus: gpuStats,
      });
    }

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
async function checkAMDSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (!isWindows) {
      // Linux/macOS check
      await execAsync('which amd-smi');
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

// amdParseFloat and amdParseInt avoid errors when amd-smi entries
// contain the string "N/A".
function amdParseFloat(value) {
    try {
        const ret = parseFloat(value);
        return ret;
    } catch(error) {
        return 0.0;
    }
}

function amdParseInt(value) {
    try {
        const ret = parseInt(value);
        return ret;
    } catch(error) {
        return 0;
    }
}

async function getAMDGpuStats(isWindows: boolean) {
  // Execute command
  const command = 'amd-smi static --json && echo ";" && amd-smi metric --json';
  // Execute command
  const { stdout } = await execAsync(command, {
    env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
  });
  var data = stdout.split(';');

  var sdata = {};
  var mdata = {};
  try {
      sdata = JSON.parse(data[0]);
      mdata = JSON.parse(data[1]);
  } catch (error) {
    console.error('Failed to parse output of amd-smi returned json: ', error);
    return [];
  }

  var gpus = sdata["gpu_data"].map(d => {
    const i = amdParseInt(d["gpu"]);
    const gpu_data = mdata["gpu_data"][i];
    const mem_total = amdParseFloat(gpu_data["mem_usage"]["total_vram"]["value"]);
    const mem_used =  amdParseFloat(gpu_data["mem_usage"]["used_vram"]["value"]);
    const mem_free =  amdParseFloat(gpu_data["mem_usage"]["free_visible_vram"]["value"]);
    const mem_utilization = ((1.0 - (mem_total - mem_free)) / mem_total) * 100;

    return {
      index: i,
      name: d["asic"]["market_name"],
      driverVersion: d["driver"]["version"],
      temperature: amdParseInt(gpu_data["temperature"]["hotspot"]["value"]),
      utilization: {
        gpu: amdParseInt(gpu_data["usage"]["gfx_activity"]["value"]),
        memory: mem_utilization,
      },
      memory: {
        total: mem_total,
        used:  mem_used,
        free:  mem_free,
      },
      power: {
        draw: amdParseFloat(gpu_data["power"]["socket_power"]["value"]),
        limit: amdParseFloat(() => {
	  try {
	    if (d["limit"]["max_power"]) {
	      return d["limit"]["max_power"]["value"];
	    } else if (d["limit"]["ppt0"]["max_power_limit"]["value"]) {
	      return d["limit"]["ppt0"]["max_power_limit"]["value"];
	    }
	  } catch (error) {
	    return 0.0;
	  }
	})
      },
      clocks: {
        graphics: amdParseInt(gpu_data["clock"]["gfx_0"]["clk"]["value"]),
        memory: amdParseInt(gpu_data["clock"]["mem_0"]["clk"]["value"]),
      },
      fan: {
        speed: amdParseFloat(gpu_data["fan"]["usage"]["value"]),
      }
    };
  });

  return gpus;
}
