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

      // Validate and clamp values to prevent astronomical numbers and NaN
      const indexNum = parseInt(index) || 0;
      const tempNum = parseInt(temperature) || 0;
      const gpuUtilNum = parseInt(gpuUtil) || 0;
      const memoryUtilNum = parseInt(memoryUtil) || 0;
      const memoryTotalNum = parseInt(memoryTotal) || 0;
      const memoryFreeNum = parseInt(memoryFree) || 0;
      const memoryUsedNum = parseInt(memoryUsed) || 0;
      const powerDrawNum = parseFloat(powerDraw) || 0;
      const powerLimitNum = parseFloat(powerLimit) || 0;
      const clockGraphicsNum = parseInt(clockGraphics) || 0;
      const clockMemoryNum = parseInt(clockMemory) || 0;
      const fanSpeedNum = parseInt(fanSpeed) || 0;

      return {
        index: indexNum,
        name: name || `GPU ${indexNum}`,
        driverVersion: driverVersion || 'Unknown',
        temperature: Math.max(0, Math.min(200, tempNum)), // Clamp to reasonable range
        utilization: {
          gpu: Math.max(0, Math.min(100, gpuUtilNum)), // Clamp to 0-100%
          memory: Math.max(0, Math.min(100, memoryUtilNum)), // Clamp to 0-100%
        },
        memory: {
          total: Math.max(0, memoryTotalNum), // Ensure non-negative
          free: Math.max(0, Math.min(memoryTotalNum, memoryFreeNum)), // Clamp to total
          used: Math.max(0, Math.min(memoryTotalNum, memoryUsedNum)), // Clamp to total
        },
        power: {
          draw: Math.max(0, powerDrawNum), // Ensure non-negative
          limit: Math.max(0, powerLimitNum), // Ensure non-negative
        },
        clocks: {
          graphics: Math.max(0, clockGraphicsNum), // Ensure non-negative
          memory: Math.max(0, clockMemoryNum), // Ensure non-negative
        },
        fan: {
          speed: Math.max(0, Math.min(100, fanSpeedNum)), // Clamp to 0-100%
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
      // Parse CSV line - rocm-smi CSV is simple comma-separated, no quotes in our case
      // But handle cases where vendor name might have commas by being more careful
      const fields = line.split(',').map(f => f.trim());
      
      // Parse device name (card0, card1, etc.) to get index
      const deviceName = fields[0]?.trim() || '';
      // Extract numeric part from device name (e.g., "card0" -> 0)
      const deviceMatch = deviceName.match(/\d+/);
      const index = deviceMatch ? parseInt(deviceMatch[0]) : idx;
      
      // Extract fields - order: device,GPU ID,Temperature,mclk clock speed,mclk clock level,sclk clock speed,sclk clock level,socclk clock speed,socclk clock level,Power,GPU use,Memory Activity,VRAM Total Memory,VRAM Total Used Memory,Card series,Card model,Card vendor,Card SKU
      const tempStr = fields[2]?.trim() || '';
      // Only parse temperature if it's a valid number, otherwise use null/0
      let temperature = 0;
      if (tempStr && !isNaN(parseFloat(tempStr)) && parseFloat(tempStr) > 0) {
        temperature = parseFloat(tempStr);
      }
      
      let gpuUtil = parseFloat(fields[10]?.trim() || '0') || 0;
      // rocm-smi GPU use is already a percentage, but validate and clamp to 0-100
      gpuUtil = Math.max(0, Math.min(100, gpuUtil));
      
      // Memory values from rocm-smi are in bytes, but check if they're valid
      let memoryTotal = parseFloat(fields[12]?.trim() || '0') || 0;
      let memoryUsed = parseFloat(fields[13]?.trim() || '0') || 0;
      
      // Validate memory values - ensure they're positive and used <= total
      if (memoryTotal < 0 || isNaN(memoryTotal)) memoryTotal = 0;
      if (memoryUsed < 0 || isNaN(memoryUsed)) memoryUsed = 0;
      if (memoryUsed > memoryTotal) memoryUsed = memoryTotal; // Clamp used to total
      
      const memoryFree = Math.max(0, memoryTotal - memoryUsed);
      const powerDrawStr = fields[9]?.trim() || '';
      // Parse power draw, handle cases where it might be in different formats
      let powerDraw = 0;
      if (powerDrawStr && !isNaN(parseFloat(powerDrawStr))) {
        powerDraw = parseFloat(powerDrawStr);
      }
      
      // Parse clock speeds (format: "(1000Mhz)" -> 1000)
      const mclkStr = fields[3]?.trim() || '(0Mhz)';
      const sclkStr = fields[5]?.trim() || '(0Mhz)';
      // Extract numeric value from clock strings like "(1000Mhz)" or "1000Mhz"
      const mclkMatch = mclkStr.match(/(\d+)/);
      const sclkMatch = sclkStr.match(/(\d+)/);
      const clockGraphics = mclkMatch ? parseInt(mclkMatch[1]) : 0;
      const clockMemory = sclkMatch ? parseInt(sclkMatch[1]) : 0;
      
      // Get GPU name from Card SKU (most descriptive), then Card model, then fallback
      // CSV fields: device,GPU ID,Temperature,...,Card series,Card model,Card vendor,Card SKU
      // Make sure we have enough fields (should be 18 fields: 0-17)
      const cardSku = fields.length > 17 ? (fields[17]?.trim() || '') : '';
      const cardModel = fields.length > 15 ? (fields[15]?.trim() || '') : '';
      const cardVendor = fields.length > 16 ? (fields[16]?.trim() || '') : '';
      const gpuId = fields[1]?.trim() || '';
      
      // Debug: log field counts and values to help diagnose parsing issues
      if (process.env.NODE_ENV === 'development') {
        console.log(`[GPU ${index}] Fields count: ${fields.length}, Card SKU: "${cardSku}", Card Model: "${cardModel}", Card Vendor: "${cardVendor}"`);
      }
      
      // Use Card SKU if available and not a hex ID or numeric ID, otherwise prefer Card model, then fallback
      let name = '';
      // Check Card SKU first - it should be a descriptive name like "STRXLGEN"
      if (cardSku && 
          cardSku.length > 0 &&
          !cardSku.startsWith('0x') && 
          !/^\d+$/.test(cardSku) && 
          cardSku !== gpuId &&
          cardSku !== memoryTotal.toString() &&
          cardSku !== memoryUsed.toString()) {
        // Card SKU is valid and not a numeric/hex ID
        name = cardSku;
      } else if (cardModel && 
                 cardModel.length > 0 && 
                 cardModel !== gpuId && 
                 !cardModel.startsWith('0x') &&
                 cardModel !== memoryTotal.toString()) {
        // Only use card model if it's not a hex ID or numeric value
        name = cardModel;
      } else if (cardVendor && (cardVendor.includes('AMD') || cardVendor.includes('Advanced Micro Devices'))) {
        // Fallback to vendor-based name
        name = `AMD GPU ${index}`;
      } else {
        // Final fallback
        name = `GPU ${index}`;
      }
      
      // Safety check: ensure name is never a numeric memory value
      if (name === memoryTotal.toString() || name === memoryUsed.toString() || /^\d+$/.test(name)) {
        name = `AMD GPU ${index}`;
      }

      // Convert memory from bytes to MB (rocm-smi reports in bytes)
      // Check if values are already in MB/GB by checking magnitude
      let memoryTotalMB = 0;
      let memoryUsedMB = 0;
      let memoryFreeMB = 0;
      
      if (memoryTotal > 0) {
        // If value is very large (> 1TB), assume bytes and convert to MB
        // If value is reasonable (< 1000), assume already in GB and convert to MB
        if (memoryTotal > 1024 * 1024 * 1024) {
          // Bytes - convert to MB
          memoryTotalMB = Math.round(memoryTotal / (1024 * 1024));
          memoryUsedMB = Math.round(memoryUsed / (1024 * 1024));
          memoryFreeMB = Math.round(memoryFree / (1024 * 1024));
        } else if (memoryTotal > 1000) {
          // Already in MB
          memoryTotalMB = Math.round(memoryTotal);
          memoryUsedMB = Math.round(memoryUsed);
          memoryFreeMB = Math.round(memoryFree);
        } else {
          // Assume GB, convert to MB
          memoryTotalMB = Math.round(memoryTotal * 1024);
          memoryUsedMB = Math.round(memoryUsed * 1024);
          memoryFreeMB = Math.round(memoryFree * 1024);
        }
      }

      // Calculate memory utilization percentage safely
      const memoryUtilPercent = memoryTotalMB > 0 
        ? Math.max(0, Math.min(100, Math.round((memoryUsedMB / memoryTotalMB) * 100)))
        : 0;

      // Validate temperature - only show if it's in a reasonable range (10-200°C)
      // Temperatures below 10°C are likely invalid readings, above 200°C is impossible
      const validTemperature = temperature >= 10 && temperature <= 200 ? temperature : 0;

      return {
        index: isNaN(index) ? idx : index,
        name,
        driverVersion: 'ROCm', // rocm-smi doesn't provide driver version in CSV
        temperature: validTemperature > 0 ? Math.round(validTemperature) : 0, // Use 0 if invalid, will be handled in UI
        utilization: {
          gpu: Math.round(gpuUtil),
          memory: memoryUtilPercent,
        },
        memory: {
          total: memoryTotalMB,
          free: memoryFreeMB,
          used: memoryUsedMB,
        },
        power: {
          draw: Math.max(0, powerDraw), // Ensure non-negative
          limit: 0, // rocm-smi CSV doesn't provide power limit (0 indicates unavailable)
        },
        clocks: {
          graphics: Math.max(0, clockGraphics),
          memory: Math.max(0, clockMemory),
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
