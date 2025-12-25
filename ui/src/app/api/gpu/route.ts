import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import path from 'path';
import fs from 'fs';

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
        hasAmdSmi: false,
        hasRocmSmi: false,
        gpus: gpuStats,
      });
    }

    // Check for AMD GPUs - prioritize amd-smi, fallback to rocm-smi
    const hasAmdSmi = await checkAmdSmi(isWindows);
    if (hasAmdSmi) {
      const gpuStats = await getAmdSmiGpuStats(isWindows);
      // If amd-smi returns data, use it
      if (gpuStats && gpuStats.length > 0) {
        return NextResponse.json({
          hasNvidiaSmi: false,
          hasAmdSmi: true,
          hasRocmSmi: false,
          gpus: gpuStats,
        });
      }
      // If amd-smi didn't return sufficient data, fallback to rocm-smi
      if (process.env.NODE_ENV === 'development') {
        console.log('[GPU] amd-smi returned insufficient data, falling back to rocm-smi');
      }
    }

    // Fallback to rocm-smi if amd-smi is not available or didn't return data
    const hasRocmSmi = await checkRocmSmi(isWindows);
    if (hasRocmSmi) {
      const gpuStats = await getRocmGpuStats(isWindows);
      return NextResponse.json({
        hasNvidiaSmi: false,
        hasAmdSmi: hasAmdSmi, // Indicate if amd-smi was checked but failed
        hasRocmSmi: true,
        gpus: gpuStats,
      });
    }

    // No GPU detection available
    return NextResponse.json({
      hasNvidiaSmi: false,
      hasAmdSmi: false,
      hasRocmSmi: false,
      gpus: [],
      error: 'Neither nvidia-smi, amd-smi, nor rocm-smi found. GPU detection unavailable.',
    });
  } catch (error) {
    console.error('Error fetching GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        hasAmdSmi: false,
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

async function checkAmdSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      // On Windows, try to run amd-smi directly (may be in PATH or Program Files)
      await execAsync('amd-smi --help');
    } else {
      // Linux/macOS check
      await execAsync('which amd-smi');
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

// Helper function to get venv PATH
function getVenvPath(isWindows: boolean): NodeJS.ProcessEnv {
  const projectRoot = path.resolve(process.cwd(), '..');
  let env: NodeJS.ProcessEnv = { ...process.env };
  
  // Check for venv and add its bin directory to PATH
  const venvPaths = [
    path.join(projectRoot, '.venv', 'bin'),
    path.join(projectRoot, 'venv', 'bin'),
  ];
  
  for (const venvBin of venvPaths) {
    if (fs.existsSync(venvBin)) {
      const currentPath = process.env.PATH || '';
      // Use appropriate path separator for the platform
      const pathSeparator = isWindows ? ';' : ':';
      env.PATH = `${venvBin}${pathSeparator}${currentPath}`;
      if (process.env.NODE_ENV === 'development') {
        console.log(`[AMD-SMI] Using venv PATH: ${venvBin}`);
      }
      break;
    }
  }
  
  return env;
}

// Helper function to parse N/A values
function parseValue(value: string | undefined, defaultValue: number = 0): number {
  if (!value || value === 'N/A' || value.trim() === '') {
    return defaultValue;
  }
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
}

async function getAmdSmiGpuStats(isWindows: boolean) {
  try {
    const env = getVenvPath(isWindows);
    
    // First, get list of GPUs
    const listCommand = 'amd-smi list --json';
    const { stdout: listStdout } = await execAsync(listCommand, { env });
    
    let gpuList: Array<{ gpu: number }> = [];
    try {
      const listData = JSON.parse(listStdout);
      if (Array.isArray(listData)) {
        gpuList = listData;
      } else if (listData && Array.isArray(listData.gpu_data)) {
        gpuList = listData.gpu_data;
      }
    } catch (parseError) {
      console.error('[AMD-SMI] Failed to parse GPU list JSON:', parseError);
      return [];
    }
    
    if (gpuList.length === 0) {
      console.warn('[AMD-SMI] No GPUs found in list');
      return [];
    }
    
    // Get metrics for each GPU using CSV format
    const gpus = await Promise.all(
      gpuList.map(async (gpuInfo) => {
        const gpuId = gpuInfo.gpu;
        const metricCommand = `amd-smi metric --gpu ${gpuId} --csv`;
        
        try {
          const { stdout: metricStdout } = await execAsync(metricCommand, { env });
          const lines = metricStdout.trim().split('\n').filter(line => line.trim().length > 0);
          
          if (lines.length < 2) {
            console.warn(`[AMD-SMI] No data for GPU ${gpuId}`);
            return null;
          }
          
          // Parse CSV header to find field indices
          const header = lines[0].split(',');
          const dataLine = lines[1].split(',');
          
          // Find field indices dynamically - try multiple possible field names
          const getFieldIndex = (fieldNames: string | string[]): number => {
            const names = Array.isArray(fieldNames) ? fieldNames : [fieldNames];
            for (const name of names) {
              const idx = header.findIndex(h => h.toLowerCase().includes(name.toLowerCase()));
              if (idx >= 0) return idx;
            }
            return -1;
          };
          
          const gpuIndex = getFieldIndex('gpu');
          const usageIndex = getFieldIndex(['usage', 'gpu_use', 'utilization']);
          // Temperature can be edge, temperature, temp, or junction
          const edgeIndex = getFieldIndex(['edge', 'temperature', 'temp', 'junction']);
          const totalVramIndex = getFieldIndex(['total_vram', 'vram_total', 'memory_total']);
          const usedVramIndex = getFieldIndex(['used_vram', 'vram_used', 'memory_used']);
          const freeVramIndex = getFieldIndex(['free_vram', 'vram_free', 'memory_free']);
          const socketPowerIndex = getFieldIndex(['socket_power', 'power', 'power_draw', 'tdp']);
          // Fan speed - look for max (percentage) or rpm
          const fanMaxIndex = getFieldIndex(['fan_max', 'fan_speed', 'max', 'fan_percent']);
          const fanRpmIndex = getFieldIndex(['fan_rpm', 'rpm']);
          
          // Find first available graphics clock (gfx_0_clk, gfx_1_clk, etc.)
          let gfxClkIndex = -1;
          for (let i = 0; i < header.length; i++) {
            if (header[i].toLowerCase().startsWith('gfx_') && header[i].toLowerCase().endsWith('_clk')) {
              gfxClkIndex = i;
              break;
            }
          }
          
          // Find memory clock (mem_0_clk)
          const memClkIndex = getFieldIndex('mem_0_clk');
          
          // Parse values
          const index = gpuIndex >= 0 ? parseInt(dataLine[gpuIndex] || '0') || gpuId : gpuId;
          const usage = usageIndex >= 0 ? parseValue(dataLine[usageIndex]) : 0;
          const temperature = edgeIndex >= 0 ? parseValue(dataLine[edgeIndex]) : 0;
          const memoryTotalMB = totalVramIndex >= 0 ? parseValue(dataLine[totalVramIndex]) : 0;
          const memoryUsedMB = usedVramIndex >= 0 ? parseValue(dataLine[usedVramIndex]) : 0;
          const memoryFreeMB = freeVramIndex >= 0 ? parseValue(dataLine[freeVramIndex]) : (memoryTotalMB - memoryUsedMB);
          const powerDraw = socketPowerIndex >= 0 ? parseValue(dataLine[socketPowerIndex]) : 0;
          const clockGraphics = gfxClkIndex >= 0 ? parseValue(dataLine[gfxClkIndex]) : 0;
          const clockMemory = memClkIndex >= 0 ? parseValue(dataLine[memClkIndex]) : 0;
          
          // Parse fan speed - prefer percentage (max) over RPM
          let fanSpeed = 0;
          if (fanMaxIndex >= 0 && dataLine[fanMaxIndex] && dataLine[fanMaxIndex] !== 'N/A') {
            fanSpeed = parseValue(dataLine[fanMaxIndex]);
          } else if (fanRpmIndex >= 0 && dataLine[fanRpmIndex] && dataLine[fanRpmIndex] !== 'N/A') {
            // If we have RPM but not percentage, we can't convert without max RPM, so leave as 0
            // In the future, we could store RPM separately if needed
            fanSpeed = 0;
          }
          
          // Get GPU name from static info - try multiple possible paths
          let name = `AMD GPU ${index}`;
          try {
            const staticCommand = `amd-smi static --gpu ${gpuId} --json`;
            const { stdout: staticStdout } = await execAsync(staticCommand, { env });
            const staticData = JSON.parse(staticStdout);
            
            // Try multiple possible paths for GPU name
            if (staticData && staticData.gpu_data && staticData.gpu_data[0]) {
              const gpuData = staticData.gpu_data[0];
              // Try market_name first (most common)
              if (gpuData.asic && gpuData.asic.market_name) {
                name = gpuData.asic.market_name;
              }
              // Fallback to other possible name fields
              else if (gpuData.asic && gpuData.asic.name) {
                name = gpuData.asic.name;
              }
              else if (gpuData.card && gpuData.card.market_name) {
                name = gpuData.card.market_name;
              }
              else if (gpuData.card && gpuData.card.name) {
                name = gpuData.card.name;
              }
              else if (gpuData.name) {
                name = gpuData.name;
              }
            }
          } catch (staticError) {
            // If static info fails, use default name
            if (process.env.NODE_ENV === 'development') {
              console.warn(`[AMD-SMI] Failed to get static info for GPU ${gpuId}:`, staticError);
            }
          }
          
          // Calculate memory utilization
          const memoryUtilPercent = memoryTotalMB > 0 
            ? Math.max(0, Math.min(100, Math.round((memoryUsedMB / memoryTotalMB) * 100)))
            : 0;
          
          // Validate values
          const validTemperature = temperature >= 0 && temperature <= 200 ? temperature : 0;
          const validUsage = Math.max(0, Math.min(100, usage));
          const validFanSpeed = Math.max(0, Math.min(100, fanSpeed));
          
          // Check if we have sufficient data
          // We need at least temperature OR memory
          // But if critical performance metrics (usage, power, clocks) are all missing, 
          // we should fallback to rocm-smi which likely has them
          const hasBasicData = validTemperature > 0 || memoryTotalMB > 0;
          const hasPerformanceData = validUsage > 0 || powerDraw > 0 || clockGraphics > 0 || clockMemory > 0;
          // If we have basic data but no performance data, it's better to use rocm-smi
          const hasSufficientData = hasBasicData && hasPerformanceData;
          
          if (process.env.NODE_ENV === 'development') {
            console.log(`[AMD-SMI GPU ${index}] temp=${validTemperature}°C, mem=${memoryTotalMB}MB, usage=${validUsage}%, power=${powerDraw}W, gfxClk=${clockGraphics}MHz, memClk=${clockMemory}MHz, hasData=${hasSufficientData}`);
          }
          
          return {
            index,
            name,
            driverVersion: 'ROCm',
            temperature: validTemperature > 0 ? Math.round(validTemperature) : 0,
            utilization: {
              gpu: Math.round(validUsage),
              memory: memoryUtilPercent,
            },
            memory: {
              total: Math.max(0, Math.round(memoryTotalMB)),
              free: Math.max(0, Math.round(memoryFreeMB)),
              used: Math.max(0, Math.round(memoryUsedMB)),
            },
            power: {
              draw: Math.max(0, powerDraw),
              limit: 0, // amd-smi doesn't provide power limit in metric output
            },
            clocks: {
              graphics: Math.max(0, Math.round(clockGraphics)),
              memory: Math.max(0, Math.round(clockMemory)),
            },
            fan: {
              speed: validFanSpeed > 0 ? Math.round(validFanSpeed) : 0,
            },
            _hasSufficientData: hasSufficientData, // Internal flag for fallback logic
          };
        } catch (error) {
          console.error(`[AMD-SMI] Error getting metrics for GPU ${gpuId}:`, error);
          return null;
        }
      })
    );
    
    // Filter out null results
    const validGpus = gpus.filter((gpu): gpu is NonNullable<typeof gpu> => gpu !== null);
    
    // Check if we have sufficient data - if not, return empty to trigger fallback
    const hasAnyData = validGpus.some(gpu => gpu._hasSufficientData);
    if (!hasAnyData && validGpus.length > 0) {
      if (process.env.NODE_ENV === 'development') {
        console.warn('[AMD-SMI] Insufficient data from amd-smi (missing usage/power/clocks), will fallback to rocm-smi');
        validGpus.forEach(gpu => {
          console.log(`[AMD-SMI] GPU ${gpu.index}: usage=${gpu.utilization.gpu}%, power=${gpu.power.draw}W, gfxClk=${gpu.clocks.graphics}MHz, memClk=${gpu.clocks.memory}MHz`);
        });
      }
      return [];
    }
    
    // Remove internal flag before returning
    const result = validGpus.map(({ _hasSufficientData, ...gpu }) => gpu);
    
    if (process.env.NODE_ENV === 'development') {
      console.log(`[AMD-SMI] Returning ${result.length} GPU(s) with data`);
      result.forEach(gpu => {
        console.log(`[AMD-SMI] GPU ${gpu.index}: usage=${gpu.utilization.gpu}%, power=${gpu.power.draw}W, gfxClk=${gpu.clocks.graphics}MHz, memClk=${gpu.clocks.memory}MHz`);
      });
    }
    
    return result;
  } catch (error) {
    console.error('[AMD-SMI] Error getting GPU stats:', error);
    return [];
  }
}

async function getRocmGpuStats(isWindows: boolean) {
  // Get GPU list using CSV format
  const command =
    'rocm-smi --showid --showproductname --showtemp --showuse --showmemuse --showmeminfo vram --showpower --showclocks --csv';

  try {
    const env = getVenvPath(isWindows);
    if (process.env.NODE_ENV === 'development') {
      console.log(`[ROCm] Executing: ${command}`);
      console.log(`[ROCm] PATH: ${env.PATH?.substring(0, 200)}`);
    }
    const { stdout, stderr } = await execAsync(command, { env });
    
    if (process.env.NODE_ENV === 'development') {
      console.log(`[ROCm] stdout length: ${stdout.length}, stderr length: ${stderr.length}`);
      console.log(`[ROCm] First 500 chars of stdout:`, stdout.substring(0, 500));
    }
    
    // Filter out error messages and empty lines, keep only CSV data
    const lines = stdout
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0 && !line.startsWith('Exception') && !line.startsWith('Error'));
    
    // Find the header line (should contain "device,GPU ID")
    const headerIndex = lines.findIndex(line => line.includes('device,GPU ID') || line.startsWith('device,'));
    
    if (headerIndex === -1 || lines.length < headerIndex + 2) {
      console.error('[ROCm] No valid CSV header found or no data lines');
      console.error('[ROCm] Available lines:', lines.slice(0, 5));
      return [];
    }

    // Helper function to parse CSV line properly (handles quoted fields)
    function parseCSVLine(line: string): string[] {
      const fields: string[] = [];
      let current = '';
      let inQuotes = false;
      
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          fields.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      fields.push(current.trim()); // Add last field
      return fields;
    }

    // Skip header line and process data lines
    const gpus = lines.slice(headerIndex + 1).map((line, idx) => {
      // Parse CSV line - rocm-smi CSV format has changed!
      // New format (25 fields): device,Device Name,Device ID,Device Rev,Subsystem ID,GUID,Temperature (Sensor edge) (C),mclk clock speed:,mclk clock level:,sclk clock speed:,sclk clock level:,socclk clock speed:,socclk clock level:,Current Socket Graphics Package Power (W),GPU use (%),GPU Memory Allocated (VRAM%),Memory Activity,VRAM Total Memory (B),VRAM Total Used Memory (B),Card Series,Card Model,Card Vendor,Card SKU,Node ID,GFX Version
      // Old format (18 fields): device,GPU ID,Temperature,mclk clock speed,mclk level,sclk clock speed,sclk level,socclk speed,socclk level,Power,GPU use,Memory Activity,VRAM Total,VRAM Used,Card series,Card model,Card vendor,Card SKU
      const fields = parseCSVLine(line);
      
      // Detect format based on field count
      const isNewFormat = fields.length >= 25;
      const isOldFormat = fields.length >= 18 && fields.length < 25;
      
      if (!isNewFormat && !isOldFormat) {
        console.error(`[ROCm GPU ${idx}] Unexpected field count: got ${fields.length}, expected 18 (old) or 25 (new)`);
        console.error(`[ROCm GPU ${idx}] Line: ${line.substring(0, 200)}`);
        console.error(`[ROCm GPU ${idx}] Parsed fields:`, fields);
        // Pad with empty strings to prevent index errors
        while (fields.length < 25) {
          fields.push('');
        }
      }
      
      // Debug logging in development - log ALL fields to diagnose
      if (process.env.NODE_ENV === 'development' && idx === 0) {
        console.log(`[ROCm] Parsing line with ${fields.length} fields (format: ${isNewFormat ? 'NEW' : 'OLD'})`);
        console.log(`[ROCm] Raw line: ${line.substring(0, 200)}`);
        console.log(`[ROCm] All fields:`, fields.map((f, i) => `[${i}]="${f}"`).join(', '));
        if (isNewFormat) {
          console.log(`[ROCm] Key fields - [1]Device Name="${fields[1]}", [6]Temp="${fields[6]}", [7]mclk="${fields[7]}", [9]sclk="${fields[9]}", [13]Power="${fields[13]}", [14]Usage="${fields[14]}"`);
        } else {
          console.log(`[ROCm] Key fields - [1]GPU ID="${fields[1]}", [2]Temp="${fields[2]}", [3]mclk="${fields[3]}", [5]sclk="${fields[5]}", [9]Power="${fields[9]}", [10]Usage="${fields[10]}"`);
        }
      }
      
      // Parse device name (card0, card1, etc.) to get index
      const deviceName = fields[0]?.trim() || '';
      // Extract numeric part from device name (e.g., "card0" -> 0)
      const deviceMatch = deviceName.match(/\d+/);
      const index = deviceMatch ? parseInt(deviceMatch[0]) : idx;
      
      // Extract fields based on format
      // New format: Temperature at field 6, mclk at field 7, sclk at field 9, Power at field 13, Usage at field 14, Memory Total at field 17, Memory Used at field 18
      // Old format: Temperature at field 2, mclk at field 3, sclk at field 5, Power at field 9, Usage at field 10, Memory Total at field 12, Memory Used at field 13
      const tempFieldIdx = isNewFormat ? 6 : 2;
      const mclkFieldIdx = isNewFormat ? 7 : 3;
      const sclkFieldIdx = isNewFormat ? 9 : 5;
      const powerFieldIdx = isNewFormat ? 13 : 9;
      const usageFieldIdx = isNewFormat ? 14 : 10;
      const memTotalFieldIdx = isNewFormat ? 17 : 12;
      const memUsedFieldIdx = isNewFormat ? 18 : 13;
      const cardSkuFieldIdx = isNewFormat ? 22 : 17;
      const cardModelFieldIdx = isNewFormat ? 20 : 15;
      const cardNameFieldIdx = isNewFormat ? 1 : -1; // Device Name in new format
      
      const tempStr = fields[tempFieldIdx]?.trim() || '';
      // Parse temperature - rocm-smi provides temperature in Celsius
      let temperature = 0;
      if (tempStr && tempStr !== 'N/A' && !isNaN(parseFloat(tempStr))) {
        const tempValue = parseFloat(tempStr);
        // Validate temperature is in reasonable range (0-200°C)
        if (tempValue >= 0 && tempValue <= 200) {
          temperature = tempValue;
        }
      }
      
      // Debug logging in development
      if (process.env.NODE_ENV === 'development' && idx === 0) {
        console.log(`[ROCm GPU ${index}] Format: ${isNewFormat ? 'NEW (25 fields)' : 'OLD (18 fields)'}`);
        console.log(`[ROCm GPU ${index}] Temperature raw (field[${tempFieldIdx}]): "${tempStr}", parsed: ${temperature}`);
      }
      
      // GPU use (%) - field index depends on format
      if (fields.length < usageFieldIdx + 1) {
        if (process.env.NODE_ENV === 'development') {
          console.error(`[ROCm GPU ${index}] ERROR: Not enough fields for usage! Expected at least ${usageFieldIdx + 1}, got ${fields.length}`);
        }
      }
      const gpuUtilStr = fields[usageFieldIdx]?.trim() || '0';
      let gpuUtil = 0;
      if (gpuUtilStr && gpuUtilStr !== 'N/A' && !isNaN(parseFloat(gpuUtilStr))) {
        const parsed = parseFloat(gpuUtilStr);
        // Validate it's a reasonable percentage (0-100)
        if (parsed >= 0 && parsed <= 100) {
          gpuUtil = parsed;
        } else if (process.env.NODE_ENV === 'development') {
          console.error(`[ROCm GPU ${index}] ERROR: Invalid usage value "${gpuUtilStr}" (parsed as ${parsed})`);
        }
      } else if (process.env.NODE_ENV === 'development') {
        console.error(`[ROCm GPU ${index}] ERROR: Could not parse usage from field[${usageFieldIdx}]="${gpuUtilStr}"`);
      }
      // rocm-smi GPU use is already a percentage, but validate and clamp to 0-100
      gpuUtil = Math.max(0, Math.min(100, gpuUtil));
      
      // Debug logging in development
      if (process.env.NODE_ENV === 'development' && idx === 0) {
        console.log(`[ROCm GPU ${index}] Usage - field[${usageFieldIdx}]="${gpuUtilStr}", parsed=${gpuUtil}%`);
      }
      
      // Memory values from rocm-smi are in bytes, but check if they're valid
      // Field indices depend on format
      if (fields.length < memUsedFieldIdx + 1) {
        if (process.env.NODE_ENV === 'development') {
          console.error(`[ROCm GPU ${index}] Insufficient fields: expected at least ${memUsedFieldIdx + 1}, got ${fields.length}`);
        }
      }
      
      const memoryTotalStr = fields[memTotalFieldIdx]?.trim() || '0';
      const memoryUsedStr = fields[memUsedFieldIdx]?.trim() || '0';
      let memoryTotal = parseFloat(memoryTotalStr) || 0;
      let memoryUsed = parseFloat(memoryUsedStr) || 0;
      
      // Debug logging in development
      if (process.env.NODE_ENV === 'development' && idx === 0) {
        console.log(`[ROCm GPU ${index}] Memory raw: total (field[${memTotalFieldIdx}])="${memoryTotalStr}", used (field[${memUsedFieldIdx}])="${memoryUsedStr}"`);
        console.log(`[ROCm GPU ${index}] Memory parsed: total=${memoryTotal}, used=${memoryUsed}`);
      }
      
      // Validate memory values - ensure they're positive and used <= total
      if (memoryTotal < 0 || isNaN(memoryTotal)) memoryTotal = 0;
      if (memoryUsed < 0 || isNaN(memoryUsed)) memoryUsed = 0;
      if (memoryUsed > memoryTotal) memoryUsed = memoryTotal; // Clamp used to total
      
      const memoryFree = Math.max(0, memoryTotal - memoryUsed);
      // Power draw - field index depends on format
      if (fields.length < powerFieldIdx + 1) {
        if (process.env.NODE_ENV === 'development') {
          console.error(`[ROCm GPU ${index}] ERROR: Not enough fields for power! Expected at least ${powerFieldIdx + 1}, got ${fields.length}`);
        }
      }
      const powerDrawStr = fields[powerFieldIdx]?.trim() || '';
      // Parse power draw, handle cases where it might be in different formats
      let powerDraw = 0;
      // Check if the field looks like a clock value (contains "Mhz" or "MHz") and skip it
      if (powerDrawStr && powerDrawStr.toLowerCase().includes('mhz')) {
        // This field contains a clock value, not power - skip parsing
        if (process.env.NODE_ENV === 'development') {
          console.warn(`[ROCm GPU ${index}] Power field[${powerFieldIdx}] contains clock value "${powerDrawStr}", skipping`);
        }
      } else if (powerDrawStr && powerDrawStr !== 'N/A') {
        // Try to extract numeric value (handle formats like "123.45" or "(123.45)")
        const powerMatch = powerDrawStr.match(/(\d+\.?\d*)/);
        if (powerMatch) {
          const parsed = parseFloat(powerMatch[1]);
          // Validate it's a reasonable power value (0-1000W)
          if (parsed >= 0 && parsed <= 1000) {
            powerDraw = parsed;
          } else if (process.env.NODE_ENV === 'development') {
            console.error(`[ROCm GPU ${index}] ERROR: Invalid power value "${powerDrawStr}" (parsed as ${parsed}W)`);
          }
        } else if (process.env.NODE_ENV === 'development') {
          console.error(`[ROCm GPU ${index}] ERROR: Could not parse power from field[${powerFieldIdx}]="${powerDrawStr}"`);
        }
      }
      
      // Debug logging in development
      if (process.env.NODE_ENV === 'development' && idx === 0) {
        console.log(`[ROCm GPU ${index}] Power - field[${powerFieldIdx}]="${powerDrawStr}", parsed=${powerDraw}W`);
      }
      
      // Parse clock speeds (format: "(1000Mhz)" -> 1000)
      // mclk = memory clock, sclk = graphics/core clock
      // Field indices depend on format
      const mclkStr = fields[mclkFieldIdx]?.trim() || '(0Mhz)';
      const sclkStr = fields[sclkFieldIdx]?.trim() || '(0Mhz)';
      
      // Debug logging to verify we're reading the right fields
      if (process.env.NODE_ENV === 'development' && idx === 0) {
        console.log(`[ROCm GPU ${index}] Format: ${isNewFormat ? 'NEW (25 fields)' : 'OLD (18 fields)'}`);
        console.log(`[ROCm GPU ${index}] Raw clock fields - field[${mclkFieldIdx}]="${fields[mclkFieldIdx]}", field[${sclkFieldIdx}]="${fields[sclkFieldIdx]}"`);
        console.log(`[ROCm GPU ${index}] Parsed clock strings - mclkStr="${mclkStr}", sclkStr="${sclkStr}"`);
      }
      
      // Extract numeric value from clock strings like "(1000Mhz)" or "1000Mhz"
      // Handle both formats: "(1472Mhz)" and raw numbers
      const mclkMatch = mclkStr.match(/(\d+)/);
      const sclkMatch = sclkStr.match(/(\d+)/);
      // Graphics clock is sclk (system/core clock), memory clock is mclk
      let clockGraphics = sclkMatch ? parseInt(sclkMatch[1]) : 0;
      let clockMemory = mclkMatch ? parseInt(mclkMatch[1]) : 0;
      
      // Validate clock speeds are reasonable (0-5000 MHz for graphics, 0-3000 MHz for memory)
      // If the value is way too high, it might be in Hz instead of MHz - convert it
      if (clockGraphics > 10000) {
        // Likely in Hz, convert to MHz
        clockGraphics = Math.round(clockGraphics / 1000000);
        if (process.env.NODE_ENV === 'development') {
          console.warn(`[ROCm GPU ${index}] Graphics clock appears to be in Hz (${clockGraphics * 1000000}), converted to ${clockGraphics}MHz`);
        }
      }
      if (clockMemory > 10000) {
        // Likely in Hz, convert to MHz
        clockMemory = Math.round(clockMemory / 1000000);
        if (process.env.NODE_ENV === 'development') {
          console.warn(`[ROCm GPU ${index}] Memory clock appears to be in Hz (${clockMemory * 1000000}), converted to ${clockMemory}MHz`);
        }
      }
      
      // Final validation after potential conversion
      if (clockGraphics > 5000 || clockGraphics < 0) {
        if (process.env.NODE_ENV === 'development') {
          console.error(`[ROCm GPU ${index}] ERROR: Invalid graphics clock ${clockGraphics}MHz from field[${sclkFieldIdx}]="${fields[sclkFieldIdx]}"`);
        }
        clockGraphics = 0;
      }
      if (clockMemory > 3000 || clockMemory < 0) {
        if (process.env.NODE_ENV === 'development') {
          console.error(`[ROCm GPU ${index}] ERROR: Invalid memory clock ${clockMemory}MHz from field[${mclkFieldIdx}]="${fields[mclkFieldIdx]}"`);
        }
        clockMemory = 0;
      }
      
      // Debug logging in development
      if (process.env.NODE_ENV === 'development' && idx === 0) {
        console.log(`[ROCm GPU ${index}] Clocks parsed - mclk: ${clockMemory}MHz, sclk: ${clockGraphics}MHz`);
        console.log(`[ROCm GPU ${index}] All field values:`, {
          field0: fields[0],
          field1: fields[1],
          field2: fields[2],
          field3: fields[3],
          field4: fields[4],
          field5: fields[5],
          field9: fields[9],
          field10: fields[10],
        });
      }
      
      // Get GPU name from Card SKU (most descriptive), then Card model, then Device Name (new format), then fallback
      // Field indices depend on format
      const cardSku = fields.length > cardSkuFieldIdx ? (fields[cardSkuFieldIdx]?.trim() || '') : '';
      const cardModel = fields.length > cardModelFieldIdx ? (fields[cardModelFieldIdx]?.trim() || '') : '';
      const deviceNameField = cardNameFieldIdx >= 0 && fields.length > cardNameFieldIdx ? (fields[cardNameFieldIdx]?.trim() || '') : '';
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

      const gpuData = {
        index: isNaN(index) ? idx : index,
        name,
        driverVersion: 'ROCm', // rocm-smi doesn't provide driver version in CSV
        temperature: temperature > 0 ? Math.round(temperature) : 0, // Temperature already validated above
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
      
      // Debug logging in development
      if (process.env.NODE_ENV === 'development' && idx === 0) {
        console.log(`[ROCm GPU ${gpuData.index}] Final values: usage=${gpuData.utilization.gpu}%, power=${gpuData.power.draw}W, gfxClk=${gpuData.clocks.graphics}MHz, memClk=${gpuData.clocks.memory}MHz`);
      }
      
      return gpuData;
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
