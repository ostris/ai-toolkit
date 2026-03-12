import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import { getMacGpuTelemetryPath, readMacGpuTelemetrySnapshot } from '@/server/macGpuTelemetry';

const execAsync = promisify(exec);
const UNKNOWN_METRIC = -1;

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';
    const isMac = platform === 'darwin';

    if (isMac) {
      return NextResponse.json(await getMacGpuStats());
    }

    // Check if nvidia-smi is available
    const hasNvidiaSmi = await checkNvidiaSmi(isWindows);

    if (!hasNvidiaSmi) {
      return NextResponse.json({
        hasNvidiaSmi: false,
        gpus: [],
        error: 'nvidia-smi not found or not accessible',
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

function toKnownMetric(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : UNKNOWN_METRIC;
}

function pickKnownMetric(...values: Array<number | undefined>): number {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
      return value;
    }
  }
  return UNKNOWN_METRIC;
}

function roundIfKnown(value: number, digits = 0): number {
  if (value === UNKNOWN_METRIC) {
    return UNKNOWN_METRIC;
  }
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function selectMacController(controllers: unknown[]) {
  if (!Array.isArray(controllers) || controllers.length === 0) {
    return null;
  }

  const appleController = controllers.find(controller => {
    if (!controller || typeof controller !== 'object') {
      return false;
    }
    const record = controller as Record<string, unknown>;
    const searchable = [record.vendor, record.model, record.name].filter(Boolean).join(' ').toLowerCase();
    return searchable.includes('apple');
  });

  return (appleController as Record<string, unknown> | undefined) ?? (controllers[0] as Record<string, unknown>);
}

async function getMacGpuStats() {
  const totalMemMb = Math.round(os.totalmem() / (1024 * 1024));
  const freeMemMb = Math.round(os.freemem() / (1024 * 1024));
  const usedMemMb = totalMemMb - freeMemMb;
  const fallbackMemoryUtil = totalMemMb > 0 ? Math.round((usedMemMb / totalMemMb) * 100) : UNKNOWN_METRIC;

  const telemetry = await readMacGpuTelemetrySnapshot();

  let controller: Record<string, unknown> | null = null;
  let systemInformationError: string | null = null;
  try {
    const siModule = await import('systeminformation');
    const si = siModule.default;
    const graphics = await si.graphics();
    controller = selectMacController(graphics.controllers as unknown[]);
  } catch (error) {
    systemInformationError = error instanceof Error ? error.message : String(error);
  }

  const controllerName =
    (typeof controller?.name === 'string' && controller.name) ||
    (typeof controller?.model === 'string' && controller.model) ||
    'Apple Silicon (MPS)';
  const driverVersion = (typeof controller?.driverVersion === 'string' && controller.driverVersion) || 'N/A';

  const memoryTotal = pickKnownMetric(toKnownMetric(controller?.memoryTotal), totalMemMb);
  const memoryUsed = pickKnownMetric(toKnownMetric(controller?.memoryUsed), usedMemMb);
  const memoryFree = pickKnownMetric(toKnownMetric(controller?.memoryFree), freeMemMb);
  const memoryUtil =
    memoryTotal > 0 && memoryUsed >= 0 ? Math.round((memoryUsed / memoryTotal) * 100) : fallbackMemoryUtil;

  const temperature = roundIfKnown(
    pickKnownMetric(
      telemetry?.temperatureC,
      toKnownMetric(controller?.temperatureGpu),
    ),
    1,
  );
  const powerDraw = roundIfKnown(
    pickKnownMetric(
      telemetry?.powerDrawW,
      toKnownMetric(controller?.powerDraw),
    ),
    2,
  );
  const powerLimit = roundIfKnown(
    pickKnownMetric(
      telemetry?.powerLimitW,
      toKnownMetric(controller?.powerLimit),
    ),
    2,
  );
  const clockGraphics = roundIfKnown(
    pickKnownMetric(
      telemetry?.clockMHz,
      toKnownMetric(controller?.clockCore),
    ),
  );
  const clockMemory = roundIfKnown(toKnownMetric(controller?.clockMemory));
  const utilizationGpu = roundIfKnown(
    pickKnownMetric(
      telemetry?.utilizationGpuPercent,
      toKnownMetric(controller?.utilizationGpu),
    ),
    1,
  );

  const fanRpm = roundIfKnown(
    pickKnownMetric(
      telemetry?.fanRpm,
    ),
  );
  const fanPercent = roundIfKnown(toKnownMetric(controller?.fanSpeed), 1);

  const errorParts: string[] = [];
  if (!telemetry) {
    errorParts.push(
      `No fresh macOS telemetry cache found at ${getMacGpuTelemetryPath()} (start collector for power/clock/temp/fan)`,
    );
  }
  if (systemInformationError) {
    errorParts.push(`systeminformation unavailable: ${systemInformationError}`);
  }

  return {
    hasNvidiaSmi: true,
    gpus: [
      {
        index: 0,
        name: controllerName,
        driverVersion,
        temperature,
        utilization: {
          gpu: utilizationGpu,
          memory: memoryUtil,
        },
        memory: {
          total: memoryTotal,
          free: memoryFree,
          used: memoryUsed,
        },
        power: {
          draw: powerDraw,
          limit: powerLimit,
        },
        clocks: {
          graphics: clockGraphics,
          memory: clockMemory,
        },
        fan: {
          speed: fanPercent,
          rpm: fanRpm,
        },
        telemetrySource: telemetry?.source ?? 'systeminformation',
      },
    ],
    ...(errorParts.length > 0 ? { error: errorParts.join(' | ') } : {}),
  };
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
