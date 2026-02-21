import { access, readFile } from 'fs/promises';
import { constants as fsConstants } from 'fs';

export interface MacGpuTelemetrySnapshot {
  timestampMs: number;
  source: string;
  temperatureC?: number;
  fanRpm?: number;
  powerDrawW?: number;
  powerLimitW?: number;
  clockMHz?: number;
  utilizationGpuPercent?: number;
}

const DEFAULT_TELEMETRY_PATH = '/tmp/ai-toolkit-mac-gpu-telemetry.json';
const DEFAULT_MAX_AGE_MS = 15_000;

function asFiniteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function getMaxAgeMs() {
  const fromEnv = Number(process.env.AI_TOOLKIT_MAC_GPU_TELEMETRY_MAX_AGE_MS);
  if (!Number.isFinite(fromEnv) || fromEnv <= 0) {
    return DEFAULT_MAX_AGE_MS;
  }
  return Math.floor(fromEnv);
}

export function getMacGpuTelemetryPath() {
  return process.env.AI_TOOLKIT_MAC_GPU_TELEMETRY_PATH || DEFAULT_TELEMETRY_PATH;
}

function parseTelemetryPayload(raw: string): MacGpuTelemetrySnapshot | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }

  if (!parsed || typeof parsed !== 'object') {
    return null;
  }

  const payload = parsed as Record<string, unknown>;
  const timestampMs = asFiniteNumber(payload.timestampMs ?? payload.timestamp);
  if (timestampMs === undefined) {
    return null;
  }

  const snapshot: MacGpuTelemetrySnapshot = {
    timestampMs,
    source: typeof payload.source === 'string' && payload.source ? payload.source : 'powermetrics-cache',
  };

  const temperatureC = asFiniteNumber(payload.temperatureC);
  const fanRpm = asFiniteNumber(payload.fanRpm);
  const powerDrawW = asFiniteNumber(payload.powerDrawW);
  const powerLimitW = asFiniteNumber(payload.powerLimitW);
  const clockMHz = asFiniteNumber(payload.clockMHz);
  const utilizationGpuPercent = asFiniteNumber(payload.utilizationGpuPercent);

  if (temperatureC !== undefined) snapshot.temperatureC = temperatureC;
  if (fanRpm !== undefined) snapshot.fanRpm = fanRpm;
  if (powerDrawW !== undefined) snapshot.powerDrawW = powerDrawW;
  if (powerLimitW !== undefined) snapshot.powerLimitW = powerLimitW;
  if (clockMHz !== undefined) snapshot.clockMHz = clockMHz;
  if (utilizationGpuPercent !== undefined) snapshot.utilizationGpuPercent = utilizationGpuPercent;

  return snapshot;
}

export async function readMacGpuTelemetrySnapshot(): Promise<MacGpuTelemetrySnapshot | null> {
  const telemetryPath = getMacGpuTelemetryPath();
  try {
    await access(telemetryPath, fsConstants.R_OK);
  } catch {
    return null;
  }

  try {
    const raw = await readFile(telemetryPath, 'utf8');
    const snapshot = parseTelemetryPayload(raw);
    if (!snapshot) {
      return null;
    }
    if (Date.now() - snapshot.timestampMs > getMaxAgeMs()) {
      return null;
    }
    return snapshot;
  } catch {
    return null;
  }
}
