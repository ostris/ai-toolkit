import { MonitorHistoryPoint, MonitorSample } from '@/types';

export const MONITOR_TICK_MS = 500;
export const MONITOR_HISTORY_LENGTH = 120_000 / MONITOR_TICK_MS; // 2 minutes of samples

/**
 * Reduce a full sample to the slim point kept in the rolling history:
 * load + memory only. Used by both the server (to build the backlog sent on
 * connect) and the client (to extend that backlog from live samples).
 */
export function historyPointFromSample(sample: MonitorSample): MonitorHistoryPoint {
  return {
    t: sample.t,
    cpu: {
      load: sample.cpu?.currentLoad ?? 0,
      memUsedMb: sample.cpu ? sample.cpu.totalMemory - sample.cpu.availableMemory : 0,
    },
    gpus: sample.gpu.gpus.map(gpu => ({
      load: gpu.utilization.gpu,
      memUsedMb: gpu.memory.used,
    })),
  };
}
