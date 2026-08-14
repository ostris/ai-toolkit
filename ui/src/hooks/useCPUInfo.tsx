'use client';

import useMonitorStream from '@/hooks/useMonitorStream';

/**
 * CPU stats from the shared /api/monitor SSE stream. Data arrives live every
 * MONITOR_TICK_MS, so `reloadInterval` is accepted only for call-site
 * compatibility with the old polling implementation and is ignored.
 */
export default function useCPUInfo(reloadInterval: null | number = null) {
  void reloadInterval;
  const { cpu, connected } = useMonitorStream();

  const isCPUInfoLoaded = cpu !== null;
  const status: 'idle' | 'loading' | 'success' | 'error' = cpu !== null ? 'success' : connected ? 'loading' : 'idle';

  return {
    cpuInfo: cpu,
    isCPUInfoLoaded,
    status,
    // The stream is always live; nothing to refresh manually.
    refreshCpuInfo: async () => {},
  };
}
