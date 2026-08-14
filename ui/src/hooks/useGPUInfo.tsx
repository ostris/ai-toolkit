'use client';

import { GpuInfo } from '@/types';
import { useMemo } from 'react';
import useMonitorStream from '@/hooks/useMonitorStream';

/**
 * GPU stats from the shared /api/monitor SSE stream. Data arrives live every
 * MONITOR_TICK_MS, so `reloadInterval` is accepted only for call-site
 * compatibility with the old polling implementation and is ignored.
 */
export default function useGPUInfo(gpuIds: null | number[] = null, reloadInterval: null | number = null) {
  void reloadInterval;
  const { gpu, connected } = useMonitorStream();

  // Key on contents, not identity — call sites often build gpuIds inline.
  const gpuIdsKey = gpuIds ? gpuIds.join(',') : null;
  const gpuList: GpuInfo[] = useMemo(() => {
    if (!gpu) return [];
    let gpus = [...gpu.gpus].sort((a, b) => a.index - b.index);
    if (gpuIdsKey !== null) {
      const ids = gpuIdsKey === '' ? [] : gpuIdsKey.split(',').map(Number);
      gpus = gpus.filter(g => ids.includes(g.index));
    }
    return gpus;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gpu, gpuIdsKey]);

  const isGPUInfoLoaded = gpu !== null;
  const status: 'idle' | 'loading' | 'success' | 'error' = gpu !== null ? 'success' : connected ? 'loading' : 'idle';

  return {
    gpuList,
    isGPUInfoLoaded,
    status,
    // The stream is always live; nothing to refresh manually.
    refreshGpuInfo: async () => {},
  };
}
