'use client';

import { CpuInfo } from '@/types';
import { useState } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

export default function useCPUInfo(reloadInterval: null | number = null) {
  const [cpuInfo, setCpuInfo] = useState<CpuInfo | null>(null);
  const [isCPUInfoLoaded, setIsLoaded] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const fetchCpuInfo = async () => {
    setStatus('loading');
    try {
      const data: CpuInfo = await apiClient.get('/api/cpu').then(res => res.data);
      setCpuInfo(data);
      setStatus('success');
    } catch (err) {
      console.error(`Failed to fetch CPU data: ${err instanceof Error ? err.message : String(err)}`);
      setStatus('error');
    } finally {
      setIsLoaded(true);
    }
  };

  usePollLoop(fetchCpuInfo, reloadInterval);

  return { cpuInfo, isCPUInfoLoaded, status, refreshCpuInfo: fetchCpuInfo };
}
