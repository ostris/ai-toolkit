'use client';

import { GPUApiResponse, GpuInfo } from '@/types';
import { useEffect, useState } from 'react';

export default function useGPUInfo(gpuIds: null | number[] = null) {
  const [gpuList, setGpuList] = useState<GpuInfo[]>([]);
  const [isGPUInfoLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    const fetchGpuInfo = async () => {
      try {
        const response = await fetch('/api/gpu');

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data: GPUApiResponse = await response.json();
        let gpus = data.gpus.sort((a, b) => a.index - b.index);
        if (gpuIds) {
          gpus = gpus.filter(gpu => gpuIds.includes(gpu.index));
        }

        setGpuList(gpus);
      } catch (err) {
        console.log(`Failed to fetch GPU data: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setIsLoaded(true);
      }
    };

    // Fetch immediately on component mount
    fetchGpuInfo();
  }, []);

  return { gpuList, setGpuList, isGPUInfoLoaded };
}
