'use client';

import { GPUApiResponse } from '@/types';
import { useEffect, useState } from 'react';

export default function useGPUInfo() {
  const [gpuList, setGpuList] = useState<number[]>([]);
  const [isGPUInfoLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    const fetchGpuInfo = async () => {
      try {
        const response = await fetch('/api/gpu');

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data: GPUApiResponse = await response.json();
        setGpuList(data.gpus.map(gpu => gpu.index).sort());
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
