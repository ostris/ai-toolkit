// components/GpuMonitor.tsx
import React, { useState, useEffect } from 'react';
import { GPUApiResponse } from '@/types';
import Loading from '@/components/Loading';

const GpuMonitor: React.FC = () => {
  const [gpuData, setGpuData] = useState<GPUApiResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    const fetchGpuInfo = async () => {
      try {
        const response = await fetch('/api/gpu');

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data: GPUApiResponse = await response.json();
        setGpuData(data);
        setLastUpdated(new Date());
        setError(null);
      } catch (err) {
        setError(`Failed to fetch GPU data: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setLoading(false);
      }
    };

    // Fetch immediately on component mount
    fetchGpuInfo();

    // Set up interval to fetch every 1 seconds
    const intervalId = setInterval(fetchGpuInfo, 1000);

    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  // Helper to format memory values
  const formatMemory = (mb: number): string => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb} MB`;
  };

  // Helper to determine background color based on utilization
  const getUtilizationColor = (percent: number): string => {
    if (percent < 30) return 'bg-green-100';
    if (percent < 70) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  // Helper to determine text color based on utilization
  const getUtilizationTextColor = (percent: number): string => {
    if (percent < 30) return 'text-green-800';
    if (percent < 70) return 'text-yellow-800';
    return 'text-red-800';
  };

  // Helper to determine temperature color
  const getTemperatureColor = (temp: number): string => {
    if (temp < 50) return 'text-green-600';
    if (temp < 80) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (loading) {
    return <Loading />;
  }

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">Error!</strong>
        <span className="block sm:inline"> {error}</span>
      </div>
    );
  }

  if (!gpuData) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <span className="block sm:inline">No GPU data available.</span>
      </div>
    );
  }

  if (!gpuData.hasNvidiaSmi) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">No NVIDIA GPUs detected!</strong>
        <span className="block sm:inline"> nvidia-smi is not available on this system.</span>
        {gpuData.error && <p className="mt-2 text-sm">{gpuData.error}</p>}
      </div>
    );
  }

  if (gpuData.gpus.length === 0) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <span className="block sm:inline">No GPUs found, but nvidia-smi is available.</span>
      </div>
    );
  }

  return (
    <div className="">
      <div className="flex justify-between items-center mb-2">
        <h1 className="text-md">GPU Monitor</h1>
        <div className="text-xs text-gray-500">Last updated: {lastUpdated?.toLocaleTimeString()}</div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {gpuData.gpus.map(gpu => (
          <div
            key={gpu.index}
            className="bg-gray-900 rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300 px-2 py-2"
          >
            <div className="bg-gray-800 text-white px-2 py-1 flex justify-between items-center">
              <h2 className="font-bold text-sm truncate">{gpu.name}</h2>
              <span className="text-xs bg-gray-700 rounded px-1 py-0.5">GPU #{gpu.index}</span>
            </div>

            <div className="p-2">
              <div className="mb-2 flex items-center">
                <p className="text-xs text-gray-500 mr-1">Temperature:</p>
                <p className={`text-sm font-bold ${getTemperatureColor(gpu.temperature)}`}>{gpu.temperature}°C</p>
              </div>

              <div className="">
                <p className="text-xs text-gray-600 mb-0.5">GPU Utilization</p>
                <div className="w-full bg-gray-500 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full ${gpu.utilization.gpu < 30 ? 'bg-green-500' : gpu.utilization.gpu < 70 ? 'bg-yellow-500' : 'bg-red-500'}`}
                    style={{ width: `${gpu.utilization.gpu}%` }}
                  ></div>
                </div>
                <p className="text-right text-xs mt-0.5">{gpu.utilization.gpu}%</p>
              </div>

              <div className="mb-2">
                <p className="text-xs text-gray-600 mb-0.5">Memory Utilization</p>
                <div className="w-full bg-gray-500 rounded-full h-1.5">
                  <div
                    className="h-1.5 rounded-full bg-blue-500"
                    style={{ width: `${(gpu.memory.used / gpu.memory.total) * 100}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs mt-0.5">
                  <span>
                    {formatMemory(gpu.memory.used)} / {formatMemory(gpu.memory.total)}
                  </span>
                  <span>{((gpu.memory.used / gpu.memory.total) * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 mb-2">
                <div>
                  <p className="text-xs text-gray-500 mb-0.5">Power</p>
                  <p className="text-sm font-medium">
                    {gpu.power.draw.toFixed(1)}W / {gpu.power.limit.toFixed(1)}W
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-0.5">Memory Clock</p>
                  <p className="text-sm font-medium">{gpu.clocks.memory} MHz</p>
                </div>
              </div>

              <div className="mt-1 pt-1 border-t border-gray-600 grid grid-cols-2 gap-2">
                <div>
                  <p className="text-xs text-gray-500 mb-0.5">Graphics Clock</p>
                  <p className="text-sm font-medium">{gpu.clocks.graphics} MHz</p>
                </div>
                <div className="">
                  <p className="text-xs text-gray-500 mb-0.5">Driver Version</p>
                  <p className="text-sm font-medium">{gpu.driverVersion}</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GpuMonitor;
