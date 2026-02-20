import React from 'react';
import { CpuInfo } from '@/types';
import { Thermometer, Zap, Clock, HardDrive, Fan, Cpu } from 'lucide-react';

interface CPUWidgetProps {
  cpu: CpuInfo | null;
}

export default function CPUWidget({ cpu }: CPUWidgetProps) {
  const formatMemory = (mb: number | string): string => {
    if (typeof mb === 'string' || isNaN(mb) || mb < 0) return 'N/A';
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
  };

  const getUtilizationColor = (value: number): string => {
    return value < 30 ? 'bg-emerald-500' : value < 70 ? 'bg-amber-500' : 'bg-rose-500';
  };

  const getTemperatureColor = (temp: number): string => {
    return temp < 50 ? 'text-emerald-500' : temp < 80 ? 'text-amber-500' : 'text-rose-500';
  };

  if (!cpu) {
    return (
      <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all duration-300 border border-gray-800">
        <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <h2 className="font-semibold text-gray-100">CPU Info</h2>
          </div>
        </div>
        <div className="p-4">
          <p className="text-sm text-gray-400">No CPU data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all duration-300 border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <h2 className="font-semibold text-gray-100">{cpu.name}</h2>
          {/* <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300">#{1}</span> */}
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Temperature, Fan, and Utilization Section */}
        <div className="grid grid-cols-2 gap-4">
          <div className="">
            <div className="flex items-center space-x-2 mb-1 mt-1">
              <Cpu className="w-4 h-4 text-gray-400" />
              <p className="text-xs text-gray-400">CPU Load</p>
              <span className="text-xs text-gray-300 ml-auto">
                {typeof cpu.currentLoad === 'number' ? `${cpu.currentLoad.toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1">
              <div
                className={`h-1 rounded-full transition-all ${getUtilizationColor(cpu.currentLoad)}`}
                style={{ width: typeof cpu.currentLoad === 'number' ? `${cpu.currentLoad}%` : '0%' }}
              />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2 mb-1 mt-1">
              <HardDrive className="w-4 h-4 text-blue-400" />
              <p className="text-xs text-gray-400">Memory</p>
              <span className="text-xs text-gray-300 ml-auto">
                {typeof cpu.totalMemory === 'number' && typeof cpu.availableMemory === 'number' && cpu.totalMemory > 0
                  ? `${(((cpu.totalMemory - cpu.availableMemory) / cpu.totalMemory) * 100).toFixed(1)}%`
                  : 'N/A'}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1">
              <div
                className="h-1 rounded-full bg-blue-500 transition-all"
                style={{
                  width: typeof cpu.totalMemory === 'number' && typeof cpu.availableMemory === 'number' && cpu.totalMemory > 0
                    ? `${((cpu.totalMemory - cpu.availableMemory) / cpu.totalMemory) * 100}%`
                    : '0%'
                }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-0.5">
              {formatMemory(cpu.totalMemory - cpu.availableMemory)} / {formatMemory(cpu.totalMemory)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
