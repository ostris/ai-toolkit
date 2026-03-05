import React from 'react';
import { GpuInfo } from '@/types';
import { Thermometer, Zap, Clock, HardDrive, Fan, Cpu } from 'lucide-react';

interface GPUWidgetProps {
  gpu: GpuInfo;
}

const UNKNOWN_METRIC = -1;

export default function GPUWidget({ gpu }: GPUWidgetProps) {
  const isKnownMetric = (value: number | undefined): value is number => {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0;
  };

  const formatMemory = (mb: number): string => {
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
  };

  const getUtilizationColor = (value: number): string => {
    if (!isKnownMetric(value)) return 'bg-gray-600';
    return value < 30 ? 'bg-emerald-500' : value < 70 ? 'bg-amber-500' : 'bg-rose-500';
  };

  const getTemperatureColor = (temp: number): string => {
    if (!isKnownMetric(temp)) return 'text-gray-500';
    return temp < 50 ? 'text-emerald-500' : temp < 80 ? 'text-amber-500' : 'text-rose-500';
  };

  const memoryPercent =
    gpu.memory.total > 0 && gpu.memory.used >= 0 ? (gpu.memory.used / gpu.memory.total) * 100 : UNKNOWN_METRIC;

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all duration-300 border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <h2 className="font-semibold text-gray-100">{gpu.name}</h2>
          <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300"># {gpu.index}</span>
        </div>
        {gpu.telemetrySource ? (
          <span className="text-[10px] uppercase tracking-wide text-gray-400">{gpu.telemetrySource}</span>
        ) : null}
      </div>

      <div className="p-4 space-y-4">
        {/* Temperature, Fan, and Utilization Section */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Thermometer className={`w-4 h-4 ${getTemperatureColor(gpu.temperature)}`} />
              <div>
                <p className="text-xs text-gray-400">Temperature</p>
                <p className={`text-sm font-medium ${getTemperatureColor(gpu.temperature)}`}>
                  {isKnownMetric(gpu.temperature) ? `${gpu.temperature}°C` : 'N/A'}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Fan className="w-4 h-4 text-blue-400" />
              <div>
                <p className="text-xs text-gray-400">Fan Speed</p>
                <p className="text-sm font-medium text-blue-400">
                  {isKnownMetric(gpu.fan.rpm)
                    ? `${Math.round(gpu.fan.rpm)} RPM`
                    : isKnownMetric(gpu.fan.speed)
                      ? `${gpu.fan.speed}%`
                      : 'N/A'}
                </p>
              </div>
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2 mb-1">
              <Cpu className="w-4 h-4 text-gray-400" />
              <p className="text-xs text-gray-400">GPU Load</p>
              <span className="text-xs text-gray-300 ml-auto">
                {isKnownMetric(gpu.utilization.gpu) ? `${gpu.utilization.gpu}%` : 'N/A'}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1">
              <div
                className={`h-1 rounded-full transition-all ${getUtilizationColor(gpu.utilization.gpu)}`}
                style={{ width: `${isKnownMetric(gpu.utilization.gpu) ? gpu.utilization.gpu : 0}%` }}
              />
            </div>
            <div className="flex items-center space-x-2 mb-1 mt-3">
              <HardDrive className="w-4 h-4 text-blue-400" />
              <p className="text-xs text-gray-400">Memory</p>
              <span className="text-xs text-gray-300 ml-auto">
                {isKnownMetric(memoryPercent) ? `${memoryPercent.toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1">
              <div
                className="h-1 rounded-full bg-blue-500 transition-all"
                style={{ width: `${isKnownMetric(memoryPercent) ? memoryPercent : 0}%` }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-0.5">
              {formatMemory(gpu.memory.used)} / {formatMemory(gpu.memory.total)}
            </p>
          </div>
        </div>

        {/* Power and Clocks Section */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-gray-800">
          <div className="flex items-start space-x-2">
            <Clock className="w-4 h-4 text-purple-400" />
            <div>
              <p className="text-xs text-gray-400">Clock Speed</p>
              <p className="text-sm text-gray-200">
                {isKnownMetric(gpu.clocks.graphics) ? `${gpu.clocks.graphics} MHz` : 'N/A'}
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-2">
            <Zap className="w-4 h-4 text-amber-400" />
            <div>
              <p className="text-xs text-gray-400">Power Draw</p>
              <p className="text-sm text-gray-200">
                {isKnownMetric(gpu.power.draw) ? `${gpu.power.draw.toFixed(1)}W` : 'N/A'}
                {isKnownMetric(gpu.power.limit) ? (
                  <span className="text-gray-400 text-xs"> / {gpu.power.limit.toFixed(1)}W</span>
                ) : null}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
