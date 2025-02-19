'use client';

import GpuMonitor from '@/components/GPUMonitor';

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>
      <GpuMonitor />
    </div>
  );
}
