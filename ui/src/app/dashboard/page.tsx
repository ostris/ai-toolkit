'use client';

import GpuMonitor from '@/components/GPUMonitor';

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold mb-8">Dashboard</h1>
      <GpuMonitor />
    </div>
  );
}
