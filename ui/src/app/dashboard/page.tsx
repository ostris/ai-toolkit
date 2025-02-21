'use client';

import GpuMonitor from '@/components/GPUMonitor';
import { TopBar, MainContent } from '@/components/layout';

export default function Dashboard() {
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Dashboard</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <GpuMonitor />
      </MainContent>
    </>
  );
}
