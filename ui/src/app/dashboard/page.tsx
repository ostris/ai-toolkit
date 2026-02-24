'use client';

import GpuMonitor from '@/components/GPUMonitor';
import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';

export default function Dashboard() {
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Dashboard</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Link
            href="/merge"
            className="inline-flex items-center px-4 py-2 rounded-md text-gray-200 bg-green-800 hover:bg-green-700"
          >
            🧬 Merge LoRAs
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <GpuMonitor />
        <div className="w-full mt-4">
          <div className="flex justify-between items-center mb-2">
            <h1 className="text-md">Queues</h1>
            <div className="text-xs text-gray-500 flex items-center gap-4">
              <Link href="/merge" className="text-green-400 hover:text-green-300">
                🧬 Merge LoRAs
              </Link>
              <Link href="/jobs">View All</Link>
            </div>
          </div>
          <JobsTable onlyActive />
        </div>
      </MainContent>
    </>
  );
}
