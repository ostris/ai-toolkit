'use client';

import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';

export default function Dashboard() {
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg hidden md:block">Training Queue</h1>
          <h1 className="text-lg md:hidden">Queue</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Link href="/jobs/new" className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md text-sm whitespace-nowrap h-[30px]">
            New Job
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <JobsTable />
      </MainContent>
    </>
  );
}
