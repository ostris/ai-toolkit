'use client';

import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';

export default function Dashboard() {
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Queue</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Link href="/jobs/new" className="text-white bg-slate-600 px-3 py-1 rounded-md">
            New Training Job
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <JobsTable />
      </MainContent>
    </>
  );
}
