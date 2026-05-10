'use client';

import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { Plus } from 'lucide-react';

export default function Dashboard() {
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-sm md:text-lg">Queue</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Link href="/jobs/new" className="text-white bg-slate-600 px-3 py-1 rounded-md flex items-center gap-1.5">
            <Plus className="w-4 h-4" />
            <span className="whitespace-nowrap">New Training Job</span>
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <JobsTable />
      </MainContent>
    </>
  );
}
