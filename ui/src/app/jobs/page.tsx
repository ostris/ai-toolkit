'use client';

import { useState } from 'react';
import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import ImportJobModal from '@/components/ImportJobModal';
import Link from 'next/link';

export default function Dashboard() {
  const [importOpen, setImportOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Queue</h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex gap-2">
          <button
            onClick={() => setImportOpen(true)}
            className="text-white bg-slate-700 hover:bg-slate-600 px-3 py-1 rounded-md"
          >
            Import Job
          </button>
          <Link href="/jobs/new" className="text-white bg-slate-600 hover:bg-slate-500 px-3 py-1 rounded-md">
            New Training Job
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <JobsTable key={refreshKey} />
      </MainContent>
      <ImportJobModal
        isOpen={importOpen}
        onClose={() => setImportOpen(false)}
        onSuccess={() => setRefreshKey(k => k + 1)}
      />
    </>
  );
}
