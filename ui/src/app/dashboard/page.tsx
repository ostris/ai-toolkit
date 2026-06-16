'use client';

import GpuMonitor from '@/components/GPUMonitor';
import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { useLanguage } from '@/components/LanguageProvider';

export default function Dashboard() {
  const { t } = useLanguage();

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">{t('dashboard.title')}</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <GpuMonitor />
        <div className="w-full mt-4">
          <div className="flex justify-between items-center mb-2">
            <h1 className="text-md">{t('dashboard.queues')}</h1>
            <div className="text-xs text-gray-500">
              <Link href="/jobs">{t('common.viewAll')}</Link>
            </div>
          </div>
          <JobsTable onlyActive />
        </div>
      </MainContent>
    </>
  );
}
