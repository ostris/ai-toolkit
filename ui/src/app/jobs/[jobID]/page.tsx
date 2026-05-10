'use client';

import { useState, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { MdDashboard, MdImage, MdShowChart, MdCode } from 'react-icons/md';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import SampleImages, { SampleImagesMenu } from '@/components/SampleImages';
import JobOverview from '@/components/JobOverview';
import { redirect } from 'next/navigation';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import JobLossGraph from '@/components/JobLossGraph';
import { Job } from '@prisma/client';

type PageKey = 'overview' | 'samples' | 'config' | 'loss_log';

interface Page {
  name: string;
  value: PageKey;
  icon: React.ComponentType<{ className?: string }>;
  component: React.ComponentType<{ job: Job }>;
  menuItem?: React.ComponentType<{ job?: Job | null }> | null;
  mainCss?: string;
  jobTypes?: string[];
}

const pages: Page[] = [
  {
    name: 'Overview',
    value: 'overview',
    icon: MdDashboard,
    component: JobOverview,
    mainCss: 'pt-14 md:pt-24 pb-16 md:pb-0',
  },
  {
    name: 'Samples',
    value: 'samples',
    icon: MdImage,
    component: SampleImages,
    menuItem: SampleImagesMenu,
    mainCss: 'pt-14 md:pt-24 pb-16 md:pb-0',
    jobTypes: ['train'],
  },
  {
    name: 'Loss',
    value: 'loss_log',
    icon: MdShowChart,
    component: JobLossGraph,
    mainCss: 'pt-14 md:pt-24 pb-16 md:pb-4',
    jobTypes: ['train'],
  },
  {
    name: 'Config',
    value: 'config',
    icon: MdCode,
    component: JobConfigViewer,
    mainCss: 'pt-14 md:pt-[80px] px-0 pb-16 md:pb-0',
  },
];

export default function JobPage({ params }: { params: { jobID: string } }) {
  const usableParams = use(params as any) as { jobID: string };
  const jobID = usableParams.jobID;
  const { job, status, refreshJob } = useJob(jobID, 5000);
  const [pageKey, setPageKey] = useState<PageKey>('overview');

  const page = pages.find(p => p.value === pageKey);
  const jobType = job?.job_type || 'unknown';

  let title = `Job: ${job?.name || 'Loading...'}`;
  if (jobType === 'caption') {
    title = `Captioning: ${job?.job_ref || 'Loading...'}`;
  }

  const visiblePages = pages.filter(p => !p.jobTypes || p.jobTypes.includes(jobType));

  return (
    <>
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => redirect('/jobs')}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="min-w-0 flex-1">
          <h1 className="text-sm md:text-lg truncate">{title}</h1>
        </div>
        {job && (
          <>
            <div className="hidden md:block">
              <JobActionBar job={job} onRefresh={refreshJob} hideView afterDelete={() => redirect('/jobs')} autoStartQueue />
            </div>
            <div className="md:hidden">
              <JobActionBar job={job} onRefresh={refreshJob} hideView afterDelete={() => redirect('/jobs')} autoStartQueue variant="menu" />
            </div>
          </>
        )}
      </TopBar>

      {/* Desktop: tab bar */}
      <div className="bg-gray-800 absolute top-12 left-0 w-full h-8 items-center px-2 text-sm overflow-x-auto hidden md:flex z-[5]">
        {visiblePages.map(p => (
          <Button
            key={p.value}
            onClick={() => setPageKey(p.value)}
            className={`px-4 py-1 h-8 flex items-center gap-1.5 ${p.value === pageKey ? 'bg-gray-300 dark:bg-gray-700 text-white' : ''}`}
          >
            <p.icon className="text-sm" />
            {p.name}
          </Button>
        ))}
        {page?.menuItem && (
          <>
            <div className="flex-grow"></div>
            <page.menuItem job={job} />
          </>
        )}
      </div>

      <MainContent className={page?.mainCss}>
        {status === 'loading' && job == null && <p>Loading...</p>}
        {status === 'error' && job == null && <p>Error fetching job</p>}
        {job && pages.map(p => {
          const Component = p.component;
          return p.value === pageKey ? <Component key={p.value} job={job} /> : null;
        })}
      </MainContent>

      {/* Mobile: bottom tab bar */}
      <div className="md:hidden fixed bottom-0 left-0 w-full bg-gray-900 border-t border-gray-700 flex items-center justify-around z-20 h-14">
        {visiblePages.map(p => (
          <button
            key={p.value}
            onClick={() => setPageKey(p.value)}
            className={`flex flex-col items-center justify-center flex-1 h-full text-xs gap-0.5 ${p.value === pageKey ? 'text-white' : 'text-gray-500'}`}
          >
            <p.icon className="text-lg" />
            {p.name}
          </button>
        ))}
      </div>
    </>
  );
}
