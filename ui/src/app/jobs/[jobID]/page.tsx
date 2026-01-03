'use client';

import { useState, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { LuHouse, LuImage, LuFileCode, LuChartLine } from 'react-icons/lu';
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
  shortName: string;
  icon: React.ComponentType<{ className?: string }>;
  value: PageKey;
  component: React.ComponentType<{ job: Job }>;
  menuItem?: React.ComponentType<{ job?: Job | null }> | null;
  mainCss?: string;
}

const pages: Page[] = [
  {
    name: 'Overview',
    shortName: 'Overview',
    icon: LuHouse,
    value: 'overview',
    component: JobOverview,
    mainCss: 'pt-24',
  },
  {
    name: 'Samples',
    shortName: 'Samples',
    icon: LuImage,
    value: 'samples',
    component: SampleImages,
    menuItem: SampleImagesMenu,
    mainCss: 'pt-24',
  },
  {
    name: 'Loss Graph',
    shortName: 'Loss',
    icon: LuChartLine,
    value: 'loss_log',
    component: JobLossGraph,
    mainCss: 'pt-24',
  },
  {
    name: 'Config File',
    shortName: 'Config',
    icon: LuFileCode,
    value: 'config',
    component: JobConfigViewer,
    mainCss: 'pt-[80px] px-0 pb-0',
  },
];

export default function JobPage({ params }: { params: { jobID: string } }) {
  const usableParams = use(params as any) as { jobID: string };
  const jobID = usableParams.jobID;
  const { job, status, refreshJob } = useJob(jobID, 5000);
  const [pageKey, setPageKey] = useState<PageKey>('overview');

  const page = pages.find(p => p.value === pageKey);

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div className="hidden md:flex items-center">
          <Button className="text-gray-500 dark:text-gray-300 px-3" onClick={() => redirect('/jobs')}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="flex-1 min-w-0 flex items-center">
          <h1 className="text-lg truncate">
            <span className="hidden md:inline">Job: </span>
            {job?.name}
          </h1>
        </div>
        {job && (
          <JobActionBar
            job={job}
            onRefresh={refreshJob}
            hideView
            afterDelete={() => {
              redirect('/jobs');
            }}
            autoStartQueue={true}
          />
        )}
      </TopBar>
      <MainContent className={pages.find(page => page.value === pageKey)?.mainCss}>
        {status === 'loading' && job == null && <p>Loading...</p>}
        {status === 'error' && job == null && <p>Error fetching job</p>}
        {job && (
          <>
            {pages.map(page => {
              const Component = page.component;
              return page.value === pageKey ? <Component key={page.value} job={job} /> : null;
            })}
          </>
        )}
      </MainContent>
      <div className="bg-gray-800 absolute top-12 left-0 w-full h-8 flex items-center px-1 md:px-2 text-sm">
        {pages.map(p => (
          <Button
            key={p.value}
            onClick={() => setPageKey(p.value)}
            className={`px-2 md:px-4 py-1 h-8 flex items-center gap-1 ${p.value === pageKey ? 'bg-gray-300 dark:bg-gray-700' : ''}`}
          >
            <p.icon className="w-4 h-4 md:hidden" />
            <span className="hidden md:inline">{p.name}</span>
            <span className="md:hidden text-xs">{p.shortName}</span>
          </Button>
        ))}
        {page?.menuItem && (
          <>
            <div className="flex-grow"></div>
            <page.menuItem job={job} />
          </>
        )}
      </div>
    </>
  );
}
