'use client';

import { useState, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import SampleImages, {SampleImagesMenu} from '@/components/SampleImages';
import JobOverview from '@/components/JobOverview';
import { redirect } from 'next/navigation';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import { Job } from '@prisma/client';

type PageKey = 'overview' | 'samples' | 'config';

interface Page {
  name: string;
  value: PageKey;
  component: React.ComponentType<{ job: Job }>;
  menuItem?: React.ComponentType<{ job?: Job | null }> | null;
  mainCss?: string;
}

const pages: Page[] = [
  {
    name: 'Overview',
    value: 'overview',
    component: JobOverview,
    mainCss: 'pt-24',
  },
  {
    name: 'Samples',
    value: 'samples',
    component: SampleImages,
    menuItem: SampleImagesMenu,
    mainCss: 'pt-24',
  },
  {
    name: 'Config File',
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
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => redirect('/jobs')}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Job: {job?.name}</h1>
        </div>
        <div className="flex-1"></div>
        {job && (
          <JobActionBar
            job={job}
            onRefresh={refreshJob}
            hideView
            afterDelete={() => {
              redirect('/jobs');
            }}
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
      <div className="bg-gray-800 absolute top-12 left-0 w-full h-8 flex items-center px-2 text-sm">
        {pages.map(page => (
          <Button
            key={page.value}
            onClick={() => setPageKey(page.value)}
            className={`px-4 py-1 h-8  ${page.value === pageKey ? 'bg-gray-300 dark:bg-gray-700' : ''}`}
          >
            {page.name}
          </Button>
        ))}
        {
          page?.menuItem && (
            <>
            <div className='flex-grow'>
            </div>
              <page.menuItem job={job} />
            </>
          )
        }
      </div>
    </>
  );
}
