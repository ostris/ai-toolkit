'use client';

import { useMemo, useState, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import { startJob, stopJob } from '@/utils/jobs';
import SampleImages from '@/components/SampleImages';
import JobOverview from '@/components/JobOverview';
import { JobConfig } from '@/types';

type PageKey = 'overview' | 'samples';

interface Page {
  name: string;
  value: PageKey;
}

const pages: Page[] = [
  { name: 'Overview', value: 'overview' },
  { name: 'Samples', value: 'samples' },
];

export default function JobPage({ params }: { params: { jobID: string } }) {
  const usableParams = use(params as any) as { jobID: string };
  const jobID = usableParams.jobID;
  const { job, status, refreshJob } = useJob(jobID, 5000);
  const [pageKey, setPageKey] = useState<PageKey>('overview');

  const numSamples = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      const sampleConfig = jobConfig.config.process[0].sample;
      return sampleConfig.prompts.length;
    }
    return 10;
  }, [job]);

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Job: {job?.name}</h1>
        </div>
        <div className="flex-1"></div>
        {job?.status === 'running' && (
          <Button
            onClick={async () => {
              await stopJob(jobID);
              refreshJob();
            }}
            className="bg-red-500 text-white px-4 py-1 rounded-sm"
          >
            Stop
          </Button>
        )}
        {(job?.status === 'stopped' || job?.status === 'error') && (
          <Button
            onClick={async () => {
              await startJob(jobID);
              refreshJob();
            }}
            className="bg-green-800 text-white px-4 py-1 rounded-sm"
          >
            Start
          </Button>
        )}
      </TopBar>
      <MainContent className="pt-24">
        {status === 'loading' && job == null && <p>Loading...</p>}
        {status === 'error' && job == null && <p>Error fetching job</p>}
        {job && (
          <>
            {pageKey === 'overview' && <JobOverview job={job} />}
            {pageKey === 'samples' && <SampleImages job={job} />}
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
      </div>
    </>
  );
}
