'use client';

import { useEffect, useState, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import { startJob, stopJob } from '@/utils/jobs';

export default function JobPage({ params }: { params: { jobID: string } }) {
  const usableParams = use(params as any) as { jobID: string };
  const jobID = usableParams.jobID;
  const { job, status, refreshJobs } = useJob(jobID, 5000);

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
                refreshJobs();
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
                refreshJobs();
            }}
            className="bg-green-800 text-white px-4 py-1 rounded-sm"
          >
            Start
          </Button>
        )}
      </TopBar>
      <MainContent>
        {status === 'loading' && job == null && <p>Loading...</p>}
        {status === 'error' && job == null && <p>Error fetching job</p>}
        {job && (
          <>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="">
                <h2 className="text-lg font-semibold">Job Details</h2>
                  <p className="text-gray-400">ID: {job.id}</p>
                  <p className="text-gray-400">Name: {job.name}</p>
                  <p className="text-gray-400">GPUs: {job.gpu_ids}</p>
                  <p className="text-gray-400">Status: {job.status}</p>
                  <p className="text-gray-400">Info: {job.info}</p>
                  <p className="text-gray-400">Step: {job.step}</p>
              </div>
            </div>
          </>
        )}
      </MainContent>
    </>
  );
}
