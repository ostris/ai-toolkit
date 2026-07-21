'use client';

import { useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

type UseJobsListProps = {
  onlyActive?: boolean;
  reloadInterval?: number | null;
  job_type?: string | null;
};

export default function useJobsList({
  onlyActive = false,
  reloadInterval = null,
  job_type = null,
}: UseJobsListProps = {}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const isFetchingRef = useRef(false);

  const refreshJobs = () => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;
    setStatus('loading');
    const params: Record<string, string> = {};
    if (job_type) {
      params.job_type = job_type;
    }
    if (onlyActive) {
      params.only_active = 'true';
    }
    return apiClient
      .get('/api/jobs', { params })
      .then(res => res.data)
      .then(data => {
        console.log('Jobs:', data);
        if (data.error) {
          console.log('Error fetching jobs:', data.error);
          setStatus('error');
        } else {
          setJobs(data.jobs);
          setStatus('success');
        }
      })
      .catch(error => {
        console.error('Error fetching jobs:', error);
        setStatus('error');
      })
      .finally(() => {
        isFetchingRef.current = false;
      });
  };
  usePollLoop(refreshJobs, reloadInterval);

  return { jobs, setJobs, status, refreshJobs };
}
