'use client';

import { useEffect, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

export default function useJobsList(onlyActive = false, reloadInterval: null | number = null) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJobs = () => {
    setStatus('loading');
    apiClient
      .get('/api/jobs')
      .then(res => res.data)
      .then(data => {
        console.log('Jobs:', data);
        if (data.error) {
          console.log('Error fetching jobs:', data.error);
          setStatus('error');
        } else {
          const list = Array.isArray(data.jobs) ? data.jobs : [];
          setJobs(onlyActive ? list.filter((job: Job) => ['running', 'queued', 'stopping'].includes(job.status)) : list);
          setStatus('success');
        }
      })
      .catch(error => {
        console.error('Error fetching jobs:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshJobs();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refreshJobs();
      }, reloadInterval);
      return () => clearInterval(interval);
    }
  }, []);

  return { jobs, setJobs, status, refreshJobs };
}
