'use client';

import { useEffect, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

export default function useJob(jobID: string, reloadInterval: null | number = null) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJob = () => {
    setStatus('loading');
    apiClient
      .get(`/api/jobs?id=${jobID}`)
      .then(res => res.data)
      .then(data => {
        console.log('Job:', data);
        setJob(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshJob();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refreshJob();
      }, reloadInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [jobID]);

  return { job, setJob, status, refreshJob };
}
