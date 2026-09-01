'use client';

import { useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

export default function useJobByRef(jobRef: string | null, reloadInterval: null | number = null) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJob = () => {
    setStatus('loading');
    return apiClient
      .get(`/api/jobs?job_ref=${jobRef}`)
      .then(res => res.data)
      .then(data => {
        console.log('Job:', data);
        setJob(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching job:', error);
        setStatus('error');
      });
  };

  usePollLoop(refreshJob, reloadInterval, [jobRef]);

  return { job, setJob, status, refreshJob };
}
