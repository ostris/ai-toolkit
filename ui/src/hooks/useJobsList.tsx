'use client';

import { useEffect, useState } from 'react';
import { Job } from '@prisma/client';

export default function useJobsList() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJobs = () => {
    setStatus('loading');
    fetch('/api/jobs')
      .then(res => res.json())
      .then(data => {
        console.log('Jobs:', data);
        setJobs(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshJobs();
  }, []);

  return { jobs, setJobs, status, refreshJobs };
}
