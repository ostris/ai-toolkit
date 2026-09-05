'use client';

import { useState } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

export default function useSampleImages(jobID: string, reloadInterval: null | number = null) {
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshSampleImages = () => {
    setStatus('loading');
    return apiClient
      .get(`/api/jobs/${jobID}/samples`)
      .then(res => res.data)
      .then(data => {
        console.log('Fetched sample images:', data);
        if (data.samples) {
          setSampleImages(data.samples);
        }
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };

  usePollLoop(refreshSampleImages, reloadInterval, [jobID]);

  return { sampleImages, setSampleImages, status, refreshSampleImages };
}
