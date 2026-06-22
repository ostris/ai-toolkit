'use client';

import { useEffect, useState } from 'react';
import { Queue } from '@prisma/client';
import { apiClient } from '@/utils/api';

export default function useQueueList() {
  const [queues, setQueues] = useState<Queue[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshQueues = () => {
    setStatus('loading');
    apiClient
      .get('/api/queue')
      .then(res => res.data)
      .then(data => {
        console.log('Queues:', data);
        if (data.error) {
          console.log('Error fetching queues:', data.error);
          setStatus('error');
        } else {
          setQueues(data.queues);
          setStatus('success');
        }
      })
      .catch(error => {
        console.error('Error fetching queues:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshQueues();
  }, []);

  return { queues, setQueues, status, refreshQueues };
}
