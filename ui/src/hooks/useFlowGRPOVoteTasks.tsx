'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface FlowGRPOCandidateView {
  id: string;
  order_index: number;
  prompt: string;
  negative_prompt: string;
  seed: number;
  guidance_scale: number;
  num_inference_steps: number;
  sampler: string;
  scheduler: string;
  image_path: string;
  image_url: string;
  status: string;
}

export interface FlowGRPOVoteTaskView {
  id: string;
  prompt: string;
  negative_prompt: string;
  width: number | null;
  height: number | null;
  seed: number | null;
  guidance_scale: number | null;
  num_inference_steps: number | null;
  sampler: string | null;
  scheduler: string | null;
  error?: string | null;
  status: string;
  created_at: string;
  candidates: FlowGRPOCandidateView[];
}

export default function useFlowGRPOVoteTasks(
  jobID: string,
  reloadInterval: number | null = 3000,
  taskStatus = 'requested,generating,open,voted',
) {
  const [tasks, setTasks] = useState<FlowGRPOVoteTaskView[]>([]);
  const [requestStatus, setRequestStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshTasks = () => {
    setRequestStatus('loading');
    apiClient
      .get(`/api/grpo/jobs/${jobID}/tasks`, { params: { status: taskStatus } })
      .then(res => res.data)
      .then(data => {
        setTasks(data.tasks || []);
        setRequestStatus('success');
      })
      .catch(error => {
        console.error('Error fetching Flow-GRPO vote tasks:', error);
        setRequestStatus('error');
      });
  };

  useEffect(() => {
    refreshTasks();
    if (reloadInterval) {
      const interval = setInterval(refreshTasks, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, taskStatus]);

  return { tasks, status: requestStatus, refreshTasks };
}
