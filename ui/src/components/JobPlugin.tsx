'use client';

import { useEffect, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

interface JobPluginProps {
  job: Job;
}

export default function JobPlugin({ job }: JobPluginProps) {
  const [html, setHtml] = useState<string | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');

  // An iframe src request cannot send the Authorization header the API requires,
  // so fetch the html through apiClient and load it via srcDoc instead.
  useEffect(() => {
    apiClient
      .get(`/api/jobs/${job.id}/plugin`, { responseType: 'text' })
      .then(res => {
        setHtml(res.data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching plugin:', error);
        setStatus('error');
      });
  }, [job.id]);

  if (status === 'loading') {
    return <p className="p-4">Loading plugin...</p>;
  }
  if (status === 'error' || html == null) {
    return <p className="p-4">Error loading plugin</p>;
  }

  // Sandboxed iframe keeps the plugin's scripts and styles isolated from the app.
  // allow-scripts lets its <script> tags run, but it cannot touch the parent page,
  // cookies, or storage. The plugin sizes itself to 100% of its container, so the
  // iframe must have an explicit size.
  // 80px = top bar (48px) + tab bar (32px)
  return <iframe srcDoc={html} sandbox="allow-scripts" className="block w-full h-[calc(100dvh-80px)] border-0" />;
}
