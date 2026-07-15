'use client';

import { useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';

interface FileObject {
  path: string;
  size: number;
}

const clean = (text: string): string => {
  // remove \x1B[A\x1B[A
  text = text.replace(/\x1B\[A/g, '');
  return text;
};

export default function useJobLog(jobID: string, reloadInterval: null | number = null) {
  const [log, setLog] = useState<string>('');
  const didInitialLoadRef = useRef(false);
  // Byte offset into the log file that we've already consumed. Sent to the
  // server so it only returns newly appended content.
  const offsetRef = useRef<number | null>(null);
  // Guards against overlapping requests: if a poll fires while a request is
  // still in flight, both would read the same offset and append the same
  // bytes twice.
  const inFlightRef = useRef(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = () => {
    if (inFlightRef.current) {
      return;
    }
    inFlightRef.current = true;
    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);
    const offset = offsetRef.current;
    apiClient
      .get(`/api/jobs/${jobID}/log`, offset !== null ? { params: { offset } } : undefined)
      .then(res => res.data)
      .then(data => {
        offsetRef.current = data.offset ?? null;
        const cleanLog = clean(data.log ?? '');
        if (data.reset) {
          // Log was reset/truncated (or initial load) — replace everything.
          setLog(cleanLog);
        } else if (cleanLog) {
          // Incremental — append only the new content.
          setLog(prev => prev + cleanLog);
        }
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
        console.error('Error fetching log:', error);
        setStatus('error');
      })
      .finally(() => {
        inFlightRef.current = false;
      });
  };

  useEffect(() => {
    // New job — start fresh.
    offsetRef.current = null;
    didInitialLoadRef.current = false;
    inFlightRef.current = false;
    setLog('');
    refresh();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refresh();
      }, reloadInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [jobID]);

  return { log, setLog, status, refresh };
}
