'use client';

import { useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';
import { TerminalEmulator } from '@/utils/terminalEmulator';

interface FileObject {
  path: string;
  size: number;
}

export default function useJobLog(jobID: string, reloadInterval: null | number = null) {
  const [log, setLog] = useState<string>('');
  // Emulates a terminal over the raw log stream so carriage returns, cursor
  // movement, and erase sequences collapse lines like a real terminal.
  const terminalRef = useRef<TerminalEmulator | null>(null);
  if (terminalRef.current === null) {
    terminalRef.current = new TerminalEmulator();
  }
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
        const terminal = terminalRef.current!;
        if (data.reset) {
          // Log was reset/truncated (or initial load) — replace everything.
          terminal.reset();
        }
        terminal.write(data.log ?? '');
        setLog(terminal.toString());
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
    terminalRef.current?.reset();
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

  return { log, status, refresh };
}
