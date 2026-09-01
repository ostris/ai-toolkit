'use client';

import { useEffect, useState } from 'react';
import { CpuInfo, GPUApiResponse, MonitorHistoryPoint, MonitorInit, MonitorSample } from '@/types';
import { isAuthorizedState } from '@/utils/api';
import { historyPointFromSample, MONITOR_HISTORY_LENGTH } from '@/utils/monitorSample';

export interface MonitorStreamState {
  connected: boolean;
  lastUpdated: Date | null;
  cpu: CpuInfo | null;
  gpu: GPUApiResponse | null;
  history: MonitorHistoryPoint[];
}

/**
 * One shared SSE connection to /api/monitor for the whole app, no matter how
 * many components subscribe. Connects while at least one hook instance is
 * mounted, reconnects automatically, and keeps the 2-minute rolling history
 * (seeded by the server's backlog on connect) up to date from live samples.
 *
 * Native EventSource can't send the Authorization header the middleware
 * expects, so this reads the SSE stream through fetch.
 */
let state: MonitorStreamState = {
  connected: false,
  lastUpdated: null,
  cpu: null,
  gpu: null,
  history: [],
};
const listeners = new Set<(s: MonitorStreamState) => void>();
let refCount = 0;
let running = false;
let abortController: AbortController | null = null;

function emit(next: MonitorStreamState) {
  state = next;
  for (const listener of listeners) {
    listener(state);
  }
}

function handleEventBlock(block: string) {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }
  if (dataLines.length === 0) return;

  if (event === 'init') {
    const init: MonitorInit = JSON.parse(dataLines.join('\n'));
    emit({
      connected: true,
      lastUpdated: new Date(init.t),
      cpu: init.cpu,
      gpu: init.gpu,
      history: init.history,
    });
  } else if (event === 'sample') {
    const sample: MonitorSample = JSON.parse(dataLines.join('\n'));
    const history = [...state.history, historyPointFromSample(sample)];
    if (history.length > MONITOR_HISTORY_LENGTH) {
      history.splice(0, history.length - MONITOR_HISTORY_LENGTH);
    }
    emit({
      connected: true,
      lastUpdated: new Date(sample.t),
      cpu: sample.cpu,
      gpu: sample.gpu,
      history,
    });
  }
}

async function runLoop() {
  if (running) return;
  running = true;
  try {
    while (refCount > 0) {
      abortController = new AbortController();
      try {
        const headers: Record<string, string> = { Accept: 'text/event-stream' };
        const token = localStorage.getItem('AI_TOOLKIT_AUTH');
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }
        const res = await fetch('/api/monitor', {
          headers,
          cache: 'no-store',
          signal: abortController.signal,
        });
        if (res.status === 401) {
          // Mirror the axios interceptor's behavior
          localStorage.removeItem('AI_TOOLKIT_AUTH');
          isAuthorizedState.set(false);
          throw new Error('unauthorized');
        }
        if (!res.ok || !res.body) {
          throw new Error(`monitor stream returned HTTP ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let sep;
          while ((sep = buffer.indexOf('\n\n')) !== -1) {
            const block = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            handleEventBlock(block);
          }
        }
      } catch (err) {
        if (!abortController.signal.aborted) {
          console.error(`Monitor stream error: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      if (state.connected) {
        emit({ ...state, connected: false });
      }
      if (refCount <= 0) break;
      await new Promise(r => setTimeout(r, 2000));
    }
  } finally {
    running = false;
  }
}

export default function useMonitorStream(): MonitorStreamState {
  const [snapshot, setSnapshot] = useState(state);

  useEffect(() => {
    const listener = (s: MonitorStreamState) => setSnapshot(s);
    listeners.add(listener);
    refCount++;
    setSnapshot(state);
    runLoop();
    return () => {
      listeners.delete(listener);
      refCount--;
      if (refCount <= 0) {
        abortController?.abort();
      }
    };
  }, []);

  return snapshot;
}
