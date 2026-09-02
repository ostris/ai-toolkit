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
 * One SSE connection to /api/monitor per browser, not per tab. Browsers cap
 * HTTP/1.1 at 6 connections per host, so one held stream per tab meant the
 * 6th tab could not load anything at all. Tabs elect a leader over a
 * BroadcastChannel: the leader holds the stream and forwards every event,
 * followers apply the forwarded events. When the leader goes away (tab
 * closed, hook unmounted, tab frozen) a follower takes over.
 *
 * Within a tab the state is shared by every hook instance. Connects while at
 * least one hook instance is mounted, reconnects automatically, and keeps the
 * 2-minute rolling history (seeded by the server's backlog on connect) up to
 * date from live samples.
 *
 * Native EventSource can't send the Authorization header the middleware
 * expects, so the leader reads the SSE stream through fetch.
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

const CHANNEL_NAME = 'ai-toolkit-monitor-stream';
// Leader is presumed gone after this much silence; forwarded samples arrive
// every MONITOR_TICK_MS so a healthy leader never comes close.
const LEADER_SILENCE_MS = 3000;
// How long a hello goes unanswered before the tab takes the stream itself.
const CLAIM_WAIT_MS = 700;
const tabId = Math.random().toString(36).slice(2) + Date.now().toString(36);

type Role = 'idle' | 'follower' | 'leader';
type ChannelMessage =
  | { type: 'hello'; id: string }
  | { type: 'lead'; id: string }
  | { type: 'snapshot'; id: string; state: MonitorStreamState }
  | { type: 'event'; id: string; event: string; data: string }
  | { type: 'resign'; id: string };

let role: Role = 'idle';
let channel: BroadcastChannel | null = null;
let lastLeaderMsgAt = 0;
let claimTimer: ReturnType<typeof setTimeout> | null = null;
let watchdogTimer: ReturnType<typeof setInterval> | null = null;

function emit(next: MonitorStreamState) {
  state = next;
  for (const listener of listeners) {
    listener(state);
  }
}

function post(msg: ChannelMessage) {
  try {
    channel?.postMessage(msg);
  } catch {
    // channel closed
  }
}

function applyEvent(event: string, data: string) {
  if (event === 'init') {
    const init: MonitorInit = JSON.parse(data);
    emit({
      connected: true,
      lastUpdated: new Date(init.t),
      cpu: init.cpu,
      gpu: init.gpu,
      history: init.history,
    });
  } else if (event === 'sample') {
    const sample: MonitorSample = JSON.parse(data);
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
  const data = dataLines.join('\n');
  applyEvent(event, data);
  if (role === 'leader') {
    post({ type: 'event', id: tabId, event, data });
  }
}

async function runLoop() {
  if (running) return;
  running = true;
  try {
    while (refCount > 0 && role === 'leader') {
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
      if (refCount <= 0 || role !== 'leader') break;
      await new Promise(r => setTimeout(r, 2000));
    }
  } finally {
    running = false;
  }
}

function cancelClaim() {
  if (claimTimer) {
    clearTimeout(claimTimer);
    claimTimer = null;
  }
}

function becomeLeader() {
  cancelClaim();
  if (role === 'leader') return;
  role = 'leader';
  post({ type: 'lead', id: tabId });
  runLoop();
}

function becomeFollower() {
  cancelClaim();
  if (role === 'leader') {
    abortController?.abort();
  }
  role = 'follower';
  lastLeaderMsgAt = Date.now();
}

// Ask for a leader; take over if nobody answers. Jitter spreads out tabs
// that lost the same leader at the same moment.
function startClaim() {
  if (claimTimer || role !== 'follower') return;
  post({ type: 'hello', id: tabId });
  claimTimer = setTimeout(becomeLeader, CLAIM_WAIT_MS + Math.random() * 300);
}

function onMessage(ev: MessageEvent<ChannelMessage>) {
  const msg = ev.data;
  if (!msg || msg.id === tabId) return;
  if (msg.type === 'hello') {
    if (role === 'leader') {
      post({ type: 'lead', id: tabId });
      post({ type: 'snapshot', id: tabId, state });
    }
    return;
  }
  if (msg.type === 'resign') {
    if (role === 'follower') startClaim();
    return;
  }
  // lead / snapshot / event: another tab is leading
  if (role === 'leader') {
    // Two leaders (simultaneous claims, or a thawed tab). Lowest id wins.
    if (msg.id < tabId) {
      becomeFollower();
    } else {
      post({ type: 'lead', id: tabId });
      return;
    }
  }
  lastLeaderMsgAt = Date.now();
  cancelClaim();
  if (msg.type === 'snapshot') {
    emit(msg.state);
  } else if (msg.type === 'event') {
    applyEvent(msg.event, msg.data);
  }
}

function onPageHide() {
  if (role === 'leader') {
    post({ type: 'resign', id: tabId });
  }
}

function start() {
  if (typeof BroadcastChannel === 'undefined') {
    role = 'leader';
    runLoop();
    return;
  }
  channel = new BroadcastChannel(CHANNEL_NAME);
  channel.onmessage = onMessage;
  window.addEventListener('pagehide', onPageHide);
  role = 'follower';
  lastLeaderMsgAt = Date.now();
  startClaim();
  watchdogTimer = setInterval(() => {
    if (role === 'follower' && Date.now() - lastLeaderMsgAt > LEADER_SILENCE_MS) {
      if (state.connected) {
        emit({ ...state, connected: false });
      }
      startClaim();
    }
  }, 1000);
}

function stop() {
  cancelClaim();
  if (watchdogTimer) {
    clearInterval(watchdogTimer);
    watchdogTimer = null;
  }
  if (role === 'leader') {
    post({ type: 'resign', id: tabId });
    abortController?.abort();
  }
  role = 'idle';
  if (channel) {
    window.removeEventListener('pagehide', onPageHide);
    channel.close();
    channel = null;
  }
}

export default function useMonitorStream(): MonitorStreamState {
  const [snapshot, setSnapshot] = useState(state);

  useEffect(() => {
    const listener = (s: MonitorStreamState) => setSnapshot(s);
    listeners.add(listener);
    refCount++;
    setSnapshot(state);
    if (refCount === 1) {
      start();
    }
    return () => {
      listeners.delete(listener);
      refCount--;
      if (refCount <= 0) {
        stop();
      }
    };
  }, []);

  return snapshot;
}
