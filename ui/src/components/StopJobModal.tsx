'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import { Job } from '@prisma/client';
import { Loader2, CheckCircle2, AlertTriangle, X, Square } from 'lucide-react';

interface Props {
  open: boolean;
  job: Job | null;
  onClose: () => void;
  onComplete?: () => void;
}

type Phase = 'confirm' | 'stopping' | 'done' | 'error';

export default function StopJobModal({ open, job, onClose, onComplete }: Props) {
  const [phase, setPhase] = useState<Phase>('confirm');
  const [poll, setPoll] = useState<Job | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [log, setLog] = useState<{ ts: number; text: string }[]>([]);
  const startRef = useRef<number>(0);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastInfoRef = useRef<string>('');
  const lastStatusRef = useRef<string>('');

  // Reset state when reopened
  useEffect(() => {
    if (open) {
      setPhase('confirm');
      setPoll(null);
      setElapsedMs(0);
      setLog([]);
      lastInfoRef.current = '';
      lastStatusRef.current = '';
    }
  }, [open, job?.id]);

  // Polling lifecycle
  useEffect(() => {
    if (phase !== 'stopping' || !job) return;
    let cancelled = false;
    startRef.current = Date.now();
    elapsedTimerRef.current = setInterval(() => {
      setElapsedMs(Date.now() - startRef.current);
    }, 200);

    const tick = async () => {
      if (cancelled) return;
      try {
        const res = await apiClient.get(`/api/jobs?id=${job.id}`);
        const fresh: Job | null = res.data || null;
        if (cancelled) return;
        if (fresh) {
          setPoll(fresh);
          // Append to log on info/status changes.
          if (fresh.info && fresh.info !== lastInfoRef.current) {
            lastInfoRef.current = fresh.info;
            setLog(l => [...l, { ts: Date.now(), text: fresh.info }]);
          }
          if (fresh.status && fresh.status !== lastStatusRef.current) {
            lastStatusRef.current = fresh.status;
            setLog(l => [...l, { ts: Date.now(), text: `status → ${fresh.status}` }]);
          }
          if (fresh.status === 'stopped' || fresh.status === 'completed' || fresh.status === 'failed') {
            setPhase('done');
            onComplete?.();
            return;
          }
        }
      } catch (err) {
        // Server might be momentarily unavailable during a restart; keep polling.
        console.warn('Poll failed:', err);
      }
      if (!cancelled) {
        pollTimerRef.current = setTimeout(tick, 700);
      }
    };
    tick();

    return () => {
      cancelled = true;
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
      if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    };
  }, [phase, job?.id]);

  const triggerStop = () => {
    if (!job) return;
    setPhase('stopping');
    setLog(l => [...l, { ts: Date.now(), text: 'Sending stop request...' }]);
    // Fire-and-forget — the endpoint is synchronous on the server but the
    // status messages it writes to the DB along the way will surface via
    // the polling loop.
    apiClient
      .get(`/api/jobs/${job.id}/stop`)
      .then(() => {
        setLog(l => [...l, { ts: Date.now(), text: 'Stop request completed.' }]);
      })
      .catch(err => {
        console.error('Stop request failed:', err);
        setLog(l => [
          ...l,
          { ts: Date.now(), text: `Stop request failed: ${err?.message || err}` },
        ]);
        setPhase('error');
      });
  };

  const markStopped = async () => {
    if (!job) return;
    try {
      await apiClient.get(`/api/jobs/${job.id}/mark_stopped`);
      setLog(l => [...l, { ts: Date.now(), text: 'Job force-marked as stopped.' }]);
      setPhase('done');
      onComplete?.();
    } catch (err: any) {
      setLog(l => [
        ...l,
        { ts: Date.now(), text: `Mark stopped failed: ${err?.message || err}` },
      ]);
    }
  };

  if (!open || !job) return null;

  const elapsedSec = (elapsedMs / 1000).toFixed(1);
  const status = poll?.status || job.status;
  const info = poll?.info || job.info || '';
  const step = poll?.step ?? job.step;
  const stuck = phase === 'stopping' && elapsedMs > 20_000;

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg w-[95vw] max-w-xl flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center px-4 py-3 border-b border-gray-800">
          <Square className="w-5 h-5 text-red-400 mr-2" />
          <div className="text-lg flex-1">Stop Job</div>
          {phase !== 'stopping' && (
            <button className="text-gray-300 hover:text-white" onClick={onClose}>
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        <div className="p-4 space-y-3 text-sm">
          {phase === 'confirm' && (
            <>
              <div className="text-gray-200">
                Stop <span className="font-medium">{job.name}</span>?
              </div>
              <div className="text-xs text-gray-400">
                The training process will receive a termination signal. You can resume the job
                later — checkpoints saved so far are preserved.
              </div>
              <div className="text-xs text-amber-300 bg-amber-900/20 border border-amber-700/40 rounded p-2">
                On Windows the underlying <code>taskkill /T /F</code> can take a few seconds
                while the GPU process flushes. This dialog will show live status until the
                worker confirms the job stopped.
              </div>
            </>
          )}

          {phase === 'stopping' && (
            <>
              <div className="flex items-center gap-2 text-gray-200">
                <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
                <span>Stopping job — elapsed {elapsedSec}s</span>
              </div>
              <div className="rounded border border-gray-800 bg-gray-950 p-2 text-xs">
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <div className="text-gray-500 uppercase text-[10px]">Status</div>
                    <div className="text-gray-200">{status}</div>
                  </div>
                  <div>
                    <div className="text-gray-500 uppercase text-[10px]">Step</div>
                    <div className="text-gray-200 tabular-nums">{step ?? 0}</div>
                  </div>
                  <div>
                    <div className="text-gray-500 uppercase text-[10px]">PID</div>
                    <div className="text-gray-200 tabular-nums">{poll?.pid ?? job.pid ?? '—'}</div>
                  </div>
                </div>
                {info && (
                  <div className="mt-2">
                    <div className="text-gray-500 uppercase text-[10px]">Last message</div>
                    <div className="text-gray-200 break-words">{info}</div>
                  </div>
                )}
              </div>
              <LogView log={log} />
              {stuck && (
                <div className="text-xs text-amber-300 bg-amber-900/20 border border-amber-700/40 rounded p-2 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <div>
                    This has been running for over 20 seconds. The worker process may be hung.
                    You can force-mark it stopped below; the actual process may still be running
                    in the background.
                  </div>
                </div>
              )}
            </>
          )}

          {phase === 'done' && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-green-400">
                <CheckCircle2 className="w-5 h-5" />
                <span>Job stopped after {elapsedSec}s.</span>
              </div>
              <div className="text-xs text-gray-400">Final status: {status}.</div>
              {info && <div className="text-xs text-gray-300">{info}</div>}
              <LogView log={log} />
            </div>
          )}

          {phase === 'error' && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-red-400">
                <AlertTriangle className="w-5 h-5" />
                <span>Failed to stop the job.</span>
              </div>
              <LogView log={log} />
            </div>
          )}
        </div>

        <div className="px-4 py-3 border-t border-gray-800 flex items-center justify-end gap-2">
          {phase === 'confirm' && (
            <>
              <button
                onClick={onClose}
                className="px-3 py-1 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded"
              >
                Cancel
              </button>
              <button
                onClick={triggerStop}
                className="px-3 py-1 text-sm text-white bg-red-600 hover:bg-red-500 rounded"
              >
                Stop Job
              </button>
            </>
          )}
          {phase === 'stopping' && (
            <>
              {stuck && (
                <button
                  onClick={markStopped}
                  className="px-3 py-1 text-sm text-white bg-amber-700 hover:bg-amber-600 rounded mr-auto"
                >
                  Force-mark Stopped
                </button>
              )}
              <button
                disabled
                className="px-3 py-1 text-sm text-gray-400 bg-gray-800 rounded cursor-wait"
              >
                Working...
              </button>
            </>
          )}
          {(phase === 'done' || phase === 'error') && (
            <button
              onClick={onClose}
              className="px-3 py-1 text-sm text-white bg-gray-700 hover:bg-gray-600 rounded"
            >
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function LogView({ log }: { log: { ts: number; text: string }[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [log.length]);
  if (log.length === 0) return null;
  return (
    <div
      ref={ref}
      className="max-h-40 overflow-auto rounded border border-gray-800 bg-black/40 p-2 text-[11px] font-mono space-y-0.5"
    >
      {log.map((entry, i) => {
        const t = new Date(entry.ts);
        const hh = t.getHours().toString().padStart(2, '0');
        const mm = t.getMinutes().toString().padStart(2, '0');
        const ss = t.getSeconds().toString().padStart(2, '0');
        return (
          <div key={i} className="text-gray-300">
            <span className="text-gray-500">[{hh}:{mm}:{ss}]</span> {entry.text}
          </div>
        );
      })}
    </div>
  );
}
