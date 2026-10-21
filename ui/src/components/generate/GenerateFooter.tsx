'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronUp, ChevronDown, Loader2 } from 'lucide-react';
import useJobLog from '@/hooks/useJobLog';

interface Props {
  /** engine job id: the log panel follows its log.txt */
  jobId: string | null;
  /** one-line status shown in the collapsed bar */
  status: string;
  busy?: boolean;
  progress?: { step: number; total: number | null } | null;
}

export default function GenerateFooter({ jobId, status, busy, progress }: Props) {
  const [open, setOpen] = useState(false);
  const { log, status: logStatus } = useJobLog(jobId ?? '', jobId && open ? 1500 : null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const [follow, setFollow] = useState(true);

  const lines = useMemo(() => {
    const splits = log.split('\n');
    return splits.length > 1500 ? splits.slice(splits.length - 1500) : splits;
  }, [log]);

  const handleScroll = () => {
    const el = logRef.current;
    if (!el) return;
    setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 10);
  };

  useEffect(() => {
    if (open && follow && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log, open, follow]);

  const pct = progress?.total ? Math.min(100, (progress.step / progress.total) * 100) : null;

  return (
    <div className="absolute left-0 right-0 bottom-0 z-30 border-t border-gray-800 bg-gray-900 shadow-[0_-4px_16px_rgba(0,0,0,0.4)]">
      {/* progress hairline */}
      {pct !== null && (
        <div className="h-0.5 bg-gray-800">
          <div className="h-0.5 bg-blue-500 transition-all" style={{ width: `${pct}%` }} />
        </div>
      )}
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 h-9 text-xs text-gray-300 hover:bg-gray-800/60"
        title={open ? 'Hide log' : 'Show log'}
      >
        {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400 shrink-0" /> : <span className="w-2 h-2 rounded-full bg-green-500 shrink-0" />}
        <span className="truncate flex-1 text-left font-mono">{status}</span>
        {progress?.total ? (
          <span className="text-gray-500 shrink-0">
            {progress.step}/{progress.total}
          </span>
        ) : null}
        {open ? <ChevronDown className="w-4 h-4 shrink-0" /> : <ChevronUp className="w-4 h-4 shrink-0" />}
      </button>
      {open && (
        <div className="h-[40vh] bg-gray-950 border-t border-gray-800 relative">
          <div ref={logRef} onScroll={handleScroll} className="absolute inset-0 overflow-y-auto p-3 text-[11px] leading-4 text-gray-300 font-mono">
            {!jobId && <div className="text-gray-500">No engine job.</div>}
            {jobId && logStatus === 'loading' && !log && <div className="text-gray-500">Loading log…</div>}
            {jobId && logStatus === 'error' && <div className="text-rose-400">Error loading log</div>}
            {lines.map((line, i) => (
              <pre key={i} className="whitespace-pre-wrap break-all">
                {line}
              </pre>
            ))}
          </div>
          {!follow && (
            <button
              type="button"
              onClick={() => {
                setFollow(true);
                if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
              }}
              className="absolute bottom-3 right-4 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-[11px] text-gray-300 hover:bg-gray-700"
            >
              Follow
            </button>
          )}
        </div>
      )}
    </div>
  );
}
