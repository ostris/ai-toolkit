'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Cpu } from 'lucide-react';
import useJobByRef from '@/hooks/useJobByRef';
import useJobLog from '@/hooks/useJobLog';
import useGPUInfo from '@/hooks/useGPUInfo';
import GPUWidget from '@/components/GPUWidget';
import JobActionBar from '@/components/JobActionBar';
import { getTotalSteps } from '@/utils/jobs';

interface CaptionMonitorProps {
  datasetPath: string;
  // Reports the current docked height (px) so the page can pad content to match.
  onHeightChange?: (height: number) => void;
}

// A small floating module that pops up from the bottom while a dataset is being
// auto-captioned. It is a simplified version of the job page: action bar, GPU
// info, and the live log.
export default function CaptionMonitor({ datasetPath, onHeightChange }: CaptionMonitorProps) {
  const { job, status, refreshJob } = useJobByRef(datasetPath, 3000);
  const [collapsed, setCollapsed] = useState(false);

  const isActive = !!(job && (job.status === 'running' || job.status === 'queued'));

  // Reset the collapsed state whenever a new captioning run starts.
  useEffect(() => {
    if (isActive) setCollapsed(false);
  }, [job?.id, isActive]);

  const gpuIds = useMemo(() => {
    if (!job) return [];
    if (job.gpu_ids === 'mps') return [0];
    return job.gpu_ids.split(',').map(id => parseInt(id));
  }, [job?.gpu_ids]);

  const { gpuList, isGPUInfoLoaded } = useGPUInfo(gpuIds, 5000);
  const { log, status: statusLog } = useJobLog(job?.id ?? '', isActive ? 2000 : null);

  const totalSteps = job ? getTotalSteps(job) : 0;
  const progress = totalSteps > 0 ? (job!.step / totalSteps) * 100 : 0;

  const logRef = useRef<HTMLDivElement>(null);
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);

  const logLines: string[] = useMemo(() => {
    let splits: string[] = log.split(/\n|\r\n/);
    splits = splits.map(line => line.split(/\r/).pop()) as string[];
    const maxLines = 1000;
    if (splits.length > maxLines) {
      splits = splits.slice(splits.length - maxLines);
    }
    return splits;
  }, [log]);

  const handleScroll = () => {
    if (logRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logRef.current;
      setIsScrolledToBottom(scrollHeight - scrollTop - clientHeight < 10);
    }
  };

  useEffect(() => {
    if (logRef.current && isScrolledToBottom) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log, isScrolledToBottom]);

  // Animate the docked height instead of translating, so the collapsed panel
  // never extends below the container and adds scroll. Inactive -> 0 (hidden),
  // collapsed -> just the header bar, expanded -> full panel.
  const HEADER_HEIGHT = 44;
  const PANEL_HEIGHT = 300;
  let height = 0;
  if (isActive) height = collapsed ? HEADER_HEIGHT : PANEL_HEIGHT;

  useEffect(() => {
    onHeightChange?.(height);
  }, [height, onHeightChange]);

  return (
    <div
      className="absolute bottom-0 left-0 w-full z-40 overflow-hidden transition-[height] duration-300"
      style={{ height: `${height}px` }}
    >
      <div
        className="bg-gray-900 border-t border-gray-700 shadow-2xl flex flex-col"
        style={{ height: `${PANEL_HEIGHT}px` }}
      >
        {/* Header / action bar */}
        <div
          className="bg-gray-800 px-3 flex items-center gap-2 flex-shrink-0"
          style={{ height: `${HEADER_HEIGHT}px` }}
        >
          <span className="px-2 py-0.5 rounded-full text-xs bg-emerald-500/10 text-emerald-500 whitespace-nowrap">
            {job?.status ?? '...'}
          </span>
          <h2 className="text-sm text-gray-100 truncate min-w-0 flex-shrink">{job?.info || 'Captioning...'}</h2>
          {totalSteps > 0 && (
            <div className="hidden sm:flex items-center gap-2 flex-1 min-w-0">
              <div className="flex-1 bg-gray-700 rounded-full h-2 min-w-0">
                <div className="h-2 rounded-full bg-blue-500 transition-all" style={{ width: `${progress}%` }} />
              </div>
              <span className="text-xs text-gray-300 whitespace-nowrap">
                {job!.step} / {totalSteps}
              </span>
            </div>
          )}
          <div className="flex items-center gap-3 flex-shrink-0">
            {job && <JobActionBar job={job} onRefresh={refreshJob} autoStartQueue={true} menuAnchor="top end" />}
            <button
              onClick={() => setCollapsed(c => !c)}
              className="text-gray-400 hover:text-gray-100"
              title={collapsed ? 'Show' : 'Hide'}
            >
              {collapsed ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Body: log + GPU info */}
        <div className="flex flex-1 min-h-0">
          <div className="flex-1 min-w-0 p-3">
            <div className="bg-gray-950 rounded-lg h-full relative">
              <div
                ref={logRef}
                className="text-xs text-gray-300 absolute inset-0 p-3 overflow-y-auto"
                onScroll={handleScroll}
              >
                {statusLog === 'loading' && 'Loading log...'}
                {statusLog === 'error' && 'Error loading log'}
                {['success', 'refreshing'].includes(statusLog) && (
                  <div>
                    {logLines.map((line, index) => (
                      <pre key={index}>{line}</pre>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="w-96 flex-shrink-0 overflow-y-auto p-3 pl-0 hidden md:block">
            {isGPUInfoLoaded && gpuList.length > 0 ? (
              <GPUWidget gpu={gpuList[0]} />
            ) : (
              <div className="flex items-center gap-2 text-xs text-gray-500 p-2">
                <Cpu className="w-4 h-4" /> Loading GPU info...
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
