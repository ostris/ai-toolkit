import FilesWidget from '@/components/FilesWidget';
import GPUWidget from '@/components/GPUWidget';
import useGPUInfo from '@/hooks/useGPUInfo';
import useJobLog from '@/hooks/useJobLog';
import { getTotalSteps } from '@/utils/jobs';
import { Job } from '@prisma/client';
import { Info } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

interface JobOverviewProps {
  job: Job;
}

export default function JobOverview({ job }: JobOverviewProps) {
  const gpuIds = useMemo(() => job.gpu_ids.split(',').map(id => parseInt(id)), [job.gpu_ids]);
  const { log, setLog, status: statusLog, refresh: refreshLog } = useJobLog(job.id, 2000);
  const logRef = useRef<HTMLDivElement>(null);
  // Track whether we should auto-scroll to bottom
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);

  const { gpuList, isGPUInfoLoaded } = useGPUInfo(gpuIds, 5000);
  const totalSteps = getTotalSteps(job);
  const progress = (job.step / totalSteps) * 100;
  const isStopping = job.stop && job.status === 'running';

  const logLines: string[] = useMemo(() => {
    // split at line breaks on \n or \r\n but not \r
    let splits: string[] = log.split(/\n|\r\n/);

    splits = splits.map(line => {
      return line.split(/\r/).pop();
    }) as string[];

    // only return last 100 lines max
    const maxLines = 1000;
    if (splits.length > maxLines) {
      splits = splits.slice(splits.length - maxLines);
    }

    return splits;
  }, [log]);

  // Handle scroll events to determine if user has scrolled away from bottom
  const handleScroll = () => {
    if (logRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logRef.current;
      // Consider "at bottom" if within 10 pixels of the bottom
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
      setIsScrolledToBottom(isAtBottom);
    }
  };

  // Auto-scroll to bottom only if we were already at the bottom
  useEffect(() => {
    if (logRef.current && isScrolledToBottom) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log, isScrolledToBottom]);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
        return 'bg-emerald-500/10 text-emerald-500';
      case 'stopping':
        return 'bg-amber-500/10 text-amber-500';
      case 'stopped':
        return 'bg-gray-500/10 text-gray-400';
      case 'completed':
        return 'bg-blue-500/10 text-blue-500';
      case 'error':
        return 'bg-rose-500/10 text-rose-500';
      default:
        return 'bg-gray-500/10 text-gray-400';
    }
  };

  let status = job.status;
  if (isStopping) {
    status = 'stopping';
  }

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
      {/* Job Information Panel */}
      <div className="col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
        <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
          <h2 className="text-gray-100">
            <Info className="w-5 h-5 mr-2 -mt-1 text-amber-400 inline-block" /> {job.info}
          </h2>
          <span className={`px-3 py-1 rounded-full text-sm ${getStatusColor(job.status)}`}>{job.status}</span>
        </div>

        <div className="p-4 space-y-6 flex flex-col flex-grow">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Progress</span>
              <span className="text-gray-200">
                Step {job.step} of {totalSteps}
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div className="h-2 rounded-full bg-blue-500 transition-all" style={{ width: `${progress}%` }} />
            </div>
          </div>

          {/* Job Info Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Status</h3>
              <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{job.status}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Step</h3>
              <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{job.step}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">GPUs</h3>
              <p className="text-sm font-medium text-gray-200">
                {job.use_multi_gpu ? (
                  <span className="text-blue-600 dark:text-blue-400">
                    Multi-GPU ({job.num_gpus} GPUs)
                  </span>
                ) : (
                  `GPU ${job.gpu_ids}`
                )}
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Speed</h3>
              <p className="text-sm font-medium text-gray-200">{job.speed_string || 'N/A'}</p>
            </div>
          </div>

          {/* Log - Now using flex-grow to fill remaining space */}
          <div className="bg-gray-950 rounded-lg p-4 relative flex-grow min-h-60">
            <div
              ref={logRef}
              className="text-xs text-gray-300 absolute inset-0 p-4 overflow-y-auto"
              onScroll={handleScroll}
            >
              {statusLog === 'loading' && 'Loading log...'}
              {statusLog === 'error' && 'Error loading log'}
              {['success', 'refreshing'].includes(statusLog) && (
                <div>
                  {logLines.map((line, index) => {
                    return <pre key={index}>{line}</pre>;
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* GPU Widget Panel */}
      <div className="col-span-1">
        <div>{isGPUInfoLoaded && gpuList.length > 0 && <GPUWidget gpu={gpuList[0]} />}</div>
        <div className="mt-4">
          <FilesWidget jobID={job.id} />
        </div>
      </div>
    </div>
  );
}
