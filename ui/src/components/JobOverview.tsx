import { Job } from '@prisma/client';
import useGPUInfo from '@/hooks/useGPUInfo';
import useCPUInfo from '@/hooks/useCPUInfo';
import GPUWidget from '@/components/GPUWidget';
import CPUWidget from '@/components/CPUWidget';
import FilesWidget from '@/components/FilesWidget';
import { getTotalSteps } from '@/utils/jobs';
import { Cpu, HardDrive, Info, Gauge, GitBranch } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import useJobLog from '@/hooks/useJobLog';
import { apiClient } from '@/utils/api';

interface JobOverviewProps {
  job: Job;
}

export default function JobOverview({ job }: JobOverviewProps) {
  const gpuIds = useMemo(() => {
    if (job.gpu_ids === 'mps') {
      return [0]; // For MPS, we can just return a single GPU ID since it's virtualized
    }
    return job.gpu_ids.split(',').map(id => parseInt(id));
  }, [job.gpu_ids]);
  const { log, status: statusLog, refresh: refreshLog } = useJobLog(job.id, 2000);
  const logRef = useRef<HTMLDivElement>(null);
  const [parentJob, setParentJob] = useState<Job | null>(null);
  const [childJobs, setChildJobs] = useState<Job[]>([]);
  
  // Track whether we should auto-scroll to bottom
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(gpuIds, 5000);
  const { cpuInfo, isCPUInfoLoaded } = useCPUInfo(5000);
  const totalSteps = getTotalSteps(job);
  const progress = (job.step / totalSteps) * 100;
  const isStopping = job.stop && job.status === 'running';

  // Fetch fork information
  useEffect(() => {
    const fetchForkInfo = async () => {
      try {
        const response = await apiClient.get(`/api/jobs/${job.id}/fork`);
        if (response.data) {
          setParentJob(response.data.parentJob);
          setChildJobs(response.data.childJobs || []);
        }
      } catch (error) {
        console.error('Error fetching fork info:', error);
      }
    };

    fetchForkInfo();
  }, [job.id]);

  const logLines: string[] = useMemo(() => {
    // Log is already terminal-rendered by useJobLog — one entry per line.
    let splits: string[] = log.split('\n');

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

  const jobType = job?.job_type || 'unknown';

  let status = job.status;
  if (isStopping) {
    status = 'stopping';
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:gap-6 md:grid-cols-3">
      {/* Job Information Panel */}
      <div className="md:col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
        <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
          <h2 className="text-gray-100">
            <Info className="w-5 h-5 mr-2 -mt-1 text-amber-600 dark:text-amber-400 inline-block" /> {job.info}
          </h2>
          <span className={`px-3 py-1 rounded-full text-sm ${getStatusColor(job.status)}`}>{job.status}</span>
        </div>

        <div className="p-4 space-y-6 flex flex-col flex-grow">
          {/* Progress Bar */}
          {totalSteps > 0 && (
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
          )}

          {/* Job Info Grid */}
          <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
            <div className="flex items-center space-x-4">
              <HardDrive className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <div>
                <p className="text-xs text-gray-400">Job Name</p>
                <p className="text-sm font-medium text-gray-200">{job.name}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Cpu className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              <div>
                <p className="text-xs text-gray-400">Assigned GPUs</p>
                <p className="text-sm font-medium text-gray-200">GPUs: {job.gpu_ids}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Gauge className="w-5 h-5 text-green-600 dark:text-green-400" />
              <div>
                <p className="text-xs text-gray-400">Speed</p>
                <p className="text-sm font-medium text-gray-200">{job.speed_string == '' ? '?' : job.speed_string}</p>
              </div>
            </div>
          </div>

          {/* Fork Information */}
          {(parentJob || childJobs.length > 0) && (
            <div className="border-t border-gray-700 pt-4 space-y-3">
              {parentJob && (
                <div className="flex items-center space-x-4 text-sm">
                  <GitBranch className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0" />
                  <div className="flex-grow">
                    <p className="text-xs text-gray-400">Forked from</p>
                    <p className="text-gray-200">
                      <a href={`/jobs/${parentJob.id}`} className="hover:text-blue-400 underline">
                        {parentJob.name}
                      </a>
                      {job.fork_from_step && <span className="text-gray-500 ml-2">(at step {job.fork_from_step})</span>}
                    </p>
                  </div>
                </div>
              )}

              {childJobs.length > 0 && (
                <div className="text-sm">
                  <p className="text-xs text-gray-400 mb-2">Child Forks ({childJobs.length})</p>
                  <div className="space-y-1">
                    {childJobs.map((childJob) => (
                      <div key={childJob.id} className="flex items-center space-x-2 pl-4">
                        <GitBranch className="w-3 h-3 text-green-600 dark:text-green-400" />
                        <a href={`/jobs/${childJob.id}`} className="text-gray-300 hover:text-blue-400 underline text-xs">
                          {childJob.name}
                        </a>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

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
      <div className="md:col-span-1">
        <div>{isCPUInfoLoaded && cpuInfo && <CPUWidget cpu={cpuInfo} />}</div>
        <div className="mt-4">{isGPUInfoLoaded && gpuList.length > 0 && <GPUWidget gpu={gpuList[0]} />}</div>
        {jobType === 'train' && (
          <div className="mt-4">
            <FilesWidget jobID={job.id} jobName={job.name} />
          </div>
        )}
      </div>
    </div>
  );
}
