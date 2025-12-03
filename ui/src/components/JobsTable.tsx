import { useMemo } from 'react';
import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { GpuInfo, JobConfig } from '@/types';
import JobActionBar from './JobActionBar';
import { Job, Queue } from '@prisma/client';
import useQueueList from '@/hooks/useQueueList';
import classNames from 'classnames';
import { startQueue, stopQueue } from '@/utils/queue';
import { CgSpinner } from 'react-icons/cg';
import useGPUInfo from '@/hooks/useGPUInfo';

interface JobsTableProps {
  autoStartQueue?: boolean;
  onlyActive?: boolean;
}

export default function JobsTable({ onlyActive = false }: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList(onlyActive, 5000);
  const { queues, status: queueStatus, refreshQueues } = useQueueList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  const columns: TableColumn[] = [
    {
      title: 'Name',
      key: 'name',
      render: row => (
        <Link href={`/jobs/${row.id}`} className="font-medium whitespace-nowrap">
          {['running', 'stopping'].includes(row.status) ? (
            <CgSpinner className="inline animate-spin mr-2 text-blue-400" />
          ) : null}
          {row.name}
        </Link>
      ),
    },
    {
      title: 'Steps',
      key: 'steps',
      render: row => {
        const jobConfig: JobConfig = JSON.parse(row.job_config);
        const totalSteps = jobConfig.config.process[0].train.steps;

        return (
          <div>
            <div className="text-xs text-gray-400">
              {row.step} / {totalSteps}
            </div>
            <div className="bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full"
                style={{ width: `${(row.step / totalSteps) * 100}%` }}
              ></div>
            </div>
          </div>
        );
      },
    },
    {
      title: 'GPU',
      key: 'gpu_ids',
    },
    {
      title: 'Status',
      key: 'status',
      render: row => {
        let statusClass = 'text-gray-400';
        if (row.status === 'completed') statusClass = 'text-green-400';
        if (row.status === 'failed') statusClass = 'text-red-400';
        if (row.status === 'running') statusClass = 'text-blue-400';

        return <span className={statusClass}>{row.status}</span>;
      },
    },
    {
      title: 'Info',
      key: 'info',
      className: 'truncate max-w-xs',
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'text-right',
      render: row => {
        return <JobActionBar job={row} onRefresh={refreshJobs} autoStartQueue={false} />;
      },
    },
  ];

  const jobsDict = useMemo(() => {
    if (!isGPUInfoLoaded) return {};
    if (jobs.length === 0) return {};
    let jd: { [key: string]: { name: string; jobs: Job[] } } = {};
    gpuList.forEach(gpu => {
      jd[`${gpu.index}`] = { name: `${gpu.name}`, jobs: [] };
    });
    jd['Idle'] = { name: 'Idle', jobs: [] };
    jobs.forEach(job => {
      const gpu = gpuList.find(gpu => job.gpu_ids?.split(',').includes(gpu.index.toString())) as GpuInfo;
      const key = `${gpu?.index || '0'}`;
      if (['queued', 'running', 'stopping'].includes(job.status) && key in jd) {
        jd[key].jobs.push(job);
      } else {
        jd['Idle'].jobs.push(job);
      }
    });
    // sort the queued/running jobs by queue position
    Object.keys(jd).forEach(key => {
      if (key === 'Idle') return;
      jd[key].jobs.sort((a, b) => {
        if (a.queue_position === null) return 1;
        if (b.queue_position === null) return -1;
        return a.queue_position - b.queue_position;
      });
    });
    return jd;
  }, [jobs, queues, isGPUInfoLoaded]);

  let isLoading = status === 'loading' || queueStatus === 'loading' || !isGPUInfoLoaded;

  // if job dict is populated, we are always loaded
  if (Object.keys(jobsDict).length > 0) isLoading = false;

  return (
    <div>
      {Object.keys(jobsDict)
        .sort()
        .filter(key => key !== 'Idle')
        .map(gpuKey => {
          const queue = queues.find(q => `${q.gpu_ids}` === gpuKey) as Queue;
          return (
            <div key={gpuKey} className="mb-6">
              <div
                className={classNames(
                  'text-md flex px-2 md:px-4 py-1 rounded-t-lg items-center gap-2',
                  { 'bg-green-900': queue?.is_running },
                  { 'bg-red-900': !queue?.is_running },
                )}
              >
                <div className="flex items-center gap-2 flex-1 py-2 min-w-0">
                  <h2 className="font-semibold text-gray-100 truncate text-sm md:text-base">{jobsDict[gpuKey].name}</h2>
                  <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300 whitespace-nowrap shrink-0"># {queue?.gpu_ids}</span>
                </div>
                <div className="text-sm text-gray-300 flex items-center gap-2 shrink-0">
                  {queue?.is_running ? (
                    <>
                      {/* Mobile: dot indicator, Desktop: text */}
                      <span className="hidden md:inline text-green-400 italic">Queue Running</span>
                      <span className="md:hidden w-2 h-2 rounded-full bg-green-400" title="Queue Running"></span>
                      <button
                        onClick={async () => {
                          await stopQueue(queue.gpu_ids as string);
                          refresh();
                        }}
                        className="text-xs bg-red-900 hover:bg-red-800 px-2 py-1 rounded"
                      >
                        STOP
                      </button>
                    </>
                  ) : (
                    <>
                      {/* Mobile: dot indicator, Desktop: text */}
                      <span className="hidden md:inline text-red-400 italic">Queue Stopped</span>
                      <span className="md:hidden w-2 h-2 rounded-full bg-red-400" title="Queue Stopped"></span>
                      <button
                        onClick={async () => {
                          await startQueue(gpuKey);
                          refresh();
                        }}
                        className="text-xs bg-green-700 hover:bg-green-600 px-2 py-1 rounded"
                      >
                        START
                      </button>
                    </>
                  )}
                </div>
              </div>
              <UniversalTable
                columns={columns}
                rows={jobsDict[gpuKey].jobs}
                isLoading={isLoading}
                onRefresh={refresh}
                theadClassName={queue?.is_running ? 'bg-green-950' : 'bg-red-950'}
              />
            </div>
          );
        })}
      {!onlyActive && Object.keys(jobsDict).includes('Idle') && (
        <div className="mb-6 opacity-50">
          <div className="text-md flex px-4 py-1 rounded-t-lg bg-slate-600">
            <div className="flex items-center space-x-2 flex-1 py-2">
              <h2 className="font-semibold text-gray-100">Idle</h2>
            </div>
          </div>
          <UniversalTable columns={columns} rows={jobsDict['Idle'].jobs} isLoading={isLoading} onRefresh={refresh} />
        </div>
      )}
    </div>
  );
}
