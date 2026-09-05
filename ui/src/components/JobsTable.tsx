import { useMemo, useState } from 'react';
import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { GpuInfo } from '@/types';
import JobActionBar from './JobActionBar';
import { Job, Queue } from '@prisma/client';
import useQueueList from '@/hooks/useQueueList';
import classNames from 'classnames';
import { startQueue, stopQueue } from '@/utils/queue';
import { CgSpinner } from 'react-icons/cg';
import useGPUInfo from '@/hooks/useGPUInfo';
import { openConfirm } from '@/components/ConfirmModal';
import { deleteJob, getTotalSteps, stopJob } from '@/utils/jobs';
import { Trash2 } from 'lucide-react';

interface JobsTableProps {
  autoStartQueue?: boolean;
  onlyActive?: boolean;
  job_type?: string | null;
}

export default function JobsTable({ onlyActive = false, job_type = null }: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList({ onlyActive, reloadInterval: 5000, job_type });
  const { queues, status: queueStatus, refreshQueues } = useQueueList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleteProgress, setDeleteProgress] = useState<{ done: number; total: number } | null>(null);

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  const isDeleting = deleteProgress !== null;
  const allSelected = jobs.length > 0 && jobs.every(job => selectedIds.has(job.id));

  const toggleRow = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleAll = () => {
    setSelectedIds(allSelected ? new Set() : new Set(jobs.map(job => job.id)));
  };

  const onMassDelete = () => {
    const jobsToDelete = jobs.filter(job => selectedIds.has(job.id));
    if (jobsToDelete.length === 0) return;
    const runningCount = jobsToDelete.filter(job => job.status === 'running').length;
    let message = `Are you sure you want to delete ${jobsToDelete.length} job${
      jobsToDelete.length === 1 ? '' : 's'
    }? This will also permanently remove them from your disk.`;
    if (runningCount > 0) {
      message += ` WARNING: ${runningCount} of them ${
        runningCount === 1 ? 'is' : 'are'
      } currently running and will be stopped first.`;
    }
    openConfirm({
      title: 'Delete Jobs',
      message: message,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        setDeleteProgress({ done: 0, total: jobsToDelete.length });
        for (let i = 0; i < jobsToDelete.length; i++) {
          const job = jobsToDelete[i];
          try {
            if (job.status === 'running') {
              try {
                await stopJob(job.id);
              } catch (e) {
                console.error('Error stopping job before deleting:', e);
              }
            }
            await deleteJob(job.id);
            setSelectedIds(prev => {
              const next = new Set(prev);
              next.delete(job.id);
              return next;
            });
          } catch (e) {
            console.error('Error deleting job:', job.name, e);
          }
          setDeleteProgress({ done: i + 1, total: jobsToDelete.length });
          refreshJobs();
        }
        setDeleteProgress(null);
        refresh();
      },
    });
  };

  const columns: TableColumn[] = [
    {
      title: (
        <input
          type="checkbox"
          checked={allSelected}
          onChange={toggleAll}
          disabled={isDeleting}
          className="cursor-pointer accent-blue-500"
        />
      ),
      key: 'select',
      className: 'w-8',
      render: row => (
        <input
          type="checkbox"
          checked={selectedIds.has(row.id)}
          onChange={() => toggleRow(row.id)}
          disabled={isDeleting}
          className="cursor-pointer accent-blue-500"
        />
      ),
    },
    {
      title: 'Name',
      key: 'name',
      render: row => {
        let title = row.name;
        let href = `/jobs/${row.id}`;
        // if (row.job_type === 'train') title = `Train: ${title}`;
        if (row.job_type === 'caption') {
          let splits = row.job_ref.split(/[/\\]/);
          const datasetPath = `${splits[splits.length - 1]}`;
          href = `/datasets/${datasetPath}`;
          title = (
            <>
              <small className="opacity-50">CAPTION: </small> {datasetPath}
            </>
          );
        }
        return (
          <Link href={href} className="font-medium whitespace-nowrap">
            {['running', 'stopping'].includes(row.status) ? (
              <CgSpinner className="inline animate-spin mr-2 text-blue-400" />
            ) : null}
            {title}
          </Link>
        );
      },
    },
    {
      title: 'Steps',
      key: 'steps',
      render: row => {
        const totalSteps = getTotalSteps(row);
        if (!totalSteps) {
          return <></>;
        }

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
      if (key === 'Idle') {
        jd[key].jobs.sort((a, b) => {
          // sort by updated_at, newest first
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        });
      } else {
        jd[key].jobs.sort((a, b) => {
          if (a.queue_position === null) return 1;
          if (b.queue_position === null) return -1;
          return a.queue_position - b.queue_position;
        });
      }
    });
    return jd;
  }, [jobs, queues, isGPUInfoLoaded]);

  let isLoading = status === 'loading' || queueStatus === 'loading' || !isGPUInfoLoaded;

  // if job dict is populated, we are always loaded
  if (Object.keys(jobsDict).length > 0) isLoading = false;

  return (
    <div>
      {(selectedIds.size > 0 || isDeleting) && (
        <div className="fixed top-16 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2 bg-gray-800 rounded-lg border border-gray-700 shadow-lg">
          {isDeleting ? (
            <>
              <CgSpinner className="inline animate-spin text-red-400" />
              <span className="text-sm text-gray-300">
                Deleting {deleteProgress.done} / {deleteProgress.total}...
              </span>
            </>
          ) : (
            <>
              <span className="text-sm text-gray-300 flex-1">
                {selectedIds.size} job{selectedIds.size === 1 ? '' : 's'} selected
              </span>
              <button
                onClick={() => setSelectedIds(new Set())}
                className="text-xs text-gray-300 bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded"
              >
                Clear
              </button>
              <button
                onClick={onMassDelete}
                className="text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded flex items-center gap-1"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete Selected
              </button>
            </>
          )}
        </div>
      )}
      {Object.keys(jobsDict)
        .sort()
        .filter(key => key !== 'Idle')
        .map(gpuKey => {
          const queue = queues.find(q => `${q.gpu_ids}` === gpuKey) as Queue;
          return (
            <div key={gpuKey} className="mb-6">
              <div
                className={classNames(
                  'text-md flex flex-wrap gap-y-1 px-2 sm:px-4 py-1 rounded-t-lg',
                  { 'bg-green-600 dark:bg-green-900': queue?.is_running },
                  { 'bg-red-600 dark:bg-red-900': !queue?.is_running },
                )}
              >
                <div className="flex items-center space-x-2 flex-1 min-w-0 py-2">
                  <h2 className="font-semibold text-white truncate">{jobsDict[gpuKey].name}</h2>
                  <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300 flex-shrink-0">
                    # {queue?.gpu_ids}
                  </span>
                </div>
                <div className="text-sm text-gray-300 italic flex items-center flex-shrink-0">
                  {queue?.is_running ? (
                    <>
                      <span className="text-green-100 dark:text-green-400 mr-2">Queue Running</span>
                      <button
                        onClick={async () => {
                          await stopQueue(queue.gpu_ids as string);
                          refresh();
                        }}
                        className="ml-2 sm:ml-4 text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded"
                      >
                        STOP
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="text-red-100 dark:text-red-400 mr-2">Queue Stopped</span>
                      <button
                        onClick={async () => {
                          await startQueue(gpuKey);
                          refresh();
                        }}
                        className="ml-2 sm:ml-4 text-xs text-white bg-green-600 hover:bg-green-700 px-2 py-1 rounded"
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
                theadClassName={
                  queue?.is_running
                    ? 'bg-green-700 dark:bg-green-950 text-white dark:text-gray-400'
                    : 'bg-red-700 dark:bg-red-950 text-white dark:text-gray-400'
                }
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
