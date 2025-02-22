import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { JobConfig } from '@/types';
import { Eye, Trash2, Pen, Play, Pause } from 'lucide-react';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { startJob, stopJob, deleteJob, getAvaliableJobActions } from '@/utils/jobs';

interface JobsTableProps {}

export default function JobsTable(props: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList();
  const isLoading = status === 'loading';

  const columns: TableColumn[] = [
    {
      title: 'Name',
      key: 'name',
      render: row => (
        <Link href={`/jobs/${row.id}`} className="font-medium whitespace-nowrap">
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
          <div className="flex items-center">
            <span>
              {row.step} / {totalSteps}
            </span>
            <div className="w-16 bg-gray-700 rounded-full h-1.5 ml-2">
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
        const { canDelete, canEdit, canStop, canStart } = getAvaliableJobActions(row);
        return (
          <div>
            {canStart && (
              <Button
                onClick={async () => {
                  if (!canStart) return;
                  await startJob(row.id);
                  refreshJobs();
                }}
                className={`ml-2 opacity-100`}
              >
                <Play />
              </Button>
            )}
            {canStop && (
              <Button
                onClick={() => {
                  if (!canStop) return;
                  openConfirm({
                    title: 'Stop Job',
                    message: `Are you sure you want to stop the job "${row.name}"? You CAN resume later.`,
                    type: 'info',
                    confirmText: 'Stop',
                    onConfirm: async () => {
                      await stopJob(row.id);
                      refreshJobs();
                    },
                  });
                }}
                className={`ml-2 opacity-100`}
              >
                <Pause />
              </Button>
            )}
            <Link href={`/jobs/${row.id}`} className="ml-2 text-gray-200 hover:text-gray-100 inline-block">
              <Eye />
            </Link>
            {canEdit && (
              <Link href={`/jobs/new?id=${row.id}`} className="ml-2 hover:text-gray-100 inline-block">
                <Pen />
              </Link>
            )}
            {canDelete && (
              <Button
                onClick={() => {
                  if (!canDelete) return;
                  openConfirm({
                    title: 'Delete Job',
                    message: `Are you sure you want to delete the job "${row.name}"? This will also permanently remove it from your disk.`,
                    type: 'warning',
                    confirmText: 'Delete',
                    onConfirm: async () => {
                      await deleteJob(row.id);
                      refreshJobs();
                    },
                  });
                }}
                className={`ml-2 opacity-100`}
              >
                <Trash2 />
              </Button>
            )}
          </div>
        );
      },
    },
  ];

  return <UniversalTable columns={columns} rows={jobs} isLoading={isLoading} onRefresh={refreshJobs} />;
}
