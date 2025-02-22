import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { JobConfig } from '@/types';

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
  ];

  return <UniversalTable columns={columns} rows={jobs} isLoading={isLoading} onRefresh={refreshJobs} />;
}
