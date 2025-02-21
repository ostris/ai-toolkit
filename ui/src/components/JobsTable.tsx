import useJobsList from '@/hooks/useJobsList';
import Loading from './Loading';
import { JobConfig } from '@/types';
import Link from 'next/link';

interface JobsTableProps {}

export default function JobsTable(props: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList();
  const isLoading = status === 'loading';

  return (
    <div className="w-full bg-gray-900 rounded-md shadow-md">
      {isLoading ? (
        <div className="p-4 flex justify-center">
          <Loading />
        </div>
      ) : jobs.length === 0 ? (
        <div className="p-6 text-center text-gray-400">
          <p className="text-sm">No jobs available</p>
          <button
            onClick={() => refreshJobs()}
            className="mt-2 px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded transition-colors"
          >
            Refresh
          </button>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-gray-300">
            <thead className="text-xs uppercase bg-gray-800 text-gray-400">
              <tr>
                <th className="px-3 py-2">Name</th>
                <th className="px-3 py-2">Steps</th>
                <th className="px-3 py-2">GPU</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Info</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job, index) => {
                const jobConfig: JobConfig = JSON.parse(job.job_config);
                const totalSteps = jobConfig.config.process[0].train.steps;

                // Style for alternating rows
                const rowClass = index % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800';

                // Style based on job status
                let statusClass = 'text-gray-400';
                if (job.status === 'completed') statusClass = 'text-green-400';
                if (job.status === 'failed') statusClass = 'text-red-400';
                if (job.status === 'running') statusClass = 'text-blue-400';

                return (
                  <tr key={job.id} className={`${rowClass} border-b border-gray-700 hover:bg-gray-700`}>
                    <td className="px-3 py-2 font-medium whitespace-nowrap">
                      <Link href={`/jobs/${job.id}`}>{job.name}</Link></td>
                    <td className="px-3 py-2">
                      <div className="flex items-center">
                        <span>
                          {job.step} / {totalSteps}
                        </span>
                        <div className="w-16 bg-gray-700 rounded-full h-1.5 ml-2">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full"
                            style={{ width: `${(job.step / totalSteps) * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-2">{job.gpu_id}</td>
                    <td className={`px-3 py-2 ${statusClass}`}>{job.status}</td>
                    <td className="px-3 py-2 truncate max-w-xs">{job.info}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
