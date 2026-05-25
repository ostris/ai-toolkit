'use client';

import Link from 'next/link';
import { CgSpinner } from 'react-icons/cg';
import useJobsList from '@/hooks/useJobsList';
import { JobConfig } from '@/types';

export default function ActiveJobWidget() {
  const { jobs } = useJobsList({ onlyActive: true, reloadInterval: 5000 });

  if (!jobs || jobs.length === 0) return null;

  return (
    <div className="px-3 pb-2">
      <div className="w-[196px]">
        <ul className="max-h-48 overflow-y-auto space-y-2">
          {jobs.map(job => {
            let totalSteps: number | undefined;
            try {
              const cfg: JobConfig = JSON.parse(job.job_config);
              totalSteps = cfg.config.process[0].train?.steps;
            } catch {
              totalSteps = undefined;
            }
            const isTrain = job.job_type === 'train';
            const pct = isTrain && totalSteps ? Math.min(100, (job.step / totalSteps) * 100) : 0;
            const isRunning = job.status === 'running' || job.status === 'stopping';

            let label = job.name;
            if (job.job_type === 'caption') {
              const splits = (job.job_ref ?? '').split(/[/\\]/);
              label = splits[splits.length - 1] || job.name;
            }

            let statusColor = 'text-gray-400';
            if (job.status === 'running') statusColor = 'text-blue-400';
            if (job.status === 'queued') statusColor = 'text-yellow-400';
            if (job.status === 'stopping') statusColor = 'text-orange-400';

            return (
              <li key={job.id} className="min-w-0">
                <Link
                  href={`/jobs/${job.id}`}
                  className="block px-3 py-2 bg-gray-800 hover:bg-gray-950 rounded-lg transition-colors min-w-0"
                >
                  <div className="flex items-center gap-1.5 min-w-0">
                    {isRunning && <CgSpinner className="animate-spin text-blue-400 flex-shrink-0" />}
                    <span className="text-xs text-gray-100 truncate min-w-0 flex-1">{label}</span>
                  </div>
                  {isTrain && totalSteps ? (
                    <div className="mt-1.5">
                      <div className="bg-gray-700 rounded-full h-1">
                        <div className="bg-blue-500 h-1 rounded-full transition-all" style={{ width: `${pct}%` }} />
                      </div>
                      <div className="flex justify-between gap-2 mt-0.5 min-w-0">
                        <span className={`text-[10px] truncate min-w-0 ${statusColor}`}>{job.info || job.status}</span>
                        <span className="text-[10px] text-gray-400 flex-shrink-0">
                          {job.step} / {totalSteps}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className={`text-[10px] mt-0.5 truncate ${statusColor}`}>{job.info || job.status}</div>
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
