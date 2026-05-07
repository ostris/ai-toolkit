'use client';

import useJobsList from '@/hooks/useJobsList';
import FlowGRPOVotingPanel from '@/components/FlowGRPOVotingPanel';
import { isFlowGRPOJob } from '@/utils/jobs';

export default function FlowGRPOQueueSection() {
  const { jobs } = useJobsList({ onlyActive: true, reloadInterval: 5000 });
  const flowJobs = jobs.filter(isFlowGRPOJob);

  if (flowJobs.length === 0) {
    return null;
  }

  return (
    <div className="mt-8 space-y-6">
      <div>
        <h2 className="text-lg text-gray-100">Live Flow-GRPO Voting</h2>
        <p className="text-sm text-gray-400">Open vote tasks from active Flow-GRPO jobs can be resolved directly from the queue.</p>
      </div>
      {flowJobs.map(job => (
        <FlowGRPOVotingPanel key={job.id} job={job} compact />
      ))}
    </div>
  );
}
