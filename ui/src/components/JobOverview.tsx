import { Job } from '@prisma/client';

interface JobOverviewProps {
  job: Job;
}

export default function JobOverview({ job }: JobOverviewProps) {
  return (
    <>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="">
          <h2 className="text-lg font-semibold">Job Details</h2>
          <p className="text-gray-400">ID: {job.id}</p>
          <p className="text-gray-400">Name: {job.name}</p>
          <p className="text-gray-400">GPUs: {job.gpu_ids}</p>
          <p className="text-gray-400">Status: {job.status}</p>
          <p className="text-gray-400">Info: {job.info}</p>
          <p className="text-gray-400">Step: {job.step}</p>
        </div>
      </div>
    </>
  );
}
