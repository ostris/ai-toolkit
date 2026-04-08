import prisma from '../prisma';

import { Job, Queue } from '@prisma/client';
import startJob from './startJob';

/**
 * Parse gpu_ids string into a set of individual GPU IDs.
 * e.g., "0,1" -> Set{"0", "1"}, "0" -> Set{"0"}
 */
function parseGpuIds(gpuIds: string): Set<string> {
  return new Set(
    gpuIds
      .split(',')
      .map(s => s.trim())
      .filter(s => s.length > 0),
  );
}

/**
 * Check if two gpu_ids strings have any overlapping GPUs.
 */
function gpuIdsOverlap(a: string, b: string): boolean {
  const setA = parseGpuIds(a);
  for (const gpu of parseGpuIds(b)) {
    if (setA.has(gpu)) return true;
  }
  return false;
}

/**
 * Check if two gpu_ids strings represent the same set of GPUs
 * (order-independent equality).
 */
function gpuIdsEqual(a: string, b: string): boolean {
  const setA = parseGpuIds(a);
  const setB = parseGpuIds(b);
  if (setA.size !== setB.size) return false;
  for (const gpu of setA) {
    if (!setB.has(gpu)) return false;
  }
  return true;
}

export default async function processQueue() {
  const queues: Queue[] = await prisma.queue.findMany({
    orderBy: {
      id: 'asc',
    },
  });

  // Build a set of all occupied GPU IDs from currently running/stopping jobs.
  // This prevents multi-GPU jobs from conflicting with single-GPU jobs and vice versa.
  const allRunningJobs: Job[] = await prisma.job.findMany({
    where: {
      status: { in: ['running', 'stopping'] },
    },
  });

  const occupiedGpus = new Set<string>();
  for (const rj of allRunningJobs) {
    for (const gpu of parseGpuIds(rj.gpu_ids)) {
      occupiedGpus.add(gpu);
    }
  }

  for (const queue of queues) {
    if (!queue.is_running) {
      // stop any running jobs whose GPUs overlap with this queue
      const runningJobs: Job[] = allRunningJobs.filter(
        j => gpuIdsOverlap(j.gpu_ids, queue.gpu_ids) && j.status === 'running',
      );

      for (const job of runningJobs) {
        // Don't stop a job if it belongs to a different queue that IS running.
        // e.g., a stopped queue for GPU "0" must not kill a multi-GPU job on "0,1"
        // that is managed by an active queue for "0,1".
        const belongsToActiveQueue = queues.some(
          q => q.id !== queue.id && q.is_running && gpuIdsOverlap(q.gpu_ids, job.gpu_ids),
        );
        if (belongsToActiveQueue) continue;

        console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
        await prisma.job.update({
          where: { id: job.id },
          data: {
            return_to_queue: true,
            info: 'Stopping job...',
          },
        });
      }
    }
    if (queue.is_running) {
      // Check if any running job uses GPUs that overlap with this queue
      const hasOverlappingRunningJob = allRunningJobs.some(
        rj => (rj.status === 'running' || rj.status === 'stopping') && gpuIdsOverlap(rj.gpu_ids, queue.gpu_ids),
      );

      if (hasOverlappingRunningJob) {
        // GPUs are busy, skip to next queue
        continue;
      } else {
        // find the next job in the queue whose GPUs exactly match this queue
        const queuedJobs: Job[] = await prisma.job.findMany({
          where: {
            status: 'queued',
          },
          orderBy: {
            queue_position: 'asc',
          },
        });
        const nextJob: Job | null = queuedJobs.find(j => gpuIdsEqual(j.gpu_ids, queue.gpu_ids)) ?? null;
        if (nextJob) {
          // Verify all GPUs needed by this job are currently free
          const jobGpus = parseGpuIds(nextJob.gpu_ids);
          const allFree = [...jobGpus].every(gpu => !occupiedGpus.has(gpu));

          if (allFree) {
            console.log(`Starting job ${nextJob.id} on GPU(s) ${nextJob.gpu_ids}`);
            await startJob(nextJob.id);
            // Mark these GPUs as occupied for remaining queue iterations
            for (const gpu of jobGpus) {
              occupiedGpus.add(gpu);
            }
          } else {
            console.log(`Job ${nextJob.id} needs GPU(s) ${nextJob.gpu_ids} but some are occupied, skipping`);
          }
        } else {
          // No queued jobs for this queue. Only auto-stop if there are also no
          // running/stopping jobs that belong to THIS queue — otherwise the queue
          // must stay active so that stopped single-GPU queues don't kill its job
          // (the belongsToActiveQueue check relies on the queue being is_running).
          const hasActiveJob = allRunningJobs.some(rj => gpuIdsEqual(rj.gpu_ids, queue.gpu_ids));
          if (!hasActiveJob) {
            console.log(`No more jobs in queue for GPU(s) ${queue.gpu_ids}, stopping queue`);
            await prisma.queue.update({
              where: { id: queue.id },
              data: { is_running: false },
            });
          }
        }
      }
    }
  }
}
