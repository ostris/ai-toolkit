import prisma from '../prisma';

import { Job, Queue } from '@prisma/client';
import startJob from './startJob';

const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1000;

function isPrismaTimeoutError(error: unknown): boolean {
  return (
    error instanceof Error &&
    'code' in error &&
    (error as { code: string }).code === 'P1008'
  );
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (isPrismaTimeoutError(error) && attempt < MAX_RETRIES) {
        const delay = BASE_DELAY_MS * Math.pow(2, attempt);
        console.warn(
          `Database timeout (attempt ${attempt + 1}/${MAX_RETRIES + 1}), retrying in ${delay}ms...`
        );
        await sleep(delay);
      } else {
        throw error;
      }
    }
  }
  // unreachable, but satisfies TypeScript
  throw new Error('withRetry: exceeded max retries');
}

export default async function processQueue() {
  const queues: Queue[] = await withRetry(() =>
    prisma.queue.findMany({
      orderBy: {
        id: 'asc',
      },
    })
  );

  for (const queue of queues) {
    if (!queue.is_running) {
      // stop any running jobs first
      const runningJobs: Job[] = await withRetry(() =>
        prisma.job.findMany({
          where: {
            status: 'running',
            gpu_ids: queue.gpu_ids,
          },
        })
      );

      for (const job of runningJobs) {
        console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
        await withRetry(() =>
          prisma.job.update({
            where: { id: job.id },
            data: {
              return_to_queue: true,
              info: 'Stopping job...',
            },
          })
        );
      }
    }
    if (queue.is_running) {
      // first see if one is already running, status of running or stopping
      const runningJob: Job | null = await withRetry(() =>
        prisma.job.findFirst({
          where: {
            status: { in: ['running', 'stopping'] },
            gpu_ids: queue.gpu_ids,
          },
        })
      );

      if (runningJob) {
        // already running, nothing to do
        continue; // skip to next queue
      } else {
        // find the next job in the queue
        const nextJob: Job | null = await withRetry(() =>
          prisma.job.findFirst({
            where: {
              status: 'queued',
              gpu_ids: queue.gpu_ids,
            },
            orderBy: {
              queue_position: 'asc',
            },
          })
        );
        if (nextJob) {
          console.log(`Starting job ${nextJob.id} on GPU(s) ${nextJob.gpu_ids}`);
          await startJob(nextJob.id);
        } else {
          // no more jobs, stop the queue
          console.log(`No more jobs in queue for GPU(s) ${queue.gpu_ids}, stopping queue`);
          await withRetry(() =>
            prisma.queue.update({
              where: { id: queue.id },
              data: { is_running: false },
            })
          );
        }
      }
    }
  }
}
