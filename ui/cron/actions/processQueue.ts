import prisma from '../prisma';
import path from 'path';
import fs from 'fs';

import { Job, Queue } from '@prisma/client';
import startJob from './startJob';
import { getTrainingFolder } from '../paths';

// Check if a process with given PID is still running
function isProcessAlive(pid: number): boolean {
  try {
    // Signal 0 doesn't actually send a signal, just checks if process exists
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    // ESRCH means process not found (dead)
    // EPERM means process exists but we don't have permission (still alive)
    if (e.code === 'ESRCH') {
      return false;
    }
    // If permission denied, process is alive
    return e.code === 'EPERM';
  }
}

// Check if a job's process is still alive, mark as crashed if not
async function checkJobProcessHealth(job: Job): Promise<boolean> {
  try {
    const trainingRoot = await getTrainingFolder();
    const trainingFolder = path.join(trainingRoot, job.name);
    const pidPath = path.join(trainingFolder, 'pid.txt');

    if (!fs.existsSync(pidPath)) {
      // No PID file - can't verify, assume dead
      console.warn(`No PID file found for job ${job.name}, marking as crashed`);
      return false;
    }

    const pidStr = fs.readFileSync(pidPath, 'utf-8').trim();
    const pid = parseInt(pidStr, 10);

    if (isNaN(pid) || pid <= 0) {
      console.warn(`Invalid PID in file for job ${job.name}: ${pidStr}`);
      return false;
    }

    const alive = isProcessAlive(pid);
    if (!alive) {
      console.log(`Process ${pid} for job ${job.name} is no longer running`);
    }
    return alive;
  } catch (e) {
    console.error(`Error checking process health for job ${job.name}:`, e);
    return false; // Assume dead on error
  }
}

export default async function processQueue() {
  // First, check all running/stopping jobs to see if their processes are still alive
  const allRunningJobs: Job[] = await prisma.job.findMany({
    where: {
      status: { in: ['running', 'stopping'] },
    },
  });

  for (const job of allRunningJobs) {
    // Re-fetch job to get latest status (process might have updated it before exiting)
    const currentJob = await prisma.job.findUnique({
      where: { id: job.id },
    });

    if (!currentJob) continue;

    // If status already changed to completed/stopped/error, skip PID check
    if (!['running', 'stopping'].includes(currentJob.status)) {
      continue;
    }

    const isAlive = await checkJobProcessHealth(currentJob);
    if (!isAlive) {
      // Double-check status hasn't changed (race condition protection)
      const finalCheck = await prisma.job.findUnique({
        where: { id: job.id },
      });

      if (finalCheck && ['running', 'stopping'].includes(finalCheck.status)) {
        console.log(`Job ${job.name} process died unexpectedly, marking as error`);
        await prisma.job.update({
          where: { id: job.id },
          data: {
            status: 'error',
            info: 'Process crashed or was killed externally',
          },
        });
      }
    }
  }

  const queues: Queue[] = await prisma.queue.findMany({
    orderBy: {
      id: 'asc',
    },
  });

  for (const queue of queues) {
    if (!queue.is_running) {
      // stop any running jobs first
      const runningJobs: Job[] = await prisma.job.findMany({
        where: {
          status: 'running',
          gpu_ids: queue.gpu_ids,
        },
      });

      for (const job of runningJobs) {
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
      // first see if one is already running, status of running or stopping
      const runningJob: Job | null = await prisma.job.findFirst({
        where: {
          status: { in: ['running', 'stopping'] },
          gpu_ids: queue.gpu_ids,
        },
      });

      if (runningJob) {
        // already running, nothing to do
        continue; // skip to next queue
      } else {
        // find the next job in the queue
        const nextJob: Job | null = await prisma.job.findFirst({
          where: {
            status: 'queued',
            gpu_ids: queue.gpu_ids,
          },
          orderBy: {
            queue_position: 'asc',
          },
        });
        if (nextJob) {
          console.log(`Starting job ${nextJob.id} on GPU(s) ${nextJob.gpu_ids}`);
          await startJob(nextJob.id);
        } else {
          // no more jobs, stop the queue
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
