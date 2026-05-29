import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // Set both stop flag and save_now flag so the trainer will
  // save a checkpoint before terminating.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      save_now: true,
      info: 'Stopping job with checkpoint save...',
    },
  });

  // Send SIGINT to the process if we have a PID
  if (job.pid != null) {
    console.log(`Attempting to stop job ${jobID} with PID ${job.pid}`);
    try {
      if (process.platform === 'win32') {
        const { execSync } = require('child_process');
        execSync(`taskkill /PID ${job.pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(job.pid, 'SIGINT');
      }
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'stopped',
          info: 'Job stopped with checkpoint saved',
          stop: false,
          save_now: false,
        },
      });
    } catch (e) {
      // Process may have already exited — that's fine
      console.error('Error sending signal to process:', e);
    }
  } else {
    console.warn(`No PID found for job ${jobID}, cannot send stop signal`);
  }

  return NextResponse.json(job);
}
