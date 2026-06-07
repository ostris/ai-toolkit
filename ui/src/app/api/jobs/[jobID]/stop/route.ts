import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
const isWindows = process.platform === 'win32';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      info: 'Stopping job...',
    },
  });

  // Send SIGINT to the process if we have a PID
  if (job.pid != null) {
    console.log(`Attempting to stop job ${jobID} with PID ${job.pid}`);
    try {
      if (isWindows) {
        // Windows doesn't support SIGINT for arbitrary processes.
        // Use taskkill with /T (tree) to send a CTRL+C-like termination.
        const { execSync } = require('child_process');
        execSync(`taskkill /PID ${job.pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(job.pid, 'SIGINT');
      }
      // if it killed it, mark it stopped in the database
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'stopped',
          info: 'Job stopped',
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
