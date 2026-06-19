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

  // Mark the job as stopping. The frontend modal polls /api/jobs?id=<id> and
  // will see this update almost immediately, before the kill below completes.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      info: job.pid != null
        ? `Stopping... sending termination signal to PID ${job.pid}`
        : 'Stopping... (no PID tracked)',
    },
  });

  if (job.pid != null) {
    console.log(`Attempting to stop job ${jobID} with PID ${job.pid}`);
    try {
      if (isWindows) {
        const { execSync } = require('child_process');
        // /T = kill the whole process tree, /F = force. This is the slow
        // step on Windows (often 1–5 seconds while child processes flush).
        execSync(`taskkill /PID ${job.pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(job.pid, 'SIGINT');
      }
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'stopped',
          info: 'Job stopped successfully.',
        },
      });
    } catch (e: any) {
      // Process may have already exited — record that fact instead of
      // swallowing it silently.
      const msg = e?.message || String(e);
      console.error('Error sending signal to process:', msg);
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'stopped',
          info: `Process already gone or kill failed: ${msg}`,
        },
      });
    }
  } else {
    await prisma.job.update({
      where: { id: jobID },
      data: {
        status: 'stopped',
        info: 'No PID was tracked. Marked as stopped.',
      },
    });
  }

  const updated = await prisma.job.findUnique({ where: { id: jobID } });
  return NextResponse.json(updated);
}
