import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const isWindows = process.platform === 'win32';

// How long a stopping job gets to run its graceful shutdown (KeyboardInterrupt
// -> on_error -> final DB write) before we assume it is hung and force-kill it.
const GRACEFUL_STOP_TIMEOUT_MS = 60_000;
const BACKSTOP_POLL_MS = 2_000;

const isProcessAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    // EPERM means it exists but belongs to someone else, which still counts.
    return e?.code === 'EPERM';
  }
};

// Windows: the trainer's stop watcher sees the `stop` flag and raises SIGINT
// inside the process, so the graceful path runs without us sending anything.
// This backstop only exists for a job that is too hung to notice the flag.
// Poll (rather than one long timer) so we stop watching the moment the pid
// dies and never touch a recycled pid.
const scheduleForceKillBackstop = (pid: number, jobID: string) => {
  const startedAt = Date.now();
  const timer = setInterval(() => {
    if (!isProcessAlive(pid)) {
      clearInterval(timer);
      return;
    }
    if (Date.now() - startedAt < GRACEFUL_STOP_TIMEOUT_MS) return;
    clearInterval(timer);
    console.warn(`Job ${jobID} (pid ${pid}) still alive ${GRACEFUL_STOP_TIMEOUT_MS / 1000}s after stop request, force killing`);
    execAsync(`taskkill /PID ${pid} /T /F`, { windowsHide: true }).catch(() => {
      // already gone
    });
  }, BACKSTOP_POLL_MS);
  timer.unref?.();
};

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
        // No external SIGINT possible on Windows (the job runs under pythonw
        // with no console), and none is needed: the `stop` flag written above
        // is the signal. The trainer's stop watcher polls it and raises
        // SIGINT in-process, which runs the same KeyboardInterrupt shutdown
        // as Linux -- final DB write and the closing log lines.
        scheduleForceKillBackstop(job.pid, jobID);
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
