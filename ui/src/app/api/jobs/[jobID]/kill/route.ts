import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { listEngines } from '@/server/inferenceEngine';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const isWindows = process.platform === 'win32';

// Force stop: SIGKILL the job process (tree-kill on Windows) and mark the row
// stopped. For jobs that ignored the graceful stop (a hung model load).
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  // the pid from the row, else (inference engines) the one the live process
  // published in its engine.json
  let pid: number | null = job?.pid ?? null;
  if (pid == null) {
    const engine = (await listEngines(true)).find(e => e.jobId === jobID);
    pid = engine?.endpoint?.pid ?? null;
  }
  if (!job && pid == null) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  let killed = false;
  if (pid != null) {
    try {
      if (isWindows) {
        await execAsync(`taskkill /PID ${pid} /T /F`, { windowsHide: true });
      } else {
        process.kill(pid, 'SIGKILL');
      }
      killed = true;
    } catch (e: any) {
      console.warn(`kill ${jobID} pid ${pid}: ${e?.message || e}`);
    }
  }
  if (job) {
    await prisma.job.update({
      where: { id: jobID },
      data: { stop: true, status: 'stopped', info: 'Job force stopped', pid: null },
    });
  }
  return NextResponse.json({ ok: true, killed });
}
