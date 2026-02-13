import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { getTrainingFolder } from '@/server/settings';
import path from 'path';
import fs from 'fs';

const prisma = new PrismaClient();

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const markStopped = (info: string) =>
    prisma.job.update({
      where: { id: jobID },
      data: {
        stop: true,
        status: 'stopped',
        info,
      },
    });

  if (job.status !== 'running') {
    const updated = await markStopped(job.status === 'stopped' ? job.info : 'Job stopped');
    return NextResponse.json(updated);
  }

  let pid: number | null = null;
  try {
    const trainingRoot = await getTrainingFolder();
    const pidPath = path.join(trainingRoot, job.name, 'pid.txt');
    if (fs.existsSync(pidPath)) {
      const raw = fs.readFileSync(pidPath, 'utf8').trim();
      const n = parseInt(raw, 10);
      if (Number.isInteger(n) && n > 0) pid = n;
    }
  } catch {
    // pid file missing or unreadable — treat as no process
  }

  if (pid === null) {
    const updated = await markStopped('Job stopped');
    return NextResponse.json(updated);
  }

  if (!isProcessAlive(pid)) {
    const updated = await markStopped('Job stopped');
    return NextResponse.json(updated);
  }

  try {
    process.kill(pid, 'SIGTERM');
  } catch {
    // Process may have exited; still mark as stopped
  }

  const updated = await markStopped('Job stopped');
  return NextResponse.json(updated);
}
