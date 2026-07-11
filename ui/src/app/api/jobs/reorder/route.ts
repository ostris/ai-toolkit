import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const ACTIVE_JOB_STATUSES = ['queued', 'running', 'stopping'] as const;
const QUEUE_POSITION_STEP = 1000;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const jobIds = Array.isArray(body.jobIds) ? body.jobIds.filter((id: unknown): id is string => typeof id === 'string') : [];

    if (jobIds.length < 2) {
      return NextResponse.json({ error: 'At least two job IDs are required' }, { status: 400 });
    }

    const jobs = await prisma.job.findMany({
      where: { id: { in: jobIds } },
      select: { id: true, gpu_ids: true, status: true },
    });

    if (jobs.length !== jobIds.length) {
      return NextResponse.json({ error: 'One or more jobs were not found' }, { status: 404 });
    }

    const gpuIds = new Set(jobs.map(job => job.gpu_ids));
    if (gpuIds.size !== 1) {
      return NextResponse.json({ error: 'Jobs must belong to the same queue' }, { status: 400 });
    }

    if (jobs.some(job => !ACTIVE_JOB_STATUSES.includes(job.status as (typeof ACTIVE_JOB_STATUSES)[number]))) {
      return NextResponse.json({ error: 'Only active queue jobs can be reordered' }, { status: 400 });
    }

    await prisma.$transaction(
      jobIds.map((id: string, index: number) =>
        prisma.job.update({
          where: { id },
          data: { queue_position: (index + 1) * QUEUE_POSITION_STEP },
        }),
      ),
    );

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to reorder jobs' }, { status: 500 });
  }
}
