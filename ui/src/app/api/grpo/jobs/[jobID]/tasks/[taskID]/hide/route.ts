import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const ACTIVE_TASK_STATUSES = new Set(['requested', 'generating', 'open', 'voted']);

export async function POST(
  _request: Request,
  { params }: { params: { jobID: string; taskID: string } },
) {
  try {
    const task = await prisma.flowGRPOVoteTask.findFirst({
      where: {
        id: params.taskID,
        job_id: params.jobID,
      },
      select: {
        id: true,
        status: true,
      },
    });

    if (!task) {
      return NextResponse.json({ error: 'Task not found' }, { status: 404 });
    }

    if (ACTIVE_TASK_STATUSES.has(task.status)) {
      return NextResponse.json({ error: 'Active tasks cannot be hidden' }, { status: 400 });
    }

    await prisma.flowGRPOVoteTask.update({
      where: {
        id: task.id,
      },
      data: {
        status: 'hidden',
      },
    });

    return NextResponse.json({ ok: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to hide Flow-GRPO vote task' }, { status: 500 });
  }
}

